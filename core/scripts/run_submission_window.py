#!/usr/bin/env python3
"""Execute a finite, hash-pinned and previously authorized Kaggle window.

The default command only validates the plan. ``run`` is the real submission
path. An intent is durably written before each upload; an ambiguous upload is
only reconciled against Kaggle, never retried. No answer file is constructed or
edited here. Feedback can stop the frozen candidate sequence, never tune rows.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuhkx.submission import validate_submission_files

SLUG = "cuhk-x-competition-large-model-track"
TERMINAL = {"COMPLETE", "ERROR", "CANCELLED", "INVALID", "FAILED"}


def utc(value):
    result = datetime.fromisoformat(str(value))
    return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_plan(path, expected_hash):
    path = Path(path).resolve()
    if len(expected_hash) != 64 or sha(path) != expected_hash:
        raise ValueError("plan SHA256 mismatch")
    plan = json.loads(path.read_text())
    if plan["schema_version"] != 1 or plan["competition"] != SLUG:
        raise ValueError("unexpected competition plan")
    start, stop = utc(plan["not_before_utc"]), utc(plan["expires_utc"])
    if start >= stop or start.date() != stop.date():
        raise ValueError("plan must describe one finite UTC submission day")
    if plan["daily_limit"] != 5 or not 1 <= plan["max_submissions"] <= 2:
        raise ValueError("window budget must be at most two of the five daily submissions")
    if not 1 <= len(plan["candidates"]) <= plan["max_submissions"]:
        raise ValueError("candidate count exceeds budget")
    ids, hashes, messages = set(), set(), set()
    for index, candidate in enumerate(plan["candidates"]):
        if (candidate["id"] in ids or candidate["sha256"] in hashes
                or candidate["message"] in messages or not candidate["message"].strip()):
            raise ValueError("duplicate or empty candidate identity")
        ids.add(candidate["id"]); hashes.add(candidate["sha256"]); messages.add(candidate["message"])
        if len(candidate["message"]) > 200:
            raise ValueError("submission message too long")
        if not Path(candidate["submission"]).is_absolute():
            raise ValueError("candidate submission must be an absolute path")
        if candidate["public_prediction_inputs"] or not candidate["purpose"]:
            raise ValueError("new independent submissions cannot inherit public prediction tables")
        if index and not math.isfinite(float(candidate["minimum_previous_public_score"])):
            raise ValueError("follow-up requires a frozen aggregate-feedback stop threshold")
        report = validate_submission_files(plan["test_csv"], plan["sample_csv"], candidate["submission"])
        if report.rows != 682 or report.columns != ("qa_id", "prediction") or report.sha256 != candidate["sha256"]:
            raise ValueError("candidate differs from frozen 682-row competition contract")
        if not candidate["evidence"]:
            raise ValueError("candidate must pin its local validation and model provenance")
        for entry in candidate["evidence"]:
            if sha(entry["path"]) != entry["sha256"]:
                raise ValueError("local evidence SHA256 mismatch")
    for entry in plan["sources"]:
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("submission worker source changed")
    return plan


def normalize_submission(row):
    status = getattr(row.status, "name", str(row.status)).split(".")[-1]
    return {
        "ref": int(row.ref), "date": utc(row.date).isoformat(), "message": row.description,
        "file_name": row.file_name, "status": status, "public_score": row.public_score,
    }


@contextlib.contextmanager
def bounded_api_call(seconds=120):
    def expired(signum, frame):
        raise TimeoutError("Kaggle request exceeded fixed timeout")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        # Do not persist raw SDK output, signed upload URLs or credential-bearing errors.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class KaggleService:
    def __init__(self):
        with bounded_api_call():
            from kaggle.api.kaggle_api_extended import KaggleApi
            self.api = KaggleApi()
            self.api.authenticate()

    def contract(self):
        from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest
        request = ApiGetCompetitionRequest(); request.competition_name = SLUG
        with bounded_api_call(), self.api.build_kaggle_client() as client:
            row = client.competitions.competition_api_client.get_competition(request)
        return {"deadline": utc(row.deadline).isoformat(), "daily_limit": row.max_daily_submissions,
                "entered": row.user_has_entered, "disabled": row.submissions_disabled}

    def submissions(self):
        # All records are requested with pagination; do not infer quota from a truncated page.
        from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
        results, token, seen = [], "", set()
        for _ in range(100):
            request = ApiListSubmissionsRequest()
            request.competition_name = SLUG; request.page_token = token; request.page_size = 100
            with bounded_api_call(), self.api.build_kaggle_client() as client:
                response = client.competitions.competition_api_client.list_submissions(request)
            results.extend(normalize_submission(row) for row in response.submissions)
            token = response.next_page_token
            if not token:
                if len({row["ref"] for row in results}) != len(results):
                    raise ValueError("submission pagination returned duplicate references")
                return results
            if token in seen:
                raise ValueError("submission pagination did not advance")
            seen.add(token)
        raise ValueError("submission pagination exceeded finite bound")

    def submit(self, candidate):
        with bounded_api_call():
            response = self.api.competition_submit(candidate["submission"], candidate["message"], SLUG, quiet=True)
        return int(response.ref or 0)


class WindowRunner:
    def __init__(self, plan_path, plan_hash, output, service, now=None, sleep=None):
        self.plan_path, self.plan_hash = Path(plan_path), plan_hash
        self.plan = validate_plan(self.plan_path, plan_hash)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.service = service
        self.now = now or (lambda: datetime.now(UTC))
        self.sleep = sleep or time.sleep
        self.state_path = self.output / "state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state["plan_sha256"] != plan_hash:
                raise ValueError("cannot resume a different plan in the same window")
        else:
            self.state = {"plan_sha256": plan_hash, "candidates": {}, "status": "starting"}

    def save(self, status=None):
        if status:
            self.state["status"] = status
        self.state["updated_at_utc"] = self.now().isoformat()
        self.state["pid"] = os.getpid()
        write_json(self.state_path, self.state)

    def cancelled(self):
        return (self.output / "CANCEL").exists()

    def log_snapshot(self, rows):
        write_json(self.output / "latest_submissions.json", {
            "queried_at_utc": self.now().isoformat(), "submissions": rows,
        })

    def record_ledger(self, candidate, accepted):
        if not self.plan.get("ledger_path"):
            return
        path = Path(self.plan["ledger_path"])
        with path.open("r+", encoding="utf-8", newline="") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            reader = csv.DictReader(handle); fields = reader.fieldnames
            rows = list(reader)
            run_id = f"{self.plan['id']}_{candidate['id']}_ref{accepted['ref']}"
            if any(row["run_id"] == run_id for row in rows):
                return
            values = {
                "run_id": run_id, "created_at_utc": accepted["date"],
                "parent_run": candidate["parent_run"], "method": "scheduled_independent_submission",
                "config": candidate["model_config"], "validation_protocol": candidate["local_validation"],
                "validation_overall": candidate.get("validation_overall", ""),
                "validation_worst_subject": candidate.get("validation_worst_subject", ""),
                "submission_path": candidate["submission"], "sha256": candidate["sha256"],
                "kaggle_ref": str(accepted["ref"]), "public_score": str(accepted["public_score"]),
                "status": accepted["status"].lower(), "kaggle_message": candidate["message"],
                "notes": "Hash-pinned finite worker; exact API terminal recorded; no answer editing",
            }
            if not {"run_id", "submission_path", "sha256", "kaggle_ref", "public_score", "status"} <= set(fields):
                raise ValueError("unexpected ledger schema")
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
            writer.writerow(values); handle.flush(); os.fsync(handle.fileno())

    def run(self):
        if self.state["status"] in {"complete", "cancelled", "stopped_after_feedback", "expired", "manual_review_required"}:
            return self.state
        self.save("waiting_for_window")
        while self.now() < utc(self.plan["not_before_utc"]):
            if self.cancelled():
                self.save("cancelled"); return self.state
            self.sleep(min(30, (utc(self.plan["not_before_utc"]) - self.now()).total_seconds()))
            self.save()
        previous_score = None
        for index, candidate in enumerate(self.plan["candidates"]):
            if self.cancelled():
                self.save("cancelled"); return self.state
            if index and (previous_score is None or previous_score < candidate["minimum_previous_public_score"]):
                self.save("stopped_after_feedback"); return self.state
            state = self.state["candidates"].setdefault(candidate["id"], {})
            # Query before upload and on every resume. Never repeat an existing intent.
            rows = self.service.submissions(); self.log_snapshot(rows)
            matches = [row for row in rows if row["message"] == candidate["message"]]
            if len(matches) > 1:
                self.save("manual_review_required"); return self.state
            if matches:
                state["ref"] = matches[0]["ref"]
            elif not state.get("intent_utc"):
                if self.now() >= utc(self.plan["expires_utc"]):
                    self.save("expired"); return self.state
                contract = self.service.contract()
                if (contract["daily_limit"] != self.plan["daily_limit"] or not contract["entered"]
                        or contract["disabled"] or self.now() >= utc(contract["deadline"])):
                    self.save("manual_review_required"); return self.state
                used = sum(utc(row["date"]).date() == self.now().date() for row in rows)
                if used >= contract["daily_limit"]:
                    self.save("quota_exhausted"); return self.state
                validate_plan(self.plan_path, self.plan_hash)
                state["intent_utc"] = self.now().isoformat()
                state["used_before_submit"] = used
                self.save("upload_intent_recorded")
                try:
                    ref = self.service.submit(candidate)
                    if ref:
                        state["ref"] = ref
                except Exception as error:  # noqa: BLE001 - reconcile any uncertain upload, never resend
                    state["upload_exception_type"] = type(error).__name__
                self.save("reconciling_upload")
            # The first post-upload action is a submissions query, with no initial sleep.
            deadline = utc(state.get("intent_utc", self.now().isoformat()))
            deadline = deadline.timestamp() + self.plan.get("terminal_timeout_seconds", 7200)
            while self.now().timestamp() <= deadline:
                if self.cancelled():
                    self.save("cancelled"); return self.state
                try:
                    rows = self.service.submissions(); self.log_snapshot(rows)
                except Exception as error:  # noqa: BLE001 - bounded read retry, no raw SDK secrets in logs
                    state["query_exception_type"] = type(error).__name__
                    self.save("query_retry")
                    self.sleep(30); continue
                matches = [row for row in rows if row["message"] == candidate["message"]]
                if len(matches) > 1:
                    self.save("manual_review_required"); return self.state
                if matches:
                    accepted = matches[0]
                    if state.get("ref") and state["ref"] != accepted["ref"]:
                        self.save("manual_review_required"); return self.state
                    state.update({"ref": accepted["ref"], "accepted": accepted})
                    if accepted["status"] in TERMINAL:
                        self.record_ledger(candidate, accepted)
                        self.save("terminal_received")
                        if accepted["status"] != "COMPLETE" or accepted["public_score"] == "":
                            self.save("manual_review_required"); return self.state
                        previous_score = float(accepted["public_score"])
                        if not math.isfinite(previous_score) or not 0 <= previous_score <= 1:
                            self.save("manual_review_required"); return self.state
                        best = self.state.get("best_independent_candidate")
                        if best is None or previous_score > float(best["public_score"]):
                            self.state["best_independent_candidate"] = {"id": candidate["id"], **accepted}
                        self.save("terminal_received")
                        break
                self.save("waiting_for_terminal")
                self.sleep(30)
            else:
                self.save("manual_review_required"); return self.state
        candidates = list(self.state["candidates"].items())
        best_id, best = max(candidates, key=lambda item: float(item[1]["accepted"]["public_score"]))
        self.state["best_independent_candidate"] = {"id": best_id, **best["accepted"]}
        self.save("complete")
        return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plan = validate_plan(args.plan, args.plan_sha256)
    if args.command == "validate":
        print(json.dumps({"valid": True, "plan_sha256": args.plan_sha256,
                          "candidates": len(plan["candidates"]), "submission_performed": False}))
        return
    if args.output_dir is None:
        parser.error("run requires --output-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "window.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = WindowRunner(args.plan, args.plan_sha256, args.output_dir, KaggleService())
        try:
            state = runner.run()
        except Exception as error:  # noqa: BLE001 - persist the stop and redact potentially sensitive errors
            runner.state["exception_type"] = type(error).__name__
            runner.save("manual_review_required")
            raise SystemExit(2) from None
        print(json.dumps({"status": state["status"], "plan_sha256": args.plan_sha256}))


if __name__ == "__main__":
    main()
