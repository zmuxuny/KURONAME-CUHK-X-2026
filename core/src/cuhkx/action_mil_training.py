"""Actual generic R34 last-block fitting on subject-scoped HAU partial labels."""

from __future__ import annotations

import gc
import hashlib
import json
import time

import numpy as np

from cuhkx import action_mil as core
from cuhkx.harn_tail import configure_torch


def fitting_digest(bank, units, targets):
    indices, available = core.window_rows(bank, units)
    digest = hashlib.sha256()
    digest.update(json.dumps({"units": units, "targets_shape": list(targets.shape),
                              "targets_dtype": str(targets.dtype)}, sort_keys=True).encode())
    digest.update(np.ascontiguousarray(targets).tobytes())
    for rows, mask in zip(indices, available, strict=True):
        digest.update(np.ascontiguousarray(rows).tobytes())
        digest.update(np.ascontiguousarray(mask).tobytes())
        digest.update(np.ascontiguousarray(bank.features[rows[mask]]).tobytes())
    return digest.hexdigest()


def forward_clip(model, bank, indices, available, torch):
    values = np.asarray(bank.features[indices[available]]).copy()
    tensor = torch.from_numpy(values).to("cuda", dtype=torch.float32)
    mask = torch.as_tensor(available, device="cuda", dtype=torch.bool)
    with torch.autocast(device_type="cuda"):
        output = model(tensor)
    logits = torch.zeros((1, 8, 2, 40), device="cuda", dtype=torch.float32)
    logits[0, mask] = output.float()
    return logits, mask[None]


def fit(bank, units, targets, train_subjects, validation_subjects, settings, seed):
    core.check_fit_scope(units, train_subjects, validation_subjects)
    if len(units) != len(targets):
        raise ValueError("MIL training units and partial targets differ")
    weights, supported = core.positive_weights(targets)
    digest = fitting_digest(bank, units, targets)
    indices, availability = core.window_rows(bank, units)
    torch = configure_torch()
    torch.cuda.set_per_process_memory_fraction(settings["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random = np.random.default_rng(seed)
    model = core.network(settings["initial_weights"]).cuda()
    optimizer = torch.optim.AdamW([
        {"params": model[0].parameters(), "lr": settings["last_block_lr"]},
        {"params": model[3].parameters(), "lr": settings["classifier_lr"]}], weight_decay=settings["weight_decay"])
    accumulation = settings["accumulation_clips"]
    steps = settings["epochs"] * ((len(units) + accumulation - 1) // accumulation)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    scaler = torch.amp.GradScaler("cuda")
    positive_weights = torch.from_numpy(weights).cuda()
    history, started, optimizer_steps = [], time.perf_counter(), 0
    for epoch in range(settings["epochs"]):
        model.train()
        order = random.permutation(len(units))
        total, skipped = 0., 0
        for start in range(0, len(order), accumulation):
            batch = order[start:start + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for row in batch:
                logits, mask = forward_clip(model, bank, indices[row], availability[row], torch)
                truth = torch.as_tensor(targets[row:row + 1].astype(np.float32), device="cuda")
                loss = core.masked_loss(logits, mask, truth, positive_weights, settings["temperature"])
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite MIL training loss")
                scaler.scale(loss / len(batch)).backward()
                total += float(loss.detach())
            old_scale = scaler.get_scale()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                schedule.step()
                optimizer_steps += 1
            else:
                skipped += 1
        entry = {"epoch": epoch + 1, "training_clips": len(units), "mean_training_loss": total / len(units),
                 "skipped_steps": skipped, "optimizer_steps": optimizer_steps,
                 "elapsed_seconds": time.perf_counter() - started}
        history.append(entry)
        print(json.dumps(entry), flush=True)
    state = {name: value.detach().cpu().numpy().astype(np.float16) if value.is_floating_point()
             else value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()}
    if any(not np.isfinite(value).all() for value in state.values()):
        raise ValueError("nonfinite retained MIL parameters")
    result = {"kind": "cuhkx_hau_r34_mil_v1", "state": state, "settings": settings, "seed": seed,
              "training_subjects": sorted(train_subjects), "validation_subjects": sorted(validation_subjects),
              "training_units": list(units), "training_targets": targets.copy(), "fitting_content_sha256": digest,
              "positive_weights": weights, "supported_actions": supported, "history": history,
              "fit_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    del model, optimizer, schedule, scaler, logits, loss, truth
    gc.collect()
    torch.cuda.empty_cache()
    return result


def predict(bundle, bank, units, *, allow_training=False):
    from cuhkx.metadata import subject_id

    if not allow_training and {subject_id(unit) for unit in units} & set(bundle["training_subjects"]):
        raise ValueError("MIL prediction subject appeared in neural fitting")
    torch = configure_torch()
    model = core.network()
    model.load_state_dict({name: torch.from_numpy(value) for name, value in bundle["state"].items()})
    model.eval().cuda()
    indices, availability = core.window_rows(bank, units)
    logits, probabilities = [], []
    with torch.inference_mode():
        correction = torch.from_numpy(np.log(bundle["positive_weights"])).cuda()
        for rows, mask in zip(indices, availability, strict=True):
            values, visible = forward_clip(model, bank, rows, mask, torch)
            pooled = core.pool_logits(values, visible, bundle["settings"]["temperature"])
            probability = (pooled - correction).sigmoid().cpu().numpy()[0]
            probability[~bundle["supported_actions"]] = .5
            logits.append(pooled.cpu().numpy()[0])
            probabilities.append(probability)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return np.asarray(probabilities), np.asarray(logits)


class ProbabilityIdentity:
    """Adapter for already computed neural probabilities in the frozen joint decoder."""

    def predict(self, matrix):
        return np.asarray(matrix)
