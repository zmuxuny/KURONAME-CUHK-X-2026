"""Strict legacy q/v bias conversion for current Transformers VideoMAE.

Reference: Transformers v4.25.1 VideoMAESelfAttention uses q_bias/v_bias
and an identically zero key bias. No pretrained tensor may be discarded.
"""

from __future__ import annotations

from pathlib import Path

from cuhkx.validation import sha256_file


def convert_legacy_state(source, layers):
    import torch

    converted = dict(source)
    mapping = {}
    for index in range(layers):
        prefix = f"videomae.encoder.layer.{index}.attention.attention."
        for old, new in (("q_bias", "query.bias"), ("v_bias", "value.bias")):
            if prefix + new in converted or prefix + old not in converted:
                raise ValueError("checkpoint does not match the declared legacy bias layout")
            converted[prefix + new] = converted.pop(prefix + old)
            mapping[prefix + old] = prefix + new
        if prefix + "key.bias" in converted:
            raise ValueError("legacy attention unexpectedly contains a key bias")
        converted[prefix + "key.bias"] = torch.zeros_like(converted[prefix + "value.bias"])
    return converted, mapping


def load_runtime(settings):
    import cv2
    import torch
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VideoMAEImageProcessor

    from cuhkx.harn_tail import configure_torch

    configure_torch()
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(settings["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    directory = Path(settings["model_directory"])
    for entry in settings["model_files"]:
        if sha256_file(directory / entry["name"]) != entry["sha256"]:
            raise ValueError("local VideoMAE asset changed")
    config = VideoMAEConfig.from_pretrained(directory, local_files_only=True)
    if (config.num_frames != 16 or config.hidden_size != 768 or config.num_labels != 174
            or not config.use_mean_pooling or config.num_hidden_layers != 12):
        raise ValueError("VideoMAE checkpoint is not the declared SSv2 base model")
    config._attn_implementation = "sdpa"
    source = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
    state, mapping = convert_legacy_state(source, config.num_hidden_layers)
    model = VideoMAEForVideoClassification(config)
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    for key, value in source.items():
        torch.testing.assert_close(actual[mapping.get(key, key)], value, rtol=0, atol=0)
    added = set(actual) - {mapping.get(key, key) for key in source}
    assert len(added) == 12 and all(key.endswith("key.bias") for key in added)
    assert all(torch.count_nonzero(actual[key]) == 0 for key in added)
    model.cuhkx_load_audit = {"strict_loading": True, "source_tensors": len(source),
                              "target_tensors": len(actual), "mapped_bias_tensors": len(mapping),
                              "all_source_tensors_exact_before_fp16": True,
                              "added_zero_key_bias_tensors": sorted(added),
                              "missing_tensors": [], "unexpected_tensors": []}
    processor = VideoMAEImageProcessor.from_pretrained(directory, local_files_only=True)
    return torch, model.eval().half().cuda(), processor
