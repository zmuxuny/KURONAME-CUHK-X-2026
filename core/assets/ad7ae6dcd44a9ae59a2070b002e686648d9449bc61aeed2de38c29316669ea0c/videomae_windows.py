"""Local VideoMAE descriptors from explicit, label-free video records."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cuhkx.validation import sha256_file
from cuhkx.video import crop_video_frames, sample_video_windows


def load_runtime(settings):
    import cv2
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    from cuhkx.harn_tail import configure_torch

    configure_torch()
    cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(settings["gpu_memory_fraction"])
    torch.cuda.reset_peak_memory_stats()
    directory = Path(settings["model_directory"])
    for entry in settings["model_files"]:
        if sha256_file(directory / entry["name"]) != entry["sha256"]:
            raise ValueError("local VideoMAE asset changed")
    processor = VideoMAEImageProcessor.from_pretrained(directory, local_files_only=True)
    model = VideoMAEForVideoClassification.from_pretrained(
        directory, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().cuda()
    if (model.config.num_frames != 16 or model.config.hidden_size != 768
            or model.config.num_labels != 174 or not model.config.use_mean_pooling):
        raise ValueError("VideoMAE checkpoint is not the declared SSv2 base model")
    return torch, model, processor


def read_windows(record, settings):
    """Resolve media only through its declared path; IDs and labels are unused."""
    import cv2

    path = Path(record["path"])
    if sha256_file(path) != record["raw"]["sha256"]:
        raise ValueError("raw video changed after the extraction declaration")
    windows = sample_video_windows(path, settings["windows"], 16, settings["window_seconds"])
    shape = windows.shape
    flat = windows.reshape(-1, *shape[2:])
    cropped = crop_video_frames(flat, tuple(record["roi"]))
    if record["modality"] == "IR":
        gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in cropped])
        cropped = np.repeat(gray[..., None], 3, axis=-1)
    elif record["modality"] != "Depth_Color":
        raise ValueError("unsupported VideoMAE view")
    return cropped.reshape(settings["windows"], 16, *cropped.shape[1:])


def encode_window(runtime, frames, *, check_classifier=False):
    torch, model, processor = runtime
    pixels = processor(list(frames), return_tensors="pt")["pixel_values"]
    if pixels.shape != (1, 16, 3, 224, 224) or not torch.isfinite(pixels).all():
        raise ValueError("VideoMAE preprocessing produced an invalid tensor")
    pixels = pixels.to(device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        tokens = model.videomae(pixels).last_hidden_state
        features = model.fc_norm(tokens.mean(dim=1))
        logits = model.classifier(features)
        if check_classifier:
            torch.testing.assert_close(logits, model(pixel_values=pixels).logits, rtol=0, atol=0)
    features, logits = features.float().cpu().numpy()[0], logits.float().cpu().numpy()[0]
    if features.shape != (768,) or logits.shape != (174,) or not np.isfinite(features).all() or not np.isfinite(logits).all():
        raise ValueError("VideoMAE emitted invalid descriptors")
    return features, logits
