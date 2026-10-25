"""Explicit official V-JEPA2.1-B encoder loading; no hub URL or predictor use."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from cuhkx.validation import sha256_file


def load_encoder(config, device='cuda'):
    weights = Path(config['weights'])
    if sha256_file(weights) != config['weights_sha256']:
        raise ValueError('Official JEPA checkpoint changed')
    source = Path(config['source_root'])
    for item in config['source_files']:
        if sha256_file(source / item['path']) != item['sha256']:
            raise ValueError('Pinned JEPA implementation changed')
    sys.path[:0] = [config['dependency_target'], str(source)]
    from app.vjepa_2_1.models.vision_transformer import vit_base
    from evals.hub.preprocessor import vjepa2_preprocessor

    checkpoint = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
    original = checkpoint['ema_encoder']
    state = {}
    for key, value in original.items():
        cleaned = key.replace('module.', '').replace('backbone.', '')
        if cleaned in state:
            raise ValueError('Ambiguous author checkpoint prefix mapping')
        state[cleaned] = value
    model = vit_base(img_size=(384, 384), patch_size=16, num_frames=16, tubelet_size=2,
                     use_sdpa=True, use_silu=False, wide_silu=True, uniform_power=False,
                     use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
    model.load_state_dict(state, strict=True)
    exact = all(torch.equal(model.state_dict()[key], value) for key, value in state.items())
    if not exact:
        raise ValueError('Encoder state differs before deployment precision conversion')
    audit = {'checkpoint_keys': list(checkpoint), 'checkpoint_key': 'ema_encoder',
             'source_tensors': len(state), 'strict_loading': True,
             'exact_before_precision_conversion': exact,
             'parameters': sum(p.numel() for p in model.parameters()),
             'input_shape': [1, 3, 16, 384, 384], 'output_shape': [1, 4608, 768],
             'return_hierarchical': model.return_hierarchical,
             'precision': 'bfloat16 frozen encoder, outputs converted exactly to float32'}
    model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    return model, vjepa2_preprocessor(crop_size=384), audit


@torch.inference_mode()
def encode(model, processor, frames):
    pixels = processor(frames)[0][None]
    if pixels.shape != (1, 3, 16, 384, 384):
        raise ValueError('Unexpected JEPA author preprocessing axes')
    device = next(model.parameters()).device
    # Author RoPE promotes q/k to fp32. Autocast reconciles q/k/v at SDPA,
    # preserving the original implementation and the declared BF16 execution.
    with torch.autocast(device.type, dtype=torch.bfloat16):
        tokens = model(pixels.to(device=device, dtype=torch.bfloat16))
    if tokens.shape != (1, 4608, 768) or not torch.isfinite(tokens).all():
        raise ValueError('Expected finite final-layer dense JEPA tokens')
    return tokens.float().reshape(1, 8, 576, 768), pixels
