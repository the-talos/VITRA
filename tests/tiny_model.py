"""Tiny random-weight VITRA model for tests that must run without a checkpoint.

The backbone is a few-layer PaliGemma built from a config (the real one is a gated
download), the action head is ``DiT-T``. ``predict_action`` runs end to end with a
stand-in processor, so the same dispatch code as the released model is exercised.
"""

from unittest import mock

import numpy as np
import torch
from transformers import (
    BatchFeature,
    Gemma2Config,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    SiglipVisionConfig,
)

from vitra.models.vla import vitra_paligemma

IMAGE_TOKEN = 250
NUM_IMAGE_TOKENS = 4
HIDDEN = 64
CHUNK = 4


class FakeProcessor:
    """Stand-in for ``PaliGemmaProcessor``: fixed token ids and a random-free pixel tensor."""

    tokenizer = None

    def __call__(self, text, images, return_tensors="pt"):
        input_ids = torch.tensor([[IMAGE_TOKEN] * NUM_IMAGE_TOKENS + [2, 5, 6, 7, 8]])
        pixels = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0 - 0.5)
        pixels = pixels.permute(2, 0, 1).unsqueeze(0)
        return BatchFeature({"input_ids": input_ids, "pixel_values": pixels})


def _tiny_backbone():
    text = Gemma2Config(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=16,
    )
    vision = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
        projection_dim=HIDDEN,
    )
    config = PaliGemmaConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_index=IMAGE_TOKEN,
        projection_dim=HIDDEN,
        hidden_size=HIDDEN,
    )
    return PaliGemmaForConditionalGeneration(config)


def build_tiny_vitra(use_bf16=False, seed=0):
    """A random-weight ``VITRA_Paligemma`` on the CPU in float32 (the repo's inference dtype)."""
    configs = {
        "fwd_pred_next_n": CHUNK,
        "train_dataset": {"action_type": "angle"},
        "state_encoder": {"state_dim": 212},
        "vlm": {"name": "paligemma"},
        "untied_cognition_token": True,
        "cognition_token_id": 10,
    }
    torch.manual_seed(seed)
    with mock.patch.object(vitra_paligemma, "build_vlm", return_value=(FakeProcessor(), _tiny_backbone())):
        model = vitra_paligemma.VITRA_Paligemma(
            configs=configs,
            train_setup_configs={"freeze_option": "freeze_vision_encoder"},
            act_model_configs={"model_type": "DiT-T", "token_size": HIDDEN, "action_dim": 192},
            fwd_pred_next_n=CHUNK,
            use_state="DiT",
            use_fov=True,
            use_bf16=use_bf16,
        )
    # DiT zero-initialises its adaLN and output layers, so a fresh model returns the same
    # actions for every input. Fill those tensors so the comparison is not vacuous.
    with torch.no_grad():
        for param in model.act_model.parameters():
            if not param.abs().any():
                param.normal_(0.0, 0.05)
    return model.eval()


def tiny_inputs(seed=1):
    """Image, state, state mask and fov for ``predict_action`` (batch of one)."""
    generator = torch.Generator().manual_seed(seed)
    image = (torch.rand(28, 28, 3, generator=generator) * 255).to(torch.uint8).numpy()
    state = torch.randn(1, 212, generator=generator)
    state_mask = torch.ones(1, 212, dtype=torch.bool)
    fov = torch.tensor([[1.2, 0.9]])
    return image, state, state_mask, fov


def fixed_noise(seed=7):
    """Patch ``torch.randn`` so the diffusion sampler starts from the same noise on any device.

    The noise is drawn on the CPU in float32 from a fixed seed and then moved, so a CPU
    run and an accelerator run denoise from the same starting point.
    """
    real_randn = torch.randn

    def randn(*size, device=None, dtype=None, **kwargs):
        generator = torch.Generator().manual_seed(seed)
        noise = real_randn(*size, generator=generator, dtype=torch.float32)
        return noise.to(device=device, dtype=dtype or torch.float32)

    return mock.patch.object(torch, "randn", randn)
