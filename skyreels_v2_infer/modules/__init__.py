import os

import torch
from safetensors.torch import load_file

from .clip import CLIPModel
from .t5 import T5EncoderModel
from .transformer import WanModel
from .vae import WanVAE


def get_vae(model_path, device="cuda", weight_dtype=torch.float32) -> WanVAE:
    vae = WanVAE(model_path).to(device).to(weight_dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    return vae


def get_transformer(model_path, device="cuda", weight_dtype=torch.bfloat16) -> WanModel:
    """
    load safetensor
    """
    config_path = os.path.join(model_path, "config.json")
    transformer = WanModel.from_config(config_path).to(weight_dtype).to(device)
    state_dict = {}
    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            state_dict.update(load_file(file_path))
    transformer.load_state_dict(state_dict, strict=False)
    transformer.requires_grad_(False)
    transformer.eval()
    transformer.to(device=device, dtype=weight_dtype)
    return transformer


def get_text_encoder(model_path, device="cuda", weight_dtype=torch.bfloat16) -> T5EncoderModel:
    t5_model = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(model_path, "google", "umt5-xxl")
    text_encoder = T5EncoderModel(checkpoint_path=t5_model, tokenizer_path=tokenizer_path).to(device).to(weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    return text_encoder


def get_image_encoder(model_path, device="cuda", weight_dtype=torch.bfloat16) -> CLIPModel:
    checkpoint_path = os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    tokenizer_path = os.path.join(model_path, "xlm-roberta-large")
    image_enc = CLIPModel(checkpoint_path, tokenizer_path).to(weight_dtype).to(device)
    image_enc.requires_grad_(False)
    image_enc.eval()
    return image_enc
