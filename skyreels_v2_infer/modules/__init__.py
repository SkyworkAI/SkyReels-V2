import gc
import os

import torch
from safetensors.torch import load_file

from .clip import CLIPModel
from .t5 import T5EncoderModel
from .transformer import WanModel
from .vae import WanVAE


def download_model(model_id):
    if not os.path.exists(model_id):
        from huggingface_hub import snapshot_download

        model_id = snapshot_download(repo_id=model_id)
    return model_id


def get_vae(model_path, device="cuda", weight_dtype=torch.float32) -> WanVAE:
    vae = WanVAE(model_path).to(device).to(weight_dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return vae


def get_transformer(model_path, device="cuda", weight_dtype=torch.bfloat16) -> WanModel:
    config_path = os.path.join(model_path, "config.json")
    transformer = WanModel.from_config(config_path).to(weight_dtype).to(device)

    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            state_dict = load_file(file_path)
            transformer.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    transformer.requires_grad_(False)
    transformer.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return transformer


def get_text_encoder(model_path, device="cuda", weight_dtype=torch.bfloat16) -> T5EncoderModel:
    t5_model = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(model_path, "google", "umt5-xxl")
    text_encoder = T5EncoderModel(checkpoint_path=t5_model, tokenizer_path=tokenizer_path).to(device).to(weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return text_encoder


def get_image_encoder(model_path, device="cuda", weight_dtype=torch.bfloat16) -> CLIPModel:
    checkpoint_path = os.path.join(model_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    tokenizer_path = os.path.join(model_path, "xlm-roberta-large")
    image_enc = CLIPModel(checkpoint_path, tokenizer_path).to(weight_dtype).to(device)
    image_enc.requires_grad_(False)
    image_enc.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return image_enc
