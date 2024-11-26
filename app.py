import torch
import streamlit as st
from PIL import Image
import qrcode
from pathlib import Path
import requests
import io
import os

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

st.set_page_config(page_title="QR Code Art Generator", layout="wide")

if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None

@st.cache_resource
def load_models():
    controlnet = ControlNetModel.from_pretrained(
        "DionTimmer/controlnet_qrcode-control_v11p_sd21", 
        torch_dtype=torch.float16
    ).to("cuda")

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    return pipe

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}

def main():
    st.title("QR Code Art Generator")
    
    pipe = load_models()

    with st.sidebar:
        qr_code_content = st.text_input("QR Code Content", help="QR Code Content or URL")
        prompt = st.text_input("Prompt", help="Prompt that guides the generation towards")
        negative_prompt = st.text_input("Negative Prompt", 
                                      value="ugly, disfigured, low quality, blurry, nsfw")
        
        guidance_scale = st.slider("Guidance Scale", 0.0, 50.0, 20.0, 0.25)
        controlnet_conditioning_scale = st.slider("Controlnet Conditioning Scale", 
                                                0.0, 5.0, 1.5, 0.01)
        strength = st.slider("Strength", 0.0, 1.0, 0.9, 0.01)
        
        sampler = st.selectbox("Sampler", list(SAMPLER_MAP.keys()))
        seed = st.number_input("Seed", value=-1)
        
        generate_button = st.button("Generate")

    if generate_button:
        if not prompt:
            st.error("Prompt is required")
            return
        
        if not qr_code_content:
            st.error("QR Code Content is required")
            return

        with st.spinner("Generating QR Code art..."):
            # Generate QR Code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(qr_code_content)
            qr.make(fit=True)

            qrcode_image = qr.make_image(fill_color="black", back_color="white")
            qrcode_image = resize_for_condition_image(qrcode_image, 768)

            pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)
            
            generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=qrcode_image,
                control_image=qrcode_image,
                width=768,
                height=768,
                guidance_scale=float(guidance_scale),
                controlnet_conditioning_scale=float(controlnet_conditioning_scale),
                generator=generator,
                strength=float(strength),
                num_inference_steps=150,
            )
            
            st.session_state.generated_image = output.images[0]

    if st.session_state.generated_image is not None:
        st.image(st.session_state.generated_image, caption="Generated QR Code Art")

if __name__ == "__main__":
    main()