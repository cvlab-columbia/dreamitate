import math
import os
from glob import glob
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from scripts.util.detection.nsfw_and_watermark_dectection import \
    DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config

import open_clip

os.environ['GRADIO_TEMP_DIR'] = "outputs/gradio_temp"
import gradio as gr
from functools import partial

def convert_frame(frame, size=None):
    
    if size is not None:
        original_height, original_width = frame.shape[:2]
        new_width = int(original_height * (size[0]/size[1]))
        crop_start = (original_width - new_width) // 2
        cropped_image = frame[:, crop_start:crop_start + new_width]
        frame = cv2.resize(cropped_image, size, interpolation=cv2.INTER_AREA)

    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = ToTensor()(frame).numpy()
    frame = frame * 2.0 - 1.0

    return frame

def prepare_model(
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd_xt",
    device: str = "cuda",
    output_folder: Optional[str] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "svd_image_decoder":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_image_decoder.yaml"
    elif version == "svd_xt_image_decoder":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(
            output_folder, "outputs/simple_video_sample/svd_xt_image_decoder/"
        )
        model_config = "scripts/sampling/configs/svd_xt_image_decoder.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
    )
    
    return model, filter

def inference(
    image_input_top,
    image_input_side,
    text,
    motion_bucket_id,
    model, 
    filter,
    fps_id: int = 16,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 25,
    device: str = "cuda",
    num_frames = 25,
    output_folder = os.environ['GRADIO_TEMP_DIR'],
    frame_width: int = 768,
    frame_height: int = 448,
):  
    image_input_top = convert_frame(image_input_top, (frame_width, frame_height))
    image_input_side = convert_frame(image_input_side, (frame_width, frame_height))

    cond_frames = np.concatenate((np.expand_dims(image_input_top, axis=0).repeat(13, axis=0), np.expand_dims(image_input_side, axis=0).repeat(12, axis=0)), axis=0)
    cond_frames = (cond_frames + cond_aug * np.random.randn(*cond_frames.shape))
    cond_frames = torch.from_numpy(cond_frames.astype(np.float32)).to(device)

    H, W = image_input_top.shape[1:]
    assert image_input_top.shape[0] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    value_dict = {}
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = cond_frames
    value_dict["cond_frames_without_noise"] = open_clip.tokenize(text)

    with torch.no_grad():
        with torch.autocast(device):

            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

            def denoiser(input, sigma, c):
                return model.denoiser(
                    model.model, input, sigma, c, **additional_model_inputs
                )

            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            base_count = len(glob(os.path.join(output_folder, "*.mp4")))
            video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MP4V"),
                fps_id + 1,
                (samples.shape[-1], samples.shape[-2]),
            )

            # samples = embed_watermark(samples)
            samples = filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            for frame in vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            writer.release()
    
    return video_path
    

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[1]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )

    if device == "cuda":
        # with torch.device(device):
        model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter

if __name__ == "__main__":
    model, filter = prepare_model()

    def inference_wrapper(image_input_top, image_input_side, text, motion_bucket_id, fps_id):
        return inference(model=model, filter=filter, image_input_top=image_input_top, image_input_side=image_input_side, text=text, motion_bucket_id=motion_bucket_id, fps_id=fps_id)

    demo = gr.Blocks()

    with demo:
        with gr.Row():
            image_input_top = gr.Image(label="Top Image (View 1)")
            image_input_side = gr.Image(label="Side Image (View 2)")
            text_input = gr.Textbox(label="Enter Clip Text Embbeding", value="Rotate the object on the black table counter clockwise with two hands")
            motion_bucket_id = gr.Number(label="Enter motion_bucket_id", value=200)
            fps_id = gr.Number(label="Enter fps_id", value=6)
        with gr.Row():
            video_output = gr.Video()
        
        gr.Button("Run").click(
            inference_wrapper,
            inputs=[image_input_top, image_input_side, text_input, motion_bucket_id, fps_id],
            outputs=video_output
        )

    demo.launch(share=True)