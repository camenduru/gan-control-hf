#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import pathlib
import subprocess
import sys
import tarfile

import gradio as gr
import huggingface_hub
import numpy as np
import PIL.Image
import torch

if os.environ.get('SYSTEM') == 'spaces':
    subprocess.call('git apply ../patch'.split(), cwd='gan-control')

sys.path.insert(0, 'gan-control/src')

from gan_control.inference.controller import Controller

TITLE = 'amazon-research/gan-control'
DESCRIPTION = '''This is an unofficial demo for https://github.com/amazon-research/gan-control.

Expected execution time on Hugging Face Spaces: 7s (for one image)
'''
ARTICLE = '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.gan-control" alt="visitor badge"/></center>'

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


def download_models() -> None:
    model_dir = pathlib.Path('controller_age015id025exp02hai04ori02gam15')
    if not model_dir.exists():
        path = huggingface_hub.hf_hub_download(
            'hysts/gan-control',
            'controller_age015id025exp02hai04ori02gam15.tar.gz',
            use_auth_token=TOKEN)
        with tarfile.open(path) as f:
            f.extractall()


@torch.inference_mode()
def run(
    seed: int,
    truncation: float,
    yaw: int,
    pitch: int,
    age: int,
    hair_color_r: float,
    hair_color_g: float,
    hair_color_b: float,
    nrows: int,
    ncols: int,
    controller: Controller,
    device: torch.device,
) -> PIL.Image.Image:
    seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
    batch_size = nrows * ncols
    latent_size = controller.config.model_config['latent_size']
    latent = torch.from_numpy(
        np.random.RandomState(seed).randn(batch_size,
                                          latent_size)).float().to(device)

    initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(
        latent=latent, truncation=truncation)
    res0 = controller.make_resized_grid_image(initial_image_tensors,
                                              nrow=ncols)

    pose_control = torch.tensor([[yaw, pitch, 0]], dtype=torch.float32)
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(
        latent=initial_latent_w,
        input_is_latent=True,
        orientation=pose_control)
    res1 = controller.make_resized_grid_image(image_tensors, nrow=ncols)

    age_control = torch.tensor([[age]], dtype=torch.float32)
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(
        latent=initial_latent_w, input_is_latent=True, age=age_control)
    res2 = controller.make_resized_grid_image(image_tensors, nrow=ncols)

    hair_color = torch.tensor([[hair_color_r, hair_color_g, hair_color_b]],
                              dtype=torch.float32) / 255
    hair_color = torch.clamp(hair_color, 0, 1)
    image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(
        latent=initial_latent_w, input_is_latent=True, hair=hair_color)
    res3 = controller.make_resized_grid_image(image_tensors, nrow=ncols)

    return res0, res1, res2, res3


def main():
    args = parse_args()
    device = torch.device(args.device)

    download_models()

    path = 'controller_age015id025exp02hai04ori02gam15/'
    controller = Controller(path, device)

    func = functools.partial(run, controller=controller, device=device)
    func = functools.update_wrapper(func, run)

    gr.Interface(
        func,
        [
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(0, 1, step=0.1, default=0.7, label='Truncation'),
            gr.inputs.Slider(-90, 90, step=1, default=30, label='Yaw'),
            gr.inputs.Slider(-90, 90, step=1, default=0, label='Pitch'),
            gr.inputs.Slider(15, 75, step=1, default=75, label='Age'),
            gr.inputs.Slider(
                0, 255, step=1, default=186, label='Hair Color (R)'),
            gr.inputs.Slider(
                0, 255, step=1, default=158, label='Hair Color (G)'),
            gr.inputs.Slider(
                0, 255, step=1, default=92, label='Hair Color (B)'),
            gr.inputs.Slider(1, 3, step=1, default=1, label='Number of Rows'),
            gr.inputs.Slider(
                1, 5, step=1, default=5, label='Number of Columns'),
        ],
        [
            gr.outputs.Image(type='pil', label='Generated Image'),
            gr.outputs.Image(type='pil', label='Head Pose Controlled'),
            gr.outputs.Image(type='pil', label='Age Controlled'),
            gr.outputs.Image(type='pil', label='Hair Color Controlled'),
        ],
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
