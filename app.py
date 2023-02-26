#!/usr/bin/env python

from __future__ import annotations

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

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run('patch -p1'.split(), cwd='gan-control', stdin=f)

sys.path.insert(0, 'gan-control/src')

from gan_control.inference.controller import Controller

DESCRIPTION = '''GAN-Control

This is an unofficial demo for https://github.com/amazon-research/gan-control.
'''

TOKEN = os.getenv('HF_TOKEN')


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


download_models()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = 'controller_age015id025exp02hai04ori02gam15/'
controller = Controller(path, device)
func = functools.partial(run, controller=controller, device=device)

gr.Interface(
    fn=func,
    inputs=[
        gr.Slider(label='Seed', minimum=0, maximum=1000000, step=1, value=0),
        gr.Slider(label='Truncation',
                  minimum=0,
                  maximum=1,
                  step=0.1,
                  value=0.7),
        gr.Slider(label='Yaw', minimum=-90, maximum=90, step=1, value=30),
        gr.Slider(label='Pitch', minimum=-90, maximum=90, step=1, value=0),
        gr.Slider(label='Age', minimum=15, maximum=75, step=1, value=75),
        gr.Slider(label='Hair Color (R)',
                  minimum=0,
                  maximum=255,
                  step=1,
                  value=186),
        gr.Slider(label='Hair Color (G)',
                  minimum=0,
                  maximum=255,
                  step=1,
                  value=158),
        gr.Slider(label='Hair Color (B)',
                  minimum=0,
                  maximum=255,
                  step=1,
                  value=92),
        gr.Slider(label='Number of Rows',
                  minimum=1,
                  maximum=3,
                  step=1,
                  value=1),
        gr.Slider(label='Number of Columns',
                  minimum=1,
                  maximum=5,
                  step=1,
                  value=5),
    ],
    outputs=[
        gr.Image(label='Generated Image', type='pil'),
        gr.Image(label='Head Pose Controlled', type='pil'),
        gr.Image(label='Age Controlled', type='pil'),
        gr.Image(label='Hair Color Controlled', type='pil'),
    ],
    description=DESCRIPTION,
).queue().launch(show_api=False)
