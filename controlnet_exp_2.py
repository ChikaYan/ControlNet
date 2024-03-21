from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import imageio


apply_openpose = OpenposeDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


control = imageio.imread('/home/tw554/IMavatar/data_pa/data/datasets/gesture/gesture/all/dwpose/1690.png') / 255.
control = torch.from_numpy(control).permute([2,0,1])[:3,...].unsqueeze(0).cuda().float()


def process(control, prompt, ddim_steps, strength, scale, eta):
    with torch.no_grad():
        B, C, H, W = control.shape

        

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])]}
        # cond = {"c_concat": [control], "c_crossattn": [prompt]}
        # un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""])]}
        shape = (4, H // 8, W // 8)


        model.control_scales = [strength] * 13 
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size=1,
                                                     shape=shape, conditioning=cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)


 
        results = model.decode_first_stage(samples) # [B, C, H, W] in range [-1, 1]

    return results



process(
    control=control,
    prompt="normal man",
    ddim_steps=20,
    strength=1.,
    scale=15.,
    eta=0.,
    )

