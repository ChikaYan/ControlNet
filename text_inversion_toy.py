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

from real_dataset import FaceDataset

from pathlib import Path

import torchvision

from cldm.cldm import ControlLDM




def log_image(control, prompt_embed, ddim_steps, strength, scale, eta):
    B, C, H, W = control.shape

    cond = {"c_concat": [control], "c_crossattn": [prompt_embed]}
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


if __name__ == '__main__':

    log_dir = Path("/home/tw554/pointavatar_gs/ext/ControlNet/log/text_inversion_toy/decoder_all_data")
    log_dir.mkdir(exist_ok=True)


    apply_openpose = OpenposeDetector()

    model: ControlLDM = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_openpose.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    initial_prompt = "talking man"
    prompt_embed = torch.nn.Parameter(model.get_learned_conditioning([initial_prompt]).detach().requires_grad_(True))
    # prompt_embed = torch.autograd.Variable(torch.rand_like(model.get_learned_conditioning([initial_prompt])), requires_grad = True)

    optimizer = torch.optim.AdamW([
        {'name': 'text_embed', 'params': [prompt_embed], 'lr': 1e-5},
        {'name': 'diffusion_decoder', 
         'params': list(model.model.diffusion_model.output_blocks.parameters()) + list(model.model.diffusion_model.out.parameters()), 
         'lr': 1e-5},
        ], lr=0.)

    DATASET_PATH = Path('/home/tw554/IMavatar/data_pa/data/datasets/gesture/gesture/all/')

    gts = list(sorted((DATASET_PATH / 'image').glob('*.png'), key=lambda x: int(x.stem)))
    controls = list(sorted((DATASET_PATH / 'dwpose').glob('*.png'), key=lambda x: int(x.stem)))

    dataset_size = len(gts)
    # dataset_size = 1
    gts = gts[:dataset_size]
    controls = gts[:dataset_size]


    N_ITER = 100000
    

    for iter, data_id in enumerate(np.random.permutation(list(range(N_ITER)))):
        control_p = controls[data_id % dataset_size]
        gt_p = gts[data_id % dataset_size]

        control = imageio.imread(str(control_p)) / 255.
        control = torch.from_numpy(control).permute([2,0,1])[:3,...].unsqueeze(0).cuda().float()

        gt = imageio.imread(str(gt_p)) / 255.
        gt = torch.from_numpy(gt).permute([2,0,1])[:3,...].unsqueeze(0).cuda().float()
        gt = gt * 2. - 1.

        # control needs to be in [0, 1]
        # gt needs to be in [-1, 1]


        encoder_posterior = model.encode_first_stage(gt) # VAE encoder
        z = model.get_first_stage_encoding(encoder_posterior).detach() # sample from VAE latent
        cond = {
            'c_crossattn': [prompt_embed],
            'c_concat': [control],
        }

        loss, loss_dict = model(z, cond)


        # z_decode = model.decode_first_stage(z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"Iter: {iter}, loss: {loss}")


        # logging
        if iter % 100 == 0:
            pred = log_image(
                    control=control,
                    prompt_embed=prompt_embed,
                    ddim_steps=20,
                    strength=1.,
                    scale=15.,
                    eta=0.,
                )
                        
            torchvision.utils.save_image((torch.concat([pred, gt], axis=0)+ 1.) / 2., str(log_dir / f'{iter:06d}.png'))

