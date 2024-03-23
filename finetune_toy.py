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

import wandb

from tqdm import tqdm

from pyhocon import ConfigFactory
import argparse




def log_image(model, ddim_sampler, control, prompt_embed, ddim_steps, strength, scale, eta):
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






def train(conf):

    log_dir = Path("./log") / conf['run_name']
    log_dir.mkdir(exist_ok=True, parents=True)

    ckpt_dir = log_dir / 'ckpt'
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    train_img_dir = log_dir / 'train'
    train_img_dir.mkdir(exist_ok=True, parents=True)


    model: ControlLDM = create_model('./models/cldm_v15.yaml').cpu()

    ckpt_path = './models/control_sd15_openpose.pth'
    resume = False
    if conf['resume'] and (ckpt_dir / 'latest.pth').exists():
        print('Resuming from previously trained ckpt!')
        ckpt_path = str(ckpt_dir / 'latest.pth')
        resume = True

    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    initial_prompt = "talking man"
    prompt_embed = torch.nn.Parameter(model.get_learned_conditioning([initial_prompt]).detach().requires_grad_(True))



    optimizer = torch.optim.AdamW([
        {'name': 'text_embed', 'params': [prompt_embed], 'lr': 1e-5},
        {'name': 'diffusion_decoder', 
         'params': list(model.model.diffusion_model.output_blocks.parameters()) + list(model.model.diffusion_model.out.parameters()), 
         'lr': 1e-5},
        ], lr=0.)
    
    if resume:
        optimizer.load_state_dict(torch.load(str(ckpt_dir / 'opt_latest.pth')))



    train_dataset = FaceDataset(
        data_folder=conf['dataset.data_folder'],
        subject_name=conf['subject'],
        json_name='flame_params.json',
        use_background=False,
        load_body_ldmk=False,
        is_eval=False,
        **conf.get_config('dataset.train')
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=conf['batch_size'],
                                                    shuffle=True,
                                                    collate_fn=train_dataset.collate_fn,
                                                    num_workers=4,
                                                    )

    test_dataset = FaceDataset(
        data_folder=conf['dataset.data_folder'],
        subject_name=conf['subject'],
        json_name='flame_params.json',
        use_background=False,
        load_body_ldmk=False,
        is_eval=False,
        **conf.get_config('dataset.test')
    )

    TEST_IDS = np.array(conf['dataset.test_log_ids']) - 2672 + 500

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=conf['batch_size'],
                                                    shuffle=True,
                                                    collate_fn=test_dataset.collate_fn,
                                                    num_workers=4,
                                                    )

    NO_HEAD = conf.get('dataset.no_head', False)


    N_EPOCHS = conf['n_epochs']
    iter = -1
    test_iter = -1

    for ep_i in range(N_EPOCHS):
        for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(train_dataloader), desc=f"epoch: {ep_i:03d}"):
            iter += 1

            for k, v in model_input.items():
                try:
                    model_input[k] = v.cuda()
                except:
                    model_input[k] = v
            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.cuda()
                except:
                    ground_truth[k] = v

            control = einops.rearrange(ground_truth['dwpose_im'], 'b h w c -> b c h w')
            B, _, H, W = control.shape
            gt = einops.rearrange(ground_truth['rgb'].reshape([B, H, W, 3]), 'b h w c -> b c h w').float()
            gt = gt * 2. - 1.
            # control needs to be in [0, 1]
            # gt needs to be in [-1, 1]

            if NO_HEAD:
                head_mask = einops.rearrange(ground_truth['head_mask'].reshape([B, H, W, 1]), 'b h w c -> b c h w')
                gt[head_mask.repeat([1,3,1,1]) > 0.] = 1.


            encoder_posterior = model.encode_first_stage(gt) # VAE encoder
            z = model.get_first_stage_encoding(encoder_posterior).detach() # sample from VAE latent
            cond = {
                'c_crossattn': [prompt_embed],
                'c_concat': [control],
            }

            loss, loss_dict = model(z, cond)


            z_decode = model.decode_first_stage(z)
            torchvision.utils.save_image((torch.concat([z_decode, gt], axis=0)+ 1.) / 2., 'gt_encode.png')


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(loss_dict)

            if iter % conf['log_im_every'] == 0:
                model.eval()
                # log train
                pred = log_image(
                        model=model,
                        ddim_sampler=ddim_sampler,
                        control=control,
                        prompt_embed=prompt_embed,
                        ddim_steps=20,
                        strength=1.,
                        scale=15.,
                        eta=0.,
                    )
                        
                torchvision.utils.save_image((torch.concat([pred, gt], axis=0)+ 1.) / 2., str(train_img_dir / f'{iter:06d}.png'))

                # log test
                test_iter += 1
                _, _, test_data = test_dataset[TEST_IDS[test_iter % len(TEST_IDS)]]
                for k, v in test_data.items():
                    try:
                        test_data[k] = v.cuda()
                    except:
                        test_data[k] = v

                control = einops.rearrange(test_data['dwpose_im'][None,...], 'b h w c -> b c h w')
                B, _, H, W = control.shape
                gt = einops.rearrange(test_data['rgb'][None,...].reshape([B, H, W, 3]), 'b h w c -> b c h w').float()
                gt = gt * 2. - 1.
                if NO_HEAD:
                    head_mask = einops.rearrange(ground_truth['head_mask'].reshape([B, H, W, 1]), 'b h w c -> b c h w')
                    gt[head_mask.repeat([1,3,1,1]) > 0.] = 1.

                pred = log_image(
                        model=model,
                        ddim_sampler=ddim_sampler,
                        control=control,
                        prompt_embed=prompt_embed,
                        ddim_steps=20,
                        strength=1.,
                        scale=15.,
                        eta=0.,
                    )
                
                torchvision.utils.save_image((torch.concat([pred, gt], axis=0)+ 1.) / 2., str(train_img_dir / f'{iter:06d}_test.png'))

                model.train()

        if ep_i % conf['ckpt_every_ep'] == 0 and ep_i != 0 and not conf['no_ckpt']:
            torch.save(model.state_dict(), str(ckpt_dir / 'latest.pth'))
            torch.save(optimizer.state_dict(), str(ckpt_dir / 'opt_latest.pth'))

    # saving model ckpt
    if not conf['no_ckpt']:
        torch.save(model.state_dict(), str(ckpt_dir / 'latest.pth'))
        torch.save(optimizer.state_dict(), str(ckpt_dir / 'opt_latest.pth'))


    # end of test logging
    model.eval()
    test_log_dir = log_dir / 'test'
    test_log_dir.mkdir(exist_ok=True, parents=True)

    for i, model_input , ground_truth in tqdm(test_dataloader, desc=f"end of training logging"):
        for k, v in ground_truth.items():
            try:
                ground_truth[k] = v.cuda()
            except:
                ground_truth[k] = v

        control = einops.rearrange(ground_truth['dwpose_im'], 'b h w c -> b c h w')
        B, _, H, W = control.shape
        gt = einops.rearrange(ground_truth['rgb'].reshape([B, H, W, 3]), 'b h w c -> b c h w').float()
        gt = gt * 2. - 1.
        if NO_HEAD:
            head_mask = einops.rearrange(ground_truth['head_mask'].reshape([B, H, W, 1]), 'b h w c -> b c h w')
            gt[head_mask.repeat([1,3,1,1]) > 0.] = 1.

                
        pred = log_image(
                model=model,
                ddim_sampler=ddim_sampler,
                control=control,
                prompt_embed=prompt_embed,
                ddim_steps=20,
                strength=1.,
                scale=15.,
                eta=0.,
            )
        
        # import pdb; pdb.set_trace()
        
        torchvision.utils.save_image((torch.concat([pred, gt], axis=0)+ 1.) / 2., str(test_log_dir / f"{model_input['img_name'].item():06d}_test.png"))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='conf/finetune/default.conf')
    opt = parser.parse_args()

    conf = ConfigFactory.parse_file(opt.conf)

    wandb.init(
        project='finetune_diffusion_toy', 
        name=conf['run_name'], 
        group=conf['subject'], 
        config=conf.as_plain_ordered_dict(), 
        mode=conf['wandb_mode'],
        )
    train(conf)