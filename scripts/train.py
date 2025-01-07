#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import time
from safetensors import safe_open
from omniflow.utils.ema import EMAModel
import torch.utils.data
from transformers.trainer_pt_utils import LabelSmoother
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection,CLIPVisionModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast,CLIPImageProcessor
import torch.nn.functional as F
import diffusers
from diffusers import (
    AutoencoderKL,
)
from omniflow.utils.scheduler import OmniFlowMatchEulerDiscreteScheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from omniflow.models.omni_flow import OmniFlowTransformerModel
from omniflow.pipelines.omniflow_pipeline import  OmniFlowPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
import torch.distributed as dist
import glob
from omniflow.models.audio_vae import load_audio_vae
from omniflow.utils.text_encode import encode_prompt_train,cat_and_pad,encode_prompt_for_decoder
if is_wandb_available():
    import wandb
from torch import nn
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)
# VAL_FILES = glob.glob('Your Validation Folder/*.jpg')
# VAL_FILES_AUDIO = sorted(glob.glob('Your Validation Folder/*.mp3'))
VAL_FILES = ['./assets/girl.png']
VAL_FILES_AUDIO = ['./assets/car engine.mp3']
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from itertools import chain
import random
from omniflow.models.text_vae import LLamaForLatentConnector

import torch
from omniflow.models.encoders import LanguageBindAudioProcessor,LanguageBindAudio
import yaml
from transformers import AutoTokenizer
from transformers import AutoConfig
def load_yaml(fp: str):
    with open(fp, 'r') as file:
        data = yaml.safe_load(file)
    return data


def n_get_sigmas(noise_scheduler_copy,device,timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
    
def n_compute_text_embeddings(device,prompt, text_encoders, tokenizers,add_token_embed=True,train=False):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt_train(
            text_encoders, tokenizers, prompt, args.max_sequence_length,add_token_embed=add_token_embed,
            normalize=True,drops = list(
                np.random.rand() > 0.5 for _ in range(4)
            ) if train else [False,False,False,False]
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds
import torch
from einops import rearrange


def all_gather_cat(input):
    output = [torch.zeros_like(input) \
        for _ in range(dist.get_world_size())]
    dist.all_gather(output, input)
    return torch.cat(tuple(output))
class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)
    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
def prepare_inputs(transformer,args,
                   text_encoder_one,text_encoder_two,text_encoder_three,
                   device,batch,
                   vae,tokenizer_three,text_encoders,
                   tokenizers,
                   tokenizer_one,
                   tokenizer_two,
                   weight_dtype,
                   noise_scheduler_copy,
                   noise_scheduler,
                   image_encoder,
                   audio_vae_factor,
                   audiovae,
                   text_vae_tokenizer,
                   text_vae,  
                   audio_encoder,  
                   anchor=False,
                   mm_encoder=None,
                   ):
    with torch.no_grad():
        models_to_accumulate = [transformer]

        task = batch['task']
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
        prompts = np.array(batch["prompts"]) # img
        prompts2 = np.array(batch["prompts2"]) # aud
    
        bsz = len(prompts)
        # image input
        model_input = vae.encode(pixel_values.to(vae.device)).latent_dist.sample() # [4, 16, 64, 64]
        model_input = model_input * vae.config.scaling_factor
        model_input = model_input.to(dtype=weight_dtype)

        # audio input
        raw_audio_embeds = batch['audio'].to(model_input.device) # [N, 1, 1024, 64]
        raw_audio_embeds = audiovae.encode(raw_audio_embeds.float()).latent_dist.sample().mul_(audiovae.config.scaling_factor) 
        raw_audio_embeds = raw_audio_embeds.to(model_input)
        
        # Sample noise that we'll add to the latents
        bsz = model_input.shape[0]
        
        add_token_embed=True 
        # sample bsz * 3 noise levels for different modalities
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz * 3,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        if args.uniform_flow:
            indices = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz*3,), device='cpu', dtype=torch.long
                    )
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
        timesteps,timesteps_text,timesteps_audio = timesteps.chunk(3)    
        sigmas = n_get_sigmas(noise_scheduler_copy,device,timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        sigma_text= n_get_sigmas(noise_scheduler_copy,device,timesteps_text, n_dim=model_input.ndim, dtype=model_input.dtype)
        sigmas_audio = n_get_sigmas(noise_scheduler_copy,device,timesteps_audio, n_dim=model_input.ndim, dtype=model_input.dtype)
        loss_text_factor= 1
        loss_aud_factor = 1
        loss_img_factor = 1
    # setup
    can_generate_text = True
    if np.random.rand() < 0.1:
        can_generate_text = False
    prepend_clip = False
    if task  in ['text2img','text2aud'] and np.random.rand() < 0.5:
        prepend_clip = True
        
    # set up proper path
    if task in ['text2img','text2aud']:
        loss_text_factor = 0
        if np.random.rand() < 0.8 or prepend_clip:
            sigma_text = sigma_text * 0
            timesteps_text = timesteps_text * 0
    
    if task in ['img2aud','aud2img']:
        loss_text_factor = 0
        sigma_text = sigma_text * 0 + 1
        timesteps_text = timesteps_text * 0 + 1000
        
    if batch['drop_text'] is not None:
        timesteps_text[batch['drop_text']] = 1000
        sigma_text[batch['drop_text']] =  1
    
    if batch['drop_aud'] is not None:
        timesteps_audio[batch['drop_aud']] = 1000
        sigmas_audio[batch['drop_aud']] =  1
        
    if batch['drop_img'] is not None:
        timesteps[batch['drop_img']] = 1000
        sigmas[batch['drop_img']] =  1
        
    if task in ['img2text','img2aud']:
        loss_img_factor = 0
        if np.random.rand() < 0.8:
            sigmas = sigmas * 0
            timesteps = timesteps * 0
        
    if task in ['text2aud','aud2text']:
        loss_img_factor = 0
        sigmas = sigmas * 0 + 1
        timesteps = timesteps * 0 + 1000
          
    if task in ['aud2text','aud2img']:
        loss_aud_factor = 0
        if np.random.rand() < 0.8:
            sigmas_audio = sigmas_audio * 0
            timesteps_audio = timesteps_audio * 0
        
    if task in ['text2img','img2text']:
        loss_aud_factor = 0
        sigmas_audio = sigmas_audio * 0 + 1
        timesteps_audio = timesteps_audio * 0 + 1000
        
    if batch['name'] in ['flux','t2i_2m']: # 't2i_2m'
        loss_text_factor = loss_text_factor * 0
        
    if batch['name'] in ['audiocaps_acc']:
        loss_aud_factor = loss_aud_factor * 0
    if not can_generate_text or prepend_clip:
        loss_text_factor = loss_text_factor * 0

    if task in ['img2text','img2aud']:
        pool_mode = 'img'
    elif task in ['aud2img','aud2text']:
        pool_mode = 'aud'
    else:
        pool_mode = 'text'
        
    if task  == ['text2img','aud2img']  and np.random.rand() < 0.3:
        pool_mode = 'img'
    elif task  in ['text2aud','img2aud'] and np.random.rand() < 0.3:
        pool_mode = 'aud'
        
    if task == 'any2any':
        pool_mode = np.random.choice(['img','aud','text'])
        if np.random.rand() < 0.2:
            # 
            if np.random.rand() < 0.5:
                timesteps = timesteps * 0 + 1000
                sigmas = sigmas * 0 + 1
            else:
                loss_img_factor = 0
                timesteps = timesteps * 0
                sigmas = sigmas * 0 
        if np.random.rand() < 0.2:
            #
            if np.random.rand() < 0.5:
                timesteps_audio = timesteps_audio * 0 + 1000
                sigmas_audio = sigmas_audio * 0 + 1
            else:
                loss_aud_factor = 0
                timesteps_audio = timesteps_audio * 0 
                sigmas_audio = sigmas_audio * 0 
        if np.random.rand() < 0.2:
            # loss_text_factor = 0
            if np.random.rand() < 0.5:
                timesteps_text = timesteps_text * 0 + 1000
                sigma_text = sigma_text * 0 + 1
            else:
                loss_text_factor = 0
                timesteps_text = timesteps_text * 0 
                sigma_text = sigma_text * 0
        use_img_caption = (timesteps < timesteps_audio ) *( torch.rand_like(timesteps) < 0.3)
        use_img_caption =  use_img_caption.cpu().numpy()
        prompts[use_img_caption]=prompts2[use_img_caption]
    prompts = prompts.tolist()
    target_labels = tokenize_prompt(tokenizer_three, prompts)
    target_labels = target_labels.to(device)
        

        
        
    # Text encode

    prompt_embeds, pooled_prompt_embeds = n_compute_text_embeddings(
        device,
        prompts, text_encoders, tokenizers,add_token_embed=add_token_embed,
        train=False
    )    
    prompt_embeds_vae = text_vae.encode(prompts,input_ids=None,tokenizer=tokenizer_three,sample=True)
    prompt_embeds_vae_uncond = text_vae.encode(prompts,input_ids=None,tokenizer=tokenizer_three,drop=True)

        
    if not can_generate_text:
        prompt_embeds_vae *= 0

    
    l_vae = prompt_embeds_vae.shape[1]
    
    

        
    if prepend_clip:
        prompt_embeds = cat_and_pad([prompt_embeds,prompt_embeds_vae],max_dim=4096)
        prompt_embeds_vae_uncond = None
        prompt_embeds_uncond = None
    else:
        prompt_embeds = cat_and_pad([prompt_embeds_vae],max_dim=4096)
        prompt_embeds_uncond = cat_and_pad([prompt_embeds_vae_uncond],max_dim=4096)
        


    # Sample a random timestep for each image
    # for weighting schemes where we sample timesteps non-uniformly
    targets = encode_prompt_for_decoder(prompts,text_vae_tokenizer,device=transformer.device)
    target_labels = targets['labels']


    with torch.no_grad():
        #assert loss_text_factor + loss_img_factor + loss_aud_factor == 1,task
        if pool_mode == 'img':
                image_embeds = image_encoder(pixel_values=batch['clip_values'].to(image_encoder.dtype).to(image_encoder.device)).image_embeds
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
                pooled_prompt_embeds[...,:image_embeds.shape[-1]] =image_embeds
                if batch['drop_img'] is not None:
                    pooled_prompt_embeds[batch['drop_img']] = 0
        elif pool_mode == 'aud':
                audio_embeds = audio_encoder.get_image_features(pixel_values=batch['audio_clip'].to(audio_encoder.dtype).to(audio_encoder.device)) # 768
                pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
                pooled_prompt_embeds[...,:audio_embeds.shape[-1]] =audio_embeds
                if batch['drop_aud'] is not None:
                    pooled_prompt_embeds[batch['drop_aud']] = 0
        else:
            if batch['drop_text'] is not None:
                pooled_prompt_embeds[batch['drop_text']] = 0
            
    # independently drop 
        
   
    pooled_prompt_embeds = pooled_prompt_embeds.detach()
            
            

    drop_pool = (torch.rand(pooled_prompt_embeds.shape[0]) < 0.85).view(-1,1).to(pooled_prompt_embeds)

    pooled_prompt_embeds = pooled_prompt_embeds * drop_pool
    sigma_text = sigma_text.view(-1,1,1)
    
    noise = torch.randn_like(model_input)
    noise_text = torch.randn_like(prompt_embeds)
    
    
    noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input
    noisy_prompt_embeds = sigma_text * noise_text +  (1.0 -sigma_text ) * prompt_embeds 

    noise_audio = torch.randn_like(raw_audio_embeds)
    sigmas_audio = sigmas_audio.view(-1,1,1,1)
    noisy_audio_embeds = sigmas_audio * noise_audio + (1.0 -sigmas_audio ) * raw_audio_embeds 

    # if can_generate_text:
    noisy_prompt_embeds[:,-l_vae:,prompt_embeds_vae.shape[-1]:]=0
    noisy_prompt_embeds = noisy_prompt_embeds.detach()
    # breakpoint()
    return (noisy_model_input,timesteps,timesteps_text,timesteps_audio,noisy_prompt_embeds,
        noisy_audio_embeds,sigma_text,prompt_embeds,pooled_prompt_embeds,targets,prompt_embeds_uncond,
        sigmas,sigmas_audio,model_input,
        loss_img_factor,
        loss_text_factor,
        loss_aud_factor,
        noise_scheduler_copy,
        raw_audio_embeds,
        task,
        prompts,
        noise,
        noise_text,
        noise_audio,
        target_labels,
        prompt_embeds_vae_uncond
        )
    
def compute_decode_loss_weight(timesteps_text,num_train_timesteps):
    _t = timesteps_text /  num_train_timesteps #scheduler.config.num_train_timesteps
    _tmax = 0.6
    decode_loss_weight = 0.5 * (torch.cos(torch.pi * torch.clip(_t /_tmax ,0,1 )) + 1)
    return decode_loss_weight


class WeightedLabelSmoother(LabelSmoother):
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels,sample_weight=None, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        if sample_weight is not None:
            bsz = nll_loss.shape[0]
            nll_loss = nll_loss * sample_weight.view(bsz,1,1)
            smoothed_loss = smoothed_loss * sample_weight.view(bsz,1,1)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss

def compute_loss(transformer,noisy_model_input,timesteps,timesteps_text,timesteps_audio,noisy_prompt_embeds,
                 noisy_audio_embeds,sigma_text,prompt_embeds,pooled_prompt_embeds,targets,prompt_embeds_uncond,
                 sigmas,sigmas_audio,model_input,
                 loss_img_factor,
                 loss_text_factor,
                 loss_aud_factor,
                 noise_scheduler_copy,
                 last_lr,
                 raw_audio_embeds,
                 task,
                 prompts,
                noise,
                noise_text,
                noise_audio,
                text_vae,
                target_labels,
                do_decode,
                prompt_embeds_vae_uncond,
                precondition_text_outputs=False,
                anchor=False,
                batch=None,
                 ):
    # transformer.train()
    output_dict = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            timestep_text=timesteps_text,
            timestep_audio=timesteps_audio,
            encoder_hidden_states=noisy_prompt_embeds,
            audio_hidden_states=noisy_audio_embeds,
            sigma_text=sigma_text,
            target_prompt_embeds=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            targets=targets,
            return_dict=False,
            use_text_output=True,
            prompt_embeds_uncond = None if np.random.rand() < 0.5 else prompt_embeds_uncond,
            detach_logits=not anchor,
            split_cond=False,
            text_vae=text_vae,
            text_x0=precondition_text_outputs,
            decode_text=True
        ) # B X 16 X 64 X 64 
    # assert anchor
    model_pred = output_dict['output']
    model_pred_audio = output_dict['audio_hidden_states']
   
    model_pred_text =  output_dict['model_pred_text']
    logits  = output_dict['logits']
    logits_labels = output_dict['logits_labels']
    
    v_theta = noise - model_input
    
    v_theta_audio = noise_audio - raw_audio_embeds
    raw_text_embeds = prompt_embeds[...,:model_pred_text.shape[-1]]
    noise_text = noise_text[...,:model_pred_text.shape[-1]]

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
    weighting_text = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigma_text)
    weighting_audio = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas_audio)
    # flow matching loss
    if batch['drop_img'] is not None:
        weighting[batch['drop_img']] = 0

    loss_img =  (weighting.float() * (model_pred - v_theta.float()) ** 2).mean()

    with torch.no_grad():
        weighting_text = weighting_text.view(-1,1,1)
        if batch['drop_text'] is not None:
            weighting_text[batch['drop_text']] = 0
    loss_text_norm_acc = None
    if precondition_text_outputs:
        loss_text = (weighting_text.float() * (model_pred_text.float() - raw_text_embeds.float().detach()) ** 2).mean()
        norm_1 = F.normalize(model_pred_text,dim=-1,eps=1e-4).float() # N L D
        norm_2 = F.normalize(raw_text_embeds,dim=-1,eps=1e-4).float().detach()
        loss_text_norm = (weighting_text.float() * (norm_1 - norm_2) ** 2).mean()
        loss_text_norm = loss_text_norm * 0.1#+ regularization
    
    else:
        v_theta_text = noise_text - raw_text_embeds # v_theta_text  
        loss_text = (weighting_text.float() * (model_pred_text.float() - v_theta_text.float()) ** 2).mean()
        loss_text_norm = 0
    weighting_audio = weighting_audio.view(-1,1,1,1)
    
    loss_audio = (weighting_audio.float() * (model_pred_audio - v_theta_audio.float()) ** 2).mean()
    

    if anchor:
        logits = output_dict['logits']
        logits_labels = output_dict['logits_labels']
        # 0.1
        label_smoother = WeightedLabelSmoother(epsilon=0.0, ignore_index= -100)
        decode_loss_tgt_weight = torch.ones(len(timesteps_text)).to(logits)
        if anchor:
            decode_loss_weight = torch.ones(len(timesteps_text)).to(logits)
        else:
            decode_loss_weight = compute_decode_loss_weight(timesteps_text,noise_scheduler_copy.config.num_train_timesteps)
        if batch['drop_text'] is not None:
            decode_loss_weight[batch['drop_text']] = 0
            decode_loss_tgt_weight[batch['drop_text']] = 0
        decode_loss_pred = label_smoother([logits], target_labels, shift_labels=True,
                                        sample_weight=decode_loss_weight)
        decode_loss_tgt = label_smoother([logits_labels], target_labels, shift_labels=True,sample_weight=decode_loss_tgt_weight)
        decode_loss = None
    else:
        decode_loss_pred = 0
        decode_loss_tgt = 0
        decode_loss=None

    loss = loss_img * loss_img_factor + \
    (loss_text+loss_text_norm )  *loss_text_factor   + \
        loss_audio * loss_aud_factor  + \
        ( decode_loss_tgt + decode_loss_pred)  *loss_text_factor  * 0.1
        

    logs = {"loss": loss.detach().item(), "lr": last_lr,
                "loss_aud_factor":loss_aud_factor,
                "loss_img_factor":loss_img_factor,
                "loss_text_factor":loss_text_factor
                }
    if loss_text_factor > 0:
        logs.update({
                "loss_text":loss_text.detach().item(),
                "loss_text_norm":loss_text_norm.detach().item(),
                
        })
        with torch.no_grad():
            logs.update(
                {
                    "text_embed_mean":raw_text_embeds.mean().item(),
                    "text_embed_std":raw_text_embeds.std().item(),
                }
            )
        if loss_text_norm_acc:
            logs['loss_text_norm_acc'] = loss_text_norm_acc
        if anchor:
            logs.update({
                "decode_loss_tgt":decode_loss_tgt.detach().item(),
                "decode_loss":decode_loss_pred.detach().item(),
            })
    if loss_img_factor > 0:
        logs.update({
                "loss_img":loss_img.detach().item(),
        })
    if loss_aud_factor > 0:
        logs.update({
                "loss_audio":loss_audio.detach().item(),
        })
    with torch.no_grad():
        model_pred = model_pred * (-sigmas) + noisy_model_input
        model_pred_audio = model_pred_audio * (-sigmas_audio) + noisy_audio_embeds
        target = model_input
    return loss,decode_loss,logs,task,model_pred,logits,target,prompts,model_pred_audio,model_pred_audio,raw_audio_embeds,model_pred_text,raw_text_embeds

def forward_pass(transformer,args,
                   text_encoder_one,text_encoder_two,text_encoder_three,
                   accelerator,batch,
                   vae,tokenizer_three,text_encoders,
                   tokenizers,
                   tokenizer_one,
                   tokenizer_two,
                   weight_dtype,
                   noise_scheduler_copy,
                   noise_scheduler,
                   image_encoder,
                   audio_vae_factor,
                   audiovae,
                   text_vae_tokenizer,    
                    last_lr,
                    text_vae,
                    audio_encoder,
                    do_decode=False,
                    precondition_text_outputs=False,
                    anchor=False,
                    mm_encoder=None
                ):

    (noisy_model_input,timesteps,timesteps_text,timesteps_audio,noisy_prompt_embeds,
            noisy_audio_embeds,sigma_text,prompt_embeds,pooled_prompt_embeds,targets,prompt_embeds_uncond,
            sigmas,sigmas_audio,model_input,
            loss_img_factor,
            loss_text_factor,
            loss_aud_factor,
            noise_scheduler_copy,
            raw_audio_embeds,
            task,
            prompts,
            noise,
            noise_text,
            noise_audio,
            target_labels,
            prompt_embeds_vae_uncond,
        ) =prepare_inputs(transformer,args,
                    text_encoder_one,text_encoder_two,text_encoder_three,
                    accelerator,batch,
                    vae,tokenizer_three,text_encoders,
                    tokenizers,
                    tokenizer_one,
                    tokenizer_two,
                    weight_dtype,
                    noise_scheduler_copy,
                    noise_scheduler,
                    image_encoder,
                    audio_vae_factor,
                    audiovae,
                    text_vae_tokenizer,    
                    text_vae, 
                    audio_encoder,
                    anchor,
                    mm_encoder=mm_encoder
                    )
    loss,decode_loss,logs,task,model_pred,logits,target,prompts,model_pred_audio,model_pred_audio,raw_audio_embeds,model_pred_text,raw_text_embeds = compute_loss(transformer,noisy_model_input,timesteps,timesteps_text,timesteps_audio,noisy_prompt_embeds,
                 noisy_audio_embeds,sigma_text,prompt_embeds,pooled_prompt_embeds,targets,prompt_embeds_uncond,
                 sigmas,sigmas_audio,model_input,
                 loss_img_factor,
                 loss_text_factor,
                 loss_aud_factor,
                 noise_scheduler_copy,
                 last_lr,
                 raw_audio_embeds,
                 task,
                 prompts,
                noise,
                noise_text,
                noise_audio,
                text_vae,
                target_labels,
                do_decode,
                prompt_embeds_vae_uncond,
                precondition_text_outputs=precondition_text_outputs,
                anchor=anchor,
                batch=batch,
                 )
    return loss,decode_loss,logs,task,model_pred,logits,target,prompts,model_pred_audio,model_pred_audio,raw_audio_embeds,model_pred_text.detach(),raw_text_embeds.detach()


class MultiDatasetBatchSampler(BatchSampler):
    def __init__(self, datasets, weights, batch_size,shuffle=True,local_rank=0,world_size=1,
                 seed=12345):
        self.datasets_length = np.array([len(x) for x in datasets])
        self.datasets_start_index = np.cumsum( self.datasets_length)
        self.datasets_start_index = np.concatenate([[0],self.datasets_start_index])
        self.datasets_start_index,self.length = self.datasets_start_index[:-1],self.datasets_start_index[-1]
        self.dataset_weight = torch.tensor(weights).float()
        self.batch_size = batch_size
        self.local_rank = local_rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
    def set_seed(self,seed):
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        
    def __iter__(self):
        batch = []
        # if self.shuffle:
        #     random.shuffle(dataset_order)
        n_batches = self.length // self.batch_size
        for idx in range(n_batches):
            select_dataset = torch.multinomial(self.dataset_weight,self.world_size,replacement=True,generator=self.generator)[self.local_rank].item()
            selected_index = self.rng.integers(0,self.datasets_length[select_dataset],(self.world_size,self.batch_size))[self.local_rank]
            selected_index = selected_index + self.datasets_start_index[select_dataset]
            #print(f"RANK:{self.local_rank }",selected_index)
            indices =  selected_index.tolist()
            print(f"Local Rank {self.local_rank} Batch ID {idx} : {indices}")
            yield indices

    def __len__(self):
        return self.length // self.batch_size




def center_crop_and_resize(image_path, desired_size):
    # Open the image
    image = Image.open(image_path).convert('RGB')
    
    # Get dimensions
    width, height = image.size
    
    # Calculate the size of the largest square
    new_side = min(width, height)
    
    # Calculate the cropping box
    left = (width - new_side) / 2
    top = (height - new_side) / 2
    right = (width + new_side) / 2
    bottom = (height + new_side) / 2
    
    # Crop the image to the largest square
    image = image.crop((left, top, right, bottom))
    
    # Resize the image to the desired size
    image = image.resize((desired_size, desired_size))
    
    return image

@torch.no_grad()
def mel_spectrogram_to_waveform(
        vae, mel,
    ):
        if len(mel.size()) == 4:
            mel = mel.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = vae.vocoder(mel)
        waveform = waveform.cpu().detach().numpy()
        return waveform
   
@torch.no_grad()
def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    global_step,
    is_final_validation=False,
    prefix='',
    do_image=True,
    do_audio=True,
    do_text=True,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    #pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()
    all_texts = []
    specs = []
    audio_guidance_scale = [1,2,4,7]
    with autocast_ctx:
        phase_name = f"test_{prefix}" if is_final_validation else f"validation_{prefix}"
        if do_image:
            images = [pipeline(**pipeline_args, generator=generator,add_token_embed=1,height=args.resolution,width=args.resolution).images[0] for _ in range(args.num_validation_images)]    
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(phase_name, np_images, global_step, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            phase_name: [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                            ]
                        }
                    )
        if do_image: # a 2 i
            images = []
            for ref in [
                'assets/car engine.mp3', # carr
            ]:
                img = pipeline(prompt='',input_aud=ref,add_token_embed=1,height=512,width=512,task='a2i')[0][0]
                images.append(img) 
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(phase_name, np_images, global_step, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            'a2i'+phase_name: [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                            ]
                        }
                    )
        if do_text:
            all_texts = []
            for _idx,img_path in enumerate(VAL_FILES):
                #input_img = Image.open(img_path).convert('RGB')
                input_img = center_crop_and_resize(img_path,args.resolution)
                texts,_ = pipeline("",
                                   guidance_scale=2,
                                   input_img=input_img,height=512,width=512,add_token_embed=1,task='i2t')#[0]
                texts2,text_uncond = pipeline("",
                                    input_img=input_img,height=512,width=512,add_token_embed=1,task='i2t',
                                    guidance_scale=1
                                    #,num_inference_steps=100
                                    )
                texts = texts[0]
                texts2 = texts2[0]
                text_uncond = text_uncond[0]
                # breakpoint()
                all_texts.append((texts,texts2,text_uncond))
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    text_table = []
                    for row in all_texts:
                        text_table.append(row)
                    text_table = pd.DataFrame(text_table)
                    text_table.columns = ["text",'text_nocfg','text_uncond']
                    html = wandb.Html(text_table.to_html(),inject=True)
                    tracker.log({f"text_{prefix}":html})
           
        if do_audio:
            for cfg in audio_guidance_scale:
                spec,_ = pipeline(prompt="Politican making speeach",
                                            add_token_embed=1,
                                            height=args.resolution,width=args.resolution,
                                            task='t2a',
                                            num_inference_steps=50,
                                            guidance_scale=cfg)
                specs.append(torch.tensor(spec[0][0]))  
            if wandb.run is not None:
                wandb_log_spec(specs,f"audio_{prefix}") 

        



    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    parser.add_argument(
        "--tokenizer",
        type=str,
        default='/localhome/jacklishufan/TinyLlama_v1.1' ,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    
    
    parser.add_argument(
        '--debug_val',action='store_true'
    )
    parser.add_argument(
        '--skip_load_text_decoder',action='store_true'
    )
    
    
    parser.add_argument(
        '--anchor',action='store_true'
    )


    parser.add_argument(
        '--image_bind',action='store_true'
    )
    
    
    
    
    parser.add_argument(
        '--precondition_text_outputs',action='store_true'
    )
    #L1260 of https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_sd3.py#L1620
    parser.add_argument(
        "--text_decoder_ckpt",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )
    
    

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument("--ema_interval", type=int, default=100, help="How many times to repeat the training data.")


    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo",
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a cute dog on a ship",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=250,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--ema_momentum",
        type=float,
        default=0.9999,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--use_ema",
        action='store_true'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=0, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        '--text_vae',
        type=str,
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--dev",
        type=str,
        default=None,
        help="Debug setup",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--uniform_flow",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    parser.add_argument(
        "--ema_init",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    
    
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    
    parser.add_argument('--ema_validation',action='store_true')
    parser.add_argument('--ema_start',type=int,default=-1)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--lr_text_factor",type=float,default=1)
    parser.add_argument("--lr_aud_factor",type=float,default=1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        is_train=True,
        image_processor=None,
        dataset = 'text2img',
        name = 'none',
        audio=None,
        audio_processor=None,
        audio_processor_clip=None,
        random_flip = False,
        task_weight=[0.5,0.5],
    ):
        self.size = size
        self.center_crop = center_crop
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.audio_processor_clip = audio_processor_clip
        self.dataset = dataset
        self.name = name
        self.task_weight = task_weight
        
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        if os.path.isdir(instance_data_root):
            df = pd.read_csv(os.path.join(instance_data_root,'metadata.csv')) #
        else:
            df = pd.read_csv(instance_data_root) #
        keys = df.columns
        if 'coco_caption' in keys: # do not use raw text for LAION-COCO, use coco-style ones
          df['img_path'] = df[['file_name']].apply(lambda row: os.path.join(instance_data_root,*row), axis=1)
          df['caption'] = df['coco_caption']
        if 'caption_llava_short' in keys:
            df['caption'] = df['caption_llava_short']
        # breakpoint()
        if 'img_file' in keys:
            df['img_path'] = df['img_file']
            
        if 'audio' in keys:
            df['audio_data'] = df['audio']

          
        self.data = df#.to_dict(orient='records')
        transforms_list = []
        transforms_list.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR))
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        if random_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        self.is_train = is_train
        self.train_transforms = transforms.Compose(
            [
                *transforms_list,
                train_crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.audio_clip_shape = (1,3,112,1036)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        if 'img_path' in item:
            image = Image.open(item['img_path']).convert('RGB')
            if image.height == 1 or image.width == 1:
                image = image.resize((224,224))
            image_clip = self.image_processor(image,return_tensors="pt").pixel_values
            image = self.train_transforms(image)
            has_img = True
        else:
            zero_image = np.zeros((256,256,3),np.uint8)
            image = self.train_transforms(Image.fromarray(zero_image))
            image_clip = torch.zeros(1,3,224,224).float()
            has_img = False
        audio = torch.zeros(1,1,1024,64).float()
        if 'audio_path' in item:
            try:
                x = self.audio_processor.feature_extraction_vae(item['audio_path'])
                audio = x['fbank'].unsqueeze(0) # 1 1024 64 -> 1 1 1024 64
                #model.first_stage_model.freq_split_subband(x['fbank']).unsqueeze(1)
                audio_clip = self.audio_processor_clip([item['audio_path']])['pixel_values']
            except:
                audio = torch.zeros(1,1,1024,64).float()
                print(f"AUDIO CLIP VALUE ERROR {item['audio_path']} !!!!")
                audio_clip = torch.zeros(1,3,112,1036).float()
            has_audio = True
        else:
            audio = torch.zeros(1,1,1024,64).float()
            #audio_clip = torch.zeros(1,1024).float()
            audio_clip = torch.zeros(1,3,112,1036).float()
            has_audio = False
            
        if 'caption' in item and self.name not in [ 'audioset']:
            caption = str(item['caption'])
            if 'caption_alt' in item and np.random.rand() < 0.5:
                caption = item['caption_alt']
            if caption.startswith('@'):
                caption = caption[1:]
            if '@' in caption:
                caption = caption.replace('@',',')
            has_caption = True
        else:
            caption = ''
            has_caption = False
            
        if 'caption2' in item:
            caption2 = str(item['caption2'])
            has_caption2 = True
        else:
            caption2 = ''
            has_caption2 = False
            
        
        assert torch.tensor([has_caption,has_audio,has_img]).sum() >=2 or self.name in ['audioset'],self.name
        if self.dataset == 'any2any':
            assert torch.tensor([has_caption,has_audio,has_img,has_caption2]).sum() >=4
        payload = dict(
            image = image,
            has_image = image,
            caption=caption,
            image_clip = image_clip,
            audio = audio,
            has_audio=has_audio,
            dataset=self.dataset,
            audio_clip=audio_clip,
            has_caption=has_caption,
            caption2=caption2,
            weight=self.task_weight
        )
        return payload
    
def collate_fn(examples):
    pixel_values = [example["image"] for example in examples]
    clip_values = torch.cat([example["image_clip"] for example in examples])
    prompts = list([example["caption"] for example in examples])
    prompts2 = list([example["caption2"] for example in examples])


    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    audio = torch.cat([example["audio"] for example in examples])
    audio_clip = torch.cat([example["audio_clip"] for example in examples])
    all_datasets = np.unique([x['dataset'] for x in examples])
    assert len(all_datasets) == 1,all_datasets
    if examples[0]['dataset'] == 'text2img':
        task = np.random.choice([
            'text2img',
            'img2text',
        ],p=examples[0]['weight'])
    elif examples[0]['dataset'] == 'text2aud':
        task = np.random.choice([
            'text2aud',
            'aud2text',
        ],p=examples[0]['weight'])
        if not examples[0]['has_caption']:
            task = 'text2aud'
    elif examples[0]['dataset'] == 'img2aud':
        task = np.random.choice([
            'img2aud',
            'aud2img',
        ],p=examples[0]['weight'])
    elif examples[0]['dataset'] == 'any2any':
        task = 'any2any'
    else:
        raise AssertionError(f"Invalid Dataset:{examples[0]['dataset'] }")
    drop_img = drop_text = drop_aud = None

    if task in ['text2img','text2aud']:
        drop_text =  (np.random.rand(len(prompts))  <0.15).nonzero()[0]

    elif task in ['img2text','img2aud']:
        drop_img =  (np.random.rand(len(prompts))  <0.15).nonzero()[0]

    elif task in ['aud2text','aud2img']:
        drop_aud=  (np.random.rand(len(prompts))  <0.15).nonzero()[0]

    elif task == 'any2any':
        pass
    else:
        print(task)
        raise NotImplemented 
    batch = {"pixel_values": pixel_values, "prompts": prompts,"task":task,"clip_values":clip_values,
             "audio":audio,"audio_clip":audio_clip,
              'drop_img':drop_img,'drop_aud':drop_aud,"drop_text":drop_text,
              "name":examples[0]['dataset'],
              "drop_img":drop_img,
              "prompts2":prompts2
             }

    return batch
from matplotlib import pyplot as plt
def wandb_log_spec(specs,key):
    fig,axes = plt.subplots(1,len(specs),figsize=(len(specs)*3,3))
    for i,x in enumerate(specs):
        axes[i].imshow(x.cpu().detach().T.numpy(), aspect='auto')
    wandb.log({ key:wandb.Image(fig)
    })
    plt.close(fig)
class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def load_finetune_checkpoint(self, path):
    m = torch.load(path)['state_dict']
    model_dict = self.state_dict()
    for k in m.keys():
        if 'fc_vidout' in k or 'fc_total' in k:
            continue

        if k in model_dict:
            pname = k
            pval = m[k]
            model_dict[pname] = pval.clone().to(model_dict[pname].device)

    self.load_state_dict(model_dict)
def load_safe_tensors(fp,model):
    tensors = torch.load(fp,map_location='cpu')
    
    #res = model.load_state_dict(tensors,strict=False)
    
    #
    model_dict = model.state_dict()
    keys_to_pop = []
    for k,v in tensors.items():
        if k in model_dict and model_dict[k].shape != v.shape:
            print(f"----------ERROR: SIZE MISMATCH {k}: {model_dict[k].shape} {v.shape}----------")
            keys_to_pop.append(k)
    for k in keys_to_pop:
        tensors.pop(k)
    res = model.load_state_dict(tensors,strict=False)
    print(f"Loaded {fp}:{res}")
    del tensors
    torch.cuda.empty_cache()

def load_safe_tensors_ema(fp,model):
    tensors = torch.load(fp,map_location='cpu')
    res = model.load_state_dict(tensors)
    print(f"Loaded {fp}:{res}")
    del tensors
    torch.cuda.empty_cache()

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    second_accelerator = None
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        # args.pretrained_model_name_or_path,
        # subfolder="tokenizer",
        # revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        'google/flan-t5-large',
        revision=args.revision,
    )


    # Load scheduler and models
    noise_scheduler = OmniFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",shift=1
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    noise_scheduler_pipeline = copy.deepcopy(noise_scheduler)
    
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
        'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        projection_dim=768
       # args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = T5EncoderModel.from_pretrained('google/flan-t5-large')

    
    image_encoder = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',projection_dim=768)
    audio_encoder = LanguageBindAudio.from_pretrained('LanguageBind/LanguageBind_Audio_FT')
    audio_encoder.text_model = nn.Identity()

    mm_encoder = nn.Identity()
    mm_encoder.eval()
    image_processor=  CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    audio_processor_clip = LanguageBindAudioProcessor(audio_encoder.config)

    text_encoder_one.eval()
    text_encoder_two.eval()
    text_encoder_three.eval()
    tokenizer_three = T5TokenizerFast.from_pretrained(
        'google/flan-t5-large',
        revision=args.revision,
    )
    #text_decoder  = LlamaForTextDecoding.from_pretrained(args.text_decoder_ckpt)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    audiovae,audio_processor = load_audio_vae()
    text_vae_tokenizer=AutoTokenizer.from_pretrained(
        args.tokenizer 
    )
    text_vae_tokenizer.add_special_tokens({'pad_token': '[PAD]'})     

    config = AutoConfig.from_pretrained(args.text_vae)
    text_vae  = LLamaForLatentConnector._from_config(
                                                    config,
                                                    torch_dtype=torch.bfloat16)
    # text_vae = LLamaForLatentConnector.from_pretrained(args.text_vae,torch_dtype=torch.bfloat16)

    text_vae.prepare_tokenizer(text_vae_tokenizer)
    text_vae.set_encoder(text_encoder_three)
  
    vae_scale_factor = (
            2 ** (len(vae.config.block_out_channels) - 1) 
    )
    vae_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    model_cls = OmniFlowTransformerModel
    
    if args.dev == 'debug':
        print("Using Debug!!!")
        # Load a 2 layer model for code sanity check
        transformer = model_cls.from_config(
        'scripts/dev_transformer'
        )
    else:
        transformer = model_cls.from_config(
            args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant,
            low_cpu_mem_usage=False,
        )

    if not args.skip_load_text_decoder:
        if second_accelerator is  None:
            transformer.set_text_decoder(text_vae)
    if args.load and not args.dev:

        fp = args.load
        if os.path.isdir(fp):
            fp = os.path.join(fp,'transformer/diffusion_pytorch_model.bin')
            fp_ema = os.path.join(args.load,'ema_transformer.pt')
        load_safe_tensors(fp,transformer)
    if args.skip_load_text_decoder:
        if second_accelerator is  None:
            transformer.set_text_decoder(text_vae)


    transformer.requires_grad_(True)
    text_vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    audio_encoder.requires_grad_(False)
    mm_encoder.requires_grad_(False)
    audiovae.requires_grad_(True)
    vae.requires_grad_(False)

    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    
    text_vae.encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if args.use_ema and  accelerator.is_main_process:
        ema_transformer = EMAModel(transformer.parameters(),decay=args.ema_momentum)#.to(torch.bfloat16)
        if args.load and args.load_ema:
            load_safe_tensors_ema(fp_ema,ema_transformer)
            ema_transformer.decay = args.ema_momentum
        if args.ema_init:
            ema_transformer.copy_to(transformer.parameters())
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=torch.float32)
    audiovae.to(accelerator.device, dtype=torch.float32)
    text_vae.to(accelerator.device)
    mm_encoder.to(accelerator.device, dtype=torch.bfloat16)

    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    audio_encoder.to(accelerator.device, dtype=weight_dtype)
    audio_encoder.visual_projection.to(accelerator.device, dtype=weight_dtype)
    audio_encoder.vision_model.to(accelerator.device, dtype=weight_dtype)
    
        
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        text_vae.gradient_checkpointing_disable()
        image_encoder.gradient_checkpointing_enable()
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
            if args.use_ema and  accelerator.is_main_process:
                target = os.path.join(output_dir, "ema_transformer.pt")
                state_dict = ema_transformer.state_dict()
                torch.save(state_dict,target)
                del state_dict


            for i, model in enumerate(models):
                if isinstance(unwrap_model(model), OmniFlowTransformerModel):
                    unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"),safe_serialization=False)
                elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                    if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                        hidden_size = unwrap_model(model).config.hidden_size
                        if hidden_size == 768:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder"))
                        elif hidden_size == 1280:
                            unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_2"))
                    else:
                        unwrap_model(model).save_pretrained(os.path.join(output_dir, "text_encoder_3"))
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                
    def load_model_hook(models, input_dir):
        if args.use_ema and  accelerator.is_main_process:
            target = os.path.join(input_dir, "ema_transformer.pt")
            state_dict = torch.load(target,map_location='cpu')
            ema_transformer.load_state_dict(state_dict)
            ema_transformer.decay = args.ema_momentum
            del state_dict
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            if isinstance(unwrap_model(model), OmniFlowTransformerModel):
                load_model = OmniFlowTransformerModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
            elif isinstance(unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)):
                try:
                    load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder")
                    model(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                except Exception:
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(input_dir, subfolder="text_encoder_2")
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(input_dir, subfolder="text_encoder_3")
                            model(**load_model.config)
                            model.load_state_dict(load_model.state_dict())
                        except Exception:
                            raise ValueError(f"Couldn't load the model of type: ({type(model)}).")
            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    
    if  args.lr_text_factor != 1 or  args.lr_aud_factor != 1:
        text_parameters = []
        aud_parameters = []
        img_parameters = []
        text_keys = ['text','context','add_k','add_q','add_v','time_image_embed']
        aud_keys = ['aud','audio']
        text_decoder_keys = ['text_decoder']
        text_decoder_parms = []
        for name, param in transformer.named_parameters():
            if any([x in name for x in text_decoder_keys]):
                text_decoder_parms.append(param)
            elif any([x in name for x in text_keys]) and 'time_text_embed' not in name:
                text_parameters.append(param)
            elif  any([x in name for x in aud_keys]):
                aud_parameters.append(param)
            else:
                img_parameters.append(param)
        transformer_parameters_with_lr = [
            {"params": text_parameters, "lr": args.learning_rate * args.lr_text_factor},  # 10x learning rate for 'text' params
            {"params": aud_parameters, "lr": args.learning_rate  * args.lr_aud_factor},    # Regular learning rate for the rest
            {"params": img_parameters, "lr": args.learning_rate}    # Regular learning rate for the rest,
        ]
    else:
        transformer_parameters_with_lr = [{"params": transformer.parameters(), "lr": args.learning_rate}]


    params_to_optimize = transformer_parameters_with_lr

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )
        
    def build_dataset(
        instance_data_root='',
        dataset='text2img',
        name='',   
        task_weight=[0.5,0.5],
    ):
        return DreamBoothDataset(
            instance_data_root=instance_data_root,
            instance_prompt=args.instance_prompt,
            class_prompt=args.class_prompt,
            class_data_root=None,
            class_num=args.num_class_images,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            random_flip=args.random_flip,
            image_processor=image_processor,
            dataset=dataset,
            name=name,
            audio_processor=audio_processor,
            audio_processor_clip=audio_processor_clip,
            task_weight=task_weight,
        )
        
    dataset_config = load_yaml(args.instance_data_dir)
    all_datasets = {}
    for k,v in dataset_config['datasets'].items():
        if k in dataset_config['weights']:
            all_datasets[k] = build_dataset(**v)
    train_datasets_weights = []
    for  k,v in dataset_config['weights'].items():
        train_datasets_weights.append(
            (
                all_datasets[k],v,k
            )
        )

    train_datasets = list(x[0] for x in train_datasets_weights)
    weights = list(x[1] for x in train_datasets_weights)
    for _d,_,_n in train_datasets_weights:
        print(f"Sanity Check:{_n},{_d.task_weight}")
        r = _d[0]
    print(f"A:{accelerator.process_index}")
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    sampler = MultiDatasetBatchSampler(train_datasets,weights=weights,batch_size=args.train_batch_size,shuffle=True,
                                               local_rank=accelerator.process_index,world_size=accelerator.num_processes,
                                               seed=int(time.time()))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=args.train_batch_size,
        # shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
        batch_sampler=sampler
    )
    
    
    
    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.


    try:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_batch_size
        
    except:
        pass
    accelerator.state.deepspeed_plugin.deepspeed_config['universal_checkpoint'] = True
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer,optimizer, lr_scheduler
    )

    
    optimizer2,scheduler2 =None,None

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    if accelerator.is_main_process:
        tracker_name = "dreambooth-sd3"
        accelerator.init_trackers(tracker_name, config=vars(args))
        wandb.config.update(dataset_config)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            acc2_path = os.path.join(args.output_dir, path,'acc2')
            if os.path.exists(acc2_path):
                second_accelerator.load_state(acc2_path)
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps* args.gradient_accumulation_steps),
        initial=initial_global_step* args.gradient_accumulation_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    do_ema = False
    audio_vae_factor = 1
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        do_log_img = 0
        do_log_text=0
        do_log_aud = 0
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                forward_kwargs = dict(
                    args=args,
                   text_encoder_one=text_encoder_one,
                   text_encoder_two=text_encoder_two,
                   text_encoder_three=text_encoder_three,
                   accelerator=accelerator.device,
                   batch=batch,
                   vae=vae,
                   tokenizer_three=tokenizer_three,
                   text_encoders=text_encoders,
                   tokenizers=tokenizers,
                   tokenizer_one=tokenizer_one,
                   tokenizer_two=tokenizer_two,
                   weight_dtype=weight_dtype,
                   noise_scheduler_copy=noise_scheduler_copy,
                   noise_scheduler=noise_scheduler,
                   image_encoder=image_encoder,
                   audio_vae_factor=audio_vae_factor,
                   audiovae=audiovae,
                   text_vae_tokenizer=text_vae_tokenizer,   
                   last_lr=lr_scheduler.get_last_lr()[0],
                   text_vae=text_vae,
                   audio_encoder=audio_encoder,
                   do_decode = second_accelerator is not None,
                   precondition_text_outputs=args.precondition_text_outputs,
                   anchor=args.anchor,
                   mm_encoder=None,
                )
                loss,decode_loss,logs,task,model_pred,logits,target,prompts,model_pred_audio,model_pred_audio,raw_audio_embeds,model_pred_text,raw_text_embeds = transformer(
                    kkwargs=forward_kwargs,
                    forward_function=forward_pass
                )
 
 

                accelerator.backward(loss)
                if global_step % 10 == 0:
                    total_norm = 0
                    parameters = [(k,p ) for (k,p) in transformer.named_parameters() if p.grad is not None and p.requires_grad]
                    for k,p in parameters:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                        # print(k,param_norm)
                    total_norm = total_norm ** 0.5
                    with torch.no_grad():
                        total_norm = torch.tensor(total_norm).float().to(accelerator.device)
                        total_norm = accelerator.reduce(total_norm)
                    logs.update(dict(total_norm=total_norm))
    
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    transformer.zero_grad()
                if second_accelerator is not None:
                    second_accelerator.backward(decode_loss)
                    optimizer2.step()
                    scheduler2.step()
                    optimizer2.zero_grad()
                if accelerator.sync_gradients:
                    if args.use_ema  and  accelerator.is_main_process:
                        torch.cuda.empty_cache()
                        if global_step == args.ema_start:
                            ema_transformer = EMAModel(transformer.parameters(),decay=args.ema_momentum,foreach=False)#.to(torch.bfloat16)
                            if wandb.run is not None:
                                wandb.log(dict(ema_init=1))
                        elif global_step > args.ema_start and (global_step+1)% args.ema_interval == 0:
                            ema_transformer.step(transformer.parameters())
                            with torch.no_grad():    
                                if wandb.run is not None:
                                    wandb.log(dict(ema_update=1))
                        torch.cuda.empty_cache()
                        gc.collect()
                    params_to_clip = (
                         transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                if accelerator.sync_gradients:
                    if (global_step+1)% 100 == 0:
                        torch.cuda.empty_cache()

            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            if accelerator.sync_gradients:
                
                global_step += 1

                if accelerator.is_main_process or 1:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                if accelerator.is_main_process:
                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)
                        accelerator.wait_for_everyone()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if second_accelerator is not None:
                            save_path_2 = os.path.join(args.output_dir, f"checkpoint-{global_step}/acc2")
                            second_accelerator.save_state(save_path_2)
                        logger.info(f"Saved state to {save_path}")
            


            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if accelerator.sync_gradients:
                if  wandb.run is not None and global_step % 100 == 1:
                    do_log_text=1
                    do_log_img=1
                    do_log_aud=1
                if task in ['text2img','aud2img'] and do_log_img:
                    do_log_img = 0
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        img_decode = vae.decode(torch.cat([model_pred.to(vae.device,dtype=vae.dtype)[:1],target[:1]]),return_dict=False)[0]
                        img = vae_processor.postprocess(img_decode)
                    wandb.log({"train/img":[wandb.Image(x) for x in [img[0],img[1]]]})
                elif task == 'img2text' and do_log_text and logits is not None:
                    do_log_text = 0
                    torch.cuda.empty_cache()
                    transformer.eval()
                    logits = text_vae.generate(latents=model_pred_text,max_length=256,do_sample=True)
                    logits_gt = text_vae.generate(latents=raw_text_embeds,max_length=256,do_sample=True)
                    transformer.train()
                    text = text_vae_tokenizer.batch_decode(logits)
                    text_gt = text_vae_tokenizer.batch_decode(logits_gt)
                    text_table = []
                    for text_output,text_output_gt,text_target in zip(text,text_gt,prompts):
                        text_table.append((text_output,text_output_gt,text_target))
                    _df = pd.DataFrame(text_table)
                    _df.columns = ['text','text_gt_decode','target']
                    html = wandb.Html((_df.to_html()),inject=True)
                    wandb.log({"train/text":html})
                elif task == 'text2aud' and do_log_aud:
                    do_log_aud = 0
                    with torch.no_grad():
                        img_decode = audiovae.decode(1 / audiovae.config.scaling_factor * torch.cat([model_pred_audio.to(vae.device,dtype=vae.dtype),raw_audio_embeds])).sample.detach().cpu()
                        #img = vae_processor.postprocess(img_decode)
                        wandb_log_spec([img_decode[0][0],img_decode[args.train_batch_size][0]],"train/aud")
                        del img_decode
                    # wandb.log({"train/aud":[wandb.Image(x) for x in [img_decode[0][0],img_decode[args.train_batch_size][0]]]})
                if global_step >= args.max_train_steps:
                    break
            
            if accelerator.sync_gradients and (global_step % args.val_every == 0 or global_step == 1 ): #
                #global_step == 1
                use_ema = args.ema_validation and do_ema
                do_ema = not do_ema
               
                if args.use_ema and  accelerator.is_main_process and use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_transformer.store(transformer.parameters())
                    ema_transformer.copy_to(transformer.parameters())
                transformer.eval()
                pipeline = OmniFlowPipeline(
                        vae=vae,
                        audio_vae=audiovae,
                        audio_processor=audio_processor,
                        audio_processor_clip=audio_processor_clip,
                        audio_encoder=audio_encoder,
                        text_encoder=accelerator.unwrap_model(text_encoder_one),
                        text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                        text_encoder_3=accelerator.unwrap_model(text_encoder_three),
                        transformer=accelerator.unwrap_model(transformer),
                        text_vae_tokenizer=text_vae_tokenizer,
                        crop_size=args.resolution,
                        image_processor=image_processor,
                        image_encoder=accelerator.unwrap_model(image_encoder),
                        scheduler=noise_scheduler_pipeline,
                        text_vae=text_vae,
                        text_x0 = args.precondition_text_outputs,
                        mm_encoder = None,
                        cfg_mode='new',
                        tokenizer=tokenizer_one,
                        tokenizer_2=tokenizer_two,
                        tokenizer_3=tokenizer_three,
                )
                
                pipeline_args = {"prompt": args.validation_prompt}
                print(f'Step {global_step }Do Ema? {use_ema}')
                log_validation(
                        pipeline=pipeline,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        global_step=global_step,
                        prefix='ema_' if use_ema else 'no_ema',
                        do_audio=True,
                )
                transformer.train()
                del pipeline
                if args.use_ema and  accelerator.is_main_process and use_ema:
                    # Switch back to the original UNet parameters.
                    ema_transformer.restore(transformer.parameters())
                torch.cuda.empty_cache()
                gc.collect()


    # Save the lora layers
    accelerator.wait_for_everyone()
    
    # Checks if the accelerator has performed an optimization step behind the scenes
    if 1:
        # progress_bar.update(1)
        global_step += 1

        if accelerator.is_main_process or 1:
            if 1:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        if accelerator.is_main_process:
                            for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                accelerator.wait_for_everyone()

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
