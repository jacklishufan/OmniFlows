# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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
# limitations under the License.

import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from omniflow.utils.text_encode import _encode_prompt_with_t5,cat_and_pad
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
# from diffusers.models.transformers import SD3Transformer2DModel
from omniflow.models.omni_flow import OmniFlowTransformerModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from PIL import Image
from transformers import AutoTokenizer
from omniflow.models.text_vae import LLamaForLatentConnector
from transformers import AutoConfig
from transformers import CLIPVisionModelWithProjection,CLIPImageProcessor
from torch import nn
from omniflow.models.encoders import LanguageBindAudioProcessor,LanguageBindAudio
from omniflow.utils.ema import EMAModel
from omniflow.models.audio_vae import load_audio_vae
from omniflow.utils.scheduler import OmniFlowMatchEulerDiscreteScheduler
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

from torchvision import transforms
# import imagebind.data as imagebind_data

def load_safe_tensors(fp,model):
    tensors = torch.load(fp,map_location='cpu')
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
    
    
class OmniFlowPipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`OmniFlowTransformerModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    @staticmethod
    def load_pretrained(path,device='cuda',weight_dtype = torch.bfloat16,load_ema=False):

        tokenizer_one = CLIPTokenizer.from_pretrained(
            'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
        )
        tokenizer_two = CLIPTokenizer.from_pretrained(
            path,
            subfolder="tokenizer_2",
        )
        tokenizer_three = T5TokenizerFast.from_pretrained(
                'google/flan-t5-large',
        )
        text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
            'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
            projection_dim=768
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            path,
            subfolder="text_encoder_2",
        )
        text_encoder_three = T5EncoderModel.from_pretrained('google/flan-t5-large')
        text_encoder_three.eval()
        text_encoder_two.eval()
        text_encoder_one.eval()
        vae = AutoencoderKL.from_pretrained(
            path,
            subfolder="vae",
        )
        text_vae_tokenizer=AutoTokenizer.from_pretrained(
            path,
            subfolder="vae_tokenizer",
        )
        text_vae_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        config = AutoConfig.from_pretrained(os.path.join(path,"text_vae"))
        text_vae  = LLamaForLatentConnector._from_config(
                                                        config,
                                                        torch_dtype=torch.bfloat16)
        text_vae.prepare_tokenizer(text_vae_tokenizer)
        text_vae.set_encoder(text_encoder_three)
        text_vae.prepare_tokenizer(text_vae_tokenizer)
        text_vae.set_encoder(text_encoder_three)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained('laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',projection_dim=768)
        image_processor=  CLIPImageProcessor.from_pretrained('laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        transformer = OmniFlowTransformerModel.from_config(
            # 'dev_transformer_v2'
            path,
            subfolder="transformer",
        )
        transformer.set_text_decoder(text_vae)
        audio_encoder = LanguageBindAudio.from_pretrained('LanguageBind/LanguageBind_Audio_FT')
        audio_encoder.text_model = nn.Identity()
        audio_processor_clip = LanguageBindAudioProcessor(audio_encoder.config)

        
        
        transformer.requires_grad_(False)
        text_vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        audio_encoder.requires_grad_(False)
        #text_vae.qformer.requires_grad_(True)
        text_vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        text_encoder_three.requires_grad_(False)
        # weight_dtype = torch.bfloat16

        # device = device
        fp = os.path.join(path,'transformer/diffusion_pytorch_model.bin')
        fp_ema = os.path.join(path,'transformer/ema_transformer.pt')

        load_safe_tensors(fp,transformer)
    
        
        text_encoder_one.to(device, dtype=weight_dtype)
        text_encoder_two.to(device, dtype=weight_dtype)
        text_encoder_three.to(device, dtype=weight_dtype)
        image_encoder.to(device, dtype=weight_dtype)
        transformer.to(device, dtype=weight_dtype)
        text_vae.to(device, dtype=weight_dtype)
        audio_encoder.to(device, dtype=weight_dtype)
        vae.to(device)
        audiovae,audio_processor = load_audio_vae()
        audiovae.to(device)
        audiovae.requires_grad_(False)
        noise_scheduler = OmniFlowMatchEulerDiscreteScheduler.from_pretrained(
            path, subfolder="scheduler",shift=3#,flux=True#,shift=1.0,flux=True
        )
        pipeline = OmniFlowPipeline(
        # base_model_path,
        scheduler=noise_scheduler,
        vae=vae,
        audio_processor=audio_processor,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        tokenizer_3=tokenizer_three,
        transformer=transformer,
        text_vae_tokenizer=text_vae_tokenizer,
        # torch_dtype=weight_dtype,
        image_encoder=image_encoder,
        text_vae=text_vae,
        crop_size=512,
        image_processor=image_processor,
        audio_vae=audiovae,
        text_x0=True,
        audio_encoder=audio_encoder,
        audio_processor_clip=audio_processor_clip,
        )
        # pipeline.to()
        if load_ema:
            ema_model = EMAModel(transformer.parameters())
            load_safe_tensors_ema(fp_ema,ema_model)
            ema_model.copy_to(transformer.parameters())
        return pipeline


        pass
    def enable_ema(self,path):
        device = self.transformer
        self.transformer.to('cpu')
        ema_model = EMAModel(self.transformer.parameters())
        fp_ema = os.path.join(path,'transformer/ema_transformer.pt')
        load_safe_tensors_ema(fp_ema,ema_model)
        self.transformer.to(device)
        ema_model.copy_to(self.transformer.parameters())
        
    def disable_ema(self,path):
        # device = self.transformer
        # self.transformer.to('cpu')
        # ema_model = EMAModel(self.transformer.parameters())
        fp = os.path.join(path,'transformer/diffusion_pytorch_model.bin')
        load_safe_tensors(fp,self.transformer)
        # self.transformer.to(device)
        # ema_model.copy_to(self.transformer.parameters())
        
    def __init__(
        self,
        transformer: OmniFlowTransformerModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        crop_size=512,
        text_vae_tokenizer = None,
        image_encoder = None,
        image_processor = None,
        audio_vae = None,
        audio_processor=None,
        audio_processor_clip=None,
        text_vae=None,
        text_x0=None,
        audio_encoder=None,
        mm_encoder=None,
        cfg_mode='old',
    ):
        super().__init__()
        self.text_x0 = text_x0
        self.cfg_mode = cfg_mode
        self.audio_encoder=audio_encoder
        self.mm_encoder = mm_encoder
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.text_vae_tokenizer = text_vae_tokenizer
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        
        transforms_list = []
        transforms_list.append(transforms.Resize(crop_size, interpolation=transforms.InterpolationMode.BILINEAR))
        train_crop = transforms.CenterCrop(crop_size) 
        self.img_transforms = transforms.Compose(
            [
                *transforms_list,
                train_crop,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_encoder = image_encoder
        self.encoder_image_processor = image_processor
        self.audio_vae = audio_vae
        self.audio_processor = audio_processor
        self.audio_processor_clip = audio_processor_clip
        self.text_vae = text_vae
        
    def call_mm_encoder(self,**kwargs):
        return self.mm_encoder(kwargs)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        add_token_embed: bool = False
    ):
        dtype = dtype or self.text_encoder.dtype
        device = device or self._execution_device
        batch_size = len(prompt)
        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )
        return _encode_prompt_with_t5(
            self.text_encoder_3,
            self.tokenizer_3,
            max_sequence_length,
            prompt,
            num_images_per_prompt,
            device = device,
            add_token_embed=add_token_embed  
        )


    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        add_token_embed:bool = False,
        use_t5: bool = False,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            # print(prompt_embed.shape,prompt_2_embed.shape)
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
            if use_t5:
                t5_prompt_embed = self._get_t5_prompt_embeds(
                    prompt=prompt_3,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    add_token_embed=add_token_embed,
                )

                clip_prompt_embeds = torch.nn.functional.pad(
                    clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
                )

                prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            else:
                prompt_embeds = clip_prompt_embeds
            if add_token_embed:
                prompt_embeds = (prompt_embeds - prompt_embeds.mean(-1,keepdim=True)) / (prompt_embeds.std(-1,keepdim=True)+1e-9)
                #prompt_embeds = F.normalize(prompt_embeds,dim=-1,p=2) * np.sqrt(prompt_embeds.shape[-1])
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
            # if self.mm_encoder is not None:
            #     txt_imagebind = imagebind_data.load_and_transform_text(prompt,'cuda')
            #     pooled_txt_imagebind = self.call_mm_encoder(text=txt_imagebind)['text']
            #     pooled_prompt_embeds[...,768:1792] = pooled_txt_imagebind
            #     pooled_prompt_embeds[...,1792:] = 0
                    
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)
            if use_t5:
                t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                    prompt=negative_prompt_3,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                )

                negative_clip_prompt_embeds = torch.nn.functional.pad(
                    negative_clip_prompt_embeds,
                    (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
                )
                negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            else:
                negative_prompt_embeds = negative_clip_prompt_embeds

            if add_token_embed:
                negative_prompt_embeds = (negative_prompt_embeds - negative_prompt_embeds.mean(-1,keepdim=True)) / (negative_prompt_embeds.std(-1,keepdim=True)+1e-9)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )
            # if self.mm_encoder is not None:
            #     neg_txt_imagebind = imagebind_data.load_and_transform_text(negative_prompt,'cuda')
            #     neg_pooled_txt_imagebind = self.call_mm_encoder(text=neg_txt_imagebind)['text']
            #     negative_pooled_prompt_embeds[...,768:1792] = neg_pooled_txt_imagebind
            #     negative_pooled_prompt_embeds[...,1792:] = 0

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        elif prompt_3 is not None and (not isinstance(prompt_3, str) and not isinstance(prompt_3, list)):
            raise ValueError(f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        add_token_embed: bool = False,
        task: str = 't2i',
        input_img = None,
        v_pred=True,
        split_cond=False,
        overwrite_audio=None,
        overwrite_audio_t=None,
        input_aud=None,
        return_embed=False,
        drop_text=False,
        drop_image=False,
        drop_audio=False,
        use_text_output=True,
        use_t5=False,
        drop_pool=False,
        mm_cfgs=[],
        bypass=False,
        no_clip=False,
        cfg_mode=None
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        if cfg_mode is not None:
            self.cfg_mode = cfg_mode
        if bypass:
            if task =='a2i':
                imgs =  self("",input_aud=input_aud,height=512,width=512,add_token_embed=1,task='a2t',return_embed=False,guidance_scale=4,drop_pool=drop_pool)
                task = 't2i'
                
                input_aud = None
                prompt = imgs[0][0].replace('<s>','').replace('</s>','')
            if task =='i2a':
                imgs =  self("",input_img=input_img,height=512,width=512,add_token_embed=1,task='i2t',return_embed=False,guidance_scale=2,drop_pool=drop_pool)
                task = 't2a'
                
                input_img = None
                prompt = imgs[0][0].replace('<s>','').replace('</s>','')
                #print(prompt)
                # prompt = 'a dog is barking'
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        text_vae_tokenizer = self.text_vae_tokenizer
        # 1. Check inputs. Raise error if not correct
        if task in ['t2i','t2a']:    
            self.check_inputs(
                prompt,
                prompt_2,
                prompt_3,
                height,
                width,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        self._guidance_scale = guidance_scale 
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            add_token_embed=add_token_embed,
            use_t5=use_t5,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        # breakpoint()
        if self.text_vae is not None:
            prompt_embeds_vae = self.text_vae.encode(prompt,input_ids=None,tokenizer=self.tokenizer_3)
            negative_prompt_embeds_vae = self.text_vae.encode(negative_prompt or '',input_ids=None,tokenizer=self.tokenizer_3)
            l_vae = prompt_embeds_vae.shape[1]
        # prepare latents
        latents = None
        latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
        )
        if self.transformer.use_audio_mae:
            prompt_embeds_audio = torch.randn(1,8,768).to(prompt_embeds)
        else:
            prompt_embeds_audio = torch.randn(1,8,256,16).to(prompt_embeds)
        
        
        assert task in ['t2i','a2i','i2t','a2t','t2a','i2a']
        if self.do_classifier_free_guidance:
            if task in ['a2i','a2t','t2i','i2t']:
                 prompt_embeds_audio = prompt_embeds_audio.repeat(2,*([1]*len(prompt_embeds_audio.shape[1:])
                                                                    ))
                 
            elif task in ['ai2t','at2i']:
                prompt_embeds_audio = prompt_embeds_audio.repeat(4,*([1]*len(prompt_embeds_audio.shape[1:])
                                                                    ))
                
            elif task in ['ait']:
                prompt_embeds_audio = prompt_embeds_audio.repeat(4,*([1]*len(prompt_embeds_audio.shape[1:])
                                                                    ))
            if task in ['i2a','i2t','t2a','a2t']: #  not in ['t2i','a2i']
                latents = latents.repeat(2,1,1,1)
            elif task in ['it2a','ai2t']:
                latents = latents.repeat(4,1,1,1)
            elif task in ['ait']:
                latents = latents.repeat(4,1,1,1)
                
        # prepare prompt_embeds
        if task in ['t2i','t2a','it2a','at2i']:
            # hack
            if no_clip==True:
                if self.do_classifier_free_guidance:
                    prompt_embeds_vae_to_append =  torch.cat([negative_prompt_embeds_vae, prompt_embeds_vae], dim=0)
                    prompt_embeds = cat_and_pad([prompt_embeds_vae_to_append],max_dim=4096)
                    # negative_prompt_embeds = cat_and_pad([negative_prompt_embeds,negative_prompt_embeds_vae],max_dim=4096)
                else:
                    prompt_embeds = cat_and_pad([prompt_embeds_vae],max_dim=4096)
            elif self.text_vae is not None:
                if self.do_classifier_free_guidance:
                    prompt_embeds_vae_to_append =  torch.cat([negative_prompt_embeds_vae, prompt_embeds_vae], dim=0)
                    prompt_embeds = cat_and_pad([prompt_embeds,prompt_embeds_vae_to_append],max_dim=4096)
                    # negative_prompt_embeds = cat_and_pad([negative_prompt_embeds,negative_prompt_embeds_vae],max_dim=4096)
                else:
                    prompt_embeds = cat_and_pad([prompt_embeds,prompt_embeds_vae],max_dim=4096)
            else:
                 prompt_embeds = cat_and_pad([prompt_embeds],max_dim=4096)
        elif task in ['i2t','a2t','ai2t']:
            prompt_embeds = randn_tensor((1,*prompt_embeds_vae.shape[1:]),device=self.transformer.device,dtype=self.transformer.dtype)
            prompt_embeds = cat_and_pad([prompt_embeds],4096)
        else:
            assert prompt_embeds.shape[0] == 2
            prompt_embeds = randn_tensor((1,*prompt_embeds_vae.shape[1:]),device=self.transformer.device,dtype=self.transformer.dtype)
            prompt_embeds =  cat_and_pad([prompt_embeds_vae],max_dim=4096)
            if self.do_classifier_free_guidance:
                prompt_embeds = prompt_embeds.repeat(2,1,1)
        
        if task in ['i2t','i2a','ai2t']:
            # update latents,pooled_prompt_embeds
            pixel_values = self.img_transforms(input_img)[None]
            latents = self.vae.encode(pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.sample()
            if self.do_classifier_free_guidance:
                latents_null = self.vae.encode(0*pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.mean
                if task == 'ai2t':
                    latents = torch.cat([latents_null,latents,latents])
                else:
                    latents = torch.cat([latents_null,latents])
            
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(device)
            
            image_clip = self.encoder_image_processor(input_img,return_tensors="pt").pixel_values
            image_clip = image_clip.to(self.image_encoder.device).to(self.image_encoder.dtype)
            with torch.no_grad():
                image_embeds = self.image_encoder(pixel_values=image_clip).image_embeds
                # if self.mm_encoder is not None:
                #     image_embeds_2 = self.call_mm_encoder(vision=image_clip)['vision']
                #     image_embeds = torch.cat([image_embeds,image_embeds_2],dim=-1)
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            # print(pooled_prompt_embeds.shape)
            pooled_prompt_embeds[...,:image_embeds.shape[-1]] = image_embeds
            

            if self.do_classifier_free_guidance:
                with torch.no_grad():
                    image_embeds_null = self.image_encoder(pixel_values=image_clip * 0).image_embeds
                    # if self.mm_encoder is not None:
                    #     image_embeds_null_2 = self.call_mm_encoder(vision=image_clip * 0)['vision']
                    #     image_embeds_null = torch.cat([image_embeds_null,image_embeds_null_2],dim=-1)
                assert pooled_prompt_embeds.shape[0] == 2
                pooled_prompt_embeds[0][...,:image_embeds.shape[-1]] = image_embeds_null
                pooled_prompt_embeds[0] *= 0
            # print(prompt_embeds.shape)
        elif task in ['a2i','a2t']:
            # update prompt_embeds_audio,pooled_prompt_embeds
            pixel_values = self.audio_processor.feature_extraction_vae(input_aud)['fbank'].unsqueeze(0)
            prompt_embeds_audio = self.audio_vae.encode(pixel_values.to(device=self.audio_vae.device,dtype=self.audio_vae.dtype)).latent_dist.sample()
            if self.do_classifier_free_guidance:
                prompt_embeds_audio_null = self.audio_vae.encode(0*pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.mean
                if task == 'ai2t':
                    prompt_embeds_audio = torch.cat([prompt_embeds_audio,prompt_embeds_audio_null,prompt_embeds_audio])
                else:
                    prompt_embeds_audio = torch.cat([prompt_embeds_audio_null,prompt_embeds_audio])
            
            prompt_embeds_audio = prompt_embeds_audio * self.audio_vae.config.scaling_factor
            prompt_embeds_audio = prompt_embeds_audio.to(device).to(prompt_embeds.dtype)
            
            audio_clip = self.audio_processor_clip(input_aud)['pixel_values']
            with torch.no_grad():
                audio_embeds = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype))
                # if self.mm_encoder is not None:
                #     audio_imb = imagebind_data.load_and_transform_audio_data([input_aud],'cpu')
                #     audio_imb = audio_imb.to(device).to(prompt_embeds.dtype)
                #     audio_embeds_2 = self.call_mm_encoder(audio=audio_imb)['audio']
                #     audio_embeds = torch.cat([audio_embeds,audio_embeds_2],dim=-1)
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            pooled_prompt_embeds[...,:audio_embeds.shape[-1]] = audio_embeds
            if self.do_classifier_free_guidance:
                assert pooled_prompt_embeds.shape[0] == 2
                with torch.no_grad():
                    audio_embeds_null = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype) * 0)
                    # if self.mm_encoder is not None:
                    #     audio_embeds_null_2 = self.call_mm_encoder(audio=audio_imb * 0)['audio']
                    #     audio_embeds_null = torch.cat([audio_embeds_null,audio_embeds_null_2],dim=-1)
                assert pooled_prompt_embeds.shape[0] == 2
                pooled_prompt_embeds[0][...,:audio_embeds.shape[-1]] = audio_embeds_null
                # pooled_prompt_embeds[0] *= 0
                #pooled_prompt_embeds 
        if task == 'ai2t':
            pixel_values = self.img_transforms(input_img)[None]
            latents = self.vae.encode(pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.sample()
            if self.do_classifier_free_guidance:
                latents_null = self.vae.encode(0*pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.mean
                latents = torch.cat([torch.rand_like(latents_null),latents,latents])
            
            latents = latents * self.vae.config.scaling_factor
            latents = latents.to(device)
            
            image_clip = self.encoder_image_processor(input_img,return_tensors="pt").pixel_values
            image_clip = image_clip.to(self.image_encoder.device).to(self.image_encoder.dtype)
            with torch.no_grad():
                image_embeds = self.image_encoder(pixel_values=image_clip).image_embeds
                # if self.mm_encoder is not None:
                #     image_embeds_2 = self.call_mm_encoder(vision=image_clip)['vision']
                #     image_embeds = torch.cat([image_embeds,image_embeds_2],dim=-1)
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)[:1].repeat(3,1)
            #pooled_prompt_embeds[...,:image_embeds.shape[-1]] = image_embeds
            if self.do_classifier_free_guidance:
                pooled_prompt_embeds[1,:image_embeds.shape[-1]] = image_embeds
                # with torch.no_grad():
                #     image_embeds_null = self.image_encoder(pixel_values=image_clip * 0).image_embeds
                #     if self.mm_encoder is not None:
                #         image_embeds_null_2 = self.call_mm_encoder(vision=image_clip * 0)['vision']
                #         image_embeds_null = torch.cat([image_embeds_null,image_embeds_null_2],dim=-1)
                # assert pooled_prompt_embeds.shape[0] == 2
                # pooled_prompt_embeds[0][...,:image_embeds.shape[-1]] = image_embeds_null
                # pooled_prompt_embeds[0] *= 0
            
            pixel_values = self.audio_processor.feature_extraction_vae(input_aud)['fbank'].unsqueeze(0)
            prompt_embeds_audio = self.audio_vae.encode(pixel_values.to(device=self.audio_vae.device,dtype=self.audio_vae.dtype)).latent_dist.sample()
            if self.do_classifier_free_guidance:
                prompt_embeds_audio_null = self.audio_vae.encode(0*pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.mean
                prompt_embeds_audio = torch.cat([prompt_embeds_audio,torch.rand_like(prompt_embeds_audio_null),prompt_embeds_audio])
            
            prompt_embeds_audio[[0,2]] = prompt_embeds_audio[[0,2]] * self.audio_vae.config.scaling_factor
            prompt_embeds_audio = prompt_embeds_audio.to(device).to(prompt_embeds.dtype)
            
            audio_clip = self.audio_processor_clip(input_aud)['pixel_values']
            with torch.no_grad():
                audio_embeds = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype))
                # if self.mm_encoder is not None:
                #     audio_imb = imagebind_data.load_and_transform_audio_data([input_aud],'cpu')
                #     audio_imb = audio_imb.to(device).to(prompt_embeds.dtype)
                #     audio_embeds_2 = self.call_mm_encoder(audio=audio_imb)['audio']
                #     audio_embeds = torch.cat([audio_embeds,audio_embeds_2],dim=-1)

            #pooled_prompt_embeds[-1,:audio_embeds.shape[-1]] = audio_embeds
            if self.do_classifier_free_guidance:
                pooled_prompt_embeds[0,:image_embeds.shape[-1]] = audio_embeds

        if task == 'at2i':
            print(pooled_prompt_embeds.shape,prompt_embeds.shape)
            if self.do_classifier_free_guidance:
                pooled_prompt_embeds_null,pooled_prompt_embeds_text = pooled_prompt_embeds.chunk(2)
                prompt_embeds_null,prompt_embeds_text = prompt_embeds.chunk(2)
                pooled_prompt_embeds = torch.cat(
                    [pooled_prompt_embeds_text,pooled_prompt_embeds_text,pooled_prompt_embeds_null,pooled_prompt_embeds_null]
                )
                prompt_embeds = torch.cat(
                    [prompt_embeds_text,prompt_embeds_text,torch.randn_like(prompt_embeds_null),torch.randn_like(prompt_embeds_null)]
                )
                
                pixel_values = self.audio_processor.feature_extraction_vae(input_aud)['fbank'].unsqueeze(0)
                prompt_embeds_audio = self.audio_vae.encode(pixel_values.to(device=self.audio_vae.device,dtype=self.audio_vae.dtype)).latent_dist.sample()
                prompt_embeds_audio = prompt_embeds_audio * self.audio_vae.config.scaling_factor
                if self.do_classifier_free_guidance:
                    prompt_embeds_audio_null = self.audio_vae.encode(0*pixel_values.to(device=self.vae.device,dtype=self.vae.dtype)).latent_dist.mean
                    prompt_embeds_audio_null = prompt_embeds_audio * self.audio_vae.config.scaling_factor
                    null_audio = torch.rand_like(prompt_embeds_audio_null)
                prompt_embeds_audio = torch.cat([prompt_embeds_audio,null_audio,prompt_embeds_audio,null_audio])
                
            
            prompt_embeds_audio = prompt_embeds_audio.to(device).to(prompt_embeds.dtype)
            
            audio_clip = self.audio_processor_clip(input_aud)['pixel_values']
            with torch.no_grad():
                audio_embeds = self.audio_encoder.get_image_features(pixel_values=audio_clip.to(self.audio_encoder.device).to(self.audio_encoder.dtype))
                # if self.mm_encoder is not None:
                #     audio_imb = imagebind_data.load_and_transform_audio_data([input_aud],'cpu')
                #     audio_imb = audio_imb.to(device).to(prompt_embeds.dtype)
                #     audio_embeds_2 = self.call_mm_encoder(audio=audio_imb)['audio']
                #     audio_embeds = torch.cat([audio_embeds,audio_embeds_2],dim=-1)

            #pooled_prompt_embeds[-1,:audio_embeds.shape[-1]] = audio_embeds
            if self.do_classifier_free_guidance:
                pooled_prompt_embeds[2,:audio_embeds.shape[-1]] = audio_embeds
                # prompt_embeds = torch.cat([prompt_embeds_text,prompt_embeds,prompt_embeds_null,prompt_embeds_null])
                # pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_text,pooled_prompt_embeds_null,pooled_prompt_embeds_text,pooled_prompt_embeds_null])
            # pooled_prompt_embeds *= 0
            pooled_prompt_embeds[-1] *= 0
            
                
        if task in ['t2i','t2a',]:
            timesteps_text = [0] * batch_size
            timesteps_text = torch.tensor(timesteps_text).to(self.device)
            if self.do_classifier_free_guidance:
                timesteps_text = timesteps_text.repeat(2)
                if self.cfg_mode == 'new':
                    timesteps_text[0] = 1000
                    prompt_embeds[0] = torch.randn_like(prompt_embeds[0] )
                    pooled_prompt_embeds[0] *= 0
        if task in ['i2a','a2i']:
            timesteps_text = [0] * batch_size 
            timesteps_text = torch.tensor(timesteps_text).to(self.device)+ 1000

                    
        if task in ['i2t','i2a',]:
            timesteps_img = [0] * batch_size
            timesteps_img = torch.tensor(timesteps_img).to(self.device)
            if self.do_classifier_free_guidance:
                timesteps_img = timesteps_img.repeat(2)
                if self.cfg_mode == 'new':
                    timesteps_img[0] = 1000
                    latents[0] = torch.randn_like(latents[0] )
                    pooled_prompt_embeds[0] *= 0
            
        if task in ['t2a','a2t']:
            timesteps_img = [0] * batch_size  
            timesteps_img = torch.tensor(timesteps_img).to(self.device) + 1000
            
        if task in ['a2t','a2i']:
            timesteps_aud = [0] * batch_size
            timesteps_aud = torch.tensor(timesteps_aud).to(self.device) 
            if self.do_classifier_free_guidance:
                timesteps_aud = timesteps_aud.repeat(2)
                if self.cfg_mode == 'new':
                    timesteps_aud[0] = 1000
                    prompt_embeds_audio[0] = torch.randn_like(prompt_embeds_audio[0])
                    pooled_prompt_embeds[0] *= 0
    
        if task in ['t2i','i2t']:
            timesteps_aud = [0] * batch_size 
            timesteps_aud = torch.tensor(timesteps_aud).to(self.device) + 1000
            

        


        x0 = None
        prompt_embeds[:,-l_vae:,prompt_embeds_vae.shape[-1]:] = 0
        if drop_pool:
            pooled_prompt_embeds = pooled_prompt_embeds * 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latents = latents.to(device=self.transformer.device,dtype=self.transformer.dtype)
                if task == 'ai2t':
                    prompt_embed_input = torch.cat([prompt_embeds] * 3) if self.do_classifier_free_guidance else prompt_embeds
                    timestep = t.expand(prompt_embed_input.shape[0])
                    #print(prompt_embed_input.shape,timestep.shape,timesteps_aud.shape,pooled_prompt_embeds.shape)
                    # breakpoint()
                    # print(timestep.shape,latent_model_input.shape)
                    _y = self.transformer(
                        hidden_states=latents,
                        timestep=timesteps_img,
                        timestep_text=timestep,
                        timestep_audio=timesteps_aud,
                        encoder_hidden_states=prompt_embed_input,
                        audio_hidden_states=prompt_embeds_audio,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        use_text_output=True,
                        decode_text=True,
                        split_cond=split_cond,
                        drop_text=drop_text,
                        drop_audio=drop_audio,
                        drop_image=drop_image
                    )
                    if v_pred and not self.text_x0:
                        noise_pred = _y['model_pred_text']
                    else:
                        x0 = _y['model_pred_text']
                        curr_latent_text = prompt_embed_input[...,:x0.shape[-1]]
                        noise_pred = self.scheduler.get_eps(t,x0,curr_latent_text) 
                        
                elif task in ['t2i','a2i','at2i']:

                    if task == 'at2i':
                        latent_model_input = torch.cat([latents] * 4) if self.do_classifier_free_guidance else latents
                        timestep = t.expand(latent_model_input.shape[0])
                    else:
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        timestep = t.expand(latent_model_input.shape[0])
                    # breakpoint()

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        timestep_text=timesteps_text,
                        timestep_audio=timesteps_aud,
                        audio_hidden_states=prompt_embeds_audio,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        use_text_output=use_text_output,
                        decode_text=True,
                        split_cond=split_cond,
                        drop_text=drop_text,
                        drop_audio=drop_audio,
                        drop_image=drop_image
                    )['output']
                elif task in ['t2a','i2a']:
                    if overwrite_audio is not None:
                        assert not self.do_classifier_free_guidance 
                        ## DEBUG
                        prompt_embeds_audio = overwrite_audio.to(prompt_embeds_audio)
                        noise_audio = torch.randn_like(prompt_embeds_audio)
                        timestep = torch.tensor([overwrite_audio_t]).to(noise_audio.device)
                        sigmas_audio = self.scheduler.sigmas[num_inference_steps-overwrite_audio_t]
                        sigmas_audio = sigmas_audio.view(-1,1,1,1)
                        prompt_embeds_audio_input = sigmas_audio * noise_audio + (1.0 -sigmas_audio ) * prompt_embeds_audio
                        prompt_embeds_audio_input = prompt_embeds_audio_input.to(self.transformer.dtype)
                    else: 
                        prompt_embeds_audio_input = torch.cat([prompt_embeds_audio] * 2) if self.do_classifier_free_guidance else prompt_embeds_audio
                        timestep = t.expand(prompt_embeds_audio_input.shape[0])
                    # print(prompt_embeds_audio_input.shape)
                    #print(pooled_prompt_embeds.mean())
                    
                    _y = self.transformer(
                        hidden_states=latents,
                        timestep=timesteps_img,
                        timestep_text=timesteps_text,
                        timestep_audio=timestep,
                        audio_hidden_states=prompt_embeds_audio_input,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        use_text_output=True,
                        decode_text=True,
                        split_cond=split_cond,
                        drop_text=drop_text,
                        drop_audio=drop_audio,
                        drop_image=drop_image
                    )
                    if v_pred:
                        noise_pred = _y['audio_hidden_states']
                    else:
                        x0 = _y['audio_hidden_states']
                        noise_pred = self.scheduler.get_eps(t,x0,prompt_embeds_audio) 
                        noise_pred = noise_pred.to(x0)      
                    if overwrite_audio is not None:
                        x0 =   noise_pred * (-sigmas_audio) + prompt_embeds_audio_input  
                        x0 = 1 / self.audio_vae.config.scaling_factor * x0
                        spec = self.audio_vae.decode( x0.float())  
                        return spec.sample.float().cpu().numpy()           
                elif task in ['i2t','a2t']:
                    prompt_embed_input = torch.cat([prompt_embeds] * 2) if self.do_classifier_free_guidance else prompt_embeds
                    timestep = t.expand(prompt_embed_input.shape[0])
                    _y = self.transformer(
                        hidden_states=latents,
                        timestep=timesteps_img,
                        timestep_text=timestep,
                        timestep_audio=timesteps_aud,
                        encoder_hidden_states=prompt_embed_input,
                        audio_hidden_states=prompt_embeds_audio,
                        pooled_projections=pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        use_text_output=True,
                        decode_text=True,
                        split_cond=split_cond,
                        drop_text=drop_text,
                        drop_audio=drop_audio,
                        drop_image=drop_image
                    )
                    if v_pred and not self.text_x0:
                        noise_pred = _y['model_pred_text']
                    else:
                        x0 = _y['model_pred_text']
                        curr_latent_text = prompt_embed_input[...,:x0.shape[-1]]
                        noise_pred = self.scheduler.get_eps(t,x0,curr_latent_text) 
                    # vx = self.scheduler.step(None, t, curr_latent_text, return_dict=False,x0=x0)[0]
                    # self.scheduler.  
                    # next  = 
                    # breakpoint()                
                else:
                    raise NotImplemented
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                if task in ['t2i','a2i','at2i']:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                elif task in ['i2t','a2t','ai2t']:
                    prompt_embeds = self.scheduler.step(noise_pred, t, prompt_embeds[...,:noise_pred.shape[-1]], return_dict=False)[0]
                    prompt_embeds = cat_and_pad([prompt_embeds],4096).to(latents_dtype)
                elif task in ['i2a','t2a']:
                    prompt_embeds_audio = self.scheduler.step(noise_pred, t, prompt_embeds_audio, return_dict=False)[0]
                else:
                    raise NotImplemented
                    
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()
        if task in ['i2t','a2t','ai2t']:
            prompt_embeds = prompt_embeds[...,:prompt_embeds_vae.shape[-1]]
            # if self.do_classifier_free_guidance:
            #     _,extra_cond = _y['extra_cond'].chunk(2)
            # else:
            #     extra_cond = _y['extra_cond']
            # prompt_embeds = torch.cat([extra_cond,prompt_embeds],dim=1)
            tokens1 = self.text_vae.generate(latents=prompt_embeds,max_length=256,do_sample=False)
            z =  self.text_vae.encode(prompt,input_ids=None,tokenizer=self.tokenizer_3,drop=True)
            tokens2 =  self.text_vae.generate(latents=z,max_length=256,do_sample=False)
            # logits = self.transformer.text_decoder(inputs_embeds=prompt_embeds)[:,77:]
            if self.text_vae_tokenizer is not None and type(tokens1[0]) is not str:
                text = self.text_vae_tokenizer.batch_decode( tokens1)
                text2 = self.text_vae_tokenizer.batch_decode( tokens2)
            else:
                text = tokens1
                text2 = tokens2
            if return_embed:
                return text,text2,prompt_embeds #,prompt_embeds
            else:
                return text,text2
        elif task in ['t2a','i2a']:
            prompt_embeds_audio = 1 / self.audio_vae.config.scaling_factor * prompt_embeds_audio
            spec = self.audio_vae.decode( prompt_embeds_audio.float())
            if hasattr(spec,'sample'):
                spec = spec.sample 
            return spec.float().cpu().numpy(),x0
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
