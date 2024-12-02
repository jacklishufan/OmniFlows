from diffusers import  ModelMixin,ConfigMixin#,PeftAdapterMixin,FromOriginalModelMixin
from diffusers.configuration_utils  import ConfigMixin, register_to_config
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from omniflow.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Tuple
import inspect
from einops import rearrange
from functools import partial
from typing import Any, Dict, List, Optional, Union
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.llama.modeling_llama import LlamaConfig,LlamaModel
import torch
import torch.nn as nn
import deepspeed

from transformers.activations import ACT2FN    
class NNMLP(nn.Module):
    def __init__(self, input_size,hidden_size,activation='gelu'):
        super().__init__()

        self.linear_1 = nn.Linear(input_size, hidden_size, bias=True)
        self.act = ACT2FN[activation]
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    
class OmniFlowTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        audio_input_dim: int = 8,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0
        decoder_config:str = '',
        add_audio=True,
        add_clip=False,
        use_audio_mae=False,
        drop_text=False,
        drop_image=False,
        drop_audio=False,
        qk_norm: Optional[str] = 'layer_norm',
    ):
        super().__init__()
        default_out_channels = in_channels
        self.add_clip = add_clip
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        
        if add_audio:
            # 1 8 256 16

            
            self.time_image_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
            self.audio_input_dim = audio_input_dim
            self.use_audio_mae = use_audio_mae
            self.audio_patch_size = 2
            if use_audio_mae:
                self.audio_embedder = nn.Linear(audio_input_dim, self.config.caption_projection_dim)
            else:
                self.audio_embedder = PatchEmbed(
                    height=256,
                    width=16,
                    patch_size=self.audio_patch_size ,
                    in_channels=self.audio_input_dim,
                    embed_dim=self.config.caption_projection_dim,
                    pos_embed_max_size=192 # hardcode
                    #pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
                )
            
            self.time_aud_embed = CombinedTimestepTextProjEmbeddings(
                embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
            )
            #
         
            self.norm_out_aud = AdaLayerNormContinuous(self.config.caption_projection_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
            #self.proj_out_aud = nn.Linear(self.config.caption_projection_dim, self.config.audio_input_dim)
            if use_audio_mae:
                self.proj_out_aud = nn.Linear(self.config.caption_projection_dim, self.config.audio_input_dim)
            else: 
                self.proj_out_aud = nn.Linear(self.inner_dim, self.audio_patch_size * self.audio_patch_size * self.audio_input_dim, bias=True)
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)
        bert_config = LlamaConfig(1,hidden_size=self.config.joint_attention_dim,num_attention_heads=32,num_hidden_layers=2)
        if self.add_audio:
            self.context_decoder = nn.ModuleDict(dict(
                #transformer=LlamaModel(bert_config),
                projection=nn.Linear(self.config.caption_projection_dim,self.config.joint_attention_dim)
            ))
        self.text_out_dim = 1536 # 1536
        self.text_output = nn.Linear(self.config.joint_attention_dim,self.text_out_dim)
        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only= i == num_layers - 1,
                    context_output=i <num_layers or self.add_audio,
                    audio_output=add_audio,
                    delete_img=drop_image,
                    delete_aud=drop_audio,
                    delete_text=drop_text,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.add_audio = add_audio
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.norm_out_text = AdaLayerNormContinuous(self.joint_attention_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        if self.add_clip:
            self.n_cond_tokens = 8
            self.clip_proj = nn.Sequential(
                NNMLP(self.config.pooled_projection_dim, self.config.caption_projection_dim),
                nn.Linear(self.config.caption_projection_dim,self.config.caption_projection_dim*self.n_cond_tokens)
            )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        if decoder_config:
            self.text_decoder = build_from_config(decoder_config)
        else:
            self.text_decoder = None
    #     self.apply(self._init_weights)
            
    # def _init_weights(self, module):
    #     std = 0.02
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()

    def set_text_decoder(self,model):
        self.text_decoder = model
        self.text_out_dim = model.vae_dim # 1536
        self.text_output = nn.Linear(self.config.joint_attention_dim,self.text_out_dim)
        
    def set_audio_pooler(self,model):
        self.audio_pooler = model
        
    def get_decoder(self):
        return self.text_decoder
    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `fuse_lora()`.")

        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        from peft.tuners.tuners_utils import BaseTunerLayer

        merge_kwargs = {"safe_merge": self._safe_fusing}

        if isinstance(module, BaseTunerLayer):
            if self.lora_scale != 1.0:
                module.scale_layer(self.lora_scale)

            # For BC with prevous PEFT versions, we need to check the signature
            # of the `merge` method to see if it supports the `adapter_names` argument.
            supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
            if "adapter_names" in supported_merge_kwargs:
                merge_kwargs["adapter_names"] = adapter_names
            elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                    " to the latest version of PEFT. `pip install -U peft`"
                )

            module.merge(**merge_kwargs)

    def unfuse_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `unfuse_lora()`.")
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(module, BaseTunerLayer):
            module.unmerge()

    def forward(
        self,
        hidden_states: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        timestep_text: torch.LongTensor = None,
        timestep_audio: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        use_text_output: bool = False,
        target_prompt_embeds=None, # target 
        decode_text=False,
        sigma_text=None,
        detach_logits=False,
        prompt_embeds_uncond=None,
        targets=None,
        audio_hidden_states = None,
        split_cond = False,
        text_vae=None,
        text_x0=True,
        drop_text=False,
        drop_image=False,
        drop_audio=False,
        kkwargs = None,
        forward_function = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if kkwargs is not None:
            assert forward_function is not None
            return forward_function(transformer=self,**kkwargs)
        
        encoder_hidden_states_base = encoder_hidden_states.clone()
        hidden_states_base = hidden_states
        # set useless branch to null
        do_image = not drop_image
        do_audio = (not drop_audio ) and ( self.add_audio)
        do_text = (not drop_text) #and use_text_output
        #assert do_text
        if do_image:
            height, width = hidden_states.shape[-2:]
            hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
            # print(pooled_projections.shape)
            temb = self.time_text_embed(timestep, pooled_projections) # image
        else:
            hidden_states = None
            temb = 0
           
        
        if do_audio:
            if audio_hidden_states is None:
                if self.use_audio_mae:
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0],8,self.audio_input_dim).to(encoder_hidden_states)
                else:
                    audio_hidden_states = torch.zeros(encoder_hidden_states.shape[0],8,256,16).to(encoder_hidden_states)
                # print(audio_hidden_states.shape)
                timestep_audio = timestep_text * 0
            
            temb_audio = self.time_aud_embed(timestep_audio,pooled_projections)
            audio_hidden_states = self.audio_embedder(audio_hidden_states)
            if not split_cond:
                temb = temb + temb_audio
                temb_audio = None
        else:
            audio_hidden_states = None
            temb_audio = None
            

            
        if do_text:
            if use_text_output:
                temb_text = self.time_image_embed(timestep_text, pooled_projections)
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)
            if use_text_output:
                if not split_cond:
                    temb = temb + temb_text
                    temb_text = None
            else:
                temb_text = None
        else:
            encoder_hidden_states = None
            temb_text = None
    
        assert not self.add_clip

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                #ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                ckpt_kwargs = dict()
                if self.add_audio:
                    encoder_hidden_states, hidden_states,audio_hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        audio_hidden_states,
                        temb_text,
                        temb_audio,
                        **ckpt_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = deepspeed.checkpointing.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        temb_text,
                        **ckpt_kwargs,
                    )

            else:
                if self.add_audio:
                    encoder_hidden_states, hidden_states,audio_hidden_states = block(
                        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,audio_hidden_states=audio_hidden_states, temb=temb,
                        temb_text=temb_text,temb_audio=temb_audio,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                        temb_text=temb_text
                    )

            # controlnet residual
            assert block_controlnet_hidden_states is None
            # if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            #     interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
            #     hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        if do_image:
            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size
            height = height // patch_size
            width = width // patch_size

            hidden_states = hidden_states.reshape(
                shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
            )
        else:
            output = None


        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)
        assert not return_dict
        logits = None
        logits_labels = None
        if do_text and use_text_output:
            encoder_hidden_states = self.context_decoder['projection'](encoder_hidden_states)
            encoder_hidden_states = self.norm_out_text(encoder_hidden_states,temb_text if temb_text is not None else temb)
            encoder_hidden_states = self.text_output(encoder_hidden_states)
            model_pred_text = encoder_hidden_states #x0 prediction
            if decode_text and targets is not None:
                logits = None
                logits_labels = None
                if self.text_decoder is not None:
                    if detach_logits:
                        with torch.no_grad():
                            if prompt_embeds_uncond is not None:
                                raw_text_embeds_input =  prompt_embeds_uncond[...,:self.text_out_dim]#.detach()
                            else:
                                raw_text_embeds_input = target_prompt_embeds[...,:self.text_out_dim]#.detach()
                            if text_x0:
                                model_pred_text_clean = model_pred_text
                            else:
                                noisy_prompt_embeds = encoder_hidden_states_base[...,:model_pred_text.shape[-1]]
                                model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[...,:model_pred_text.shape[-1]]
                            latents_decode = torch.cat([model_pred_text_clean,raw_text_embeds_input],dim=0).detach()
                    else:
                        if prompt_embeds_uncond is not None:
                            raw_text_embeds_input =  prompt_embeds_uncond[...,:self.text_out_dim]#.detach()
                        else:
                            raw_text_embeds_input = target_prompt_embeds[...,:self.text_out_dim]#.detach()
                        if text_x0:
                            model_pred_text_clean = model_pred_text
                        else:
                            noisy_prompt_embeds = encoder_hidden_states_base[...,:model_pred_text.shape[-1]]
                            model_pred_text_clean = model_pred_text * (-sigma_text) + noisy_prompt_embeds[...,:model_pred_text.shape[-1]]
                        latents_decode = torch.cat([model_pred_text_clean,raw_text_embeds_input],dim=0)
                    # print(latents_decode.shape,targets['input_ids'].shape)
                    logits_all = self.text_decoder(latents=latents_decode,
                                        input_ids=targets['input_ids'].repeat(2,1),
                                        attention_mask=None, # attention_mask
                                        labels=None,
                                        return_dict=False
                                    )[0]
                    logits,logits_labels = logits_all.chunk(2)
        else:
            model_pred_text = None

    
        if do_audio:
                audio_hidden_states = self.norm_out_aud(audio_hidden_states,temb_audio if temb_audio is not None else temb)
                audio_hidden_states = self.proj_out_aud(audio_hidden_states)
                if not self.use_audio_mae:
                    patch_size_audio = self.audio_patch_size 
                    height_audio = 256  // patch_size_audio
                    width_audio = 16 // patch_size_audio
                    # N X [(256/16) X (16 / 16)] X [ (16 16 8)]
                    # breakpoint()
                    audio_hidden_states = rearrange(
                        audio_hidden_states,
                        'n (h w) (hp wp c) -> n c (h hp) (w wp)',
                        h=height_audio,
                        w=width_audio,
                        hp=patch_size_audio,
                        wp=patch_size_audio,
                        c=self.audio_input_dim
                    )
                    # audio_hidden_states = audio_hidden_states.reshape(
                    #     shape=(audio_hidden_states.shape[0], height_audio, width_audio, patch_size_audio, patch_size_audio, self.audio_input_dim)
                    # )
                    # audio_hidden_states = torch.einsum("nhwpqc->nchpwq", audio_hidden_states)
                    # audio_hidden_states = audio_hidden_states.reshape(
                    #     shape=(audio_hidden_states.shape[0], self.audio_input_dim, height_audio * patch_size_audio, width_audio * patch_size_audio)
                    # )
        else:
                audio_hidden_states = None
        return dict(output=output,
                    model_pred_text=model_pred_text,
                    encoder_hidden_states=encoder_hidden_states,
                    logits=logits,
                    extra_cond=None,
                    logits_labels=logits_labels,
                    audio_hidden_states=audio_hidden_states,
                    )

        #return Transformer2DModelOutput(sample=output)

