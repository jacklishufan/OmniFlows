from diffusers.models.attention import *
from diffusers.models.attention_processor import *


@maybe_allow_in_graph
class AttentionAudioVideo(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        aud_kv_proj_dim=None,
        delete_img=False,
        delete_aud=False,
        delete_text=False
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.aud_kv_proj_dim = aud_kv_proj_dim

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applys qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            from diffusers.models.normalization import RMSNorm
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
            self.norm_added_q = RMSNorm(dim_head, eps=eps)
            self.norm_added_k = RMSNorm(dim_head, eps=eps)
                        # self.norm_added_q = RMSNorm(dim_head, eps=eps)
            # self.norm_added_k = RMSNorm(dim_head, eps=eps)
            self.norm_aud_q = RMSNorm(dim_head, eps=eps)
            self.norm_aud_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )
        if not delete_img:
            self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

            if not self.only_cross_attention:
                # only relevant for the `AddedKVProcessor` classes
                self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
                self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            else:
                self.to_k = None
                self.to_v = None

        if not delete_text:
            if self.added_kv_proj_dim is not None:
                self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim)
                self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim)
                if self.context_pre_only is not None:
                    self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
                
        if not delete_aud:    
            if self.aud_kv_proj_dim is not None:
                self.aud_k_proj = nn.Linear(aud_kv_proj_dim, self.inner_kv_dim)
                self.aud_v_proj = nn.Linear(aud_kv_proj_dim, self.inner_kv_dim)
                if self.context_pre_only is not None:
                    self.aud_q_proj = nn.Linear(aud_kv_proj_dim, self.inner_dim)
                self.to_aud_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)
                    
        
        if not delete_img:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))

        if not delete_text:
            if self.context_pre_only is not None and not self.context_pre_only:
                self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        r"""
        Set whether to use npu flash attention from `torch_npu` or not.

        """
        if use_npu_flash_attention:
            processor = AttnProcessorNPU()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_0,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
            ),
        )

        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and is_custom_diffusion:
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for custom diffusion for attention processor type {self.processor}"
                )
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            if is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.info(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_custom_diffusion:
                attn_processor_class = (
                    CustomDiffusionAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else CustomDiffusionAttnProcessor
                )
                processor = attn_processor_class(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            else:
                # set attention processor
                # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
                # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
                # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
                processor = (
                    AttnProcessor2_0()
                    if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                    else AttnProcessor()
                )

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        audio_hidden_states:Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_hidden_states=audio_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @torch.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)

        self.fused_projections = fuse



class JointAttnProcessorAudio3_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self,use_dual_attention=False):
        self.use_dual_attention = use_dual_attention
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: AttentionAudioVideo,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        audio_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # sample_obj = 
        if hidden_states is not None:
            residual = hidden_states
            cutoff = residual.shape[1]
            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            cutoff = 0
        if encoder_hidden_states is not None:
            enc_len = encoder_hidden_states.shape[1]
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            enc_len = 0

      #  batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        queries = []
        keys = []
        values = []
        do_image = hidden_states is not None
        do_text = encoder_hidden_states is not None
        do_audio = audio_hidden_states is not None
        if do_image:
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            query = attn.to_q(hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            key = attn.to_k(hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            value = attn.to_v(hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            if self.use_dual_attention:
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
            queries.append(query)
            keys.append(key)
            values.append(value)
        else:
            query = None
            key = None
            value = None

        if do_text:
            batch_size = encoder_hidden_states.shape[0]
            seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            if self.use_dual_attention:
                if attn.norm_added_q is not None:
                    encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
                if attn.norm_added_k is not None:
                    encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)
                    
            queries.append(encoder_hidden_states_query_proj)
            keys.append(encoder_hidden_states_key_proj)
            values.append(encoder_hidden_states_value_proj)

        if do_audio:
            batch_size = audio_hidden_states.shape[0]
            seq_len = audio_hidden_states.shape[1]
            audio_hidden_states_query_proj = attn.aud_q_proj(audio_hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            audio_hidden_states_key_proj = attn.aud_k_proj(audio_hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            audio_hidden_states_value_proj = attn.aud_v_proj(audio_hidden_states).view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
            if self.use_dual_attention:
                if attn.norm_aud_q is not None:
                    audio_hidden_states_query_proj = attn.norm_aud_q(audio_hidden_states_query_proj)
                if attn.norm_aud_k is not None:
                    audio_hidden_states_key_proj = attn.norm_aud_k(audio_hidden_states_key_proj)
            queries.append(audio_hidden_states_query_proj)
            keys.append(audio_hidden_states_key_proj)
            values.append(audio_hidden_states_value_proj)
        else:
            pass
                
        # attention
        query = torch.cat(queries, dim=2)
        key = torch.cat(keys, dim=2)
        value = torch.cat(values, dim=2)
        batch_size = query.shape[0]
        #inner_dim = key.shape[-1]
        head_dim = query.shape[-1] #inner_dim // attn.heads

        if not self.use_dual_attention:
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)


        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        
        hidden_states, encoder_hidden_states,audio_hidden_states = (
            hidden_states[:, : cutoff],
            hidden_states[:, cutoff :cutoff+enc_len],
            hidden_states[:, cutoff+enc_len:],
        )
        if do_image: 
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        else:
            hidden_states = None
        if do_text:
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        else:
            encoder_hidden_states = None
        
        if do_audio:
            audio_hidden_states = attn.to_aud_out(audio_hidden_states)
        else:
            audio_hidden_states = None
        
        
        return hidden_states, encoder_hidden_states,audio_hidden_states


@maybe_allow_in_graph
class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, 
                 context_pre_only=False, context_output=True,audio_output=False,delete_img=False,delete_aud=False,delete_text=False,
                  qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        ):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if not delete_img:
            if use_dual_attention:
                self.norm1 = SD35AdaLayerNormZeroX(dim)
            else:
                self.norm1 = AdaLayerNormZero(dim)

        if not delete_text:
            if context_norm_type == "ada_norm_continous":
                self.norm1_context = AdaLayerNormContinuous(
                    dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm"
                )
            elif context_norm_type == "ada_norm_zero":
                self.norm1_context = AdaLayerNormZero(dim)
            else:
                raise ValueError(
                    f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
                )
        if hasattr(F, "scaled_dot_product_attention"):
            if audio_output:
                processor = JointAttnProcessorAudio3_0(use_dual_attention= qk_norm == 'rms_norm')
            else:
                processor = JointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.audio_output = audio_output
        if audio_output:
            self.attn = AttentionAudioVideo(
                query_dim=dim,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                aud_kv_proj_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=qk_norm,
                processor=processor,
                delete_img=False,
                delete_aud=False,
                delete_text=False
            )
        else:
            self.attn = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=qk_norm,
                processor=processor,
            )
        if use_dual_attention:
            if audio_output:
                self.attn2 = None
            else:
                self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=None,
                    dim_head=attention_head_dim,
                    heads=num_attention_heads,
                    out_dim=dim,
                    bias=True,
                    processor=JointAttnProcessor2_0(),
                    qk_norm=qk_norm,
                    eps=1e-6,
                )
        else:
            self.attn2 = None
            
        if not delete_img:

            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not delete_text:
            if context_output:
                self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
                self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            else:
                self.norm2_context = None
                self.ff_context = None
            
        if not delete_aud:
            if audio_output:
                self.ff_audio = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
                self.norm1_audio = AdaLayerNormZero(dim)
                self.norm2_audio = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) #AdaLayerNormZero(dim, elementwise_affine=False, eps=1e-6)
        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor,
        audio_hidden_states:torch.FloatTensor = None,temb_text=None,temb_audio=None
    ):
        encoder_hidden_states_base = encoder_hidden_states
        if hidden_states is not None:
            if self.use_dual_attention:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                    hidden_states, emb=temb
                )
            else:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            if temb_text is None:
                temb_text = temb
            if temb_audio is None:
                temb_audio = temb
        else:
            norm_hidden_states = None
            
        if encoder_hidden_states is not None:
            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb_text)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb_text
                )
        else:
            norm_encoder_hidden_states = None
        if audio_hidden_states is not None:
            norm_audio_hidden_states, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp = self.norm1_audio(
            audio_hidden_states, emb=temb_audio
        )
        else:
            norm_audio_hidden_states = None

        # Attention.
        if  self.audio_output:
            # assert audio_hidden_states is not None
            attn_output, context_attn_output,audio_attn_output = self.attn(
                hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
                audio_hidden_states=norm_audio_hidden_states
            )
        else:
            attn_output, context_attn_output = self.attn(
                            hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states,
            )


        # Process attention outputs for the `hidden_states`.
        
        if hidden_states is not None:
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output
            if self.use_dual_attention and self.attn2 is not None:
                attn_output2 = self.attn2(hidden_states=norm_hidden_states2)
                attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
                hidden_states = hidden_states + attn_output2
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output

        if encoder_hidden_states is not None:
            if self.context_pre_only:
                encoder_hidden_states = encoder_hidden_states_base
            else:
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output
                if self.norm2_context is not None:
                    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
         
                    context_ff_output = self.ff_context(norm_encoder_hidden_states)
                    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output
                    encoder_hidden_states = encoder_hidden_states + context_ff_output
        if audio_hidden_states is not None:
            audio_attn_output = a_gate_msa.unsqueeze(1) * audio_attn_output
            audio_hidden_states = audio_hidden_states + audio_attn_output
            norm_audio_hidden_states = self.norm2_audio(audio_hidden_states)
            norm_audio_hidden_states = norm_audio_hidden_states * (1 + a_scale_mlp[:, None]) + a_shift_mlp[:, None]
            audio_ff_output = self.ff_audio(norm_audio_hidden_states)
            audio_ff_output = a_gate_mlp.unsqueeze(1) * audio_ff_output
            audio_hidden_states = audio_hidden_states + audio_ff_output
            
            
        res = [encoder_hidden_states, hidden_states]
        if  self.audio_output:
            res.append(audio_hidden_states)
        return res

