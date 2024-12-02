from transformers import LlamaModel,LlamaPreTrainedModel,LlamaConfig,LlamaForCausalLM,Cache,GPT2LMHeadModel
from transformers.modeling_outputs import ModelOutput,CausalLMOutputWithCrossAttentions
from typing import Tuple,Dict,Any
from transformers.models.llama.modeling_llama import CausalLMOutputWithPast
from torch import nn
from typing import Union,Optional,List
import torch
from transformers import AutoTokenizer
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
import torch.nn.functional as F
from transformers import T5EncoderModel
from diffusers.utils.accelerate_utils import apply_forward_hook
from transformers import Blip2QFormerConfig,Blip2QFormerModel,T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput,BaseModelOutput
from omniflow.utils.text_encode import encode_prompt_vae
from transformers.activations import ACT2FN
import numpy as np
import warnings

class MLP(nn.Module):
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


class VAEMixIn:

    @apply_forward_hook
    def encode(self,prompt=None,input_ids=None,drop=False,tokenizer=None,sample=False,legacy=None):
        assert self.encoder is not None
        legacy = self.vae_dim == 1536
        if type(prompt) == str:
            prompt = [prompt]
        prompt_embeds, pooled_prompt_embeds = encode_prompt_vae(
                            text_encoders=self.encoder,
                            tokenizers=tokenizer,
                            prompt=prompt,
                            max_sequence_length=77,
                            text_input_ids_list=[input_ids
                                                ],
                            normalize=True
        )
        not_null = torch.tensor([len(x) > 0 for x in prompt]).view(-1,1,1).to(prompt_embeds)
        if legacy:
            not_null=1
        prompt_embeds = prompt_embeds * not_null
        # if drop:
        #     prompt_embeds = prompt_embeds * 0
        if sample:
            prompt_embeds = self.connect(prompt_embeds,1)[0]
        else:
            prompt_embeds = self.connect_mean(prompt_embeds,1)
        if drop:
            prompt_embeds = prompt_embeds * 0
        return prompt_embeds

        
    @apply_forward_hook
    def connect(self, bert_fea, nsamples=1):
        # (batch_size, nz)
        query = self.query.repeat(bert_fea.shape[0],1,1)
        # breakpoint()
        # self.qformer.apply(self.qformer._init_weights)
        query_results = self.qformer(query,encoder_hidden_states=bert_fea)[0]
        mean, logvar = self.vae_proj(query_results).chunk(2, -1)
        mean = self.q_norm(mean)
        z = mean + torch.rand_like(mean) * 0.001
        return z,torch.tensor(0,device=mean.device,dtype=mean.dtype)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        z = self.reparameterize(mean, logvar, nsamples)
        KL = 0.5 * (mean.pow(2) + logvar.exp() - logvar - 1).mean(dim=-1).mean()
        return z, KL

    @apply_forward_hook
    def connect_mean(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)
        query = self.query.repeat(bert_fea.shape[0],1,1)
        #query = self.q_norm(query)
        query_results = self.qformer(query,encoder_hidden_states=bert_fea)[0]
        
        mean, logvar = self.vae_proj(query_results).chunk(2, -1)
        mean = self.q_norm(mean)
        # pdb.set_trace()
        # mean, logvar = mean.squeeze(0), logvar.squeeze(0)

        # (batch, nsamples, nz)
        
        return mean
        
        
    
    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """

        std = logvar.mul(0.5).exp()
        mu_expd = mu.repeat(nsamples,1,1)
        std_expd = std.repeat(nsamples,1,1)
        eps = torch.rand_like(mu_expd)

        return mu_expd + torch.mul(eps, std_expd)

__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class LatentConnector(VAEMixIn):
    
    def __init__(self, config):
        super().__init__(config)
        self.vae_proj =  nn.Linear(4096,4096*2)
 
class LLamaForLatentConnector(LlamaForCausalLM,VAEMixIn):
    
    def __init__(self, config,prompt_size=1024,vae_dim=1536,
                 vae_hidden_size=1536,n_tokens=32,pre_norm=False,norm_type='layer_norm',vae_scale=1.0):
        super().__init__(config)
        
        
        if hasattr(config,'vae_config'):
            vae_config = config.vae_config
            prompt_size = vae_config['prompt_size']
            vae_dim = vae_config['vae_dim']
            vae_hidden_size = vae_config['vae_hidden_size']
            n_tokens = vae_config['n_tokens']
            pre_norm = vae_config['pre_norm']
            norm_type = vae_config['norm_type']
            vae_scale = vae_config['vae_scale']

        self.prompt_size = prompt_size
        self.vae_dim = vae_dim
        self.vae_hidden_size = vae_hidden_size
        self.pre_norm = pre_norm
        self.vae_scale = vae_scale

        q_config = Blip2QFormerConfig(hidden_size=vae_hidden_size,num_hidden_layers=6,encoder_hidden_size=prompt_size)
        self.query = nn.Parameter(torch.randn(1,n_tokens,vae_hidden_size))
        norm_dim =  vae_hidden_size if pre_norm else vae_dim
        if norm_type == 'rms_norm':
            self.q_norm = nn.RMSNorm(norm_dim,eps=1e-6)
        else:
            self.q_norm = nn.LayerNorm(norm_dim,elementwise_affine=False)
            
        self.qformer = Blip2QFormerModel(q_config)
        self.vae_proj =  nn.Linear(vae_hidden_size,vae_dim*2)
        self.input_proj = nn.Linear(vae_dim,config.hidden_size) if config.hidden_size != vae_dim else nn.Identity()
       
        self.post_init()
        self.encoder = None
        
    def set_vae_scale(self,v):
        self.vae_scale = 1.0
        
    def set_encoder(self,encoder):
        self.encoder = encoder
        
    def prepare_tokenizer(self,tokenizer):
         return tokenizer
     
    @apply_forward_hook
    def connect(self, bert_fea, nsamples=1):
        # (batch_size, nz)
        query = self.query.repeat(bert_fea.shape[0],1,1)
        # breakpoint()
        # self.qformer.apply(self.qformer._init_weights)
        query_results = self.qformer(query,encoder_hidden_states=bert_fea)[0]
        if self.pre_norm:
            query_results =  self.q_norm(query_results)
        mean, logvar = self.vae_proj(query_results).chunk(2, -1)
        # do not use logvar when used as VAE
        if not self.pre_norm:
            mean = self.q_norm(mean)
        z = mean + torch.rand_like(mean) * 0.001 # do not use logvar when used as VAE, use a fixed std err instead
        return z * self.vae_scale ,torch.tensor(0,device=mean.device,dtype=mean.dtype)


    @apply_forward_hook
    def connect_mean(self, bert_fea, nsamples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """

        # (batch_size, nz)
        query = self.query.repeat(bert_fea.shape[0],1,1)
        #query = self.q_norm(query)

        query_results = self.qformer(query,encoder_hidden_states=bert_fea)[0]
        
        if self.pre_norm:
            query_results =  self.q_norm(query_results)
        mean, logvar = self.vae_proj(query_results).chunk(2, -1)
        if not self.pre_norm:
            mean = self.q_norm(mean)

        
        return mean  * self.vae_scale
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        reduction:str = 'mean'
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        inputs_embeds = self.model.embed_tokens(input_ids)
        seq_len = inputs_embeds.shape[1]
        input_ids = None
        if latents is not None:
            latents = latents / self.vae_scale
            latents = self.input_proj(latents)
            inputs_embeds = torch.cat([latents,inputs_embeds],dim=1)
        

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0][:,-seq_len:]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction=reduction)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if 'past_key_values' in outputs else None,
            hidden_states=outputs.hidden_states if 'hidden_states' in outputs else None,
            attentions=outputs.attentions if 'attentions' in outputs else None,
        )
        

    def generate(self,input_ids=None,latents=None,*args,**kwargs):
        with torch.no_grad():
            if latents is not None:
                if input_ids is None:
                    input_ids = torch.tensor([[1]*(latents.shape[1]+1)]*len(latents)).cuda()
                y = self.forward(latents=latents,input_ids=torch.tensor([[]]*len(latents)).cuda().long(),use_cache=True)
                cache = y.past_key_values
                return super().generate(*args,**kwargs,past_key_values=cache,input_ids=input_ids)
            else: 
                return super().generate(*args,**kwargs)
            
