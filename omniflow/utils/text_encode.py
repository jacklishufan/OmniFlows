import torch

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
    add_token_embed=True,
    dtype=None
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    text_input_ids = text_input_ids.to(device)
   
    prompt_embeds = text_encoder(text_input_ids)[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def cat_and_pad(embeds,max_dim=None):
    assert type(embeds) == list
    new_embeds = []
    if max_dim == None:
        max_dim = max([x.shape[-1] for x in embeds])
    for embed in embeds:
        new_embeds.append(
            torch.nn.functional.pad(
                embed,(0,max_dim-embed.shape[-1])
            )
        )
    return torch.cat(new_embeds,dim=-2)


def encode_prompt_vae(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
    add_token_embed=True,
    normalize=False
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders,
        tokenizers,
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders.device,
        add_token_embed=add_token_embed
    )
    prompt_embeds = t5_prompt_embed
    if normalize:
        prompt_embeds = (prompt_embeds - prompt_embeds.mean(-1,keepdim=True)) / (prompt_embeds.std(-1,keepdim=True)+1e-9)
    return prompt_embeds, None



def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds



def encode_prompt_train(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
    add_token_embed=True,
    normalize=False,
    use_t5=False,
    drops = [False,False,False,False]
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        if drops[2*i]:
            prompt_embeds *= 0
        if drops[2*i+1]:
            pooled_prompt_embeds  *= 0
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)
    if use_t5:
        t5_prompt_embed = _encode_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
            device=device if device is not None else text_encoders[-1].device,
            add_token_embed=add_token_embed
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    else: 
        prompt_embeds = clip_prompt_embeds
    if normalize:
        prompt_embeds = (prompt_embeds - prompt_embeds.mean(-1,keepdim=True)) / (prompt_embeds.std(-1,keepdim=True)+1e-9)
    return prompt_embeds, pooled_prompt_embeds



def encode_prompt_for_decoder(prompt,tokenizer,append_eos=True,device=None,pad_length=32):
    if 'gpt2' in tokenizer.name_or_path:
        prompt = list([(tokenizer.bos_token+x+tokenizer.eos_token if not x.endswith(tokenizer.eos_token) else x ) for x in prompt])
    else:
        prompt = list([(x+tokenizer.eos_token if not x.endswith(tokenizer.eos_token) else x ) for x in prompt])
    z = tokenizer(
            prompt,
            padding="longest",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
    )
    if device is not None:
        z = z.to(device)
    labels = z.input_ids.clone()
    labels[[labels == tokenizer.pad_token_id]] = -100
    attention_mask = z.attention_mask
    attention_mask_pre = torch.ones(attention_mask.shape[0],pad_length).to(attention_mask)
    attention_mask = torch.cat([attention_mask_pre,attention_mask],dim=1)
    return dict(
        input_ids=z.input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )