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
   
    if add_token_embed:
        prompt_embeds = text_encoder(text_input_ids)[0]
        #token_embeds = text_encoder.encoder.embed_tokens(text_input_ids)
        #prompt_embeds = prompt_embeds + token_embeds
        #prompt_embeds=token_embeds
    else:
        prompt_embeds = text_encoder(text_input_ids)[0]
        #prompt_embeds = F.normalize(prompt_embeds,dim=-1)+F.normalize(token_embeds,dim=-1)
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
        # breakpoint()
        prompt_embeds = (prompt_embeds - prompt_embeds.mean(-1,keepdim=True)) / (prompt_embeds.std(-1,keepdim=True)+1e-9)
        #prompt_embeds = F.normalize(prompt_embeds,dim=-1,p=2) * np.sqrt(prompt_embeds.shape[-1])
    return prompt_embeds, None

