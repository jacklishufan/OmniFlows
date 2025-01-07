
#
export MODEL_PATH='ckpts/v0.5'
export OUTPUT_DIR=/path/to/save/checkpoint
export DATA_DIR=config/data.yml
export NGPUS=8
export LOAD_PATH=ckpts/v0.5/
export VAE_PATH=ckpts/v0.5/text_vae
accelerate launch --config_file scripts/config.json --main_process_port 29507 --num_processes $NGPUS scripts/train.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --instance_data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --gradient_accumulation_steps 1 \
  --train_batch_size=8 \
  --ema_start 1000 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --checkpointing_steps=1000 \
  --ema_momentum 0.999 \
  --checkpoints_total_limit 2 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1000 \
  --dataloader_num_workers 1 \
  --text_vae $VAE_PATH \
  --max_train_steps=100000 \
  --ema_validation \
  --gradient_checkpointing \
  --ema_interval 100 \
  --lr_text_factor 1 \
  --lr_aud_factor 1 \
  --tokenizer ckpts/v0.5/vae_tokenizer \
  ${@} \
  --use_ema \
  --precondition_text_output \
  --load $LOAD_PATH \
  --resume_from_checkpoint latest \
