export MODEL_DIR="/path/to/stable-diffusion-2-1/"
export CLIP_MODEL_DIR="/path/to/clip-vit-large-patch14/"
export OUTPUT_DIR="./runs/celeb/"
export NCCL_P2P_DISABLE=1
export DATASET_LOADING_SCRIPT_PATH="./my_dataset/train_dataset_loading_script.py"
export TORCH_DISTRIBUTED_DEBUG="INFO"
export WANDB__SERVICE_WAIT="300"

accelerate launch --main_process_port=29500 --mixed_precision="fp16" --multi_gpu -m examples.referencenet.train_referencenet \
	--pretrained_model_name_or_path=$MODEL_DIR \
	--pretrained_clip_model_name_or_path=$CLIP_MODEL_DIR \
	--output_dir=$OUTPUT_DIR \
	--dataset_loading_script_path=$DATASET_LOADING_SCRIPT_PATH \
	--resolution=512 \
	--learning_rate=1e-5 \
	--validation_source_image "./my_dataset/test/00482.png" \
	--validation_conditioning_image "./my_dataset/test/14795.png" \
	--train_batch_size=1 \
	--tracker_project_name="celeb" \
	--checkpointing_steps=10000 \
	--num_validation_images=1 \
	--validation_steps=1000 \
	--mixed_precision="fp16" \
	--gradient_checkpointing \
	--use_8bit_adam \
	--enable_xformers_memory_efficient_attention \
	--gradient_accumulation_steps=8 \
	--resume_from_checkpoint="latest" \
	--set_grads_to_none \
	--max_train_steps=60000 \
	--conditioning_dropout_prob=0.1 \
	--seed=0 \
	--report_to="wandb" \
	--random_flip \
	--dataloader_num_workers=8
