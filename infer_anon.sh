export PRETRAINED_MODEL_DIR="/data/han-wei/models/stable-diffusion-2-1/"
export PRETRAINED_CLIP_MODEL_DIR="/data/han-wei/models/clip-vit-large-patch14/"
export MODEL_DIR="/data/han-wei/trash/fams/"
export OUTPUT_DIR="results/anon/"
export DATASET_LOADING_SCRIPT_PATH="my_dataset/my_dataset_anon.py"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG="INFO"

accelerate launch --main_process_port=29500 --multi_gpu -m examples.referencenet.infer_referencenet \
	--pretrained_model_name_or_path=$PRETRAINED_MODEL_DIR \
	--pretrained_clip_model_name_or_path=$PRETRAINED_CLIP_MODEL_DIR \
	--output_dir=$OUTPUT_DIR \
	--model_path=$MODEL_DIR \
	--dataset_loading_script_path=$DATASET_LOADING_SCRIPT_PATH \
	--resolution "512" \
	--seed "3" \
	--guidance_scale "4.0" \
	--num_inference_steps "200" \
	--anonymization_degree_start "1.25" \
	--anonymization_degree_end "1.25" \
	--num_anonymization_degrees "1" \
	--center_crop \
	--max_test_samples "2" \
	--vis_input \
	--dataloader_num_workers "8"
