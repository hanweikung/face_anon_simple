#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import logging
import math
import os
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
    cat_referencenet_states,
)
from examples.referencenet.infer_referencenet import combine_images
from PIL import Image
from PIL import ImageFile

import torchvision.transforms.v2 as transforms_v2

import albumentations as A

ImageFile.LOAD_TRUNCATED_IMAGES = True

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


def set_parts_of_model_for_gradient_computation(module):
    # Include attention blocks in gradient computation
    for attn_processor_name, attn_processor in module.attn_processors.items():
        attn_module = module
        for n in attn_processor_name.split(".")[:-1]:
            attn_module = getattr(attn_module, n)
        attn_module.requires_grad_(True)

    # Include transformer blocks in gradient computation
    tb_type = type(module.down_blocks[0].attentions[0].transformer_blocks[0])
    attn_modules = [module for module in torch_dfs(module) if isinstance(module, tb_type)]
    attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
    [attn_module.requires_grad_(True) for attn_module in attn_modules]

    return module


def recursive_multiply(element, tensor):
    if isinstance(element, tuple) or isinstance(element, list):
        for element in element:
            recursive_multiply(element, tensor)
    elif torch.is_tensor(element):
        # In-place multiplication
        element.mul_(tensor)
    else:
        raise ValueError("Invalid type encountered in the element")


def log_validation(
    vae, unet, feature_extractor, image_encoder, referencenet, conditioning_referencenet, args, accelerator, step
):
    logger.info("Running validation... ")
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionReferenceNetPipeline(
        unet=accelerator.unwrap_model(unet),
        referencenet=accelerator.unwrap_model(referencenet),
        conditioning_referencenet=accelerator.unwrap_model(conditioning_referencenet),
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )

    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_conditioning_image) == len(args.validation_source_image):
        validation_conditioning_images = args.validation_conditioning_image
        validation_source_images = args.validation_source_image
    elif len(args.validation_conditioning_image) == 1:
        validation_conditioning_images = args.validation_conditioning_image * len(args.validation_source_image)
        validation_source_images = args.validation_source_image
    elif len(args.validation_source_image) == 1:
        validation_conditioning_images = args.validation_conditioning_image
        validation_source_images = args.validation_source_image * len(args.validation_conditioning_image)
    else:
        raise ValueError(
            "number of `args.validation_conditioning_image` and `args.validation_source_image` should be checked in `parse_args`"
        )
    image_logs = []

    image_path = os.path.join(args.output_dir, "outputs")
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    for i, (validation_source_image, validation_conditioning_image) in enumerate(
        zip(validation_source_images, validation_conditioning_images)
    ):
        source_image_filename_without_ext = Path(validation_source_image).stem
        conditioning_image_filename_without_ext = Path(validation_conditioning_image).stem

        # Source images
        validation_source_image = Image.open(validation_source_image).convert("RGB")
        validation_source_image = transforms_v2.Resize(size=(args.resolution, args.resolution), antialias=True)(
            validation_source_image
        )

        # Driving images
        validation_conditioning_image = Image.open(validation_conditioning_image).convert("RGB")
        validation_conditioning_image = transforms_v2.Resize(size=(args.resolution, args.resolution), antialias=True)(
            validation_conditioning_image
        )

        images = []
        for n in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(
                    source_image=validation_source_image,
                    conditioning_image=validation_conditioning_image,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=200,
                    guidance_scale=4.0,
                    generator=generator,
                ).images[0]

            images.append(image)

            combined_images = combine_images([validation_source_image, validation_conditioning_image, image])
            save_to = Path(
                image_path,
                f"{i:03}_src_{source_image_filename_without_ext}_drv_{conditioning_image_filename_without_ext}_{n:03}.png",
            )
            combined_images.save(save_to, format="PNG")

        image_logs.append(
            {
                "validation_source_image": validation_source_image,
                "validation_conditioning_image": validation_conditioning_image,
                "images": images,
            }
        )

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for i, log in enumerate(image_logs):
                images = log["images"]
                validation_source_image = log["validation_source_image"]
                validation_conditioning_image = log["validation_conditioning_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_source_image))
                formatted_images.append(np.asarray(validation_conditioning_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(f"{i:05}", formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_source_image = log["validation_source_image"]
                validation_conditioning_image = log["validation_conditioning_image"]

                formatted_images.append(wandb.Image(validation_source_image, caption="Source"))
                formatted_images.append(wandb.Image(validation_conditioning_image, caption="Conditioning"))

                for image in images:
                    image = wandb.Image(image, caption="Generated")
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_clip_model_name_or_path",
        type=str,
        default="openai/clip-vit-large-patch14",
        required=False,
        help="Path to pretrained CLIP model or model identifier from huggingface.co/models.",
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
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--dataset_loading_script_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset loading script file",
    )
    parser.add_argument(
        "--source_image_column",
        type=str,
        default="source_image",
        help="The column of the dataset containing the referencenet source image.",
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the referencenet conditioning image.",
    )
    parser.add_argument(
        "--ground_truth_column",
        type=str,
        default="ground_truth",
        help="The column of the dataset containing the ground truth image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
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
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_conditioning_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the referencenet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_source_image`s, a"
            " a single `--validation_source_image` to be used with all `--validation_conditioning_image`s, or a single"
            " `--validation_conditioning_image` that will be used with all `--validation_source_image`s."
        ),
    )
    parser.add_argument(
        "--validation_source_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of source images evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_conditioning_image`s, a single `--validation_conditioning_image`"
            " to be used with all source images, or a single source image that will be used with all `--validation_conditioning_image`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_conditioning_image`, `--validation_source_image` pair",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_source_image` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditioning image used during training",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_loading_script_path is None:
        raise ValueError("Need a script to load training dataset.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def make_train_dataset(args, feature_extractor, accelerator):
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    dataset = load_dataset(path=args.dataset_loading_script_path, split="train", trust_remote_code=True)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    if args.source_image_column is None:
        source_image_column = column_names[0]
        logger.info(f"image column defaulting to {source_image_column}")
    else:
        source_image_column = args.source_image_column
        if source_image_column not in column_names:
            raise ValueError(
                f"`--source_image_column` value '{args.source_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[1]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.ground_truth_column is None:
        ground_truth_column = column_names[2]
        logger.info(f"ground truth column defaulting to {ground_truth_column}")
    else:
        ground_truth_column = args.ground_truth_column
        if ground_truth_column not in column_names:
            raise ValueError(
                f"`--ground_truth_column` value '{args.ground_truth_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def extract_features(images):
        features = []
        for image in images:
            feature = feature_extractor(images=image, do_rescale=False, return_tensors="pt").pixel_values[0]
            features.append(feature)
        return features

    # The pipeline expects two images as inputs named image and image0 and will output numpy arrays.
    albumentations_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5 if args.random_flip else 0.0),
        ],
        additional_targets={"image0": "image", "image1": "image"},
    )

    torchvision_transforms = transforms_v2.Compose(
        [
            transforms_v2.ToImage(),
            transforms_v2.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms_v2.CenterCrop(args.resolution),
            transforms_v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # Create a Normalize transform
    normalize_transforms = transforms_v2.Compose([transforms_v2.Normalize(mean=[0.5], std=[0.5])])

    def preprocess_train(examples):
        source_images = [image.convert("RGB") for image in examples[source_image_column]]
        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        ground_truth = [image.convert("RGB") for image in examples[ground_truth_column]]

        for i, (src, cond, gt) in enumerate(zip(source_images, conditioning_images, ground_truth)):
            # Convert PIL image to numpy array
            src = np.array(src)
            cond = np.array(cond)
            gt = np.array(gt)

            # Apply the same augmentation with the same parameters to multiple images
            augmented = albumentations_transform(image=src, image0=cond, image1=gt)

            # Convert numpy array to PIL image
            source_images[i] = Image.fromarray(augmented["image"])
            conditioning_images[i] = Image.fromarray(augmented["image0"])
            ground_truth[i] = Image.fromarray(augmented["image1"])

        source_images = [torchvision_transforms(image) for image in source_images]
        conditioning_images = [torchvision_transforms(image) for image in conditioning_images]
        ground_truth = [torchvision_transforms(image) for image in ground_truth]

        examples["source_images"] = [normalize_transforms(image) for image in source_images]
        examples["conditioning_images"] = [normalize_transforms(image) for image in conditioning_images]
        examples["ground_truth"] = [normalize_transforms(image) for image in ground_truth]

        examples["clip_source_images"] = extract_features(source_images)
        examples["clip_conditioning_images"] = extract_features(conditioning_images)
        examples["clip_ground_truth"] = extract_features(ground_truth)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    ground_truth = [example["ground_truth"] for example in examples]
    source_images = [example["source_images"] for example in examples]
    conditioning_images = [example["conditioning_images"] for example in examples]
    clip_ground_truth = [example["clip_ground_truth"] for example in examples]
    clip_source_images = [example["clip_source_images"] for example in examples]
    clip_conditioning_images = [example["clip_conditioning_images"] for example in examples]

    ground_truth = torch.stack(ground_truth)
    ground_truth = ground_truth.to(memory_format=torch.contiguous_format).float()

    source_images = torch.stack(source_images)
    source_images = source_images.to(memory_format=torch.contiguous_format).float()

    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(memory_format=torch.contiguous_format).float()

    clip_ground_truth = torch.stack(clip_ground_truth)
    clip_ground_truth = clip_ground_truth.to(memory_format=torch.contiguous_format).float()

    clip_source_images = torch.stack(clip_source_images)
    clip_source_images = clip_source_images.to(memory_format=torch.contiguous_format).float()

    clip_conditioning_images = torch.stack(clip_conditioning_images)
    clip_conditioning_images = clip_conditioning_images.to(memory_format=torch.contiguous_format).float()

    return {
        "ground_truth": ground_truth,
        "source_images": source_images,
        "conditioning_images": conditioning_images,
        "clip_ground_truth": clip_ground_truth,
        "clip_source_images": clip_source_images,
        "clip_conditioning_images": clip_conditioning_images,
    }


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_clip_model_name_or_path)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_clip_model_name_or_path)
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    referencenet = ReferenceNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.train()
    # unet.requires_grad_(False)
    # for p in unet.down_blocks[0].attentions[0].transformer_blocks[0].parameters():
    #     print(p.requires_grad)  # Expect to be False
    #     break
    # unet = set_parts_of_model_for_gradient_computation(unet)
    # Check if gradient will be calculated on the tensor
    # for p in unet.down_blocks[0].attentions[0].transformer_blocks[0].parameters():
    #     print(p.requires_grad)  # Expect to be True
    #     break

    # referencenet.train()
    referencenet.requires_grad_(False)
    referencenet = set_parts_of_model_for_gradient_computation(referencenet)

    # conditioning_referencenet.train()
    conditioning_referencenet.requires_grad_(False)
    conditioning_referencenet = set_parts_of_model_for_gradient_computation(conditioning_referencenet)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if False:
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for model in models:
                    sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "referencenet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, type(accelerator.unwrap_model(referencenet))):
                    # load transformers style into model
                    load_model = ReferenceNetModel.from_pretrained(input_dir, subfolder="referencenet")
                else:
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, referencenet.parameters()))
    trainable_params += list(filter(lambda p: p.requires_grad, conditioning_referencenet.parameters()))

    optimizer = optimizer_cls(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = make_train_dataset(args, feature_extractor, accelerator)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

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
    )

    # Prepare everything with our `accelerator`.
    unet, referencenet, conditioning_referencenet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, referencenet, conditioning_referencenet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora image_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_conditioning_image")
        tracker_config.pop("validation_source_image")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
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
            # Get the most recent checkpoint
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
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    last_step = noise_scheduler.config.num_train_timesteps - 1
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, referencenet, conditioning_referencenet):
                # Ground truth
                ground_truth = batch["ground_truth"].to(weight_dtype)
                clip_ground_truth = batch["clip_ground_truth"].to(weight_dtype)

                # Source images
                source_images = batch["source_images"].to(weight_dtype)
                clip_source_images = batch["clip_source_images"].to(weight_dtype)

                # Driving images
                conditioning_images = batch["conditioning_images"].to(weight_dtype)
                clip_conditioning_images = batch["clip_conditioning_images"].to(weight_dtype)

                # Convert images to latent space
                latents = vae.encode(ground_truth).latent_dist.sample()
                latents *= vae.config.scaling_factor
                source_latents = vae.encode(source_images).latent_dist.sample()
                source_latents *= vae.config.scaling_factor
                conditioning_latents = vae.encode(conditioning_images).latent_dist.sample()
                conditioning_latents *= vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                ref_timesteps = torch.zeros_like(timesteps)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Fix zero PSNR
                noisy_latents[timesteps == last_step] = noise[timesteps == last_step]

                # Get the source image embedding
                source_image_embeds = image_encoder(clip_source_images).pooler_output.unsqueeze(1)

                # Get the conditioning image embedding
                conditioning_image_embeds = image_encoder(clip_conditioning_images).pooler_output.unsqueeze(1)

                # Conditioning dropout to support classifier-free guidance during inference.
                random_p = torch.rand(bsz, device=accelerator.device, generator=generator)
                if args.conditioning_dropout_prob is not None:
                    # Sample masks for the source images.
                    image_mask = 1 - (random_p < args.conditioning_dropout_prob).to(source_images.dtype)
                    # Final image conditioning.
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    source_latents = image_mask * source_latents

                    image_mask = image_mask.reshape(bsz, 1, 1)
                    source_image_embeds = image_mask * source_image_embeds

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Referencenet pass
                referencenet_sample, referencenet_states = referencenet(
                    sample=source_latents,
                    timestep=ref_timesteps,
                    encoder_hidden_states=source_image_embeds,
                    return_dict=False,
                )

                if False:
                    # Sample masks for the referencenet states.
                    referencenet_states_mask = 1 - (random_p < args.conditioning_dropout_prob).to(referencenet.dtype)
                    # Final referencenet states conditioning.
                    referencenet_states_mask = referencenet_states_mask.reshape(bsz, 1, 1)
                    recursive_multiply(referencenet_states, referencenet_states_mask)

                conditioning_referencenet_sample, conditioning_referencenet_states = conditioning_referencenet(
                    sample=conditioning_latents,
                    timestep=ref_timesteps,
                    encoder_hidden_states=conditioning_image_embeds,
                    return_dict=False,
                )

                concatenated_embeds = torch.cat([source_image_embeds, conditioning_image_embeds], dim=1)
                concatenated_referencenet_states = cat_referencenet_states(
                    referencenet_states,
                    conditioning_referencenet_states,
                    dim=1,
                )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=concatenated_embeds,
                    referencenet_states=concatenated_referencenet_states,
                ).sample
                # Fix error by adding with 0 weight
                model_pred += 0 * (referencenet_sample + conditioning_referencenet_sample)

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
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

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_conditioning_image is not None and global_step % args.validation_steps == 0:
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        log_validation(
                            vae=vae,
                            unet=unet,
                            feature_extractor=feature_extractor,
                            image_encoder=image_encoder,
                            referencenet=referencenet,
                            conditioning_referencenet=conditioning_referencenet,
                            args=args,
                            accelerator=accelerator,
                            step=global_step,
                        )
                        if args.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(Path(args.output_dir, "unet"))
        referencenet = accelerator.unwrap_model(referencenet)
        referencenet.save_pretrained(Path(args.output_dir, "referencenet"))
        conditioning_referencenet = accelerator.unwrap_model(conditioning_referencenet)
        conditioning_referencenet.save_pretrained(Path(args.output_dir, "conditioning_referencenet"))

        # Run a final round of inference.
        log_validation(
            vae=vae,
            unet=unet,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            args=args,
            accelerator=accelerator,
            step=global_step,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
