import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms_v2
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from tqdm import tqdm
from accelerate import Accelerator

from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        required=False,
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
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path to the model trained by yourself",
    )
    parser.add_argument(
        "--dataset_loading_script_path",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset loading script file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test-infer/",
        help="The output directory where predictions are saved",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the test dataset will be resized to this resolution",
    )
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")
    parser.add_argument(
        "--anonymization_degree_start",
        type=float,
        default=0.0,
        help="Increasing the anonymization scale value encourages the model to produce images that diverge significantly from the conditioning image.",
    )
    parser.add_argument("--anonymization_degree_end", type=float, default=0.0)
    parser.add_argument("--num_anonymization_degrees", type=int, default=1)
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
        "--max_test_samples",
        type=int,
        default=None,
        help="Truncate the number of test examples to this value if set.",
    )
    parser.add_argument(
        "--vis_input",
        action="store_true",
        help="If set, save the input and generated images together as a single output image for easy visualization",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=1,
        help=(
            "The batch size for the test dataloader per device should be set to 1."
            "This setting does not affect performance, no matter how large the batch size is."
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

    args = parser.parse_args()
    return args


def combine_images(images):
    # Get the total width and maximum height of all images
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new image with the combined width and maximum height
    new_image = Image.new("RGB", (total_width, max_height))

    # Paste each image onto the new image horizontally
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image


def make_test_dataset(args):
    ds = load_dataset(path=args.dataset_loading_script_path, split="test", trust_remote_code=True)

    # Preprocessing the datasets.
    image_transforms = transforms_v2.Compose(
        [
            transforms_v2.Resize(args.resolution, interpolation=transforms_v2.InterpolationMode.BILINEAR),
            transforms_v2.CenterCrop(args.resolution)
            if args.center_crop
            else transforms_v2.RandomCrop(args.resolution),
        ]
    )

    def preprocess_test(examples):
        images = [image.convert("RGB") for image in examples["source_image"]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples["conditioning_image"]]
        conditioning_images = [image_transforms(image) for image in conditioning_images]

        examples["source_image"] = images
        examples["conditioning_image"] = conditioning_images

        return examples

    if args.max_test_samples is not None:
        max_test_samples = min(args.max_test_samples, len(ds))
        ds = ds.select(range(max_test_samples))

    test_dataset = ds.with_transform(preprocess_test)
    return test_dataset


def collate_fn(examples):
    source_images = [example["source_image"] for example in examples]
    conditioning_images = [example["conditioning_image"] for example in examples]
    source_image_paths = [example["source_image_path"] for example in examples]
    conditioning_image_paths = [example["conditioning_image_path"] for example in examples]

    return {
        "source_images": source_images,
        "conditioning_images": conditioning_images,
        "source_image_paths": source_image_paths,
        "conditioning_image_paths": conditioning_image_paths,
    }


if __name__ == "__main__":
    args = parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    os.makedirs(args.output_dir, exist_ok=True)

    if args.vis_input:
        output_vis_dir = Path(args.output_dir, "vis")
        output_vis_dir.mkdir(parents=True, exist_ok=True)

    generator = None

    # create & load model
    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
    referencenet = ReferenceNetModel.from_pretrained(args.model_path, subfolder="referencenet")
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        args.model_path, subfolder="conditioning_referencenet"
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_clip_model_name_or_path)
    image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_clip_model_name_or_path)
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)

    test_dataset = make_test_dataset(args)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.test_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = accelerator.prepare(test_dataloader)

    # Generate the list of evenly spaced numbers
    anonymization_degrees = np.linspace(
        args.anonymization_degree_start, args.anonymization_degree_end, args.num_anonymization_degrees
    )

    for step, batch in enumerate(tqdm(test_dataloader)):
        # Group corresponding items from each key together
        grouped_items = list(
            zip(
                batch["source_images"],
                batch["conditioning_images"],
                batch["source_image_paths"],
                batch["conditioning_image_paths"],
            )
        )

        for source_image, conditioning_image, source_image_path, conditioning_image_path in grouped_items:
            source_image_name = Path(source_image_path).stem
            conditioning_image_name = Path(conditioning_image_path).stem

            for index, anonymization_degree in enumerate(anonymization_degrees):
                filename = f"{source_image_name}-{conditioning_image_name}_{index:03}.png"
                save_to = Path(args.output_dir, filename)

                if save_to.is_file():
                    continue

                if args.seed is not None:
                    # create a generator for reproducibility; notice you don't place it on the GPU!
                    generator = torch.manual_seed(args.seed)

                image = pipe(
                    source_image=source_image,
                    conditioning_image=conditioning_image,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    anonymization_degree=anonymization_degree,
                ).images[0]

                image.save(save_to)
                save_vis_to = Path(output_vis_dir, filename)

                if args.vis_input and not save_vis_to.is_file():
                    if anonymization_degree > 0.0:
                        # face anonymization
                        combined_image = combine_images([conditioning_image, image])
                    else:
                        # face swapping
                        combined_image = combine_images([source_image, conditioning_image, image])
                    combined_image.save(save_vis_to)
