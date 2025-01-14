# Face Anonymization Made Simple (WACV 2025)

[arXiv](http://arxiv.org/abs/2411.00762)

![teaser](teaser.jpg)

Our face anonymization technique preserves the original facial expressions, head positioning, eye direction, and background elements, effectively masking identity while retaining other crucial details. The anonymized face blends seamlessly into its original photograph, making it ideal for diverse real-world applications.

## Setup

1. Clone the repository.

```bash
git clone https://github.com/hanweikung/face_anon_simple.git
```

2. Create a Python environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
```

## Usage
1. Import the library.

```python
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
```

2. Create & load models.

```python
face_model_id = "hkung/face-anon-simple"
clip_model_id = "openai/clip-vit-large-patch14"
sd_model_id = "stabilityai/stable-diffusion-2-1"

unet = UNet2DConditionModel.from_pretrained(
    face_model_id, subfolder="unet", use_safetensors=True
)
referencenet = ReferenceNetModel.from_pretrained(
    face_model_id, subfolder="referencenet", use_safetensors=True
)
conditioning_referencenet = ReferenceNetModel.from_pretrained(
    face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
)
vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
scheduler = DDPMScheduler.from_pretrained(
    sd_model_id, subfolder="scheduler", use_safetensors=True
)
feature_extractor = CLIPImageProcessor.from_pretrained(
    clip_model_id, use_safetensors=True
)
image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

pipe = StableDiffusionReferenceNetPipeline(
    unet=unet,
    referencenet=referencenet,
    conditioning_referencenet=conditioning_referencenet,
    vae=vae,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    scheduler=scheduler,
)
pipe = pipe.to("cuda")

generator = torch.manual_seed(1)
```

### Anonymize images with a single aligned face

Create an anonymized version of an image if the image contains a single face and that face has already been aligned similarly to those in the [FFHQ](https://github.com/NVlabs/ffhq-dataset) or [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans) datasets.

```python
# get an input image for anonymization
original_image = load_image("my_dataset/test/14795.png")

# generate an image that anonymizes faces
anon_image = pipe(
    source_image=original_image,
    conditioning_image=original_image,
    num_inference_steps=200,
    guidance_scale=4.0,
    generator=generator,
    anonymization_degree=1.25,
).images[0]
anon_image.save("anon.png")
```

### Anonymize images with one or multiple unaligned faces

Create an anonymized version of an image if it contains one or more unaligned faces.

```python
import face_alignment
from utils.anonymize_faces_in_image import anonymize_faces_in_image

# get an input image for anonymization
original_image = load_image("my_dataset/test/friends.jpg")

# SFD (likely best results, but slower)
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, face_detector="sfd"
)

# generate an image that anonymizes faces
anon_image = anonymize_faces_in_image(
    image=original_image,
    face_alignment=fa,
    pipe=pipe,
    generator=generator,
    face_image_size=512,
    num_inference_steps=25,
    guidance_scale=4.0,
    anonymization_degree=1.25,
)
anon_image.save("anon.png")
```

### Swap faces between two images

Create an image that swap faces.

```python
# get source and conditioning (driving) images for face swap
source_image = load_image("my_dataset/test/00482.png")
conditioning_image = load_image("my_dataset/test/14795.png")

# generate an image that swaps faces
swap_image = pipe(
    source_image=source_image,
    conditioning_image=conditioning_image,
    num_inference_steps=200,
    guidance_scale=4.0,
    generator=generator,
    anonymization_degree=0.0,
).images[0]
swap_image.save("swap.png")
```

We also provide the [demo.ipynb](https://github.com/hanweikung/face_anon_simple/blob/main/demo.ipynb) notebook, which guides you through the steps mentioned above.

## Acknowledgements

This work is built upon the [Diffusers](https://github.com/huggingface/diffusers) project. The [face extractor](https://github.com/hanweikung/face_anon_simple/blob/main/utils/extractor.py) is adapted from [DeepFaceLab](https://dagshub.com/idonov/DeepFaceLab/src/master/mainscripts/Extractor.py).