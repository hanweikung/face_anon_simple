# Face Anonymization Made Simple

## Setup

1. Clone the repository.

```bash
git clone https://github.com/hanweikung/fams.git
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
from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
```

2. Load the models.

```python
fams_model_id = "hkung/fams"
clip_model_id = "openai/clip-vit-large-patch14"
sd_model_id = "stabilityai/stable-diffusion-2-1"

unet = UNet2DConditionModel.from_pretrained(
    fams_model_id, subfolder="unet", use_safetensors=True
)
referencenet = ReferenceNetModel.from_pretrained(
    fams_model_id, subfolder="referencenet", use_safetensors=True
)
conditioning_referencenet = ReferenceNetModel.from_pretrained(
    fams_model_id, subfolder="conditioning_referencenet", use_safetensors=True
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
```

3. Create an image that anonymizes faces.

```python
generator = torch.manual_seed(1)

# get an input image for anonymization
original_image = load_image("my_dataset/14795.png")

# generate an image that anonymizes faces
anon_image = pipe(
    source_image=original_image,
    conditioning_image=original_image,
    num_inference_steps=200,
    guidance_scale=4.0,
    generator=generator,
    anonymization_degree=1.25,
).images[0]
```

4. Create an image that swap faces.

```python
# get source and conditioning (driving) images for face swap
source_image = load_image("my_dataset/00482.png")
conditioning_image = load_image("my_dataset/14795.png")

# generate an image that swaps faces
swap_image = pipe(
    source_image=source_image,
    conditioning_image=conditioning_image,
    num_inference_steps=200,
    guidance_scale=4.0,
    generator=generator,
    anonymization_degree=0.0,
).images[0]
```

We also provide the [demo.ipynb](https://github.com/hanweikung/fams/blob/main/demo.ipynb) notebook, which guides you through the steps mentioned above.