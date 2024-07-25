# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# limitations under the License.


import inspect
from typing import List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...models import AutoencoderKL
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
from diffusers import DiffusionPipeline
from ..stable_diffusion.pipeline_output import StableDiffusionPipelineOutput

from ...models.referencenet.unet_2d_condition import UNet2DConditionModel
from ...models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionReferenceNetPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionReferenceNetPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""


def cat_referencenet_states(states1, states2, dim=0):
    concatenated_states = []
    for i in range(len(states1)):
        # unet down x 3; mid x 1; up x 3
        unet_blocks = []
        for j in range(len(states1[i])):
            # cross attention down x 2; mid x 1; up x 2
            cross_attn_blocks = []
            for k in range(len(states1[i][j])):
                concatenated_cross_attn_blocks = torch.cat([states1[i][j][k], states2[i][j][k]], dim=dim)
                cross_attn_blocks.append(concatenated_cross_attn_blocks)
            unet_blocks.append(cross_attn_blocks)
        concatenated_states.append(unet_blocks)
    return concatenated_states


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionReferenceNetPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        referencenet: ReferenceNetModel,
        conditioning_referencenet: ReferenceNetModel,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            unet=unet,
            referencenet=referencenet,
            conditioning_referencenet=conditioning_referencenet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor_do_normalize = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def check_inputs(
        self,
        source_image,
        conditioning_image,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    ):
        # Check `image`
        if isinstance(self.referencenet, ReferenceNetModel):
            self.check_image(source_image, conditioning_image)
        else:
            assert False

        if not isinstance(control_guidance_start, (tuple, list)):
            control_guidance_start = [control_guidance_start]

        if not isinstance(control_guidance_end, (tuple, list)):
            control_guidance_end = [control_guidance_end]

        if len(control_guidance_start) != len(control_guidance_end):
            raise ValueError(
                f"`control_guidance_start` has {len(control_guidance_start)} elements, but `control_guidance_end` has {len(control_guidance_end)} elements. Make sure to provide the same number of elements to each list."
            )

        for start, end in zip(control_guidance_start, control_guidance_end):
            if start >= end:
                raise ValueError(
                    f"control guidance start: {start} cannot be larger or equal to control guidance end: {end}."
                )
            if start < 0.0:
                raise ValueError(f"control guidance start: {start} can't be smaller than 0.")
            if end > 1.0:
                raise ValueError(f"control guidance end: {end} can't be larger than 1.0.")

    def check_image(self, source_image, conditioning_image):
        image_batch_size = None
        for image in [source_image, conditioning_image]:
            image_is_pil = isinstance(image, PIL.Image.Image)
            image_is_tensor = isinstance(image, torch.Tensor)
            image_is_np = isinstance(image, np.ndarray)
            image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
            image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)
            image_is_np_list = isinstance(image, list) and isinstance(image[0], np.ndarray)

            if (
                not image_is_pil
                and not image_is_tensor
                and not image_is_np
                and not image_is_pil_list
                and not image_is_tensor_list
                and not image_is_np_list
            ):
                raise TypeError(
                    f"image must be passed and be one of PIL image, numpy array, torch tensor, list of PIL images, list of numpy arrays or list of torch tensors, but is {type(image)}"
                )

            if not image_batch_size:
                if image_is_pil:
                    image_batch_size = 1
                else:
                    image_batch_size = len(image)
            else:
                if image_is_pil:
                    if image_batch_size != 1:
                        raise ValueError(
                            f"Source image batch size must be same as conditioning image batch size. source image batch size: {image_batch_size}, conditioning image batch size: 1"
                        )
                elif image_batch_size != len(image):
                    raise ValueError(
                        f"Source image batch size must be same as conditioning image batch size. source image batch size: {image_batch_size}, conditioning image batch size: {len(image)}"
                    )

    def prepare_referencenet_input(
        self,
        image,
        width,
        height,
        device,
        dtype,
        do_classifier_free_guidance=False,
        is_null_conditioning=False,
        do_anonymization=False,
    ):
        init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        init_image = init_image.to(device=device, dtype=dtype)

        normalized_image = self.image_processor_do_normalize.preprocess(image, height=height, width=width).to(
            device=device, dtype=dtype
        )
        features = self.feature_extractor(images=init_image, do_rescale=False, return_tensors="pt").pixel_values.to(
            device=device
        )
        image_embeds = self.image_encoder(features).pooler_output.unsqueeze(1)
        latents = self.vae.encode(normalized_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        if do_classifier_free_guidance or do_anonymization:
            if is_null_conditioning:
                latents = torch.cat([torch.zeros_like(latents), latents])
                image_embeds = torch.cat([torch.zeros_like(image_embeds), image_embeds])
            else:
                latents = torch.cat([latents] * 2)
                image_embeds = torch.cat([image_embeds] * 2)

        return init_image, latents, image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def anonymization_degree(self):
        return self._anonymization_degree

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def do_anonymization(self):
        return self._anonymization_degree > 0

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        source_image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        conditioning_image: Union[PipelineImageInput, List[PipelineImageInput]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 2.5,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        anonymization_degree: float = 0.0,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            conditioning_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            anonymization_degree (`float`, *optional*, defaults to 0.0):
                Increasing the anonymization scale value encourages the model to produce images that diverge significantly
                from the conditioning image.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        referencenet = self.referencenet
        conditioning_referencenet = self.conditioning_referencenet

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            source_image=source_image,
            conditioning_image=conditioning_image,
        )

        self._guidance_scale = guidance_scale
        self._anonymization_degree = anonymization_degree

        # 2. Define call parameters
        image_is_pil = isinstance(source_image, PIL.Image.Image)
        if image_is_pil:
            batch_size = 1
        else:
            batch_size = len(source_image)

        device = self._execution_device

        # 4. Prepare image
        if isinstance(referencenet, ReferenceNetModel) and isinstance(conditioning_referencenet, ReferenceNetModel):
            if self.do_anonymization:
                source_image, source_latents, source_image_embeds = self.prepare_referencenet_input(
                    image=conditioning_image,
                    width=width,
                    height=height,
                    device=device,
                    dtype=referencenet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    is_null_conditioning=True,
                    do_anonymization=True,
                )
            else:
                source_image, source_latents, source_image_embeds = self.prepare_referencenet_input(
                    image=source_image,
                    width=width,
                    height=height,
                    device=device,
                    dtype=referencenet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    is_null_conditioning=True,
                    do_anonymization=False,
                )
            conditioning_image, conditioning_latents, conditioning_image_embeds = self.prepare_referencenet_input(
                image=conditioning_image,
                width=width,
                height=height,
                device=device,
                dtype=referencenet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                is_null_conditioning=False,
                do_anonymization=self.do_anonymization,
            )
            height, width = conditioning_image.shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=self.image_encoder.dtype,
            device=device,
            generator=generator,
        )

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        is_unet_compiled = is_compiled_module(self.unet)
        is_referencenet_compiled = is_compiled_module(self.referencenet)
        is_torch_higher_equal_2_1 = is_torch_version(">=", "2.1")

        if self.do_anonymization:
            source_image_embeds = self.nullify_image_embeds(source_image_embeds, self.anonymization_degree)

        referencenet_sample, referencenet_states = referencenet(
            sample=source_latents,
            timestep=0,
            encoder_hidden_states=source_image_embeds,
            return_dict=False,
        )

        if self.do_anonymization:
            referencenet_states = self.nullify_referencenet_states(referencenet_states, self.anonymization_degree)

        conditioning_referencenet_sample, conditioning_referencenet_states = conditioning_referencenet(
            sample=conditioning_latents,
            timestep=0,
            encoder_hidden_states=conditioning_image_embeds,
            return_dict=False,
        )

        concatenated_embeds = torch.cat([source_image_embeds, conditioning_image_embeds], dim=1)
        concatenated_referencenet_states = cat_referencenet_states(
            referencenet_states, conditioning_referencenet_states, dim=1
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Relevant thread:
                # https://dev-discuss.pytorch.org/t/cudagraphs-in-pytorch-2-0/1428
                if (is_unet_compiled and is_referencenet_compiled) and is_torch_higher_equal_2_1:
                    torch._inductor.cudagraph_mark_step_begin()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance or self.do_anonymization else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=concatenated_embeds,
                    referencenet_states=concatenated_referencenet_states,
                    return_dict=False,
                )[0]

                if self.do_anonymization:
                    uncond_noise_pred, cond_noise_pred = noise_pred.chunk(2)
                    if self.do_classifier_free_guidance:
                        noise_pred = (
                            1 - self.guidance_scale
                        ) * uncond_noise_pred + self.guidance_scale * cond_noise_pred
                    else:
                        noise_pred = cond_noise_pred
                elif self.do_classifier_free_guidance:
                    # perform guidance
                    uncond_noise_pred, cond_noise_pred = noise_pred.chunk(2)
                    noise_pred = (1 - self.guidance_scale) * uncond_noise_pred + self.guidance_scale * cond_noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # If we do sequential model offloading, let's offload unet and referencenet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.referencenet.to("cpu")
            self.conditioning_referencenet.to("cpu")
            torch.cuda.empty_cache()

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor_do_normalize.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionPipelineOutput(
            images=image,
            nsfw_content_detected=None,
        )

    def nullify_referencenet_states(self, referencenet_states, anonymization_degree):
        new_referencenet_states = []
        for i in range(len(referencenet_states)):
            unet_blocks = []
            for j in range(len(referencenet_states[i])):
                cross_attn_blocks = []
                for k in range(len(referencenet_states[i][j])):
                    # Split the tensor along the first dimension
                    split_tensors = torch.chunk(referencenet_states[i][j][k], chunks=2, dim=0)

                    # Select the first half of the split tensors
                    first_half_tensor = split_tensors[0]

                    # Select the second half of the split tensors
                    second_half_tensor = split_tensors[1]

                    # Nullify the second half of the split tensors
                    second_half_tensor = (
                        second_half_tensor * (1 - anonymization_degree) + first_half_tensor * anonymization_degree
                    )

                    # Concatenate the tensors along the first dimension
                    concatenated_tensor = torch.cat((first_half_tensor, second_half_tensor), dim=0)

                    cross_attn_blocks.append(concatenated_tensor)
                unet_blocks.append(tuple(cross_attn_blocks))
            new_referencenet_states.append(unet_blocks)

        return new_referencenet_states

    def nullify_image_embeds(self, image_embed, anonymization_degree):
        # Split the tensor along the first dimension
        split_tensors = torch.chunk(image_embed, chunks=2, dim=0)

        # Select the first half of the split tensors
        first_half_tensor = split_tensors[0]

        # Select the second half of the split tensors
        second_half_tensor = split_tensors[1]

        # Nullify the second half of the split tensors
        second_half_tensor = second_half_tensor * (1 - anonymization_degree)

        # Concatenate the tensors along the first dimension
        new_image_embed = torch.cat((first_half_tensor, second_half_tensor), dim=0)

        return new_image_embed
