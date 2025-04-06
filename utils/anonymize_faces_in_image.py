import face_alignment
from PIL import Image
import torch

from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
from utils.extractor import extract_faces
from utils.merger import paste_foreground_onto_background


def anonymize_faces_in_image(
    image: Image,
    face_alignment: face_alignment.FaceAlignment,
    pipe: StableDiffusionReferenceNetPipeline,
    generator: torch.Generator = None,
    face_image_size: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 4,
    anonymization_degree: float = 1.25,
) -> Image:
    face_images, image_to_face_matrices = extract_faces(
        face_alignment, image, face_image_size
    )

    anon_image = image
    for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
        # generate an image that anonymizes faces
        anon_face_image = pipe(
            source_image=face_image,
            conditioning_image=face_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=face_image_size,
            height=face_image_size,
        ).images[0]

        anon_image = paste_foreground_onto_background(
            anon_face_image, anon_image, image_to_face_mat
        )

    return anon_image
