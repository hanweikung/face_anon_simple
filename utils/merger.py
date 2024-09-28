import argparse

import cv2
import face_alignment
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paste a smaller foreground image onto a larger background image."
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        required=True,
        help="Path to output image",
    )
    args = parser.parse_args()
    return args


def paste_foreground_onto_background(fg_image, bg_image, rotation_matrix):
    fg_array = np.array(fg_image)
    bg_array = np.array(bg_image)

    # Get the dimensions (width and height) of the image
    height, width = bg_array.shape[:2]

    # Warp the foreground image twice using different constant boundary values
    warped_foreground_255 = cv2.warpAffine(
        fg_array,
        rotation_matrix,
        (width, height),
        np.empty_like(bg_array),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    warped_foreground_0 = cv2.warpAffine(
        fg_array,
        rotation_matrix,
        (width, height),
        np.empty_like(bg_array),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Create a mask by calculating the absolute difference between the two warped images
    diff = cv2.absdiff(warped_foreground_255, warped_foreground_0)
    mask = diff / 255.0

    # Place the warped foreground back into the original background image using the mask
    result = mask * bg_array + warped_foreground_0

    return Image.fromarray(result.astype("uint8"), "RGB")


if __name__ == "__main__":
    from diffusers.utils import load_image
    from extractor import FaceType, extract_faces

    args = parse_args()

    # sfd for SFD, dlib for Dlib and folder for existing bounding boxes.
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )
    face_type = FaceType.WHOLE_FACE
    init_image = load_image(args.input_image)
    face_images, image_to_face_matrices = extract_faces(fa, init_image, 256, face_type)

    result = init_image
    for face_image, image_to_face_mat in zip(face_images, image_to_face_matrices):
        result = paste_foreground_onto_background(face_image, result, image_to_face_mat)
    result.save(args.output_image)
