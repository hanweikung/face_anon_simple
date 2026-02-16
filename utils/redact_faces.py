import argparse
import cv2
import face_alignment
import numpy as np
from PIL import Image

from extractor import extract_faces, FaceType
from merger import paste_foreground_onto_background


def blur_image(image, blur_strength=51):
    """
    Apply Gaussian blur to an image.

    Args:
        image: PIL Image to blur
        blur_strength: Kernel size for Gaussian blur (must be odd number)

    Returns:
        Blurred PIL Image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Apply Gaussian blur
    blurred_array = cv2.GaussianBlur(img_array, (blur_strength, blur_strength), 0)

    # Convert back to PIL Image
    return Image.fromarray(blurred_array)


def pixelate_image(image, pixel_size=16):
    """
    Apply pixelation effect to an image.

    Args:
        image: PIL Image to pixelate
        pixel_size: Size of pixelation blocks (default: 16)

    Returns:
        Pixelated PIL Image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Get dimensions
    height, width = img_array.shape[:2]

    # Resize down and then back up to create pixelation effect
    temp_height = max(1, height // pixel_size)
    temp_width = max(1, width // pixel_size)

    # Downscale
    temp = cv2.resize(
        img_array, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR
    )

    # Upscale back to original size
    pixelated_array = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

    # Convert back to PIL Image
    return Image.fromarray(pixelated_array)


def black_out_image(image):
    """
    Overlay solid black pixels over an image.

    Args:
        image: PIL Image to black out

    Returns:
        PIL Image with solid black pixels
    """
    # Create a black image of the same size
    img_array = np.array(image)
    black_array = np.zeros_like(img_array)

    # Convert back to PIL Image
    return Image.fromarray(black_array)


def redact_faces_in_image(
    input_image_path,
    output_image_path,
    face_image_size=512,
    redaction_method="blur",
    blur_strength=51,
    pixel_size=16,
    face_type=FaceType.WHOLE_FACE,
):
    """
    Extract faces from an image, redact them (blur, pixelate, or black out),
    and merge back into the original image.

    Args:
        input_image_path: Path to input image
        output_image_path: Path to save the output image
        face_image_size: Size of extracted face images
        redaction_method: Method to redact faces ("blur", "pixelate", or "black")
        blur_strength: Kernel size for Gaussian blur (must be odd number)
        pixel_size: Size of pixelation blocks for pixelate method
        face_type: Type of face extraction (default: WHOLE_FACE)
    """
    # Load the input image
    image = Image.open(input_image_path)

    # Initialize face alignment model
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )

    # Extract faces from the image
    print(f"Extracting faces from {input_image_path}...")
    face_images, image_to_face_matrices = extract_faces(
        fa, image, face_image_size, face_type
    )

    print(f"Found {len(face_images)} face(s)")

    # Redact each face image based on the selected method
    redacted_face_images = []
    for i, face_image in enumerate(face_images):
        print(f"Redacting face {i + 1}/{len(face_images)} using {redaction_method}...")

        if redaction_method == "blur":
            redacted_face = blur_image(face_image, blur_strength)
        elif redaction_method == "pixelate":
            redacted_face = pixelate_image(face_image, pixel_size)
        elif redaction_method == "black":
            redacted_face = black_out_image(face_image)
        else:
            raise ValueError(f"Unknown redaction method: {redaction_method}")

        redacted_face_images.append(redacted_face)

    # Merge redacted faces back into the original image
    result_image = image
    for i, (redacted_face, image_to_face_mat) in enumerate(
        zip(redacted_face_images, image_to_face_matrices)
    ):
        print(f"Merging redacted face {i + 1}/{len(redacted_face_images)}...")
        result_image = paste_foreground_onto_background(
            redacted_face, result_image, image_to_face_mat
        )

    # Save the result
    result_image.save(output_image_path)
    print(f"Saved result to {output_image_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract faces from an image, redact them (blur, pixelate, or black out), and merge back."
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
        help="Path to save output image",
    )
    parser.add_argument(
        "--face_image_size",
        type=int,
        default=512,
        help="Size of extracted face images (default: 512)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="blur",
        choices=["blur", "pixelate", "black"],
        help="Redaction method: blur (Gaussian blur), pixelate (pixelation effect), or black (solid black overlay) (default: blur)",
    )
    parser.add_argument(
        "--blur_strength",
        type=int,
        default=51,
        help="Gaussian blur kernel size, must be odd (default: 51). Only used with 'blur' method.",
    )
    parser.add_argument(
        "--pixel_size",
        type=int,
        default=16,
        help="Size of pixelation blocks (default: 16). Only used with 'pixelate' method.",
    )
    parser.add_argument(
        "--face_type",
        type=str,
        default="whole_face",
        choices=["half_face", "midfull_face", "full_face", "whole_face", "head"],
        help="Type of face extraction (default: whole_face)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Convert face_type string to FaceType enum
    face_type = FaceType.fromString(args.face_type)

    # Ensure blur_strength is odd (only matters for blur method)
    if args.method == "blur" and args.blur_strength % 2 == 0:
        args.blur_strength += 1
        print(f"Adjusted blur_strength to {args.blur_strength} (must be odd)")

    redact_faces_in_image(
        input_image_path=args.input_image,
        output_image_path=args.output_image,
        face_image_size=args.face_image_size,
        redaction_method=args.method,
        blur_strength=args.blur_strength,
        pixel_size=args.pixel_size,
        face_type=face_type,
    )
