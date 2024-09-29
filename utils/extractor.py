import argparse
import math
import os
from enum import IntEnum
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import numpy.linalg as npla
from PIL import Image

from diffusers.utils import load_image

landmarks_2D_new = np.array(
    [
        [0.000213256, 0.106454],  # 17
        [0.0752622, 0.038915],  # 18
        [0.18113, 0.0187482],  # 19
        [0.29077, 0.0344891],  # 20
        [0.393397, 0.0773906],  # 21
        [0.586856, 0.0773906],  # 22
        [0.689483, 0.0344891],  # 23
        [0.799124, 0.0187482],  # 24
        [0.904991, 0.038915],  # 25
        [0.98004, 0.106454],  # 26
        [0.490127, 0.203352],  # 27
        [0.490127, 0.307009],  # 28
        [0.490127, 0.409805],  # 29
        [0.490127, 0.515625],  # 30
        [0.36688, 0.587326],  # 31
        [0.426036, 0.609345],  # 32
        [0.490127, 0.628106],  # 33
        [0.554217, 0.609345],  # 34
        [0.613373, 0.587326],  # 35
        [0.121737, 0.216423],  # 36
        [0.187122, 0.178758],  # 37
        [0.265825, 0.179852],  # 38
        [0.334606, 0.231733],  # 39
        [0.260918, 0.245099],  # 40
        [0.182743, 0.244077],  # 41
        [0.645647, 0.231733],  # 42
        [0.714428, 0.179852],  # 43
        [0.793132, 0.178758],  # 44
        [0.858516, 0.216423],  # 45
        [0.79751, 0.244077],  # 46
        [0.719335, 0.245099],  # 47
        [0.254149, 0.780233],  # 48
        [0.726104, 0.780233],  # 54
    ],
    dtype=np.float32,
)


class FaceType(IntEnum):
    # enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = (100,)  # no align at all, just embedded faceinfo

    @staticmethod
    def fromString(s):
        r = from_string_dict.get(s.lower())
        if r is None:
            raise Exception("FaceType.fromString value error")
        return r

    @staticmethod
    def toString(face_type):
        return to_string_dict[face_type]


to_string_dict = {
    FaceType.HALF: "half_face",
    FaceType.MID_FULL: "midfull_face",
    FaceType.FULL: "full_face",
    FaceType.FULL_NO_ALIGN: "full_face_no_align",
    FaceType.WHOLE_FACE: "whole_face",
    FaceType.HEAD: "head",
    FaceType.HEAD_NO_ALIGN: "head_no_align",
    FaceType.MARK_ONLY: "mark_only",
}

from_string_dict = {to_string_dict[x]: x for x in to_string_dict.keys()}
FaceType_to_padding_remove_align = {
    FaceType.HALF: (0.0, False),
    FaceType.MID_FULL: (0.0675, False),
    FaceType.FULL: (0.2109375, False),
    FaceType.FULL_NO_ALIGN: (0.2109375, True),
    FaceType.WHOLE_FACE: (0.40, False),
    FaceType.HEAD: (0.70, False),
    FaceType.HEAD_NO_ALIGN: (0.70, True),
}


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def transform_points(points, mat, invert=False):
    if invert:
        mat = cv2.invertAffineTransform(mat)
    points = np.expand_dims(points, axis=1)
    points = cv2.transform(points, mat, points.shape)
    points = np.squeeze(points)
    return points


def estimate_averaged_yaw(landmarks):
    # Works much better than solvePnP if landmarks from "3DFAN"
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    l = (
        (landmarks[27][0] - landmarks[0][0])
        + (landmarks[28][0] - landmarks[1][0])
        + (landmarks[29][0] - landmarks[2][0])
    ) / 3.0
    r = (
        (landmarks[16][0] - landmarks[27][0])
        + (landmarks[15][0] - landmarks[28][0])
        + (landmarks[14][0] - landmarks[29][0])
    ) / 3.0
    return float(r - l)


def polygon_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def get_transform_mat(image_landmarks, output_size, face_type, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array(image_landmarks)

    # estimate landmarks transform from global space to local aligned space with bounds [0..1]
    mat = umeyama(
        np.concatenate([image_landmarks[17:49], image_landmarks[54:55]]),
        landmarks_2D_new,
        True,
    )[0:2]

    # get corner points in global space
    g_p = transform_points(
        np.float32([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]), mat, True
    )
    g_c = g_p[4]

    # calc diagonal vectors between corners in global space
    tb_diag_vec = (g_p[2] - g_p[0]).astype(np.float32)
    tb_diag_vec /= npla.norm(tb_diag_vec)
    bt_diag_vec = (g_p[1] - g_p[3]).astype(np.float32)
    bt_diag_vec /= npla.norm(bt_diag_vec)

    # calc modifier of diagonal vectors for scale and padding value
    padding, remove_align = FaceType_to_padding_remove_align.get(face_type, 0.0)
    mod = (1.0 / scale) * (npla.norm(g_p[0] - g_p[2]) * (padding * np.sqrt(2.0) + 0.5))

    if face_type == FaceType.WHOLE_FACE:
        # adjust vertical offset for WHOLE_FACE, 7% below in order to cover more forehead
        vec = (g_p[0] - g_p[3]).astype(np.float32)
        vec_len = npla.norm(vec)
        vec /= vec_len
        g_c += vec * vec_len * 0.07

    elif face_type == FaceType.HEAD:
        # assuming image_landmarks are 3D_Landmarks extracted for HEAD,
        # adjust horizontal offset according to estimated yaw
        yaw = estimate_averaged_yaw(transform_points(image_landmarks, mat, False))

        hvec = (g_p[0] - g_p[1]).astype(np.float32)
        hvec_len = npla.norm(hvec)
        hvec /= hvec_len

        yaw *= np.abs(math.tanh(yaw * 2))  # Damp near zero

        g_c -= hvec * (yaw * hvec_len / 2.0)

        # adjust vertical offset for HEAD, 50% below
        vvec = (g_p[0] - g_p[3]).astype(np.float32)
        vvec_len = npla.norm(vvec)
        vvec /= vvec_len
        g_c += vvec * vvec_len * 0.50

    # calc 3 points in global space to estimate 2d affine transform
    if not remove_align:
        l_t = np.array(
            [g_c - tb_diag_vec * mod, g_c + bt_diag_vec * mod, g_c + tb_diag_vec * mod]
        )
    else:
        # remove_align - face will be centered in the frame but not aligned
        l_t = np.array(
            [
                g_c - tb_diag_vec * mod,
                g_c + bt_diag_vec * mod,
                g_c + tb_diag_vec * mod,
                g_c - bt_diag_vec * mod,
            ]
        )

        # get area of face square in global space
        area = polygon_area(l_t[:, 0], l_t[:, 1])

        # calc side of square
        side = np.float32(math.sqrt(area) / 2)

        # calc 3 points with unrotated square
        l_t = np.array([g_c + [-side, -side], g_c + [side, -side], g_c + [side, side]])

    # calc affine transform from 3 global space points to 3 local space points size of 'output_size'
    pts2 = np.float32(((0, 0), (output_size, 0), (output_size, output_size)))
    mat = cv2.getAffineTransform(l_t, pts2)
    return mat


def extract_faces(model, image, face_image_size, face_type=FaceType.WHOLE_FACE):
    array = np.array(image)
    preds = model.get_landmarks(array)

    face_images = []
    image_to_face_matrices = []
    for face_landmarks in preds:
        image_to_face_mat = get_transform_mat(
            face_landmarks, face_image_size, face_type
        )

        face_array = cv2.warpAffine(
            array,
            image_to_face_mat,
            (face_image_size, face_image_size),
            cv2.INTER_LANCZOS4,
        )

        image_to_face_matrices.append(image_to_face_mat)
        face_images.append(Image.fromarray(face_array))

    return face_images, image_to_face_matrices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="Path to input image",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Output image size",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./face_images",
        help="Path to output folder",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # sfd for SFD, dlib for Dlib and folder for existing bounding boxes.
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )
    face_type = FaceType.WHOLE_FACE
    pil_image = load_image(args.image_path)
    face_images, image_to_face_matrices = extract_faces(
        fa, pil_image, args.image_size, face_type
    )

    # Make sure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    for i, face_image in enumerate(face_images):
        face_image.save(Path(args.output_folder, f"face_{i:02}.jpg"))
