"""
Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Focused from PaddleClas deploy/python/preprocess.py. Only the classification
operators used by the knife runtime are retained.
"""

from __future__ import annotations

import importlib
import random
from functools import partial

import cv2
import numpy as np
from PIL import Image


def create_operators(params):
    """Build PaddleClas preprocessing operators from a config list."""

    assert isinstance(params, list), "operator config should be a list"
    module = importlib.import_module(__name__)
    operators = []
    for operator in params:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        operator_name = list(operator)[0]
        values = {} if operator[operator_name] is None else operator[operator_name]
        operators.append(getattr(module, operator_name)(**values))
    return operators


class UnifiedResize:
    def __init__(self, interpolation=None, backend="cv2", return_numpy=True):
        cv2_interpolation = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "random": (cv2.INTER_LINEAR, cv2.INTER_CUBIC),
        }
        pil_interpolation = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
            "random": (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC),
        }

        def cv2_resize(source, size, resample):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            return cv2.resize(source, size, interpolation=resample)

        def pil_resize(source, size, resample, return_numpy=True):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            image = Image.fromarray(source) if isinstance(source, np.ndarray) else source
            image = image.resize(size, resample)
            return np.asarray(image) if return_numpy else image

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = cv2_interpolation[interpolation.lower()]
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(cv2_resize, resample=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = pil_interpolation[interpolation.lower()]
            elif interpolation is None:
                interpolation = Image.Resampling.BILINEAR
            self.resize_func = partial(
                pil_resize,
                resample=interpolation,
                return_numpy=return_numpy,
            )
        else:
            raise ValueError(f"Resize backend must be 'cv2' or 'pil', got {backend!r}")

    def __call__(self, source, size):
        if isinstance(size, list):
            size = tuple(size)
        return self.resize_func(source, size)


class OperatorParamError(ValueError):
    pass


class ResizeImage:
    """PaddleClas image resize operator."""

    def __init__(
        self,
        size=None,
        resize_short=None,
        interpolation=None,
        backend="cv2",
        return_numpy=True,
    ):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError(
                "size and resize_short cannot both be None"
            )
        self._resize_func = UnifiedResize(
            interpolation=interpolation,
            backend=backend,
            return_numpy=return_numpy,
        )

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image_height, image_width = image.shape[:2]
        else:
            image_width, image_height = image.size
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(image_width, image_height)
            width = int(round(image_width * percent))
            height = int(round(image_height * percent))
        else:
            width = self.w
            height = self.h
        return self._resize_func(image, (width, height))


class NormalizeImage:
    """PaddleClas image normalization operator."""

    def __init__(
        self,
        scale=None,
        mean=None,
        std=None,
        order="chw",
        output_fp16=False,
        channel_num=3,
    ):
        if isinstance(scale, str):
            scale = eval(scale, {"__builtins__": {}}, {})
        assert channel_num in [3, 4]
        self.channel_num = channel_num
        self.output_dtype = "float16" if output_fp16 else "float32"
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]
        shape = (3, 1, 1) if self.order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        assert isinstance(image, np.ndarray), "invalid input 'img' in NormalizeImage"
        image = (image.astype("float32") * self.scale - self.mean) / self.std
        if self.channel_num == 4:
            image_height = image.shape[1] if self.order == "chw" else image.shape[0]
            image_width = image.shape[2] if self.order == "chw" else image.shape[1]
            pad_zeros = (
                np.zeros((1, image_height, image_width))
                if self.order == "chw"
                else np.zeros((image_height, image_width, 1))
            )
            axis = 0 if self.order == "chw" else 2
            image = np.concatenate((image, pad_zeros), axis=axis)
        return image.astype(self.output_dtype)


class ToCHWImage:
    """Convert HWC image data to CHW."""

    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image.transpose((2, 0, 1))
