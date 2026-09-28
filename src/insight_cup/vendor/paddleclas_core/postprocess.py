"""
Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Focused from PaddleClas deploy/python/postprocess.py. Only label parsing and
the classification Topk postprocessor are retained.
"""

from __future__ import annotations

import os

import numpy as np


def parse_class_id_map(class_id_map_file, delimiter):
    if class_id_map_file is None:
        return None
    if not os.path.exists(class_id_map_file):
        raise FileNotFoundError(f"Class label map not found: {class_id_map_file}")
    class_id_map = {}
    with open(class_id_map_file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            partition = line.rstrip("\n").partition(delimiter)
            class_id_map[int(partition[0])] = str(partition[-1])
    return class_id_map


class Topk:
    def __init__(
        self,
        topk=1,
        class_id_map_file=None,
        delimiter=None,
        label_list=None,
    ):
        assert isinstance(topk, int)
        self.topk = topk
        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = (
            parse_class_id_map(class_id_map_file, delimiter)
            if not label_list
            else label_list
        )

    def __call__(self, values, file_names=None):
        if file_names is not None:
            assert values.shape[0] == len(file_names)
        results = []
        for index, probabilities in enumerate(values):
            ranked = probabilities.argsort(axis=0)[-self.topk :][::-1].astype("int32")
            class_ids = []
            scores = []
            label_names = []
            for class_id in ranked:
                class_ids.append(class_id.item())
                scores.append(probabilities[class_id].item())
                if self.class_id_map is not None:
                    label_names.append(self.class_id_map[class_id.item()])
            result = {
                "class_ids": class_ids,
                "scores": np.around(scores, decimals=5).tolist(),
            }
            if file_names is not None:
                result["file_name"] = file_names[index]
            if self.class_id_map is not None:
                result["label_names"] = label_names
            results.append(result)
        return results
