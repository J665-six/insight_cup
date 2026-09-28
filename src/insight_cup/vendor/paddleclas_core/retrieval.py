# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Focused PP-ShiTu feature normalization and flat inner-product search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Apply the L2 normalization used by PaddleClas RecPredictor."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    norms = np.sqrt(np.sum(np.square(values), axis=1, keepdims=True))
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("features contain a non-finite or zero-length vector")
    return np.ascontiguousarray(np.divide(values, norms), dtype=np.float32)


@dataclass(frozen=True)
class SearchResult:
    scores: np.ndarray
    indices: np.ndarray


class FlatInnerProductIndex:
    """Faiss IndexFlatIP wrapper matching PP-ShiTu gallery semantics."""

    def __init__(self, features: np.ndarray) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu is required for PaddleClas feature retrieval"
            ) from exc

        vectors = normalize_features(features)
        self.embedding_size = int(vectors.shape[1])
        self.count = int(vectors.shape[0])
        if self.count == 0:
            raise ValueError("feature index cannot be empty")
        self._index = faiss.IndexFlatIP(self.embedding_size)
        self._index.add(vectors)

    def search(self, features: np.ndarray, topk: int) -> SearchResult:
        queries = normalize_features(features)
        if queries.shape[1] != self.embedding_size:
            raise ValueError("query and gallery embedding dimensions do not match")
        count = min(max(1, int(topk)), self.count)
        scores, indices = self._index.search(queries, count)
        return SearchResult(scores=scores, indices=indices)
