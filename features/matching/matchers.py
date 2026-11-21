"""MEMBER 4 — Feature matching with SSD and NCC.

Both matchers follow the same recipe (Template Method in `FeatureMatcher`):
  1. score matrix  S[i, j] between every descriptor i of image A and j of image B
  2. best match    for each i (lowest SSD / highest NCC)
  3. acceptance    test that rejects ambiguous / weak matches
  4. cross-check   (optional) keep i<->j only if they choose each other

Fully vectorised: the whole N x M score matrix is one matrix product.

SSD(a, b) = Σ (a_k - b_k)^2 = |a|^2 + |b|^2 - 2 a·b          (lower = better)
NCC(a, b) = Σ (a_k - ā)(b_k - b̄) / (‖a - ā‖ ‖b - b̄‖)          (higher = better, in [-1, 1])
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..core import Timer, param


@dataclass(frozen=True)
class MatchParams:
    cross_check: bool = param(True, "Cross-check (mutual best)",
                              tip="Keep a pair only if A→B and B→A agree")


@dataclass(frozen=True)
class SSDParams(MatchParams):
    ratio: float = param(0.8, "Lowe ratio", 0.3, 1.0, 0.05,
                         "Accept if best distance < ratio × second-best distance")


@dataclass(frozen=True)
class NCCParams(MatchParams):
    min_ncc: float = param(0.90, "Min NCC", 0.0, 1.0, 0.01,
                           "Accept if correlation ≥ this value")


@dataclass
class MatchResult:
    method: str
    query_idx: np.ndarray     # indices into image A keypoints
    train_idx: np.ndarray     # indices into image B keypoints
    scores: np.ndarray        # SSD or NCC value of each accepted pair
    n_query: int
    n_train: int
    elapsed_ms: float

    def __len__(self) -> int:
        return len(self.query_idx)

    def best(self, k: int) -> "MatchResult":
        """The k most confident matches (for drawing)."""
        order = np.argsort(self.scores)
        if self._higher_is_better:
            order = order[::-1]
        order = order[:k]
        return MatchResult(self.method, self.query_idx[order], self.train_idx[order],
                           self.scores[order], self.n_query, self.n_train, self.elapsed_ms)

    @property
    def _higher_is_better(self) -> bool:
        return MATCHERS[self.method].higher_is_better


class FeatureMatcher(ABC):
    name: ClassVar[str]
    higher_is_better: ClassVar[bool]
    params_cls: ClassVar[type[MatchParams]]

    def __init__(self, params: MatchParams | None = None) -> None:
        self.params = params or self.params_cls()

    def match(self, desc_a: np.ndarray, desc_b: np.ndarray) -> MatchResult:
        n, m = len(desc_a), len(desc_b)
        if n == 0 or m == 0:
            empty = np.zeros(0, np.int64)
            return MatchResult(self.name, empty, empty, np.zeros(0, np.float32), n, m, 0.0)

        with Timer() as t:
            scores = self.score_matrix(desc_a.astype(np.float32), desc_b.astype(np.float32))
            pick = np.argmax if self.higher_is_better else np.argmin
            best_j = pick(scores, axis=1)                          # step 2
            rows = np.arange(n)
            keep = self.accept(scores, best_j)                     # step 3
            if self.params.cross_check:                            # step 4
                keep &= pick(scores, axis=0)[best_j] == rows
            q, tr = rows[keep], best_j[keep]
            s = scores[q, tr]

        return MatchResult(self.name, q, tr, s.astype(np.float32), n, m, t.ms)

    @abstractmethod
    def score_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def accept(self, scores: np.ndarray, best_j: np.ndarray) -> np.ndarray: ...


class SSDMatcher(FeatureMatcher):
    name = "SSD"
    higher_is_better = False
    params_cls = SSDParams

    def score_matrix(self, a, b):
        sq_a = np.sum(a * a, axis=1)[:, None]
        sq_b = np.sum(b * b, axis=1)[None, :]
        return np.maximum(sq_a + sq_b - 2.0 * (a @ b.T), 0.0)

    def accept(self, scores, best_j):
        if scores.shape[1] < 2:
            return np.ones(len(scores), bool)
        two = np.partition(scores, 1, axis=1)[:, :2]               # best & second best
        # ratio is defined on Euclidean distance, SSD = distance^2  -> square it
        return two[:, 0] < (self.params.ratio ** 2) * two[:, 1]


class NCCMatcher(FeatureMatcher):
    name = "NCC"
    higher_is_better = True
    params_cls = NCCParams

    @staticmethod
    def _standardise(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def score_matrix(self, a, b):
        return self._standardise(a) @ self._standardise(b).T

    def accept(self, scores, best_j):
        return scores[np.arange(len(scores)), best_j] >= self.params.min_ncc


MATCHERS: dict[str, type[FeatureMatcher]] = {
    SSDMatcher.name: SSDMatcher,
    NCCMatcher.name: NCCMatcher,
}
