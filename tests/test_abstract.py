"""Interface-drift guard: an incomplete BaseHandModel subclass must fail instantiation."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.interface import BaseHandModel  # noqa: E402


class MissingInfer(BaseHandModel):
    name = "broken"

    def load(self, device: str = "cuda") -> None:
        pass

    def preprocess(self, image_bgr):
        return image_bgr

    @property
    def device(self) -> str:
        return "cpu"


def test_abc_rejects_incomplete_subclass():
    with pytest.raises(TypeError):
        MissingInfer()


def test_abc_is_abstract():
    assert BaseHandModel.__abstractmethods__  # must still have abstract members