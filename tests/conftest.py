from pathlib import Path

import pytest

from vramfit import gguf
from vramfit import shape as shape_mod

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    _, meta, tensors = gguf.read_header(FIXTURES / name)
    return shape_mod.from_gguf(meta, tensors, name)


@pytest.fixture
def llama8b():
    return load("llama-3.1-8b-instruct-q4_k_m.header.gguf")


@pytest.fixture
def qwen7b():
    return load("qwen2.5-7b-instruct-q8_0.header.gguf")


@pytest.fixture
def mqa():
    return load("tiny-mqa-f16.header.gguf")
