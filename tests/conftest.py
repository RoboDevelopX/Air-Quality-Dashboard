import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


class FakeClock:
    def __init__(self, start: float = 1_736_942_400.0) -> None:  # 2025-01-15T12:00Z
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fixture() -> Callable[[str], Any]:
    return load_fixture
