from collections.abc import Callable, Iterator
from typing import Protocol

from prooflens.data.schema import ManifestRecord


class DatasetAdapter(Protocol):
    scan: Callable[[], Iterator[ManifestRecord]]
