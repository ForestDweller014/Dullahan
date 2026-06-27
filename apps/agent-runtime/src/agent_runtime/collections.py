from __future__ import annotations

from threading import RLock
from typing import Generic, TypeVar


T = TypeVar("T")


class ThreadSafeList(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
        self._lock = RLock()

    def append(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def snapshot(self) -> list[T]:
        with self._lock:
            return list(self._items)
