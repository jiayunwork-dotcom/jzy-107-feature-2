"""通道档存取：把常用通道几何与工质参数命名保存，进程内可靠存取。

只在运行期间有效，不跨重启保留。通道档是不可变的 ChannelParams，
多个请求并发读取同一份参数不会被彼此污染。
"""
from __future__ import annotations

import threading

from .induction import ChannelParams


class ProfileNotFoundError(KeyError):
    """按名称取通道档时未找到。"""


class ProfileStore:
    """线程安全的进程内通道档仓库。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, ChannelParams] = {}

    def save(self, name: str, params: ChannelParams) -> None:
        with self._lock:
            self._profiles[name] = params

    def get(self, name: str) -> ChannelParams:
        with self._lock:
            try:
                return self._profiles[name]
            except KeyError:
                raise ProfileNotFoundError(name) from None

    def delete(self, name: str) -> None:
        with self._lock:
            try:
                del self._profiles[name]
            except KeyError:
                raise ProfileNotFoundError(name) from None

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._profiles)
