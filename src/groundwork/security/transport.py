"""An httpx transport that only connects to addresses the SSRF policy approved.

`assert_url_is_safe` checks a URL before the request, but httpx resolves the
hostname again when it opens the connection. A DNS server that answers with a
public address the first time and 169.254.169.254 the second (DNS rebinding)
would pass the check and still steer the connection inside the network.

The fix is to make the check and the connection share one resolution. httpx
hands connection setup to an httpcore network backend; `PinnedNetworkBackend`
resolves the name itself, validates every address, and connects to a validated
IP. TLS is unaffected: httpcore still sends the original hostname for SNI and
certificate verification, because only the TCP target changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import anyio
import httpcore
import httpx

from groundwork.security.ssrf import (
    Resolver,
    UnsafeURLError,
    resolve_addresses,
    resolve_safe_address,
)


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve, validate and connect in one step."""

    def __init__(
        self,
        *,
        resolver: Resolver = resolve_addresses,
        inner: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self._resolver = resolver
        self._inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        # getaddrinfo blocks, so keep it off the event loop.
        address = await anyio.to_thread.run_sync(
            lambda: resolve_safe_address(host, resolver=self._resolver)
        )
        return await self._inner.connect_tcp(
            address,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise UnsafeURLError("Unix sockets are never a valid fetch target")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class PinnedTransport(httpx.AsyncHTTPTransport):
    """`httpx.AsyncHTTPTransport` whose connections go through the pinned backend.

    httpx does not expose the network backend publicly, so this sets it on the
    connection pool it builds. If an httpx release moves that pool, construction
    fails instead of quietly falling back to unpinned connections, and
    `test_the_fetcher_transport_is_actually_pinned` says so.
    """

    def __init__(self, *, resolver: Resolver = resolve_addresses, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        pool = getattr(self, "_pool", None)
        if pool is None or not hasattr(pool, "_network_backend"):
            raise RuntimeError(
                "httpx internals changed: cannot pin fetch connections, refusing "
                "to build a transport that would connect unpinned"
            )
        pool._network_backend = PinnedNetworkBackend(resolver=resolver)
