"""Fetch-path hardening: connection pinning, address policy, bounded reads."""

from __future__ import annotations

import socket

import httpcore
import httpx
import pytest

from groundwork.config import Settings
from groundwork.providers.search import ContentFetcher
from groundwork.security.ssrf import UnsafeURLError, assert_url_is_safe, resolve_safe_address
from groundwork.security.transport import PinnedNetworkBackend, PinnedTransport

PUBLIC = "93.184.216.34"
IMDS = "169.254.169.254"


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    """Stands in for the socket layer and records where a connection was attempted."""

    def __init__(self) -> None:
        self.hosts: list[str] = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.hosts.append(host)
        raise httpcore.ConnectError("recording backend: no real sockets in tests")

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError("unused")

    async def sleep(self, seconds):
        return None


def _rebinding_dns(*answers: list[str]):
    """A DNS server that gives a different answer each time it is asked."""
    replies = iter(answers)
    return lambda host: next(replies)


async def test_connection_goes_to_the_address_that_was_validated() -> None:
    inner = _RecordingBackend()
    backend = PinnedNetworkBackend(resolver=lambda host: [PUBLIC], inner=inner)
    with pytest.raises(httpcore.ConnectError):
        await backend.connect_tcp("example.com", 443)
    assert inner.hosts == [PUBLIC], "must connect to the checked IP, not re-resolve the name"


async def test_dns_rebinding_between_check_and_connect_is_blocked() -> None:
    """The attack the URL check alone cannot stop: a public answer while the URL is
    checked, the metadata service when the connection is opened."""
    dns = _rebinding_dns([PUBLIC], [IMDS])
    assert_url_is_safe("https://rebind.example/", resolver=dns)  # first answer looks fine

    inner = _RecordingBackend()
    backend = PinnedNetworkBackend(resolver=dns, inner=inner)
    with pytest.raises(UnsafeURLError):
        await backend.connect_tcp("rebind.example", 443)
    assert inner.hosts == [], "nothing may be contacted once the answer turns unsafe"


async def test_mixed_public_and_private_answers_are_refused_at_connect() -> None:
    inner = _RecordingBackend()
    backend = PinnedNetworkBackend(resolver=lambda host: [PUBLIC, "10.0.0.5"], inner=inner)
    with pytest.raises(UnsafeURLError):
        await backend.connect_tcp("mixed.example", 80)
    assert inner.hosts == []


@pytest.mark.parametrize("address", ["100.64.0.1", "100.100.100.200", "198.18.0.1", "0.0.0.0"])
def test_non_global_addresses_are_refused(address: str) -> None:
    """Shared address space (carrier-grade NAT, Tailscale) is not the public internet."""
    with pytest.raises(UnsafeURLError):
        resolve_safe_address("internal.example", resolver=lambda host: [address])


def test_public_address_is_allowed() -> None:
    assert resolve_safe_address("example.com", resolver=lambda host: [PUBLIC]) == PUBLIC


def test_the_fetcher_transport_is_actually_pinned() -> None:
    """Guards the wiring, not only the backend. If an httpx upgrade moved the
    connection pool, PinnedTransport refuses to build rather than fetch
    unpinned, and this test fails loudly."""
    transport = PinnedTransport()
    assert isinstance(transport._pool._network_backend, PinnedNetworkBackend)


async def test_response_body_is_read_only_up_to_the_cap(monkeypatch) -> None:
    """`res.text` used to read the whole body into memory before slicing it."""
    cap = 256 * 1024
    chunk = b"x" * (1024 * 1024)
    pulled = 0

    async def enormous_page():
        nonlocal pulled
        for _ in range(64):
            pulled += len(chunk)
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=enormous_page())

    monkeypatch.setattr(socket, "getaddrinfo", lambda host, *a, **k: [(2, 1, 6, "", (PUBLIC, 0))])
    monkeypatch.setattr("groundwork.providers.search.MAX_RESPONSE_BYTES", cap)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        fetcher = ContentFetcher(Settings(_env_file=None), client=client)
        result = await fetcher.fetch("https://example.com/huge")

    assert result is not None
    assert len(result.text) <= cap
    assert pulled <= cap + len(chunk), f"read {pulled} bytes for a {cap}-byte cap"
