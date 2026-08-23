"""SSRF guard for the URL fetcher.

An agent that fetches attacker-suggested URLs is a Server-Side Request Forgery
primitive. On a cloud host, `http://169.254.169.254/` is the metadata endpoint
and can leak IAM credentials. This module is the allow/deny decision point.

Note the ordering problem this solves: checking the hostname is not enough,
because DNS can resolve a public-looking name to 127.0.0.1 (DNS rebinding).
We therefore resolve the host and check every returned address.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Hosts we refuse regardless of resolution.
BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
    }
)

# Cloud metadata addresses.
BLOCKED_ADDRESSES = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure IMDS
        ipaddress.ip_address("100.100.100.200"),  # Alibaba
    }
)

MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 5_000_000  # 5 MB, guards against decompression bombs


class UnsafeURLError(ValueError):
    """Raised when a URL fails the SSRF policy."""


def _is_disallowed_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject anything that is not a normal public unicast address."""
    if ip in BLOCKED_ADDRESSES:
        return True
    return any(
        (
            ip.is_private,
            ip.is_loopback,
            ip.is_link_local,
            ip.is_multicast,
            ip.is_reserved,
            ip.is_unspecified,
        )
    )


def resolve_addresses(hostname: str) -> list[str]:
    """Resolve a hostname to all of its addresses.

    Split out so tests can monkeypatch DNS without network access.

    `getaddrinfo` returns the address as the first element of the sockaddr
    tuple, which is typed as `str | int` because AF_PACKET and friends differ.
    We coerce to str explicitly rather than trusting the family.
    """
    infos = socket.getaddrinfo(hostname, None)
    return sorted({str(info[4][0]) for info in infos})


def assert_url_is_safe(url: str, *, resolver=resolve_addresses) -> None:
    """Raise `UnsafeURLError` unless `url` is safe to fetch.

    `resolver` is injected so the test-suite can simulate DNS rebinding.
    """
    parsed = urlparse(url)

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Scheme not allowed: {parsed.scheme!r}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise UnsafeURLError("URL has no hostname")

    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".localhost"):
        raise UnsafeURLError(f"Hostname not allowed: {hostname!r}")

    # Reject credentials in the URL (http://user:pass@host) - usually an attempt
    # to confuse naive host parsing.
    if parsed.username or parsed.password:
        raise UnsafeURLError("Credentials in URL are not allowed")

    # A literal IP in the URL still has to pass the address policy.
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None

    if literal is not None:
        if _is_disallowed_ip(literal):
            raise UnsafeURLError(f"IP address not allowed: {hostname}")
        return

    try:
        addresses = resolver(hostname)
    except OSError as exc:
        raise UnsafeURLError(f"Could not resolve host {hostname!r}: {exc}") from exc

    if not addresses:
        raise UnsafeURLError(f"Host {hostname!r} resolved to no addresses")

    for addr in addresses:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            raise UnsafeURLError(f"Unparseable address {addr!r}") from None
        if _is_disallowed_ip(ip):
            raise UnsafeURLError(
                f"Host {hostname!r} resolves to disallowed address {addr}"
            )


def is_url_safe(url: str, *, resolver=resolve_addresses) -> bool:
    """Boolean convenience wrapper."""
    try:
        assert_url_is_safe(url, resolver=resolver)
    except UnsafeURLError:
        return False
    return True
