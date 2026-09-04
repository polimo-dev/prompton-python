"""The HTTP layer: stdlib ``urllib`` only, and a seam for tests to replace it.

Nothing here retries or backs off. Retry policy belongs to the snapshot store and the log buffer,
which know what "keep serving the previous document" and "resend the same ids" mean.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ._version import USER_AGENT
from .errors import APIError, TransportError

__all__ = ["HttpResponse", "Transport", "UrllibTransport", "parse_api_error", "retry_after_seconds"]


@dataclass(frozen=True)
class HttpResponse:
    """One HTTP answer. ``headers`` keys are lowercased."""

    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Any:
        """The body decoded as JSON, or ``None`` when it is empty or not JSON."""
        if not self.body:
            return None
        try:
            return json.loads(self.body)
        except (ValueError, UnicodeDecodeError):
            return None

    @property
    def etag(self) -> str | None:
        return self.headers.get("etag")

    @property
    def last_modified(self) -> str | None:
        return self.headers.get("last-modified")


class Transport(Protocol):
    """Replace this to test without a network, or to route through your own HTTP client."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
        timeout: float = 5.0,
    ) -> HttpResponse: ...


class UrllibTransport:
    """The default transport. No third-party dependency, no connection pooling, no retries."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
        timeout: float = 5.0,
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, method=method)
        for name, value in headers.items():
            request.add_header(name, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return HttpResponse(
                    status=response.status,
                    body=response.read(),
                    headers=_lowercase(response.headers.items()),
                )
        except urllib.error.HTTPError as error:
            # An HTTPError *is* the response: 304, 404 and 429 all arrive here.
            payload = b""
            try:
                payload = error.read()
            except Exception:  # noqa: BLE001 - a body we cannot read is not an error
                payload = b""
            return HttpResponse(
                status=error.code, body=payload, headers=_lowercase(error.headers.items())
            )
        except urllib.error.URLError as error:
            raise TransportError(f"could not reach PromptOn at {url}: {error.reason}") from error
        except OSError as error:
            raise TransportError(f"could not reach PromptOn at {url}: {error}") from error


def _lowercase(items: Any) -> dict[str, str]:
    return {str(name).lower(): str(value) for name, value in items}


def build_headers(
    api_key: str | None, user_agent: str = USER_AGENT, **extra: str
) -> dict[str, str]:
    headers = {"accept": "application/json", "user-agent": user_agent}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    headers.update({name: value for name, value in extra.items() if value is not None})
    return headers


def parse_api_error(response: HttpResponse) -> APIError:
    """Turn a non-2xx PromptOn response into a typed error carrying ``code`` and ``details``."""
    body = response.json()
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return APIError(
            status=response.status,
            code=error.get("code"),
            message=error.get("message"),
            details=error.get("details") if isinstance(error.get("details"), dict) else {},
        )
    text = response.body.decode("utf-8", "replace")[:200] if response.body else ""
    return APIError(status=response.status, message=text or None)


def retry_after_seconds(response: HttpResponse) -> float | None:
    """``Retry-After`` in seconds, from the header or from ``error.details.retry_after``."""
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return max(float(raw.strip()), 0.0)
        except ValueError:
            parsed = _http_date_seconds(raw.strip())
            if parsed is not None:
                return parsed
    body = response.json()
    if isinstance(body, dict):
        error = body.get("error")
        details = error.get("details") if isinstance(error, dict) else None
        if isinstance(details, dict):
            value = details.get("retry_after")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(float(value), 0.0)
    return None


def _http_date_seconds(value: str) -> float | None:
    from email.utils import parsedate_to_datetime

    try:
        moment = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if moment is None:
        return None
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max((moment - now).total_seconds(), 0.0)


def urlencode(params: Mapping[str, Any]) -> str:
    """Query string for the parameters that are not ``None``."""
    return urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
