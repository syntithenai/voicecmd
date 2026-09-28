"""Tiny stdlib HTTP helpers (JSON + multipart) shared by service clients."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:200]}")
        self.status = status
        self.body = body


def request(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> bytes:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise HttpError(exc.code, detail, url) from None


def get_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 10.0) -> Any:
    return json.loads(request("GET", url, headers=headers, timeout=timeout) or b"null")


def post_json(
    url: str,
    payload: Any,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    method: str = "POST",
) -> Any:
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    raw = request(method, url, body=json.dumps(payload).encode("utf-8"), headers=hdrs, timeout=timeout)
    return json.loads(raw) if raw else None


def post_multipart(
    url: str,
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes, str]],
    *,
    timeout: float = 30.0,
) -> bytes:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, data, ctype) in files.items():
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return request(
        "POST",
        url,
        body=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=timeout,
    )
