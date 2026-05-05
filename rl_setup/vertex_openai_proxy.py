from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


TOKEN_TTL_SECONDS = 45 * 60
FORMAT_REMINDER = """\
Return exactly one mini-SWE-agent action.

Your response must contain:
THOUGHT: <brief reasoning>

```mswea_bash_command
<exactly one bash command>
```

Do not answer in plain prose without the fenced command block.
"""


class TokenCache:
    def __init__(self) -> None:
        self._token: str | None = None
        self._expires_at = 0.0

    def get(self) -> str:
        now = time.time()
        if self._token and now < self._expires_at:
            return self._token
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            text=True,
            capture_output=True,
        )
        self._token = completed.stdout.strip()
        self._expires_at = now + TOKEN_TTL_SECONDS
        return self._token

    def clear(self) -> None:
        self._token = None
        self._expires_at = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Expose a Vertex predict endpoint as an OpenAI chat-completions endpoint.",
    )
    parser.add_argument("--project", required=True, help="Google Cloud project id or number.")
    parser.add_argument("--location", required=True, help="Vertex AI region, e.g. us-central1.")
    parser.add_argument("--endpoint-id", required=True, help="Vertex endpoint id.")
    parser.add_argument("--dedicated-host", required=True, help="Dedicated endpoint DNS host.")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--debug", action="store_true", help="Print prompt/response previews.")
    args = parser.parse_args()

    vertex_url = (
        f"https://{args.dedicated_host}/v1/projects/{args.project}"
        f"/locations/{args.location}/endpoints/{args.endpoint_id}:predict"
    )
    token_cache = TokenCache()

    class Handler(BaseHTTPRequestHandler):
        server_version = "VertexOpenAIProxy/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [{"id": "qwen-vertex", "object": "model"}],
                    }
                )
                return
            self._send_json({"error": {"message": "not found"}}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send_json({"error": {"message": "not found"}}, status=404)
                return

            try:
                request = self._read_json()
                prompt = render_prompt(request.get("messages") or [])
                vertex_payload = {
                    "instances": [{"prompt": prompt}],
                    "parameters": {
                        "temperature": request.get("temperature", 0),
                        "maxOutputTokens": request.get(
                            "max_tokens",
                            request.get("max_completion_tokens", 2048),
                        ),
                    },
                }
                if args.debug:
                    print(f"\n--- prompt preview ---\n{prompt[-2500:]}\n--- end prompt preview ---")
                text = call_vertex(vertex_url, token_cache, vertex_payload)
                text = normalize_response(text)
                if args.debug:
                    print(f"\n--- response preview ---\n{text[:2500]}\n--- end response preview ---")
                response = {
                    "id": f"chatcmpl-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.get("model", "qwen-vertex"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
                self._send_json(response)
            except Exception as exc:
                self._send_json({"error": {"message": str(exc)}}, status=500)

        def log_message(self, fmt: str, *args: object) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8")) if raw else {}

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer((args.listen_host, args.port), Handler)
    print(f"OpenAI proxy listening on http://{args.listen_host}:{args.port}/v1")
    print(f"Forwarding to {vertex_url}")
    server.serve_forever()
    return 0


def render_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).lower()
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append(f"<|im_start|>user\n{FORMAT_REMINDER}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def call_vertex(url: str, token_cache: TokenCache, payload: dict[str, Any]) -> str:
    data = json.dumps(payload).encode("utf-8")
    for attempt in range(2):
        token = token_cache.get()
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401 and attempt == 0:
                token_cache.clear()
                continue
            raise RuntimeError(f"Vertex HTTP {exc.code}: {detail}") from exc
    else:
        raise RuntimeError("Vertex request failed before receiving a response.")

    predictions = body.get("predictions") or []
    if not predictions:
        raise RuntimeError(f"Vertex returned no predictions: {body}")
    prediction = predictions[0]
    if isinstance(prediction, str):
        return strip_vertex_echo(prediction)
    if isinstance(prediction, dict):
        for key in ("content", "text", "output", "generated_text", "prediction"):
            value = prediction.get(key)
            if isinstance(value, str):
                return strip_vertex_echo(value)
    return strip_vertex_echo(str(prediction))


def strip_vertex_echo(text: str) -> str:
    marker = "Output:\n"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def normalize_response(text: str) -> str:
    text = strip_vertex_echo(text)
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = text.strip()
    if "```mswea_bash_command" not in text and "```bash" in text:
        text = text.replace("```bash", "```mswea_bash_command", 1)
    if "```mswea_bash_command" in text and not text.lstrip().startswith("THOUGHT:"):
        text = f"THOUGHT: I will run the command below.\n\n{text}"
    return text


if __name__ == "__main__":
    raise SystemExit(main())
