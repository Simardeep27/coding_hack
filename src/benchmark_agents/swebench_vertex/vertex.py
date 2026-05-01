from __future__ import annotations

from google import genai
from google.genai import types

from benchmark_agents.swebench_vertex.config import VertexConfig


class VertexResponder:
    def __init__(self, config: VertexConfig) -> None:
        self._config = config
        self._provider = "google"
        self._client = None
        if config.model.startswith("claude-"):
            self._provider = "anthropic_vertex"
            try:
                from anthropic import AnthropicVertex
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "The Anthropic Vertex client is required for Claude models. "
                    "Install the project dependencies again so `anthropic[vertex]` is available."
                ) from exc
            self._client = AnthropicVertex(
                project_id=config.project,
                region=config.location,
            )
        else:
            self._client = genai.Client(
                vertexai=True,
                project=config.project,
                location=config.location,
            )

    @property
    def model_name(self) -> str:
        return self._config.model

    def generate_json_response(self, prompt: str) -> str:
        if self._provider == "anthropic_vertex":
            return self._generate_anthropic_json_response(prompt)
        return self._generate_google_json_response(prompt)

    def _generate_google_json_response(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_output_tokens,
                response_mime_type="application/json",
            ),
        )

        if getattr(response, "text", None):
            return response.text

        chunks: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)

        if chunks:
            return "\n".join(chunks)

        raise RuntimeError("Vertex response did not contain any text content.")

    def _generate_anthropic_json_response(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_output_tokens,
            temperature=self._config.temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        chunks: list[str] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", None)
            if block_type == "text" and block_text:
                chunks.append(block_text)

        if chunks:
            return "\n".join(chunks)

        raise RuntimeError("Anthropic Vertex response did not contain any text content.")

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()
