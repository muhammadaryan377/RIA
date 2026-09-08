"""
ARIA LLM Provider
-----------------
Unified interface over two LLM backends so the agents can run either:

  * local  - Ollama (offline, private, data never leaves the machine)
  * cloud  - DeepSeek API (hosted inference)

Every agent talks to ``chat()`` and does not need provider-specific code.
"""

import json
import logging
import os
import re
import time

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from openai import OpenAI
    DEEPSEEK_AVAILABLE = True
except ImportError:
    OpenAI = None
    DEEPSEEK_AVAILABLE = False

# Load .env centrally before module-level os.getenv calls below.
import core.config  # noqa: E402,F401

logging.basicConfig(level=logging.INFO)

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-v4-flash-vision-exp")

# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

PROVIDERS = {
    "local": {
        "label": "Local LLM (Ollama)",
        "privacy": "Your data never leaves your machine. Fully offline & secure, but slower on commodity hardware.",
        "models": {
            "sql": "aria-sql-mistral",
            "story": "aria-story-mistral",
            "suggest": "aria-goal",
            "semi": "aria-goal",
            "schema": "aria-goal",
        },
    },
    "cloud": {
        "label": "Cloud LLM (DeepSeek API)",
        "privacy": "Hosted DeepSeek inference. Your schema, queries, selected PDF evidence, and data samples may be sent to the DeepSeek API.",
        "models": {
            "sql": DEEPSEEK_MODEL,
            "story": DEEPSEEK_MODEL,
            "suggest": DEEPSEEK_MODEL,
            "semi": DEEPSEEK_MODEL,
            "schema": DEEPSEEK_MODEL,
            "vision": DEEPSEEK_VISION_MODEL,
        },
    },
}

# Local Ollama models default to a small runtime context (often 4096 tokens).
LOCAL_NUM_CTX = 8192
_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)


class LLMProvider:
    """Unified chat interface: local (Ollama) or cloud (DeepSeek)."""

    def __init__(self, provider="local", api_key=None, models=None, base_url=None):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDERS)}")
        self.provider = provider
        self.models = dict(PROVIDERS[provider]["models"])
        if models:
            self.models.update(models)

        self._cloud_client = None
        if provider == "cloud":
            if not DEEPSEEK_AVAILABLE:
                raise RuntimeError("openai package is not installed. Run: pip install openai")
            key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in .env or pass api_key.")
            self._cloud_client = OpenAI(
                api_key=key,
                base_url=(base_url or DEEPSEEK_BASE_URL).rstrip("/"),
                max_retries=0,
            )

    # -- model lookup ---------------------------------------------------

    def model_for(self, role):
        return self.models.get(role)

    # -- unified chat ---------------------------------------------------

    def chat(self, role, messages, temperature=0.1, num_predict=400, timeout=None):
        if timeout is None:
            if self.provider == "local" and role in ("story", "prescription"):
                timeout = 900
            else:
                timeout = 30 if self.provider != "local" else 60
        model = self.model_for(role)
        if not model:
            raise ValueError(f"No model configured for role '{role}'")

        try:
            if self.provider == "local":
                return self._chat_local(model, messages, temperature, num_predict, timeout)
            return self._chat_cloud(model, messages, temperature, num_predict, timeout)
        except TimeoutError:
            raise
        except Exception as exc:
            logging.warning("LLMProvider.%s chat failed (%s): %s", self.provider, role, exc)
            raise

    def chat_structured(
        self,
        role,
        messages,
        *,
        json_schema,
        schema_name="structured_response",
        temperature=0.0,
        num_predict=900,
        timeout=None,
        reasoning_effort="low",
    ):
        """Return validated-JSON-ready content from DeepSeek JSON mode.

        DeepSeek Chat Completions supports ``response_format={type: json_object}``.
        ARIA still performs its own Pydantic/schema validation after generation, so
        provider JSON mode is a formatting guarantee rather than a trust boundary.
        """
        del reasoning_effort  # DeepSeek-specific thinking is not needed for router JSON.
        if self.provider != "cloud":
            raise RuntimeError("Structured output is currently available only on the cloud provider.")
        if timeout is None:
            timeout = 20
        model = self.model_for(role)
        if not model:
            raise ValueError(f"No model configured for role '{role}'")
        if not isinstance(json_schema, dict) or json_schema.get("type") != "object":
            raise ValueError("json_schema must be a JSON Schema object definition.")

        schema_text = json.dumps(json_schema, ensure_ascii=False, separators=(",", ":"))
        structured_messages = [
            {
                "role": "system",
                "content": (
                    f"Return only one valid JSON object named {schema_name}. "
                    "It must match this JSON Schema exactly. Do not use markdown fences, commentary, or extra keys. "
                    f"JSON Schema: {schema_text}"
                ),
            },
            *messages,
        ]
        try:
            completion = self._cloud_with_retry(
                lambda: self._cloud_client.chat.completions.create(
                    model=model,
                    messages=structured_messages,
                    temperature=temperature,
                    max_tokens=num_predict,
                    timeout=timeout,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
            )
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("Structured LLM response was empty.")
            return content.strip()
        except TimeoutError:
            raise
        except Exception as exc:
            logging.warning("LLMProvider.%s structured chat failed (%s): %s", self.provider, role, exc)
            raise

    def complete(self, role, prompt, temperature=0.1, num_predict=400, timeout=None):
        if timeout is None:
            if self.provider == "local" and role == "sql":
                timeout = 900
            else:
                timeout = 30 if self.provider != "local" else 300
        model = self.model_for(role)
        if not model:
            raise ValueError(f"No model configured for role '{role}'")

        try:
            if self.provider == "local":
                return self._complete_local(model, prompt, temperature, num_predict, timeout)
            completion = self._cloud_with_retry(
                lambda: self._cloud_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=num_predict,
                    timeout=timeout,
                )
            )
            content = completion.choices[0].message.content
            return (content or "").strip()
        except TimeoutError:
            raise
        except Exception as exc:
            logging.warning("LLMProvider.%s complete failed (%s): %s", self.provider, role, exc)
            raise

    def _complete_local(self, model, prompt, temperature, num_predict, timeout):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")
        client = ollama.Client(timeout=timeout)
        response = client.generate(
            model=model,
            prompt=prompt,
            keep_alive="30m",
            options={
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": LOCAL_NUM_CTX,
                "stop": ["<|im_end|>", "<|im_start|>", "```", "<|endoftext|>", "\n\nUser:"],
            },
        )
        return response["response"].strip()

    def _chat_local(self, model, messages, temperature, num_predict, timeout):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")
        client = ollama.Client(timeout=timeout)
        response = client.chat(
            model=model,
            messages=messages,
            keep_alive="30m",
            options={"temperature": temperature, "num_predict": num_predict, "num_ctx": LOCAL_NUM_CTX},
        )
        return response["message"]["content"].strip()

    def _chat_cloud(self, model, messages, temperature, num_predict, timeout):
        completion = self._cloud_with_retry(
            lambda: self._cloud_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=num_predict,
                timeout=timeout,
            )
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Cloud LLM response was empty.")
        return content.strip()

    @staticmethod
    def _retry_after_seconds(exc):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                raw = headers.get("retry-after") or headers.get("Retry-After")
                if raw is not None and float(raw) > 0:
                    return float(raw)
            except (TypeError, ValueError):
                pass
        match = _RETRY_AFTER_RE.search(str(exc))
        if match:
            try:
                value = float(match.group(1))
                return value if value > 0 else None
            except (TypeError, ValueError):
                pass
        return None

    def _cloud_with_retry(self, fn, attempts=4):
        """Bounded retry for hosted-provider rate limits."""
        last_exc = None
        attempts = max(1, int(attempts))
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                response = getattr(exc, "response", None)
                status = getattr(exc, "status_code", None)
                if status is None and response is not None:
                    status = getattr(response, "status_code", None)
                if status != 429 or attempt >= attempts - 1:
                    raise
                hinted = self._retry_after_seconds(exc)
                wait = min(max((hinted + 0.5) if hinted else 1.5 * (2**attempt), 1.5), 20.0)
                logging.warning(
                    "Rate limit (429); retrying in %.1fs (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    attempts,
                )
                time.sleep(wait)
        raise last_exc  # pragma: no cover

    def vision(self, prompt, images, temperature=0.1, num_predict=1500, timeout=60):
        """Send rendered page images to the hosted DeepSeek vision model."""
        model = self.models.get("vision")
        if not model:
            raise RuntimeError(f"No vision model configured for provider '{self.provider}'.")
        if self.provider == "local":
            raise RuntimeError(
                "Handwritten/scanned PDF extraction requires the hosted Cloud (DeepSeek) provider."
            )
        content = [{"type": "text", "text": prompt}]
        for b64 in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return self._vision_once(prompt, content, model, temperature, num_predict, timeout)

    def _vision_once(self, prompt, content, model, temperature, num_predict, timeout):
        kwargs = {}
        if "json" in (prompt or "").lower():
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._cloud_with_retry(
            lambda: self._cloud_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=temperature,
                max_tokens=num_predict,
                timeout=timeout,
                **kwargs,
            )
        )
        content_text = completion.choices[0].message.content
        if not content_text:
            raise RuntimeError("Vision response was empty.")
        return content_text.strip()

    def __repr__(self):
        return f"LLMProvider(provider={self.provider}, models={self.models})"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def create_provider(provider="local", api_key=None, models=None, base_url=None):
    """Build an LLMProvider without silently switching backends."""
    if provider == "cloud":
        if not DEEPSEEK_AVAILABLE:
            raise RuntimeError(
                "The Cloud provider requires the OpenAI-compatible client package. "
                "Run: pip install openai"
            )
        return LLMProvider(
            "cloud",
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            models=models,
            base_url=base_url or os.getenv("DEEPSEEK_BASE_URL") or DEEPSEEK_BASE_URL,
        )
    return LLMProvider("local", models=models)
