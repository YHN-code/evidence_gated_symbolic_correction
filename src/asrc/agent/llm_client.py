from __future__ import annotations

import json
import os
import http.client
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asrc.agent.prompts import SYSTEM_PROMPT
from asrc.agent.context import DEFAULT_AGENT_ACTIONS
from asrc.utils.io import read_yaml


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    def __init__(
        self,
        message: str,
        payload: dict[str, Any] | None = None,
        content: str | None = None,
        additional_calls: int = 0,
        json_repair_calls: int = 0,
    ):
        super().__init__(message)
        self.payload = payload
        self.content = content
        self.additional_calls = max(0, int(additional_calls))
        self.json_repair_calls = max(0, int(json_repair_calls))


class LLMTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMClientConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.2
    send_temperature: bool = True
    thinking_type: str | None = None
    reasoning_effort: str | None = None
    timeout_s: int = 60
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_rounds: int = 5
    top_k_feedback: int = 5
    early_stop_relative_rmse: float = 0.01
    early_stop_patience: int = 2
    retry_without_thinking_on_empty: bool = True
    retry_without_thinking_on_truncation: bool = True
    structured_output_repair_attempts: int = 2
    agent_available_actions: list[str] | None = None
    agent_context_recent_rounds: int = 2
    agent_knowledge_top_k: int = 3
    agent_allow_accept_before_round: int = 2
    agent_acceptance_requirements: list[str] | None = None
    agent_validation_top_k: int = 8
    agent_validation_budget_total: int = 24
    agent_final_audit_enabled: bool = True
    agent_audit_recovery_rounds: int = 0
    agent_structural_audit_enabled: bool = True
    agent_structural_validation_top_k: int = 8
    agent_structural_bootstrap_samples: int = 1000
    agent_structural_confidence: float = 0.95


def _load_config_file(config_path: str | Path | None) -> dict[str, Any]:
    if not config_path:
        return {}
    payload = read_yaml(config_path)
    return payload.get("llm", payload)


def _env_or_config(env_name: str, config: dict[str, Any], key: str, default: Any = None) -> Any:
    return os.environ.get(env_name) or config.get(key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _as_string_list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value)]


def load_llm_config(config_path: str | Path | None = None) -> LLMClientConfig:
    config = _load_config_file(config_path)
    provider = _env_or_config("ASRC_LLM_PROVIDER", config, "provider", "openai-compatible")
    model = _env_or_config("ASRC_LLM_MODEL", config, "model")
    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ASRC_LLM_API_KEY")
        or config.get("api_key")
    )
    base_url = _env_or_config("ASRC_LLM_BASE_URL", config, "base_url", "https://api.openai.com/v1/chat/completions")
    temperature = float(_env_or_config("ASRC_LLM_TEMPERATURE", config, "temperature", 0.2))
    send_temperature = _as_bool(_env_or_config("ASRC_LLM_SEND_TEMPERATURE", config, "send_temperature", True))
    thinking_type = _env_or_config("ASRC_LLM_THINKING", config, "thinking")
    reasoning_effort = _env_or_config("ASRC_LLM_REASONING_EFFORT", config, "reasoning_effort")
    timeout_s = int(_env_or_config("ASRC_LLM_TIMEOUT_S", config, "timeout_s", 60))
    max_retries = int(_env_or_config("ASRC_LLM_MAX_RETRIES", config, "max_retries", 3))
    retry_backoff_s = float(_env_or_config("ASRC_LLM_RETRY_BACKOFF_S", config, "retry_backoff_s", 2.0))
    stream = _as_bool(_env_or_config("ASRC_LLM_STREAM", config, "stream", False))
    agent_cfg = config.get("agent", {}) if isinstance(config.get("agent", {}), dict) else {}
    max_tokens_value = _env_or_config("ASRC_LLM_MAX_TOKENS", config, "max_tokens")
    max_tokens = int(max_tokens_value) if max_tokens_value not in (None, "") else None
    max_completion_tokens_value = _env_or_config(
        "ASRC_LLM_MAX_COMPLETION_TOKENS",
        config,
        "max_completion_tokens",
    )
    max_completion_tokens = (
        int(max_completion_tokens_value)
        if max_completion_tokens_value not in (None, "")
        else None
    )
    max_rounds = int(os.environ.get("ASRC_LLM_MAX_ROUNDS") or config.get("max_rounds", agent_cfg.get("max_rounds", 5)))
    top_k_feedback = int(_env_or_config("ASRC_LLM_TOP_K_FEEDBACK", config, "top_k_feedback", 5))
    early_stop_relative_rmse = float(_env_or_config("ASRC_LLM_EARLY_STOP_RELATIVE_RMSE", config, "early_stop_relative_rmse", 0.01))
    early_stop_patience = int(_env_or_config("ASRC_LLM_EARLY_STOP_PATIENCE", config, "early_stop_patience", 2))
    agent_available_actions = _as_string_list(agent_cfg.get("available_actions"), DEFAULT_AGENT_ACTIONS)
    agent_context_recent_rounds = int(agent_cfg.get("context_recent_rounds", 2))
    agent_knowledge_top_k = int(agent_cfg.get("knowledge_top_k", 3))
    agent_allow_accept_before_round = int(agent_cfg.get("allow_accept_before_round", 2))
    agent_acceptance_requirements = _as_string_list(agent_cfg.get("acceptance_requirements"), None)
    agent_validation_top_k = int(agent_cfg.get("validation_top_k", 8))
    agent_validation_budget_total = int(agent_cfg.get("validation_budget_total", 24))
    agent_final_audit_enabled = _as_bool(agent_cfg.get("final_audit_enabled", True))
    agent_audit_recovery_rounds = int(agent_cfg.get("audit_recovery_rounds", 0))
    agent_structural_audit_enabled = _as_bool(agent_cfg.get("structural_audit_enabled", True))
    agent_structural_validation_top_k = int(agent_cfg.get("structural_validation_top_k", 8))
    agent_structural_bootstrap_samples = int(agent_cfg.get("structural_bootstrap_samples", 1000))
    agent_structural_confidence = float(agent_cfg.get("structural_confidence", 0.95))
    retry_without_thinking_on_empty = _as_bool(
        _env_or_config("ASRC_LLM_RETRY_WITHOUT_THINKING_ON_EMPTY", config, "retry_without_thinking_on_empty", True)
    )
    retry_without_thinking_on_truncation = _as_bool(
        _env_or_config(
            "ASRC_LLM_RETRY_WITHOUT_THINKING_ON_TRUNCATION",
            config,
            "retry_without_thinking_on_truncation",
            True,
        )
    )
    structured_output_repair_attempts = int(
        _env_or_config(
            "ASRC_LLM_STRUCTURED_OUTPUT_REPAIR_ATTEMPTS",
            config,
            "structured_output_repair_attempts",
            2,
        )
    )
    if not model:
        raise LLMConfigurationError("LLM model is required when --llm is enabled. Set ASRC_LLM_MODEL or llm.model in --llm-config.")
    if not api_key:
        raise LLMConfigurationError(
            "LLM API key is required when --llm is enabled. Set llm.api_key, OPENAI_API_KEY, or ASRC_LLM_API_KEY."
        )
    if max_tokens is not None and max_completion_tokens is not None:
        raise LLMConfigurationError(
            "Configure only one of llm.max_tokens and "
            "llm.max_completion_tokens."
        )
    return LLMClientConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        send_temperature=send_temperature,
        thinking_type=str(thinking_type) if thinking_type else None,
        reasoning_effort=str(reasoning_effort) if reasoning_effort else None,
        timeout_s=timeout_s,
        max_retries=max(0, max_retries),
        retry_backoff_s=max(0.0, retry_backoff_s),
        stream=stream,
        max_tokens=max_tokens,
        max_completion_tokens=max_completion_tokens,
        max_rounds=max(1, max_rounds),
        top_k_feedback=max(1, top_k_feedback),
        early_stop_relative_rmse=max(0.0, early_stop_relative_rmse),
        early_stop_patience=max(1, early_stop_patience),
        retry_without_thinking_on_empty=retry_without_thinking_on_empty,
        retry_without_thinking_on_truncation=retry_without_thinking_on_truncation,
        structured_output_repair_attempts=max(0, structured_output_repair_attempts),
        agent_available_actions=agent_available_actions,
        agent_context_recent_rounds=max(1, agent_context_recent_rounds),
        agent_knowledge_top_k=max(1, agent_knowledge_top_k),
        agent_allow_accept_before_round=max(1, agent_allow_accept_before_round),
        agent_acceptance_requirements=agent_acceptance_requirements or None,
        agent_validation_top_k=max(1, agent_validation_top_k),
        agent_validation_budget_total=max(1, agent_validation_budget_total),
        agent_final_audit_enabled=agent_final_audit_enabled,
        agent_audit_recovery_rounds=max(0, min(agent_audit_recovery_rounds, max(0, max_rounds - 1))),
        agent_structural_audit_enabled=agent_structural_audit_enabled,
        agent_structural_validation_top_k=max(1, agent_structural_validation_top_k),
        agent_structural_bootstrap_samples=max(100, agent_structural_bootstrap_samples),
        agent_structural_confidence=min(0.999, max(0.5, agent_structural_confidence)),
    )


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    if not text:
        raise LLMResponseError("LLM response content is empty.")

    def decode(candidate: str) -> dict[str, Any] | None:
        try:
            value: Any = json.loads(candidate, strict=False)
            if isinstance(value, str):
                value = json.loads(value.strip(), strict=False)
            return value if isinstance(value, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    decoded = decode(text)
    if decoded is not None:
        return decoded
    try:
        json.loads(text, strict=False)
    except (json.JSONDecodeError, TypeError):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            decoded = decode(text[start : end + 1])
            if decoded is not None:
                return decoded
        preview = text[:500].replace("\n", "\\n")
        raise LLMResponseError(f"LLM response content is not valid JSON. content_preview={preview!r}", content=content)
    raise LLMResponseError("LLM response JSON must be an object.", content=content)


def _json_syntax_repair_prompt(
    original_prompt: str,
    malformed_content: str,
) -> str:
    prompt_excerpt = original_prompt[:12000]
    content_excerpt = malformed_content[:16000]
    return (
        "Repair the malformed structured response below. Return exactly one valid "
        "JSON object and no Markdown or explanation. Preserve the proposal meanings, "
        "field names, and requested schema; only repair JSON syntax or complete a "
        "prematurely ended object.\n\n"
        "ORIGINAL REQUEST (for schema context):\n"
        f"{prompt_excerpt}\n\n"
        "MALFORMED RESPONSE:\n"
        f"{content_excerpt}"
    )


def _read_response_payload(request: urllib.request.Request, config: LLMClientConfig) -> dict[str, Any]:
    retryable_errors = (
        json.JSONDecodeError,
        UnicodeDecodeError,
        http.client.IncompleteRead,
        http.client.HTTPException,
        TimeoutError,
        socket.timeout,
        ConnectionError,
        urllib.error.URLError,
    )
    last_error: BaseException | None = None
    for attempt in range(1, config.max_retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except urllib.error.HTTPError:
            raise
        except retryable_errors as exc:
            last_error = exc
            if attempt > config.max_retries:
                break
            time.sleep(config.retry_backoff_s * attempt)
    raise LLMTransportError(
        f"LLM API transport/read failed after {config.max_retries + 1} attempt(s): {type(last_error).__name__}: {last_error}"
    ) from last_error


def _consume_stream_response(response: Any) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_char_count = 0
    event_count = 0
    finish_reason: str | None = None
    usage: dict[str, Any] = {}
    stream_done = False

    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            stream_done = True
            break
        event = json.loads(data)
        event_count += 1
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if content:
            content_parts.append(str(content))
        if reasoning:
            reasoning_char_count += len(str(reasoning))
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])

    if not stream_done and finish_reason is None:
        raise http.client.IncompleteRead(b"")
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": "".join(content_parts),
                },
            }
        ],
        "usage": usage,
        "_stream_transport": {
            "event_count": event_count,
            "reasoning_char_count": reasoning_char_count,
            "content_char_count": sum(len(part) for part in content_parts),
        },
    }


def _read_stream_response_payload(
    request: urllib.request.Request,
    config: LLMClientConfig,
) -> dict[str, Any]:
    retryable_errors = (
        json.JSONDecodeError,
        UnicodeDecodeError,
        http.client.IncompleteRead,
        http.client.HTTPException,
        TimeoutError,
        socket.timeout,
        ConnectionError,
        urllib.error.URLError,
    )
    last_error: BaseException | None = None
    for attempt in range(1, config.max_retries + 2):
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_s) as response:
                return _consume_stream_response(response)
        except urllib.error.HTTPError:
            raise
        except retryable_errors as exc:
            last_error = exc
            if attempt > config.max_retries:
                break
            time.sleep(config.retry_backoff_s * attempt)
    raise LLMTransportError(
        "LLM API stream failed after "
        f"{config.max_retries + 1} attempt(s): "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _build_chat_body(
    prompt: str,
    config: LLMClientConfig,
    include_thinking: bool = True,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    if include_thinking and config.thinking_type:
        body["thinking"] = {"type": config.thinking_type}
    if include_thinking and config.reasoning_effort:
        body["reasoning_effort"] = config.reasoning_effort
    if config.send_temperature:
        body["temperature"] = config.temperature
    if config.max_tokens is not None:
        body["max_tokens"] = config.max_tokens
    if config.max_completion_tokens is not None:
        body["max_completion_tokens"] = config.max_completion_tokens
    if config.stream:
        body["stream"] = True
    return body


def _send_chat_body(body: dict[str, Any], config: LLMClientConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        config.base_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    if config.stream:
        return _read_stream_response_payload(request, config)
    return _read_response_payload(request, config)


def _parse_choice_content(payload: dict[str, Any]) -> dict:
    choice = payload["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    try:
        return _extract_json_object(content)
    except LLMResponseError as exc:
        finish_reason = choice.get("finish_reason")
        reasoning_tokens = payload.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
        hint = (
            f" finish_reason={finish_reason!r} reasoning_tokens={reasoning_tokens!r}. "
            "If content is empty or truncated, increase the configured output-token limit or disable/reduce thinking mode."
        )
        raise LLMResponseError(str(exc) + hint, payload=payload, content=content) from exc


def call_llm_json(
    prompt: str,
    config: LLMClientConfig | None = None,
    config_path: str | Path | None = None,
    system_prompt: str | None = None,
    include_thinking: bool | None = None,
) -> dict:
    config = config or load_llm_config(config_path)
    thinking_enabled = (
        True if include_thinking is None else bool(include_thinking)
    )
    body = _build_chat_body(
        prompt,
        config,
        include_thinking=thinking_enabled,
        system_prompt=system_prompt,
    )
    try:
        payload = _send_chat_body(body, config)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP error {exc.code}: {detail}") from exc
    try:
        return _parse_choice_content(payload)
    except LLMResponseError as first_exc:
        def finish_reason(error: LLMResponseError) -> str | None:
            if error.payload and error.payload.get("choices"):
                return error.payload["choices"][0].get("finish_reason")
            return None

        first_finish_reason = finish_reason(first_exc)
        empty_content = not (first_exc.content or "").strip()
        truncated_content = first_finish_reason == "length"
        should_retry_without_thinking = (
            thinking_enabled
            and bool(config.thinking_type or config.reasoning_effort)
            and (
                (config.retry_without_thinking_on_empty and empty_content)
                or (
                    config.retry_without_thinking_on_truncation
                    and truncated_content
                )
            )
        )
        current_exc = first_exc
        additional_calls = 0
        retry_reason: str | None = None
        if should_retry_without_thinking:
            retry_reason = (
                "truncated_content" if truncated_content else "empty_content"
            )
            retry_body = _build_chat_body(
                prompt,
                config,
                include_thinking=False,
                system_prompt=system_prompt,
            )
            try:
                retry_payload = _send_chat_body(retry_body, config)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"LLM API HTTP error {exc.code}: {detail}"
                ) from exc
            additional_calls += 1
            try:
                parsed = _parse_choice_content(retry_payload)
                parsed["_llm_retry_without_thinking"] = True
                parsed["_llm_retry_reason"] = retry_reason
                parsed["_llm_additional_calls"] = additional_calls
                return parsed
            except LLMResponseError as retry_exc:
                current_exc = retry_exc

        syntax_repairs = 0
        current_content = (current_exc.content or "").strip()
        can_repair_syntax = bool(current_content) and finish_reason(current_exc) != "length"
        while (
            can_repair_syntax
            and syntax_repairs < config.structured_output_repair_attempts
        ):
            repair_body = _build_chat_body(
                _json_syntax_repair_prompt(prompt, current_content),
                config,
                include_thinking=False,
                system_prompt=system_prompt,
            )
            try:
                repair_payload = _send_chat_body(repair_body, config)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"LLM API HTTP error {exc.code}: {detail}"
                ) from exc
            syntax_repairs += 1
            additional_calls += 1
            try:
                parsed = _parse_choice_content(repair_payload)
                if retry_reason is not None:
                    parsed["_llm_retry_without_thinking"] = True
                    parsed["_llm_retry_reason"] = retry_reason
                parsed["_llm_json_repair_attempts"] = syntax_repairs
                parsed["_llm_additional_calls"] = additional_calls
                return parsed
            except LLMResponseError as repair_exc:
                current_exc = repair_exc
                current_content = (current_exc.content or "").strip()
                can_repair_syntax = (
                    bool(current_content)
                    and finish_reason(current_exc) != "length"
                )

        context = ""
        if retry_reason is not None:
            context += "; retry_without_thinking also failed"
        if syntax_repairs:
            context += f"; JSON syntax repair failed after {syntax_repairs} attempt(s)"
        raise LLMResponseError(
            f"{first_exc}{context}: {current_exc}",
            payload=current_exc.payload,
            content=current_exc.content,
            additional_calls=additional_calls,
            json_repair_calls=syntax_repairs,
        ) from current_exc
