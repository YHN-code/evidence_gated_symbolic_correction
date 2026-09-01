from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from asrc.agent.llm_client import (
    LLMResponseError,
    LLMTransportError,
    call_llm_json,
    load_llm_config,
)
from asrc.model_revision.proposals import (
    ProposalBatch,
    RejectedRepair,
    RepairContract,
    RepairRequest,
    RepairValidationError,
    TypedRepair,
    validate_typed_repair,
)
from asrc.utils.io import read_json, write_json_atomic


SYSTEM_PROMPT = """You propose mathematical model revisions for a controlled scientific benchmark.
Return JSON only. Do not fit coefficients, claim acceptance, or invent variables. Use only the supplied
AST operators, variables, target, and edit types. Numerical coefficients that should be fitted must be
represented as parameter nodes, not guessed constants. The locked test data and generator are unavailable.
"""


_OPERATOR_NODE_CONTRACTS: dict[str, dict[str, Any]] = {
    "baseline": {"required_fields": ["op"], "arity": 0},
    "variable": {"required_fields": ["op", "name"], "arity": 0},
    "parameter": {"required_fields": ["op", "name"], "arity": 0},
    "constant": {"required_fields": ["op", "value"], "arity": 0},
    "add": {"required_fields": ["op", "arguments"], "arity": "2_or_more"},
    "multiply": {
        "required_fields": ["op", "arguments"],
        "arity": "2_or_more",
    },
    "subtract": {"required_fields": ["op", "left", "right"], "arity": 2},
    "divide": {"required_fields": ["op", "left", "right"], "arity": 2},
    "power": {"required_fields": ["op", "left", "right"], "arity": 2},
    "negate": {"required_fields": ["op", "argument"], "arity": 1},
    "abs": {"required_fields": ["op", "argument"], "arity": 1},
    "exp": {"required_fields": ["op", "argument"], "arity": 1},
    "log": {"required_fields": ["op", "argument"], "arity": 1},
    "sin": {"required_fields": ["op", "argument"], "arity": 1},
    "cos": {"required_fields": ["op", "argument"], "arity": 1},
    "tanh": {"required_fields": ["op", "argument"], "arity": 1},
}


def _schema_example(mode: str, target: str) -> dict[str, Any]:
    edit_type = "replace_subtree" if mode == "direct_equation" else "add_term"
    return {
        "proposals": [
            {
                "proposal_id": "llm_001",
                "source": "llm",
                "edit_type": edit_type,
                "target": target,
                "expression": {
                    "op": "multiply",
                    "arguments": [
                        {"op": "parameter", "name": "coefficient"},
                        {"op": "variable", "name": "x1"},
                    ],
                },
                "rationale": "Short evidence-based reason.",
                "expected_signature": "Residual behavior that would support this structure.",
            }
        ]
    }


def build_llm_repair_prompt(
    request: RepairRequest,
    contract: RepairContract,
    *,
    mode: str,
    candidate_count: int,
    prior_structural_keys: set[str] | None = None,
    proposal_context: Mapping[str, Any] | None = None,
) -> str:
    if mode not in {"typed_repair", "direct_equation"}:
        raise ValueError("mode must be 'typed_repair' or 'direct_equation'.")
    target = sorted(contract.allowed_targets)[0]
    edit_types = (
        ["replace_subtree"]
        if mode == "direct_equation"
        else [
            "add_term",
            "multiply_term",
            "add_state_dependence",
            "add_bounded_transition",
        ]
    )
    representation = (
        "Each expression is the complete corrected model and must use replace_subtree."
        if mode == "direct_equation"
        else "Each expression is a minimal patch applied to the supplied baseline model."
    )
    payload = {
        "objective": representation,
        "requested_candidate_count": int(candidate_count),
        "baseline_expression": request.baseline_expression,
        "observed_residual_evidence": request.residual_evidence,
        "constraints": list(request.constraints),
        "allowed_variables": sorted(contract.allowed_variables),
        "allowed_target": target,
        "allowed_edit_types": edit_types,
        "allowed_operators": sorted(contract.allowed_operators),
        "operator_node_contracts": {
            operator: _OPERATOR_NODE_CONTRACTS[operator]
            for operator in sorted(contract.allowed_operators)
        },
        "maximum_depth": contract.maximum_depth,
        "maximum_nodes": contract.maximum_nodes,
        "already_seen_structural_keys": sorted(prior_structural_keys or set()),
        "required_output_example": _schema_example(mode, target),
    }
    if proposal_context:
        payload["public_scientific_context"] = dict(proposal_context)
    return (
        "Propose diverse, compact candidate structures from the observed evidence. "
        "Follow operator_node_contracts exactly: unary nodes use argument, binary "
        "nodes use left/right, and only add/multiply use arguments. "
        "Do not output Markdown or explanatory text outside JSON.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _validate_response(
    raw: Any,
    contract: RepairContract,
    *,
    mode: str,
) -> ProposalBatch:
    if not isinstance(raw, dict) or not isinstance(raw.get("proposals"), list):
        return ProposalBatch(
            (),
            (RejectedRepair("batch", "llm", "Response requires a proposals list."),),
        )
    accepted: list[TypedRepair] = []
    rejected: list[RejectedRepair] = []
    seen: set[str] = set()
    for index, item in enumerate(raw["proposals"]):
        proposal_id = (
            str(item.get("proposal_id", f"llm_{index + 1:03d}"))
            if isinstance(item, dict)
            else f"llm_{index + 1:03d}"
        )
        if isinstance(item, dict):
            item = dict(item)
            item.setdefault("source", "llm")
        try:
            proposal = validate_typed_repair(item, contract)
        except RepairValidationError as exc:
            rejected.append(RejectedRepair(proposal_id, "llm", str(exc)))
            continue
        if proposal.source != "llm":
            rejected.append(
                RejectedRepair(proposal.proposal_id, "llm", "Proposal source must be llm.")
            )
            continue
        allowed_mode = (
            proposal.edit_type == "replace_subtree"
            if mode == "direct_equation"
            else proposal.edit_type != "replace_subtree"
        )
        if not allowed_mode:
            rejected.append(
                RejectedRepair(
                    proposal.proposal_id,
                    "llm",
                    f"edit_type {proposal.edit_type!r} is invalid for mode {mode!r}.",
                )
            )
            continue
        if proposal.structural_key in seen:
            rejected.append(
                RejectedRepair(proposal.proposal_id, "llm", "Duplicate structure in response.")
            )
            continue
        seen.add(proposal.structural_key)
        accepted.append(proposal)
    return ProposalBatch(tuple(accepted), tuple(rejected))


def _repair_prompt(
    original_prompt: str,
    raw: Any,
    rejected: tuple[RejectedRepair, ...],
) -> str:
    errors = [item.reason for item in rejected]
    return (
        original_prompt
        + "\nThe previous JSON did not satisfy the contract. Return a corrected complete "
        "proposals object. Do not discuss the errors.\n"
        + json.dumps({"validation_errors": errors, "previous_response": raw}, ensure_ascii=False)
    )


def _repair_payload(repair: TypedRepair) -> dict[str, Any]:
    return {
        "proposal_id": repair.proposal_id,
        "source": repair.source,
        "edit_type": repair.edit_type,
        "target": repair.target,
        "expression": repair.expression,
        "rationale": repair.rationale,
        "expected_signature": repair.expected_signature,
    }


def _rejected_payload(repair: RejectedRepair) -> dict[str, str]:
    return {
        "proposal_id": repair.proposal_id,
        "source": repair.source,
        "reason": repair.reason,
    }


def _is_truncated_response(error: LLMResponseError) -> bool:
    payload = error.payload or {}
    choices = payload.get("choices") or []
    if choices and choices[0].get("finish_reason") == "length":
        return True
    return "finish_reason='length'" in str(error)


class LLMRepairProposer:
    source = "llm"

    def __init__(
        self,
        *,
        config_path: str,
        mode: str,
        candidates_per_call: int = 8,
        maximum_generation_calls: int = 3,
        transport_retry_attempts: int = 2,
        transport_retry_wait_seconds: float = 20.0,
        model_override: str | None = None,
        reasoning_effort_override: str | None = None,
        max_tokens_override: int | None = None,
        truncation_batch_reductions: int = 2,
        proposal_context: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if mode not in {"typed_repair", "direct_equation"}:
            raise ValueError("Unsupported LLM repair mode.")
        self.config_path = config_path
        self.mode = mode
        self.candidates_per_call = max(1, int(candidates_per_call))
        self.maximum_generation_calls = max(1, int(maximum_generation_calls))
        self.transport_retry_attempts = max(0, int(transport_retry_attempts))
        self.transport_retry_wait_seconds = max(
            0.0, float(transport_retry_wait_seconds)
        )
        self.model_override = (
            str(model_override).strip() if model_override else None
        )
        self.reasoning_effort_override = reasoning_effort_override
        self.max_tokens_override = (
            max(1, int(max_tokens_override))
            if max_tokens_override is not None
            else None
        )
        self.truncation_batch_reductions = max(
            0, int(truncation_batch_reductions)
        )
        self.proposal_context = dict(proposal_context or {})
        self.proposal_context_sha256 = hashlib.sha256(
            json.dumps(
                self.proposal_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.resume = bool(resume)
        self.on_progress = on_progress
        self.llm_calls = 0
        self.repair_calls = 0
        self.transport_failures = 0
        self.truncation_retries = 0
        self.llm_elapsed_seconds = 0.0
        self.raw_payloads: list[dict[str, Any]] = []

    def _notify(self, event: str, **fields: Any) -> None:
        if self.on_progress is not None:
            self.on_progress(event, fields)

    def _call_with_transport_retries(
        self,
        prompt: str,
        runtime: Any,
        *,
        phase: str,
    ) -> dict[str, Any]:
        total_attempts = self.transport_retry_attempts + 1
        for attempt in range(1, total_attempts + 1):
            self.llm_calls += 1
            self._notify(
                "LLM request started",
                phase=phase,
                attempt=f"{attempt}/{total_attempts}",
            )
            started_at = time.perf_counter()
            try:
                raw = call_llm_json(
                    prompt,
                    config=runtime,
                    system_prompt=SYSTEM_PROMPT,
                )
            except LLMResponseError as exc:
                self.llm_calls += int(exc.additional_calls)
                self.repair_calls += int(exc.json_repair_calls)
                if exc.json_repair_calls:
                    self._notify(
                        "LLM JSON repair attempts exhausted",
                        phase=phase,
                        attempts=exc.json_repair_calls,
                    )
                raise
            except LLMTransportError as exc:
                self.transport_failures += 1
                if attempt >= total_attempts:
                    self._notify(
                        "LLM transport retries exhausted",
                        phase=phase,
                        error=type(exc).__name__,
                    )
                    raise
                self._notify(
                    "LLM transport retry scheduled",
                    phase=phase,
                    attempt=f"{attempt}/{total_attempts}",
                    wait_seconds=f"{self.transport_retry_wait_seconds:g}",
                )
                time.sleep(self.transport_retry_wait_seconds)
                continue
            finally:
                self.llm_elapsed_seconds += time.perf_counter() - started_at
            self.llm_calls += int(
                raw.get(
                    "_llm_additional_calls",
                    1 if raw.get("_llm_retry_without_thinking") else 0,
                )
            )
            self.repair_calls += int(raw.get("_llm_json_repair_attempts", 0))
            if raw.get("_llm_json_repair_attempts"):
                self._notify(
                    "LLM malformed JSON repaired",
                    phase=phase,
                    attempts=int(raw["_llm_json_repair_attempts"]),
                    thinking=False,
                )
            self.raw_payloads.append(raw)
            return raw
        raise AssertionError("Unreachable LLM transport retry state.")

    def _write_checkpoint(
        self,
        *,
        status: str,
        completed_generation_calls: int,
        accepted: list[TypedRepair],
        rejected: list[RejectedRepair],
        error: str | None = None,
    ) -> None:
        if self.checkpoint_path is None:
            return
        payload = {
            "version": 1,
            "mode": self.mode,
            "status": status,
            "completed_generation_calls": completed_generation_calls,
            "next_generation_call": completed_generation_calls + 1,
            "accepted": [_repair_payload(item) for item in accepted],
            "rejected": [_rejected_payload(item) for item in rejected],
            "metadata": {
                "llm_calls": self.llm_calls,
                "structured_repair_calls": self.repair_calls,
                "transport_failures": self.transport_failures,
                "truncation_retries": self.truncation_retries,
                "llm_elapsed_seconds": self.llm_elapsed_seconds,
                "raw_payloads": self.raw_payloads,
                "proposal_context_sha256": self.proposal_context_sha256,
            },
        }
        if error:
            payload["error"] = error
        write_json_atomic(self.checkpoint_path, payload)

    def _restore_checkpoint(
        self,
        request: RepairRequest,
        contract: RepairContract,
    ) -> tuple[list[TypedRepair], list[RejectedRepair], set[str], int]:
        accepted: list[TypedRepair] = []
        rejected: list[RejectedRepair] = []
        seen = set(request.failed_structural_keys)
        if (
            not self.resume
            or self.checkpoint_path is None
            or not self.checkpoint_path.exists()
        ):
            return accepted, rejected, seen, 0

        payload = read_json(self.checkpoint_path)
        if payload.get("version") != 1 or payload.get("mode") != self.mode:
            raise ValueError(
                f"Incompatible LLM proposal checkpoint: {self.checkpoint_path}"
            )
        for item in payload.get("accepted", []):
            proposal = validate_typed_repair(item, contract)
            if proposal.structural_key in seen:
                continue
            seen.add(proposal.structural_key)
            accepted.append(proposal)
        rejected.extend(
            RejectedRepair(
                str(item["proposal_id"]),
                str(item["source"]),
                str(item["reason"]),
            )
            for item in payload.get("rejected", [])
        )
        metadata = payload.get("metadata", {})
        checkpoint_context_sha256 = str(
            metadata.get("proposal_context_sha256", "")
        )
        if (
            self.proposal_context
            and checkpoint_context_sha256 != self.proposal_context_sha256
        ):
            raise ValueError(
                "LLM proposal checkpoint was created with a different scientific context."
            )
        self.llm_calls = int(metadata.get("llm_calls", 0))
        self.repair_calls = int(metadata.get("structured_repair_calls", 0))
        self.transport_failures = int(metadata.get("transport_failures", 0))
        self.truncation_retries = int(metadata.get("truncation_retries", 0))
        self.llm_elapsed_seconds = float(metadata.get("llm_elapsed_seconds", 0.0))
        self.raw_payloads = list(metadata.get("raw_payloads", []))
        completed = min(
            self.maximum_generation_calls,
            max(0, int(payload.get("completed_generation_calls", 0))),
        )
        self._notify(
            "LLM generation checkpoint restored",
            completed_calls=completed,
            accepted_candidates=len(accepted),
        )
        return accepted, rejected, seen, completed

    def _call_with_repairs(
        self,
        prompt: str,
        contract: RepairContract,
    ) -> ProposalBatch:
        runtime = load_llm_config(self.config_path)
        runtime_overrides: dict[str, Any] = {}
        if self.model_override:
            runtime_overrides["model"] = self.model_override
        if self.reasoning_effort_override:
            runtime_overrides["reasoning_effort"] = self.reasoning_effort_override
        if self.max_tokens_override is not None:
            runtime_overrides.update(
                {
                    "max_tokens": self.max_tokens_override,
                    "max_completion_tokens": None,
                }
            )
        if runtime_overrides:
            runtime = replace(runtime, **runtime_overrides)
        raw = self._call_with_transport_retries(
            prompt,
            runtime,
            phase="generation",
        )
        batch = _validate_response(raw, contract, mode=self.mode)
        for _ in range(runtime.structured_output_repair_attempts):
            if batch.accepted and not batch.rejected:
                break
            raw = self._call_with_transport_retries(
                _repair_prompt(prompt, raw, batch.rejected),
                runtime,
                phase="structured_output_repair",
            )
            self.repair_calls += 1
            batch = _validate_response(raw, contract, mode=self.mode)
        return batch

    def propose(
        self,
        request: RepairRequest,
        contract: RepairContract,
    ) -> ProposalBatch:
        accepted, rejected, seen, completed_calls = self._restore_checkpoint(
            request, contract
        )
        completed_generation_calls = completed_calls
        for generation_index in range(
            completed_calls, self.maximum_generation_calls
        ):
            if len(accepted) >= request.maximum_candidates:
                break
            self._notify(
                "LLM generation call started",
                generation=f"{generation_index + 1}/{self.maximum_generation_calls}",
                accepted_candidates=len(accepted),
            )
            requested_count = min(
                self.candidates_per_call,
                request.maximum_candidates - len(accepted),
            )
            reductions_used = 0
            while True:
                prompt = build_llm_repair_prompt(
                    request,
                    contract,
                    mode=self.mode,
                    candidate_count=requested_count,
                    prior_structural_keys=seen,
                    proposal_context=self.proposal_context,
                )
                try:
                    batch = self._call_with_repairs(prompt, contract)
                except LLMTransportError as exc:
                    self._write_checkpoint(
                        status="transport_failed",
                        completed_generation_calls=generation_index,
                        accepted=accepted,
                        rejected=rejected,
                        error=str(exc),
                    )
                    raise
                except LLMResponseError as exc:
                    can_reduce = (
                        _is_truncated_response(exc)
                        and requested_count > 1
                        and reductions_used < self.truncation_batch_reductions
                    )
                    if not can_reduce:
                        self._write_checkpoint(
                            status="response_failed",
                            completed_generation_calls=generation_index,
                            accepted=accepted,
                            rejected=rejected,
                            error=str(exc),
                        )
                        raise
                    reduced_count = max(1, requested_count // 2)
                    reductions_used += 1
                    self.truncation_retries += 1
                    self._notify(
                        "LLM truncated response batch reduced",
                        generation=f"{generation_index + 1}/{self.maximum_generation_calls}",
                        previous_candidates=requested_count,
                        retry_candidates=reduced_count,
                        reduction=f"{reductions_used}/{self.truncation_batch_reductions}",
                    )
                    requested_count = reduced_count
                    continue
                break
            rejected.extend(batch.rejected)
            for proposal in batch.accepted:
                if proposal.structural_key in seen:
                    rejected.append(
                        RejectedRepair(
                            proposal.proposal_id,
                            proposal.source,
                            "Duplicate structure across LLM generation calls.",
                        )
                    )
                    continue
                proposal = replace(
                    proposal,
                    proposal_id=f"llm_{len(accepted) + 1:03d}",
                )
                seen.add(proposal.structural_key)
                accepted.append(proposal)
                if len(accepted) >= request.maximum_candidates:
                    break
            completed_generation_calls = generation_index + 1
            self._write_checkpoint(
                status="in_progress",
                completed_generation_calls=completed_generation_calls,
                accepted=accepted,
                rejected=rejected,
            )
            self._notify(
                "LLM generation call finished",
                generation=f"{generation_index + 1}/{self.maximum_generation_calls}",
                accepted_candidates=len(accepted),
                rejected_candidates=len(rejected),
            )
        status = (
            "candidate_budget_filled"
            if len(accepted) >= request.maximum_candidates
            else "generation_calls_exhausted"
        )
        self._write_checkpoint(
            status=status,
            completed_generation_calls=completed_generation_calls,
            accepted=accepted,
            rejected=rejected,
        )
        return ProposalBatch(tuple(accepted), tuple(rejected))
