# LLM confirmation reproducibility record

This record documents the optional LLM candidate source used in the prespecified 32-case confirmation. It contains no API credential and no hidden chain-of-thought.

## Frozen run

- Run ID: `semantic_evidence_guided_revision_confirmation_v1`
- Run creation time: `2026-08-11T08:48:05.129237+00:00`
- Provider/API: DeepSeek, OpenAI-compatible chat-completions endpoint
- Model: `deepseek-v4-pro`
- Reasoning: enabled; confirmation override `high`
- Temperature: omitted from the API request (`send_temperature: false`), so provider-default sampling controls applied
- Maximum output tokens: `12000`
- Generation budget: two calls per task-seed cell, at most six valid typed proposals per call
- Transport policy: five retries separated by 25 seconds

## Versioned inputs

- Confirmation protocol: `configs/semantic_evidence_guided_revision_confirmation_v1.yaml`
- Local runtime configuration template: `configs/llm_agent.local.example.yaml`
- System prompt and prompt constructor: `src/asrc/model_revision/llm_proposer.py`
- Typed expression and repair validation: `src/asrc/model_revision/proposals.py`
- API request and structured-response parsing: `src/asrc/agent/llm_client.py`

The prompt constructor receives only the baseline description, visible observations and diagnostics, allowed variables and operators, the physical contract, and the approved knowledge snippets. It does not receive the hidden reference equation or locked-test responses.

## Archived outputs

The run archive is rooted at:

`outputs/runs/semantic_evidence_guided_revision_confirmation_v1/`

Per-cell parsed response payloads and repair checkpoints are stored under:

`generation/seed_*/formulas/semantic_attribution/checkpoints/*.json`

The checkpoints record submitted proposal objects, parsing or validation repairs, and accepted or rejected typed candidates. API keys and provider-side hidden reasoning are not stored. A publication archive should include the configuration, code revision, run manifest, and these checkpoint files so that candidate attribution can be audited independently of API availability.
