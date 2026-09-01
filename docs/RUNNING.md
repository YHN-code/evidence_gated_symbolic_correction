
# Running the reproducibility package

## Tier 1: frozen-result verification

Tier 1 requires Python 3.12 only. Install the package, run `pytest`, and execute
the paper-figure generator from the repository root. This is the recommended
review path because it does not require network services or commercial software.

## Tier 2: deterministic experiment entry points

The `experiments/` directory contains the confirmation, evidence-gate,
constitutive material-point, and active-design entry points used by the study.
Use `python <script> --help` before a rerun. New results should be assigned a new
run identifier; do not overwrite the frozen paper artifacts.

## Tier 3: optional LLM proposals

The LLM is a candidate proposer, not the numerical verifier. Start from
`configs/llm_agent.example.yaml`, create `configs/llm_agent.local.yaml`, and keep
that file untracked. Never commit an API key. Provider output may change over
time; the frozen candidate/result artifacts remain the paper record.

## Tier 4: optional FLAC3D reruns

FLAC3D 7.0 is not distributed here. Install it separately, put
`flac3d700_console.exe` on `PATH` or edit a private configuration copy, and
install the optional Python transport dependency with
`python -m pip install -e ".[flac3d]"`. The supplied grid and data-file
template are sufficient for the retained controlled cavern case, subject to a
valid solver license.
