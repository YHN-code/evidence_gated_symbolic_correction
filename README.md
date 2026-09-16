
# Evidence-gated symbolic correction

This repository is the reproducibility package for **Evidence-gated symbolic correction of geomechanical models under structural ambiguity**. The method
starts from an operational baseline model, evaluates explicit symbolic
corrections, and reports whether the available evidence distinguishes among
physically admissible alternatives. Candidate expressions may be supplied by
PySR, an optional typed LLM proposer, or provenance-tracked templates. Fitting,
validation, physical checks, evidence gating, acquisition, and final acceptance
are deterministic.

## Reproducibility scope

The repository includes only the Python modules reached by the paper reproduction entry points, prespecified configurations,
compact frozen results, and frozen solver exports needed to regenerate the
manuscript figures without an LLM API or FLAC3D license. Generated figures are
not distributed in the repository. It does not include the private
research history, local API configuration, commercial FLAC3D binaries, solver
save files, source-paper PDFs, or third-party raw datasets.

The public package was exported from source commit `0b8641239cf82b2bbfa980cd6219022a78f17363`.

## Python 3.12 environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pytest -q
```

## Regenerate manuscript figures

```powershell
python experiments/make_evidence_gated_paper_figures.py `
  --output-dir paper_assets/asrc_evidence_gated/figures
```

The command writes PNG (600 dpi), PDF, and editable-text SVG files. It reads
compact frozen CSV/JSON artifacts under `outputs/runs/` and `data/frozen_cavern/`.

## Optional software

- PySR is optional: `python -m pip install -e ".[pysr]"`.
- LLM proposal generation is optional. Copy `configs/llm_agent.example.yaml` to
  an untracked local file and insert credentials only in that local copy.
- FLAC3D 7.0 reruns require a separately licensed installation. The repository
  contains no commercial binary or license material; frozen exports are enough
  to reproduce the paper figures.

See `docs/RUNNING.md` and `docs/DATA_AND_SOFTWARE.md` for details.

## Citation

Citation metadata is provided in `CITATION.cff`. A DOI and final bibliographic
record will be added after archival release.

## Update scope

The numerical implementation remains the initial public release. The figure
generator has been updated to use Times New Roman (Liberation Serif or
STIXGeneral if unavailable) and the six-MPa cavern field export.
Exact typography requires Times New Roman installed on the rendering machine.

Supplementary Section S12 is supported by frozen case-level results:
`python experiments/verify_full_domain_results.py`.
This recomputes aggregate statistics and checks paired-policy equality. It
does not regenerate candidates, refit the models, or independently prove the
algebraic certificates. The later search implementation is not substituted
for the original primary-study implementation.
