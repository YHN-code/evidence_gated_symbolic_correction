
# Data and software boundaries

## Included

- Project-generated synthetic and solver-export summary CSV/JSON files required
  by the manuscript figures.
- Prespecified YAML configurations and random-seed metadata used by the retained
  confirmation analyses.
- A compact FLAC3D grid and text template for the controlled cavern case.

## Not included

- Original literature PDFs and publisher-supplied supplementary files.
- Third-party raw or processed tables whose redistribution terms have not been
  confirmed. The manuscript identifies the source publications and reports only
  the frozen aggregate values needed for its literature evaluation.
- FLAC3D executables, licenses, save states, and temporary run directories.
- LLM credentials, request headers, private account metadata, and local config.

## Literature evaluation

The four literature-case Group-CV summaries used by Figure 8 are recorded in
`data/literature_validation_summary.csv`. These are analysis outputs, not copies
of the source publications. Users must obtain the source papers from their
publishers to reconstruct the original tables independently.
