---
name: quantumedge-phase3
description: Reproduce and verify the QuantumEdge GIC 2026 Phase 3 financial-volatility QRC workflow.
---

# QuantumEdge Phase 3 Skill

## Purpose

Run the QuantumEdge Phase 3 notebooks in the required order, retain generated outputs, and stop when strict package verification reports a failure.

## Preconditions

- Execute from the repository root.
- Public data files must exist under `data/`.
- Retained IBM hardware evidence must exist under `results/`.
- Do not expose or persist IBM credentials.
- The public GitHub URL must replace the README placeholder before final verification.

## Execution

```bash
python qbraid_skill/quantumedge_phase3/scripts/run_quantumedge.py
```

The skill executes:

1. Financial benchmark and execution studies.
2. Canonical MNIST supplementary benchmark.
3. Final figures.
4. Strict package verification.

## Success condition

`results/package_verification_summary.json` must report zero failures.

## Safety and integrity

- Do not fabricate missing data, hardware observables, job identifiers, or benchmark results.
- Do not silently replace canonical MNIST with the sklearn digits smoke test.
- Do not rerun IBM jobs when complete retained evidence is available.
- Report negative or statistically inconclusive results without altering them.
