# InterruptBench

A benchmark for evaluating web agents under **mid-task interruptions and intent updates**, built on WebArena. An agent first acts on an initial (under-specified) intent; the run is then interrupted at a chosen point and the user injects updates (e.g., clarifications, modifications, retractions), after which the agent must adapt and complete the true task.

## Repository Structure

```
InterruptBench/
├── WebArena-Env-Setup/   # Docker setup for the WebArena websites + reset server
└── Eval/                 # Evaluation harness (agent, run.py, parallel scheduler)
    └── interrupt_config/ # Interrupt data (raw/) and generation scripts
```

## 0) Environment Setup

Set up the WebArena websites (docker) and the reset server:

- [`WebArena-Env-Setup/`](WebArena-Env-Setup/) — follow its `README.md`

## 1) Data

Raw interrupt data lives in [`Eval/interrupt_config/raw/`](Eval/interrupt_config/raw/). Each JSON file is a list of items with `task_id`, `transformed_initial_intent`, and `updates`, covering different interruption types:

| File | Interruption type |
|---|---|
| `1update.json` | 1 intent update |
| `2update.json` | 2 intent updates |
| `2modification.json` | 2 modifications |
| `1retraction.json` / `2retraction.json` | 1 / 2 retractions |
| `3mixed.json` | 3 mixed updates |

## 2) Evaluation

### Single-interrupt (parallel) evaluation

End-to-end guide: environment setup, reset server, parallel scheduler (`scripts/parallel_by_sites.py`), and the two-stage baseline → replay+interrupt workflow.

- English: [`Eval/RUN_PARALLEL_EVAL.md`](Eval/RUN_PARALLEL_EVAL.md)
- 中文: [`Eval/RUN_PARALLEL_EVAL_ZH.md`](Eval/RUN_PARALLEL_EVAL_ZH.md)

### Multi-interrupt (multi-round intent update) evaluation

Extends the single-interrupt workflow to multiple interrupts, chained stage by stage (Stage 0 baseline → Stage 1 → Stage 2 → ...).

- English: [`Eval/interrupt_config/MULTI_INTERRUPT.md`](Eval/interrupt_config/MULTI_INTERRUPT.md)
- 中文: [`Eval/interrupt_config/MULTI_INTERRUPT_ZH.md`](Eval/interrupt_config/MULTI_INTERRUPT_ZH.md)
