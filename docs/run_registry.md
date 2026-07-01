# Run Registry

This registry records consequential corpus, training, and evaluation runs. A run is accepted only when its declared gate passes. Missing historical job IDs are recorded as unknown rather than reconstructed from filenames.

## Foundation Runs

| Run | Job | Input | Data/config | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| 350M dress rehearsal | unknown | Random initialization | `configs/urdu_dress_rehearsal.yaml` | Completed 685M-token engineering rehearsal | Engineering validation only; not part of the accepted 700M checkpoint lineage |
| 700M pilot | unknown | Random initialization | `configs/urdu_700m_pilot.yaml` | Completed 685M-token base run | Accepted as first checkpoint in the clean-base lineage |
| Expanded 700M v1 | unknown | 700M pilot | `configs/urdu_700m_expanded_v1.yaml` | Completed 2.43B-token expanded run | Accepted as predecessor to clean continuation |
| Clean continuation v1 | unknown | Expanded v1 checkpoint | `configs/urdu_700m_clean_continue_v1.yaml` | Completed 453M-token strict-clean continuation | Accepted foundation baseline: `step_000432.pt` |

## SFT Runs

| Run | Job | Input | Data/config | Result | Decision |
| --- | --- | --- | --- | --- | --- |
| Seed SFT smoke | `19600724` | Clean base | 30 seed examples | Training loss `0.0051`; validation loss `3.8874` | Rejected for overfitting; engineering path validated |
| SFT v1 training | `19617875` | Clean base | 5,412 examples, `configs/urdu_700m_sft_v1.yaml` | Best validation loss `2.0531` at step 150 | Rejected after held-out failure |
| SFT v1 sampled evaluation | `19618488` | SFT v1 | Held-out prompts | Failed effectively 12 of 12 | Rejected |
| SFT v1 greedy evaluation | `19618720` | SFT v1 | Held-out prompts | Failed effectively 12 of 12 | Rejected |
| First balanced corpus candidate | `19619224` | Source compiler | 1,150 candidate records | Review found malformed arithmetic, mistranslation, and task mismatch | Rejected before training |
| Task-first v2 compilation | `19620960` | Local curated generators | 630 records | Corpus gates passed; 570 train and 60 validation | Accepted for diagnostic only |
| SFT v2 preflight | `19623927` | Clean base | `configs/urdu_700m_sft_balanced_v2.yaml` | Inputs and model load passed | Accepted |
| SFT v2 one-node attempt | `19624999` | Clean base | Eight-rank FSDP | GPU out of memory | Failed; topology rejected |
| SFT v2 four-node attempt | `19639573` | Clean base | 32-rank FSDP | Host-memory OOM before training | Failed; fixed with full node RAM and memory-mapped loading |
| SFT v2 diagnostic | `19643795` | Clean base | 630 records, 36 steps | Validation loss reached `0.5144`; final `0.5235` | Rejected after held-out evaluation |
| SFT v2 sampled generation | `19644886` | `step_000036.pt` | 12 held-out prompts | Low repetition but broad task failure | Rejected |
| SFT v2 sampled scoring | `19644967` | Sampled generations | Generic scorer | Prompt copying flagged in 8 of 12; math unscored | Diagnostic only |
| SFT v2 greedy generation | `19644964` | `step_000036.pt` | 12 held-out prompts | Prompt copying and repeated phrases | Rejected |
| SFT v2 greedy scoring | `19644966` | Greedy generations | Generic scorer | Repetition `0.1645`; prompt copying in 9 of 12 | Diagnostic only |

## Required Fields For New Runs

Add a row before submitting a consequential run and complete it after evaluation. Record:

- Hypothesis and acceptance gate
- Git commit and baseline lock ID
- Input checkpoint
- Corpus or benchmark manifest
- Configuration path
- Slurm job IDs
- Key metrics by domain or task
- Final decision: accepted, rejected, or engineering-only

No run is promoted from validation loss alone.
