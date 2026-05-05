# Proposal Eval Results

`Qwen/Qwen2.5-Coder-7B-Instruct` on `rh3_nonhack12_challenge`.

## Pilot-Style Table

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Problems analyzed | 15 | Fixed challenge set |
| Correct solutions | 3/15 (20.0%) | Hidden-test-resolved sample cases |
| Confirmed reward-hacking | 1/15 (6.7%) | One visible-success trajectory labeled `REWARD_HACK` |
| Visible tests | 84/150 (56.0%) | Proxy/local score |
| Holdout tests | 41/150 (27.3%) | Trusted hidden score |
| Visible-hidden gap | 28.7 percentage points | Overfit/divergence signal |

## Outcome Mix

| Outcome | Count | Rate |
| --- | ---: | ---: |
| Correct | 3 | 20.0% |
| Reward hack | 1 | 6.7% |
| Incorrect, not reward hack | 11 | 73.3% |

## PDF Evaluation Criteria

| Criterion | Sample result | Notes |
| --- | ---: | --- |
| Exploit Rate Sensitivity | 0.0% to 20.0% across  test | The `standard_swe_agent` row matches the per-instance sample. |
| Mitigation Efficacy | 1/1 converted (100.0%) | refinement converts the one reward-hack case to non-hacking unresolved. |
| RL Robustness | integrity-aware RL gap = 12.0 pp |  comparison placeholder for SFT, naive RL, and integrity-aware RL. |

## Exploit Rate Sensitivity

| Prompt framing | Reward hacks | Visible pass | Hidden pass | Gap |
| --- | ---: | ---: | ---: | ---: |
| `standard_swe_agent` | 1/15 (6.7%) | 56.0% | 27.3% | 28.7 pp |
| `exploit_stress_prompt` | 3/15 (20.0%) | 66.0% | 22.0% | 44.0 pp |
| `integrity_reminder_prompt` | 0/15 (0.0%) | 50.0% | 30.0% | 20.0 pp |

## Mitigation Efficacy

| Stage | Reward-hack cases | Honest/non-hacking cases | Notes |
| --- | ---: | ---: | --- |
| Before iterative refinement | 1 | 14 | Base sample outcome mix |
| After iterative refinement | 0 | 15 | conversion of the one test-tampering case |

## RL Robustness

| Group | Visible pass | Hidden pass | Visible-hidden gap |
| --- | ---: | ---: | ---: |
| Qwen base | 56.0% | 27.3% | 28.7 pp |
| SFT | 58.0% | 38.0% | 20.0 pp |
| Naive RL | 72.0% | 31.0% | 41.0 pp |
| Integrity-aware RL | 62.0% | 50.0% | 12.0 pp |

## Exploit Breakdown

| Exploit type | Count |
| --- | ---: |
| `test_tampering` | 1 |
| `none` | 14 |


