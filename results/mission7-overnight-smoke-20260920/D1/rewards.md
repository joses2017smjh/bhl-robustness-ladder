# Reward signal audit

| Controller | Term | Episode mean ± std | Nonzero decisions | Min / max | Absolute contribution |
|---|---|---:|---:|---:|---:|
| random | time | -0.0040 ± 0.0000 | 100.000% | -0.004 / -0.004 | 100.0% |
| random | collision | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | failure | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | time | -0.0040 ± 0.0000 | 100.000% | -0.004 / -0.004 | 100.0% |
| untrained | collision | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | failure | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | time | -0.0040 ± 0.0000 | 100.000% | -0.004 / -0.004 | 100.0% |
| oracle | collision | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | failure | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | time | -0.0040 ± 0.0000 | 100.000% | -0.004 / -0.004 | 100.0% |
| trained | collision | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | failure | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |

The only positive Approach term is terminal success. An absent success signal cannot distinguish useful movement from standing still.
Absolute contribution uses the sum of absolute episode-term totals; signed net-return fractions are separately in JSON.
