# Reward signal audit

| Controller | Term | Episode mean ± std | Nonzero decisions | Min / max | Absolute contribution |
|---|---|---:|---:|---:|---:|
| random | time | -0.1533 ± 0.0512 | 100.000% | -0.180 / -0.011 | 9.0% |
| random | collision | -0.3000 ± 0.5196 | 0.814% | -1.200 / 0.000 | 17.6% |
| random | failure | -1.2500 ± 2.1651 | 0.326% | -5.000 / 0.000 | 73.4% |
| random | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| random | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | time | -0.1800 ± 0.0000 | 100.000% | -0.180 / -0.180 | 100.0% |
| untrained | collision | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | failure | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| untrained | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | time | -0.0640 ± 0.0453 | 100.000% | -0.180 / -0.018 | 0.8% |
| oracle | collision | -0.2000 ± 0.4243 | 1.351% | -1.200 / 0.000 | 2.4% |
| oracle | failure | -0.6250 ± 1.6536 | 0.386% | -5.000 / 0.000 | 7.5% |
| oracle | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| oracle | success | 7.5000 ± 4.3301 | 2.317% | 0.000 / 10.000 | 89.4% |
| trained | time | -0.1800 ± 0.0000 | 100.000% | -0.180 / -0.180 | 100.0% |
| trained | collision | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | failure | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | correct_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | wrong_button | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | door_crossing | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | acquisition | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |
| trained | success | 0.0000 ± 0.0000 | 0.000% | 0.000 / 0.000 | 0.0% |

The only positive Approach term is terminal success. An absent success signal cannot distinguish useful movement from standing still.
Absolute contribution uses the sum of absolute episode-term totals; signed net-return fractions are separately in JSON.
