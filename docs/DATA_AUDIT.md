# Data audit — recording quality & noise triage

Source: `data/processed` · 789 recordings · thresholds flat>0.15, snr<9.0 dB, wind>0.15

## Per-class triage

| class | keep | denoise | harvest | drop | med snr_db | med flat | med wind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| american_crow | 77 | 5 | 0 | 0 | 23.2 | 0.05 | 0.00 |
| background | 100 | 0 | 0 | 0 | 7.5 | 0.04 | 0.12 |
| california_quail | 61 | 15 | 0 | 1 | 20.8 | 0.08 | 0.00 |
| california_scrub_jay | 51 | 12 | 0 | 0 | 23.7 | 0.10 | 0.00 |
| great_horned_owl | 58 | 10 | 0 | 0 | 25.4 | 0.01 | 0.00 |
| killdeer | 117 | 14 | 0 | 0 | 26.1 | 0.03 | 0.00 |
| mourning_dove | 86 | 2 | 0 | 0 | 24.1 | 0.02 | 0.00 |
| red_tailed_hawk | 79 | 7 | 1 | 0 | 25.7 | 0.06 | 0.00 |
| western_meadowlark | 85 | 8 | 0 | 0 | 32.7 | 0.03 | 0.00 |

## Worst 10 recordings (lowest dynamic range)

| group | class | triage | snr_db | flat | wind | noise win |
| --- | --- | --- | --- | --- | --- | --- |
| background_0005 | background | keep | -3.0 | 0.93 | 0.01 | 2/3 |
| background_0007 | background | keep | -3.0 | 1.00 | 0.00 | 2/3 |
| background_0009 | background | keep | -3.0 | 1.00 | 0.00 | 2/3 |
| background_0095 | background | keep | 0.9 | 0.37 | 0.38 | 3/3 |
| background_0098 | background | keep | 1.0 | 0.52 | 0.11 | 3/3 |
| background_0076 | background | keep | 1.0 | 0.12 | 0.00 | 0/3 |
| background_0096 | background | keep | 1.0 | 0.52 | 0.12 | 3/3 |
| background_0097 | background | keep | 1.0 | 0.49 | 0.17 | 3/3 |
| background_0099 | background | keep | 1.3 | 0.38 | 0.36 | 3/3 |
| background_0077 | background | keep | 2.1 | 0.10 | 0.01 | 0/3 |
