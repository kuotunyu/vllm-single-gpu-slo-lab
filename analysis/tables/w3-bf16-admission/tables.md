## Admission trace: one row per (cell, policy, seed)

| cell | policy | seed | offered | attainment | attainment_pre | attainment_burst | attainment_recovery | goodput_rps | rejection_rate | rejection_burst | ttft_p95_burst_s | tpot_p95_burst_s | time_to_recover_s | time_to_recover_attainment_s | shim_cpu_util | probe_tpot_median_s | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | passthrough | 1 | 11937 | 0.4717 | 1 | 0.006463 | 0.7609 | 3.754 | 0 | 0 | 107 | 0.02423 | 215 | 210 | 0.0315 | 0.01857 | False |
| bf16 | passthrough | 2 | 12113 | 0.4807 | 0.9993 | 0.01311 | 0.7653 | 3.882 | 0 | 0 | 104.3 | 0.02402 | 210 | 205 | 0.0337 | 0.01858 | False |
| bf16 | passthrough | 3 | 11894 | 0.4598 | 0.9993 | 0.01506 | 0.7337 | 3.646 | 0 | 0 | 128.4 | 0.02566 | 250 | 245 | 0.0363 | 0.01962 | False |
| bf16 | hard_cap | 1 | 11937 | 0.8691 | 0.9985 | 0.6955 | 0.9992 | 6.917 | 0.1296 | 0.3029 | 0.09117 | 0.02401 | 0 | 0 | 0.028 | 0.01895 | False |
| bf16 | hard_cap | 2 | 12113 | 0.8713 | 0.9993 | 0.6972 | 0.9981 | 7.036 | 0.1274 | 0.3018 | 0.08843 | 0.02399 | 0 | 0 | 0.0271 | 0.01858 | False |
| bf16 | hard_cap | 3 | 11894 | 0.8598 | 0.9993 | 0.6797 | 0.9988 | 6.818 | 0.1393 | 0.3197 | 0.09113 | 0.02486 | 0 | 0 | 0.0311 | 0.01873 | False |
| bf16 | bounded_queue | 1 | 11937 | 0.8593 | 0.9993 | 0.6732 | 0.9983 | 6.839 | 0.1178 | 0.2753 | 1.018 | 0.02434 | 5 | 0 | 0.0284 | 0.01863 | False |
| bf16 | bounded_queue | 2 | 12113 | 0.8611 | 0.9993 | 0.6731 | 0.9981 | 6.954 | 0.1157 | 0.2741 | 1.018 | 0.02428 | 5 | 0 | 0.0285 | 0.01857 | False |
| bf16 | bounded_queue | 3 | 11894 | 0.8481 | 0.9964 | 0.6548 | 0.9974 | 6.725 | 0.1273 | 0.2921 | 1.02 | 0.02507 | 10 | 0 | 0.0298 | 0.01931 | False |


## bf16: mean over seeds (per-seed values in brackets)

| policy | n | attainment | goodput_rps | rejection_rate | attainment_burst | attainment_recovery | ttft_p95_burst_s | time_to_recover_s | time_to_recover_attainment_s |
|---|---|---|---|---|---|---|---|---|---|
| passthrough | 3 | 0.4708 (0.4717/0.4807/0.4598) | 3.761 (3.754/3.882/3.646) | 0 (0/0/0) | 0.01154 (0.006463/0.01311/0.01506) | 0.7533 (0.7609/0.7653/0.7337) | 113.2 (107/104.3/128.4) | 225 (215/210/250) | 220 (210/205/245) |
| hard_cap | 3 | 0.8668 (0.8691/0.8713/0.8598) | 6.924 (6.917/7.036/6.818) | 0.1321 (0.1296/0.1274/0.1393) | 0.6908 (0.6955/0.6972/0.6797) | 0.9987 (0.9992/0.9981/0.9988) | 0.09024 (0.09117/0.08843/0.09113) | 0 (0/0/0) | 0 (0/0/0) |
| bounded_queue | 3 | 0.8562 (0.8593/0.8611/0.8481) | 6.839 (6.839/6.954/6.725) | 0.1202 (0.1178/0.1157/0.1273) | 0.667 (0.6732/0.6731/0.6548) | 0.9979 (0.9983/0.9981/0.9974) | 1.019 (1.018/1.018/1.02) | 6.667 (5/5/10) | 0 (0/0/0) |

## bf16: paired difference vs passthrough (same trace per seed)

| policy | metric | per-seed difference | mean | same sign in every seed |
|---|---|---|---|---|
| hard_cap | attainment | 0.3974/0.3906/0.4 | 0.396 | yes |
| hard_cap | goodput_rps | 3.163/3.154/3.172 | 3.163 | yes |
| hard_cap | rejection_rate | 0.1296/0.1274/0.1393 | 0.1321 | yes |
| hard_cap | attainment_burst | 0.6891/0.6841/0.6647 | 0.6793 | yes |
| hard_cap | attainment_recovery | 0.2383/0.2328/0.2651 | 0.2454 | yes |
| hard_cap | ttft_p95_burst_s | -106.9/-104.2/-128.3 | -113.1 | yes |
| hard_cap | time_to_recover_s | -215/-210/-250 | -225 | yes |
| hard_cap | time_to_recover_attainment_s | -210/-205/-245 | -220 | yes |
| bounded_queue | attainment | 0.3876/0.3804/0.3883 | 0.3854 | yes |
| bounded_queue | goodput_rps | 3.085/3.072/3.079 | 3.078 | yes |
| bounded_queue | rejection_rate | 0.1178/0.1157/0.1273 | 0.1202 | yes |
| bounded_queue | attainment_burst | 0.6667/0.66/0.6398 | 0.6555 | yes |
| bounded_queue | attainment_recovery | 0.2373/0.2328/0.2636 | 0.2446 | yes |
| bounded_queue | ttft_p95_burst_s | -106/-103.3/-127.4 | -112.2 | yes |
| bounded_queue | time_to_recover_s | -210/-205/-240 | -218.3 | yes |
| bounded_queue | time_to_recover_attainment_s | -210/-205/-245 | -220 | yes |
