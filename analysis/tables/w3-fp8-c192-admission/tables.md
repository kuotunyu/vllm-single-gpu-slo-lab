## Admission trace: one row per (cell, policy, seed)

| cell | policy | seed | offered | attainment | attainment_pre | attainment_burst | attainment_recovery | goodput_rps | rejection_rate | rejection_burst | ttft_p95_burst_s | tpot_p95_burst_s | time_to_recover_s | time_to_recover_attainment_s | shim_cpu_util | probe_tpot_median_s | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8-c192 | hard_cap | 1 | 45624 | 0.7845 | 0.9983 | 0.501 | 0.9985 | 23.86 | 0.2142 | 0.4982 | 0.2046 | 0.04426 | 0 | 0 | 0.0773 | 0.01987 | False |
| fp8-c192 | hard_cap | 2 | 45874 | 0.7804 | 0.999 | 0.4892 | 0.9974 | 23.87 | 0.217 | 0.507 | 0.2765 | 0.04494 | 0 | 0 | 0.0881 | 0.02009 | False |
| fp8-c192 | hard_cap | 3 | 45733 | 0.7763 | 0.9972 | 0.4828 | 0.9977 | 23.67 | 0.222 | 0.5163 | 0.206 | 0.0464 | 0 | 0 | 0.084 | 0.02007 | False |


## fp8-c192: mean over seeds (per-seed values in brackets)

| policy | n | attainment | goodput_rps | rejection_rate | attainment_burst | attainment_recovery | ttft_p95_burst_s | time_to_recover_s | time_to_recover_attainment_s |
|---|---|---|---|---|---|---|---|---|---|
| hard_cap | 3 | 0.7804 (0.7845/0.7804/0.7763) | 23.8 (23.86/23.87/23.67) | 0.2177 (0.2142/0.217/0.222) | 0.491 (0.501/0.4892/0.4828) | 0.9979 (0.9985/0.9974/0.9977) | 0.229 (0.2046/0.2765/0.206) | 0 (0/0/0) | 0 (0/0/0) |
