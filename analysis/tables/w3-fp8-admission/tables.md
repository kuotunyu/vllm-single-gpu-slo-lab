## Admission trace: one row per (cell, policy, seed)

| cell | policy | seed | offered | attainment | attainment_pre | attainment_burst | attainment_recovery | goodput_rps | rejection_rate | rejection_burst | ttft_p95_burst_s | tpot_p95_burst_s | time_to_recover_s | time_to_recover_attainment_s | shim_cpu_util | probe_tpot_median_s | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8 | passthrough | 1 | 45624 | 0.1856 | 0.9981 | 0.002294 | 0.09774 | 5.644 | 0 | 0 | 270.9 | 0.05777 | 820 | 810 | 0.1028 | 0.01983 | False |
| fp8 | passthrough | 2 | 45874 | 0.1674 | 0.9977 | 0.001531 | 0.05586 | 5.119 | 0 | 0 | 274.4 | 0.05824 |  |  | 0.1034 | 0.01981 | False |
| fp8 | passthrough | 3 | 45733 | 0.2224 | 0.9985 | 0.000661 | 0.1819 | 6.781 | 0 | 0 | 258.3 | 0.05638 | 745 | 735 | 0.1036 | 0.0198 | False |
| fp8 | hard_cap | 1 | 45624 | 0.5734 | 0.999 | 0.009788 | 0.9984 | 17.44 | 0.2093 | 0.4867 | 0.266 | 0.05778 | 0 | 0 | 0.0787 | 0.02001 | False |
| fp8 | hard_cap | 2 | 45874 | 0.5743 | 0.9992 | 0.007142 | 0.997 | 17.56 | 0.2094 | 0.4894 | 0.2547 | 0.05787 | 0 | 0 | 0.0781 | 0.01985 | False |
| fp8 | hard_cap | 3 | 45733 | 0.5709 | 0.9955 | 0.005442 | 0.9978 | 17.41 | 0.2089 | 0.4854 | 0.2594 | 0.05765 | 0 | 0 | 0.0792 | 0.0198 | False |
| fp8 | bounded_queue | 1 | 45624 | 0.5717 | 0.9985 | 0.005965 | 0.9983 | 17.39 | 0.2072 | 0.482 | 1.119 | 0.05714 | 5 | 0 | 0.0797 | 0.02004 | False |
| fp8 | bounded_queue | 2 | 45874 | 0.6269 | 0.9979 | 0.1303 | 0.9972 | 19.17 | 0.1901 | 0.4439 | 1.147 | 0.05752 | 10 | 0 | 0.0811 | 0.01981 | False |
| fp8 | bounded_queue | 3 | 45733 | 0.5735 | 0.9962 | 0.01053 | 0.9986 | 17.48 | 0.1985 | 0.4617 | 1.119 | 0.05559 | 10 | 0 | 0.0795 | 0.01979 | False |


## fp8: mean over seeds (per-seed values in brackets)

| policy | n | attainment | goodput_rps | rejection_rate | attainment_burst | attainment_recovery | ttft_p95_burst_s | time_to_recover_s | time_to_recover_attainment_s |
|---|---|---|---|---|---|---|---|---|---|
| passthrough | 3 | 0.1918 (0.1856/0.1674/0.2224) | 5.848 (5.644/5.119/6.781) | 0 (0/0/0) | 0.001495 (0.002294/0.001531/0.000661) | 0.1118 (0.09774/0.05586/0.1819) | 267.9 (270.9/274.4/258.3) | 782.5 (820/n/a/745) | 772.5 (810/n/a/735) |
| hard_cap | 3 | 0.5729 (0.5734/0.5743/0.5709) | 17.47 (17.44/17.56/17.41) | 0.2092 (0.2093/0.2094/0.2089) | 0.007457 (0.009788/0.007142/0.005442) | 0.9977 (0.9984/0.997/0.9978) | 0.26 (0.266/0.2547/0.2594) | 0 (0/0/0) | 0 (0/0/0) |
| bounded_queue | 3 | 0.5907 (0.5717/0.6269/0.5735) | 18.01 (17.39/19.17/17.48) | 0.1986 (0.2072/0.1901/0.1985) | 0.04895 (0.005965/0.1303/0.01053) | 0.998 (0.9983/0.9972/0.9986) | 1.129 (1.119/1.147/1.119) | 8.333 (5/10/10) | 0 (0/0/0) |

## fp8: paired difference vs passthrough (same trace per seed)

| policy | metric | per-seed difference | mean | same sign in every seed |
|---|---|---|---|---|
| hard_cap | attainment | 0.3879/0.4069/0.3485 | 0.3811 | yes |
| hard_cap | goodput_rps | 11.8/12.45/10.63 | 11.62 | yes |
| hard_cap | rejection_rate | 0.2093/0.2094/0.2089 | 0.2092 | yes |
| hard_cap | attainment_burst | 0.007494/0.005611/0.004781 | 0.005962 | yes |
| hard_cap | attainment_recovery | 0.9006/0.9411/0.816 | 0.8859 | yes |
| hard_cap | ttft_p95_burst_s | -270.6/-274.1/-258.1 | -267.6 | yes |
| hard_cap | time_to_recover_s | -820/-745 | -782.5 | yes |
| hard_cap | time_to_recover_attainment_s | -810/-735 | -772.5 | yes |
| bounded_queue | attainment | 0.3861/0.4595/0.3511 | 0.3989 | yes |
| bounded_queue | goodput_rps | 11.74/14.05/10.7 | 12.17 | yes |
| bounded_queue | rejection_rate | 0.2072/0.1901/0.1985 | 0.1986 | yes |
| bounded_queue | attainment_burst | 0.003671/0.1288/0.009866 | 0.04745 | yes |
| bounded_queue | attainment_recovery | 0.9006/0.9414/0.8167 | 0.8862 | yes |
| bounded_queue | ttft_p95_burst_s | -269.7/-273.2/-257.2 | -266.7 | yes |
| bounded_queue | time_to_recover_s | -815/-735 | -775 | yes |
| bounded_queue | time_to_recover_attainment_s | -810/-735 | -772.5 | yes |
