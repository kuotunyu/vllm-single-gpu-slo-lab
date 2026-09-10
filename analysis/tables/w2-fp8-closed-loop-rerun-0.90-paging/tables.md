# Closed-loop

| cell | seed | concurrency | records | window_records | window_s | achieved_rps | output_tok_per_s | ttft_p50_s | ttft_p95_s | tpot_p50_s | tpot_p95_s | attainment | power_mean_w | tok_per_wh | mean_util_pct | w_per_util_point | probe_tpot_median_s | windows_committed_mb | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8 | 1 | 1 | 90 | 68 | 202.6 | 0.3357 | 44.3 | 0.02617 | 0.03425 | 0.02295 | 0.0248 | 1 | 205.1 | 700.3 | 97.3 | 2.11 | 0.01922 |  | False |
| fp8 | 1 | 2 | 200 | 156 | 236.8 | 0.6587 | 86.95 | 0.05279 | 0.08712 | 0.01761 | 0.02813 | 0.9872 | 200.6 | 1508 | 97.8 | 2.05 | 0.01872 |  | False |

{
  "fp8": {
    "r_sat_rps": 0.658658623736447,
    "r_sat_is_lower_bound": true,
    "plateau_reached": false,
    "plateau_first_concurrency": 2,
    "capacity_C": 2,
    "suspect_concurrencies_excluded": [],
    "mean_rps_by_concurrency": {
      "1": 0.3356677191027079,
      "2": 0.658658623736447
    },
    "gain_vs_prev_by_concurrency": {
      "2": 0.9622340375688916
    },
    "min_attainment_by_concurrency": {
      "1": 1.0,
      "2": 0.9871794871794872
    }
  }
}

# Open-loop

| cell | seed | offered_rps | records | achieved_rps | served_rps | ttft_p50_s | ttft_p95_s | tpot_p50_s | tpot_p95_s | attainment | attainment_ci95 | goodput_rps | w_per_util_point | probe_tpot_median_s | suspect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

{}
