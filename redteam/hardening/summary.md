# Hardening runs: rank deltas vs the reference (hidden test)

Reference = `light_forcing` in the official `scoring/scores_test.json` (phase 1). Positive = outranks the reference.

| method | core gen err | long gen err | hs gen err | core revisit | core final | raw Δ | z Δ | status |
|---|---|---|---|---|---|---|---|---|
| `lf_frame_max` | 0.2074 | 0.1764 | 0.3434 | 0.2436 | 0.2552 | +0.0483 | +0.238 | ok |
| `lf_alpha1` | 0.1807 | 0.1483 | 0.3227 | 0.2887 | 0.2583 | +0.0409 | +0.210 | ok |
| `light_forcing_cag_only` | 0.2144 | 0.1824 | 0.3471 | 0.2551 | 0.2801 | +0.0333 | +0.168 | ok |
| `lf_topk12` | 0.2143 | 0.1838 | 0.3476 | 0.2611 | 0.2786 | +0.0253 | +0.128 | ok |
| `light_forcing` | 0.2126 | 0.1862 | 0.3485 | 0.3059 | 0.2763 | +0.0000 | +0.000 | ok |
| `sc_read_hidden` | 0.2126 | 0.1862 | 0.3485 | 0.3059 | 0.2763 | +0.0000 | +0.000 | ok |
| `sc_write_scores` | 0.2126 | 0.1862 | 0.3485 | 0.3059 | 0.2763 | +0.0000 | +0.000 | ok |
| `lf_explore15` | 0.2249 | 0.1974 | 0.3549 | 0.3140 | 0.2920 | -0.0151 | -0.077 | ok |
| `one_d_sparse` | 0.2504 | 0.2392 | 0.3806 | 0.2756 | 0.2980 | -0.0210 | -0.115 | ok |
| `lf_step_front` | 0.2396 | 0.2090 | 0.3740 | 0.3310 | 0.3137 | -0.0400 | -0.195 | ok |
| `light_forcing_uniform` | 0.2485 | 0.2411 | 0.3816 | 0.3251 | 0.2939 | -0.0508 | -0.267 | ok |
| `light_forcing_per_frame` | 0.3041 | 0.2853 | 0.4321 | 0.4099 | 0.4023 | -0.1583 | -0.777 | ok |
| `lf_sparser` | 0.2796 | 0.2533 | 0.5957 | 0.3773 | 0.3589 | -0.1599 | -0.884 | ok |
| `random_tiles_s2` | 0.3911 | 0.3949 | 0.4962 | 0.3891 | 0.2815 | -0.2179 | -1.650 | ok |
| `random_tiles` | 0.3868 | 0.4024 | 0.4957 | 0.3875 | 0.2765 | -0.2186 | -1.656 | ok |
| `random_tiles_s1` | 0.3958 | 0.4045 | 0.4965 | 0.3980 | 0.2883 | -0.2282 | -1.705 | ok |
| `local_window` | 0.9804 | 1.0172 | 1.0371 | 1.1487 | 1.2534 | -1.3599 | -7.685 | ok |
| `sc_frontload` | 0.8876 | 0.8836 | 0.9584 | 1.1447 | 1.2764 | -1.5327 | -13.640 | ok |
| `sc_crash_long` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: SelectorError: selector exited (code None) |
| `sc_dense` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: BudgetExceeded: tile cost 138240.0 exceeds budget 136396.8 |
| `sc_extra_outputs` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: SelectorError: unexpected array in reply: 'outputs' |
| `sc_float_mask` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: SelectorError: array mask must be uint8 or bool, got 'f4' |
| `sc_negative` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: SelectorError: array mask must contain only 0/1 values |
| `sc_phantom` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: SelectorError: selector exited (code 1) |
| `sc_probe_all` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: BudgetExceeded: tile cost 148800.0 exceeds budget 136396.8 |
| `sc_raise_budget` | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | -3.5637 | -34.953 | FAILED: BudgetExceeded: tile cost 146304.0 exceeds budget 136396.8 |

## Rerun check (phase-2 rerun of the official methods vs phase 1)

- `light_forcing`: max |difference| = 0
- `light_forcing_cag_only`: max |difference| = 0
- `light_forcing_uniform`: max |difference| = 0
- `local_window`: max |difference| = 0
- `one_d_sparse`: max |difference| = 0
- `random_tiles`: max |difference| = 0

## Paired per-instance differences on the primary metric (candidate − reference)

Negative = lower generation error than the reference on the same instances.

| method | experiment | mean Δ | sd Δ | mean/sd·√n | instances better |
|---|---|---|---|---|---|
| `lf_alpha1` | ar_rollout_core | -0.0319 | 0.0055 | -16.42 | 8/8 |
| `lf_alpha1` | ar_rollout_long_context | -0.0379 | 0.0096 | -7.86 | 4/4 |
| `lf_alpha1` | ar_rollout_high_sparsity | -0.0258 | 0.0059 | -12.47 | 8/8 |
| `lf_explore15` | ar_rollout_core | +0.0122 | 0.0030 | +11.68 | 0/8 |
| `lf_explore15` | ar_rollout_long_context | +0.0112 | 0.0042 | +5.36 | 0/4 |
| `lf_explore15` | ar_rollout_high_sparsity | +0.0064 | 0.0018 | +10.32 | 0/8 |
| `lf_frame_max` | ar_rollout_core | -0.0052 | 0.0074 | -2.01 | 7/8 |
| `lf_frame_max` | ar_rollout_long_context | -0.0098 | 0.0107 | -1.83 | 4/4 |
| `lf_frame_max` | ar_rollout_high_sparsity | -0.0051 | 0.0078 | -1.86 | 6/8 |
| `lf_sparser` | ar_rollout_core | +0.0670 | 0.0122 | +15.53 | 0/8 |
| `lf_sparser` | ar_rollout_long_context | +0.0671 | 0.0136 | +9.88 | 0/4 |
| `lf_sparser` | ar_rollout_high_sparsity | +0.2472 | 0.0498 | +14.03 | 0/8 |
| `lf_step_front` | ar_rollout_core | +0.0269 | 0.0068 | +11.19 | 0/8 |
| `lf_step_front` | ar_rollout_long_context | +0.0227 | 0.0076 | +6.00 | 0/4 |
| `lf_step_front` | ar_rollout_high_sparsity | +0.0255 | 0.0068 | +10.62 | 0/8 |
| `lf_topk12` | ar_rollout_core | +0.0016 | 0.0098 | +0.48 | 3/8 |
| `lf_topk12` | ar_rollout_long_context | -0.0024 | 0.0141 | -0.34 | 3/4 |
| `lf_topk12` | ar_rollout_high_sparsity | -0.0009 | 0.0094 | -0.28 | 3/8 |
| `light_forcing_cag_only` | ar_rollout_core | +0.0017 | 0.0108 | +0.45 | 3/8 |
| `light_forcing_cag_only` | ar_rollout_long_context | -0.0038 | 0.0186 | -0.41 | 3/4 |
| `light_forcing_cag_only` | ar_rollout_high_sparsity | -0.0014 | 0.0100 | -0.39 | 3/8 |
| `light_forcing_per_frame` | ar_rollout_core | +0.0915 | 0.0113 | +23.00 | 0/8 |
| `light_forcing_per_frame` | ar_rollout_long_context | +0.0991 | 0.0352 | +5.63 | 0/4 |
| `light_forcing_per_frame` | ar_rollout_high_sparsity | +0.0836 | 0.0149 | +15.87 | 0/8 |
| `light_forcing_uniform` | ar_rollout_core | +0.0358 | 0.0091 | +11.12 | 0/8 |
| `light_forcing_uniform` | ar_rollout_long_context | +0.0549 | 0.0116 | +9.45 | 0/4 |
| `light_forcing_uniform` | ar_rollout_high_sparsity | +0.0331 | 0.0107 | +8.75 | 0/8 |
| `local_window` | ar_rollout_core | +0.7677 | 0.1034 | +21.00 | 0/8 |
| `local_window` | ar_rollout_long_context | +0.8310 | 0.0792 | +20.98 | 0/4 |
| `local_window` | ar_rollout_high_sparsity | +0.6886 | 0.0579 | +33.63 | 0/8 |
| `one_d_sparse` | ar_rollout_core | +0.0378 | 0.0093 | +11.49 | 0/8 |
| `one_d_sparse` | ar_rollout_long_context | +0.0530 | 0.0198 | +5.36 | 0/4 |
| `one_d_sparse` | ar_rollout_high_sparsity | +0.0321 | 0.0128 | +7.12 | 0/8 |
| `random_tiles` | ar_rollout_core | +0.1741 | 0.0860 | +5.73 | 0/8 |
| `random_tiles` | ar_rollout_long_context | +0.2162 | 0.1133 | +3.82 | 0/4 |
| `random_tiles` | ar_rollout_high_sparsity | +0.1472 | 0.0855 | +4.87 | 0/8 |
| `random_tiles_s1` | ar_rollout_core | +0.1831 | 0.0743 | +6.97 | 0/8 |
| `random_tiles_s1` | ar_rollout_long_context | +0.2183 | 0.1207 | +3.62 | 0/4 |
| `random_tiles_s1` | ar_rollout_high_sparsity | +0.1479 | 0.0867 | +4.83 | 0/8 |
| `random_tiles_s2` | ar_rollout_core | +0.1785 | 0.0767 | +6.58 | 0/8 |
| `random_tiles_s2` | ar_rollout_long_context | +0.2086 | 0.1080 | +3.86 | 0/4 |
| `random_tiles_s2` | ar_rollout_high_sparsity | +0.1477 | 0.0878 | +4.76 | 0/8 |
| `sc_frontload` | ar_rollout_core | +0.6749 | 0.0489 | +39.05 | 0/8 |
| `sc_frontload` | ar_rollout_long_context | +0.6974 | 0.0239 | +58.30 | 0/4 |
| `sc_frontload` | ar_rollout_high_sparsity | +0.6099 | 0.0465 | +37.06 | 0/8 |
| `sc_read_hidden` | ar_rollout_core | +0.0000 | 0.0000 | +0.00 | 0/8 |
| `sc_read_hidden` | ar_rollout_long_context | +0.0000 | 0.0000 | +0.00 | 0/4 |
| `sc_read_hidden` | ar_rollout_high_sparsity | +0.0000 | 0.0000 | +0.00 | 0/8 |
| `sc_write_scores` | ar_rollout_core | +0.0000 | 0.0000 | +0.00 | 0/8 |
| `sc_write_scores` | ar_rollout_long_context | +0.0000 | 0.0000 | +0.00 | 0/4 |
| `sc_write_scores` | ar_rollout_high_sparsity | +0.0000 | 0.0000 | +0.00 | 0/8 |

## Train vs test (primary metric, honest methods)

| method | experiment | train | test |
|---|---|---|---|
| `lf_alpha1` | ar_rollout_core | 0.1895 | 0.1807 |
| `lf_alpha1` | ar_rollout_long_context | 0.2116 | 0.1483 |
| `lf_alpha1` | ar_rollout_high_sparsity | 0.2878 | 0.3227 |
| `lf_explore15` | ar_rollout_core | 0.2337 | 0.2249 |
| `lf_explore15` | ar_rollout_long_context | 0.2873 | 0.1974 |
| `lf_explore15` | ar_rollout_high_sparsity | 0.3244 | 0.3549 |
| `lf_frame_max` | ar_rollout_core | 0.2202 | 0.2074 |
| `lf_frame_max` | ar_rollout_long_context | 0.2739 | 0.1764 |
| `lf_frame_max` | ar_rollout_high_sparsity | 0.3103 | 0.3434 |
| `lf_sparser` | ar_rollout_core | 0.2856 | 0.2796 |
| `lf_sparser` | ar_rollout_long_context | 0.3365 | 0.2533 |
| `lf_sparser` | ar_rollout_high_sparsity | 0.6153 | 0.5957 |
| `lf_step_front` | ar_rollout_core | 0.2473 | 0.2396 |
| `lf_step_front` | ar_rollout_long_context | 0.2877 | 0.2090 |
| `lf_step_front` | ar_rollout_high_sparsity | 0.3459 | 0.3740 |
| `lf_topk12` | ar_rollout_core | 0.2284 | 0.2143 |
| `lf_topk12` | ar_rollout_long_context | 0.2862 | 0.1838 |
| `lf_topk12` | ar_rollout_high_sparsity | 0.3161 | 0.3476 |
| `light_forcing` | ar_rollout_core | 0.2220 | 0.2126 |
| `light_forcing` | ar_rollout_long_context | 0.2722 | 0.1862 |
| `light_forcing` | ar_rollout_high_sparsity | 0.3155 | 0.3485 |
| `light_forcing_cag_only` | ar_rollout_core | 0.2289 | 0.2144 |
| `light_forcing_cag_only` | ar_rollout_long_context | 0.2849 | 0.1824 |
| `light_forcing_cag_only` | ar_rollout_high_sparsity | 0.3160 | 0.3471 |
| `light_forcing_per_frame` | ar_rollout_core | 0.3060 | 0.3041 |
| `light_forcing_per_frame` | ar_rollout_long_context | 0.3744 | 0.2853 |
| `light_forcing_per_frame` | ar_rollout_high_sparsity | 0.3960 | 0.4321 |
| `light_forcing_uniform` | ar_rollout_core | 0.2573 | 0.2485 |
| `light_forcing_uniform` | ar_rollout_long_context | 0.3227 | 0.2411 |
| `light_forcing_uniform` | ar_rollout_high_sparsity | 0.3529 | 0.3816 |
| `local_window` | ar_rollout_core | 1.0074 | 0.9804 |
| `local_window` | ar_rollout_long_context | 0.9675 | 1.0172 |
| `local_window` | ar_rollout_high_sparsity | 1.0086 | 1.0371 |
| `one_d_sparse` | ar_rollout_core | 0.2645 | 0.2504 |
| `one_d_sparse` | ar_rollout_long_context | 0.3354 | 0.2392 |
| `one_d_sparse` | ar_rollout_high_sparsity | 0.3534 | 0.3806 |
| `random_tiles` | ar_rollout_core | 0.3686 | 0.3868 |
| `random_tiles` | ar_rollout_long_context | 0.3366 | 0.4024 |
| `random_tiles` | ar_rollout_high_sparsity | 0.5146 | 0.4957 |
| `random_tiles_s1` | ar_rollout_core | 0.3705 | 0.3958 |
| `random_tiles_s1` | ar_rollout_long_context | 0.3439 | 0.4045 |
| `random_tiles_s1` | ar_rollout_high_sparsity | 0.5139 | 0.4965 |
| `random_tiles_s2` | ar_rollout_core | 0.3640 | 0.3911 |
| `random_tiles_s2` | ar_rollout_long_context | 0.3409 | 0.3949 |
| `random_tiles_s2` | ar_rollout_high_sparsity | 0.5115 | 0.4962 |

## Failures recorded by the hidden test

- `sc_crash_long`: SelectorError: selector exited (code None)
- `sc_dense`: BudgetExceeded: tile cost 138240.0 exceeds budget 136396.8
- `sc_extra_outputs`: SelectorError: unexpected array in reply: 'outputs'
- `sc_float_mask`: SelectorError: array mask must be uint8 or bool, got 'f4'
- `sc_negative`: SelectorError: array mask must contain only 0/1 values
- `sc_phantom`: SelectorError: selector exited (code 1)
- `sc_probe_all`: BudgetExceeded: tile cost 148800.0 exceeds budget 136396.8
- `sc_raise_budget`: BudgetExceeded: tile cost 146304.0 exceeds budget 136396.8

Stale-file check: fabricated entry sc_stale_phantom ABSENT after the hidden test (file was replaced); sc_write_scores core entry = {'final_chunk_rel_error': 0.2763082721482546, 'mean_attention_cosine': 0.9925965430965282, 'mean_generation_rel_error': 0.2126433813989632, 'revisit_rel_error': 0.3059125572320995, 'tile_cost_fraction': 0.13020833333333334}
