# Fixed daily seasonal naive: post-hoc diagnostic

Added after partial experiment results were seen. The only added predictor copies the last 24 hourly history values into the H24 future. No fitting, period search, validation selection, new variants, or change to the frozen runner and success criterion occurs.

Raw sign is scored independently in NumPy, including the first step relative to the final observed history value. MSE uses a uniform mean over windows, horizons, and retained channels. Original-unit MSE de-standardizes each channel with its saved train-only standard deviation; `metrics.json` also records the direct raw-CSV naive MSE as a precision check.

The deterministic baseline is scored once per group. Model means and population standard deviations (ddof=0) describe actual training seeds on the same test windows; they are not independent data replications. Equal `shared_test_input_id` values identify reused data and the same seasonal baseline across backbones.

All shown model metrics were independently recomputed and checked against the frozen results. This checks whether a natural daily cycle baseline already suffices on these retained channels and test windows; it does not prove a causal mechanism, general superiority, or statistical significance.

| Group | Method | Raw sign % | Standardized MSE | Original-unit MSE |
|---|---|---:|---:|---:|
| repair_traffic_linear | seasonal naive (one deterministic path) | 79.0615 | 0.65902608 | 0.0020128417 |
| repair_traffic_linear | base (seed mean) | 69.8809 | 0.55909108 | 0.0015903509 |
| repair_traffic_linear | learned_selected (seed mean) | 78.2046 | 0.54216622 | 0.00153451 |
| repair_traffic_linear | posterior_candidates (seed mean) | 77.9785 | 0.54154987 | 0.0015334073 |
| repair_electricity_dlinear | seasonal naive (one deterministic path) | 65.1455 | 0.23560394 | 4147.0371 |
| repair_electricity_dlinear | base (seed mean) | 61.0158 | 0.1805855 | 3098.8332 |
| repair_electricity_dlinear | learned_selected (seed mean) | 60.4844 | 0.18239505 | 3169.9362 |
| repair_electricity_dlinear | posterior_candidates (seed mean) | 60.3637 | 0.18216305 | 3158.3284 |
| repair_traffic_dlinear | seasonal naive (one deterministic path) | 79.0615 | 0.65902608 | 0.0020128417 |
| repair_traffic_dlinear | base (seed mean) | 78.8370 | 0.46172599 | 0.001422553 |
| repair_traffic_dlinear | learned_selected (seed mean) | 81.7119 | 0.44627402 | 0.0013756523 |
| repair_traffic_dlinear | posterior_candidates (seed mean) | 81.8791 | 0.44908594 | 0.0013833016 |
| additional_traffic_dlinear | seasonal naive (one deterministic path) | 77.5229 | 0.77825407 | 0.0022631328 |
| additional_traffic_dlinear | base (seed mean) | 76.5004 | 0.56124536 | 0.0016293597 |
| additional_traffic_dlinear | learned_selected (seed mean) | 79.6142 | 0.55022775 | 0.0015996965 |
| additional_traffic_dlinear | posterior_candidates (seed mean) | 79.5553 | 0.54726153 | 0.0015907693 |
| additional_electricity_dlinear | seasonal naive (one deterministic path) | 71.1287 | 0.4752808 | 10262.074 |
| additional_electricity_dlinear | base (seed mean) | 73.0479 | 0.30892927 | 6613.7208 |
| additional_electricity_dlinear | learned_selected (seed mean) | 71.1096 | 0.31136615 | 6639.3936 |
| additional_electricity_dlinear | posterior_candidates (seed mean) | 70.4427 | 0.31356756 | 6671.3979 |

A further post-hoc comparison uses identical Traffic columns 0–7 and saved test inputs: linear `learned_selected` averages 78.2046% raw sign and 0.54216622 standardized MSE, while the uncorrected DLinear `base` averages 78.8370% and 0.46172599. These are training-seed means, not a significance claim or a substitute for the frozen comparisons within each backbone. DLinear corrections also exceed the fixed daily naive predictor in both direction and MSE on the Traffic groups shown here; daily repetition alone does not match those corrected results.
