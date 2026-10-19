# Trend Lab Prototype

Run from `DTL/`:

```sh
python -m trend_lab.run --config trend_lab/configs/synthetic.json --smoke
```

Runtime dependency bounds are in [`requirements.txt`](requirements.txt). In an environment where installation is authorized, install them with `python -m pip install -r trend_lab/requirements.txt`; the runner itself does not install packages or fetch data.

Omit `--smoke` for the configured five-seed experiment. The checked-in config expands five synthetic scenarios across two frozen-base backbones. A non-empty output directory is rejected; pass a fresh `--output` directory for each run. For a local CSV, fill in [csv.template.json](configs/csv.template.json) with an existing path, explicit numeric `value_columns`, and optionally a strictly increasing `time_column`. CSV smoke uses the data loader's prefix truncation; it does not require synthetic-only `length` or `seed` fields. No CSV is created or downloaded.

Artifacts include the resolved config, JSON metrics and separate validation alpha grids, compressed predictions/probabilities with history, truth, and test origins, and reloadable PyTorch checkpoints. Temperature is selected on calibration NLL and early stopping monitors sample-weighted calibration loss. Projection strength is selected independently for the multi-scale and single-scale decoders using mean path accuracy, requiring a strict accuracy improvement, macro-F1 no lower than base by more than `1e-12`, and validation MSE at most `1.01 * base`; ties prefer lower MSE, then lower alpha. Test is used only for final reporting. Data seed is independent from model training seeds. Timings synchronize CUDA when used. Adapter `inference_seconds` is incremental and excludes the shared base forward, reported separately as `shared_base_inference_seconds`; projection inference includes the direction-head forward and decoding. Projection `fit_seconds` attributes the shared direction-head fit so it is not mistaken for a training-free method.

Checkpoint inference is available through `load_checkpoint(path, seed)` and `predict_from_checkpoint(record, standardized_history, method)` in `trend_lab.run`. The prediction function expects history already standardized with the saved train-only mean and standard deviation and returns a standardized path (or path/probabilities for `direction_head`).

`residual_direction` is an ordinary residual adapter trained with MSE plus a weighted multi-scale direction cross-entropy auxiliary term (default weight `0.1`); it is a fairness control, not a proposed new loss. `difference` is only a one-step-difference MSE baseline. Event and event-free slope adapters are lightweight mechanism prototypes.

This is runnable prototype code, not a full reproduction of the paper's methods or a complete literature-strength evaluation. In particular, PIR, GARS, CORTEX, FreDF, PS, and DILATE are not implemented. Synthetic results are pipeline checks and cannot establish real-data gains.

## ETTh1 GPU pilot

`configs/etth1_pilot.json` is a fixed first real-data pilot. From `DTL/`, run:

```sh
bash trend_lab/scripts/run_etth1_pilot.sh
```

The config uses all seven ETTh1 variables as both history inputs and prediction targets, a joint linear backbone, `L=96`, `H=24`, fixed scales `[1,2,4,8]`, three training seeds, at most 40 epochs with patience 8, training stride 4, and non-overlapping evaluation stride 24. Its chronological 60/15/10/15 split is an explicitly nonstandard pilot protocol; it is not the canonical ETTh benchmark split. Forecast targets stay within their assigned segment, while each history may include already observed values from earlier segments. Standardization and direction thresholds use training data only. Validation selects projection strengths; test results are final reporting only.

The runner script uses `/home/lgc/.conda/envs/TFB/bin/python` and visible GPU 1 by default. Override these with `TSLIB_TREND_PYTHON`, `TSLIB_TREND_CUDA_DEVICE`, or `TSLIB_TREND_RUNS_DIR`. To run the same pilot protocol on ETTh2, use `TSLIB_TREND_DATASET=ETTh2 bash trend_lab/scripts/run_etth1_pilot.sh`; this selects `configs/etth2_pilot.json`. The script runs the existing unit suite, a bounded-prefix CUDA smoke, then the fixed pilot. Each unique run directory stores the source snapshot and SHA256 manifest, the exact config, source CSV hash, environment/GPU details, logs, resolved smoke/pilot configs, JSON results and validation alpha tables, compressed predictions, and checkpoints. Do not treat the smoke metrics as research evidence.

Runner progress messages report the best checkpoint epoch restored by early stopping, not the total epochs traversed before patience expired.
