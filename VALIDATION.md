# Data and file checks

- The original TensorBoard event files and saved run configurations are included. Their SHA-256 checksums are recorded in `analysis/source_sha256.json`.
- Each of the six extracted metrics has a finite value at every expected iteration: 300 points for baseline, 300 for wide-300, and 900 for wide-900. The exported CSV contains 9,000 observations, with no missing or duplicate iteration indices.
- Success, base-contact and time-out indicators lie within [0, 1].
- Configuration comparisons show only the intended command-range and training-budget changes, plus run labels and log paths. All three runs use seed 42 and 4,096 environments.
- Reported late-window values are arithmetic means of the final 50 logged points. They agree with `analysis/scalars.csv` and `analysis/summary.json`.
- Figures show raw observations and trailing 20-iteration means. Shorter runs stop at iteration 299; no values are extrapolated.
- All three MP4 recordings passed complete decoding checks.
- All three final checkpoint files are included. Their bytes match the source checksum list, and their PyTorch ZIP containers passed integrity checks. These checks do not constitute an independent policy evaluation.
- The required-file check and archive integrity check passed. `submission_sha256.txt` records checksums for the required submission files.

The comparison uses training logs from one seed per condition. It does not establish performance on a physical robot or provide an independent held-out evaluation. Log-point averages are not pooled percentages of all completed episodes.
