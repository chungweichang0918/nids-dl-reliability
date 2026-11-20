# nids-dl-reliability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24391814.svg)](https://doi.org/10.5281/zenodo.24391814)

Code and raw results for the paper

> **How Reliable Are Deep-Learning Comparisons for Network Intrusion Detection? A Reproducibility and Evaluation-Pitfall Study on Four Benchmarks** (submitted to IEEE Access)

The study reproduces two published comparisons of deep intrusion detectors. It re-evaluates their models (DNN, CNN, SimpleRNN, LSTM, CNN-LSTM, GRU) together with the FT-Transformer, random forest, and XGBoost on NSL-KDD, UNSW-NB15, CIC-IDS2017, and CSE-CIC-IDS2018. All models are tuned with equal trial budgets and trained over repeated seeds under eight evaluation conditions. The study also covers shortcut features and adversarial robustness under unconstrained, realistic, and consistent threat models.

Code comments and console messages are in Traditional Chinese; identifiers, file names, and the paper are in English.

## Contents

| Path | What it is |
|---|---|
| `nids_core.py`, `nids_data.py`, `nids_repro.py` | Shared training/evaluation loop, data loading for all conditions, model definitions (incl. FT-Transformer) |
| `cicids.py`, `cic2018.py`, `unsw.py` | Dataset preprocessing (CIC-IDS2017, CSE-CIC-IDS2018 stratified subsample, UNSW-NB15) |
| `tune.py`, `baselines_v2.py` | Hyperparameter search with equal trial budgets (deep models / RF and XGBoost) |
| `run_v2.py`, `pipeline_v2.py`, `supervise_v2.py` | Main experiments (RQ1–RQ3) and the pipeline that runs all stages in order |
| `shortcut_v2.py` | RQ4: SHAP importance and feature ablation |
| `adversarial_v2.py` | RQ5: PGD and random-search attacks, transfer attacks, adversarial training |
| `yin_repro.py` | Case study: reproduction of Yin et al. (2017) |
| `sensitivity_v2.py`, `device_check.py`, `scaler_leak_check.py`, `scaler_control.py` | Robustness checks reported under Threats to Validity |
| `analyze_v2.py` | Aggregates all results into `results/v2/analysis/` and draws the figures |
| `paper/` | Scripts that generate the manuscript's tables and numbers from the results |
| `cloud/`, `mac_package/` | Shell scripts used to run parts of the experiments on a cloud GPU and a Mac mini |
| `results/v2/` | Raw per-run results: training curves, gradient norms, all metrics, tuning trials, selected hyperparameters (`tuned_hp.json`) |
| other `*.py` files | Earlier exploratory version of the study (v1), superseded by the `*_v2.py` scripts |

Trained model weights (about 830 MB) are not in this repository; they are archived on Zenodo: [10.5281/zenodo.24391892](https://doi.org/10.5281/zenodo.24391892).

## Setup

Python 3.12. The experiments used Keras 3 with the PyTorch backend (PyTorch 2.6, CUDA 12.4).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_laptop.txt
```

`requirements_laptop.txt` is the full environment of the main machine and includes a few packages that are not needed for the experiments. `mac_package/requirements.txt` is the environment used on Apple silicon.

Set the backend before running any script:

```bash
export KERAS_BACKEND=torch KERAS_TORCH_DEVICE=cuda PYTHONIOENCODING=utf-8
# PowerShell: $env:KERAS_BACKEND='torch'; $env:KERAS_TORCH_DEVICE='cuda'; $env:PYTHONIOENCODING='utf-8'
```

## Data

The datasets are not redistributed here. Download them from their publishers and place them as follows:

| Dataset | Source | Expected location |
|---|---|---|
| NSL-KDD | https://www.unb.ca/cic/datasets/nsl.html | `data/KDDTrain+.txt`, `data/KDDTest+.txt`, `data/KDDTest-21.txt` |
| UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset (training and testing set CSVs) | `data/unsw/test.csv` = official **training** set (175,341 records), `data/unsw/train.csv` = official **testing** set (82,332 records) — note the swapped names, which the code expects |
| CIC-IDS2017 (original) | https://www.unb.ca/cic/datasets/ids-2017.html (`MachineLearningCSV.zip`) | `data/cicids2017/original/MachineLearningCSV.zip` |
| CIC-IDS2017 (corrected, Engelen et al. / Liu et al., IEEE CNS 2022) | https://intrusion-detection.distrinet-research.be/CNS2022/ | `data/cicids2017/improved/CICIDS2017_improved.zip` |
| CSE-CIC-IDS2018 (corrected, same source) | https://intrusion-detection.distrinet-research.be/CNS2022/ | `data/cicids2018/CSECICIDS2018_improved.zip` |

Then build the CIC caches once:

```bash
python cicids.py                                             # CIC-IDS2017, original and corrected
python cic2018.py                                            # CSE-CIC-IDS2018 main subsample (seed 2018)
python cic2018.py --seed 2019 --variant improved_s2019       # second subsample (cic18b)
```

## Reproducing the experiments

Every script skips work that is already finished, can be interrupted and resumed, and can run in several processes at once (lock files split the work). Results are written to `results/v2/`. If a process crashes, delete its stale `.lock` file before restarting.

```bash
# 1. Tuning, main experiments, shortcut analysis, adversarial attacks and adversarial training
python pipeline_v2.py --models DNN CNN RNN LSTM CNN-LSTM GRU
python tune.py --models FTT && python run_v2.py --models FTT --groups main temporal
python baselines_v2.py --stage tune run                      # random forest and XGBoost (CPU)

# 2. Consistent threat model and transfer attacks (need all weights on one machine)
python adversarial_v2.py --stage consistent
python adversarial_v2.py --stage advtrain_consistent --datasets cic18
python adversarial_v2.py --stage transfer

# 3. Case study (Yin et al., 2017)
python yin_repro.py --interp record
python yin_repro.py --interp featseq
python yin_repro.py --interp trees

# 4. Checks reported under Threats to Validity
python sensitivity_v2.py
python device_check.py --label <machine> --models GRU DNN && python device_check.py --compare
python scaler_leak_check.py
python scaler_control.py --models DNN CNN && python scaler_control.py --compare
python tune.py --datasets cic18b --models DNN && python run_v2.py --datasets cic18b --models DNN --groups main conditions
python baselines_v2.py --datasets cic18b

# 5. Aggregate results and figures
python analyze_v2.py
```

The scripts in `cloud/` and `mac_package/` show how the work was split between machines in the original runs. Running everything takes several GPU-days; to check the reported numbers without retraining, run only `python analyze_v2.py`, which reads the raw results included in `results/v2/`.

## License

See [LICENSE](LICENSE). The datasets are subject to the licenses of their publishers.

## Citation

If you use this code, please cite the paper (citation details will be added after publication).
