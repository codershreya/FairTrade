# FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning
As Federated Learning (FL) gains prominence in distributed machine learning applications, achieving fairness without compromising predictive performance becomes paramount. The data being gathered from distributed clients in an FL environment often leads to class imbalance. In such scenarios, balanced accuracy rather than accuracy is the true representation of model performance. However, most state-of-the-art fair FL methods report accuracy as the measure of performance,  which can lead to misguided interpretations of the model's effectiveness to mitigate discrimination. To the best of our knowledge, this work presents the first attempt towards achieving Pareto-optimal trade-offs between balanced accuracy and fairness in a federated environment (FairTrade). By utilizing multi-objective optimization, the framework negotiates the intricate balance between model's balanced accuracy and fairness. The framework's agnostic design adeptly accommodates both statistical and causal fairness notions, ensuring its adaptability across diverse FL contexts. We provide empirical evidence of our novel framework's efficacy through extensive experiments on five real-world datasets and comparisons with six competing baselines. The empirical results underscore the significant potential of our framework in improving the trade-off between fairness and balanced accuracy in FL applications.
## The datsets used in this project
* [Adult Census](https://archive.ics.uci.edu/dataset/2/adult)
* [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* [Default](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* [Law School](https://github.com/iosifidisvasileios/FABBOO/blob/master/Data/law_dataset.arff)
## Code
### Dataset Processing Scripts

The `datasets` directory contains all the datasets used in this project. Below is a description of python scripts written to process datasets:

- `load_data_utilities.py`: Utility script for loading and preprocessing all the datasets (Adult, Bank, Default, Law).

### Utility Scripts
- `utilities.py`: Utility script for computing evaluation metrics including 'statistical parity', average treatment effect (ATE), balanced accuracy, and accuracy.

### FairTrade main scripts
The following scripts constitute the complete methodology of FairTrade
- `Fairtrade-crypten.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets with secure multiparty protocol.
- `Fairtrade.py`: Main script for the 'FairTrade' framework that orchestrates the fairness aware federated learning process on different datasets without secure multiparty protocol.

- `constraint.py`: The script contains the implementation of fairness constraints for discrimination mitigation.
  
## Running the FairTrade-crypten.py Script

To run the `FairTrade-crypten.py` script with the default settings, you can use the following command:

```bash
python FairTrade-crypten.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Running the FairTrade.py Script
To run the `FairTrade.py` script with the default settings, you can use the following command:

```bash
python FairTrade.py --fairness_notion 'stat_parity' --num_clients 3 --dataset_name 'bank' --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 --distribution_type 'random'
```
## Prerequisites

Python 3.11+, a CUDA-capable GPU is optional (falls back to CPU automatically). Install the pinned dependencies:

```bash
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on Linux/Mac
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118  # or omit --index-url for CPU-only
pip install -r requirements.txt
```

All scripts below accept `--seed` (default 42) and are deterministic: two runs with the same
seed produce bit-identical results (verified). `torch.manual_seed`/`np.random.seed`/`random.seed`
are set in `FairTrade.py` and `FairTrade_multi_attribute.py` before any data loading or model
initialization.

---

## HiWi Challenge — Tasks 1–3 (modifications summary)

This fork adds the following on top of the original FairTrade codebase, kept separate from it
wherever possible so the original single-attribute pipeline stays unchanged and reproducible:

| File | Status | Purpose |
|---|---|---|
| `FairTrade.py` | modified | GPU auto-detection (`cuda` if available, else `cpu`); `--seed` argument for reproducibility; saves a model checkpoint (`results/<dataset>/<n>_model_<split>_<notion>.pt`) at the end of training; fixes a return-value/unpacking mismatch in the main loop that made the original repo raise `ValueError` on the first communication round (see Task 4 in the report) |
| `constraint.py` | modified | Fixed a CPU/GPU device-mismatch bug in `ConstraintLoss.__init__` (see Task 4 in the report) |
| `task2_intersectional_fairness.py` | new (Task 2) | Loads a trained checkpoint and evaluates SPD for gender alone, race alone, and their 4-way intersection using `fairlearn` |
| `FairTrade_multi_attribute.py` | new (Task 3) | FairTrade extended to jointly optimize fairness for gender **and** race (`max(SPD_gender, SPD_race)`) |
| `task3_comparison_chart.py` | new (Task 3) | Generates the Task 1 vs. Task 3 before/after comparison chart |

### Task 1 — Reproduce & Understand
```bash
python FairTrade.py --dataset_name adult --num_clients 3 --fairness_notion stat_parity \
  --epochs 15 --communication_rounds 50 --mobo_optimization_rounds 10 \
  --distribution_type random --seed 42
```
Saves `results/adult/3_bal_acc_stat_parity.npy`, `3_stat_parity.npy`, and the checkpoint
`3_model_random_stat_parity.pt`.

### Task 2 — Intersectional Fairness Evaluation
Requires the Task 1 checkpoint above. Then:
```bash
python task2_intersectional_fairness.py --dataset_name adult --num_clients 3 \
  --fairness_notion stat_parity --distribution_type random
```
Saves `results/adult/task2_intersectional_summary_stat_parity.csv`,
`task2_spd_comparison_stat_parity.csv`, and `task2_spd_comparison_stat_parity.png`.

### Task 3 — Multi-attribute Extension
```bash
python FairTrade_multi_attribute.py --num_clients 3 --epochs 15 --communication_rounds 50 \
  --mobo_optimization_rounds 10 --distribution_type random --seed 42
```
Saves the joint-model checkpoint `3_model_random_multi_attr.pt` and per-round metric arrays.
Then re-run Task 2's script against it (`--fairness_notion multi_attr`) and generate the
before/after chart:
```bash
python task2_intersectional_fairness.py --dataset_name adult --num_clients 3 \
  --fairness_notion multi_attr --distribution_type random
python task3_comparison_chart.py
```
Saves `results/adult/task3_before_after_comparison.png`.

### Task 4 — Code Review & Bug Hunt
No script to run — see the PDF report for the full write-up. Primary finding: a train/test
feature-scaling leakage issue in `load_data_utilities.py` (`StandardScaler` fit before the
train/test split, present in all five dataset loaders). Also documents three bugs found and
fixed while completing Tasks 1–3: a return-value/unpacking mismatch that made the original
repo crash on its first run, a CPU/GPU device-mismatch bug, and a missing reproducibility seed.

## Citation Request
If you find this work useful in your research, please consider citing:
```bash
@inproceedings{badar2024fairtrade,
  title={FairTrade: Achieving Pareto-Optimal Trade-offs Between Balanced Accuracy and Fairness in Federated Learning},
  author={Badar, Maryam and Sikdar, Sandipan and Nejdl, Wolfgang and Fisichella, Marco},
  booktitle={Proceedings of the 38th Annual AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
