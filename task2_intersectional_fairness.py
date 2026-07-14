"""Task 2: Intersectional fairness evaluation for a trained FairTrade model.

Loads a FairTrade checkpoint (produced by FairTrade.py) and evaluates Statistical
Parity Difference (SPD) for gender alone, race alone, and their four-way
intersection (White Male, White Female, Non-White Male, Non-White Female) on the
held-out test set, using fairlearn's MetricFrame (https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html).
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate

from load_data_utilities import load_dataset

# --- palette (dataviz skill reference palette, light mode) ---
COLOR_SURFACE = '#fcfcfb'
COLOR_INK = '#0b0b0b'
COLOR_MUTED = '#898781'
COLOR_GRID = '#e1e0d9'
CATEGORICAL = ['#2a78d6', '#1baf7a', '#eda100', '#008300']  # blue, aqua, yellow, green


def create_model(input_dim):
    """3-layer MLP (64-32-1, ReLU/Sigmoid), matching FairTrade.py's base learner exactly
    so the saved state_dict loads correctly."""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


def signed_spd(y_pred_cls, group_mask):
    """FairTrade's own SPD convention: P(pred=1 | non-protected) - P(pred=1 | protected).
    group_mask is True for the non-protected/reference group (Male, White)."""
    non_protected_rate = y_pred_cls[group_mask].mean() if group_mask.sum() > 0 else 0.0
    protected_rate = y_pred_cls[~group_mask].mean() if (~group_mask).sum() > 0 else 0.0
    return float(non_protected_rate - protected_rate)


def main():
    """Loads a trained FairTrade checkpoint, reconstructs its exact test split, and reports
    gender-only, race-only, and intersectional (4-subgroup) Statistical Parity Difference --
    saving a summary table (CSV) and comparison bar chart (PNG) under results/<dataset>/."""
    parser = argparse.ArgumentParser(description="Task 2: intersectional fairness evaluation.")
    parser.add_argument("--dataset_name", type=str, default="adult")
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--fairness_notion", type=str, default="stat_parity")
    parser.add_argument("--distribution_type", type=str, default="random")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    distribution_tag = 'random' if args.distribution_type == 'random' else 'attr'
    checkpoint_path = f'./results/{args.dataset_name}/{args.num_clients}_model_{distribution_tag}_{args.fairness_notion}.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    sensitive_feature = checkpoint['sensitive_feature']
    column_names_list = checkpoint['column_names_list']
    url = f'./datasets/{args.dataset_name}.csv'

    # Same seed=42 hardcoded inside load_data_utilities' train_test_split calls, so this
    # reconstructs the identical held-out test split used to train the checkpoint.
    _, X_test, y_test, sex_list, column_names_list_reload, _ = load_dataset(
        url, args.dataset_name, args.num_clients, sensitive_feature, args.distribution_type
    )
    assert column_names_list_reload == column_names_list, "test split does not match the checkpoint's training run"

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    model = create_model(X_test.shape[1]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        y_pred = model(X_test).squeeze()
        y_pred_cls = y_pred.round().cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # sex: 0=Female, 1=Male (sklearn LabelEncoder, alphabetical). race: label-encoded
    # alphabetically -> White=4, everything else (Black/Asian-Pac-Islander/
    # Amer-Indian-Eskimo/Other)=0-3 -> Non-White.
    sex_arr = np.array(sex_list)
    race_idx = column_names_list.index('race')
    race_arr = X_test[:, race_idx].cpu().numpy()

    is_male = sex_arr == 1
    is_white = race_arr == 4

    gender_labels = np.where(is_male, 'Male', 'Female')
    race_labels = np.where(is_white, 'White', 'Non-White')
    intersection_labels = np.array([f'{r} {g}' for r, g in zip(race_labels, gender_labels)])

    # --- per-attribute and intersectional selection rates (fairlearn) ---
    mf_gender = MetricFrame(metrics=selection_rate, y_true=y_test_np, y_pred=y_pred_cls, sensitive_features=pd.Series(gender_labels, name='gender'))
    mf_race = MetricFrame(metrics=selection_rate, y_true=y_test_np, y_pred=y_pred_cls, sensitive_features=pd.Series(race_labels, name='race'))
    mf_intersection = MetricFrame(metrics=selection_rate, y_true=y_test_np, y_pred=y_pred_cls, sensitive_features=pd.Series(intersection_labels, name='intersection'))

    spd_gender = signed_spd(y_pred_cls, is_male)          # P(pred=1|Male) - P(pred=1|Female), matches FairTrade.py's own convention
    spd_race = signed_spd(y_pred_cls, is_white)            # P(pred=1|White) - P(pred=1|Non-White)
    spd_intersection_maxmin = float(mf_intersection.difference(method='between_groups'))  # max-min gap across the 4 subgroups

    # --- summary table ---
    rows = []
    for label, rate in mf_gender.by_group.items():
        rows.append({'group': label, 'attribute': 'gender', 'selection_rate': rate, 'n': int((gender_labels == label).sum())})
    for label, rate in mf_race.by_group.items():
        rows.append({'group': label, 'attribute': 'race', 'selection_rate': rate, 'n': int((race_labels == label).sum())})
    for label, rate in mf_intersection.by_group.items():
        rows.append({'group': label, 'attribute': 'intersection', 'selection_rate': rate, 'n': int((intersection_labels == label).sum())})
    summary_df = pd.DataFrame(rows)

    out_dir = f'./results/{args.dataset_name}'
    os.makedirs(out_dir, exist_ok=True)
    tag = args.fairness_notion  # distinguishes which trained model was evaluated, e.g. stat_parity vs multi_attr
    summary_df.to_csv(f'{out_dir}/task2_intersectional_summary_{tag}.csv', index=False)

    spd_summary_df = pd.DataFrame([
        {'comparison': 'Gender SPD (Male - Female)', 'spd': spd_gender},
        {'comparison': 'Race SPD (White - Non-White)', 'spd': spd_race},
        {'comparison': 'Intersectional SPD (max-min over 4 subgroups)', 'spd': spd_intersection_maxmin},
    ])
    spd_summary_df.to_csv(f'{out_dir}/task2_spd_comparison_{tag}.csv', index=False)

    print(summary_df.to_string(index=False))
    print()
    print(spd_summary_df.to_string(index=False))

    # --- bar chart: per-attribute SPD vs intersectional SPD ---
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    labels = ['Gender SPD\n(Male − Female)', 'Race SPD\n(White − Non-White)', 'Intersectional SPD\n(max − min, 4 subgroups)']
    values = [spd_gender, spd_race, spd_intersection_maxmin]
    bars = ax.bar(labels, values, color=CATEGORICAL[:3], width=0.55, zorder=3)
    ax.axhline(0, color=COLOR_MUTED, linewidth=1)
    ax.grid(axis='y', color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    top = max(values) * 1.22 if max(values) > 0 else 0.05
    ax.set_ylim(0, top)
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_INK, pad=8)
    ax.set_ylabel('Statistical Parity Difference', color=COLOR_INK)
    ax.set_title('Per-attribute vs. intersectional SPD — Adult, gender × race', color=COLOR_INK, pad=14)
    for bar, val in zip(bars, values):
        offset = top * 0.02
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset, f'{val:.3f}', ha='center', va='bottom', color=COLOR_INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/task2_spd_comparison_{tag}.png', dpi=150, facecolor=COLOR_SURFACE)
    print(f'\nSaved chart to {out_dir}/task2_spd_comparison_{tag}.png')
    print(f'Saved tables to {out_dir}/task2_intersectional_summary_{tag}.csv and {out_dir}/task2_spd_comparison_{tag}.csv')


if __name__ == '__main__':
    main()
