
# Task 3: FairTrade extended to jointly optimize fairness for two sensitive attributes
# (default: gender + race on the Adult dataset).
#
# Design: the client-side fairness penalty becomes max(DP_loss(gender), DP_loss(race)) --
# whichever attribute the current model discriminates against more heavily dominates the
# gradient. The server-side multi-objective Bayesian optimization is otherwise unchanged:
# it still optimizes two objectives (balanced accuracy, -discrimination score), where
# discrimination score is now max(|SPD_gender|, |SPD_race|) instead of a single SPD. This
# keeps the existing, already-verified 2D GP/qEHVI machinery from FairTrade.py untouched --
# only the definition of "discrimination score" changes to cover both attributes.
#
# This is a copy of FairTrade.py with the fairness computation extended; Task 1/2 code
# (FairTrade.py) is left untouched so its results stay reproducible.

import torch
import argparse
import random
from utilities import find_statistical_parity_score, all_metrics
from load_data_utilities import get_data, load_dataset
from constraint import DemographicParityLoss

from torch import nn, optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from sklearn.metrics import average_precision_score

parser = argparse.ArgumentParser(description="FairTrade jointly optimized for two sensitive attributes.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility. Default is 42.")
parser.add_argument("--num_clients", type=int, default=3, choices=[3, 5, 10, 15])
parser.add_argument("--dataset_name", type=str, default='adult', choices=['adult'],
                     help="Only 'adult' is supported: it's the only dataset with two documented sensitive attributes (sex, race) for this task.")
parser.add_argument("--epochs", type=int, default=15, help="Client training epochs. Default is 15.")
parser.add_argument("--communication_rounds", type=int, default=50)
parser.add_argument("--mobo_optimization_rounds", type=int, default=10)
parser.add_argument("--distribution_type", type=str, default='random', choices=['random', 'attribute-based'])
parser.add_argument("--second_sensitive_feature", type=str, default='race', help="Second sensitive attribute column. Default is 'race'.")
parser.add_argument("--second_sensitive_reference_value", type=str, default='White',
                     help="Reference/non-protected category of the second attribute. Default is 'White'.")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


def create_model(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )


dataset_name = args.dataset_name
num_clients = args.num_clients
epochs = args.epochs
communication_rounds = args.communication_rounds
mobo_optimization_rounds = args.mobo_optimization_rounds
distribution_type = args.distribution_type

url = './datasets/adult.csv'
sensitive_feature = 'sex'  # 'sex': 0 -> female, 1 -> male

# Look up the encoded value of the second attribute's reference category (e.g. 'White')
# dynamically, rather than hardcoding the LabelEncoder's alphabetical index -- keeps this
# working even if the raw category spelling/count changes.
raw_df = pd.read_csv(url)
second_attr_encoder = LabelEncoder()
second_attr_encoder.fit(raw_df[args.second_sensitive_feature])
reference_group_code = int(second_attr_encoder.transform([args.second_sensitive_reference_value])[0])
print(f"Second sensitive attribute: '{args.second_sensitive_feature}', reference category "
      f"'{args.second_sensitive_reference_value}' encoded as {reference_group_code}")

bal_acc_list = []
disc_score_list = []       # max(|SPD_gender|, |SPD_second|) per round -- the actual MOO objective
gender_spd_list = []       # tracked separately for the report/trade-off discussion
second_spd_list = []

clients_data, X_test, y_test, sex_list, column_names_list, ytest_potential = load_dataset(
    url, dataset_name, num_clients, sensitive_feature, distribution_type
)
X_test = X_test.to(device)
y_test = y_test.to(device)
global_model = create_model(X_test.shape[1]).to(device)

second_attr_idx = column_names_list.index(args.second_sensitive_feature)
second_attr_list_test = (X_test[:, second_attr_idx] == reference_group_code).float().cpu().tolist()

cost_false_positives = 1.0


def calculate_weights(targets, cost_false_negatives=5):
    cost_false_negatives = 10
    return torch.where(targets == 1, cost_false_negatives, cost_false_positives)


def evaluate(alpha=100, lr=0.001, cost_false_negatives=5):
    params = [torch.zeros_like(param.data) for param in global_model.parameters()]
    for client_name in clients_data.keys():
        print(client_name)
        X1, y1, s1_gender, y1_potential = get_data(client_name, clients_data)
        X1 = X1.to(device)
        y1 = y1.to(device)
        y1_potential = y1_potential.to(device)
        s1_gender = s1_gender.to(device)
        s1_second = (X1[:, second_attr_idx] == reference_group_code).float().to(device)

        model1 = create_model(X1.shape[1]).to(device)
        model1.load_state_dict(global_model.state_dict())
        optimizer1 = optim.Adam(model1.parameters(), lr=lr)
        dp_loss_gender = DemographicParityLoss(alpha=alpha)
        dp_loss_second = DemographicParityLoss(alpha=alpha)

        for epoch in range(epochs):
            optimizer1.zero_grad()
            y_pred = model1(X1)
            weights = calculate_weights(y1, cost_false_negatives)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)

            fairness_loss_gender = dp_loss_gender(X1, y_pred, s1_gender, y1_potential).to(device)
            fairness_loss_second = dp_loss_second(X1, y_pred, s1_second, y1_potential).to(device)
            # Joint fairness objective: whichever attribute is currently worse dominates the penalty.
            fairness_loss = torch.max(fairness_loss_gender, fairness_loss_second)

            loss = criterion(y_pred.view(-1), y1) + fairness_loss
            loss.backward()
            optimizer1.step()
        print(f'- Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        for param, param_sum in zip(model1.parameters(), params):
            param_sum.add_(param.data)

    average_params = [param_sum / len(clients_data) for param_sum in params]
    with torch.no_grad():
        for param_global, param_avg in zip(global_model.parameters(), average_params):
            param_global.copy_(param_avg)
    global_model.eval()

    with torch.no_grad():
        y_pred = global_model(X_test).squeeze()
        y_pred_cls = y_pred.round()
        sensitivity, specificity, bal_acc, G_mean, FN_rate, FP_rate, Precision, f1_sc, acc, auc = all_metrics(y_test.cpu(), y_pred.cpu())
        stat_parity_gender = find_statistical_parity_score(sex_list, y_test, y_pred_cls)
        stat_parity_second = find_statistical_parity_score(second_attr_list_test, y_test, y_pred_cls)
        disc_score = max(abs(stat_parity_gender), abs(stat_parity_second))
        auprc = average_precision_score(y_test.cpu(), y_pred.cpu())
        print(f'Communication round {round + 1}/{communication_rounds}')
        print(f'Test accuracy: {acc.item()}')
        print("BalanceACC: %s" % bal_acc)
        print(f"statistical parity (gender): {stat_parity_gender}")
        print(f"statistical parity ({args.second_sensitive_feature}): {stat_parity_second}")
        print(f"combined discrimination score (max |SPD|): {disc_score}")

    objectives = torch.tensor([[-disc_score, bal_acc]])
    return objectives, stat_parity_gender, stat_parity_second


for round in range(communication_rounds):
    print(f'Communication round {round + 1}/{communication_rounds}')

    bounds = torch.tensor([[100.0, 0.0], [2000.0, 0.01]])
    alpha = torch.tensor([100], dtype=torch.float32).view(1, -1)

    if round == 0:
        objectives, spd_gender, spd_second = evaluate(alpha)
    else:
        objectives, spd_gender, spd_second = evaluate(updated_alpha, updated_lr)
    disc_score_list.append(objectives[0, 0].item())
    bal_acc_list.append(objectives[0, 1].item())
    gender_spd_list.append(spd_gender)
    second_spd_list.append(spd_second)

    x_input = torch.tensor([100, 0.001], dtype=torch.float32).view(1, -1)

    models = []
    for i in range(objectives.shape[-1]):
        models.append(SingleTaskGP(x_input, objectives[:, i].unsqueeze(-1).float()))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    for i in range(mobo_optimization_rounds):
        print("Global optimization round:", i)
        fit_gpytorch_mll(mll)
        ref_point = torch.tensor([0.0001, 0.001])
        acq_func = qExpectedHypervolumeImprovement(
            model=model.float(),
            ref_point=torch.tensor([0.001, 0.001]),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
            objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
            partitioning=FastNondominatedPartitioning(ref_point, Y=objectives),
        )
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=300,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
        )

        new_objectives, spd_gender, spd_second = evaluate(candidate[0, 0].item(), candidate[0, 1].item())
        for j, m in enumerate(model.models):
            train_x = torch.cat([m.train_inputs[0], candidate])
            train_y = torch.cat([m.train_targets, new_objectives[:, j]])
            m.set_train_data(train_x, train_y, strict=False)
            if j == 0:
                train_y_0 = train_y
            else:
                train_y_1 = train_y
        train_y_all = torch.stack((train_y_0, train_y_1), dim=1)
        weights = torch.tensor([0.6, 0.4])
        weighted_sums = (train_y_all * weights).sum(dim=1)
        best_solution_idx = weighted_sums.argmax()
        best_solution = train_y_all[best_solution_idx]
        print(f"Best solution based on weighted sum: {best_solution}")
        best_train_x = train_x[best_solution_idx]
        updated_alpha, updated_lr = best_train_x.tolist()

global_model.eval()
with torch.no_grad():
    y_pred = global_model(X_test).squeeze()
    y_pred_cls = y_pred.round()
    sensitivity, specificity, bal_acc, G_mean, FN_rate, FP_rate, Precision, f1_sc, acc, auc = all_metrics(y_test.cpu(), y_pred.cpu())
    stat_parity_gender = find_statistical_parity_score(sex_list, y_test, y_pred_cls)
    stat_parity_second = find_statistical_parity_score(second_attr_list_test, y_test, y_pred_cls)
    disc_score = max(abs(stat_parity_gender), abs(stat_parity_second))
    print(f'Final test accuracy: {acc.item()}')
    print("Final BalanceACC: %s" % bal_acc)
    print(f"Final statistical parity (gender): {stat_parity_gender}")
    print(f"Final statistical parity ({args.second_sensitive_feature}): {stat_parity_second}")
    print(f"Final combined discrimination score (max |SPD|): {disc_score}")

destination = './results/'
distribution_tag = 'random' if distribution_type == 'random' else 'attr'
tag = f'{num_clients}_{distribution_tag}_multi_attr'
np.save(f'{destination}{dataset_name}/{tag}_bal_acc.npy', np.array(bal_acc_list))
np.save(f'{destination}{dataset_name}/{tag}_disc_score.npy', np.array(disc_score_list))
np.save(f'{destination}{dataset_name}/{tag}_gender_spd.npy', np.array(gender_spd_list))
np.save(f'{destination}{dataset_name}/{tag}_{args.second_sensitive_feature}_spd.npy', np.array(second_spd_list))

checkpoint_name = f'{num_clients}_model_{distribution_tag}_multi_attr.pt'
torch.save({
    'model_state_dict': global_model.state_dict(),
    'column_names_list': column_names_list,
    'sensitive_feature': sensitive_feature,
    'second_sensitive_feature': args.second_sensitive_feature,
    'second_sensitive_reference_value': args.second_sensitive_reference_value,
    'dataset_name': dataset_name,
    'num_clients': num_clients,
    'fairness_notion': 'multi_attr_max_spd',
    'distribution_type': distribution_type,
    'seed': args.seed,
}, destination + dataset_name + '/' + checkpoint_name)
print(f'Saved trained model checkpoint to {destination}{dataset_name}/{checkpoint_name}')
