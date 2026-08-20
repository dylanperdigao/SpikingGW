import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import optuna
import warnings
import os
import sys
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime
from modules.networks import FFSNN, FFSNN_GWT, CSNN, CSNN_GWT
from modules.datasets import BAF
from modules.metrics import metrics_performance, metrics_fairness

PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(PATH, os.path.pardir)))
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

DATASETS = ["BAF-Base", "BAF-TypeI", "BAF-TypeII", "BAF-TypeIII", "BAF-TypeIV", "BAF-TypeV"]
MODELS = ["FFSNN", "FFSNN_GWT", "CSNN", "CSNN_GWT"]
METRIC = "recall"

def load_dataset(dataset: str, root='./data', validation=False):
    if dataset == 'baf':
        variant = 'Base'
    elif 'baf' in dataset.lower():
        variant = dataset.split('-')[-1]
    else:
        raise ValueError("Invalid dataset")
    train_dataset = BAF(variant=variant, root=f"{root}/BAF", train=True, mode='train', validation=validation)
    test_dataset = BAF(variant=variant, root=f"{root}/BAF", train=False, mode='test', validation=validation)
    return train_dataset, test_dataset

def main(trial: optuna.Trial, dataset: str, model_name: str, epochs: int, steps: int, batch_size: int, slope: int,
         fixparams: bool, learn_betas: bool, learn_thresholds: bool, gw_thr:float, date: str, device_name: str, seed: int):

    ######################################
    # Hyperparameter Search Space
    ######################################
    if model_name == "CSNN":
        layers = 5
    elif model_name in ["FFSNN", "FFSNN_GWT", "CSNN_GWT"]:
        layers = 4
    else:
        layers = 0
    slopes = [slope] * layers
    if not fixparams:
        thresholds = [trial.suggest_float(f"threshold_{i}", 0.1, 10.0, step=0.01) for i in range(1, layers + 1)]
        betas = [trial.suggest_float(f"beta_{i}", 0.1, 1.0, step=0.01) for i in range(1, layers + 1)]
    else:
        thresholds = [1.0] * layers
        betas = [0.9] * layers
    if "GWT" in model_name:
        if gw_thr:
            gw_threshold=gw_thr
        else:
            gw_threshold = trial.suggest_float("gw_threshold", 1.5, 3.0)
    else:
        gw_threshold = None

    ######################################
    # Data Preprocessing
    ######################################
    train_dataset, test_dataset = load_dataset(dataset, validation=False)
    x_train = train_dataset.data.squeeze(1).numpy()
    y_train = train_dataset.targets.numpy()
    x_test = test_dataset.data.squeeze(1).numpy()
    y_test = test_dataset.targets.numpy()
    df_test = pd.DataFrame(x_test, columns=test_dataset.features)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ######################################
    # Data Preparation
    ######################################
    train_tensor = TensorDataset(torch.FloatTensor(x_train_scaled), torch.LongTensor(y_train))
    test_tensor = TensorDataset(torch.FloatTensor(x_test_scaled), torch.LongTensor(y_test))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    input_size = x_train_scaled.shape[1]
    device = torch.device(device_name)
    num_fraud_samples = np.sum(y_train == 1)
    num_normal_samples = np.sum(y_train == 0)
    weight_normal = 1.0 / num_normal_samples if num_normal_samples > 0 else 1.0
    weight_fraud = 1.0 / num_fraud_samples if num_fraud_samples > 0 else 1.0
    class_weights = torch.tensor([weight_normal, weight_fraud], dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() 
    # Model Instantiation
    if model_name == "FFSNN":
        model = FFSNN(input_size, betas, thresholds, slopes, learn_betas, learn_thresholds, device).to(device)
    elif model_name == "FFSNN_GWT":
        model = FFSNN_GWT(input_size, betas, thresholds, slopes, learn_betas, learn_thresholds, device, gw_threshold).to(device)
    elif model_name == "CSNN":
        model = CSNN(input_size, betas, thresholds, slopes, learn_betas, learn_thresholds, device).to(device)
    elif model_name == "CSNN_GWT":
        model = CSNN_GWT(input_size, betas, thresholds, slopes, learn_betas, learn_thresholds, device, gw_threshold).to(device)
    else:
        raise ValueError("Invalid model name")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    ###########################################################
    # Training Loop
    ###########################################################
    patience = 5  
    epochs_no_improve = 0
    best_recall = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    min_epochs = 5
    for epoch in range(epochs):
        model.train()
        losses = []
        train_p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, target in train_p_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            spk_rec, _ = model(data, steps)
            pred_spikes = spk_rec.sum(dim=0) 
            loss = criterion(pred_spikes, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            train_p_bar.set_postfix(Loss=np.mean(losses))
        model.eval()
        y_hat = []
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", leave=False):
                data, target = data.to(device), target.to(device)
                spk_rec, _ = model(data, steps)
                spike_counts = spk_rec.sum(dim=0)
                probabilities = F.softmax(spike_counts, dim=1)
                y_hat.extend(probabilities[:, 1].cpu().numpy())
            perf_metrics = metrics_performance(y_test, y_hat)
            recall = perf_metrics["recall"]
            trial.report(recall, step=epoch)
            if recall > best_recall:
                best_recall = recall
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
            if epoch >= min_epochs and epochs_no_improve >= patience:
                break
            if trial.should_prune():
                raise optuna.TrialPruned()
    trial.set_user_attr("best_epoch", epoch + 1 - epochs_no_improve if best_recall > 0 else 0)

    ######################################
    # Best Model Inference
    ######################################
    model.load_state_dict(best_model_state)
    model.eval()
    y_hat = []
    total_spikes_normal = 0
    total_spikes_fraud = 0
    elements_normal = 0
    elements_fraud = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Final Best Model Inference"):
            data, target = data.to(device), target.to(device)
            spk_rec, _ = model(data, steps)
            mask_fraud = (target == 1)
            mask_normal = (target == 0)
            if mask_fraud.sum() > 0:
                total_spikes_fraud += spk_rec[:, mask_fraud, :].sum().item()
                elements_fraud += spk_rec[:, mask_fraud, :].numel()
            if mask_normal.sum() > 0:
                total_spikes_normal += spk_rec[:, mask_normal, :].sum().item()
                elements_normal += spk_rec[:, mask_normal, :].numel()
            spike_counts = spk_rec.sum(dim=0)
            probabilities = F.softmax(spike_counts, dim=1)
            y_hat.extend(probabilities[:, 1].cpu().numpy())
    spike_rate_normal = (total_spikes_normal / elements_normal) if elements_normal > 0 else 0
    spike_rate_fraud = (total_spikes_fraud / elements_fraud) if elements_fraud > 0 else 0
    trial.set_user_attr("@energy spike_rate_normal", spike_rate_normal)
    trial.set_user_attr("@energy spike_rate_fraud", spike_rate_fraud)
    perf_metrics = metrics_performance(y_test, y_hat)
    fair_age_metrics = metrics_fairness(df_test, y_test, y_hat, "customer_age", 50)
    fair_income_metrics = metrics_fairness(df_test, y_test, y_hat, "income", 0.5)
    fair_employment_metrics = metrics_fairness(df_test, y_test, y_hat, "employment_status", 0.5)
    trial.set_user_attr("@perf accuracy", perf_metrics["accuracy"])
    trial.set_user_attr("@perf precision", perf_metrics["precision"])
    trial.set_user_attr("@perf recall", perf_metrics["recall"])
    trial.set_user_attr("@perf f1_score", perf_metrics["f1_score"])
    trial.set_user_attr("@perf fpr", perf_metrics["fpr"])
    trial.set_user_attr("@perf tnr", perf_metrics["tnr"])
    trial.set_user_attr("@perf roc_auc", perf_metrics["roc_auc"])
    trial.set_user_attr("@perf pr_auc", perf_metrics["pr_auc"])
    trial.set_user_attr("@fair fpr_ratio_age", fair_age_metrics["fpr_ratio"])
    trial.set_user_attr("@fair fpr_ratio_income", fair_income_metrics["fpr_ratio"])
    trial.set_user_attr("@fair fpr_ratio_employment", fair_employment_metrics["fpr_ratio"])
    if "gw_threshold" in trial.params:
        trial.set_user_attr("@gwt threshold_used", trial.params["gw_threshold"])
    try:
        best_v = trial.study.best_value
        print(f"Best {METRIC} so far: {best_v*100:.2f}%")
        print(f"Current trial {METRIC}: {best_recall*100:.2f}%")
        print(f"Spike Rate (Normal): {spike_rate_normal*100:.2f}% | Spike Rate (Fraud): {spike_rate_fraud*100:.2f}%")
    except ValueError:
        print("First trial: No best value yet.")
        best_v = -1.0 
    if best_recall > best_v:
        model_filename = f"{PATH}/../models/{date}-{trial.study.study_name}-t{int(trial.number)}.pth"
        os.makedirs(os.path.dirname(model_filename), exist_ok=True)
        torch.save(best_model_state, model_filename)
    return best_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize SNN/GWT on Neuromorphic Dataset")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y%m%d"), help="Date for experiment tracking (default: today)")
    parser.add_argument("--exp", type=str, required=True, help="Experiment identifier (e.g., Exp1_Ablation)")
    parser.add_argument("--dataset", type=str, default="BAF-Base", choices=DATASETS, help=f"Dataset to use for training")
    parser.add_argument("--model", type=str, default="FFSNN", choices=MODELS, help="Model to use for training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--slope", type=int, default=25, help="Fixed surrogate gradient slope value")
    parser.add_argument("--gw_thr", type=float, default=None, help="GWT Threshold")
    parser.add_argument("--fixparams", action="store_true", help="Fix betas and thresholds (except GW)")
    parser.add_argument("--learn_betas", action="store_true", help="PyTorch learns betas")
    parser.add_argument("--learn_thresholds", action="store_true", help="PyTorch learns thresholds")
    parser.add_argument("--jobs", type=int, default=5, help="Number of parallel jobs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()
    DB_NAME = f"{args.date}-EXP_{args.exp}.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=args.seed, n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=f"{args.exp}-{args.dataset}-{args.model}-S{args.steps}{'-LT' if args.learn_thresholds else ''}{'-LB' if args.learn_betas else ''}{'-FIX' if args.fixparams else ''}{'-GWTHR'+str(args.gw_thr) if args.gw_thr else ''}",         
        storage=f"sqlite:///{PATH}/../results/{DB_NAME}",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: main(
            trial, args.dataset, args.model, args.epochs, args.steps, args.batch, args.slope,
            args.fixparams, args.learn_betas, args.learn_thresholds, args.gw_thr, args.date, args.device, args.seed,
        ), 
        n_jobs=args.jobs,
        n_trials=100, 
        show_progress_bar=True
    )

######################################################
# EXPERIMENT A: ABLATION STUDY (GWT vs Baseline)
######################################################
"""
python src/main.py --exp A --model FFSNN_GWT --fixparams --device cuda:1
python src/main.py --exp A --model FFSNN_GWT --fixparams --learn_betas --device cuda:0
python src/main.py --exp A --model FFSNN_GWT --fixparams --learn_thresholds --device cuda:1
python src/main.py --exp A --model FFSNN_GWT --fixparams --learn_betas --learn_thresholds --device cuda:1
#
python src/main.py --exp A --model CSNN_GWT --fixparams --device cuda:0
python src/main.py --exp A --model CSNN_GWT --fixparams --learn_betas --device cuda:0
python src/main.py --exp A --model CSNN_GWT --fixparams --learn_thresholds --device cuda:0
python src/main.py --exp A --model CSNN_GWT --fixparams --learn_betas --learn_thresholds --device cuda:0
#
python src/main.py --exp A --model FFSNN --fixparams --device cuda:2
python src/main.py --exp A --model FFSNN --fixparams --learn_betas --device cuda:0 
python src/main.py --exp A --model FFSNN --fixparams --learn_thresholds --device cuda:2
python src/main.py --exp A --model FFSNN --fixparams --learn_betas --learn_thresholds --device cuda:2
#
python src/main.py --exp A --model CSNN --fixparams --device cuda:3
python src/main.py --exp A --model CSNN --fixparams --learn_betas --device cuda:0
python src/main.py --exp A --model CSNN --fixparams --learn_thresholds --device cuda:3
python src/main.py --exp A --model CSNN --fixparams --learn_betas --learn_thresholds --device cuda:3
"""
######################################################
# EXPERIMENT B: TEMPORAL DYNAMICS
######################################################
"""
python src/main.py --exp B --model FFSNN_GWT --steps 5  --fixparams --device cuda:1
python src/main.py --exp B --model FFSNN_GWT --steps 10 --fixparams --device cuda:1
python src/main.py --exp B --model FFSNN_GWT --steps 20 --fixparams --device cuda:1
python src/main.py --exp B --model FFSNN_GWT --steps 30 --fixparams --device cuda:1
python src/main.py --exp B --model FFSNN_GWT --steps 40 --fixparams --device cuda:1
python src/main.py --exp B --model FFSNN_GWT --steps 50 --fixparams --device cuda:1
#
python src/main.py --exp B --model CSNN_GWT --steps 5  --fixparams --device cuda:0
python src/main.py --exp B --model CSNN_GWT --steps 10 --fixparams --device cuda:0
python src/main.py --exp B --model CSNN_GWT --steps 20 --fixparams --device cuda:0
python src/main.py --exp B --model CSNN_GWT --steps 30 --fixparams --device cuda:0
python src/main.py --exp B --model CSNN_GWT --steps 40 --fixparams --device cuda:0
python src/main.py --exp B --model CSNN_GWT --steps 50 --fixparams --device cuda:0
#
python src/main.py --exp B --model FFSNN --steps 5  --fixparams --device cuda:2
python src/main.py --exp B --model FFSNN --steps 10 --fixparams --device cuda:2
python src/main.py --exp B --model FFSNN --steps 20 --fixparams --device cuda:2
python src/main.py --exp B --model FFSNN --steps 30 --fixparams --device cuda:2
python src/main.py --exp B --model FFSNN --steps 40 --fixparams --device cuda:2
python src/main.py --exp B --model FFSNN --steps 50 --fixparams --device cuda:2
#
python src/main.py --exp B --model CSNN --steps 5  --fixparams --device cuda:3
python src/main.py --exp B --model CSNN --steps 10 --fixparams --device cuda:3
python src/main.py --exp B --model CSNN --steps 20 --fixparams --device cuda:3
python src/main.py --exp B --model CSNN --steps 30 --fixparams --device cuda:3
python src/main.py --exp B --model CSNN --steps 40 --fixparams --device cuda:3
python src/main.py --exp B --model CSNN --steps 50 --fixparams --device cuda:3
"""
######################################################
# EXPERIMENT C: GWT THRESHOLD SENSITIVITY
######################################################
"""
python src/main.py --exp C --model FFSNN --fixparams --device cuda:0
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 0.5
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 1.0
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 1.5
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 2.0
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 2.5
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 3.0
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:1 --gw_thr 3.5
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:2 --gw_thr 4.0
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:2 --gw_thr 4.5
python src/main.py --exp C --model FFSNN_GWT --fixparams --device cuda:2 --gw_thr 5.0
python src/main.py --exp C --model CSNN --fixparams --device cuda:0
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:2 --gw_thr 0.5
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:2 --gw_thr 1.0
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:2 --gw_thr 1.5
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:2 --gw_thr 2.0
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:3 --gw_thr 2.5
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:3 --gw_thr 3.0
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:3 --gw_thr 3.5
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:3 --gw_thr 4.0
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:3 --gw_thr 4.5
python src/main.py --exp C --model CSNN_GWT --fixparams --device cuda:3 --gw_thr 5.0
"""