import random
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 0. REPRODUCIBILITY & UTILS
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# 1. DATA PREPARATION (LAWSCHOOL)
# =============================================================================
def load_and_process_lawschool(datapath='law_dataset.arff'):
    """
    Loads Lawschool data locally from ARFF format.
    One-hot encodes the categorical 'tier' feature.
    Scales real features to a [-1, 1] range.
    Isolates manipulable features from non-manipulable ones.
    """
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"{datapath} not found. Please ensure the file is in the working directory.")
        
    print(f"Loading local {datapath}...")
    with open(datapath, 'r') as f:
        lines = f.readlines()
        
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == '@data':
            data_start = i + 1
            break
            
    # Load into a pandas DataFrame
    df = pd.read_csv(datapath, skiprows=data_start, header=None)
    
    # Define columns exactly as ordered in the ARFF attribute metadata
    columns = [
        'decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 
        'fulltime', 'fam_inc', 'male', 'racetxt', 'tier', 'pass_bar'
    ]
    df.columns = columns
    
    # 1. Separate Target Variable
    y = df['pass_bar'].values
    X_df = df.drop(columns=['pass_bar'])
    
    # 2. Preprocess Categorical Feature ('tier')
    # One-hot encode the tier column
    X_df = pd.get_dummies(X_df, columns=['tier'], prefix='tier', dtype=float)
    tier_cols = [col for col in X_df.columns if col.startswith('tier_')]
    
    # 3. Define Manipulable vs. Non-Manipulable Features
    # Real continuous features to be scaled
    real_features = ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fam_inc']
    
    # Manipulable features include the real features and all the new one-hot encoded tier columns
    manipulable_features = real_features + tier_cols
    non_manipulable_features = ['fulltime', 'male', 'racetxt'] 
    
    # Reorder columns so manipulable features are grouped together
    ordered_cols = manipulable_features + non_manipulable_features
    X_df = X_df[ordered_cols]
    
    # Create a mask vector (1.0 for manipulable, 0.0 for non-manipulable)
    manipulable_mask = np.array(
        [1.0] * len(manipulable_features) + [0.0] * len(non_manipulable_features), 
        dtype=np.float32
    )
    
    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create explicit copies to prevent SettingWithCopyWarnings during scaling
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # 5. Scale Real Features to [-1, 1]
    # Fit scaler ONLY on the training data to prevent data leakage
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train.loc[:, real_features] = scaler.fit_transform(X_train[real_features])
    X_test.loc[:, real_features] = scaler.transform(X_test[real_features])
    
    print(f"Using {len(manipulable_features)} manipulable features (including OHE tiers) and {len(non_manipulable_features)} non-manipulable features.")
    
    return X_train.values, X_test.values, y_train, y_test, manipulable_mask

def generate_synthetic_data(d=16):
    print("Generating synthetic fallback data...")
    X = np.random.randn(1000, d)
    w_true = np.random.randn(d)
    p = 1 / (1 + np.exp(-(X @ w_true)))
    y = np.random.binomial(1, p)
    # Assume first 13 are manipulable, rest are not (matching Lawschool with OHE tiers)
    mask = np.array([1.0]*13 + [0.0]*(d-13), dtype=np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te, mask

def build_dataloaders(X_scaled, Y, batch_size=64):
    X = X_scaled.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)

    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    X_train_t = torch.from_numpy(X_train_np)
    Y_train_t = torch.from_numpy(Y_train_np)
    X_val_t = torch.from_numpy(X_val_np)
    Y_val_t = torch.from_numpy(Y_val_np)

    n_pos = torch.sum(Y_train_t == 1).item()
    n_neg = torch.sum(Y_train_t == 0).item()
    
    if n_pos == 0 or n_neg == 0: 
        weights = torch.ones_like(Y_train_t).view(-1)
    else:
        w_pos = 1.0 / n_pos
        w_neg = 1.0 / n_neg
        weights = torch.where(Y_train_t.view(-1) == 1, w_pos, w_neg)

    sampler = WeightedRandomSampler(weights, num_samples=len(Y_train_t), replacement=True)

    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, sampler=sampler)
    val_dl = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, X_train_t

# =============================================================================
# 2. MODELS & UTILS
# =============================================================================
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.out(x)

def estimate_eta(X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([('clf', LogisticRegression(solver='lbfgs', max_iter=1000, C=10, random_state=42))])
    cal = CalibratedClassifierCV(pipe, cv=3, method='isotonic')
    cal.fit(X_tr, y_tr)
    return cal

# =============================================================================
# 3. STRATEGIC LOGIC (Helpers)
# =============================================================================
def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
    batch_size, d = X_batch_np.shape
    X_manip = X_batch_np.copy()
    
    # Calculate ROI (Return on Investment) for each feature.
    # Non-manipulable features have alpha=1e9, so their ROI approaches 0.
    roi = w_np / (alpha_np + 1e-12)
    k_star = np.argmax(roi)
    
    scores = X_batch_np @ w_np + b_np
    neg_indices = np.where(scores < 0)[0]
    
    if len(neg_indices) > 0:
        gap = -scores[neg_indices] 
        w_k = w_np[k_star]
        
        if w_k > 1e-8:
            delta_k = gap / w_k
            cost = alpha_np[k_star] * delta_k
            
            do_manipulate = cost <= beta
            
            idx_to_change = neg_indices[do_manipulate]
            delta_to_apply = delta_k[do_manipulate]
            
            X_manip[idx_to_change, k_star] += (delta_to_apply + 1e-5)
            
    return X_manip

# =============================================================================
# 4. TRAINERS
# =============================================================================
class BayesTrainer:
    def __init__(self, model, train_dl, opt):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt

    def loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        margin = y_tilde * pred
        return torch.relu(1 - margin).mean()

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs), desc="Training Bayes Optimal (f*)", leave=False):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.loss(xb, y_tilde)
                l.backward()
                self.opt.step()

class StrategicTrainer:
    def __init__(self, model, train_dl, opt, alpha, beta):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.alpha = alpha.clone().detach().to(device)
        self.beta = beta

    def strat_loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde * (pred + s_gain)
        return torch.relu(1 - margin).mean()

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs), desc="Training Strategic (f*_s)", leave=False):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.strat_loss(xb, y_tilde)
                l.backward()
                self.opt.step()

class ImprovementAwareTrainer:
    def __init__(self, model, train_dl, opt, eta_model, alpha, beta):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.eta_model = eta_model
        self.alpha_np = alpha.cpu().numpy()
        self.alpha_t = alpha.clone().detach().to(device)
        self.beta = beta

    def train_step(self, xb, yb):
        w_curr = self.model.out.weight.detach().cpu().numpy().flatten()
        b_curr = self.model.out.bias.detach().cpu().item()
        
        xb_np = xb.cpu().numpy()
        X_strat = get_strategic_response_batch(xb_np, w_curr, b_curr, self.alpha_np, self.beta)
        
        probs_new = self.eta_model.predict_proba(X_strat)[:, 1]
        y_imp_np = np.random.binomial(1, probs_new)
        
        y_imp_t = torch.from_numpy(y_imp_np).float().to(device).view(-1, 1)
        y_tilde_imp = (2 * y_imp_t - 1).squeeze()
        
        pred = self.model(xb.to(device)).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha_t + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde_imp * (pred + s_gain)
        loss = torch.relu(1 - margin).mean()
        
        return loss

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs), desc="Training Improvement-Aware (f*_imp)", leave=False):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                loss = self.train_step(xb, yb)
                loss.backward()
                self.opt.step()

# =============================================================================
# 5. EVALUATION METRIC
# =============================================================================
def compute_err_imp(model, X_np, eta_model, alpha_np, beta):
    model.eval()
    w = model.out.weight.detach().cpu().numpy().flatten()
    b = model.out.bias.detach().cpu().item()
    
    X_strat = get_strategic_response_batch(X_np, w, b, alpha_np, beta)
    
    scores = X_strat @ w + b
    preds = (scores >= 0).astype(int)
    
    etas = eta_model.predict_proba(X_strat)[:, 1]
    errors = preds * (1 - etas) + (1 - preds) * etas
    
    return np.mean(errors)

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    set_seed(42)
    
    # 1. Load Lawschool Dataset and Extact Non-Manipulable Mask
    try:
        X_train, X_test, y_train, y_test, manipulable_mask = load_and_process_lawschool('law_dataset.arff')
    except Exception as e:
        print(f"Data loading error: {e}")
        X_train, X_test, y_train, y_test, manipulable_mask = generate_synthetic_data()

    # Data is already scaled in the loader! Directly assign to variables.
    X_train_sc = X_train
    X_test_final = X_test
    
    train_dl, val_dl, X_train_t = build_dataloaders(X_train_sc, y_train)
    d = X_train_sc.shape[1]
    
    print("\nTraining Eta (True Probability) Model...")
    eta_model = estimate_eta(X_train_sc, y_train, X_test_final, y_test)
    
    # -------------------------------------------------------------
    # Train Bayes Optimal ONCE (Independent of alpha and beta)
    # -------------------------------------------------------------
    print("\nTraining Bayes Optimal (f*) Model...")
    model_bayes = LinearClassifier(d, 1).to(device)
    opt_b = torch.optim.SGD(model_bayes.parameters(), lr=0.01, momentum=0.9)
    trainer_b = BayesTrainer(model_bayes, train_dl, opt_b)
    trainer_b.train(n_epochs=100) 
    
    # Extract Bayes weights for bias shift calculations
    w_star_bayes = model_bayes.out.weight.squeeze().detach()
    b_star_bayes = model_bayes.out.bias.item()
    
    # Define test grids for beta and dynamic alpha mapped to manipulable mask
    betas_to_test = [0.2, 1.0, 20.0]
    
    base_alphas = {
        "Uniform (1.0)": 1.0,
        "Uniform (2.0)": 2.0,
        "Uniform (5.0)": 5.0,
        "Uniform (20.0)": 20.0,
    }
    
    alpha_configs = {}
    for name, val in base_alphas.items():
        # If manipulable -> set to base alpha value. 
        # If not manipulable -> set to 1e9 (infinity) to zero out strategic ROI.
        alpha_tensor = torch.where(
            torch.tensor(manipulable_mask) == 1.0, 
            torch.tensor(val), 
            torch.tensor(1e9)
        )
        alpha_configs[name] = alpha_tensor.to(device)
    
    print("\n" + "="*80)
    print(f"{'VERIFICATION OF BIAS SHIFT AND ERROR ORDERING':^80}")
    print("="*80)
    
    results = []
    
    for beta in betas_to_test:
        print(f"\n{'='*60}")
        print(f" TESTING BETA = {beta}")
        print(f"{'='*60}")
        
        for name, alpha in alpha_configs.items():
            # Show configuration (excluding the 1e9 infinite costs for readability)
            display_alpha = alpha[manipulable_mask == 1.0].cpu().numpy().round(1)[0]
            print(f"\n--- Configuration: {name} (Manipulable Base Alpha: {display_alpha}) ---")
            
            # A. Evaluate Bayes Optimal (f*) under current (alpha, beta) conditions
            err_bayes = compute_err_imp(model_bayes, X_test_final, eta_model, alpha.cpu().numpy(), beta)
            
            # B. Train Strategic Optimal (f*_s)
            model_strat = LinearClassifier(d, 1).to(device)
            model_strat.load_state_dict(model_bayes.state_dict()) # Warm start
            opt_s = torch.optim.SGD(model_strat.parameters(), lr=0.01, momentum=0.9)
            trainer_s = StrategicTrainer(model_strat, train_dl, opt_s, alpha, beta)
            trainer_s.train(n_epochs=100)
            
            err_strat = compute_err_imp(model_strat, X_test_final, eta_model, alpha.cpu().numpy(), beta)

            # -------------------------------------------------------------
            # VERIFICATION 1: Theoretical Bias Shift (b_s* = b* - S(w*))
            # -------------------------------------------------------------
            b_strat_trained = model_strat.out.bias.item()
            S_w_star = beta * torch.max(w_star_bayes / (alpha + 1e-12)).item()
            b_strat_theory = b_star_bayes - S_w_star

            print(f"\n  [Bias Shift Verification]")
            print(f"    > Bayes Optimal b* : {b_star_bayes:.5f}")
            print(f"    > Theoretical b_s* : {b_strat_theory:.5f}")
            print(f"    > SGD Trained b_s         : {b_strat_trained:.5f}")
            print(f"    > Absolute Difference     : {abs(b_strat_theory - b_strat_trained):.5f}")
            
            # C. Train Improvement-Aware Optimal (f*_imp)
            model_imp = LinearClassifier(d, 1).to(device)
            model_imp.load_state_dict(model_bayes.state_dict()) # Warm start
            opt_imp = torch.optim.SGD(model_imp.parameters(), lr=0.01, momentum=0.9)
            trainer_imp = ImprovementAwareTrainer(model_imp, train_dl, opt_imp, eta_model, alpha, beta)
            trainer_imp.train(n_epochs=130) 
            
            err_imp = compute_err_imp(model_imp, X_test_final, eta_model, alpha.cpu().numpy(), beta)
            
            # -------------------------------------------------------------
            # VERIFICATION 2: Error Ordering
            # -------------------------------------------------------------
            check1 = err_strat <= (err_bayes + 1e-4) 
            check2 = err_imp <= (err_strat + 1e-4)   
            
            print(f"\n  [Error Ordering Verification]")
            print(f"    > Bayes Err (f*)          : {err_bayes:.5f}")
            print(f"    > Strat Err (f*_s)        : {err_strat:.5f}")
            print(f"    > Imp-Aware Err (f*_imp)  : {err_imp:.5f}")
            print(f"    > f*_imp <= f*_s <= f* holds? {'Yes' if check1 and check2 else 'No'}")

            results.append({
                "Beta": beta,
                "Config": name,
                "err_imp(f*)": err_bayes,
                "err_imp(f*_s)": err_strat,
                "err_imp(f*_imp)": err_imp,
                "f*_s <= f*": "PASS" if check1 else "FAIL",
                "f*_imp <= f*_s": "PASS" if check2 else "FAIL",
                "bias_diff": abs(b_strat_theory - b_strat_trained)
            })

    # --- 5. Summary Table ---
    print("\n" + "="*105)
    print(f"{'FINAL RESULTS SUMMARY':^105}")
    print("="*105)
    print(f"{'Beta':<5} | {'Config':<15} | {'f* (Bayes)':<10} | {'f*_s (Strat)':<12} | {'f*_imp (Ours)':<13} | {'Order Checks':<12} | {'Bias Diff':<10}")
    print("-" * 105)
    for r in results:
        checks = f"{r['f*_s <= f*']} / {r['f*_imp <= f*_s']}"
        print(f"{r['Beta']:<5.1f} | {r['Config']:<15} | {r['err_imp(f*)']:<10.5f} | {r['err_imp(f*_s)']:<12.5f} | {r['err_imp(f*_imp)']:<13.5f} | {checks:<12} | {r['bias_diff']:.5f}")
    print("-" * 105)