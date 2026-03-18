import random
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import arff

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
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
# 1. DATA PREPARATION (LAW SCHOOL DATASET)
# =============================================================================
def load_and_process_lawschool(datapath='law_school_clean.csv'):
    df = pd.read_csv('law_school_clean.csv')
    target = 'pass_bar'
    non_manipulable = ['fulltime', 'male', 'racetxt', 'race', 'fam_inc', 'tier']
    feature_cols = [c for c in df.columns if c not in [target] + non_manipulable]
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_cols

def generate_synthetic_data():
    X = np.random.randn(1000, 5)
    w_true = np.array([1.0, 2.0, -0.5, 0.5, 1.0])
    p = 1 / (1 + np.exp(-(X @ w_true)))
    y = np.random.binomial(1, p)
    feature_cols = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_cols

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

    # Re-implemented the WeightedRandomSampler logic
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
    
    return train_dl, val_dl

# =============================================================================
# 2. MODELS & UTILS
# =============================================================================
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.out(x)

def estimate_eta(X_tr, y_tr):
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
    def __init__(self, model, train_dl, opt, C=1.0):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.C = C

    def loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        margin = y_tilde * pred
        hinge = torch.relu(1 - margin).mean()
        reg = 0.5 * torch.sum(self.model.out.weight ** 2)
        return self.C * hinge + reg

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
    def __init__(self, model, train_dl, opt, alpha, beta, C=1.0):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.alpha = alpha.clone().detach().to(device)
        self.beta = beta
        self.C = C

    def strat_loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde * (pred + s_gain)
        hinge = torch.relu(1 - margin).mean()
        reg = 0.5 * torch.sum(w ** 2)
        return self.C * hinge + reg

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
    def __init__(self, model, train_dl, opt, eta_model, alpha, beta, C=1.0):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.eta_model = eta_model
        self.alpha_np = alpha.cpu().numpy()
        self.alpha_t = alpha.clone().detach().to(device)
        self.beta = beta
        self.C = C

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
        hinge = torch.relu(1 - margin).mean()
        # reg = 0.5 * torch.sum(w ** 2)        
        # return self.C * hinge + reg
        
        return hinge

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
def compute_err_imp(model, X_np, eta_model_oracle, alpha_np, beta):
    model.eval()
    w = model.out.weight.detach().cpu().numpy().flatten()
    b = model.out.bias.detach().cpu().item()
    
    # 1. Test set reacts strategically to the model
    X_strat = get_strategic_response_batch(X_np, w, b, alpha_np, beta)
    
    # 2. Model makes predictions on the manipulated features
    scores = X_strat @ w + b
    preds = (scores >= 0).astype(int)
     
    # 3. Calculate expected error against the *Oracle* true probabilities
    etas = eta_model_oracle.predict_proba(X_strat)[:, 1]
    errors = preds * (1 - etas) + (1 - preds) * etas
    
    return np.mean(errors)

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    set_seed(42)
    
    try:
        X_train, X_test, y_train, y_test, feature_cols = load_and_process_lawschool('law_dataset.arff')
    except Exception as e:
        print(f"Data loading error: {e}. Falling back to synthetic data.")
        X_train, X_test, y_train, y_test, feature_cols = generate_synthetic_data()

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    d = X_train_sc.shape[1]
    
    # -------------------------------------------------------------------------
    # ORACLE SETUP
    # -------------------------------------------------------------------------
    print("\nTraining ORACLE Eta Model on full training set...")
    eta_model_oracle = estimate_eta(X_train_sc, y_train)
    
    # Experiment Constants
    fixed_beta = 0.01
    
    # DYNAMIC ALPHA SIZING for Law School
    # cost_map = {
    #     'lsat': 250,     
    #     'ugpa': 250,    
    #     'zfygpa': 25,  
    #     'zgpa': 25,    
    #     'decile1b': 100, 
    #     'decile3': 100,  
    #     'fam_inc': 25.0,  
    #     'tier': 100.0     
    # }
    cost_map = {
    'lsat': 1.0,      # Prep courses can help, but hard
    'ugpa': 0.5,     # Immutable once graduated
    'zfygpa': 1.0,   # Immutable past 1st year
    'zgpa': 1.0,       # Can be somewhat improved before graduation
    'decile1b': 10.0,  # Zero-sum class rank, very hard
    'decile3': 10.0,   # Zero-sum class rank, very hard
    }
    alpha_vals = [cost_map[f] for f in cost_map]
    print(f"Assigned Alpha Costs: {dict(zip(feature_cols, alpha_vals))}")
    
    fixed_alpha = torch.tensor(alpha_vals, dtype=torch.float32, device=device)
    fixed_alpha_np = fixed_alpha.cpu().numpy()
    
    # Define 10 varying sizes for the training subset
    max_train_size = len(X_train_sc)
    min_train_size = max(500, max_train_size // 20)
    train_sizes = np.linspace(min_train_size, max_train_size, 10).astype(int)
    
    # Define number of random splits per sample size
    num_runs = 5
    
    # Tracking metrics
    mean_b, std_b = [], []
    mean_s, std_s = [], []
    mean_i, std_i = [], []
    
    print("\n" + "="*85)
    print(f"{'STARTING SAMPLE COMPLEXITY EXPERIMENT (AVERAGED OVER ' + str(num_runs) + ' RUNS)':^85}")
    print("="*85)
    
    for n in train_sizes:
        print(f"\n--- Training on Subset Size: {n} / {max_train_size} ---")
        
        runs_err_b, runs_err_s, runs_err_i = [], [], []
        
        for run in range(num_runs):
            print(f"  > Run {run + 1}/{num_runs}...")
            
            X_tr_sub, y_tr_sub = resample(
                X_train_sc, y_train, 
                n_samples=n, 
                replace=True, 
                random_state=42 + run
            )
            
            train_dl_sub, _ = build_dataloaders(X_tr_sub, y_tr_sub)
            eta_model_sub = estimate_eta(X_tr_sub, y_tr_sub)
            
            model_bayes = LinearClassifier(d, 1).to(device)
            opt_b = torch.optim.Adam(model_bayes.parameters(), lr=0.01)
            trainer_b = BayesTrainer(model_bayes, train_dl_sub, opt_b, C=1.0)
            trainer_b.train(n_epochs=50) 
            
            model_strat = LinearClassifier(d, 1).to(device)
            # model_strat.load_state_dict(model_bayes.state_dict())
            opt_s = torch.optim.Adam(model_strat.parameters(), lr=0.01)
            trainer_s = StrategicTrainer(model_strat, train_dl_sub, opt_s, fixed_alpha, fixed_beta, C=1.0)
            trainer_s.train(n_epochs=50)
            
            model_imp = LinearClassifier(d, 1).to(device)
            # model_imp.load_state_dict(model_bayes.state_dict())
            opt_imp = torch.optim.Adam(model_imp.parameters(), lr=0.01)
            trainer_imp = ImprovementAwareTrainer(model_imp, train_dl_sub, opt_imp, eta_model_sub, fixed_alpha, fixed_beta, C=1.0)
            trainer_imp.train(n_epochs=80) 
            
            err_b = compute_err_imp(model_bayes, X_test_sc, eta_model_oracle, fixed_alpha_np, fixed_beta)
            err_s = compute_err_imp(model_strat, X_test_sc, eta_model_oracle, fixed_alpha_np, fixed_beta)
            err_i = compute_err_imp(model_imp, X_test_sc, eta_model_oracle, fixed_alpha_np, fixed_beta)
            
            runs_err_b.append(err_b)
            runs_err_s.append(err_s)
            runs_err_i.append(err_i)
            
        mean_b.append(np.mean(runs_err_b))
        std_b.append(np.std(runs_err_b))
        
        mean_s.append(np.mean(runs_err_s))
        std_s.append(np.std(runs_err_s))
        
        mean_i.append(np.mean(runs_err_i))
        std_i.append(np.std(runs_err_i))
        
        print(f"  [Aggregated Results for N={n} over {num_runs} runs]")
        print(f"    err_imp(f*)     = {mean_b[-1]:.4f} ± {std_b[-1]:.4f}")
        print(f"    err_imp(f*_s)   = {mean_s[-1]:.4f} ± {std_s[-1]:.4f}")
        print(f"    err_imp(f*_imp) = {mean_i[-1]:.4f} ± {std_i[-1]:.4f}")

    # =============================================================================
    # 7. PLOTTING THE RESULTS
    # =============================================================================
    print("\nGenerating sample complexity plots with variance bounds...")
    
    mean_b, std_b = np.array(mean_b), np.array(std_b)
    mean_s, std_s = np.array(mean_s), np.array(std_s)
    mean_i, std_i = np.array(mean_i), np.array(std_i)
    
    plt.figure(figsize=(10, 6))
     
    plt.plot(train_sizes, mean_b, marker='o', linestyle='-', linewidth=2, color='tab:blue', label=r'Bayes Optimal ($f^*$)')
    plt.fill_between(train_sizes, mean_b - std_b, mean_b + std_b, color='tab:blue', alpha=0.2)
    
    plt.plot(train_sizes, mean_s, marker='s', linestyle='-', linewidth=2, color='tab:orange', label=r'Strategic ($f_s^*$)')
    plt.fill_between(train_sizes, mean_s - std_s, mean_s + std_s, color='tab:orange', alpha=0.2)
    
    plt.plot(train_sizes, mean_i, marker='^', linestyle='-', linewidth=2, color='tab:green', label=r'Improvement-Aware ($f_{imp}^*$)')
    plt.fill_between(train_sizes, mean_i - std_i, mean_i + std_i, color='tab:green', alpha=0.2)
    
    plt.xlabel('Number of Training Samples N', fontsize=12)
    plt.ylabel('Improvement Error (err_imp)', fontsize=12)
    plt.title(f'Sample Complexity vs Improvement Error (Law School Dataset, beta=1.0)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('sample_complexity_curves_lawschool.png', dpi=300)
    plt.show()
    print("Plot saved as 'sample_complexity_curves_lawschool.png'.")