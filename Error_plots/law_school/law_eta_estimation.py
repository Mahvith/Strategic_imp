"""
η(x) = P(y=1|x) Estimation for the Law School Dataset
========================================================
Target: pass_bar (passed bar exam = 1)
Positive rate: ~90.2% (highly imbalanced)

Features:
  decile1b  — 1st year class rank decile (1-10, 10=best)
  decile3   — 3rd year class rank decile (1-10, 10=best)
  lsat      — LSAT score (11-48)
  ugpa      — Undergrad GPA (1.5-4.0)
  zfygpa    — Standardized 1st year GPA
  zgpa      — Standardized cumulative GPA
  fulltime  — Full-time (1) vs part-time (2)
  fam_inc   — Family income band (1-5)
  male      — Gender (0=female, 1=male)
  racetxt   — Race (0=non-white, 1=white)
  tier      — Law school tier (1-6, higher=more prestigious)

Candidate improvable features for strategic classification:
  lsat, ugpa, decile1b, decile3, fam_inc
  (features where investment of effort plausibly increases qualification)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading and Preprocessing Law School Dataset")
print("=" * 70)

data, meta = arff.loadarff('law_dataset.arff')
df = pd.DataFrame(data)

# Decode byte strings from ARFF
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.decode('utf-8')

print(f"Raw dataset shape: {df.shape}")

# --- Encode target ---
df['y'] = (df['pass_bar'] == '1').astype(int)
print(f"Class distribution: {df['y'].value_counts().to_dict()}")
print(f"Positive (Pass bar) rate: {df['y'].mean():.3f}")
print("  ⚠ Note: Highly imbalanced — 90.2% positive")

# --- Convert categorical columns ---
df['racetxt'] = df['racetxt'].astype(int)
df['tier'] = df['tier'].astype(int)
df['fulltime_binary'] = (df['fulltime'] == 1.0).astype(int)

# --- Handle missing values ---
n_missing = df.isnull().sum().sum()
print(f"Missing values: {n_missing}")

# --- Define feature groups ---
# Improvable features: things a law school applicant/student can improve
IMPROVABLE_FEATURES = ['lsat', 'ugpa', 'decile1b', 'decile3', 'fam_inc']

# All features used for prediction
NUMERICAL_FEATURES = ['decile1b', 'decile3', 'lsat', 'ugpa',
                       'zfygpa', 'zgpa', 'fam_inc']
CATEGORICAL_FEATURES = ['fulltime_binary', 'male', 'racetxt', 'tier']

DROP_COLS = ['pass_bar', 'y', 'fulltime']  # fulltime replaced by fulltime_binary

X = df.drop(columns=DROP_COLS)
y = df['y'].values

print(f"\nFeature space: {X.shape[1]} features")
print(f"  Numerical: {NUMERICAL_FEATURES}")
print(f"  Categorical: {CATEGORICAL_FEATURES}")
print(f"  Improvable: {IMPROVABLE_FEATURES}")

# --- Preprocessing pipeline ---
# Treat tier as categorical (one-hot), others as indicated
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         CATEGORICAL_FEATURES),
    ]
)

# --- Train / Calibration / Test split ---
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
)

print(f"\nSplit sizes — Train: {len(X_train)}, Calibration: {len(X_cal)}, "
      f"Test: {len(X_test)}")
print(f"Test positive rate: {y_test.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════
# 2. MODEL TRAINING + CALIBRATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Training η(x) Estimators")
print("=" * 70)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, C=1.0, solver='lbfgs', random_state=42
    ),
    'Gradient Boosted Trees': GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=15,
        random_state=42, n_jobs=-1
    ),
}

results = {}

for name, base_model in models.items():
    print(f"\n--- {name} ---")

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('clf', base_model),
    ])
    pipe.fit(X_train, y_train)

    raw_probs = pipe.predict_proba(X_test)[:, 1]

    # Isotonic calibration
    cal_iso = CalibratedClassifierCV(pipe, method='isotonic', cv='prefit')
    cal_iso.fit(X_cal, y_cal)
    iso_probs = cal_iso.predict_proba(X_test)[:, 1]

    # Platt scaling
    cal_sig = CalibratedClassifierCV(pipe, method='sigmoid', cv='prefit')
    cal_sig.fit(X_cal, y_cal)
    sig_probs = cal_sig.predict_proba(X_test)[:, 1]

    for variant, probs in [('Raw', raw_probs),
                            ('Platt', sig_probs),
                            ('Isotonic', iso_probs)]:
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
        print(f"  {variant:10s} | Brier: {brier:.4f} | LogLoss: {ll:.4f} | "
              f"AUC: {auc:.4f} | Acc: {acc:.4f}")

    results[name] = {
        'pipe': pipe,
        'cal_iso': cal_iso,
        'cal_sig': cal_sig,
        'raw_probs': raw_probs,
        'iso_probs': iso_probs,
        'sig_probs': sig_probs,
    }

# ═══════════════════════════════════════════════════════════════════
# 3. CALIBRATION PLOTS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Calibration Reliability Diagrams")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {'Raw': '#e74c3c', 'Platt': '#3498db', 'Isotonic': '#2ecc71'}

for idx, (name, res) in enumerate(results.items()):
    ax = axes[idx]
    for variant, probs in [('Raw', res['raw_probs']),
                            ('Platt', res['sig_probs']),
                            ('Isotonic', res['iso_probs'])]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=12,
                                                 strategy='uniform')
        brier = brier_score_loss(y_test, probs)
        ax.plot(mean_pred, frac_pos, 's-', color=colors[variant],
                label=f'{variant} (Brier={brier:.4f})', markersize=4)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel('Mean predicted probability', fontsize=11)
    ax.set_ylabel('Fraction of positives', fontsize=11)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

fig.suptitle('Law School: Calibration Reliability Diagrams for η(x) Estimators',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('law_calibration_plots.png', dpi=150, bbox_inches='tight')
print("Saved: law_calibration_plots.png")

# ═══════════════════════════════════════════════════════════════════
# 4. MARGINAL MONOTONICITY VALIDATION (Assumption 1)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Validating Assumption 1 — Monotonicity of η(x)")
print("=" * 70)

best_estimators = {
    'GBT + Isotonic': results['Gradient Boosted Trees']['cal_iso'],
    'LogReg + Isotonic': results['Logistic Regression']['cal_iso'],
    'RF + Isotonic': results['Random Forest']['cal_iso'],
}

# Baseline: median numerical, mode categorical
baseline = {}
for col in NUMERICAL_FEATURES:
    baseline[col] = X_train[col].median()
for col in CATEGORICAL_FEATURES:
    baseline[col] = X_train[col].mode()[0]
baseline_df = pd.DataFrame([baseline])

# Also include non-improvable features that are interesting
ALL_FEATURES = IMPROVABLE_FEATURES + ['zfygpa', 'zgpa']
N_FEATS = len(ALL_FEATURES)

N_SAMPLE = 60
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(N_SAMPLE, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].copy()

n_rows = 2
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 10))
axes_flat = axes.flatten()

for feat_idx, feature in enumerate(ALL_FEATURES):
    ax = axes_flat[feat_idx]

    # For discrete features (decile1b, decile3, fam_inc) use actual levels
    unique_vals = sorted(X_train[feature].dropna().unique())
    if len(unique_vals) <= 15:
        feat_grid = np.array(unique_vals)
    else:
        feat_min = X_train[feature].quantile(0.02)
        feat_max = X_train[feature].quantile(0.98)
        feat_grid = np.linspace(feat_min, feat_max, 40)

    for est_name, estimator in best_estimators.items():
        x_batch = pd.concat([baseline_df] * len(feat_grid), ignore_index=True)
        x_batch[feature] = feat_grid
        eta_vals = estimator.predict_proba(x_batch)[:, 1]

        lw = 2.5 if 'GBT' in est_name else 1.5
        marker = 'o' if len(feat_grid) <= 15 else ''
        ax.plot(feat_grid, eta_vals, linewidth=lw, marker=marker, markersize=4,
                label=f'{est_name}')

    # Average marginal effect using GBT
    gbt_estimator = best_estimators['GBT + Isotonic']
    avg_eta = np.zeros(len(feat_grid))
    individual_curves = []
    for i in range(len(X_sample)):
        x_row = X_sample.iloc[[i]]
        x_batch = pd.concat([x_row] * len(feat_grid), ignore_index=True)
        x_batch[feature] = feat_grid
        etas_i = gbt_estimator.predict_proba(x_batch)[:, 1]
        avg_eta += etas_i
        individual_curves.append(etas_i)
    avg_eta /= len(X_sample)

    for curve in individual_curves:
        ax.plot(feat_grid, curve, color='gray', alpha=0.08, linewidth=0.5,
                marker=('.' if len(feat_grid) <= 15 else ''), markersize=2)

    ax.plot(feat_grid, avg_eta, 'k--', linewidth=2.5,
            marker=('s' if len(feat_grid) <= 15 else ''), markersize=4,
            label='GBT Avg. marginal')

    is_imp = feature in IMPROVABLE_FEATURES
    tag = " [IMPROVABLE]" if is_imp else " [OUTCOME]"
    color_title = 'darkgreen' if is_imp else '#555555'
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('η(x) = P(Pass Bar|x)', fontsize=11)
    ax.set_title(f'{feature}{tag}', fontsize=11, fontweight='bold', color=color_title)
    ax.grid(True, alpha=0.3)
    if feat_idx == 0:
        ax.legend(fontsize=6, loc='lower right')

    # Monotonicity check on average curve
    diffs = np.diff(avg_eta)
    n_inc = np.sum(diffs > 1e-5)
    n_dec = np.sum(diffs < -1e-5)
    total = len(diffs)
    mono_frac = n_inc / max(total, 1)
    verdict = 'YES ✓' if mono_frac > 0.85 else ('WEAK ~' if mono_frac > 0.6 else 'NO ✗')
    print(f"  {feature:15s} | ↑: {n_inc:2d} | ↓: {n_dec:2d} | "
          f"total: {total:2d} | mono: {mono_frac:.1%} | {verdict}")

# Remove unused subplot
axes_flat[7].axis('off')

fig.suptitle('Law School: Assumption 1 Validation — Monotonicity of η(x)\n'
             '(Gray: individual samples | Dashed black: average marginal | Colored: median-profile)',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('law_monotonicity_validation.png', dpi=150, bbox_inches='tight')
print("\nSaved: law_monotonicity_validation.png")

# ═══════════════════════════════════════════════════════════════════
# 5. EMPIRICAL η FROM RAW DATA (for discrete features)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Empirical η from raw data (non-parametric)")
print("=" * 70)

discrete_features = ['decile1b', 'decile3', 'fam_inc']
continuous_features = ['lsat', 'ugpa']

fig2, axes2 = plt.subplots(1, 5, figsize=(26, 5))

for idx, feature in enumerate(IMPROVABLE_FEATURES):
    ax = axes2[idx]

    if feature in discrete_features:
        # Exact empirical pass rate per level
        grouped = df.groupby(feature)['y']
        means = grouped.mean()
        counts = grouped.count()
        x_vals = sorted(df[feature].unique())
        y_vals = [means[v] for v in x_vals]
        n_vals = [counts[v] for v in x_vals]

        # Wilson CIs
        z = 1.96
        ci_lo, ci_hi = [], []
        for v in x_vals:
            n, p = counts[v], means[v]
            denom = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denom
            spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            ci_lo.append(center - spread)
            ci_hi.append(center + spread)

        ax.errorbar(x_vals, y_vals,
                     yerr=[np.array(y_vals) - np.array(ci_lo),
                           np.array(ci_hi) - np.array(y_vals)],
                     fmt='o-', capsize=3, linewidth=2, markersize=6,
                     color='#2c3e50', label='Empirical η (95% CI)')

        for i, (xv, n) in enumerate(zip(x_vals, n_vals)):
            ax.annotate(f'n={n}', (xv, y_vals[i]),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=7, color='gray')

        # GBT median-profile overlay
        gbt_vals = []
        for val in x_vals:
            x_q = baseline_df.copy()
            x_q[feature] = val
            gbt_vals.append(gbt_estimator.predict_proba(x_q)[0, 1])
        ax.plot(x_vals, gbt_vals, 's--', color='#e74c3c',
                linewidth=2, markersize=5, label='GBT η (median)')

        diffs = np.diff(y_vals)
        is_mono = all(d > 0 for d in diffs)
        print(f"  {feature:15s} | η = {[f'{v:.3f}' for v in y_vals]}")
        print(f"  {'':15s} | Strictly monotone: {'YES ✓' if is_mono else 'NO ✗'}")

    else:
        # Bin continuous features and compute pass rate per bin
        n_bins = 15
        bins = pd.cut(df[feature], bins=n_bins)
        grouped = df.groupby(bins, observed=True)['y']
        means = grouped.mean()
        counts = grouped.count()
        midpoints = [(iv.left + iv.right) / 2 for iv in means.index]
        y_vals = means.values
        n_vals = counts.values

        # Wilson CIs
        z = 1.96
        ci_lo, ci_hi = [], []
        for p_val, n_val in zip(y_vals, n_vals):
            denom = 1 + z**2 / n_val
            center = (p_val + z**2 / (2*n_val)) / denom
            spread = z * np.sqrt((p_val*(1-p_val) + z**2/(4*n_val)) / n_val) / denom
            ci_lo.append(center - spread)
            ci_hi.append(center + spread)

        ax.errorbar(midpoints, y_vals,
                     yerr=[y_vals - np.array(ci_lo),
                           np.array(ci_hi) - y_vals],
                     fmt='o-', capsize=3, linewidth=2, markersize=5,
                     color='#2c3e50', label='Empirical η (95% CI)')

        # GBT median-profile overlay
        feat_grid = np.linspace(df[feature].quantile(0.02),
                                 df[feature].quantile(0.98), 40)
        x_batch = pd.concat([baseline_df] * len(feat_grid), ignore_index=True)
        x_batch[feature] = feat_grid
        gbt_vals = gbt_estimator.predict_proba(x_batch)[:, 1]
        ax.plot(feat_grid, gbt_vals, '-', color='#e74c3c',
                linewidth=2, label='GBT η (median)')

        diffs = np.diff(y_vals)
        n_inc = np.sum(diffs > 0)
        is_mono = n_inc == len(diffs)
        mono_pct = n_inc / len(diffs)
        print(f"  {feature:15s} | {n_inc}/{len(diffs)} bins increasing "
              f"({mono_pct:.0%}) | {'YES ✓' if is_mono else 'WEAK ~' if mono_pct > 0.8 else 'NO ✗'}")

    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('η(x) = P(Pass Bar|x)', fontsize=11)
    ax.set_title(f'{feature}', fontsize=12, fontweight='bold', color='darkgreen')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

fig2.suptitle('Law School: Empirical η(x) = P(Pass Bar|x) from Raw Data\n'
              'with 95% Wilson Confidence Intervals',
              fontsize=14, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('law_empirical_eta.png', dpi=150, bbox_inches='tight')
print("\nSaved: law_empirical_eta.png")

# ═══════════════════════════════════════════════════════════════════
# 6. CONDITIONAL MONOTONICITY CHECKS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Conditional Monotonicity Checks")
print("=" * 70)

# For each improvable feature, vary it while conditioning on
# random combinations of the other improvable features
gbt_estimator = best_estimators['GBT + Isotonic']

for target_feat in IMPROVABLE_FEATURES:
    other_feats = [f for f in IMPROVABLE_FEATURES if f != target_feat]

    unique_vals = sorted(X_train[target_feat].dropna().unique())
    if len(unique_vals) > 20:
        feat_grid = np.linspace(X_train[target_feat].quantile(0.05),
                                 X_train[target_feat].quantile(0.95), 20)
    else:
        feat_grid = np.array(unique_vals)

    # Sample conditioning contexts from real data
    np.random.seed(42)
    n_contexts = 80
    context_idx = np.random.choice(len(X_train), n_contexts, replace=False)
    contexts = X_train.iloc[context_idx]

    violations = 0
    total_checks = 0
    for _, ctx_row in contexts.iterrows():
        x_batch = pd.concat([pd.DataFrame([ctx_row])] * len(feat_grid), ignore_index=True)
        x_batch[target_feat] = feat_grid
        etas = gbt_estimator.predict_proba(x_batch)[:, 1]
        diffs = np.diff(etas)
        violations += np.sum(diffs < -1e-4)
        total_checks += len(diffs)

    viol_rate = violations / max(total_checks, 1)
    verdict = 'PASS' if viol_rate < 0.10 else ('CONCERN' if viol_rate < 0.25 else 'FAIL')
    print(f"  {target_feat:15s} | violations: {violations}/{total_checks} "
          f"({viol_rate:.1%}) | {verdict}")

# ═══════════════════════════════════════════════════════════════════
# 7. CORRELATION BETWEEN η ESTIMATORS (Agreement check)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Cross-model Agreement on η(x)")
print("=" * 70)

from sklearn.metrics import mean_absolute_error

# Compare η predictions on test set across models
eta_predictions = {}
for name, res in results.items():
    eta_predictions[f'{name} (Iso)'] = res['iso_probs']

model_names = list(eta_predictions.keys())
print(f"\n{'':30s}", end='')
for n in model_names:
    print(f"{n[:12]:>14s}", end='')
print()

for i, n1 in enumerate(model_names):
    print(f"{n1:30s}", end='')
    for j, n2 in enumerate(model_names):
        corr = np.corrcoef(eta_predictions[n1], eta_predictions[n2])[0, 1]
        print(f"{'':>4s}{corr:.4f}    ", end='')
    print()

print("\nMean Absolute Difference between estimators on test set:")
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        mae = mean_absolute_error(eta_predictions[model_names[i]],
                                    eta_predictions[model_names[j]])
        print(f"  {model_names[i][:20]:20s} vs {model_names[j][:20]:20s}: MAE = {mae:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 8. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Summary — All Models")
print("=" * 70)

summary_rows = []
for name, res in results.items():
    for variant, probs in [('Raw', res['raw_probs']),
                            ('Platt', res['sig_probs']),
                            ('Isotonic', res['iso_probs'])]:
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        summary_rows.append({
            'Model': name, 'Calibration': variant,
            'Brier ↓': round(brier, 4),
            'LogLoss ↓': round(ll, 4),
            'AUC ↑': round(auc, 4),
        })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))

best_row = summary_df.loc[summary_df['Brier ↓'].idxmin()]
print(f"\n★ Best calibrated (lowest Brier): {best_row['Model']} + {best_row['Calibration']} "
      f"(Brier={best_row['Brier ↓']})")

# ═══════════════════════════════════════════════════════════════════
# 9. SAVE ESTIMATORS AND DATA
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: Saving Estimators + Processed Data")
print("=" * 70)

import pickle

for model_name, key in [('gbt', 'Gradient Boosted Trees'),
                          ('lr', 'Logistic Regression'),
                          ('rf', 'Random Forest')]:
    with open(f'law_eta_{model_name}_isotonic.pkl', 'wb') as f:
        pickle.dump(results[key]['cal_iso'], f)
    print(f"Saved: law_eta_{model_name}_isotonic.pkl")

processed = {
    'X_train': X_train, 'y_train': y_train,
    'X_cal': X_cal, 'y_cal': y_cal,
    'X_test': X_test, 'y_test': y_test,
    'IMPROVABLE_FEATURES': IMPROVABLE_FEATURES,
    'NUMERICAL_FEATURES': NUMERICAL_FEATURES,
    'CATEGORICAL_FEATURES': CATEGORICAL_FEATURES,
}
with open('law_processed_data.pkl', 'wb') as f:
    pickle.dump(processed, f)
print("Saved: law_processed_data.pkl")

print("\n" + "=" * 70)
print("DONE — Law School η estimation complete.")
print("=" * 70)
