import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def data_split(X, y, test_size, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def build_dataloaders(X, Y, batch_size=64):
    X = X.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)

    X_train_t = torch.from_numpy(X)
    Y_train_t = torch.from_numpy(Y)
    
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
        
    return train_dl

def build_dataloaders_old(X, Y, batch_size=64):
    X = X.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)

    X_train_t = torch.from_numpy(X)
    Y_train_t = torch.from_numpy(Y)
    
    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle = True)     
       
    return train_dl


def load_and_process_adult(datapath):
    """
    Loads Adult data locally, or fetches from OpenML.
    Restricts to 5 continuous features to match the 5D alpha vectors.
    """
    df = pd.read_csv(datapath)
    df = df.replace('?', np.nan).dropna().drop_duplicates(keep="first")
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    # Use the 5 continuous features to align with the 5D alpha configurations
    features = ['educational-num', "capital-gain", "capital-loss", "hours-per-week", "age"]
    
    X = df[features].values
    y = df['income'].values
    
    return X, y

def load_and_process_heloc(datapath='heloc.csv'):
    # Load dataset
    df = pd.read_csv(datapath)
    
    # Target mapping
    df['Target'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
    
    # Defined improvable features
    
    features = [
        "ExternalRiskEstimate",
        "MSinceOldestTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MSinceMostRecentInqexcl7days",
    ]
    
    # Handle HELOC special values (-7, -8, -9) 
    X_raw = df[features].replace([-7, -8, -9], np.nan)
    y = df['Target'].values
    
    # Impute missing values with median to preserve clean matrices for PyTorch
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)
    
    return X, y

def load_and_process_oulad(datapath):
    """
    Loads and processes the OULAD dataset based on user mapping.
    """
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Dataset {datapath} not found in the current directory.")

    # 1. Load Data
    print(f"Loading {datapath}...")
    df = pd.read_csv(datapath)

    # Drop student ID
    if 'id_student' in df.columns:
        df = df.drop(columns=['id_student'])

    # 2. Correct Mappings for Ordinal Features
    if 'age_band' in df.columns:
        age_map = {'0-35': 0, '35-55': 1, '55<=': 2}
        df['age_band'] = df['age_band'].map(age_map)

    if 'imd_band' in df.columns:
        imd_bands = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%',
                     '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        imd_map = {band: i for i, band in enumerate(imd_bands)}
        df['imd_band'] = df['imd_band'].map(imd_map)

    if 'highest_education' in df.columns:
        edu_map = {
            'No Formal quals': 0,
            'Lower Than A Level': 1,
            'A Level or Equivalent': 2,
            'HE Qualification': 3,
            'Post Graduate Qualification': 4
        }
        df['highest_education'] = df['highest_education'].map(edu_map)

    # Drop rows with NaN values created by mapping (if any unmapped values existed)
    df = df.dropna()

    # 3. Label Encoding for Nominal Features & Target
    if 'final_result' in df.columns:
        # Assuming binary setup for Pass vs Fail/Withdrawn. 
        # If your data has Distinction, map it to Pass (1) as well, or filter.
        # Here we just map exactly as requested, dropping others.
        df['final_result'] = df['final_result'].map({'Pass': 1, 'Fail': 0, 'Distinction': 1, 'Withdrawn': 0})
        df = df.dropna(subset=['final_result'])

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # 4. Split
    X = df.drop(columns=['final_result'])
    y = df['final_result']
    
    return X.values, y.values

def load_and_process_german(datapath):
    """
    Loads German Credit data locally from the space-separated numeric format.
    Scales features to a [-1, 1] range.
    Isolates manipulable features from non-manipulable ones via a boolean mask.
    """
        
    print(f"Loading local {datapath}...")
    
    # Read the space-separated numeric data
    df = pd.read_csv(datapath, delim_whitespace=True, header=None)
    
    # 1. Separate Target Variable
    # Column 24 contains the target: 1 for Good Credit, 2 for Bad Credit
    # We map this to standard binary: 1 (Good) and 0 (Bad)
    y = np.where(df[24].values == 1, 1, 0)
    X_df = df.drop(columns=[24])
    
    # 2. Define Manipulable vs. Non-Manipulable Features
    # The numeric dataset has 24 features (columns 0 to 23).
    # Since specific labels are omitted in the numeric version, we assume a 
    # default split (e.g., first 12 manipulable, remaining 12 non-manipulable).
    num_features = X_df.shape[1]
    num_manipulable = 12
    num_non_manipulable = num_features - num_manipulable
    
    # Create a mask vector (1.0 for manipulable, 0.0 for non-manipulable)
    manipulable_mask = np.array(
        [1.0] * num_manipulable + [0.0] * num_non_manipulable, 
        dtype=np.float32
    )
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create explicit copies to prevent SettingWithCopyWarnings during scaling
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # 4. Scale All Features to [-1, 1]
    # Fit scaler ONLY on the training data to prevent data leakage
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train.loc[:, :] = scaler.fit_transform(X_train)
    X_test.loc[:, :] = scaler.transform(X_test)
    
    # print(f"Using {num_manipulable} manipulable features and {num_non_manipulable} non-manipulable features.")
    
    return X_train.values, X_test.values, y_train, y_test, manipulable_mask


def load_and_process_lawschool_old_version(datapath):
    """
    Loads Lawschool data locally from ARFF format.
    One-hot encodes the categorical 'tier' feature.
    Scales real features to a [-1, 1] range.
    Isolates manipulable features from non-manipulable ones.
    """
        
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


# def load_and_process_lawschool(datapath):
#     """
#     Law School: predict pass_bar (binary).
#     Drop non-manipulable features: fulltime, male, racetxt.
#     """
#     df = pd.read_csv(datapath)

#     target = 'pass_bar'
#     # non_manipulable = ['male', 'race']
#     df['race'] = df['race'].apply(lambda x: 1 if 'White' in str(x) else 0)

#     feature_cols = [c for c in df.columns if c not in [target]]
#     X = df[feature_cols].values.astype(float)
#     y = df[target].values.astype(int)

#     return X, y


def load_and_process_lawschool(datapath):
    
    df = pd.read_csv(datapath)
    target = 'pass_bar'
    
    non_manipulable = ['fulltime', 'male', 'racetxt', 'race', 'fam_inc', 'tier']
    feature_cols = [c for c in df.columns if c not in [target] + non_manipulable]
    
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    
    return X, y