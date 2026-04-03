from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

# Data Cleaning
def clean_data(df):

    cleaned = df.copy()
    
    placeholder_values = ['?', 'na', 'n/a', 'nan', '']
    cleaned = cleaned.applymap(
        lambda x: np.nan if (pd.isna(x) or str(x).strip().lower() in placeholder_values)
        else x
    )

    numeric_cols = [
        "age", "bp", "sg", "al", "su",
        "bgr", "bu", "sc", "sod", "pot",
        "hemo", "pcv", "wbcc", "rbcc"
    ]

    categorical_cols = [
        "rbc", "pc", "pcc", "ba", "htn", "dm",
        "cad", "appet", "pe", "ane", "class"
    ]

    allowed_cats = {
        "rbc":   ["normal", "abnormal"],
        "pc":    ["normal", "abnormal"],
        "pcc":   ["present", "notpresent"],
        "ba":    ["present", "notpresent"],
        "htn":   ["yes", "no"],
        "dm":    ["yes", "no"],
        "cad":   ["yes", "no"],
        "appet": ["good", "poor"],
        "pe":    ["yes", "no"],
        "ane":   ["yes", "no"],
        "class": ["ckd", "notckd"]
    }


    for col in numeric_cols:
        if col not in cleaned.columns:
            continue

        cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

        cleaned.loc[cleaned[col] < 0, col] = np.nan

        if col == "age":
            cleaned.loc[cleaned[col] > 120, col] = np.nan

        if col == "bp":
            cleaned.loc[cleaned[col] > 200, col] = np.nan

        if col in ["al", "su"]:
            cleaned[col] = cleaned[col].apply(
                lambda v: v if v in [0,1,2,3,4,5] else np.nan
            )


    for col in categorical_cols:
        if col not in cleaned.columns:
            continue

        cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()

        allowed = allowed_cats[col]

        cleaned.loc[~cleaned[col].isin(allowed), col] = np.nan


    is_missing = cleaned.isna()

    return cleaned, is_missing



# Categorical Features Imputation

def impute_missing_categorical(df_train, df_val, df_test, categorical_columns):

    RSEED = 8

    X_train_cat = df_train[categorical_columns].copy()
    X_val_cat = df_val[categorical_columns].copy()
    X_test_cat = df_test[categorical_columns].copy()

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(X_train_cat)  # Fit on training data

    X_train_enc = encoder.transform(X_train_cat)
    X_val_enc = encoder.transform(X_val_cat)
    X_test_enc = encoder.transform(X_test_cat)

    dt_estimator = DecisionTreeClassifier(random_state=RSEED)

    imputer = IterativeImputer(
        estimator=dt_estimator,
        max_iter=10,
        random_state=RSEED,
    )
    
    imputer.fit(X_train_enc)
    
    X_train_imputed_enc = imputer.transform(X_train_enc)
    X_val_imputed_enc = imputer.transform(X_val_enc)
    X_test_imputed_enc = imputer.transform(X_test_enc)

    X_train_imputed_cat = pd.DataFrame(encoder.inverse_transform(X_train_imputed_enc), columns=categorical_columns)
    X_val_imputed_cat = pd.DataFrame(encoder.inverse_transform(X_val_imputed_enc), columns=categorical_columns)
    X_test_imputed_cat = pd.DataFrame(encoder.inverse_transform(X_test_imputed_enc), columns=categorical_columns)

    return X_train_imputed_cat, X_val_imputed_cat, X_test_imputed_cat

# Numerical Features Imputation
def impute_numerical_features(df_train, df_val, df_test, numerical_columns):
  
    train_num = df_train[numerical_columns].copy()
    val_num = df_val[numerical_columns].copy()
    test_num = df_test[numerical_columns].copy()

    train_imputed_ridge = train_num.copy()
    val_imputed_ridge = val_num.copy()
    test_imputed_ridge = test_num.copy()

    train_num_missing = train_imputed_ridge[train_imputed_ridge.isnull().any(axis=1)].copy()
    train_num_not_missing = train_imputed_ridge[~train_imputed_ridge.isnull().any(axis=1)].copy()

    val_num_missing = val_imputed_ridge[val_imputed_ridge.isnull().any(axis=1)].copy()
    test_num_missing = test_imputed_ridge[test_imputed_ridge.isnull().any(axis=1)].copy()

    while pd.concat([train_num_missing, val_num_missing, test_num_missing]).isna().sum().sum() != 0:

        missing_count = pd.concat([train_num_missing, val_num_missing, test_num_missing]).isna().sum()
        fewest_missing = missing_count[missing_count > 0].idxmin()


        cols_no_missing = [
            col for col in numerical_columns
            if col != fewest_missing and train_imputed_ridge[col].isna().sum() == 0
        ]


        x_train = train_num_not_missing[cols_no_missing]
        y_train = train_num_not_missing[fewest_missing]
        model = Ridge()
        model.fit(x_train, y_train)

        x_test = train_num_missing[train_num_missing[fewest_missing].isna()][cols_no_missing]
        train_imputed_ridge.loc[x_test.index, fewest_missing] = model.predict(x_test)

        x_test = val_num_missing[val_num_missing[fewest_missing].isna()][cols_no_missing]
        val_imputed_ridge.loc[x_test.index, fewest_missing] = model.predict(x_test)

        x_test = test_num_missing[test_num_missing[fewest_missing].isna()][cols_no_missing]
        test_imputed_ridge.loc[x_test.index, fewest_missing] = model.predict(x_test)

        train_num_missing = train_imputed_ridge[train_imputed_ridge.isnull().any(axis=1)].copy()
        train_num_not_missing = train_imputed_ridge[~train_imputed_ridge.isnull().any(axis=1)].copy()
        val_num_missing = val_imputed_ridge[val_imputed_ridge.isnull().any(axis=1)].copy()
        test_num_missing = test_imputed_ridge[test_imputed_ridge.isnull().any(axis=1)].copy()

    return train_imputed_ridge, val_imputed_ridge, test_imputed_ridge
    
def merge_imputed(df_cat, df_num):
    
    merged_df = pd.concat([df_cat, df_num], axis=1)

    merged_df = merged_df.reset_index(drop=True)

    return merged_df


# Classification Using a Single Split
def train_and_evaluate_single_split(X_train, X_val, y_train, y_val, cat_cols, num_cols, model, hp):

    model.set_params(**hp)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    clf_pipeline = Pipeline([('preprocessor', preprocessor),('classifier', model)])

    clf_pipeline.fit(X_train, y_train)

    y_pred = clf_pipeline.predict(X_val)

    f1 = f1_score(y_val, y_pred)

    return {'params': hp, 'F1 scores': f1}



# Classification Using Cross-Validation
def train_and_evaluate_cross_validation(X, y, model, cat_cols, num_cols, hp, cv):
  
    RSEED = 8

    model.set_params(**hp)

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    clf_pipeline = Pipeline([('preprocessor', preprocessor),('classifier', model)])

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RSEED)

    f1_scores = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf_pipeline.fit(X_train, y_train)
        y_pred = clf_pipeline.predict(X_test)

        score = f1_score(y_test, y_pred)
        f1_scores.append(score)

    avg_f1 = float(np.mean(f1_scores))
    return {'params': hp, 'Average F1 scores': avg_f1}
