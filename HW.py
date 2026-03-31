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

"""
HW.py

This file contains the skeleton for the functions you need to implement as part of your homework.
Each function corresponds to a specific task and includes instructions on what is expected.
"""

# Task 2: Data Cleaning
def clean_data(df):
    """
    Task: Data Cleaning
    --------------------
    This function should take a pandas DataFrame as input and return a cleaned DataFrame.
    
    Instructions:
    - Handle missing values in categorical and numerical columns separately.
    - Handle incorrect data points (e.g., negative or null weight values) (I know that there is no weight column!).
    - Ensure that in the cleaned dataframe all the missing or incorrect values are encoded as NaN, and that the column names are the same as in the input dataframe.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to clean.

    Returns:
    cleaned (pd.DataFrame): The cleaned DataFrame.
    is_missing (pd.DataFrame): A boolean DataFrame where each cell is True if you consider it missing.
    """
    
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



# Task 3: Categorical Features Imputation

def impute_missing_categorical(df_train, df_val, df_test, categorical_columns):
    """
    Task: Categorical Features Imputation (IterativeImputer + DecisionTreeClassifier)
    ------------------------------------------------------------------------------- 
    This function should handle missing values in categorical columns using an
    iterative, model-based imputation strategy.

    In this task, you will:
    - Work only with the categorical columns (which you have passed with `categorical_columns`).
    - Encode the categories as integers using `OrdinalEncoder`.
    - Use `IterativeImputer` with a `DecisionTreeClassifier` as the estimator to impute missing values in the encoded categorical data.
    - Decode the imputed integer codes back to the original categorical labels.

    Instructions:
    1. Create subsets of the input DataFrames (train, validation, test) that contain only the columns in `categorical_columns`.
    Call these subsets X_train_cat, X_val_cat, and X_test_cat, respectively.

    2. Initialize an `OrdinalEncoder`, setting handle_unknown='use_encoded_value', unknown_value=-1, and:
       - Fit it on the correct subset.
       - Transform the training, validation, and test categorical subsets using the fitted encoder.
       - After this step, all categorical values should be represented as numeric codes (floats/ints), but there will still be missing values (NaNs).

    3. Instantiate a `DecisionTreeClassifier` (from `sklearn.tree`).
       - Set a `random_state` for reproducibility (RSEED=8).
       - You may use default hyperparameters for the rest.

    4. Create an `IterativeImputer` (from `sklearn.impute`) with:
       - `estimator` set to the `DecisionTreeClassifier` you created.
       - A fixed `random_state` for reproducibility (RSEED=8).
       - Set `max_iter`=10.
       - Note: `IterativeImputer` works only with numeric data, which is why we first encode the categories.

    5. Fit the `IterativeImputer` on the correct subset.
       - Then use the fitted imputer to transform the encoded training, validation, and test categorical data.
       - After this step, there should be no missing values in the encoded categorical arrays.

    6. Use the fitted `OrdinalEncoder` to inverse-transform the imputed, encoded data for each split (train, validation, test), so that the values
        are converted back to the original categorical labels (strings).

    7. Wrap the final imputed data for each split into `pd.DataFrame`s:
       - The returned DataFrames should:
         * Contain only the categorical columns.
         * Have `categorical_columns` as their column names.
         * Contain no missing values in those columns.
         """


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

# Task 4: Numerical Features Imputation
def impute_numerical_features(df_train, df_val, df_test, numerical_columns):
    """
    Task: Numerical Features Imputation
    ------------------------------------
    This function should handle missing values in numerical columns using appropriate techniques.
    Again, we will skip scaling to keep things simple.
    
    Instructions:
    - The function should take three datasets as input: df_train, df_val, and df_test.
    - The function should return three datasets as output: train_imputed_ridge, val_imputed_ridge, and test_imputed_ridge.
    - Use Ridge regression in an iterative fashion to impute missing values in numerical columns.
    - Ensure that the imputation process is consistent and does not use any other imputer.
    - Follow these steps:
        1. Create a subset of the train dataset with only the numerical columns. Call this subset train_num.
        2. Create a subset of the val dataset with only the numerical columns. Call this subset val_num.
        3. Create a subset of the test dataset with only the numerical columns. Call this subset test_num.
        4a. Create a subset of train_num containing the rows with missing values. Call this subset train_num_missing.
        4b. Create a subset of train_num containing the rows without missing values. Call this subset train_num_not_missing.
        5a. Create a subset of val_num containing the rows with missing values. Call this subset val_num_missing.
        5b. Create a subset of val_num containing the rows without missing values. Call this subset val_num_not_missing.
        6a. Create a subset of test_num containing the rows with missing values. Call this subset test_num_missing.
        6b. Create a subset of test_num containing the rows without missing values. Call this subset test_num_not_missing.
        7a. Select the columns without any missing values in all three subsets.
        7b. Train a Ridge regression model on the correct subset of rows and columns.
        7c. Using Ridge regression, "predict" the missing values in the subsets that have missing values. Only predict the values in the column with the fewest missing values.
            Use the sum of the missing values in all three subsets to decide which column to impute.
        7d. Update the missing, imputed, and not missing subsets accordingly.
        8. Repeat steps 4–7 until all the missing values are imputed.
        9. Save the results in train_num_imputed_ridge, val_num_imputed_ridge, and test_num_imputed_ridge.
        10. Concatenate the imputed subsets with the subsets that did not contain missing values.
        11. Save the resulting datasets in train_imputed_ridge, val_imputed_ridge, and test_imputed_ridge.
        12. Ensure that the order of the rows in the final datasets matches the order in the original datasets.

        Hint: If you print the columns "in common" (step 7a) and the column to impute (step 7c), you should see that at each iteration the column that you imputed in the previous iteration
        becomes one of the columns "in common".
        If you print the number of the iterations, this should not exceed the number of numerical columns, so if you see more iterations you might have entered an infinite loop.

    Parameters:
    df_train (pd.DataFrame): The training DataFrame.
    df_val (pd.DataFrame): The validation DataFrame.
    df_test (pd.DataFrame): The test DataFrame.
    numerical_columns (list): A list of column names corresponding to numerical features.

    Returns:
    train_imputed_ridge (pd.DataFrame): The training DataFrame with imputed numerical features.
    val_imputed_ridge (pd.DataFrame): The validation DataFrame with imputed numerical features.
    test_imputed_ridge (pd.DataFrame): The test DataFrame with imputed numerical features.
    """

    
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
    """
    Task: Merge Imputed DataFrames
    -------------------------------
    This function should merge the imputed categorical and numerical DataFrames.

    Instructions:
    
    Merge the imputed categorical and numerical DataFrames on their indexes.
    Ensure that the resulting DataFrame contains all columns from both input DataFrames.

    Parameters:
    df_cat (pd.DataFrame): The DataFrame with imputed categorical features.
    df_num (pd.DataFrame): The DataFrame with imputed numerical features.

    Returns:
    pd.DataFrame: The merged DataFrame containing both categorical and numerical features.
    """
    merged_df = pd.concat([df_cat, df_num], axis=1)

    merged_df = merged_df.reset_index(drop=True)

    return merged_df


# Task 5: Classification Using a Single Split
def train_and_evaluate_single_split(X_train, X_val, y_train, y_val, cat_cols, num_cols, model, hp):
    """
    Task: Classification Using a Single Split
    ------------------------------------------
    This function should train a classification pipeline on the training set and evaluate it on the validation set, using the provided parameters.

    Instructions:
    - Create a classification pipeline. It should include:
        - An OrdinalEncoder for categorical features (handle_unknown='use_encoded_value', unknown_value=-1).
        - A StandardScaler for numerical features.
        - The provided classification model.
    - Use ColumnTransformer to apply the appropriate transformations to categorical and numerical features.
    - Set the model parameters using the provided hyperparameters dictionary.
    - Train the model using the training data (X_train, y_train).
    - Evaluate the model on the validation data (X_val, y_val) using F1 score.
    - Return the evaluation results as a dictionary containing:
        - 'params': the hyperparameters used,
        - 'F1 score': the computed validation F1 score.
    
    Parameters:
    X_train (pd.DataFrame): The training feature set.
    X_val (pd.DataFrame): The validation feature set.
    y_train (pd.Series): The training labels.
    y_val (pd.Series): The validation labels.
    model: The classification model to train.
    hp (dict): A dictionary of hyperparameters to set for the model.

    Returns:
    dict: A dictionary with two keys: 'params' (hyperparameters) and 'F1 score' (validation F1 score).
    """
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



# Task 6: Classification Using Cross-Validation
def train_and_evaluate_cross_validation(X, y, model, cat_cols, num_cols, hp, cv):
    """
    Task: Classification Using Cross-Validation
    --------------------------------------------
    This function should train and evaluate a classification model using cross-validation.
    
    Instructions:
    - Use cross-validation to train and evaluate the model, with shuffle set to True and using the specified number of folds (cv).
    - For each fold, create a classification pipeline similar to the one in Task 5.
    - Evaluate the model on each fold using F1 score.
    - Return the average F1 score across all folds for each parameter combination.
    - Ensure that the cross-validation process is reproducible (use random_state = 8).
    
    Parameters:
    X (pd.DataFrame): The feature set.
    y (pd.Series): The labels.
    model: The classification model to train.
    hp (dict): A dictionary of hyperparameters to set for the model.
    cv (int): The number of cross-validation folds.

    Returns:
    dict: A dictionary containing two keys: 'params' (training parameters) and 'Average F1 scores' (F1 score). Each key should have the correct value.
    """
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