[README.md](https://github.com/user-attachments/files/26467512/README.md)
# Chronic Kidney Disease Classification Pipeline

This project demonstrates a complete, end-to-end machine learning workflow for predicting Chronic Kidney Disease (CKD) using clinical data. It focuses on cleaning corrupted datasets, handling missing values, and building baseline classification models.

The core of this project is a robust preprocessing and modeling pipeline implemented in a single Python script (`CKDC.py`), designed to handle real-world noisy medical data and produce reliable classification results using F1-score evaluation.

## Technology Stack

-   **Backend & ML**: Python, Scikit-learn, Pandas, NumPy

## Key Features

-   **Robust Data Cleaning**: Handles corrupted and inconsistent values (e.g., `?`, `na`, `n/a`) and enforces domain-specific constraints on clinical features.
-   **Advanced Missing Value Imputation**:
    - Categorical features imputed using Decision Tree–based iterative imputation  
    - Numerical features imputed using a custom Ridge regression approach  
-   **End-to-End ML Workflow**: A single script (`CKDC.py`) that performs data cleaning, imputation, preprocessing, and model training.
-   **Reliable Model Evaluation**: Supports both single split validation and Stratified K-Fold cross-validation using F1-score.

## File Structure
```bash
├── CKDC.py                # Main pipeline script (cleaning, imputation, training, evaluation)
├── mix_corrupted.csv     # Training and validation dataset (corrupted)
├── test_corrupted.csv    # Independent test dataset (corrupted)
└── README.md
```

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.9 (Recommended)
-   An environment manager like `venv` or `conda`

### Setup & Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/ckd-classification
    cd ckd-classification
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Dataset:**
    -   Original dataset: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
    -   The provided files (`mix_corrupted.csv`, `test_corrupted.csv`) are corrupted versions used in this project.
    -   No additional download is required if these files are already included.

## Usage / Workflow

Run the script **from the project root directory** to execute the full pipeline.

1.  **Run the ML Workflow (Data Cleaning to Model Evaluation):**
    This script will clean the data, handle missing values, preprocess features, and train classification models.
    ```sh
    python CKDC.py
    ```

2.  **Workflow Steps:**
    - Load corrupted datasets  
    - Clean invalid and inconsistent values  
    - Impute missing categorical and numerical features  
    - Encode categorical variables and scale numerical features  
    - Train classification model(s)  
    - Evaluate performance using F1-score and cross-validation  
