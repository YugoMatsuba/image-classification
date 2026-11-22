# COMP30027 Machine Learning Project 2

This project was created with the assistance of generative AI for code syntax support, grammar/vocabulary refinement, and documentation organisation. All analysis, reasoning, model design, and interpretation are my own.

## Overview

This project investigates image classification using two datasets:

- **Task 1:** animal image classification
- **Task 2:** bird species classification

The project compares three model approaches:

1. Random Forest using ResNet-50 extracted features
2. SVM using ResNet-50 extracted features
3. CNN trained directly on image data

The final Kaggle-format prediction files are stored in the `predictions/` folder.

## Requirements

Install the required Python libraries before running the notebook:

```bash
pip install numpy pandas pillow matplotlib scikit-learn torch torchvision
```

## How to Run

Open the notebook:

```text
COMP30027_A1_YugoMatsuba.ipynb
```

Then run the notebook cells from top to bottom in this order:

```text
1. Feature Engineering
2. Random Forest
3. SVM
4. CNN
5. Diagrams
6. Error Analysis
```

## Code Block Descriptions

### 1. Feature Engineering

This section creates the feature datasets used by the classical machine learning models.

It does the following:

- Loads the provided feature CSV files.
- Combines HOG, colour histogram, and additional features into the 219-dimensional given feature set.
- Loads pretrained ResNet-50.
- Removes the final classification layer.
- Extracts 2048-dimensional average-pooling features.
- Creates ResNet features for multiple crop levels:
  - full image
  - 10% crop
  - 20% crop
  - 30% crop
  - 40% crop
  - 50% crop


### 2. Random Forest

This section trains and evaluates Random Forest models.

It does the following:

- Runs 5-fold cross-validation on Task 1 and Task 2.
- Evaluates the given 219-dimensional features.
- Evaluates ResNet-50 features at different crop levels.
- Saves validation results and summary tables.
- Trains the final Random Forest models and creates Kaggle-format prediction files.


### 3. SVM

This section trains and evaluates SVM models using an RBF kernel.

It does the following:

- Runs 5-fold cross-validation on Task 1 and Task 2.
- Evaluates the given features and ResNet features.
- Compares different C values:
  - 0.1
  - 1
  - 10
  - 100
  - 1000
- Saves validation results and summary tables.
- Creates Kaggle-format prediction files using the selected C value and crop setting.


### 4. CNN

This section trains a CNN directly on RGB image data.

It does the following:

- Runs 5-fold cross-validation for each crop level.
- Trains a CNN with four convolutional blocks.
- Evaluates mean validation accuracy for each crop level.
- Trains final CNN models on the full training data.
- Creates Kaggle-format prediction files.


### 5. Diagrams

This section creates graphs and tables used in the report.

It reads saved result CSV files from:

```text
results/
```

It creates:

- Crop trend line charts for Random Forest, SVM, and CNN.
- Feature representation comparison table.
- Best model comparison bar charts.


### 6. Error Analysis

This section creates error analysis tables.

It does the following:

- Collects validation predictions from 5-fold cross-validation.
- Finds the class with the lowest recall for each selected model.
- Finds the class it was most often confused with.
- Saves compact error analysis tables.# image-classification
