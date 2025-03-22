
Copy
# HPINN Project: Physics-Informed Neural Network for Fatigue Life Prediction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project implements a **Physics-Informed Neural Network (PINN)** to predict fatigue life cycles based on stress amplitude and defect size. The model integrates traditional physics-based approaches (Basquin's Law and Paris' Law) with deep learning to improve prediction accuracy.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Code Structure](#code-structure)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to predict the number of fatigue life cycles (`Nf`) for materials under cyclic loading, given the **stress amplitude** and **defect size**. The model integrates physics-based equations (Basquin's Law and Paris' Law) into a neural network to improve generalization and interpretability.

### Key Components:
- **Data Preprocessing**: Normalization and splitting of data into training and testing sets.
- **Physics-Informed Neural Network (PINN)**: Combines traditional physics models with deep learning.
- **Training**: Implements early stopping to prevent overfitting.
- **Evaluation**: Metrics include R² score and Mean Absolute Percentage Error (MAPE).
- **Visualization**: Plots experimental vs. predicted cycles and evaluation metrics.

---

## Features

- **Physics-Informed Learning**: Incorporates Basquin's Law and Paris' Law into the neural network.
- **Early Stopping**: Prevents overfitting during training.
- **Evaluation Metrics**: Calculates R² and MAPE for both training and testing data.
- **Visualization**: Generates plots for R², MAPE, and experimental vs. predicted cycles.
- **Excel Output**: Saves predictions to an Excel file for further analysis.

---

## Installation



Copy

2. Install the required libraries:
pip install numpy pandas torch scikit-learn scipy matplotlib

Copy

3. Place your dataset (`Data1.csv`) in the project directory.

---

## Usage

1. **Run the Script**:
Execute the Python script to train the model and generate results:
python hpin_project.py

Copy

2. **Output**:
- **Excel File**: Predictions are saved to `HPINN_Predictions1.xlsx`.
- **Plots**: R², MAPE, and experimental vs. predicted cycles are displayed.

---

## Code Structure

The project is organized into the following sections:

1. **Data Loading**:
- Loads the dataset (`Data1.csv`).
- Extracts stress amplitude, defect size, and fatigue life cycles.

2. **Data Preprocessing**:
- Normalizes the data for training.
- Splits the data into training and testing sets.

3. **Model Definition**:
- Defines the **Physics-Informed Neural Network (PINN)**.
- Combines traditional physics models (Basquin's Law and Paris' Law) with deep learning.

4. **Training**:
- Implements early stopping to prevent overfitting.
- Saves the best model checkpoint (`best_model.pth`).

5. **Evaluation**:
- Calculates R² and MAPE for training and testing data.
- Denormalizes predictions for comparison with experimental data.

6. **Visualization**:
- Plots R² and MAPE scores.
- Generates a scatter plot of experimental vs. predicted cycles.

7. **Saving Results**:
- Saves predictions to an Excel file (`HPINN_Predictions1.xlsx`).

---

## Model Architecture

The **Physics-Informed Neural Network (PINN)** consists of:
- **Input Layer**: 2 features (stress amplitude and defect size).
- **Hidden Layers**:
- Fully connected layers with ReLU activation.
- Dropout and batch normalization for regularization.
- **Physics Branches**:
- **Basquin's Law Branch**: Models fatigue life using Basquin's equation.
- **Paris' Law Branch**: Models crack growth using Paris' equation.
- **Non-Negative Branch**: Ensures non-negative predictions.
- **Output Layer**: Predicts the number of fatigue life cycles.

---

## Results

### Evaluation Metrics
- **R² Score**: Measures the goodness of fit.
- **MAPE**: Measures the percentage error in predictions.

### Plots
1. **R² and MAPE Scores**:
- Bar plots comparing training and testing performance.
2. **Experimental vs. Predicted Cycles**:
- Scatter plot with 2-factor and 3-factor bands for error analysis.

### Example Output
- **Excel File**: Contains stress, defect size, experimental cycles, and predicted cycles for both training and testing data.
- **Plots**: Visualize model performance and predictions.

---



## Acknowledgments
[Wang L, Zhu S P, Luo C, et al. Defect-driven physics-informed neural network framework for fatigue life prediction of additively manufactured materials[J/OL]. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 2023, 381(2260).
https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0386


Zhang X, Gong S, Wang Y, et al. A modified swt model for very high cycle fatigue life prediction of lpbf ti-6al-4v alloy based on single defect: Effect of building orientation[J/OL]. International Journal of Fatigue, 2024, 188: 108514. https:
//www.sciencedirect.com/science/article/pii/S0142112324003724.


Wang L, Zhu S P, Luo C, et al. Physics-guided machine learning frameworks for fatigue life prediction of am materials[J/OL]. International Journal of Fatigue, 2023, 172: 107658. https://www.sciencedirect.com/science/article/pii/S0142112323001597
- Basquin's Law and Paris' Law for fatigue life prediction.
- Physics-Informed Neural Networks (PINNs) for integrating physics into deep learning.

---

For questions or feedback, please contact Isaac Abiria at abiriaisaac@bit.ac.cn.
