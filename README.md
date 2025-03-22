
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

### Prerequisites
- Python 3.7 or higher
- Required Python libraries: `numpy`, `pandas`, `torch`, `scikit-learn`, `scipy`, `matplotlib`

### Steps
1. Clone the repository:
git clone https://github.com/yourusername/HPINN-Project.git
cd HPINN-Project

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset**: Provided by [source].
- **References**:
- Basquin's Law and Paris' Law for fatigue life prediction.
- Physics-Informed Neural Networks (PINNs) for integrating physics into deep learning.

---

For questions or feedback, please contact [Your Name] at [your.email@example.com].
