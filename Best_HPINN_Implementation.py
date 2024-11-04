import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'font.family': 'serif'
})

# Load data
data = pd.read_csv('fatigue_data3.csv')
Nf = data['cycle'].values
defect_size = data['defect_size'].values
delta_sigma = data['stress'].values
Y = 0.5
delta_K0 = Y * delta_sigma * np.sqrt(np.pi * defect_size)
Nf_area = Nf / defect_size

# Fit functions
def basquin_func(Nf, sigma_f_prime, b):
    return sigma_f_prime * (Nf ** b)

def fit_func(delta_K0, a_p, b_p):
    return a_p * delta_K0 ** b_p

# Fit Basquin's and Paris' curves
basquin_params, _ = curve_fit(basquin_func, Nf, delta_sigma)
sigma_f_prime, b_basquin = basquin_params

paris_params, _ = curve_fit(fit_func, delta_K0, Nf_area)
a_paris, b_paris = paris_params
m = -b_paris
C = 1 / (a_paris * (1 - b_paris / 2))

# Prepare data for neural networks
X = data[['stress', 'defect_size']].values
y = data[['cycle']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def normalize(X_train, X_test, y_train, y_test):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    
    X_train_normalized = (X_train - X_mean) / X_std
    X_test_normalized = (X_test - X_mean) / X_std
    y_train_normalized = (y_train - y_mean) / y_std
    y_test_normalized = (y_test - y_mean) / y_std
    
    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, X_mean, X_std, y_mean, y_std

# PINN model
class PINN(nn.Module):
    def __init__(self, use_basquin=True, use_paris=True, use_non_neg=True):
        super(PINN, self).__init__()
        self.use_basquin = use_basquin
        self.use_paris = use_paris
        self.use_non_neg = use_non_neg

        self.fc1 = nn.Linear(2, 63)
        self.fc2 = nn.Linear(63, 63)
        self.fc3 = nn.Linear(63, 63)
        
        if use_basquin:
            self.phys_fc1_basquin = nn.Linear(63, 21)
            self.phys_fc2_basquin = nn.Linear(21, 21)
        
        if use_paris:
            self.phys_fc1_paris = nn.Linear(63, 21)
            self.phys_fc2_paris = nn.Linear(21, 21)
        
        if use_non_neg:
            self.phys_fc1_non_neg = nn.Linear(63, 21)
            self.phys_fc2_non_neg = nn.Linear(21, 21)
        
        output_dim = sum([21 for use in [use_basquin, use_paris, use_non_neg] if use])
        self.fc_final = nn.Linear(max(output_dim, 1), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(63)
        self.batch_norm2 = nn.BatchNorm1d(63)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        phys_outputs = []
        
        if self.use_basquin:
            x_phys_basquin = self.relu(self.phys_fc1_basquin(x))
            x_phys_basquin2 = self.relu(self.phys_fc2_basquin(x_phys_basquin))
            phys_outputs.append(x_phys_basquin2)
        
        if self.use_paris:
            x_phys_paris = self.relu(self.phys_fc1_paris(x))
            x_phys_paris2 = self.relu(self.phys_fc2_paris(x_phys_paris))
            phys_outputs.append(x_phys_paris2)
        
        if self.use_non_neg:
            x_phys_non_neg = self.relu(self.phys_fc1_non_neg(x))
            x_phys_non_neg2 = self.relu(self.phys_fc2_non_neg(x_phys_non_neg))
            phys_outputs.append(x_phys_non_neg2)
        
        x_phys_combined = torch.cat(phys_outputs, dim=1) if phys_outputs else torch.zeros(x.size(0), 1, device=x.device)
        x_out = self.fc_final(x_phys_combined)
        
        return x_out

# Early stopping
def train_model_with_early_stopping(model, loss_fn, X_train, y_train, X_val, y_val, batch_size=5, max_epochs=3000, patience=1000, print_interval=100):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in data_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = loss_fn(val_output, y_val)
        
        if epoch % print_interval == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

# Main execution
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, X_mean, X_std, y_mean, y_std = normalize(X_train, X_test, y_train, y_test)

X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_normalized, dtype=torch.float32)

pinn_model = PINN(use_basquin=True, use_paris=True, use_non_neg=True)
train_model_with_early_stopping(pinn_model, nn.MSELoss(), X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=10, max_epochs=3000, patience=100, print_interval=100)

pinn_model.load_state_dict(torch.load('best_model.pth'))
pinn_model.eval()
with torch.no_grad():
    y_pred_train_pinn = pinn_model(X_train_tensor)
    y_pred_test_pinn = pinn_model(X_test_tensor)

# Denormalize predictions
y_pred_train_pinn = y_pred_train_pinn * y_std + y_mean
y_pred_test_pinn = y_pred_test_pinn * y_std + y_mean
y_train_tensor = y_train_tensor * y_std + y_mean
y_test_tensor = y_test_tensor * y_std + y_mean

# Calculate evaluation metrics for PINN
train_r2_pinn = r2_score(y_train_tensor.numpy(), y_pred_train_pinn.numpy())
test_r2_pinn = r2_score(y_test_tensor.numpy(), y_pred_test_pinn.numpy())
train_mape_pinn = mean_absolute_percentage_error(y_train_tensor.numpy(), y_pred_train_pinn.numpy())
test_mape_pinn = mean_absolute_percentage_error(y_test_tensor.numpy(), y_pred_test_pinn.numpy())

# Plot R² and MAPE scores with distinct colors and annotations
plt.figure(figsize=(12, 6))

# R² Scores
plt.subplot(1, 2, 1)
r2_scores = [train_r2_pinn, test_r2_pinn]
r2_labels = ['Train PINN', 'Test PINN']
bars_r2 = plt.bar(r2_labels, r2_scores, color=['skyblue', 'salmon'])
plt.title('R² Scores')
plt.ylabel('R² Score')
plt.grid(True, linestyle='--')

# Annotate R² bars
for bar in bars_r2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')

# MAPE Scores
plt.subplot(1, 2, 2)
mape_scores = [train_mape_pinn, test_mape_pinn]
mape_labels = ['Train PINN', 'Test PINN']
bars_mape = plt.bar(mape_labels, mape_scores, color=['skyblue', 'salmon'])
plt.title('MAPE Scores')
plt.ylabel('MAPE')
plt.grid(True, linestyle='--')

# Annotate MAPE bars
for bar in bars_mape:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')

plt.tight_layout()
plt.show()

# Plot Experimental vs. Predicted Cycles for PINN with annotations
plt.figure(figsize=(8, 8))
plt.scatter(y_train_tensor.numpy(), y_pred_train_pinn.numpy(), color='blue', label='Train Data')
plt.scatter(y_test_tensor.numpy(), y_pred_test_pinn.numpy(), color='red', label='Test Data')
plt.plot([y_train_tensor.min(), y_train_tensor.max()], [y_train_tensor.min(), y_train_tensor.max()], 'k--', label='Perfect Prediction')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Experimental Cycles')
plt.ylabel('Predicted Cycles')
plt.title('HPINN - Experimental vs Predicted Fatigue Life Cycles')
plt.legend()
plt.grid(True, linestyle='--')

plt.show()
