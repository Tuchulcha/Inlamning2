import pandas as pd
import xgboost as xgb
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter # For logging values within training loop
import torch.onnx
from sklearn.model_selection import KFold

# Load dataset
df = pd.read_csv('AmazonDataSales_v2.csv', low_memory=False)
# Drop all columns except 'amount', 'category', 'size', 'quantity'
df = df[['amount', 'category', 'size', 'qty']]

# One-hot encode the 'category', 'size', and 'qty' columns
# Select all columns except 'amount' as feature columns
feature_columns = df.columns.drop('amount')
# One-hot encode the feature columns
df_encoded = pd.get_dummies(df, columns=feature_columns)

# Assuming 'df' contains your dataset
X = df_encoded.drop('amount', axis=1)  # Features
y = df['amount']  # Target

#Network stopped converging, the only things I did was add hooks and remove this line
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import KFold

# Define the number of folds for cross-validation
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Convert to PyTorch tensors just once
X_tensor = torch.tensor(X_np)
y_tensor = torch.tensor(y_np)

# Define the training and evaluation process within a cross-validation loop
fold_perf = []

for fold, (train_ids, test_ids) in enumerate(kfold.split(X_tensor)):
    # Split data
    X_train, X_test = X_tensor[train_ids], X_tensor[test_ids]
    y_train, y_test = y_tensor[train_ids], y_tensor[test_ids]
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    
    # Model instantiation (inside the loop to reset weights)
    model = FeedForwardRegressor(input_size, hidden_size1, hidden_size2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training and evaluation
    print(f"Fold {fold + 1}/{num_folds}")
    train_model(model, train_loader, criterion, optimizer, writer, num_epochs=5)  # Ensure this function is adapted for CV
    fold_perf.append(evaluate_model(model, test_loader))  # Adapt this to return and store fold performance metrics

# After cross-validation, you can aggregate `fold_perf` to get an overall performance metric


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Logging during training
writer = SummaryWriter()

# Function to register hooks for monitoring activations
def register_activation_hooks(model, writer):
    def hook_fn(module, input, output):
        writer.add_histogram(f"{module.__class__.__name__}_activations", output)

    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.Linear):
            # Use a closure to capture the current layer
            layer.register_forward_hook(lambda module, input, output, layer=layer: hook_fn(layer, input, output))


# Function to register hooks for monitoring gradients
def register_gradient_hooks(model, writer):
    for name, parameter in model.named_parameters():
        def hook(grad, name=name):  # Capture current value of name
            writer.add_histogram(f"{name}_gradients", grad)
        parameter.register_hook(hook)


# Function to log weights, needs no fancy hooks
def log_weights(model, writer, epoch):
    for name, param in model.named_parameters():
        writer.add_histogram(f"{name}_weights", param, epoch)


class FeedForwardRegressor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(FeedForwardRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)  # Output layer for regression

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Model instantiation and move to device
input_size = X_train.shape[1]
hidden_size1 = 2
hidden_size2 = 2
model = FeedForwardRegressor(input_size, hidden_size1, hidden_size2).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



#Training function with logging

def train_model(model, train_loader, criterion, optimizer, writer, num_epochs=10):
    model.train()
    register_activation_hooks(model, writer)  # Register activation hooks // register_forward_hook
    register_gradient_hooks(model, writer)     # Register gradient hooks


    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log weights and gradients for the first batch in each epoch
            if batch_idx == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"{name}_weights", param, epoch) #parameter_hook
                    writer.add_histogram(f"{name}_grads", param.grad, epoch) #?_hook

            # Log weights at the end of each epoch
            log_weights(model, writer, epoch)

        # Log loss at each epoch
        writer.add_scalar('Loss/train', total_loss/len(train_loader), epoch)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}')

# Evaluation function

def evaluate_model(model, test_loader):
    model.eval()
    targets_list = []
    outputs_list = []
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in test_loader:
            # Move data to the device
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))  # Add an extra dimension to targets

            total_loss += loss.item()
            targets_list.append(targets.cpu())
            outputs_list.append(outputs.cpu())
        
        # Concatenate all batches
        all_targets = torch.cat(targets_list, dim=0)
        all_outputs = torch.cat(outputs_list, dim=0)

        # Calculate R-squared score
        r2 = r2_score(all_targets.numpy(), all_outputs.numpy())
        
        print(f'Test Loss: {total_loss/len(test_loader)}')
        print(f'R-squared: {r2}')

# Run training
train_model(model, train_loader, criterion, optimizer, writer, num_epochs=5)

# and evaluation
evaluate_model(model, test_loader) 



