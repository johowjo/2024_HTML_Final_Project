import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv("./data/train_data.csv")
test_data = pd.read_csv("./data/same_season_test_data.csv")

# Prepare the features and labels
train_x = train_data.drop(columns=["id", "home_team_win", "date", "home_team_season", "away_team_season", "home_team_abbr", "away_team_abbr", "home_pitcher", "away_pitcher", "is_night_game"])
train_y = train_data["home_team_win"]

train_x.fillna(train_x.mean(), inplace=True)

test_x = test_data.drop(columns=["id", "home_team_season", "away_team_season", "home_team_abbr", "away_team_abbr", "home_pitcher", "away_pitcher", "is_night_game"])
test_x.fillna(test_x.mean(), inplace=True)

# Label encode the target variable
train_y = train_y.replace({"True": 1, "False": 0})
test_y = pd.Series(np.zeros(len(test_x)))

# Convert to numpy arrays for PyTorch
train_x = train_x.to_numpy().astype(np.float32)
test_x = test_x.to_numpy().astype(np.float32)

train_y = train_y.to_numpy().astype(np.float32)

# Convert to PyTorch tensors
train_x_tensor = torch.tensor(train_x)
train_y_tensor = torch.tensor(train_y)
test_x_tensor = torch.tensor(test_x)

# Define a simple feedforward neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model, loss function, and optimizer
input_size = train_x.shape[1]
model = SimpleNN(input_size)

criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(train_x_tensor).squeeze()  # Squeeze to remove extra dimension
    loss = criterion(outputs, train_y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Make predictions on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    test_predictions = model(test_x_tensor).squeeze()
    test_predictions = (test_predictions >= 0.5).float()  # Convert probabilities to binary (0 or 1)

# accuracy = accuracy_score(train_y_tensor.numpy(), test_predictions.numpy())
# print(f"Accuracy: {accuracy * 100:.2f}%")

# Save predictions to a CSV file
count = 0
with open("pytorch_predictions.csv", "w") as file:
    file.write("id,home_team_win\n")
    for p in test_predictions:
        file.write(f"{count},")
        count += 1
        if p == 0:
            file.write("True\n")
        else:
            file.write("False\n")

print("Predictions saved to pytorch_predictions.csv")

