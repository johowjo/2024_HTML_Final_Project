import pandas as pd
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
train_data = pd.read_csv("./data/train_data.csv")
test_data = pd.read_csv("./data/same_season_test_data.csv")

# Prepare features and target for training
train_x = train_data.drop(columns=["id", "home_team_win", "date", "home_team_season", "away_team_season", 
                                   "home_team_abbr", "away_team_abbr", "home_pitcher", "away_pitcher", "is_night_game"])
train_y = train_data["home_team_win"]

# Handle missing values (fill with mean)
train_x.fillna(train_x.mean(), inplace=True)

# Encode categorical target variable (True/False to 1/0)
train_y = train_y.replace({"True": 1, "False": 0})

# One-hot encode categorical columns in the test set
test_x = test_data.drop(columns=["id", "home_team_season", "away_team_season", 
                                 "home_team_abbr", "away_team_abbr", "home_pitcher", 
                                 "away_pitcher", "is_night_game"])
test_x.fillna(test_x.mean(), inplace=True)

# Label encoding for categorical variables in test set
test_x = pd.get_dummies(test_x, drop_first=True)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# Initialize the Support Vector Classifier
svm = SVC(kernel='linear', C=1, random_state=42)  # Linear kernel, regularization parameter C

# Train the model
svm.fit(X_train, y_train)

# Make predictions
train_pred = svm.predict(X_train)
val_pred = svm.predict(X_val)

# Evaluate accuracy
train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Make predictions on the test set
test_pred = svm.predict(test_x)

# Save predictions to a file
with open("svm_predictions.csv", "w") as f:
    f.write("id,home_team_win\n")
    for i, pred in enumerate(test_pred):
        f.write(f"{i},{'True' if pred == 1 else 'False'}\n")

