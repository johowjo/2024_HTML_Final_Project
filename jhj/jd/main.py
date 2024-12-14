import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('../data/train_data.csv')

# Fill empty fields with 0
data.fillna(0, inplace=True)

# Define features and target variable
# Assuming 'home_team_win' is the target variable
target = 'home_team_win'
features = data.drop(columns=[target, 'id', 'date', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher'])
labels = data[target]

# Convert categorical features to numerical using one-hot encoding
features = pd.get_dummies(features)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators = 200, max_depth = 20)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


# Load the test data
same_season_test_data = pd.read_csv('../data/same_season_test_data.csv')

# List of columns to drop (adjust according to the actual columns in your test data)
columns_to_drop = ['id', 'date', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher']

# Drop only columns that exist in the test data
existing_columns_to_drop = [col for col in columns_to_drop if col in same_season_test_data.columns]
same_season_test_data_cleaned = same_season_test_data.drop(columns=existing_columns_to_drop)

# Fill missing values with 0 (to match the train data processing)
same_season_test_data_cleaned.fillna(0, inplace=True)

# Apply one-hot encoding to categorical features
same_season_test_data_cleaned = pd.get_dummies(same_season_test_data_cleaned)

# Ensure that the test data has the same columns as the training data
missing_cols = set(features.columns) - set(same_season_test_data_cleaned.columns)
for col in missing_cols:
    same_season_test_data_cleaned[col] = 0

# Remove any extra columns in the test data that were not in the training data
extra_cols = set(same_season_test_data_cleaned.columns) - set(features.columns)
same_season_test_data_cleaned.drop(columns=extra_cols, inplace=True)

# Align columns order with the training data
same_season_test_data_cleaned = same_season_test_data_cleaned[features.columns]

# Ensure the scaler was fitted with training data before transforming test data
# You must use the same scaler fitted on the training data, here 'scaler' is assumed to be the fitted scaler
#same_season_test_data_scaled = scaler.transform(same_season_test_data_cleaned)
scaler = StandardScaler()
scaler.fit(features)  # Fit the scaler to your training data features

same_season_test_data_scaled = scaler.transform(same_season_test_data_cleaned)


# Make predictions using the trained model
y_pred_test = model.predict(same_season_test_data_scaled)

# Create a DataFrame with the original IDs and predicted results
predictions_df = same_season_test_data[['id']]  # Retain 'id' from the original test data
predictions_df['predicted_home_team_win'] = y_pred_test

# Save the predictions to a CSV file
predictions_df.to_csv('./submission1.csv', index=False)

print("Predictions and original data saved to 'predictions_with_original_data.csv'.")


target = 'home_team_win'
same_season_test_data_cleaned = same_season_test_data.drop(columns=[target, 'id', 'date', 'home_team_abbr', 'away_team_abbr', 'home_pitcher', 'away_pitcher'])

# Fill missing values with 0
same_season_test_data_cleaned.fillna(0, inplace=True)

# Apply one-hot encoding to categorical features (same as training data)
same_season_test_data_cleaned = pd.get_dummies(same_season_test_data_cleaned)

# Ensure the columns in the test set match the training set
missing_cols = set(X_train.columns) - set(same_season_test_data_cleaned.columns)
for c in missing_cols:
    same_season_test_data_cleaned[c] = 0

same_season_test_data_cleaned = same_season_test_data_cleaned[X_train.columns]

# Step 2: Make predictions on the preprocessed test data
y_pred_test = model.predict(same_season_test_data_cleaned)

# Step 3: Create a DataFrame with the predictions and save it as a CSV
predictions_df = pd.DataFrame({
    'id': same_season_test_data['id'],  # Assuming the 'id' column exists in the test data
    'predicted_home_team_win': y_pred_test
})

# Save the predictions to a CSV file
predictions_df.to_csv('submission2.csv', index=False)

print("Predictions saved to 'predictions.csv'.")
