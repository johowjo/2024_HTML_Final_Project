import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column, 'date', 'home_pitcher', 'away_pitcher', 'season', 'home_team_season', 'away_team_season'], errors='ignore')
    y = df[target_column] if target_column in df.columns else None

    X = pd.get_dummies(X, columns=['home_team_abbr', 'away_team_abbr'], drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y

train_data = pd.read_csv('./data/train_data.csv')
test_data = pd.read_csv('./data/2024_test_data.csv')
sample_submission = pd.read_csv('./data/2024_sample_submission.csv')

X_train, y_train = preprocess_data(train_data, 'home_team_win')
X_test, _ = preprocess_data(test_data, None)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

submission = sample_submission.copy()
submission['home_team_win'] = predictions
submission.to_csv('./result/submission_pla.csv', index=False)
