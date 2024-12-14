import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from liblinear.liblinearutil import *

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column, 'date', 'home_pitcher', 'away_pitcher'], errors='ignore')
    y = df[target_column] if target_column in df.columns else None

    X = pd.get_dummies(X, columns=['home_team_abbr', 'away_team_abbr', 'season', 'home_team_season', 'away_team_season'], drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y

train_data = pd.read_csv('./data/train_data.csv')
test_data = pd.read_csv('./data/same_season_test_data.csv')
sample_submission = pd.read_csv('./data/same_season_sample_submission.csv')

X_train, y_train = preprocess_data(train_data, 'home_team_win')
X_test, _ = preprocess_data(test_data, None)

X_train = X_train.to_numpy().astype(np.float64)
y_train = y_train.to_numpy().astype(np.float64).tolist()
X_test = X_test.to_numpy().astype(np.float64)

lam = 0.01
C = 1 / lam
param = f'-s 0 -c {C} -q'
model = train(y_train, X_train, param)

y_test = [0] * X_test.shape[0]
predictions, p_test_acc, _ = predict(y_test, X_test, model, options='-q')

predictions = [True if pred == 1 else False for pred in predictions]

submission = sample_submission.copy()
submission['home_team_win'] = predictions
submission.to_csv('./result/submission_l2reg.csv', index=False)
