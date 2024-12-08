import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def preprocess_2(df,numeric,season_year=-1):
    if season_year != -1:
        df = df[df['season'] == int(season_year)]
    if 'home_team_win' in df.columns:
        df['home_team_win'] = df['home_team_win'].map({True: 1, False: -1})
    df['is_night_game'] = df['is_night_game'].map({True: 1, False: -1})
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'season' in df.columns:
        df = df.drop(columns=['season'])
    if 'home_team_win' in df.columns:
        X = df.drop(columns=['home_team_win'])
        y = df['home_team_win']
        y = y.to_numpy()
    else:
        X = df
        y = []
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_features = X.select_dtypes(include=['number']).columns
    X[num_features] = num_imputer.fit_transform(X[num_features])
    X_numeric = X.select_dtypes(include=['number'])
    
    cat_features = X.select_dtypes(exclude=['number']).columns
    X[cat_features] = cat_imputer.fit_transform(X[cat_features])
    X[cat_features] = X[cat_features].astype('category')
    X = pd.get_dummies(X, drop_first=True)

    if numeric: 
        X = X_numeric.to_numpy()
    else:
        X = X.to_numpy()
    return X, y

def preprocess(file_path,numeric,season_year=-1):
    df = pd.read_csv(file_path)
    if season_year != -1:
        df = df[df['season'] == int(season_year)]
    if 'home_team_win' in df.columns:
        df['home_team_win'] = df['home_team_win'].map({True: 1, False: -1})
    df['is_night_game'] = df['is_night_game'].map({True: 1, False: -1})
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'season' in df.columns:
        df = df.drop(columns=['season'])
    if 'home_team_win' in df.columns:
        X = df.drop(columns=['home_team_win'])
        y = df['home_team_win']
        y = y.to_numpy()
    else:
        X = df
        y = []
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_features = X.select_dtypes(include=['number']).columns
    X[num_features] = num_imputer.fit_transform(X[num_features])
    X_numeric = X.select_dtypes(include=['number'])
    
    cat_features = X.select_dtypes(exclude=['number']).columns
    X[cat_features] = cat_imputer.fit_transform(X[cat_features])
    X[cat_features] = X[cat_features].astype('category')
    X = pd.get_dummies(X, drop_first=True)

    if numeric: 
        X = X_numeric.to_numpy()
    else:
        X = X.to_numpy()
    return X, y

def blending_with_random_seed_single_season(season):
    models = []
    alphas = []
    for i in range(1,10):
        X, y = preprocess('train_data.csv', True, season)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.fit_transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_test)
        error_rate = 1 - accuracy_score(y_test, y_pred)
        print(error_rate)
        alphas.append(1/error_rate - 1)
    alphas = np.log(alphas)
    return models, alphas    

def blending_with_random_seed():
    models = []
    alphas = []
    for i in range(1,10):
        X, y = preprocess('train_data.csv', True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.fit_transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_test)
        error_rate = 1 - accuracy_score(y_test, y_pred)
        print(error_rate)
        alphas.append(1/error_rate - 1)
    alphas = np.log(alphas)
    return models, alphas    

def blending_with_year():
    models = []
    alphas = []
    for i in range(2016, 2024):
        if i == 2020:
            continue
        X, y = preprocess('train_data.csv', True, i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.fit_transform(X_test)
        model = LogisticRegression()
        model.fit(X_train, y_train)
        models.append(model)
        y_pred = model.predict(X_test)
        error_rate = 1 - accuracy_score(y_test, y_pred)
        print(error_rate)
        alphas.append(1/error_rate - 1)
    alphas = np.log(alphas)
    return models, alphas

def write_to_file(y_pred, filename):
    data_mapped = np.where(y_pred == -1, True, True)
    ids = np.arange(0, len(data_mapped))
    df = pd.DataFrame({
        'id': ids,
        'home_team_win': data_mapped
    })
    df.to_csv(filename, index=False)
    print("CSV file created successfully.")

X, y = preprocess('train_data.csv', True)
#X_t, _= preprocess('same_season_test_data.csv', True)
X_t, _ = preprocess('2024_test_data.csv',True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
print(len(X_t))
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)
X_t = scalar.fit_transform(X_t)
base_model = DecisionTreeClassifier(max_depth=2)
adaboost_model = AdaBoostClassifier(base_model, n_estimators=10, random_state=42)
adaboost_model.fit(X_train, y_train)
y_pred = adaboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"n_estimators={10}, Accuracy={accuracy}")

y_pred_t = np.zeros(len(X_t))
y_pred = np.zeros(len(y_test))
models, alphas = blending_with_year()
for i in range(0, len(alphas)):
    pred = models[i].predict(X_test)
    pred_t = models[i].predict(X_t)
    print(pred)
    y_pred += alphas[i]*pred
    y_pred_t += alphas[i]*pred_t
y_pred = np.sign(y_pred)
y_pred_t = np.sign(y_pred_t)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
write_to_file(y_pred_t, "log_b_year.csv")

y_pred = np.zeros(len(y_test))
y_pred_t = np.zeros(len(X_t))
models, alphas = blending_with_random_seed()
for i in range(0, len(alphas)):
    pred = models[i].predict(X_test)
    pred_t = models[i].predict(X_t)
    print(pred)
    y_pred += alphas[i]*pred
    y_pred_t += alphas[i]*pred_t
y_pred = np.sign(y_pred)
y_pred_t = np.sign(y_pred_t)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
write_to_file(y_pred_t, "log_b_seed.csv")

'''
df = pd.read_csv("same_season_test_data.csv")
unique_seasons = df['season'].dropna().unique()
for i in df[df['season'].isna()].index:
    df.loc[i, 'season'] = np.random.choice(unique_seasons)

#randomly assign season year

total_y_predt = np.zeros(len(X_t))
for i in range(2016, 2024):
    if i == 2020:
        continue
    X, y = preprocess('train_data.csv', True, i)
    X_t, _= preprocess_2(df, True, i)
    year_indices = df[df['season'] == i].index
    print(year_indices)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.fit_transform(X_test)
    X_t = scalar.fit_transform(X_t)

    y_pred = np.zeros(len(y_test))
    y_pred_t = np.zeros(len(X_t))
    models, alphas = blending_with_random_seed_single_season(i)
    for j in range(0, len(alphas)):
        pred = models[j].predict(X_test)
        pred_t = models[j].predict(X_t)
        y_pred += alphas[j]*pred
        y_pred_t += alphas[j]*pred_t
    y_pred = np.sign(y_pred)
    y_pred_t = np.sign(y_pred_t)
    total_y_predt[year_indices] = y_pred_t
    print(f"Accuracy for {i} year: {accuracy_score(y_test, y_pred)}")
write_to_file(total_y_predt, "log_b_yaer_seed.csv")
        
'''