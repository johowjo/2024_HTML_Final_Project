import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('train_data.csv')
season_year = input("Please input the sepcific season year (2016~2023/2020): ")
if(season_year != "" and season_year != "2020"):
    df = df[df['season'] == int(season_year)]
print("DataFrame:\n", df)
label='home_team_win'
df_true = df[df[label] == True]
df_false = df[df[label] == False]

column1 = 'home_team_abbr'
column2 = 'away_team_abbr'
feature_win_counts = df_true.groupby(column1).size()+df_false.groupby(column2).size()
feature_loss_counts = df_false.groupby(column1).size()+df_true.groupby(column2).size()
plt.figure()
feature_win_counts.plot(kind='bar', alpha=0.5, color='blue', label='Win', position=1, width=0.2)
feature_loss_counts.plot(kind='bar', alpha=0.5, color='red', label='Loss', position=0, width=0.2)
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig(f'./Data/totalwin_loss.png', bbox_inches='tight')

for column in df.columns:
    if column != label and column != 'date' and column != 'id' and column != 'home_pitcher' and column != 'away_pitcher' and column != 'home_team_season':
        if df[column].dtype == 'object':
            feature_true_counts = df_true.groupby(column).size()
            feature_false_counts = df_false.groupby(column).size()
            plt.figure()
            plt.subplots_adjust(left=0.1)
            feature_true_counts.plot(kind='bar', alpha=0.5, color='blue', label='Home_Win', position=1, width=0.2)
            feature_false_counts.plot(kind='bar', alpha=0.5, color='red', label='Home_Loss', position=0, width=0.2)
            plt.xlabel(f'{column}')
            plt.ylabel('Frequency')
            plt.legend(loc='upper right')
            plt.savefig(f'./Data/{column}.png', bbox_inches='tight')
        else:
            df[column] = df[column].fillna(df[column].mean())
            plt.figure()
            plt.hist(df_true[column], bins=30, alpha=0.5, label='Home_Win', color='blue', align='mid')
            plt.hist(df_false[column], bins=30, alpha=0.5, label='Home_Loss', color='red', align='mid')
            plt.title(f'Feature "{column}" vs Label (Numerical)')
            plt.xlabel(column)
            plt.legend(loc='upper right')
            plt.ylabel('Label')
            plt.savefig(f'./Data/{column}.png')


data_array = df.to_numpy()