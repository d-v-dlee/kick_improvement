import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

kick_df = pd.read_csv('field_goals.csv')

#drop two rows of null values
kick_df = kick_df.dropna()

#drop kick description column
kick_df.drop('desc', inplace=True, axis=1)

#add column that splits classifies kick to a 5-yard interval
bins = [15, 30, 35, 40, 45, 50, 55, 60, 100]
names = ['<30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+']

kick_df['Kick Range'] = pd.cut(kick_df['FieldGoalDistance'], bins, labels=names)

kicks_by_distance = pd.crosstab(kick_df['FieldGoalDistance'], kick_df['FieldGoalResult'])
kicks_by_distance['prob'] = ((kicks_by_distance[1] / (kicks_by_distance[0] + kicks_by_distance[1])) - .001).round(3)
kicks_by_distance['prob'][66] = 0.05
kicks_by_distance['prob'][68] = 0.05
kicks_by_distance['prob'][71] = 0.05

def prior_prob_bar():
    _ = kicks_by_distance['prob'].plot(kind="bar", figsize=(15, 8), color='green', alpha=0.5)

def all_kicks_bar():
    """
    shows all kicks and attempts at 5 yard ranges
    """
    
    N = 8
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, kick_df[kick_df['FieldGoalResult'] == 1]['Kick Range'].value_counts().sort_index(), width, color='#d62728')
    p2 = plt.bar(ind, kick_df[kick_df['FieldGoalResult'] == 0]['Kick Range'].value_counts().sort_index(), width,
    )

    plt.ylabel('Number of Field Goals')
    plt.title('All Field Goal Attempts and Makes')
    plt.xticks(ind, ('<30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+'))
    plt.yticks(np.arange(0, 1600, 100))
    plt.legend((p1[0], p2[0]), ('Makes', 'Misses'))

    plt.show()

def kick_jitter():
    noise = np.random.rand(kick_df.shape[0]) * 0.2
    plt.scatter(kick_df.iloc[:, 0], kick_df.iloc[:, 1] + noise, s=5, color="k",
                alpha=0.1)
    plt.yticks([0, 1])
    plt.ylabel("Field Goal Result")
    plt.xlabel("Field Goal Distance")
    plt.title("Field Goal Results by Distance")

def home_stadium_visuals(hometeam =str):
    """
    input string of HomeTeam like 'DEN' or 'SEA'
    returns bar graphs of missed and made fields goals
    """
    home_kick = kick_df[(kick_df['HomeTeam'] == hometeam)].copy()
    
    N = 8
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, home_kick[home_kick['FieldGoalResult'] == 1]['Kick Range'].value_counts().sort_index(), width, color='#d62728')
    p2 = plt.bar(ind, home_kick[home_kick['FieldGoalResult'] == 0]['Kick Range'].value_counts().sort_index(), width,
                )

    plt.ylabel('Number of Field Goals')
    plt.title(f'All Field Goal Attempts and Makes in {hometeam} Stadium')
    plt.xticks(ind, ('<30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+'))
    plt.yticks(np.arange(0, 60, 5))
    plt.legend((p1[0], p2[0]), ('Makes', 'Misses'))

    plt.show()

def kicker_visuals(kickername=str):
    """
    input string of KickerName ex 'J.Tucker'
    returns bar graph of missed and made field goals
    """
    player_kick = kick_df[kick_df['KickerName']== kickername].copy()
    N = 8
    ind = np.arange(N)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, player_kick[player_kick['FieldGoalResult'] == 1]['Kick Range'].value_counts().sort_index(), width, color='#d62728')
    p2 = plt.bar(ind, player_kick[player_kick['FieldGoalResult'] == 0]['Kick Range'].value_counts().sort_index(), width,
                )

    plt.ylabel('Number of Field Goals')
    plt.title(f'Field Goal Attempts and Makes By {kickername}')
    plt.xticks(ind, ('<30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+'))
    plt.yticks(np.arange(0, 50, 5))
    plt.legend((p1[0], p2[0]), ('Makes', 'Misses'))

    plt.show()

def kicker_yard_visuals(kickername=str):
    """
    input a string of a KickerName ex. 'J.Tucker.'
    returns a bar graph of all attempts and misses by exact yardage
    """
    player_kick = kick_df[kick_df['KickerName']== kickername].copy()
    makes_by_range = pd.crosstab(player_kick['FieldGoalDistance'], player_kick['FieldGoalResult'])
    makes_by_range.plot(kind="bar", figsize=(15, 8))

def stadium_yard_visuals(hometeam=str):
    """
    input a string of a HomeTeam ex. 'SEA'
    returns a bar graph of all attempts and misses by exact yardage
    """
    stad_kick = kick_df[kick_df['HomeTeam']== hometeam].copy()
    makes_by_range = pd.crosstab(stad_kick['FieldGoalDistance'], stad_kick['FieldGoalResult'])
    makes_by_range.plot(kind="bar", figsize=(15, 8))





