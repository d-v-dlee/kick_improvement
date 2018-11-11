import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier

def clean_kick_data(csv_name):
    '''
    input csv name 
    output cleaned dataframe with kickers of minimum 25 kicks, dropped null values,
    optional yardage bins currently hashed out, dropped description
    '''
    kick_df = pd.read_csv(csv_name)
    kick_df = kick_df.dropna()
    kick_df.drop('desc', inplace=True, axis=1)
    # bins = [15, 30, 35, 40, 45, 50, 55, 60, 100]
    # names = ['<30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+']
    # kick_df['Kick Range'] = pd.cut(kick_df['FieldGoalDistance'], bins, labels=names)
    kick_df_min = kick_df.groupby('KickerName').filter(lambda x: len(x) >= 25)
    kick_df1 = kick_df_min[['FieldGoalDistance', 'HomeTeam', 'Month', 'KickerName', 'FieldGoalResult']]

    return kick_df1

def kick_dictionary(df, column_name):
    """
    create dictionary with stadium or player name as key 
    and corresponding number as values.
    ex {'J.Tucker':1, 'B.Walsch':2}
    ex {'SEA':1, etc}
    """

    unique_vals = df[column_name].unique()
    
    kicker_dict = {}
    i = 1
    for unique in unique_vals:
        kicker_dict[unique] = i
        i += 1
    return kicker_dict

def replace_kicker_with_num(df, column_name):
    kicker_dict = kick_dictionary(df, column_name)
    kicker_nums = df.replace({column_name:kicker_dict})
    return kicker_nums


def data_splitter(df, frac_per=.3, random_state=30):
    '''
    input dataframe
    return x% of original dataframe and X and y 
    '''
    test_df = df.sample(frac=frac_per, random_state=30)
    final_test_x = test_df.iloc[:, :-2]
    final_test_y = test_df.iloc[:, -1]
    return test_df, final_test_x, final_test_y

def model_comparison(data, y):
    '''
    input data (X) and y to get log loss scores of different models
    '''
    X_train, X_test, y_train, y_test = train_test_split(data,y)
    logistic_model = LogisticRegression()
    gradient_boost_model = GradientBoostingClassifier(learning_rate=0.005, max_depth=6, max_features='log2', min_samples_leaf=4, n_estimators=500, subsample=0.25)
    random_forest_model = RandomForestClassifier(n_estimators=300, max_depth=3, verbose=1)

    logistic_model.fit(X_train, y_train)
    gradient_boost_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    p_random_forest = random_forest_model.predict_proba(X_test)
    p_gradient_boost =  gradient_boost_model.predict_proba(X_test)
    p_logistic = logistic_model.predict_proba(X_test)

    ensemble_p = (p_random_forest[:,1] + p_gradient_boost[:,1] + p_logistic[:,1])/3

    random_forest_ll = log_loss(y_test, p_random_forest )
    gradient_boost_ll = log_loss(y_test, p_gradient_boost )
    logistic_ll = log_loss(y_test, p_logistic )
    ensemble_ll = log_loss(y_test, ensemble_p )

    # fig, axs = plot_partial_dependence(gradient_boost_model, X = X_train, features = [0,1,2,3,(1,4)],
    #                                        feature_names=list(X_train.columns),
    #                                        n_jobs=1, grid_resolution=100, figsize = (20, 20))
    # plt.show()                                       


    print("Ensemble Log Loss " + str(ensemble_ll))
    print("Gradient Boost Log Loss " + str(gradient_boost_ll))
    print("Random Forest Log Loss " + str(random_forest_ll))
    print("Logistic Log Loss " + str(logistic_ll))

def kicker_dataframe(kickers_dict, stadium_dict, kicker_name, home_team, month):
    """
    creates dataframe of possible kicks from 25-62 yards
    """
    kicker_num = kickers_dict[kicker_name]
    stadium_num = stadium_dict[home_team]
    possible_kicks = pd.DataFrame({'FieldGoalDistance': list(range(25, 62)),
                                    'HomeTeam': [stadium_num for x in range(37)],
                                    'Month': [month for x in range(37)],
                                    'KickerName': [kicker_num for x in range(37)]

    })
    return possible_kicks
