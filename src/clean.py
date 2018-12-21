import pandas as pd
import numpy as np
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt



def make_dummies(test_col, train_unique_vals, col_name):

    """
    Return a df containing len(train_unique_vals) columns for 
    each unique value in train_unique_vals. If the test_col has more 
    unique values that are not seen in train_unique_vals, value
    will be 0
    """

    dummies = {}
    for val in train_unique_vals:
        dummies[col_name + '_' + val] = (test_col == val).astype(int)
    return pd.DataFrame(dummies, index = test_col.index)


def make_dummies_dataframe(data, categories):

    """
    creates dummy variables for multiple categories
    ex categories = ['city', 'phone'], make_dummies_dataframe(data, categories)

    """
    dummy_dfs = []
    for category in categories:
        temp_df = make_dummies(data[category], data[category].unique(), category)
        dummy_dfs.append(temp_df)
    for i in dummy_dfs:
        data_transformed = pd.concat([data, i], axis=1)
        data = data_transformed
    return data_transformed



def clean_kick_data(csv_name):
    '''
    input: csv name 
    output: cleaned dataframe
        numeric month values --> strings (9 -> 'September')
        kickers filtered out for those with a minumum of 25 kicks
        dummy columns created for categories HomeTeam, KickerName, Month 
            those columns are then dropped

    final return: df that is (5178, 93) including 51 unique kickers
    '''
    kick_df = pd.read_csv(csv_name)
    kick_df = kick_df.dropna()
    kick_df.drop('desc', inplace=True, axis=1)
    kick_df['Month'] = kick_df['Month'].replace(9, 'September')
    kick_df['Month'] = kick_df['Month'].replace(10, 'October')
    kick_df['Month'] = kick_df['Month'].replace(11, 'November')
    kick_df['Month'] = kick_df['Month'].replace(12, 'December')
    kick_df['Month'] = kick_df['Month'].replace(1, 'January')
    kick_df_min = kick_df.groupby('KickerName').filter(lambda x: len(x) >= 25)
    kick_df1 = kick_df_min[['FieldGoalDistance', 'HomeTeam', 'Month', 'KickerName', 'FieldGoalResult']]

    categories = ['HomeTeam', 'KickerName', 'Month']
    fg_df = make_dummies_dataframe(kick_df1, categories)
    fg_df.drop(columns = ['HomeTeam', 'KickerName', 'Month'], inplace=True)

    return fg_df

columns = ['HomeTeam_DEN', 'HomeTeam_BUF',
       'HomeTeam_PIT', 'HomeTeam_NO', 'HomeTeam_NYJ', 'HomeTeam_CAR',
       'HomeTeam_CHI', 'HomeTeam_CLE', 'HomeTeam_DET', 'HomeTeam_IND',
       'HomeTeam_SF', 'HomeTeam_STL', 'HomeTeam_DAL', 'HomeTeam_WAS',
       'HomeTeam_SD', 'HomeTeam_NE', 'HomeTeam_ATL', 'HomeTeam_PHI',
       'HomeTeam_KC', 'HomeTeam_HOU', 'HomeTeam_GB', 'HomeTeam_BAL',
       'HomeTeam_TB', 'HomeTeam_ARI', 'HomeTeam_OAK', 'HomeTeam_NYG',
       'HomeTeam_SEA', 'HomeTeam_CIN', 'HomeTeam_TEN', 'HomeTeam_MIN',
       'HomeTeam_MIA', 'HomeTeam_JAC', 'HomeTeam_LA', 'HomeTeam_JAX',
       'HomeTeam_LAC', 'KickerName_J.Tucker', 'KickerName_S.Gostkowski',
       'KickerName_R.Bironas', 'KickerName_M.Bryant', 'KickerName_G.Hartley',
       'KickerName_N.Folk', 'KickerName_R.Lindell', 'KickerName_S.Hauschka',
       'KickerName_R.Gould', 'KickerName_C.Sturgis', 'KickerName_B.Cundiff',
       'KickerName_B.Walsh', 'KickerName_S.Janikowski', 'KickerName_P.Dawson',
       'KickerName_G.Zuerlein', 'KickerName_J.Feely', 'KickerName_D.Bailey',
       'KickerName_J.Brown', 'KickerName_A.Henery', 'KickerName_K.Forbath',
       'KickerName_R.Bullock', 'KickerName_N.Novak', 'KickerName_R.Succop',
       'KickerName_A.Vinatieri', 'KickerName_M.Crosby',
       'KickerName_D.Carpenter', 'KickerName_G.Gano', 'KickerName_J.Scobee',
       'KickerName_M.Prater', 'KickerName_S.Suisham', 'KickerName_M.Nugent',
       'KickerName_S.Graham', 'KickerName_C.Parkey', 'KickerName_C.Santos',
       'KickerName_B.McManus', 'KickerName_C.Catanzaro', 'KickerName_P.Murray',
       'KickerName_C.Barth', 'KickerName_T.Coons', 'KickerName_A.Franks',
       'KickerName_J.Myers', 'KickerName_J.Lambo', 'KickerName_D.Hopkins',
       'KickerName_C.Boswell', 'KickerName_R.Aguayo', 'KickerName_W.Lutz',
       'KickerName_Z.Gonzalez', 'KickerName_A.Rosas', 'KickerName_K.Fairbairn',
       'KickerName_J.Elliott', 'KickerName_H.Butker', 'Month_September',
       'Month_October', 'Month_November', 'Month_December', 'Month_January']

final_columns = ['FieldGoalDistance', 'HomeTeam_DEN', 'HomeTeam_BUF',
       'HomeTeam_PIT', 'HomeTeam_NO', 'HomeTeam_NYJ', 'HomeTeam_CAR',
       'HomeTeam_CHI', 'HomeTeam_CLE', 'HomeTeam_DET', 'HomeTeam_IND',
       'HomeTeam_SF', 'HomeTeam_STL', 'HomeTeam_DAL', 'HomeTeam_WAS',
       'HomeTeam_SD', 'HomeTeam_NE', 'HomeTeam_ATL', 'HomeTeam_PHI',
       'HomeTeam_KC', 'HomeTeam_HOU', 'HomeTeam_GB', 'HomeTeam_BAL',
       'HomeTeam_TB', 'HomeTeam_ARI', 'HomeTeam_OAK', 'HomeTeam_NYG',
       'HomeTeam_SEA', 'HomeTeam_CIN', 'HomeTeam_TEN', 'HomeTeam_MIN',
       'HomeTeam_MIA', 'HomeTeam_JAC', 'HomeTeam_LA', 'HomeTeam_JAX',
       'HomeTeam_LAC', 'KickerName_J.Tucker', 'KickerName_S.Gostkowski',
       'KickerName_R.Bironas', 'KickerName_M.Bryant', 'KickerName_G.Hartley',
       'KickerName_N.Folk', 'KickerName_R.Lindell', 'KickerName_S.Hauschka',
       'KickerName_R.Gould', 'KickerName_C.Sturgis', 'KickerName_B.Cundiff',
       'KickerName_B.Walsh', 'KickerName_S.Janikowski', 'KickerName_P.Dawson',
       'KickerName_G.Zuerlein', 'KickerName_J.Feely', 'KickerName_D.Bailey',
       'KickerName_J.Brown', 'KickerName_A.Henery', 'KickerName_K.Forbath',
       'KickerName_R.Bullock', 'KickerName_N.Novak', 'KickerName_R.Succop',
       'KickerName_A.Vinatieri', 'KickerName_M.Crosby',
       'KickerName_D.Carpenter', 'KickerName_G.Gano', 'KickerName_J.Scobee',
       'KickerName_M.Prater', 'KickerName_S.Suisham', 'KickerName_M.Nugent',
       'KickerName_S.Graham', 'KickerName_C.Parkey', 'KickerName_C.Santos',
       'KickerName_B.McManus', 'KickerName_C.Catanzaro', 'KickerName_P.Murray',
       'KickerName_C.Barth', 'KickerName_T.Coons', 'KickerName_A.Franks',
       'KickerName_J.Myers', 'KickerName_J.Lambo', 'KickerName_D.Hopkins',
       'KickerName_C.Boswell', 'KickerName_R.Aguayo', 'KickerName_W.Lutz',
       'KickerName_Z.Gonzalez', 'KickerName_A.Rosas', 'KickerName_K.Fairbairn',
       'KickerName_J.Elliott', 'KickerName_H.Butker', 'Month_September',
       'Month_October', 'Month_November', 'Month_December', 'Month_January']

log_model = pickle.load(open("log.pkl", "rb"))

def make_kicker_prob(HomeTeam, KickerName, Month):
    """
    input: 
    HomeTeam in format 'HomeTeam_SEA';
    KickerName in format 'KickerName_N.Folk'
    Month in format 'Month_November'

    output figure of probability
    """
    
    kick_dict = OrderedDict()
    kick_dict[HomeTeam] = [1] * 45
    kick_dict[KickerName] = [1] * 45
    kick_dict[Month] = [1] * 45
    kick_dict['FieldGoalDistance'] = list(range(20, 65))

    inputs = [HomeTeam, KickerName, Month]
    other_columns = [x for x in columns if x not in inputs]
    for column in other_columns:
        kick_dict[column] = [0] * 45
    
    ready_df = pd.DataFrame.from_dict(kick_dict)
    model_ready_df = ready_df[final_columns]
    p_logistic = log_model.predict_proba(model_ready_df)
    plt.bar(model_ready_df['FieldGoalDistance'], p_logistic[:, 1])

def get_probs(HomeTeam, KickerName, Month):
    kick_dict = OrderedDict()
    kick_dict[HomeTeam] = [1] * 45
    kick_dict[KickerName] = [1] * 45
    kick_dict[Month] = [1] * 45
    kick_dict['FieldGoalDistance'] = list(range(20, 65))

    inputs = [HomeTeam, KickerName, Month]
    other_columns = [x for x in columns if x not in inputs]
    for column in other_columns:
        kick_dict[column] = [0] * 45
    
    ready_df = pd.DataFrame.from_dict(kick_dict)
    model_ready_df = ready_df[final_columns]
    p_logistic = log_model.predict_proba(model_ready_df)
    return p_logistic





