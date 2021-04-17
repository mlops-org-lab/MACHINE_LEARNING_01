""" import package """
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle




""" load data """
file_path = './pre_dataset.csv'
print('---------- load data from {} ---------------'.format(file_path))
df_data = pd.read_csv(file_path)
print(df_data.head())



lab = '信用分'

X = df_data.loc[df_data[lab].notnull(), (df_data.columns != lab) & (df_data.columns != '用户编码')]
y = df_data.loc[df_data[lab].notnull()][lab]
X_pred = df_data.loc[df_data[lab].isnull(), (df_data.columns != lab) & (df_data.columns != '用户编码')]


y_counts = 0
y_scores = np.zeros(5)
for i in range(5):
    print("{}、".format(i + 1), 'loading model from ./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i))
    with open('./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i), 'rb') as f:
        lgb_model = pickle.load(f)
    y_scores[y_counts] = lgb_model.best_score['valid_0']['acc_score']
    y_counts += 1

print(y_scores, y_scores.mean())




y_counts = 0
y_scores = np.zeros(5)
for i in range(5):
    print("{}、".format(i + 1), 'loading model from ./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i))
    with open('./lgb_model_l2/lgb_model_l2_{}.pickle'.format(i), 'rb') as f:
        lgb_model = pickle.load(f)
    y_scores[y_counts] = lgb_model.best_score['valid_0']['acc_score']
    y_counts += 1

print(y_scores, y_scores.mean())

