""" import package """
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import pickle
from data_preparation import preparation
import random
from sklearn.metrics import mean_absolute_error



''' preparation '''
ran = random.randint(0, 9)
input_file = './test/test_dataset_{}.csv'.format(ran)
output_file = './test_pre/test_pre_dataset_{}.csv'.format(ran)
print("-------------- do preparation for {} and output file as {}".format(input_file, output_file))
pre = preparation(input_file, output_file)
pre.execution()



""" load data """
print('---------- load data from {} ---------------'.format(output_file))
df_data = pd.read_csv(output_file)
print(df_data.head())



""" model validation """
lab = '信用分'
X_pred = df_data.loc[df_data[lab], (df_data.columns != lab) & (df_data.columns != '用户编码')]


print('--------------- download model(lgb_model_l1) and do prediction ------------------')
y_counts = 0
y_pred_l1 = np.zeros([5, X_pred.shape[0]])
y_pred_all_l1 = np.zeros(X_pred.shape[0])
for i in range(5):
    print("{}、".format(i + 1), 'loading model from ./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i))
    with open('./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i), 'rb') as f:
        lgb_model = pickle.load(f)
    y_pred_l1[y_counts] = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)
    y_pred_all_l1 += y_pred_l1[y_counts]
    y_counts += 1

y_pred_all_l1 /= y_counts
print('y_pred_all_l1[:5]:', y_pred_all_l1[:5])



print('--------------- download model(lgb_model_l2) and do prediction ------------------')
y_counts = 0
y_pred_l2 = np.zeros([5, X_pred.shape[0]])
y_pred_all_l2 = np.zeros(X_pred.shape[0])
for i in range(5):
    print("{}、".format(i + 1), 'loading model from ./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i))
    with open('./lgb_model_l2/lgb_model_l2_{}.pickle'.format(i), 'rb') as f:
        lgb_model = pickle.load(f)
    y_pred_l2[y_counts] = lgb_model.predict(X_pred, num_iteration=lgb_model.best_iteration)
    y_pred_all_l2 += y_pred_l2[y_counts]
    y_counts += 1

y_pred_all_l2 /= y_counts
print('y_pred_all_l2[:5]:', y_pred_all_l2[:5])



print('------------------ do model fusion ----------------')
submit = pd.DataFrame()
submit['id'] = df_data['用户编码']

submit['score1'] = y_pred_all_l1
submit['score2'] = y_pred_all_l2

submit = submit.sort_values('score1')
submit['rank'] = np.arange(submit.shape[0])

min_rank = 100
max_rank = 50000 - min_rank

l1_ext_rate = 1
l2_ext_rate = 1 - l1_ext_rate
il_ext = (submit['rank'] <= min_rank) | (submit['rank'] >= max_rank)

l1_not_ext_rate = 0.5
l2_not_ext_rate = 1 - l1_not_ext_rate
il_not_ext = (submit['rank'] > min_rank) & (submit['rank'] < max_rank)

submit['score'] = 0
submit.loc[il_ext, 'score'] = (submit[il_ext]['score1'] * l1_ext_rate + submit[il_ext]['score2'] * l2_ext_rate + 1 + 0.25)
submit.loc[il_not_ext, 'score'] = submit[il_not_ext]['score1'] * l1_not_ext_rate + submit[il_not_ext]['score2'] * l2_not_ext_rate + 0.25
submit['score'] = submit['score'].apply(lambda x: round(x))
# """ output result """
# submit[['id', 'score']].to_csv('submit.csv', index=False)


y_true = df_data[lab]
y_pred = submit['score']
score = 1 / (1 + mean_absolute_error(y_true, y_pred))
print('------------------- output score result for {} ------------------'.format(input_file))
print('score:', score)