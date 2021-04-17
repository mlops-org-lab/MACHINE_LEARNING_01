""" import package """
import pandas as pd
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.metrics import mean_absolute_error


""" load data """
file_path = './pre_dataset.csv'
print('---------- load data from {} ---------------'.format(file_path))
df_data = pd.read_csv(file_path)
print(df_data.head())


lab = '信用分'
X = df_data.loc[df_data[lab].notnull(), (df_data.columns != lab) & (df_data.columns != '用户编码')]
y = df_data.loc[df_data[lab].notnull()][lab]
X_pred = df_data.loc[df_data[lab].isnull(), (df_data.columns != lab) & (df_data.columns != '用户编码')]




def feval_lgb(y_pred, train_data):
    y_true = train_data.get_label()
    #y_pred = np.argmax(y_pred.reshape(7, -1), axis=0)

    score = 1 / (1 + mean_absolute_error(y_true, y_pred))
    return 'acc_score', score, True



""" 模型参数 """
lgb_param_l1 = {
    'learning_rate': 0.01, #梯度下降的步长
    'boosting_type': 'gbdt',#梯度提升决策树
    'objective': 'regression_l1', #任务目标（L1 loss, alias=mean_absolute_error, mae）
    'metric': 'None',
    'min_child_samples': 46,# 一个叶子上数据的最小数量
    'min_child_weight': 0.01,
    'feature_fraction': 0.6,#每次迭代中选择前60%的特征
    'bagging_fraction': 0.8,#不进行重采样的情况下随机选择部分数据
    'bagging_freq': 2, #每2次迭代执行bagging
    'num_leaves': 31,#一棵树上的叶子数
    'max_depth': 5,#树的最大深度
    'lambda_l2': 1, # 表示的是L2正则化
    'lambda_l1': 0,# 表示的是L1正则化
    'n_jobs': -1,
    'seed': 4590,
    'verbose': -1
}



n_fold = 5
y_counts = 0
for n in range(1):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2019 + n)
    kf = kfold.split(X, y)

    for i, (train_iloc, test_iloc) in enumerate(kf):
        print("{}、".format(i + 1), end='')
        X_train, X_test, y_train, y_test = X.iloc[train_iloc, :], X.iloc[test_iloc, :], y[train_iloc], y[test_iloc]
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        lgb_model = lgb.train(train_set=lgb_train, valid_sets=lgb_valid, feval=feval_lgb,
                              params=lgb_param_l1,num_boost_round=6000, verbose_eval=-1, early_stopping_rounds=100)

        with open('./lgb_model_l1/lgb_model_l1_{}.pickle'.format(i), 'wb') as f:
            pickle.dump(lgb_model, f)
        y_counts += 1




lgb_param_l2 = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'None',
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'num_leaves': 40,
    'max_depth': 7,
    'lambda_l2': 1,
    'lambda_l1': 0,
    'n_jobs': -1,
    'verbose': -1    #去除warning
}



n_fold = 5
y_counts = 0
for n in range(1):
    kfold = KFold(n_splits=n_fold, shuffle=True, random_state=2019 + n)
    kf = kfold.split(X, y)

    for i, (train_iloc, test_iloc) in enumerate(kf):
        print("{}、".format(i + 1), end='')
        X_train, X_test, y_train, y_test = X.iloc[train_iloc, :], X.iloc[test_iloc, :], y[train_iloc], y[test_iloc]
        #print(len(y_test))
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
        lgb_model = lgb.train(train_set=lgb_train, valid_sets=lgb_valid, feval=feval_lgb,
                              params=lgb_param_l2,num_boost_round=6000, verbose_eval=-1, early_stopping_rounds=100)
        with open('./lgb_model_l2/lgb_model_l2_{}.pickle'.format(i), 'wb') as f:
            pickle.dump(lgb_model, f)
        y_counts += 1
