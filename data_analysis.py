""" import package """
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

'''config matplotlib attribution'''
plt.style.use("bmh")
plt.rc('font', family='SimHei', size=13)
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

""" load data """
file_path = './data/train_dataset.csv'
print('---------- load data from {} ---------------'.format(file_path))
df_data = pd.read_csv(file_path)


""" data's info """
print("---------- data's info ---------------")
df_data.info()
print("共有数据集：", df_data.shape[0])



""" data's type """
print("---------- data's type ---------------")
for i,name in enumerate(df_data.columns):
    name_sum = df_data[name].value_counts().shape[0]
    print("{}、{}      The number of types of features is：{}".format(i + 1, name, name_sum))


""" data's describe """
print("---------- data's describe ---------------")
print(df_data.describe())


""" Observe the data distribution of training / testing dataset """
print('---------- Observe the data distribution of training / test dataset ---------------')
print(df_data[df_data['信用分'].isnull()].describe())
print(df_data[df_data['信用分'].notnull()].describe())


""" Tailing / sequence feature analysis """
print("-------------- Tailing / sequence feature analysis --------------")
f, ax = plt.subplots(figsize=(20, 6))

sns.scatterplot(data=df_data, x='当月通话交往圈人数', y='信用分', color='k', ax=ax)
plt.show()

import seaborn as sns
name_list = ['当月旅游资讯类应用使用次数', '当月火车类应用使用次数', '当月物流快递类应用使用次数', '当月网购类应用使用次数',
             '当月视频播放类应用使用次数', '当月金融理财类应用使用总次数', '当月飞机类应用使用次数', '用户年龄',
             '用户当月账户余额（元）', '用户账单当月总费用（元）', '用户近6个月平均消费值（元）', '缴费用户最近一次缴费金额（元）']

f, ax = plt.subplots(3, 4, figsize=(20, 20))
for i,name in enumerate(name_list):
    sns.scatterplot(data=df_data, x=name, y='信用分', color='b', ax=ax[i // 4][i % 4])
plt.show()


f, ax = plt.subplots(1, 3, figsize=(20, 6))
sns.kdeplot(data=df_data['当月飞机类应用使用次数'], color='r', shade=True, ax=ax[0])
sns.kdeplot(data=df_data['当月火车类应用使用次数'], color='c', shade=True, ax=ax[1])
sns.kdeplot(data=df_data['当月旅游资讯类应用使用次数'], color='b', shade=True, ax=ax[2])
plt.show()


""" Discrete characteristic analysis """
print("-------------------- Discrete characteristic analysis --------------------")
f, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.boxplot(data=df_data, x='用户最近一次缴费距今时长（月）', y='信用分', ax=ax[0])
sns.boxplot(data=df_data, x='缴费用户当前是否欠费缴费', y='信用分', ax=ax[1])
plt.show()


name_list = ['当月是否体育场馆消费', '当月是否到过福州山姆会员店', '当月是否景点游览', '当月是否看电影', '当月是否逛过福州仓山万达',
             '是否4G不健康客户', '是否大学生客户', '是否经常逛商场的人', '是否黑名单客户', '用户实名制是否通过核实']

f, ax = plt.subplots(2, 5, figsize=(20, 12))
for i,name in enumerate(name_list):
    sns.boxplot(data=df_data, x=name, y='信用分', ax=ax[i // 5][i % 5])
plt.show()


f, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df_data, x='用户话费敏感度', y='信用分', ax=ax)
plt.show()


'''Association characteristics'''
print("----------------- Association characteristics ---------------------")
f, ax = plt.subplots(figsize=(20, 6))
sns.boxenplot(data=df_data, x='当月是否逛过福州仓山万达', y='信用分', hue='当月是否到过福州山姆会员店', ax=ax)
plt.show()


""" The exploration of discrete model """
print("------------- The exploration of discrete model -----------------")
f, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(1, 5, figsize=(20, 6))
sns.boxplot(data=df_data, x='当月是否逛过福州仓山万达', y='信用分', hue='是否经常逛商场的人', ax=ax0)
sns.boxplot(data=df_data, x='当月是否到过福州山姆会员店', y='信用分', hue='是否经常逛商场的人', ax=ax1)
sns.boxplot(data=df_data, x='当月是否看电影', y='信用分', hue='是否经常逛商场的人', ax=ax2)
sns.boxplot(data=df_data, x='当月是否景点游览', y='信用分', hue='是否经常逛商场的人', ax=ax3)
sns.boxplot(data=df_data, x='当月是否体育场馆消费', y='信用分', hue='是否经常逛商场的人', ax=ax4)
plt.show()


""" Continuous exploration """
print("------------- Continuous exploration -----------------")
f, ax = plt.subplots(1, 2, figsize=(20, 6))

sns.scatterplot(data=df_data, x='用户账单当月总费用（元）', y='信用分', color='b', ax=ax[0])
sns.scatterplot(data=df_data, x='用户当月账户余额（元）', y='信用分', color='r', ax=ax[1])
plt.show()


f, ax = plt.subplots(1, 2, figsize=(20, 6))

sns.scatterplot(data=df_data, x='用户账单当月总费用（元）', y='信用分', color='b', ax=ax[0])
sns.scatterplot(data=df_data, x='用户近6个月平均消费值（元）', y='信用分', color='r', ax=ax[1])
plt.show()


f, [ax0, ax1, ax2, ax3] = plt.subplots(1, 4, figsize=(20, 6))
sns.scatterplot(data=df_data, x='当月网购类应用使用次数', y='信用分', hue='是否经常逛商场的人', ax=ax0)
sns.scatterplot(data=df_data, x='当月物流快递类应用使用次数', y='信用分', hue='是否经常逛商场的人', ax=ax1)
sns.scatterplot(data=df_data, x='当月金融理财类应用使用总次数', y='信用分', hue='是否经常逛商场的人', ax=ax2)
sns.scatterplot(data=df_data, x='当月视频播放类应用使用次数', y='信用分', hue='是否经常逛商场的人', ax=ax3)
plt.show()

f, [ax0, ax1, ax2, ax3] = plt.subplots(1, 4, figsize=(20, 6))
sns.scatterplot(data=df_data, x='当月飞机类应用使用次数', y='信用分', hue='是否经常逛商场的人', ax=ax0)
sns.scatterplot(data=df_data, x='当月火车类应用使用次数', y='信用分', hue='是否经常逛商场的人', ax=ax1)
sns.scatterplot(data=df_data, x='当月旅游资讯类应用使用次数', y='信用分', hue='是否经常逛商场的人', ax=ax2)
sns.scatterplot(data=df_data, x='用户网龄（月）', y='信用分', hue='是否经常逛商场的人', ax=ax3)
plt.show()


''' Mining and extracting the information of data hiding '''
print("------------ Mining and extracting the information of data hiding ------------")
df_data['缴费方式'] = 0
df_data.loc[(df_data['缴费用户最近一次缴费金额（元）'] != 0) & (df_data['缴费用户最近一次缴费金额（元）'] % 10 == 0), '缴费方式'] = 1
df_data.loc[(df_data['缴费用户最近一次缴费金额（元）'] != 0) & (df_data['缴费用户最近一次缴费金额（元）'] % 10 > 0), '缴费方式'] = 2
f, ax = plt.subplots(figsize=(20, 6))
sns.boxplot(data=df_data, x='缴费方式', y='信用分', ax=ax)
plt.show()



df_data['信用资格'] = df_data['用户网龄（月）'].apply(lambda x: 1 if x > 12 else 0)
f, ax = plt.subplots(figsize=(10, 6))
sns.boxenplot(data=df_data, x='信用资格', y='信用分', ax=ax)
plt.show()


df_data['敏度占比'] = df_data['用户话费敏感度'].map({1:1, 2:3, 3:3, 4:4, 5:8})
f, ax = plt.subplots(1, 2, figsize=(20, 6))
sns.boxenplot(data=df_data, x='敏度占比', y='信用分', ax=ax[0])
sns.boxenplot(data=df_data, x='用户话费敏感度', y='信用分', ax=ax[1])
plt.show()

