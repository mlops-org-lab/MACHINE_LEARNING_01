import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class preparation:

    def __init__(self, input_data_file, output_data_file):
        self.input_data_file = input_data_file
        self.output_data_file = output_data_file

    def execution(self):
        """ load data """
        file_path = './data/train_dataset.csv'
        print('---------- load data from {} ---------------'.format(file_path))
        df_data = pd.read_csv(file_path)


        """ Primary exploration project """
        print('---------- Primary exploration project ---------------')
        df_data.drop(df_data[df_data['当月通话交往圈人数'] > 1750].index, inplace=True)
        df_data.reset_index(drop=True, inplace=True)
        """ 0替换np.nan，通过线下验证发现数据实际情况缺失值数量大于0值数量，np.nan能更好的还原数据真实性 """
        na_list = ['用户年龄', '缴费用户最近一次缴费金额（元）', '用户近6个月平均消费值（元）','用户账单当月总费用（元）']
        for na_fea in na_list:
            df_data[na_fea].replace(0, np.nan, inplace=True)
        """ 话费敏感度0替换，通过线下验证发现替换为中位数能比np.nan更好的还原数据真实性 """
        df_data['用户话费敏感度'].replace(0, df_data['用户话费敏感度'].mode()[0], inplace=True)


        """ Intermediate exploration engineering """
        print('---------- Intermediate exploration engineering ---------------')
        """ x / (y + 1) 避免无穷值Inf，采用高斯平滑 + 1 """
        df_data['话费稳定'] = df_data['用户账单当月总费用（元）'] / (df_data['用户当月账户余额（元）'] + 1)
        df_data['相比稳定'] = df_data['用户账单当月总费用（元）'] / (df_data['用户近6个月平均消费值（元）'] + 1)
        df_data['缴费稳定'] = df_data['缴费用户最近一次缴费金额（元）'] / (df_data['用户近6个月平均消费值（元）'] + 1)
        df_data['当月是否去过豪华商场'] = (df_data['当月是否逛过福州仓山万达'] + df_data['当月是否到过福州山姆会员店']).map(lambda x: 1 if x > 0 else 0)
        df_data['应用总使用次数'] = df_data['当月网购类应用使用次数'] + df_data['当月物流快递类应用使用次数'] + df_data['当月金融理财类应用使用总次数'] + df_data['当月视频播放类应用使用次数'] + df_data['当月飞机类应用使用次数'] + df_data['当月火车类应用使用次数'] + df_data['当月旅游资讯类应用使用次数']

        """ Advanced exploration engineering """
        print('---------- Advanced exploration engineering ---------------')
        df_data['缴费方式'] = 0
        df_data.loc[(df_data['缴费用户最近一次缴费金额（元）'] != 0) & (df_data['缴费用户最近一次缴费金额（元）'] % 10 == 0), '缴费方式'] = 1
        df_data.loc[(df_data['缴费用户最近一次缴费金额（元）'] != 0) & (df_data['缴费用户最近一次缴费金额（元）'] % 10 > 0), '缴费方式'] = 2
        df_data['信用资格'] = df_data['用户网龄（月）'].apply(lambda x: 1 if x > 12 else 0)
        df_data['敏度占比'] = df_data['用户话费敏感度'].map({1:1, 2:3, 3:3, 4:4, 5:8})


        """ output preparation file"""
        print('---------- output preparation file in {}---------------'.format(self.output_data_file))
        df_data.to_csv(self.output_data_file,index=False)


if __name__ == '__main__':
    pre = preparation('data/train_dataset.csv', 'pre_dataset.csv')
    pre.execution()