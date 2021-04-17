import pandas as pd

""" load data """
file_path = './data/train_dataset.csv'
print('---------- load data from {} ---------------'.format(file_path))
train_data = pd.read_csv(file_path)

""" data's attribution """
print("---------- data's attribution ---------------")
print("The total number of dataset：", train_data.shape[0])

output_path = './data/train_dataset.csv'
print('---------- download data in {} ---------------'.format(output_path))
train_data.to_csv(output_path, index=False)
