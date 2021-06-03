from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


def Normalization(data):
    for i in range(len(data[0])):
        # 查找最大值
        max_num = np.amax(data[:, i])
        min_num = np.amin(data[:, i])

        # 使用 min-max normalization 标准化
        for j in range(len(data)):
            data[j][i] = (data[j][i] - min_num) / (max_num - min_num)

    return data


file_name = "../Dataset/WSN-DS/binary_wsn-ds_0.1.xlsx"
data_training = pd.read_excel(file_name, sheet_name='train data', usecols=range(0, 18), engine='openpyxl')
data_testing = pd.read_excel(file_name, sheet_name='test data', usecols=range(0, 18), engine='openpyxl')
train_target = pd.read_excel(file_name, sheet_name='train data', usecols=[18], engine='openpyxl')
test_target = pd.read_excel(file_name, sheet_name='test data', usecols=[18], engine='openpyxl')
print("Data loaded")
# 变化矩阵形式
dt_training = Normalization(data_training.to_numpy())  # 正则化
dt_testing = Normalization(data_testing.to_numpy())  # 正则化
dt_target_training = train_target.to_numpy()
dt_target_testing = test_target.to_numpy()

model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
model.fit(dt_training, dt_target_training)

dataset_predict_y = model.predict(dt_testing)
correct_predicts = sum([1 if dataset_predict_y[i] == dt_target_testing[i] else 0 for i in range(len(dataset_predict_y))])
accuracy = 100 * correct_predicts / len(dt_testing)
print('C5,correct prediction num:{},accuracy:{:.2f}%'
      .format(correct_predicts, accuracy))

# 98.56？
