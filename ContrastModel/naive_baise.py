# 使用GaussianNB分类器构建朴素贝叶斯模型
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# 获取数据
file_name = "../Dataset/WSN-DS/binary_wsn-ds.xlsx"
data_training = pd.read_excel(file_name, sheet_name='train data', usecols=range(0, 18), engine='openpyxl')
data_testing = pd.read_excel(file_name, sheet_name='test data', usecols=range(0, 18), engine='openpyxl')
train_target = pd.read_excel(file_name, sheet_name='train data', usecols=[18], engine='openpyxl')
test_target = pd.read_excel(file_name, sheet_name='test data', usecols=[18], engine='openpyxl')
print("Data loaded")

# 变化矩阵形式
dt_training = data_training.to_numpy()
dt_testing = data_testing.to_numpy()
dt_target_training = train_target.to_numpy()
dt_target_testing = test_target.to_numpy()

gaussianNB = GaussianNB()
gaussianNB.fit(dt_training, dt_target_training)

# 评估本模型在整个数据集上的表现
dataset_predict_y = gaussianNB.predict(dt_testing)
print(dataset_predict_y)
correct_predicts = sum([1 if dataset_predict_y[i] == dt_target_testing[i] else 0 for i in range(len(dataset_predict_y))])
accuracy = 100 * correct_predicts / len(dt_testing)
print('GaussianNB,correct prediction num:{},accuracy:{:.2f}%'
      .format(correct_predicts, accuracy))

# 87.32%
