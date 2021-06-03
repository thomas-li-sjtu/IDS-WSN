from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 获取数据
file_name = "../Dataset/WSN-DS/binary_wsn-ds_0.1.xlsx"
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

# 随机森林
clf = RandomForestClassifier(n_estimators=5, max_depth=None, min_samples_split=2, random_state=0)
model = clf.fit(dt_training, dt_target_training)
dataset_predict_y = clf.predict(dt_testing)  # make a prediction for the test data point.

correct_predicts = sum([1 if dataset_predict_y[i] == dt_target_testing[i] else 0 for i in range(len(dataset_predict_y))])
accuracy = 100 * correct_predicts / len(dt_testing)
print('random forest,correct prediction num:{},accuracy:{:.2f}%'
      .format(correct_predicts, accuracy))

# 99+
