from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd

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

# PCA
n_comp = 10
pca = PCA(n_comp)
pca.fit(dt_training)  # finding pca axes
Xtrain_proj = pca.transform(dt_training)  # projecting training data onto pca axes
Xtest_proj = pca.transform(dt_testing)  # projecting test data onto pca axes

# The SVM Section
clf = svm.SVC(gamma=0.001, C=100.)  # instantiating an SVM classifier
clf.fit(Xtrain_proj, dt_target_training)  # apply SVM to training data and draw boundaries.
dataset_predict_y = clf.predict(Xtest_proj)  # make a prediction for the test data point.

correct_predicts = sum([1 if dataset_predict_y[i] == dt_target_testing[i] else 0 for i in range(len(dataset_predict_y))])
accuracy = 100 * correct_predicts / len(dt_testing)
print('pca+svm,correct prediction num:{},accuracy:{:.2f}%'
      .format(correct_predicts, accuracy))

# 93.43% (97.5%)
