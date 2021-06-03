import numpy as np
import torch
import torch.nn.functional as Fun
import torch.utils.data as Data
import pandas as pd


class Mlp(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Mlp, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐藏层网络
        self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络

    def forward(self, x):
        x = Fun.relu(self.hidden(x))  # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
        x = self.out(x)  # 输出层不用激活函数
        return x


def Normalization(data):
    for i in range(len(data[0])):
        # 查找最大值
        max_num = np.amax(data[:, i])
        min_num = np.amin(data[:, i])

        # 使用 min-max normalization 标准化
        for j in range(len(data)):
            data[j][i] = (data[j][i] - min_num) / (max_num - min_num)

    return data


def load_data(file_name):
    if 'wsn-ds' in file_name:
        data_training = pd.read_excel(file_name, sheet_name='train data', usecols=range(0, 18), engine='openpyxl')
        data_testing = pd.read_excel(file_name, sheet_name='test data', usecols=range(0, 18), engine='openpyxl')
        train_target = pd.read_excel(file_name, sheet_name='train data', usecols=[18], engine='openpyxl')
        test_target = pd.read_excel(file_name, sheet_name='test data', usecols=[18], engine='openpyxl')
    else:
        inject = 'noise_inject'
        # inject = 'short_term_inject'
        # inject = 'fixed_inject'
        data_training = pd.read_excel(file_name, sheet_name=inject + '_train', usecols=range(0, 20), engine='openpyxl')
        data_testing = pd.read_excel(file_name, sheet_name=inject + '_test', usecols=range(0, 20), engine='openpyxl')
        train_target = pd.read_excel(file_name, sheet_name=inject + '_train', usecols=[20], engine='openpyxl')
        test_target = pd.read_excel(file_name, sheet_name=inject + '_test', usecols=[20], engine='openpyxl')
    print("Data loaded")
    # 变化矩阵形式
    dt_training = Normalization(data_training.to_numpy())  # 正则化
    dt_testing = Normalization(data_testing.to_numpy())  # 正则化
    dt_target_training = train_target.to_numpy()
    dt_target_testing = test_target.to_numpy()

    dt_training = torch.from_numpy(dt_training).float()  # x(torch tensor)
    dt_target_training = torch.from_numpy(dt_target_training).float()  # y(torch tensor)
    dt_testing = torch.from_numpy(dt_testing).float()
    dt_target_testing = torch.from_numpy(dt_target_testing).float()

    return dt_training, dt_target_training, dt_testing, dt_target_testing


# 超参数
batch_size = 64
epochs = 2
n_feature = 18
n_hidden = 20
n_output = 1

file_name = '../Dataset/WSN-DS/binary_wsn-ds_0.1.xlsx'
x, y, dt_testing, dt_target_testing = load_data(file_name)
# 将输入和输出封装进Data.TensorDataset()类对象
torch_dataset = Data.TensorDataset(x, y)
# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
    batch_size=batch_size,  # 每块的大小
    shuffle=False,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多进程（multiprocess）来读数据
    # drop_last=True,  # 最后一组删除
)

net = Mlp(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)  # n_feature:输入的特征维度,n_hiddenb:神经元个数,n_output:输出的类别个数
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 优化器选用随机梯度下降方式
loss_func = torch.nn.BCEWithLogitsLoss()  # 对于二分类一般采用的交叉熵损失函数,
for epoch in range(epochs):
    for step, (batch_x, batch_y) in enumerate(loader):
        out = net(batch_x.view(-1, n_feature))
        loss = loss_func(out, batch_y)  # 计算lossy
        optimizer.zero_grad()  # # 梯度清零
        loss.backward()  # 前馈操作
        optimizer.step()  # 使用梯度优化器
    print('Epoch: ', epoch)
    # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

sigmoid = torch.nn.Sigmoid()
out = sigmoid(net(dt_testing.view(-1, n_feature)))  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
print(out.data)
pred_y = np.around(out.data.numpy())
print(pred_y)
target_y = dt_target_testing.data.numpy()

acc = sum([1 if target_y[i] == pred_y[i] else 0 for i in range(len(target_y))])/target_y.size
print("acc :", acc)

# WSN_0.1  正则 97.6  非正则 91
# WSN
