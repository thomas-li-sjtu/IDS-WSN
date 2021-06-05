import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd

torch.manual_seed(1)


class LSTM_RNN(nn.Module):
    """搭建LSTM神经网络"""
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,   # rnn 隐藏单元数
                            num_layers=num_layers,     # rnn 层数
                            batch_first=True,
                            # If ``True``, then the input and output tensors are provided as
                            # (batch, seq, feature). Default: False
                            )
        self.output_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lstm_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        lstm_out, (h_n, h_c) = self.lstm(x, None)   # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        output = self.output_layer(lstm_out[:, -1, :])   # 选择最后时刻lstm的输出
        return output


# preprocessing (数据的标准化)
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


# 设置超参数
epoches = 2
batch_size = 64
time_step = 1
input_size = 18
learning_rate = 0.01
hidden_size = 64
num_layers = 2


def main():
    x, y, dt_testing, dt_target_testing = load_data(file_name='../Dataset/WSN-DS/binary_wsn-ds.xlsx')
    # 将输入和输出封装进Data.TensorDataset()类对象
    torch_dataset = Data.TensorDataset(x, y)
    # 把 dataset 放入 DataLoader
    train_loader = Data.DataLoader(
        dataset=torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
        # drop_last=True,  # 最后一组删除
    )

    lstm = LSTM_RNN()
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = lstm(batch_x.view(-1, 1, 18))
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                sigmoid = torch.nn.Sigmoid()
                out = sigmoid(lstm(dt_testing.view(-1, 1, 18)))  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
                pred_y = np.around(out.data.numpy())
                accuracy = ((pred_y == dt_target_testing.data.numpy()).astype(int).sum()) / float(dt_target_testing.nelement())
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

    sigmoid = torch.nn.Sigmoid()
    out = sigmoid(lstm(dt_testing.view(-1, 1, 18)))  # out是一个计算矩阵，可以用Fun.softmax(out)转化为概率矩阵
    pred_y = np.around(out.data.numpy())
    accuracy = ((pred_y == dt_target_testing.data.numpy()).astype(int).sum()) / float(dt_target_testing.nelement())
    print('| test accuracy: %.4f' % accuracy)


if __name__ == "__main__":
    main()

# WSN-DS_0.1  正则：0.9801  非正则：
# WSN-DS      正则：98.09%  非正则：
