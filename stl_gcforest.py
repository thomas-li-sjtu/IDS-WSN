import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import torch.utils.data as Data
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--training-data', '-i',
                        default='data/train.txt',
                        dest='training_data',
                        help='Path to training data file (one password per line) (default: data/train.txt)')

    parser.add_argument('--output-dir', '-o',
                        required=True,
                        dest='output_dir',
                        help='Output directory. If directory doesn\'t exist it will be created.')

    parser.add_argument('--save-every', '-s',
                        type=int,
                        default=5000,
                        dest='save_every',
                        help='Save model checkpoints after this many iterations (default: 5000)')

    return parser.parse_args()


class autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=18,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = torch.nn.Linear(in_features=64, out_features=5)  # 编码的维度为

        self.rnn_2 = torch.nn.LSTM(
            input_size=5,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out_2 = torch.nn.Linear(in_features=64, out_features=18)  # out_features = input_size

    def forward(self, x):
        # 一下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output, (h_n, c_n) = self.rnn(x)
        # output_in_last_timestep = output[:,-1,:] # 也是可以的
        output_in_last_timestep = h_n[-1, :, :]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        encode = self.out(output_in_last_timestep)

        output1, (h_n1, c_n1) = self.rnn_2(encode.view(-1, 1, 5))
        # output_in_last_timestep = output[:,-1,:] # 也是可以的
        output_in_last_timestep1 = h_n1[-1, :, :]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        decode = self.out_2(output_in_last_timestep1)
        return encode, decode


# preprocessing (数据的标准化)
def Normalization(data):

    for i in range(len(data[0])):
        # 查找最大值
        max = np.amax(data[:, i])
        min = np.amin(data[:, i])

        # 使用 min-max normalization 标准化
        for j in range(len(data)):
            data[j][i] = (data[j][i] - min) / (max - min)

    return data


def load_data():
    # 获取数据
    file_name = "Dataset/WSN-DS/binary_wsn-ds_0.1.xlsx"
    data_training = pd.read_excel(file_name, sheet_name='train data', usecols=range(0, 18), engine='openpyxl')
    data_testing = pd.read_excel(file_name, sheet_name='test data', usecols=range(0, 18), engine='openpyxl')
    train_target = pd.read_excel(file_name, sheet_name='train data', usecols=[18], engine='openpyxl')
    test_target = pd.read_excel(file_name, sheet_name='test data', usecols=[18], engine='openpyxl')
    print("Data loaded")
    # 变化矩阵形式
    x = Normalization(data_training.to_numpy())
    y = train_target.to_numpy()
    dt_testing = Normalization(data_testing.to_numpy())
    dt_target_testing = test_target.to_numpy()

    x = torch.from_numpy(x).float()  # x(torch tensor)
    y = torch.from_numpy(y).float()  # y(torch tensor)


def run(args):
    # 超参数
    batch_size = 20
    epoches = 10

    # 将输入和输出封装进Data.TensorDataset()类对象
    torch_dataset = Data.TensorDataset(x, y)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多进程（multiprocess）来读数据
    )

    net = autoencoder()
    # 训练
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_F = torch.nn.MSELoss()
    for epoch in range(epoches):  # 数据集只迭代一次
        for step, (batch_x, batch_y) in enumerate(loader):
            _, decode = net(batch_x.view(-1, 1, 18))
            loss = loss_F(decode, batch_x)  # 计算loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

    encode, _ = net(x.view(-1, 1, 18))
    encode = encode.squeeze(1).detach().numpy()
    print(encode.shape)
    print(y)
    y = y.detach().numpy()

    # 根据encoder结果分类
    clf = RandomForestClassifier(n_estimators=5, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, encode, y, cv=6, scoring='accuracy')
    print(scores)
    # model = clf.fit(dt_training, dt_target_training)
    # dataset_predict_y = clf.predict(dt_testing)  # make a prediction for the test data point.
    # correct_predicts = sum([1 if dataset_predict_y[i] == dt_target_testing[i] else 0 for i in range(len(dataset_predict_y))])
    # accuracy = 100 * correct_predicts / len(dt_testing)
    # print('random forest,correct prediction num:{},accuracy:{:.2f}%'
    #       .format(correct_predicts, accuracy))


if __name__ == '__main__':
    args = parse_args()
    run(args)