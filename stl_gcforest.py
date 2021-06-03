import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import torch.utils.data as Data
import argparse
from deepforest import CascadeForestClassifier


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', '-i',
                        default='Dataset/WSN-DS/binary_wsn-ds_0.1.xlsx',
                        # default='Dataset/IBRL/IBRL_data.xlsx',
                        dest='file_name',
                        help='Path to training data file (default: Dataset/WSN-DS/binary_wsn-ds_0.1.xlsx)')

    parser.add_argument('--batch_size', '-b',
                        type=int,
                        default=32,
                        dest='batch_size',
                        help='Batch size (default: 32).')

    parser.add_argument('--epochs', '-e',
                        type=int,
                        default=20,
                        dest='epochs',
                        help='Training epochs (default: 20)')

    parser.add_argument('--dimensions', '-d',
                        type=int,
                        default=20,
                        dest='dimensions',
                        help='Dimension reduction result (default: 5)')

    parser.add_argument('--hidden_size', '-hs',
                        type=int,
                        default=64,
                        dest='hidden_size',
                        help='Hidden cells of LSTM (default: 64)')

    parser.add_argument('--folds', '-f',
                        type=int,
                        default=10,
                        dest='folds',
                        help='Number of folds (default: 10)')

    parser.add_argument('--characteristic', '-c',
                        type=int,
                        default=18,
                        dest='characteristic',
                        help='Number of characteristic (default: 10)')

    return parser.parse_args()


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_features):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_features = out_features

        self.rnn = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.out = torch.nn.Linear(in_features=64, out_features=self.out_features)  # 编码的维度为
        self.rnn_2 = torch.nn.LSTM(
            input_size=self.out_features,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.out_2 = torch.nn.Linear(in_features=64, out_features=self.input_size)  # out_features = input_size

    def forward(self, x):
        # 以下关于shape的注释只针对单项
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size] # 虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output, (h_n, c_n) = self.rnn(x)
        # output_in_last_timestep = output[:,-1,:] # 也是可以的
        output_in_last_timestep = h_n[-1, :, :]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        encode = self.out(output_in_last_timestep)

        output1, (h_n1, c_n1) = self.rnn_2(encode.view(-1, 1, self.out_features))
        # output_in_last_timestep = output[:,-1,:] # 也是可以的
        output_in_last_timestep1 = h_n1[-1, :, :]
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        decode = self.out_2(output_in_last_timestep1)
        return encode, decode


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


def random_forest(encode_train, y, encode_test, dt_target_testing, args):
    rfc = RandomForestClassifier(n_estimators=5, max_depth=None, min_samples_split=2, random_state=0)
    scores = cross_val_score(rfc, encode_train, y, cv=args.folds, scoring='accuracy')  # k折交叉验证
    print(scores)
    scores = cross_val_score(rfc, encode_test, dt_target_testing, cv=args.folds, scoring='accuracy')
    print('random forest: {}'.format(scores))
    # [0.97784895 0.98211903 0.98425407 0.98532159 0.99572992 0.99839872 0.99252536 0.97463962 0.97650828 0.98585158]
    #
    # model = rfc.fit(encode_train, y)
    # dataset_predict_y = rfc.predict(encode_test)
    # correct_predicts = sum([1 if dataset_predict_y[i] == dt_target_testing[i] else 0
    #                         for i in range(len(dataset_predict_y))])
    # accuracy = 100 * correct_predicts / len(encode_test)
    # print('random forest,correct prediction num:{},accuracy:{:.2f}%'
    #       .format(correct_predicts, accuracy))


def deep_forest(encode_train, y, encode_test, dt_target_testing):
    from sklearn.metrics import accuracy_score

    model = CascadeForestClassifier(n_estimators=5, random_state=0)
    model.fit(encode_train, y)
    y_pred = model.predict(encode_test)
    acc = accuracy_score(dt_target_testing, y_pred) * 100
    print("\ndeep forest Testing Accuracy: {:.3f} %".format(acc))


def run(args):
    # 超参数
    batch_size = args.batch_size
    epochs = args.epochs

    x, y, dt_testing, dt_target_testing = load_data(args.file_name)
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

    net = AutoEncoder(input_size=args.characteristic, hidden_size=args.hidden_size, out_features=args.dimensions)
    # 训练
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_F = torch.nn.MSELoss()
    for epoch in range(epochs):  # 数据集只迭代一次
        for step, (batch_x, batch_y) in enumerate(loader):
            _, decode = net(batch_x.view(-1, 1, args.characteristic))
            loss = loss_F(decode, batch_x)  # 计算lossy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: ', epoch)
        # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

    # 降维编码
    encode_train, _ = net(x.view(-1, 1, args.characteristic))
    encode_train = encode_train.squeeze(1).detach().numpy()
    y = y.detach().numpy()
    encode_test, _ = net(dt_testing.view(-1, 1, args.characteristic))
    encode_test = encode_test.squeeze(1).detach().numpy()
    dt_target_testing = dt_target_testing.detach().numpy()

    # 根据encode结果分类
    # random forest
    random_forest(encode_train, y, encode_test, dt_target_testing, args)
    # deep forest
    # deep_forest(encode_train, y, encode_test, dt_target_testing)


if __name__ == '__main__':
    args = parse_args()
    if 'wsn' in args.file_name:
        args.characteristic = 18
    else:
        args.characteristic = 20

    run(args)
