import pandas as pd
import numpy as np
import collections
import random
import pickle


def load(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def dump(file_path, data):
    file = open(file_path, "wb")
    pickle.dump(data, file)


def normalize(data):
    """
    归一化
    :param data: 输入数据
    :return:
    """
    data_max, data_min = data.max(), data.min()
    data = (data - data_min) / (data_max - data_min)
    return data


# 三种故障注入方式
def noise_fault(origin, attributes, noise_times=3, num=500, duration=5):
    """
    噪音故障注入
    :param origin: 原始数据
    :param attributes: 要混入噪音的属性
    :param noise_times: 噪音的标准差是原始数据标准差的多少倍
    :param num: 噪音数目
    :param duration: 噪音持续时间
    :return: 加入噪音的数据
    """
    temperature, humidity, light, voltage = zip(*origin)
    tmp_dict = {1: list(temperature), 2: list(humidity), 3: list(light), 4: list(voltage),
                'labels': [1] * len(temperature)}
    ran_index = sorted(random.sample(range(len(tmp_dict[1]) - duration), int(num / duration)))  # 噪音开始的位置
    for i in attributes:
        std_noise = np.std(np.array(tmp_dict[i]), ddof=1) * noise_times
        nosie = np.random.normal(loc=0.0, scale=std_noise, size=num)

        # 混入噪音
        counter = 0
        for index in ran_index:
            for j in range(index, index + duration):  # 每次噪音持续时间为 duration 次采样
                tmp_dict[i][j] += nosie[counter]
                counter += 1
                tmp_dict['labels'][j] = 0

    return zip(tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict['labels'])


def short_term_fault(origin, attributes, f=1.5, nums=500):
    """
    短时故障注入
    :param origin: 原始数据
    :param attributes:
    :param f: 倍数关系
    :param nums: 混入故障的数据数目
    :return:
    """
    temperature, humidity, light, voltage = zip(*origin)
    tmp_dict = {1: list(temperature), 2: list(humidity), 3: list(light), 4: list(voltage),
                'labels': [1] * len(temperature)}
    ran_index = sorted(random.sample(range(len(tmp_dict[1])), nums))  # 噪音开始的位置
    for i in attributes:
        # 混入噪音
        counter = 0
        for index in ran_index:
            # print(tmp_dict[i][index])
            tmp_dict[i][index] += tmp_dict[i][index] * f
            # print(tmp_dict[i][index])
            counter += 1
            tmp_dict['labels'][index] = 0

    # for i in range(len(tmp_dict['labels'])):  # 查看标签是否正确
    #     if tmp_dict['labels'][i] == 0:
    #         print(tmp_dict[1][i])

    return zip(tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict['labels'])


def fixed_fault(origin, attributes, error_data=-0.1, num=500, duration=5):
    """
    固定故障注入
    :param origin: 原始数据
    :param attributes:
    :param error_data: 固定故障的值
    :param num: 混入故障的数据数目
    :param duration: 故障的持续时间
    :return:
    """
    temperature, humidity, light, voltage = zip(*origin)
    tmp_dict = {1: list(temperature), 2: list(humidity), 3: list(light), 4: list(voltage),
                'labels': [1] * len(temperature)}
    ran_index = sorted(random.sample(range(len(tmp_dict[1]) - duration), int(num / duration)))  # 故障开始的位置
    for i in attributes:
        # 混入噪音
        counter = 0
        for index in ran_index:
            for j in range(index, index + duration):  # 每次故障持续时间为 duration 次采样
                tmp_dict[i][j] = error_data
                counter += 1
                tmp_dict['labels'][j] = 0

    return zip(tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict['labels'])


def save_data(data, inject_node, attributes, writer, sheet_name):
    """
    保存数据到excel
    :param data:
    :param inject_node: 注入故障的节点号
    :param attributes: 混入故障的节点属性
    :param writer: ExcelWriter
    :param sheet_name: 对应表名
    :return:
    """
    tmp_t, tmp_h, tmp_l, tmp_v, labels = zip(*list(data[inject_node]))
    data[inject_node], labels = list(zip(tmp_t, tmp_h, tmp_l, tmp_v)), \
                                np.reshape(np.array(list(labels)).transpose(), (len(labels), -1))
    # print(data[1][0])
    # print(data[2][0])
    # print(data[3][0])
    # print(data[33][0])
    # print(data[35][0])
    # print(labels.shape)

    # 合并 5 个节点的测量结果，形成 20 个特征
    data = np.array(list(zip(data[1], data[2], data[3], data[33], data[35])))
    # print(data.shape)  # (6823, 5, 4)
    data = np.reshape(data, (len(data), -1))
    data = np.hstack((data, labels))
    # print(data.shape)  # (6823, 21)
    # for i in data[0]:
    #     print(i)

    train, test = data[:int(len(data) * 0.7)], data[int(len(data) * 0.7):]
    train_frame = pd.DataFrame(train, index=None,
                               columns=['1_t', '1_h', '1_l', '1_v',
                                        '2_t', '2_h', '2_l', '2_v',
                                        '3_t', '3_h', '3_l', '3_v',
                                        '33_t', '33_h', '33_l', '33_v',
                                        '35_t', '35_h', '35_l', '35_v',
                                        'label'])
    test_frame = pd.DataFrame(test, index=None,
                              columns=['1_t', '1_h', '1_l', '1_v',
                                       '2_t', '2_h', '2_l', '2_v',
                                       '3_t', '3_h', '3_l', '3_v',
                                       '33_t', '33_h', '33_l', '33_v',
                                       '35_t', '35_h', '35_l', '35_v',
                                       'label'])
    train_frame.to_excel(writer, index=None, sheet_name=sheet_name+"_train")
    test_frame.to_excel(writer, index=None, sheet_name=sheet_name+"_test")


def run():
    aver_data = load("IBRL_average.pickle")

    normal_data = {}
    for key, value in aver_data.items():
        counter = 0  # 需要填补的数目
        tmp_normal_data = collections.defaultdict(lambda: [])
        for time, data in value:
            if time > '2004-03-18':
                continue
            if not data:  # 平滑填补
                counter += 1
                tmp_normal_data['temperature'].append(sum(tmp_normal_data['temperature'][-5:]) / 5)
                tmp_normal_data['humidity'].append(sum(tmp_normal_data['humidity'][-5:]) / 5)
                tmp_normal_data['light'].append(sum(tmp_normal_data['light'][-5:]) / 5)
                tmp_normal_data['voltage'].append(sum(tmp_normal_data['voltage'][-5:]) / 5)
                continue
            tmp_normal_data['temperature'].append(data[0])
            tmp_normal_data['humidity'].append(data[1])
            tmp_normal_data['light'].append(data[2])
            tmp_normal_data['voltage'].append(data[3])
            if tmp_normal_data['humidity'][-1] < 0:  # 特殊情况——错误修正
                tmp_normal_data['humidity'][-1] = sum(tmp_normal_data['humidity'][-5:-1]) / 4
            if tmp_normal_data['temperature'][-1] < 16:  # 特殊情况——错误修正
                tmp_normal_data['temperature'][-1] = sum(tmp_normal_data['temperature'][-5:-1]) / 4
            if tmp_normal_data['voltage'][-1] < 2:  # 特殊情况——错误修正
                tmp_normal_data['voltage'][-1] = sum(tmp_normal_data['voltage'][-5:-1]) / 4
        # print(max(tmp_normal_data['temperature']), min(tmp_normal_data['temperature']))
        # print(max(tmp_normal_data['humidity']), min(tmp_normal_data['humidity']))
        # print(max(tmp_normal_data['light']), min(tmp_normal_data['light']))
        # print(max(tmp_normal_data['voltage']), min(tmp_normal_data['voltage']))
        # 归一化
        normal_t, normal_h, normal_l, normal_v = normalize(np.array(tmp_normal_data['temperature'])), \
                                                 normalize(np.array(tmp_normal_data['humidity'])), \
                                                 normalize(np.array(tmp_normal_data['light'])), \
                                                 normalize(np.array(tmp_normal_data['voltage']))
        print(counter)
        normal_data[key] = list(zip(normal_t, normal_h, normal_l, normal_v))
    print(len(normal_data[1]))
    # 共 6823 个数据，700 条填补

    writer = pd.ExcelWriter("IBRL_data.xlsx")
    # 加入噪音故障
    tmp = normal_data[3]
    normal_data[3] = noise_fault(normal_data[3], attributes=[1, 2, 3, 4])
    save_data(normal_data, inject_node=3, attributes=[1, 2, 3, 4], writer=writer, sheet_name="noise_inject")
    # # 加入短时故障
    normal_data[3] = tmp
    tmp = normal_data[35]
    normal_data[35] = short_term_fault(normal_data[35], attributes=[1, 2, 3, 4])
    save_data(normal_data, inject_node=35, attributes=[1, 2, 3, 4], writer=writer, sheet_name="short_term_inject")
    # # 加入固定故障
    normal_data[35] = tmp
    tmp = normal_data[33]
    normal_data[33] = fixed_fault(normal_data[33], attributes=[1, 2, 3, 4])
    save_data(normal_data, inject_node=33, attributes=[1, 2, 3, 4], writer=writer, sheet_name="fixed_inject")

    writer.save()


if __name__ == '__main__':
    run()
