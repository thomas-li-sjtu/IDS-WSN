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
def noise_fault(origin, attributes, noise_times=3, num=100):
    """
    噪音故障注入
    :param origin: 原始数据
    :param noise_times: 噪音的标准差是原始数据标准差的多少倍
    :param num: 噪音数目
    :return: 加入噪音的数据
    """
    temperature, humidity, light, voltage = zip(*origin)
    tmp_dict = {1: list(temperature), 2: list(humidity), 3: list(light), 4: list(voltage), 'labels': [1]*len(temperature)}
    for i in attributes:
        std_noise = np.std(np.array(tmp_dict[i]), ddof=1) * noise_times
        nosie = np.random.normal(loc=0.0, scale=std_noise, size=num)
        ran_index = sorted(random.sample(range(len(tmp_dict[i])), 25))  # 噪音开始的位置

        # 混入噪音
        counter = 0
        for index in ran_index:
            for j in range(index, index+int(num/25)):  # 每次噪音持续时间为 num/25 次采样
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
    :param nums:
    :return:
    """
    temperature, humidity, light, voltage = zip(*origin)
    tmp_dict = {1: list(temperature), 2: list(humidity), 3: list(light), 4: list(voltage), 'labels': [1]*len(temperature)}
    for i in attributes:
        ran_index = sorted(random.sample(range(len(tmp_dict[i])), nums))  # 噪音开始的位置

        # 混入噪音
        counter = 0
        for index in ran_index:
            print(tmp_dict[i][index])
            tmp_dict[i][index] += tmp_dict[i][index]*f
            print(tmp_dict[i][index])
            counter += 1
            tmp_dict['labels'][index] = 0

    return zip(tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict['labels'])


def fixed_fault(origin, attributes, error_data=-0.1, num=100):
    """
    固定故障注入
    :param origin:
    :param attributes:
    :param num:
    :param error_data:
    :return:
    """
    temperature, humidity, light, voltage = zip(*origin)
    tmp_dict = {1: list(temperature), 2: list(humidity), 3: list(light), 4: list(voltage), 'labels': [1]*len(temperature)}
    for i in attributes:
        ran_index = sorted(random.sample(range(len(tmp_dict[i])), 25))  # 噪音开始的位置

        # 混入噪音
        counter = 0
        for index in ran_index:
            for j in range(index, index+int(num/25)):  # 每次噪音持续时间为 num/25 次采样
                tmp_dict[i][j] = error_data
                counter += 1
                tmp_dict['labels'][j] = 0

    return zip(tmp_dict[1], tmp_dict[2], tmp_dict[3], tmp_dict[4], tmp_dict['labels'])


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
                tmp_normal_data['temperature'].append(sum(tmp_normal_data['temperature'][-5:])/5)
                tmp_normal_data['humidity'].append(sum(tmp_normal_data['humidity'][-5:])/5)
                tmp_normal_data['light'].append(sum(tmp_normal_data['light'][-5:])/5)
                tmp_normal_data['voltage'].append(sum(tmp_normal_data['voltage'][-5:])/5)
                continue
            tmp_normal_data['temperature'].append(data[0])
            tmp_normal_data['humidity'].append(data[1])
            tmp_normal_data['light'].append(data[2])
            tmp_normal_data['voltage'].append(data[3])
            if tmp_normal_data['humidity'][-1] < 0:  # 特殊情况——错误修正
                tmp_normal_data['humidity'][-1] = sum(tmp_normal_data['humidity'][-5:-1])/4
            if tmp_normal_data['temperature'][-1] < 16:  # 特殊情况——错误修正
                tmp_normal_data['temperature'][-1] = sum(tmp_normal_data['temperature'][-5:-1])/4
            if tmp_normal_data['voltage'][-1] < 2:  # 特殊情况——错误修正
                tmp_normal_data['voltage'][-1] = sum(tmp_normal_data['voltage'][-5:-1])/4
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
        print(len(normal_t))
        normal_data[key] = list(zip(normal_t, normal_h, normal_l, normal_v))
    # 共 6823 个数据，700 条填补

    # # 加入噪音故障
    normal_data[2] = noise_fault(normal_data[2], attributes=[1, 2])
    normal_data[3] = noise_fault(normal_data[3], attributes=[3, 4])
    # # 加入其他故障
    normal_data[35] = short_term_fault(normal_data[35], attributes=[1])
    normal_data[33] = fixed_fault(normal_data[33], attributes=[2])


if __name__ == '__main__':
    run()
