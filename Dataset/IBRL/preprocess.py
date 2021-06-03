import pandas as pd
import collections
import datetime
import pickle


def dump(file_path, data):
    file = open(file_path, "wb")
    pickle.dump(data, file)


def split_time_ranges(from_time, to_time, win_length, intervals=0, flow=False):
    """
    切分时间窗口
    :param from_time:
    :param to_time:
    :param win_length: 窗口长度
    :param intervals: 滑动步长
    :param flow: 是否滑动
    :return:
    """
    if not flow:
        intervals = win_length
    from_time, to_time = pd.to_datetime(from_time), pd.to_datetime(to_time)
    time_range = list(pd.date_range(from_time, to_time, freq='%sS' % intervals))
    if to_time not in time_range:
        time_range.append(to_time)
    time_range = [item.strftime("%Y-%m-%d %H:%M:%S") for item in time_range]
    time_ranges = []
    for item in time_range:
        f_time = item
        t_time = (datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(seconds=win_length))
        if t_time >= to_time:
            t_time = to_time.strftime("%Y-%m-%d %H:%M:%S")
            time_ranges.append([f_time, t_time])
            break
        time_ranges.append([f_time, t_time.strftime("%Y-%m-%d %H:%M:%S")])
    return time_ranges


def split_data_by_time(data):
    """
    根据时间窗口划分传入的数据（传入的数据是一个传感器的所有数据）
    :param data: 某个传感器的所有数据
    :return: 一个二维列表，元素为tuple（timestamp，series data）
    """
    from_time = '2004-02-28 01:10:00'  # 起始时间
    to_time = '2004-03-25 01:10:00'  # 间隔时间
    win_length = 60 * 20  # 时间窗口大小，单位s
    split_time = split_time_ranges(from_time, to_time, win_length, intervals=60*4, flow=True)  # 获得时间窗口

    # 根据时间窗口切分数据，变为二维数组
    index_time = 0
    split_data, tmp_split_data = [], []
    for i in range(len(data)):
        tmp = data[i][:-1].split(' ')
        tmp_time = ' '.join([tmp[0], tmp[1].split('.')[0]])
        if tmp_time > to_time:  # 是否超过计入时间
            break
        while split_time[index_time][1] < tmp_time:
            if tmp_split_data:
                split_data.append(tuple([split_time[index_time][0], tmp_split_data]))
            else:
                split_data.append(tuple([split_time[index_time][0], []]))
            tmp_split_data = []
            index_time += 1
        # print(split_time[index_time], tmp_time)
        if split_time[index_time][1] > tmp_time > split_time[index_time][0]:
            tmp_split_data.append(data[i])
    return split_data


def calc_average(data):
    """
    输入时间窗口分割后的数据，统计时间窗口内的均值，作为此时间段内的测量值
    :param data:
    :return:
    """
    calc_data = []
    for time, series in data:
        if not series:
            calc_data.append(tuple([time, []]))
            continue
        record_temperature, record_humidity, record_light, record_voltage = 0, 0, 0, 0
        for record in series:  # 一个窗口内的和
            record_temperature += float(record.split(' ')[-4])
            record_humidity += float(record.split(' ')[-3])
            record_light += float(record.split(' ')[-2])
            record_voltage += float(record.split(' ')[-1])
        # 计算均值
        record_temperature = record_temperature/len(series)
        record_humidity = record_humidity/len(series)
        record_light = record_light/len(series)
        record_voltage = record_voltage/len(series)
        calc_data.append(tuple([time, [record_temperature, record_humidity, record_light, record_voltage]]))
    return calc_data


def run():
    id_value = collections.defaultdict(lambda: [])
    with open("data.txt", "r") as file:
        for i in file:
            tmp = i[:-1].split(" ")
            if len(tmp) == 8 and tmp[3] != '' and int(tmp[3]) <= 58:
                id_value[tmp[3]].append(i[:-1])

    # 分割时间窗口
    sort_data = {key: sorted(id_value[key]) for key, value in id_value.items()}
    data_1 = split_data_by_time(sort_data['1'])
    print(sorted(list(set([time.split(" ")[0] for time, data in data_1 if not data]))))
    data_2 = split_data_by_time(sort_data['2'])
    print()
    print(sorted(list(set([time.split(" ")[0] for time, data in data_2 if not data]))))
    data_3 = split_data_by_time(sort_data['3'])
    print()
    print(sorted(list(set([time.split(" ")[0] for time, data in data_3 if not data]))))
    data_33 = split_data_by_time(sort_data['33'])
    print()
    print(sorted(list(set([time.split(" ")[0] for time, data in data_33 if not data]))))
    data_35 = split_data_by_time(sort_data['35'])
    print()
    print(sorted(list(set([time.split(" ")[0] for time, data in data_35 if not data]))))

    # 计算窗口内均值
    data_1_aver = calc_average(data_1)
    data_2_aver = calc_average(data_2)
    data_3_aver = calc_average(data_3)
    data_33_aver = calc_average(data_33)
    data_35_aver = calc_average(data_35)
    ibrl_dict = {1: data_1_aver, 2: data_2_aver, 3: data_3_aver, 33: data_33_aver, 35: data_35_aver}

    dump("IBRL_average.pickle", ibrl_dict)


if __name__ == '__main__':
    run()