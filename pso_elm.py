import random
import math
import scipy
from scipy import linalg
import numpy as np
import pandas as pd
import openpyxl
import argparse


def args_parser():
    pass


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


def Data():
    # # 获得数据
    # file_name = 'datasets.xlsx'
    # data_training = pd.read_excel(file_name, sheet_name='training', usecols=range(1, 14))
    # data_testing = pd.read_excel(file_name, sheet_name='testing', usecols=range(1, 14))
    # train_target = pd.read_excel(file_name, sheet_name='training', usecols=[14])
    # test_target = pd.read_excel(file_name, sheet_name='testing', usecols=[14])

    # 获取数据
    file_name = "Dataset/WSN-DS/binary_wsn-ds.xlsx"
    data_training = pd.read_excel(file_name, sheet_name='train data', usecols=range(0, 18), engine='openpyxl')
    data_testing = pd.read_excel(file_name, sheet_name='test data', usecols=range(0, 18), engine='openpyxl')
    train_target = pd.read_excel(file_name, sheet_name='train data', usecols=[18], engine='openpyxl')
    test_target = pd.read_excel(file_name, sheet_name='test data', usecols=[18], engine='openpyxl')
    print("Data loaded")

    # 变化矩阵形式
    Data.dt_training = Normalization(data_training.to_numpy())
    Data.dt_testing = Normalization(data_testing.to_numpy())
    Data.dt_target_training = train_target.to_numpy()
    Data.dt_target_testing = test_target.to_numpy()


def Hidden_layer(input_weights, biases, n_hidden_node, data_input):
    # 初始化 input weight
    # input_weight = input_weights.reshape(n_hidden_node, 13)
    input_weight = input_weights.reshape(n_hidden_node, 18)

    # 初始化 bias
    bias = biases

    # 转置 input weight
    transpose_input_weight = np.transpose(input_weight)

    # output hidden layer 矩阵
    hidden_layer = []
    for data in range(len(data_input)):
        # data input 乘以 input weight transpose
        h = np.matmul(data_input[data], transpose_input_weight)

        # 加上 bias
        h_output = np.add(h, bias)
        hidden_layer.append(h_output)

    return hidden_layer


# 激活 hidden layer
def Activation(hidden_layer):
    for row in range(len(hidden_layer)):
        for col in range(len(hidden_layer[row])):
            hidden_layer[row][col] = 1 / (1 + np.exp((hidden_layer[row][col] * (-1))))
    activation = hidden_layer

    return activation


# matriks moore penrose pseudo-inverse menggunakan SVD
def Pseudoinverse(hidden_layer):
    h_pseudo_inverse = scipy.linalg.pinv2(hidden_layer, cond=None, rcond=None, return_rank=False, check_finite=True)

    return h_pseudo_inverse


# 计算 output weight
def Output_weight(pseudo_inverse, target):
    beta = np.matmul(pseudo_inverse, target)

    return beta


# 计算预测结果 data testing
def Target_output(testing_hidden_layer, output_weight):
    target = np.matmul(testing_hidden_layer, output_weight)

    # 基于分类的目标矩阵映射
    prediction = []
    for result in range(len(target)):
        dist_target_0 = abs(target[result] - 0)
        dist_target_1 = abs(target[result] - 1)
        min_dist = min(dist_target_0, dist_target_1)
        if min_dist == dist_target_0:
            predict = 0
        elif min_dist == dist_target_1:
            predict = 1
        prediction.append(predict)

    return prediction


# 初始化粒子 / posisi partikel (input weight & bias)
def Particle(n_inputWeights, n_biases):
    # 初始化 input weight
    input_weights = []
    for input_weight in range(0, n_inputWeights):
        input_weights.append(round(random.uniform(-1.0, 1.0), 2))

    # 初始化 bias
    biases = []
    for bias in range(0, n_biases):
        biases.append(round(random.random(), 2))

    return input_weights + biases


# 速度初始化
def Velocity_0(n_particles):
    return [0] * n_particles


# evaluasi akurasi
def Evaluate(actual, prediction):
    true = 0
    for i in range(min(len(actual), len(prediction))):
        if actual[i] == prediction[i]:
            true += 1
    # 准确率
    accuracy = round(((true / len(prediction)) * 100), 2)

    return accuracy


# 获取 pBest
def Pbest(particles, fitness):
    fitness = np.expand_dims(fitness, axis=1)
    pbest = np.hstack((particles, fitness))

    return pbest


# 比较 pbest_t 和 pbest_t+1
def Comparison(pbest_t, pbest_t_1):
    for i in range(min(len(pbest_t), len(pbest_t_1))):
        if pbest_t[i][-1] > pbest_t_1[i][-1]:
            pbest_t_1[i] = pbest_t[i]
        else:
            pbest_t_1[i] = pbest_t_1[i]

    return pbest_t_1


# 获得群体中最好的粒子
def Gbest(particles, fitness):
    # fitness / akurasi terbaik
    best_fitness = np.amax(fitness)

    # 具有最佳适应度的粒子
    particle = fitness.index(best_fitness)
    best_particle = particles[particle]

    # gbest
    gbest = np.hstack((best_particle, best_fitness))

    return gbest


# update 速度
def Velocity_update(pbest, gbest, w, c1, c2, particles, velocity):
    # 寻找每个特征的边界
    interval = []
    for i in range(len(particles[0])):
        x_max = np.amax(np.array(particles)[:, i])
        x_min = np.amin(np.array(particles)[:, i])
        k = round(random.random(), 1)
        v_max_i = np.array(((x_max - x_min) / 2) * k)
        v_min_i = np.array(v_max_i * -1)
        intvl = np.hstack((v_min_i, v_max_i))
        interval.append(intvl)

    # update 速度
    r1 = round(random.random(), 1)
    r2 = round(random.random(), 1)
    for i in range(min(len(particles), len(velocity), len(pbest), len(gbest))):
        for j in range(min(len(particles[i]) - 1, len(pbest[i]) - 1)):
            velocity[i] = (w * velocity[i]) + (c1 * r1 * (pbest[i][j] - particles[i][j])) + (
                        c2 * r2 * (gbest[i] - particles[i][j]))

    return velocity


# update 粒子位置
def Position_update(current_position, velocity_update):
    for i in range(min(len(current_position), len(velocity_update))):
        for j in range(len(current_position[i])):
            current_position[i][j] = (current_position[i][j] + velocity_update[i])

    return current_position


# ELM
def Elm(particles, n_input_weights, n_hidden_node):
    fitness = []

    for i in range(len(particles)):
        # proses elm
        # -----------------training---------------------#

        # input weight
        input_weights = np.array(particles[i][0:n_input_weights])

        # bias
        biases = np.array(particles[i][n_input_weights:len(particles[i])])

        # 计算输出矩阵 hidden layer 在 data training
        hidden_layer_training = Hidden_layer(input_weights, biases, n_hidden_node, Data.dt_training)

        # 激活输出矩阵 hidden layer data training
        activation_training = Activation(hidden_layer_training)

        # 彭罗斯矩阵
        pseudo_training = Pseudoinverse(activation_training)

        # 计算 output weight 在 data training
        output_training = Output_weight(pseudo_training, Data.dt_target_training)

        # -----------------testing--------------------#

        # 计算输出矩阵 hidden layer 在 data testing
        hidden_layer_testing = Hidden_layer(input_weights, biases, n_hidden_node, Data.dt_testing)

        # 激活输出矩阵 hidden layer data testing
        activation_testing = Activation(hidden_layer_testing)

        # 计算预测结果
        prediction = Target_output(hidden_layer_testing, output_training)

        # 准确率
        accuracy = Evaluate(Data.dt_target_testing, prediction)
        fitness.append(accuracy)

    return fitness


def Run():
    # 初始化 PSO
    fitures = 18
    n_hidden_node = 10  # 隐藏节点数 / partikel bias
    n_input_weights = n_hidden_node * fitures  # 粒子 input weight
    population = 200  # 每次迭代中的总体
    max_iter = 30  # 迭代最大值
    w = 0.5  # 惯性缓冲器
    c1 = 1  # 速度常数 1
    c2 = 1  # 速度常数 2

    # data
    Data()

    # 速度初始化
    velocity_t = Velocity_0(population)

    # 初始位置
    particles = []
    # 人口减少
    for pop in range(population):
        particle = Particle(n_input_weights, n_hidden_node)
        particles.append(particle)

    # fitness 每一个粒子 = akurasi elm
    fitness_t = Elm(particles, n_input_weights, n_hidden_node)

    # 初始化 Pbest
    pbest_t = Pbest(particles, fitness_t)

    # 初始化 Gbest
    gbest_t = Gbest(particles, fitness_t)

    for iteration in range(max_iter):
        # update 速度
        velocity_t_1 = Velocity_update(pbest_t, gbest_t, w, c1, c2, particles, velocity_t)

        # update 粒子位置
        particles_t_1 = Position_update(particles, velocity_t_1)

        # elm
        fitness_t_1 = Elm(particles_t_1, n_input_weights, n_hidden_node)

        # update pbest
        pbest_t_1 = Pbest(particles_t_1, fitness_t_1)
        pbest_t_1 = Comparison(pbest_t, pbest_t_1)

        # update gbest
        gbest_t_1 = Gbest(particles_t_1, fitness_t_1)

        # update params#
        pbest_t = pbest_t_1
        gbest_t = gbest_t_1
        particles = particles_t_1
        velocity_t = velocity_t_1

        print(gbest_t_1[-1])


    print('Input Weights')
    print(gbest_t_1[0:n_input_weights])
    print('')
    print('Biases')
    print(gbest_t_1[n_input_weights:len(gbest_t_1) - 1])
    print('')
    print('Accuracy')
    print(gbest_t_1[-1])


if __name__ == '__main__':
    Run()

# WSN  非正则：   正则：
