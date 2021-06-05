import pickle
import numpy as np
import collections
import matplotlib.pyplot as plt


def load(file_path: str):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def dump(file_path, data):
    file = open(file_path, "wb")
    pickle.dump(data, file)


def draw_bar():
    data_wsn = {'Naive Baise': 87.32, 'c5': 97.57, 'PCA-SVM': 93.43, 'MLP': 97.64, 'RNN': 98.09, 'VAE-DF': 99.42, 'SAE-SVM': 99.0.}

    import matplotlib.pyplot as plt
    import numpy as np
    # 柱高信息
    bar_width = 0.15
    # 绘制柱状图
    plt.bar(data_wsn.keys(), data_wsn.values(), bar_width, align="center", alpha=0.5)
    font2 = {'family': 'SimSun',
             'weight': 'heavy',
             'size': 14,
             }
    plt.ylabel("检测率", font2)
    plt.ylim((85, 100))
    plt.legend(bbox_to_anchor=(0.5, -0.31), loc=8, ncol=3)
    plt.tick_params(labelsize=10)  # 坐标刻度字体大小
    plt.tight_layout()
    plt.savefig('覆盖率-训练轮数.png', dpi=400)
    plt.show()

draw_bar()