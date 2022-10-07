# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2020/11/21
# 主要是用来自动化协助我们画实验结果图
# -------------------------------------------------------------------------------
from my_util import init_path

path, run_platform = init_path()
if run_platform == "linux":
    import matplotlib

    matplotlib.use('Agg')
from matplotlib import pyplot
from my_objects import ShowResultImageParm

# 自动显示
def show_result_image(image_param):
    title_dict = {0: "(a)", 1: "(b)", 2: "(c)"}
    import matplotlib.pyplot as plt

    assert isinstance(image_param, ShowResultImageParm)
    x_trains = image_param.x_trains
    y_trains = image_param.y_trains
    image_count = len(str(y_trains[0][0]).split(","))
    for j in range(0, image_count):
        labels = image_param.labels
        assert len(labels) == len(y_trains)
        xlable = image_param.xlable
        ylable = image_param.images_ylabes[j]
        title = title_dict.get(j)
        names = [str(x) for x in list(x_trains)]
        temp = ['o', '*', '^', '+', 'p', '<', 'h', 'd', '1', '2']
        x = list(range(len(names)))
        new_yrains = []
        for y_train in y_trains:
            temp_y = []
            for y in y_train:
                temp_y.append(float(str(y).split(",")[j]))
            new_yrains.append(temp_y)
        for i in range(0, len(labels)):
            try:
                plt.plot(x, new_yrains[i], marker=temp[i], ms=10, label=labels[i])
            except Exception as e:
                print(e)
                print("====================except=================")
                image_param.print_info()
                return
        plt.legend()  # 让图例生效
        plt.xticks(x, names, rotation=2)
        plt.margins(0)
        plt.subplots_adjust(bottom=0.10)
        plt.xlabel(xlable)  # X轴标签
        plt.ylabel(ylable)  # Y轴标签
        yticks = get_yticks(new_yrains)
        pyplot.yticks(yticks)
        plt.title(title)  # 标题
        plt.legend(
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc=3,
            ncol=6,
            mode="expand",
            borderaxespad=0.0,
        )
        if image_param.need_save:
            path = (
                image_param.result_image_path
                + "/"
                + image_param.images_ylabes[j]
                + "_"
                + image_param.fig_path
            )
            try:
                plt.savefig(path, dpi=900)
            except:
                path = image_param.result_image_path + "/{}_test.png".format(j)
                plt.savefig(path, dpi=900)
        if image_param.need_show:
            plt.show()
        plt.close()


def get_yticks(y_trains):
    temp = []
    for y_train in y_trains:
        for x in y_train:
            temp.append(float(x))
    min_y = min(temp)
    max_y = max(temp)
    temp_a = [
        0,
        0.05,
        0.1,
        0.15,
        0.20,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]
    for i in range(1, len(temp_a)):
        if min_y < temp_a[i]:
            min_y = i - 1
            break
    for i in range(1, len(temp_a)):
        if max_y <= temp_a[i]:
            max_y = i
            break
    return temp_a[min_y : max_y + 1]


#############################
# 对于与其它对比算法将ONMI、OMEGA、F-Score合成一张图，
# 需要将title修改
#############################
def show_result_image2(image_param, other_param=None):
    import matplotlib.pyplot as plt

    assert isinstance(image_param, ShowResultImageParm)
    title_dict = {
        "0.2_0": "(a) ",
        "0.2_1": "(b) ",
        "0.2_2": "(c) ",
        "0.3_0": "(d) ",
        "0.3_1": "(e) ",
        "0.3_2": "(f) ",
    }
    x_trains = image_param.x_trains
    y_trains = image_param.y_trains
    image_count = len(str(y_trains[0][0]).split(","))
    plt.figure(figsize=(23, 6), dpi=30)
    for j in range(0, image_count):
        plt.subplot(1, 3, j + 1)
        labels = image_param.labels
        assert len(labels) == len(y_trains)
        xlable = image_param.xlable
        ylable = image_param.images_ylabes[j]
        names = [str(x) for x in list(x_trains)]
        temp = ['o', '*', '^', '+', 'p', '<', 'h', 'd', '1', '2']
        x = list(range(len(names)))
        new_yrains = []
        for y_train in y_trains:
            temp_y = []
            for y in y_train:
                temp_y.append(float(str(y).split(",")[j]))
            new_yrains.append(temp_y)
        for i in range(0, len(labels)):
            try:
                plt.plot(x, new_yrains[i], ms=10, label=labels[i], marker=temp[i])
            except Exception as e:
                print(e)
                return
        plt.xticks(x, names, rotation=2)
        plt.xlabel(xlable)  # X轴标签
        plt.ylabel(ylable)  # Y轴标签
        yticks = get_yticks(new_yrains)
        pyplot.yticks(yticks)
        plt.title(
            "{} muw={}".format(
                title_dict.get(str(other_param) + "_" + str(j)), other_param
            )
        )  # 标题
    num1 = 1.05
    num2 = 0.72
    num3 = 3
    num4 = 0
    # if other_param == 0.2:
    # plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc=3,
        ncol=6,
        mode="expand",
        borderaxespad=0.0,
    )
    path = image_param.result_image_path + "/{}_test.png".format(other_param)
    plt.savefig(path, dpi=900)
    plt.show()
    plt.close()


# 从txt文件中读取实验结果暂存的txt文件的数据，用于重新生成数据结果图，
# 因为放在论文中的实验结果图需要各种微调等
def read_pick_to_image():
    base_path = "./result_images/"
    result_file = "1_2_n_mut_om_OK"
    file_path = base_path + result_file + "/pickle.txt"
    import pickle

    with open(file_path, 'rb') as f2:
        try:
            for i in range(100):
                param = pickle.load(f2)
                assert isinstance(param, ShowResultImageParm)
                param.x_trains = [2, 3, 4, 5, 6, 7, 8]
                param.labels = [
                    'ODPI',
                    'CPM',
                    'EADP',
                    'SLPA',
                    'LFR_EX',
                    'DEMON',
                    'MOSES',
                ]
                param.result_image_path = "./result_images/ok_images"
                param.images_ylabes = ["ONMI", r'$\Omega$ Index', "F-Score"]
                param.need_show = True
                param.need_save = False
                other_param = 0.3
                # if i == 0:
                #     other_param = 0.2
                # show_result_image2(param, other_param)
                show_result_image(param)
        except Exception as e:
            print("===============================================")
            print(e)
            print("===============================================")
            return


if __name__ == '__main__':
    read_pick_to_image()
    # x_trains = [2, 4, 6, 8]
    # y_train_1 = ["0.8549032857142856,0.5", "0.8012512857142857,0.6", "0.746615857142857,0.7", "0.6966934285714286,0.8"]
    # y_train_2 = ["0.37274260000000004,0.125", "0.35266929999999996,0.254", "0.26617919999999995,0.257",
    #              "0.3286446,0.68"]
    # y_train_3 = ["0.49628101999999996,0.7414", "0.541,0.44444315", "0.215,0.42112183000000003", "0.38348271,0.98"]
    # y_train_4 = ["0.37274260000000004,0.325", "0.251,0.35266929999999996", "0.1547,0.26617919999999995",
    #              "0.658,0.3286446"]
    #
    # y_trains = [y_train_1, y_train_2, y_train_3, y_train_4]
    # labels = ['MYDPC', 'CPM', 'SLPA', 'LFR_EX']
    # image_param = ShowResultImageParm()
    # image_param.x_trains = x_trains
    # image_param.y_trains = y_trains
    # image_param.labels = labels
    # image_param.xlable = "om"
    # image_param.result_image_path = "./datasets"
    # image_param.need_show = True
    # image_param.need_save = False
    # image_param.title = "muw-0.2-om"
    # image_param.fig_path = "muw-0.2-om.png"
    # show_result_image(image_param)
