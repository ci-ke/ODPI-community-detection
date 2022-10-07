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
def show_result_image(image_param, index=0):
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
        # plt.title(title)  # 标题
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
            except Exception as e:
                # 这是由于OMEGA得名字不是正常得字母
                path = image_param.result_image_path + "/{}_OMEGA.png".format(index)
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


# 从txt文件中读取实验结果暂存的txt文件的数据，用于重新生成数据结果图，
# 因为放在论文中的实验结果图需要各种微调等
def read_pick_to_image():
    base_path = "./result_images/"
    result_file = "5_1_muw_contrast_OK"
    file_path = base_path + result_file + "/pickle.txt"
    import pickle

    with open(file_path, 'rb') as f2:
        try:
            for i in range(100):
                param = pickle.load(f2)
                assert isinstance(param, ShowResultImageParm)
                param.x_trains = [2, 3, 4, 5, 6, 7, 8]
                param.labels = ['ADPC', 'CPM', 'EADP', 'SLPA', 'LFM', 'DEMON', 'MOSES']
                param.result_image_path = "./result_images/ok_images/muw_contrast"
                param.images_ylabes = ["ONMI", r'$\Omega$ Index', "F-Score"]
                param.xlable = r"$O_m$"
                param.need_show = True
                param.need_save = True
                show_result_image(param, i)
        except Exception as e:
            print("===============================================")
            print(e)
            print("===============================================")
            return


# 基于扩展的自适应密度峰值聚类算法的重叠社区发现研究
# 基于扩展的自适应密度峰值聚类的重叠社区发现算法研究
# 对自适应扩展的密度峰值聚类算法在重叠社区发现中的应用研究
# 基于自适应的重叠社区发现算法研究
# 重叠社区发现算法研究
# 自适应密度峰值重叠社区发现方法研究

if __name__ == '__main__':
    read_pick_to_image()
