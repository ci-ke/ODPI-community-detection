# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2020/12/9
# -------------------------------------------------------------------------------


# 生成多组图像，muw为生成图像的控制参数，每幅图中以u对比，横坐标为on重叠节点个数
from my_objects import ShowResultImageParm, AlgorithmParam
import random


# 命名规则：generate+不变的一个参数+多条曲线的参数+横坐标

#############################################################
# 1) 生成一组图像 n=1000
# 主要用于观察每个重叠节点所属社区的个数对实验的影响
# 图像中的曲线控制：muw
# 横坐标：om (重叠节点的所属的社区个数)
#############################################################
def generate_n_muw_om():
    params = []
    show_image_params = []
    nodes = [1000]
    muws = [0.1, 0.2, 0.3]
    oms = [2, 3, 4, 5, 6, 7, 8]
    minc_maxc = 0.04
    for n in nodes:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "n-{}-muw-om".format(n)
        # 网络图的路径
        show_image_param.fig_path = "n-{}-muw-om.png".format(n)
        show_image_param.xlable = "om"
        labels = []
        for_1 = []
        for muw in muws:
            labels.append("muw={}".format(muw))
            for_2 = []
            for om in oms:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.muw = muw
                param.mut = 0.1  # 为了让曲线好看点
                param.n = n
                param.on = n * param.on_weight
                param.om = om
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(om) for om in oms]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


#############################################################
# 1_2) 生成一组图像 n=1000
# 主要用于观察每个重叠节点所属社区的个数对实验的影响
# 图像中的曲线控制：muw
# 横坐标：om (重叠节点的所属的社区个数)
#############################################################
def generate_n_mut_om():
    params = []
    show_image_params = []
    nodes = [1000]
    muts = [0.1, 0.2, 0.3]
    oms = [2, 3, 4, 5, 6, 7, 8]
    minc_maxc = 0.04
    for n in nodes:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "n-{}-mut-om".format(n)
        # 网络图的路径
        show_image_param.fig_path = "n-{}-mut-om.png".format(n)
        show_image_param.xlable = "om"
        labels = []
        for_1 = []
        for mut in muts:
            labels.append("mut={}".format(mut))
            for_2 = []
            for om in oms:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.mut = mut
                param.n = n
                param.on = n * param.on_weight
                param.om = om
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(om) for om in oms]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


#############################################################
# 2) 生成多组图像 每组图像中muw不同
# 主要用于观察实验的节点的个数
# 图像中的曲线控制：n(节点的个数)
# 横坐标：om (每个重叠节点归属的社区个数)
#############################################################
def generate_muw_n_om():
    params = []
    show_image_params = []
    muws = [0.0]
    nodes = [1000, 3000]
    oms = [2, 3, 4, 5, 6, 7, 8]
    minc_maxc = 0.04
    for muw in muws:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "muw-{}-n-om".format(muw)
        # 网络图的路径
        show_image_param.fig_path = "muw-{}-n-om.png".format(muw)
        show_image_param.xlable = "om"
        labels = []
        for_1 = []
        for n in nodes:
            labels.append("n={}".format(n))
            for_2 = []
            for om in oms:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.muw = muw
                param.n = n
                param.on = n * param.on_weight
                param.om = om
                param.k = 20
                if n > 1000:
                    param.u = 0.3  # 不能那么高
                    param.c = 0.8
                else:
                    param.u = 0.6
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(om) for om in oms]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


#############################################################
# 3) 生成多组图像 每组图像中muw不同而已
# 主要用于观察候选的重叠节点和真实的重叠节点之间的一种关系
# 图像中的曲线控制：u (控制候选重叠节点的参数情况)
# 横坐标：on (重叠节点的个数)
#############################################################
def generate_mut_u_on():
    params = []
    show_image_params = []
    n = 1000
    muts = [0.0, 0.1]
    us = [0.1, 1.0]  # 控制候选重叠节点的个数
    # 可能成为候选节点的控制参数
    on_weights = [0.05, 0.10, 0.20, 0.30]  # 重叠节点个数
    minc_maxc = 0.04
    for mut in muts:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "muts-{}-u-on".format(mut)
        # 网络图的路径
        show_image_param.fig_path = "muts-{}-u-on.png".format(mut)
        show_image_param.xlable = "on"
        labels = []
        for_1 = []
        for u in us:
            labels.append("u={}".format(u))
            for_2 = []
            for on_weight in on_weights:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.mut = mut
                param.n = n
                param.on = n * on_weight
                param.u = u
                if u == 1.0:
                    param.c = 0.1
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(on_weight * n) for on_weight in on_weights]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


def generate_muw_u_on():
    params = []
    show_image_params = []
    n = 1000
    muws = [0.1, 0.2, 0.3]
    us = [0.1, 1.0]  # 控制候选重叠节点的个数
    # 可能成为候选节点的控制参数
    on_weights = [0.05, 0.15, 0.25, 0.35]  # 重叠节点个数
    minc_maxc = 0.04
    for muw in muws:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "muws-{}-u-on".format(muw)
        # 网络图的路径
        show_image_param.fig_path = "muws-{}-u-on.png".format(muw)
        show_image_param.xlable = "on"
        labels = []
        for_1 = []
        for u in us:
            labels.append("u={}".format(u))
            for_2 = []
            for on_weight in on_weights:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.muw = muw
                param.n = n
                param.on = n * on_weight
                param.u = u
                if u == 1.0:
                    param.c = 0.1
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(on_weight * n) for on_weight in on_weights]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


#############################################################
# 5) 生成多组图像 每组图像中重叠个数不同而已
# 主要用于观察候选(就要得到的是当om越大的时候，c应该比较小才合适)
# 图像中的曲线控制：c (在二次划分的时候，重叠节点划分的一个阈值)
# 横坐标：om
#############################################################
def generate_on_c_om():
    params = []
    show_image_params = []
    n = 1000
    on_weights = [0.05, 0.20, 0.40]
    cs = [0.2, 0.8]  # 二次分配的时候
    oms = [2, 3, 4, 5, 6, 7, 8]
    minc_maxc = 0.04
    for on_weight in on_weights:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "on-{}-c-om".format(on_weight * n)
        # 网络图的路径
        show_image_param.fig_path = "on-{}-c-om.png".format(on_weight * n)
        show_image_param.xlable = "om"
        labels = []
        for_1 = []
        for c in cs:
            labels.append("c={}".format(c))
            for_2 = []
            for om in oms:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.n = n
                param.on = n * on_weight
                param.om = om
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(om) for om in oms]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


#############################################################
# 6,7) 用于生成与其他对比算法得到的结果的运行图
#############################################################
def generate_contrast(muw=0.0, n=1000, mut=0.0):
    params = []
    n = n
    muw = muw
    oms = [2, 3, 4, 5, 6, 7, 8]
    minc_maxc = 0.04

    show_image_param = ShowResultImageParm()
    show_image_param.need_save = True
    show_image_param.need_show = False
    # 网络图的名称
    show_image_param.title = "muw-{}-mut-{}-n-{}-om".format(muw, mut, n)
    # 网络图的路径
    show_image_param.fig_path = "muw-{}-mut-{}-n-{}-om.png".format(muw, mut, n)
    show_image_param.xlable = "om"
    show_image_param.labels = [
        "ODPI",
        "CPM",
        "EADP",
        "SLPA",
        "LFR_EX",
        "DEMON",
        "MOSES",
    ]
    show_image_param.x_trains = [int(om) for om in oms]
    for om in oms:
        param = AlgorithmParam(need_calcualte_other_algorithm=True)
        param.minc = n * minc_maxc
        param.maxc = n * minc_maxc + 20
        param.muw = muw
        param.mut = mut
        param.n = n
        param.on = n * param.on_weight
        param.om = om
        if n > 1000:
            param.u = 0.3  # 不能那么高
            param.c = 0.8
        else:
            param.u = 0.6
        param.k = 20
        params.append(param)

    return params, [show_image_param]


#############################################################
# 8) 主要用于生成一些lfr网络用于测试 lfr网络对于自使用dc的一个敏感度
#############################################################
def generate_test_lfr_for_dc_params():
    params = []
    muw = 0.2
    n = 1000
    om = 2
    minc_maxc = 0.04
    ks = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 平均度
    for k in ks:
        param = AlgorithmParam()
        param.need_calcualte_other_algorithm = False
        param.need_show_image = False
        param.k = k
        param.minc = n * minc_maxc
        param.maxc = n * minc_maxc + 20
        param.muw = muw
        param.n = n
        param.on = n * param.on_weight
        param.om = om
        params.append(param)
    return params


#############################################################
# 8) 主要用于生成一些lfr网络用于测试 候选重叠节点的情况
#############################################################
def generate_test_lfr_for_u_params():
    params = []
    muts = [0.2]
    n = 1000
    us = [0.1, 0.3, 0.5, 0.8, 1.0]
    on_weights = [0.05, 0.20, 0.40]
    minc_maxc = 0.04
    for mut in muts:
        labels = []
        for u in us:
            for on_weight in on_weights:
                param = AlgorithmParam(need_calcualte_other_algorithm=False)
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.mut = mut
                param.n = n
                param.u = u
                param.on = n * on_weight
                param.c = 0.01
                params.append(param)
    return params


#############################################################
# 9) 主要用于生成一些lfr网络用于测试 候选重叠节点的情况
#############################################################
def generate_test_lfr_for_c_params():
    params = []
    muts = [0.2]
    n = 1000
    u = 0.6
    cs = [0.01, 0.03, 0.05, 0.1, 0.5]
    on_weights = [0.05, 0.20, 0.40]
    minc_maxc = 0.04
    for mut in muts:
        for c in cs:
            for on_weight in on_weights:
                param = AlgorithmParam(need_calcualte_other_algorithm=False)
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.mut = mut
                param.n = n
                param.u = u
                param.on = n * on_weight
                param.c = c
                params.append(param)
    return params


def test():
    params = []
    show_image_params = []
    muws = [0.0, 0.2]
    n = 1000
    ks = [10, 20, 30]
    oms = [2, 4, 6, 8]
    minc_maxc = 0.04
    for muw in muws:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "muw-{}-n-om".format(muw)
        # 网络图的路径
        show_image_param.fig_path = "muw-{}-n-om.png".format(muw)
        show_image_param.xlable = "om"
        labels = []
        for_1 = []
        for k in ks:
            labels.append("k={}".format(k))
            for_2 = []
            for om in oms:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.muw = muw
                param.n = n
                param.u = 0.8
                param.on = n * param.on_weight
                param.om = om
                param.k = k
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(om) for om in oms]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params


def test1():
    params = []
    show_image_params = []
    muts = [0.0, 0.2]
    n = 1000
    ks = [10, 20, 30]
    oms = [2, 4, 6, 8]
    minc_maxc = 0.04
    for mut in muts:
        show_image_param = ShowResultImageParm()
        show_image_param.need_save = True
        show_image_param.need_show = False
        # 网络图的名称
        show_image_param.title = "mut-{}-n-om".format(mut)
        # 网络图的路径
        show_image_param.fig_path = "mut-{}-n-om.png".format(mut)
        show_image_param.xlable = "om"
        labels = []
        for_1 = []
        for k in ks:
            labels.append("k={}".format(k))
            for_2 = []
            for om in oms:
                param = AlgorithmParam()
                param.need_calcualte_other_algorithm = False
                param.minc = n * minc_maxc
                param.maxc = n * minc_maxc + 20
                param.mut = mut
                param.n = n
                param.u = 0.8
                param.on = n * param.on_weight
                param.om = om
                param.k = k
                for_2.append(param)
            for_1.append(for_2)
        show_image_param.labels = labels
        show_image_param.x_trains = [int(om) for om in oms]
        show_image_params.append(show_image_param)
        params.append(for_1)
    return params, show_image_params
