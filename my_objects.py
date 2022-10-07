# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2020/11/12
# -------------------------------------------------------------------------------
from collections import defaultdict
import os


# 该算法的参数
class AlgorithmParam(object):
    def __init__(self, true_dataset=False, need_calcualte_other_algorithm=True):
        # 生成网络的参数
        self.n = 1000
        self.k = 10
        self.maxk = 40
        self.minc = self.n * 0.04
        self.maxc = self.n * 0.06
        self.mut = 0.0
        self.muw = 0.0
        self.on = 10  # 重叠节点的个数
        self.on_weight = 0.05  # 重叠节点的所占比例
        self.om = 2
        # 算法的参数
        self.dataset = None
        self.need_show_image = False
        self.enveloped_weight = 1.0  # 就是是否判断该节点为包络节点占的比重
        self.cc_weight = 0.7  # 在计算节点的距离的时候，邻居节点得到的相关信息做的参考值（实验得出的结果）

        if true_dataset:
            self.u = 0.0  # 找到候选节点的可能性
            self.center_node_r_weight = (
                0.8  # 就是在选择中心节点的时候，作为中心节点的node_r * center_node_r_weight > averge_node_r
            )
            self.tt_weight = 2  # 节点拓扑结构所占的权重
            self.dc_weight = 0.1  # 计算dc所占的权重
            self.c = 0.1  # 在二次划分的时候[0.2, 0.4, 0.6, 0.8, 1.0]
            self.node_g_weight = 2  # 在node_g_1所占的权重比 [2, 4, 6]
            self.t = 0.2  # 在计算ls_cc的时候，有一个t为0-1之间的参数来控制共同节点对链接权重的贡献程度
        else:
            self.u = 0.6  # 找到候选节点的可能性
            self.center_node_r_weight = 0.5
            self.tt_weight = 3  # 节点拓扑结构所占的权重
            self.dc_weight = 0.2  # 计算dc所占的权重
            self.c = 0.1  # 在二次划分的时候[0.2, 0.4, 0.6, 0.8, 1.0]
            self.node_g_weight = 3  # 在node_g_1所占的权重比 [2, 4, 6]
            self.t = 0.2  # 在计算ls_cc的时候，有一个t为0-1之间的参数来控制共同节点对链接权重的贡献程度
        self.need_calcualte_maxw = False  # 在计算ls_cc的时候，有一个maxw的值，如果是有权重的网络，则该值为True
        self.need_calcualte_other_algorithm = need_calcualte_other_algorithm
        self.dataset_type = 'gml'  # 对于真实数据集在构造生成G的时候，有gml格式的，也有txt格式的
        self.run_ture_dataset = False  # 是否一次性全部运行真实数据集
        self.is_known_dataset = True  # 该真实数据集是否已知真实社区结构

    def __str__(self):
        return "n: {}, k: {}, maxk: {}, minc: {}, maxc: {}, muw: {}, mut: {}, on: {}, om: {}, u: {}, c: {}".format(
            self.n,
            self.k,
            self.maxk,
            self.minc,
            self.maxc,
            self.muw,
            self.mut,
            self.on,
            self.om,
            self.u,
            self.c,
        )


# 自定义一个结果集的类，方便后续将实验结果保存到mysql或文件中，甚至方便打印，避免print散落在在各处
class MyResultInof(object):
    def __init__(self):
        self.G = None
        self.true_overlapping_nodes = []  # 网络真实的重叠节点
        self.true_community_num = 0
        self.find_overlapping_nodes = []  # 算法发现的的重叠节点
        self.min_om = 2  # 重叠节点最少划分到的社区个数
        self.max_om = 2  # 重叠节点最多划分到的社区个数
        self.mapping_overlapping_nodes = []  # 真实的重叠节点与网络重叠节点的交集
        self.ascending_nod_p_nodes = []  # 所有按照node_p进行排序之后的nodes
        self.ascending_nod_r_nodes = []  # 所有按照node_r进行排序之后的nodes
        self.center_nodes = []  # 中心节点
        self.ls_zero_nodes = []  # 到所有中心节点的ls都是0的节点
        self.not_enveloped_nodes = []  # 非包络节点
        self.community_nodes_dict = {}  # 每个社区对应的节点集合
        self.not_overlapping_communit_nodes_dict = {}  # 未执行seconde_step(),得到的非重叠节点划分情况
        self.node_community_dict = defaultdict(list)  # 每个节点对应的社区信息
        self.overlapping_candidates = []  # 所有的可能的候选的重叠节点
        self.special_candidate_nodes = []  # LFR合成网络生成的不合理重叠节点（通过算法检测出来的关键字图只有1个节点等）

        # 以下是算法的评估值
        self.onmi = 0.0  # 算法得到onmi值
        self.omega = 0.0  # 算法得到的omega值
        self.f1_score = 0.0  # 算法得到的f1-score分数
        self.modularity = 0.0  # 算法得到的modularity

        self.spend_seconds = 0  # 算法执行消耗的时间(单位s)

        # 以下是其他算法的ONMI的值
        self.CPM_ONMI = 0.0
        self.CPM_OMEGA = 0.0
        self.CPM_F1_SCORE = 0.0
        self.CPM_MODULARITY = 0.0
        self.EADP_ONMI = 0.0
        self.EADP_OMEGA = 0.0
        self.EADP_F1_SCORE = 0.0
        self.EADP_MODULARITY = 0.0
        self.SLPA_ONMI = 0.0
        self.SLPA_OMEGA = 0.0
        self.SLPA_F1_SCORE = 0.0
        self.SLPA_MODULARITY = 0.0
        self.LFM_EX_ONMI = 0.0
        self.LFM_EX_OMEGA = 0.0
        self.LFM_EX_F1_SCORE = 0.0
        self.LFM_EX_MODULARITY = 0.0
        self.GCE_ONMI = 0.0
        self.GCE_OMEGA = 0.0
        self.GCE_F1_SCORE = 0.0
        self.GCE_MODULARITY = 0.0
        self.DEMON_ONMI = 0.0
        self.DEMON_OMEGA = 0.0
        self.DEMON_F1_SCORE = 0.0
        self.DEMON_MODULARITY = 0.0
        self.MOSES_ONMI = 0.0
        self.MOSES_OMEGA = 0.0
        self.MOSES_F1_SCORE = 0.0
        self.MOSES_MODULARITY = 0.0


class EvaluationResult(object):
    def __init__(self, onmi=0.0, omega=0.0, f1_score=0.0):
        # onmi 评价指标
        self.onmi = onmi
        self.CPM_ONMI = 0.0
        self.EADP_ONMI = 0.0
        self.SLPA_ONMI = 0.0
        self.LFM_EX_ONMI = 0.0
        self.GCE_ONMI = 0.0
        self.DEMON_ONMI = 0.0
        self.MOSES_ONMI = 0.0

        # omega 评价指标
        self.omega = omega
        self.CPM_OMEGA = 0.0
        self.EADP_OMEGA = 0.0
        self.SLPA_OMEGA = 0.0
        self.LFM_EX_OMEGA = 0.0
        self.GCE_OMEGA = 0.0
        self.DEMON_OMEGA = 0.0
        self.MOSES_OMEGA = 0.0

        # f1_score 评价指标
        self.f1_score = f1_score
        self.CPM_F1_SCORE = 0.0
        self.EADP_F1_SCORE = 0.0
        self.SLPA_F1_SCORE = 0.0
        self.LFM_EX_F1_SCORE = 0.0
        self.GCE_F1_SCORE = 0.0
        self.DEMON_F1_SCORE = 0.0
        self.MOSES_F1_SCORE = 0.0


# 将节点的所有信息统一定义为一个类，之后节点想过的所有信息应该统一放在NodeInfo中
class NodeInfo(object):
    def __init__(self):
        self.node = None
        self.node_p = 0.0  # 表示的就是节点的揉
        self.node_g = 0.0  # 表示的就是节点的伽马
        self.node_p_1 = 0.0  # 表示的归一化之后的揉
        self.node_g_1 = 0.0  # 表示的归一化之后的伽马
        self.node_r = 0.0  # 表示的就是归一化之后的揉*伽马
        self.node_dr = 0.0
        self.node_w = 0.0  # 表示某个节点的所占的权重
        self.node_w_1 = 0.0
        self.is_center_node = False  # 表示该节点是否为中心节点，默认都不是，因为中心节点是需要选出来的
        self.is_enveloped_node = True  # 是否为包络节点（讲道理，这里是不是定义为是否为重叠节点更加合适？论文是这么定义的）
        self.communities = []  # 表示每个节点划分的社区编号，因为是重叠社区，一个节点可能隶属多个社区

    def gatherAttrs(self):
        return ",".join(
            "{}={}".format(k, getattr(self, k)) for k in list(self.__dict__.keys())
        )

    # 对节点的信息打印重写，方便程序打印输出
    def __str__(self):
        return "[{}:{}]".format(self.__class__.__name__, self.gatherAttrs())


class ShowResultImageParm(object):
    def __init__(self):
        self.x_trains = []
        self.y_trains = []
        self.labels = []
        self.xlable = "Os"
        self.ylable = "ONMI"
        self.title = "a simple image"
        self.fig_path = "n-1000-muw-on_2.png"
        self.need_save = False
        self.need_show = True
        self.result_image_path = None
        # 与y_train_i.append(str(evaluation_result.onmi) + "," + str(evaluation_result.omega)) 这个一致
        # 在上面增加或者顺序调换，这里也需要做出相应的更改，这个主要用于表示结果图像的纵坐标
        self.images_ylabes = ["ONMI", r'$\Omega$ Index', "F-Score"]

    # 对节点的信息打印重写，方便程序打印输出
    def print_info(self):
        self.write_all_result_to_txt("*" * 40)
        print('*' * 40)
        print("x_trains")
        print(self.x_trains)
        self.write_all_result_to_txt("x_trains: {}".format(self.x_trains))
        self.write_all_result_to_txt("lables: {}".format(self.labels))
        self.write_all_result_to_txt("xlable: {}".format(self.xlable))
        self.write_all_result_to_txt("title: {}".format(self.title))
        print("y_trains")
        for y_train in self.y_trains:
            print(y_train)
            self.write_all_result_to_txt(y_train)
        print("lables")
        print(self.labels)
        print("xlable", self.xlable)
        print("title", self.title)
        print("*" * 40)
        self.write_all_result_to_txt("*" * 40)

    def write_all_result_to_txt(self, lines):
        txt_path = self.result_image_path + "/result.txt"
        with open(txt_path, "a") as f:
            f.write(str(lines) + "\n")
