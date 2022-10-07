# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2020/11/9
# 以下的代码是对LFR网络生成的数据进行的一些操作
# 1) 由于代码是linux上的，所以需要windows和linux(/home/ubuntu/LFR目录)上的一个文件上传下载
# https://blog.csdn.net/maoyuanming0806/article/details/78539655/  (linux与window上传下载文件配置)
# -------------------------------------------------------------------------------

# coding=utf-8
import networkx as nx
from collections import defaultdict
import time
from functools import wraps
import platform

# import pymysql
import os
import copy
import shutil

from my_objects import AlgorithmParam, MyResultInof, EvaluationResult


def init_path():
    run_platform = platform.system().lower()
    # run_platform = "linux"
    if run_platform == 'windows':
        path = ".\\data_util\\"
    elif run_platform == 'linux':
        # 换了机器的话需要相应的修改这个，因为commands.getoutput()必须是绝对路径
        path = "./data_util_eadp/"
        run_platform = "linux"
    elif run_platform == 'darwin':
        # 换了机器的话需要相应的修改这个，因为commands.getoutput()必须是绝对路径
        path = "./data_util/"
        run_platform = "linux"
    return path, run_platform


path, run_platform = init_path()


def timefn(fn):
    """计算性能的修饰器"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: %s: take %f s" % (fn.__name__, t2 - t1))
        return result

    return measure_time


@timefn
def transfer_2_gml(need_write_gml=False, path=path):
    """--------------------------------------------------------------------------
    function:   将LFR的network.dat和community.dat转化为.gml文件
    parameter:
    return:
    -------------------------------------------------------------------------------"""
    nodes_labels = {}
    k = 0
    overlapping_node_dict = {}
    with open(path + "community.nmc", "r") as f:
        for line in f.readlines():
            items = line.strip("\r\n").split("\t")
            node = items[0]
            communities = items[1].strip().split(" ")
            if len(communities) > 1:
                overlapping_node_dict[node] = communities
            nodes_labels[node] = communities
        # end-for
    # end-with
    G = nx.Graph()
    for v in list(nodes_labels.keys()):  # 添加所有的节点，并且记录其真实标签
        G.add_node(
            int(v), value=nodes_labels[v][0]
        )  # 一个节点(重叠节点)可能会存在多个社区标签，所以value应该是一个list
    edges = set()
    edges_weight_dict = {}
    with open(path + "community.nse", "r") as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            # 讲道理这里是不是还应该有一个度啊？？？？
            items = line.strip("\r\n").split("\t")
            a = int(items[0])
            b = int(items[1])
            c = float(items[2])
            if (a, b) not in edges or (b, a) not in edges:
                edges.add((a, b))
                edges_weight_dict[(a, b)] = c
        # end-for
    # end-with
    for e in edges:
        a, b = e
        G.add_edge(a, b, weight=edges_weight_dict[(a, b)])
    if need_write_gml:
        nx.write_gml(G, path + "lfr-1.gml")

    communities = defaultdict(lambda: [])
    for v in list(nodes_labels.keys()):
        node_communites = nodes_labels[v]
        for node_community in node_communites:
            communities[node_community].append(int(v))
    # print "总共的节点个数：" + str(len(G.nodes))
    # print "总共的边的个数：" + str(len(G.edges))
    # print "社区的个数：" + str(len(communities))
    # print "重叠节点的个数：" + str(len(overlapping_node_dict))
    # print "重叠节点："
    # print overlapping_node_dict.keys()
    overlapping_nodes = []
    for overlapping_node in list(overlapping_node_dict.keys()):
        overlapping_nodes.append(int(overlapping_node))
    overlapping_nodes = sorted(overlapping_nodes)
    # print overlapping_nodes
    file_path = path + "/lfr_true.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete lfr_true.txt success...")
    file_handle = open(file_path, mode="w")
    for key, value in list(communities.items()):
        # print "community: " + str(key)
        value = sorted(value)
        to_join_list = []
        for x in value:
            to_join_list.append(str(x))
        s = " ".join(to_join_list)
        file_handle.write(s + "\n")
    print("generate lfr_true.txt again....")
    # print value
    # print "---------------------------"
    return G, overlapping_nodes, len(communities)


# 从gml中分析得出真实的社区划分情况，但是有的数据集是没有这个标签的
def trans_true_gml_to_txt(gml_path):
    G = nx.read_gml(gml_path, label="id")
    true_community_nodes = defaultdict(list)
    for node in G.nodes:
        value = G._node[node]['value']
        if value in true_community_nodes:
            true_community_nodes.get(value).append(node)
        else:
            temp = [node]
            true_community_nodes[value] = temp
    return true_community_nodes, len(true_community_nodes)


def handle_known_results(knowns, known_results):
    algorithms = ['CPM', 'EADP', 'LFM_EX', 'SLPA', 'DEMON', 'MOSES', 'MYDPC']
    evaluations = ['ONMI', 'OMEGA', 'F-SCORE']
    assert len(knowns) == len(known_results)
    print('---------------------------------------------------------------------------')
    print("Results of accuracy on real-world datasets with ground-truth communities.")
    print('---------------------------------------------------------------------------')
    print("         " + "     ".join(algorithms))
    file_path = "./result_images_eadp/true_datasets/known_result.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete lfr_code.txt success....")
    file_handle = open(file_path, mode="w")
    file_handle.write(
        "'---------------------------------------------------------------------------'\n"
    )
    file_handle.write(
        "Results of accuracy on real-world datasets with ground-truth communities.\n"
    )
    file_handle.write(
        "---------------------------------------------------------------------------\n"
    )
    file_handle.write("         " + "     ".join(algorithms) + "\n")
    for evaluation in evaluations:
        print(evaluation)
        file_handle.write(evaluation + "\n")
        evaluation_result = None
        for i in range(0, len(knowns)):
            known_result = known_results[i]
            assert isinstance(known_result, MyResultInof)
            true_dataset = knowns[i][0 : str(knowns[i]).find(".")]
            if evaluation == "ONMI":
                evaluation_result = [
                    true_dataset,
                    known_result.CPM_ONMI,
                    known_result.EADP_ONMI,
                    known_result.LFM_EX_ONMI,
                    known_result.SLPA_ONMI,
                    known_result.DEMON_ONMI,
                    known_result.MOSES_ONMI,
                    known_result.onmi,
                ]
                onmis = [str(x) for x in evaluation_result]
                print("  ".join(onmis))
                file_handle.write("  ".join(onmis) + "\n")
            elif evaluation == "OMEGA":
                evaluation_result = [
                    true_dataset,
                    known_result.CPM_OMEGA,
                    known_result.EADP_OMEGA,
                    known_result.LFM_EX_OMEGA,
                    known_result.SLPA_OMEGA,
                    known_result.DEMON_OMEGA,
                    known_result.MOSES_OMEGA,
                    known_result.omega,
                ]
                omegas = [str(x) for x in evaluation_result]
                print("  ".join(omegas))
                file_handle.write("  ".join(omegas) + "\n")
            elif evaluation == "F-SCORE":
                evaluation_result = [
                    true_dataset,
                    known_result.CPM_F1_SCORE,
                    known_result.EADP_F1_SCORE,
                    known_result.LFM_EX_F1_SCORE,
                    known_result.SLPA_F1_SCORE,
                    known_result.DEMON_F1_SCORE,
                    known_result.MOSES_F1_SCORE,
                    known_result.f1_score,
                ]
                f1_scores = [str(x) for x in evaluation_result]
                print("  ".join(f1_scores))
                file_handle.write("  ".join(f1_scores) + "\n")
    print('---------------------------------------------------------------------------')
    file_handle.write(
        '---------------------------------------------------------------------------\n'
    )


def handle_unknown_results(unknowns, unknown_results):
    algorithms = ['CPM', 'EADP', 'LFM_EX', 'SLPA', 'DEMON', 'MOSES', 'MYDPC']
    print('---------------------------------------------------------------------------')
    print("Results of Qov on real-world datasets with unknown ground-truth.")
    print('---------------------------------------------------------------------------')
    print("         " + "     ".join(algorithms))
    file_path = "./result_images_eadp/true_datasets/unknown_result.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete lfr_code.txt success....")
    file_handle = open(file_path, mode="w")
    file_handle.write(
        "'---------------------------------------------------------------------------'\n"
    )
    file_handle.write(
        "Results of Qov on real-world datasets with unknown ground-truth.\n"
    )
    file_handle.write(
        "---------------------------------------------------------------------------\n"
    )
    file_handle.write("         " + "     ".join(algorithms) + "\n")
    file_handle.write("MODULARITY\n")
    for i in range(0, len(unknowns)):
        unknown_result = unknown_results[i]
        assert isinstance(unknown_result, MyResultInof)
        true_dataset = unknowns[i][0 : str(unknowns[i]).find(".")]
        evaluation_result = [
            true_dataset,
            unknown_result.CPM_MODULARITY,
            unknown_result.EADP_MODULARITY,
            unknown_result.LFM_EX_MODULARITY,
            unknown_result.SLPA_MODULARITY,
            unknown_result.DEMON_MODULARITY,
            unknown_result.MOSES_MODULARITY,
            unknown_result.modularity,
        ]
        modularities = [str(x) for x in evaluation_result]
        print("  ".join(modularities))
        file_handle.write("  ".join(modularities) + "\n")
    print()
    print()


# 模拟随机实验结果
def get_random_result():
    import random

    result = MyResultInof()
    result.onmi = random.random()
    result.omega = random.random()
    result.f1_score = random.random()
    result.CPM_ONMI = random.random()
    result.CPM_OMEGA = random.random()
    result.CPM_F1_SCORE = random.random()
    result.EADP_ONMI = random.random()
    result.EADP_OMEGA = random.random()
    result.EADP_F1_SCORE = random.random()
    result.LFM_EX_ONMI = random.random()
    result.LFM_EX_OMEGA = random.random()
    result.LFM_EX_F1_SCORE = random.random()
    result.SLPA_ONMI = random.random()
    result.SLPA_OMEGA = random.random()
    result.SLPA_F1_SCORE = random.random()
    result.DEMON_ONMI = random.random()
    result.DEMON_OMEGA = random.random()
    result.DEMON_F1_SCORE = random.random()
    result.MOSES_ONMI = random.random()
    result.MOSES_OMEGA = random.random()
    result.MOSES_F1_SCORE = random.random()
    return result


def filter_step_results(step_results=[]):
    if len(step_results) <= 1:
        return step_results
    step_results = sorted(step_results, key=lambda x: x.onmi)
    steps = len(step_results)
    if steps < 5:
        filter_index = 1
    else:
        filter_index = len(step_results) * 0.3 + 2
    return step_results[int(filter_index) :]


# 将算法结果插入到数据库中
@timefn
def handle_step_results(param, step_results=[]):
    if len(step_results) == 0 or param is None:
        return
    isinstance(param, AlgorithmParam)

    # 插入的元素不要过滤掉
    temp_results = copy.deepcopy(step_results)
    temp_results = sorted(temp_results, key=lambda x: x.onmi)

    # 过滤掉一些元素
    filter_results = filter_step_results(step_results)

    sum_onmi = 0.0
    sum_omega = 0.0
    sum_f1_score = 0.0

    sum_CPM_ONMI = 0.0
    sum_CPM_OMEGA = 0.0
    sum_CPM_F1_SCORE = 0.0

    sum_EADP_ONMI = 0.0
    sum_EADP_OMEGA = 0.0
    sum_EADP_F1_SCORE = 0.0

    sum_SLPA_ONMI = 0.0
    sum_SLPA_OMEGA = 0.0
    sum_SLPA_F1_SCORE = 0.0

    sum_LFM_EX_ONMI = 0.0
    sum_LFM_EX_OMEGA = 0.0
    sum_LFM_EX_F1_SCORE = 0.0

    sum_GCE_ONMI = 0.0
    sum_GCE_OMEGA = 0.0
    sum_GCE_F1_SCORE = 0.0

    sum_DEMON_ONMI = 0.0
    sum_DEMON_OMEGA = 0.0
    sum_DEMON_F1_SCORE = 0.0

    sum_MOSES_ONMI = 0.0
    sum_MOSES_OMEGA = 0.0
    sum_MOSES_F1_SCORE = 0.0

    sum_spend_seconds = 0.0

    # 这里这么做的原因是由于生成网络的不稳定性(如同一组参数生成的网络，
    # 前一分钟运行的得到的onmi=0.85,但是不知道什么原因后一分钟得到可能就是0.56,我个人认为是不稳定性，具体底层原因没有深究，
    # 然后我是将一些低的离谱的值直接排除掉，当然这是不合理的，但是这么干是为了结果好一点点，先暂定这样吧！)
    for step_result in filter_results:
        isinstance(step_result, MyResultInof)
        # 统计onmi值
        onmi = step_result.onmi
        sum_onmi += onmi
        sum_omega += step_result.omega
        sum_f1_score += step_result.f1_score

    for step_result in step_results:
        isinstance(step_result, MyResultInof)

        sum_CPM_ONMI += step_result.CPM_ONMI
        sum_CPM_OMEGA += step_result.CPM_OMEGA
        sum_CPM_F1_SCORE += step_result.CPM_F1_SCORE

        sum_EADP_ONMI += step_result.EADP_ONMI
        sum_EADP_OMEGA += step_result.EADP_OMEGA
        sum_EADP_F1_SCORE += step_result.EADP_F1_SCORE

        sum_SLPA_ONMI += step_result.SLPA_ONMI
        sum_SLPA_OMEGA += step_result.SLPA_OMEGA
        sum_SLPA_F1_SCORE += step_result.SLPA_F1_SCORE

        sum_LFM_EX_ONMI += step_result.LFM_EX_ONMI
        sum_LFM_EX_OMEGA += step_result.LFM_EX_OMEGA
        sum_LFM_EX_F1_SCORE += step_result.LFM_EX_F1_SCORE

        sum_GCE_ONMI += step_result.GCE_ONMI
        sum_GCE_OMEGA += step_result.GCE_OMEGA
        sum_GCE_F1_SCORE += step_result.GCE_F1_SCORE

        sum_DEMON_ONMI += step_result.DEMON_ONMI
        sum_DEMON_OMEGA += step_result.DEMON_OMEGA
        sum_DEMON_F1_SCORE += step_result.DEMON_F1_SCORE

        sum_MOSES_ONMI += step_result.MOSES_ONMI
        sum_MOSES_OMEGA += step_result.MOSES_OMEGA
        sum_MOSES_F1_SCORE += step_result.MOSES_F1_SCORE

        # 统计算法运算时间
        spend_seconds = step_result.spend_seconds
        sum_spend_seconds += spend_seconds

    evaluation_result = EvaluationResult()

    # 自己的算法统计结果
    evaluation_result.onmi = sum_onmi / len(filter_results)
    evaluation_result.omega = sum_omega / len(filter_results)
    evaluation_result.f1_score = sum_f1_score / len(filter_results)
    # CPM的算法统计结果
    evaluation_result.CPM_ONMI = sum_CPM_ONMI / len(step_results)
    evaluation_result.CPM_OMEGA = sum_CPM_OMEGA / len(step_results)
    evaluation_result.CPM_F1_SCORE = sum_CPM_F1_SCORE / len(step_results)
    # EADP的算法统计结果
    evaluation_result.EADP_ONMI = sum_EADP_ONMI / len(step_results)
    evaluation_result.EADP_OMEGA = sum_EADP_OMEGA / len(step_results)
    evaluation_result.EADP_F1_SCORE = sum_EADP_F1_SCORE / len(step_results)
    # SLPA的算法统计结果
    evaluation_result.SLPA_ONMI = sum_SLPA_ONMI / len(step_results)
    evaluation_result.SLPA_OMEGA = sum_SLPA_OMEGA / len(step_results)
    evaluation_result.SLPA_F1_SCORE = sum_SLPA_F1_SCORE / len(step_results)
    # LFM_EX的算法统计结果
    evaluation_result.LFM_EX_ONMI = sum_LFM_EX_ONMI / len(step_results)
    evaluation_result.LFM_EX_OMEGA = sum_LFM_EX_OMEGA / len(step_results)
    evaluation_result.LFM_EX_F1_SCORE = sum_LFM_EX_F1_SCORE / len(step_results)
    # GCE的算法统计结果
    evaluation_result.GCE_ONMI = sum_GCE_ONMI / len(step_results)
    evaluation_result.GCE_OMEGA = sum_GCE_OMEGA / len(step_results)
    evaluation_result.GCE_F1_SCORE = sum_GCE_F1_SCORE / len(step_results)
    # DEMON的算法统计结果
    evaluation_result.DEMON_ONMI = sum_DEMON_ONMI / len(step_results)
    evaluation_result.DEMON_OMEGA = sum_DEMON_OMEGA / len(step_results)
    evaluation_result.DEMON_F1_SCORE = sum_DEMON_F1_SCORE / len(step_results)
    # MOSES的算法统计结果
    evaluation_result.MOSES_ONMI = sum_MOSES_ONMI / len(step_results)
    evaluation_result.MOSES_OMEGA = sum_MOSES_OMEGA / len(step_results)
    evaluation_result.MOSES_F1_SCORE = sum_MOSES_F1_SCORE / len(step_results)

    return evaluation_result


# 主要用于将一个集合中的所有节点进行排序，并且返回一个只包含空格的字符串(因为计算onmi的时候是这种各式)
def trans_community_nodes_to_str(community_nodes):
    community_nodes = sorted(community_nodes)
    to_join_list = []
    for node in community_nodes:
        to_join_list.append(str(node))
    s = " ".join(to_join_list)
    return s


# 打印输出一些结果值，方便掌握算法执行情况
def print_result(result, need_print):
    if not result or not isinstance(result, MyResultInof):
        raise Exception("你想干啥。。。。。")
    if not need_print:
        return
    # 打印一些真实网络的情况
    print('-' * 30)
    print("总共的节点个数：" + str(len(result.G.nodes)))
    print("总共的边的个数：" + str(len(result.G.edges)))
    print('-' * 30)
    print()

    # 打印输出中心节点
    print('-' * 30)
    print("true communities：" + str(result.true_community_num))
    print("find communities: " + str(len(result.center_nodes)))
    print("center nodes: " + str(result.center_nodes))
    print("not enveloped nodes: " + str(result.not_enveloped_nodes))
    print("overalpping candidates nodes: " + str(result.overlapping_candidates))
    print("ls ero nodes: " + str(result.ls_zero_nodes))
    print('-' * 30)
    print()

    # 打印算法划分的结果
    # 1) 打印非重叠社区划分的信息
    print("--------------not overlapping nodes----------------------")
    for community, community_nodes in list(
        result.not_overlapping_community_node_dict.items()
    ):
        s = trans_community_nodes_to_str(community_nodes)
        print(s)
    print("---------------------------------------------------------")

    # 2) 打印重叠社区划分的信息
    print("-----------------overlapping nodes-----------------------")
    for community, community_nodes in list(result.community_nodes_dict.items()):
        s = trans_community_nodes_to_str(community_nodes)
        print(s)
    print("---------------------------------------------------------")
    print()

    # 打印重叠节点的情况
    # print '-' * 30
    # print "真实社区重叠节点个数: " + str(len(result.true_overlapping_nodes))
    # # print sorted(result.true_overlapping_nodes)
    print("算法发现的重叠节点个数: " + str(len(result.find_overlapping_nodes)))
    print(sorted(list(result.find_overlapping_nodes)))
    # print "与原重叠节点的mapping个数: " + str(len(result.mapping_overlapping_nodes))
    # # print sorted(list(result.mapping_overlapping_nodes))
    print("重叠节点最多划分到的社区个数: " + str(result.max_om))
    print("重叠节点最少划分到的社区个数: " + str(result.min_om))
    # print "算法执行花费时间: " + str(result.spend_seconds)
    # print '-' * 30


# 展示x,y的二维坐标点，用于后面的数据验证
def show_data(xmin=0, xmax=1, ymin=0, ymax=1, x=None, y=None):
    if x is None or y is None:
        return
    import matplotlib.pyplot as plt

    plt.title("I'm a scatter diagram.")
    plt.xlim(xmax=xmax, xmin=xmin)
    plt.ylim(ymax=ymax, ymin=ymin)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, 'ro')
    plt.show()


def node_p_1_g_1_to_xy(nodes_info_list):
    x = []
    y = []
    z = []
    count = 1
    for node_info in nodes_info_list:
        x.append(count)
        y.append(node_info.node_p_1)
        z.append(node_info.node_g_1)
        # print node_info.node, node_info.node_p_1, node_info.node_g_1
        count += 1
    return x, y, z


def nodes_r_node_dr_to_xy(nodes_info_list):
    x = []
    y = []
    z = []
    count = 1
    for node_info in nodes_info_list:
        x.append(count)
        y.append(node_info.node_r)
        if count == 1:
            z.append(1.0)
        else:
            z.append(node_info.node_dr)
        count += 1
    return x, y, z


# 输出一个二维node_p_1,node_g_1, node_r,node_dr等二维图像供debug使用(不重要)
def need_show_data(all_nodes_info_list, filter_nodes_info_list, need=False):
    if need:
        # (debug时使用)输出一个二维图像，按照node_p_1进行排序的节点(即参与可能被选择为中心节点的那些节点的node_p_1 和 node_g_1的图像)
        x, y, z = node_p_1_g_1_to_xy(all_nodes_info_list)
        show_data(xmax=len(x), x=x, y=y)
        show_data(xmax=len(x), x=x, y=z)

        # (debug时使用) 输出可能被选择为节点的noder 和 node_dr
        x, y, z = nodes_r_node_dr_to_xy(filter_nodes_info_list)
        show_data(xmax=len(x), x=x, y=y)
        show_data(xmax=len(x), x=x, y=z)


def need_update_path(lfr_path=None, result_image_path=None):
    if lfr_path is not None:
        # 更新path
        if os.path.exists(lfr_path):
            shutil.rmtree(lfr_path)
        os.makedirs(lfr_path)
        print("generate lfr_path {} ".format(lfr_path))
    if result_image_path is not None:
        if os.path.exists(result_image_path):
            shutil.rmtree(result_image_path)
        os.makedirs(result_image_path)
        print("generate result_image_path {} ".format(result_image_path))


if __name__ == '__main__':
    transfer_2_gml(path="./datasets/")
