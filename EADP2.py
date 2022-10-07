# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# https://blog.csdn.net/qq_40587374/article/details/86597293(数据集图)
# 其中dolphins的数据图比真实的的数据集标签多1
# Author:       liuligang
# Date:         2020/9/14
# https://github.com/RapidsAtHKUST/CommunityDetectionCodes   社区发现算法的总结
# -------------------------------------------------------------------------------


import math
import os
import shutil
import sys
import time
from collections import defaultdict

import networkx as nx
import numpy as np

from my_evaluation import generate_network
from my_generate_param import *
from my_objects import MyResultInof, AlgorithmParam, NodeInfo
from my_help_image import show_result_image
from my_util import timefn, need_show_data, print_result, \
    trans_community_nodes_to_str, need_update_path
from my_util import transfer_2_gml, path, run_platform, handle_step_results, get_random_result
import pickle

# 1) Distance function
a = 0.1  # 计算cc(i,j)的时候使用的，一个较小的正值，避免分母为0的情况
b = 0.1  # 计算dist(i, j)的时候使用的，表示当i,j时孤立的节点的时候
# c = 0.8  # 在second_step()分配重叠节点的时候使用的。 todo 11.10 这个在论文后面的实验中有作对比
G = nx.Graph()
node_outgoing_weight_dict = {}
node_knn_neighbors_dict = {}
node_influence_dict = {}
dist_martix = None
ls_martix = None
dist_martix_2 = None # 主要用于将该距离矩阵转换成网络

all_nodes_info_list = []


# 计算G中最大权重
def calculate_maxw(need=False):
    if not need:
        return 1.0
    res = 0.0
    for u, v in G.edges:
        res = max(res, G[u][v]['weight'])
    return res


# 计算cc(i, j)，表示的是节点i,j的共同节点对节点i和节点j的链接强度的贡献，因此这个方法应该是考虑的节点的共同邻居节点
def calculate_cc_ij(nodei, nodej, V_ij=None, maxw=1.0, t=0.2):
    if V_ij is None:
        V_ij = nx.common_neighbors(G, nodei, nodej)
    r = 1.0  # 暂定1.0，todo 没有弄明白论文中的r所指的意思?????
    res = 0.0
    for node in V_ij:
        w_ipj = min(G[nodei][node]['weight'], G[nodej][node]['weight'])
        # 其实这里会发现，针对如果是无权重的图，temp就是等于0的
        temp = math.pow(((w_ipj - maxw) / (r * t + a)), 2)
        res = res + w_ipj * math.exp(-temp)
    return res


def calculate_node_outgoing_weight(node):
    res = 0.0
    for n in G.neighbors(node):
        res = res + G[node][n]['weight']
    return res


# 计算ls(i, j)，同时考虑直接链接权重和共同节点的共享，所以讲道理这个函数是考虑的cc(i,j)和i,j的之间的权重值
def calculate_ls_ij(nodei, nodej, maxw=1.0, cc_weight=0.5, tt_weight=2, t=0.2):
    N_i = list(nx.neighbors(G, nodei))
    N_j = list(nx.neighbors(G, nodej))
    V_ij = list(nx.common_neighbors(G, nodei, nodej))
    cc_ij = calculate_cc_ij(nodei, nodej, V_ij, maxw, t)
    t = (float(len(V_ij))) / len(set(N_i) | set(N_j)) + 1.0
    # i,j之间有边连接
    if G.has_edge(nodei, nodej):
        A_ij = G[nodei][nodej]['weight']
    else:
        A_ij = 0.0
    I_i = node_outgoing_weight_dict[nodei]
    I_j = node_outgoing_weight_dict[nodej]

    res = ((cc_ij * cc_weight + A_ij + t * tt_weight) * (len(V_ij) + 1)) / min(
        I_i, I_j)
    return res


# 计算节点i,j的distance
def calculate_dist_ij(nodei, nodej, maxw=1.0, cc_weight=0.5, tt_weight=2,
                      t=0.2):
    # 判断两个节点中是否存在至少一个节点为孤立节点
    if G.degree(nodei) == 0 or G.degree(nodej) == 0:
        ls_ij = 0.0
    else:
        ls_ij = calculate_ls_ij(nodei, nodej, maxw, cc_weight, tt_weight, t)
    res = 1 / (ls_ij + b)
    return res, ls_ij


def calculate_node_t_k(nodei):
    knn_nodes = nx.neighbors(G, nodei)
    try:
        return max([dist_martix[nodei][nodej] for nodej in knn_nodes if
                dist_martix != 1 / b])
    except:
        return 0.0

def calculate_dc(dc_weight=0.1):
    node_t_k_dict = {}
    for node in G.nodes:
        node_t_k = calculate_node_t_k(node)
        node_t_k_dict[node] = node_t_k
    u_k = float(sum(list(node_t_k_dict.values()))) / len(node_t_k_dict)
    temp = 0.0
    for node in G.nodes:
        temp += math.pow((node_t_k_dict[node] - u_k), 2)
    dc = math.sqrt(temp / (len(G.nodes) - 1)) + u_k
    return dc * dc_weight


# 初始化所有的节点之间的距离
@timefn  # 统计一下该函数调用花费的时间
def init_dist_martix(cc_weight=0.5, tt_weight=2, t=0.2,
                     need_calcualte_maxw=False):
    n = len(G.nodes)
    # 这里需要注意一点，因为使用了二维数组存储节点之间的dist数据，所以节点必须以数字表示，
    # 并且节点的下标必须是以0或者1开始
    # 对于非数字的graph，需要map转换一下
    dist_martix = [[1 / b for i in range(n + 1)] for i in range(n + 1)]
    dist_martix_2 = [[0.0 for i in range(n + 1)] for i in range(n + 1)]
    dist_martix_2 = np.array(dist_martix_2)
    ls_martix = [[0 for i in range(n + 1)] for i in range(n + 1)]
    # a = np.zeros([n+1, n+1])
    nodes = sorted(list(G.nodes))
    maxw = calculate_maxw(need_calcualte_maxw)
    for i in range(0, n):
        nodei = nodes[i]
        if G.degree(nodei) == 0:
            continue
        for j in range(i + 1, n):
            nodej = nodes[j]
            if dist_martix[nodei][nodej] == 1 / b:
                dist_ij, ls_ij = calculate_dist_ij(nodei, nodej, maxw, cc_weight,
                                                   tt_weight, t)
                dist_martix[nodei][nodej] = dist_ij
                dist_martix[nodej][nodei] = dist_ij
                dist_martix_2[nodei][nodej] = dist_ij
                dist_martix_2[nodej][nodei] = dist_ij
                ls_martix[nodei][nodej] = ls_ij
                ls_martix[nodej][nodei] = ls_ij
    return dist_martix, ls_martix, dist_martix_2

# 求网络的平均度
def calculate_knn():
    sum = 0
    # 返回的是每个节点的度的情况
    node_degree_tuple = nx.degree(G)
    for _, degree in node_degree_tuple:
        sum += degree
    return int(sum / len(node_degree_tuple))


# 计算一个节点的knn的邻居节点的集合 todo 这个方法有很严重的歧义，中英文版的论文给的不一样
def calculate_node_knn_neighbor(nodei, knn):
    knn_nodes = nx.neighbors(G, nodei)
    # 我个人觉得这里不一定是邻居节点,应该是将所有的节点的dist进行排序，取最近的k个节点
    # knn_nodes = [node for node in G.nodes if node != nodei]
    # 得到节点的所有邻居节点之间的dist
    node_neighbors_dist_tuple_list = [(x, dist_martix[nodei][x]) for x in
                                      knn_nodes]
    # 对所有的邻居节点进行排序········································
    node_neighbors_dist_tuple_list = sorted(node_neighbors_dist_tuple_list,
                                            key=lambda x: x[1])
    # 找到最小的k个邻居节点
    res = []
    k = len(node_neighbors_dist_tuple_list)
    # 如果不够就取所有的
    if k < knn:
        knn = k
    for i in range(knn):
        nodej = node_neighbors_dist_tuple_list[i][0]
        res.append(nodej)
    return res


# 计算每个节点的揉
def calculate_nodep(node, dc=0.2):
    # 找到最小的k个邻居节点，这里不按照论文的来，这里就是算所有邻居节点
    # knn_neighbors = calculate_node_knn_neighbor(node)
    # knn_neighbors = node_knn_neighbors_dict.get(node)
    knn_neighbors = list(G.neighbors(node))
    res = 0.0
    # 如果不够就取所有的
    for knn_neighbor in knn_neighbors:
        # a = float(dist_martix[node][knn_neighbor])
        temp = math.pow((float(dist_martix[node][knn_neighbor]) / dc), 2)
        res = res + math.exp(-temp)
    return res


# 初始化所有的节点的信息
@timefn
def init_all_nodes_info(node_g_weight=2, dc_weight=0.1, dc=None):
    res = []
    all_node_p = []
    all_node_w = []
    node_p_dict = {}
    if dc is None:
        dc = calculate_dc(dc_weight)
    print("dc: " + str(dc))
    # 初始化所有节点的影响力
    # node_influence_dict = init_all_nodes_influence(0.4, 0.4)
    # 1) 初始化所有的
    for node in G.nodes:
        node_p = calculate_nodep(node, dc)
        node_p_dict[node] = node_p
        node_w = node_outgoing_weight_dict[node]
        t = NodeInfo()
        t.node = node
        t.node_p = node_p
        t.node_w = node_w
        res.append(t)
        all_node_p.append(node_p)
        all_node_w.append(node_w)

    # 2) 对揉进行归一化
    min_node_p = min(all_node_p)
    max_node_p = max(all_node_p)
    min_node_w = min(all_node_w)
    max_node_w = max(all_node_w)
    for node_info in res:
        node_p = node_info.node_p
        node_p_1 = (node_p - min_node_p) / (max_node_p - min_node_p)
        node_info.node_p_1 = node_p_1
        node_w = node_info.node_w
        node_w_1 = (node_w - min_node_w) / (max_node_w - min_node_w)
        node_info.node_w_1 = node_w_1

    # 3) 初始化所有节点的伽马
    # 计算每个节点的伽马函数，由于这个方法外部不会调用，就暂且定义在方法内部吧，问题不大！
    def calculate_node_g(nodei, node_list):
        if len(node_list) == 0:
            return 1.0 / b
        temp = []
        for nodej in node_list:
            temp.append(node_g_weight * dist_martix[nodei][nodej])
        return min(temp)

    # 按照所有节点的揉进行升序排序
    res = sorted(res, key=lambda x: x.node_p)
    all_node_g = []
    for i in range(len(res)):
        # 当揉为最大的时候，取最大的dist
        if i == len(res) - 1:
            res[i].node_g = max(all_node_g)
            all_node_g.append(res[i].node_g)
        else:
            node_info = res[i]
            node = node_info.node
            # 因为res是根据揉排好序的，所有i之后的所有节点对应的揉都是大于当前的, 这里应该是需要加上后面的if
            node_list = [res[x].node for x in range(i + 1, len(res)) if
                         res[x].node_p > node_info.node_p]
            node_g = calculate_node_g(node, node_list)
            # todo 想不通为什么这里会有计算出node_g = 10.0的情况？？？？
            if node_g == 1.0 / b:
                node_g = res[i - 1].node_g
            all_node_g.append(node_g)
            node_info.node_g = node_g
    # 4) 对所有的节点的伽马进行归一化，并且求出r
    max_node_g = max(all_node_g)
    min_node_g = min(all_node_g)
    node_node_r_dict = {}
    for node_info in res:
        node_g = node_info.node_g
        node_g_1 = (node_g - min_node_g) / (max_node_g - min_node_g)
        node_info.node_g_1 = node_g_1
        # 且顺便计算出node_r
        node_r = node_info.node_p_1 * node_info.node_g_1
        node_info.node_r = node_r
        node_node_r_dict[node_info.node] = node_r

    # 打印一些节点的局部密度信息
    print("====" * 30)
    for node_info in res:
        if str(node_info.node) in ["1", "34", "28", "22"]:
            print(node_info.node, node_info.node_p_1)
    print("====" * 30)
    return res, node_node_r_dict, node_p_dict


# 打印一下初始化之后的节点的信息，所有节点按照p进行排序
def print_node_info():
    for node_info in all_nodes_info_list:
        print("节点： { %s } node_p_1的值: { %f } node_g_1的值：{ %f } node_r的值： { %f } " \
              % (node_info.node, node_info.node_p_1, node_info.node_g_1,
                 node_info.node_r))


# 讲道理这里应该还需要过滤一些更不不可能成为clustering node的节点
def filter_corredpond_nodes(all_nodes_info_list):
    all_nodes_info_list = sorted(all_nodes_info_list, key=lambda x: x.node_p)
    count = int(0.8 * len(all_nodes_info_list))
    sum_node_p = 0.0
    for i in range(count):
        sum_node_p += all_nodes_info_list[i].node_p
    averge_eighty_percen_node_p = float(sum_node_p) / count

    sum_node_r = 0.0
    all_nodes_info_list = sorted(all_nodes_info_list, key=lambda x: x.node_r)
    for i in range(count):
        sum_node_r += all_nodes_info_list[i].node_r
    averge_eighty_percen_node_r = float(sum_node_r) / count

    filter_nodes_info_list = []
    for node_info in all_nodes_info_list:
        if node_info.node_p < averge_eighty_percen_node_p or node_info.node_r < averge_eighty_percen_node_r:
            pass
        else:
            filter_nodes_info_list.append(node_info)
            sum_node_r += node_info.node_r
    averge_node_r = sum_node_r / len(filter_nodes_info_list)
    return filter_nodes_info_list, averge_node_r


# 初始化所有的节点的node_dr信息，并返回最大的node_dr以及对应的index
def init_filter_nodes_dr(filter_nodes_info_list):
    # 第一个节点应该是没有node_dr的，所以从第二个节点开始
    for i in range(1, len(filter_nodes_info_list)):
        a = filter_nodes_info_list[i - 1]
        b = filter_nodes_info_list[i]
        node_dr = b.node_r - a.node_r
        b.node_dr = node_dr


# ================================================================================
# 以上的所有代码应该是初始化好了所有的节点的信息，
# 包括揉，伽马，还有d伽马等信息。那么讲道理下面的步骤就应该是
# 1) 自动计算中心节点
# 2) 将节点划分到对应的社区
# ================================================================================

# 得到一维的线性拟合的参数a和b
def calculate_predict_node_dr(node_info_list, node_index):
    list_x = []
    list_y = []
    for i in range(len(node_info_list)):
        node_info = node_info_list[i]
        list_x.append(i + 1)
        list_y.append(node_info.node_dr)
    z = np.polyfit(list_x, list_y, 1)
    return z[0] * node_index + z[1]


# list_x = [1, 2, 3, 4, 5, 6]
# list_y = [2.5, 3.51, 4.45, 5.52, 6.47, 7.51]
# print calculate_linear_fitting_number(list_x, list_y, 8)
# 可以在这一步打印出节点的一些信息，进行验证
# for node in all_nodes_info_list:
#     print node.node, node.node_r, node.node_dr

# 算法二的核心，自动计算出node center
@timefn
def select_center(node_info_list, averge_node_r, center_node_r_weight=0.8):
    def calculate_max_node_dr(node_info_list):
        max_index = -1
        max_node_dr = -1
        for i in range(1, len(node_info_list)):
            node_info = node_info_list[i];
            t = node_info.node_dr
            if max_node_dr < t:
                max_node_dr = t
                max_index = i
        return max_node_dr, max_index

    res = -1
    # 这里的循环的过程不就会导致一种结果，那就是只要某个max_index是center，
    # 那么之后的所有节点不就肯定都是啦？？？
    while len(node_info_list) > 3:
        _, max_index = calculate_max_node_dr(node_info_list)
        temp_node_info = node_info_list[max_index]
        true_node_dr = temp_node_info.node_dr
        # 将所有的前面的进行拟合
        node_info_list = node_info_list[0:max_index]
        if len(
                node_info_list) < 3 or temp_node_info.node_dr < averge_node_r * center_node_r_weight:
            break
        predict_node_dr = calculate_predict_node_dr(node_info_list, max_index)
        if 2 * (true_node_dr - predict_node_dr) > true_node_dr:
            res = max_index
        else:
            break
    return res


# 初始化所有的中心节点,因为后面的节点划分社区都需要用到这个
def init_center_node(filter_nodes_info_list_index, filter_nodes_info_list,
                     all_nodes_info_dict):
    center_node_dict = {}
    community = 1
    # 因为从 filter_node_info_list_index 到最后都是中心节点
    for i in range(filter_nodes_info_list_index, len(filter_nodes_info_list)):
        filter_node_info = filter_nodes_info_list[i]
        node_info = all_nodes_info_dict.get(filter_node_info.node)
        node_info.is_center_node = True
        # 设置中心节点的社区，从编号1开始
        node_info.communities.append(community)
        # 将center_node的信息加入到center_node_list中，因为first_step会使用到该信息
        center_node_dict[node_info.node] = community
        community += 1
    return center_node_dict

@timefn
def select_center_kmeans(node_info_list, all_nodes_info_dict,
                         center_node_r_weight=1.0):
    ######################################
    #  使用聚类的方法进行中心节点的选择
    #  1）先使用KMeans进行二分聚类
    #  2）将最好的一类的最差的节点和最差的一类的最好的节点做一个差值做一个比较
    ######################################
    from my_select_center import select_kmeans
    k = 2
    center_node_list = []
    bad_list, midle_list, best_list = select_kmeans(node_info_list, k)
    center_node_list.extend(best_list)  # 作为中心节点添加进去
    averge_bad_node_r = sum(
        [node_info.node_r for node_info in node_info_list]) / len(bad_list)
    bad_list, midle_list, best_list = select_kmeans(bad_list, k)  # 将差的一类继续二分
    min_best_nodr_r = min([node_info.node_r for node_info in best_list])

    if min_best_nodr_r > averge_bad_node_r * center_node_r_weight:
        center_node_list.extend(best_list)

    # 初始化所有的中心节点信息
    center_node_dict = {}
    community = 1
    for node_info_temp in center_node_list:
        node_info = all_nodes_info_dict.get(node_info_temp.node)
        node_info.is_center_node = True
        # 设置中心节点的社区，从编号1开始
        node_info.communities.append(community)
        # 将center_node的信息加入到center_node_list中，因为first_step会使用到该信息
        center_node_dict[node_info.node] = community
        community += 1
    return center_node_dict


# 统计一下该节点和所有的中心节点的值都是0的情况，因为这种节点是随意划分的，需要思考一个方法把这种节点也正确划分
# 这里除了到中心节点为0的情况，还有一种情况就是到所有的中心节点的距离同相同
def calculate_zeor_ls_with_center_node(center_nodes=[], all_nodes=[]):
    all_ls_zero_nodes = []
    # 统计到所有中心节点为0的情况,或者到所有中心节点的ls强度都相同的点
    for node in all_nodes:
        temp = []
        if node not in center_nodes:
            for center_node in center_nodes:
                temp.append(ls_martix[node][center_node])
            # 如果到所有中心节点为0，或者到所有中心节点的距离都为0的话，那么该节点就不能随意划分
            if len(temp) == 0:
                continue
            if max(temp) == 0 or (len(temp) != 0 and max(temp) - min(temp) == 0):
                all_ls_zero_nodes.append(node)
    return all_ls_zero_nodes


# 将一些与中心节点的ls距离都是0的值进行划分，不能随意简单的划分
def divide_ls_zero_node(node, all_nodes_info_list, node_community_dict,
                        center_nodes_community):
    index = 0
    length = len(all_nodes_info_list)
    for i in range(0, length):
        if node == all_nodes_info_list[i].node:
            index = i
            break
    waiting_node_info = all_nodes_info_list[index]
    waiting_node = waiting_node_info.node
    waiting_node_p = waiting_node_info.node_p
    min_dist = 1000
    community = -1
    lg_node_p_list = []
    for i in range(index + 1, length):
        node_info = all_nodes_info_list[i]
        if node_info.node_p > waiting_node_p:
            lg_node_p_list.append(all_nodes_info_list[i])
    if len(lg_node_p_list) == 0:
        # 随意划分一个，但是这种情况几乎没有
        community = random.choice(center_nodes_community)
    else:
        # 先看它的邻居分配情况
        node_neighbors = nx.neighbors(G, node)
        node_neighbors_community_dict = {}
        for node_neighbor in node_neighbors:
            t = node_community_dict.get(node_neighbor, [-1])[0]
            if t != -1:
                if t in node_neighbors_community_dict:
                    node_neighbors_community_dict[t] = node_neighbors_community_dict[t] + \
                                                       ls_martix[node][node_neighbor]
                else:
                    node_neighbors_community_dict[t] = ls_martix[node][node_neighbor]
        temp = list(node_neighbors_community_dict.values())
        max_neighbor_ls = -1000
        for key, value in list(node_neighbors_community_dict.items()):
            if value > max_neighbor_ls:
                max_neighbor_ls = value
                community = key
        # 如果根据邻居还得不出该节点应该划分的社区，那么就按照下面的这种方式进行划分

        if len(temp) != 0 and len(temp) != 1 and max(temp) == min(temp):
            for node_info in lg_node_p_list:
                if dist_martix[node_info.node][waiting_node] < min_dist:
                    min_dist = dist_martix[node_info.node][waiting_node]
                    if node_info.node in node_community_dict:
                        community = node_community_dict.get(node_info.node)[0]
    if community == -1:
        # 随意划分一个，但是这种情况几乎没有
        community = random.choice(center_nodes_community)
    waiting_node_info.communities = []
    waiting_node_info.communities.append(community)
    # 这个结构主要是下面判断一个节点是否为包络节点需要使用到，所以在这里返回出去
    node_community_dict[waiting_node] = [community]

# 将一些与中心节点的ls距离都是0 或者都相等的节点进行划分，不能随意简单的划分
# 分配逻辑：
# 1）计算该节点到中心节点的的最短路径，将该节点划分到对应的最短的中心节点即可
# 2）如果到所有的中心节点的最短路径都相同的话，那么选择p值最大的中心节点
# 3）如果上面的两种情况都分配不了，那么就随机分配到一个中心节点即可（其实走到这一步的概率非常小）
def divide_ls_zero_node_2(ls_zero_nodes, G_temp, node_community_dict, center_node_dict, all_nodes_info_dict):

    can_not_match_nodes = []
    for ls_zero_node in ls_zero_nodes:
        zero_with_center_node_dist_dict = {}
        for center_node in list(center_node_dict.keys()):
            try:
                shortest_path_length = nx.shortest_path_length(G_temp, source=ls_zero_node,
                                                               target=center_node, weight='weight')
                zero_with_center_node_dist_dict[center_node] = shortest_path_length
            except:
                continue # 对于孤立的节点就是这种报异常的情况
        zero_dist_values = list(zero_with_center_node_dist_dict.values())
        if len(zero_with_center_node_dist_dict) == 0 \
                or max(zero_dist_values) - min(zero_dist_values) <= 0.01:
            can_not_match_nodes.append(ls_zero_node)
        else:
            zero_node_community = None
            min_with_center_node_dist = 100000
            for center_node in list(zero_with_center_node_dist_dict.keys()):
                with_center_node_dist = zero_with_center_node_dist_dict[center_node]
                if with_center_node_dist < min_with_center_node_dist:
                    min_with_center_node_dist = with_center_node_dist
                    zero_node_community = center_node_dict[center_node]
            waiting_node_info = all_nodes_info_dict.get(ls_zero_node)
            waiting_node_info.communities = []
            waiting_node_info.communities.append(zero_node_community)
            # 这个结构主要是下面判断一个节点是否为包络节点需要使用到，所以在这里返回出去
            node_community_dict[waiting_node_info.node] = [zero_node_community]

    print("=================================================")
    print("ls_zero_nodes: " + str(ls_zero_nodes))
    print("can_not_match_nodes: " + str(can_not_match_nodes))
    print("==================================================")
    # 通过上面都划分不了的节点（其实走到这一步剩余的节点是非常是少的节点，讲道理这里可以随意划分，
    # 但是我为了每次运行结果是固定的，先调用之前的按照邻居的节点划分情况进行划分），讲道理这里的
    # 操作对结果不是很影响。
    for ls_zero_node in can_not_match_nodes:
        divide_ls_zero_node(ls_zero_node, all_nodes_info_list, node_community_dict,
                            list(center_node_dict.values()))
        # zero_node_community = random.choice(center_node_dict.values())
        # waiting_node_info = all_nodes_info_dict.get(ls_zero_node)
        # waiting_node_info.communities = []
        # waiting_node_info.communities.append(zero_node_community)
        # node_community_dict[waiting_node_info.node] = [zero_node_community]

# 第一步将所有的非中心节点进行划分
@timefn
def first_step(center_node_dict, all_nodes_info_dict):
    # node_community_dict 就是记录所有的节点的划分的社区信息{}, 因为很多地方会使用到这个
    # node_community_dict = center_node_dict.copy()
    node_community_dict = defaultdict(list)
    for node in list(center_node_dict.keys()):
        node_community_dict[node] = [center_node_dict[node]]
    ls_zero_nodes = calculate_zeor_ls_with_center_node(
        list(node_community_dict.keys()), list(G.nodes))

    for node_info in all_nodes_info_list:
        waiting_node = node_info.node
        # 1) 先将所有的非中心节点且不是到所有的中心节点都不是零的值先进行划分
        if not node_info.is_center_node and waiting_node not in ls_zero_nodes:
            community = -1
            min_dist = -1000000
            for node in list(center_node_dict.keys()):
                node_ij_weight = 0.0
                if G.has_edge(waiting_node, node):
                    node_ij_weight = G[waiting_node][node]['weight']
                ls_ij = ls_martix[node_info.node][node] + node_ij_weight
                if ls_ij > min_dist:
                    community = center_node_dict.get(node)
                    min_dist = ls_ij
            node_info.communities = []
            node_info.communities.append(community)
            # 这个结构主要是下面判断一个节点是否为包络节点需要使用到，所以在这里返回出去
            node_community_dict[waiting_node] = [community]

    # 2) 将所有的零节点(也就是该节点到所有的中心节点都的强度都是0)划分，这一步也非常重要
    # todo 2021.03.18 这里的逻辑改了，与老孔商量之后，这里改为计算这些ls_zero的节点到所有中心节点的一个最短距离
    # 中心节点划分的社区
    center_nodes_community = list(center_node_dict.values())
    for ls_zeor_node in ls_zero_nodes:
        divide_ls_zero_node(ls_zeor_node, all_nodes_info_list, node_community_dict,
                            center_nodes_community)
    # 像这些ls_zero的节点划分，最后通过最短路径的方式进行划分
    # G_temp = nx.from_numpy_array(dist_martix_2)
    # divide_ls_zero_node_2(ls_zero_nodes, G_temp, node_community_dict,  center_node_dict, all_nodes_info_dict)
    return node_community_dict, ls_zero_nodes


# 计算每个节点的knn个邻居节点的ls的值之和
def calculate_node_knn_neighboor_ls(nodei, knn_node_neighbors,
                                    node_community_dict, comminity=None):
    res = 0.0
    for nodej in knn_node_neighbors:
        if comminity is None:
            res += ls_martix[nodei][nodej]
        else:
            if node_community_dict.get(nodej)[0] == comminity:
                res += ls_martix[nodei][nodej]
    return res


# 计算非包络节点的membership, 用于二次划分时将该节点划分到一个新的社区
def calculate_node_membership(nodei, node_community_dict):
    # 得到nodei的knn的邻居节点
    nodei_knn = node_knn_neighbors_dict[nodei]
    # 得到nodei的knn个邻居节点以及它们的划分社区信息
    node_knn_community_to_node_dict = {} # 记录一下所有邻居的社区划分情况，
    for nodej in nodei_knn:
        # 邻居节点的社区编号
        nodej_community = node_community_dict.get(nodej)[0]
        if nodej_community in node_knn_community_to_node_dict:
            node_knn_community_to_node_dict.get(nodej_community).append(nodej)
        else:
            node_knn_community_to_node_dict[nodej_community] = [nodej]
    node_membership_dict = {}
    # 对于每一个接待你进行划分
    for community_c in list(node_knn_community_to_node_dict.keys()):
        res = 0.0
        node_knn_c = node_knn_community_to_node_dict.get(community_c)
        for nodej in node_knn_c:
            # 得到是邻居节点的邻居节点集合，即是论文中的公式中p<- knn_j
            nodej_knn = node_knn_neighbors_dict[nodej]
            a = calculate_node_knn_neighboor_ls(nodej, nodej_knn, node_community_dict,
                                                community_c)
            b = calculate_node_knn_neighboor_ls(nodej, nodej_knn, node_community_dict)
            res += ls_martix[nodei][nodej] * (a / b)
        # 更新结果
        node_membership_dict[community_c] = res
    return node_membership_dict


# 划分重叠节点出来
@timefn
def second_step(node_community_dict, c, enveloped_weight=0.5,
                overlapping_candidates=[]):
    not_enveloped_nodes = []
    t_range = [] # 记录一下t的大致范围，从而在实验中可以好好设计一下合理的c
    neighbor_all_different_count = 0 # 该节点的社区和所有的邻居社区都不同的候选重叠节点个数（因为此种节点不会被划分到重叠社区中）
    for node_info in all_nodes_info_list:
        nodei = node_info.node
        if not node_info.is_center_node:
            # # 计算该节点是否为包络节点
            # node_neighbors = list(nx.neighbors(G, nodei))
            community = node_info.communities[0]
            if nodei in overlapping_candidates:
                # 说明该节点就不是包络节点
                node_info.is_enveloped_node = False
            else:
                not_enveloped_nodes.append(node_info.node)
            # 如果不是包络节点，那么会进行二次划分
            if not node_info.is_enveloped_node:
                # 1) 如果该节点和它的所有邻居划分社区都不相同，那么该节点先不管
                # 说明该节点和所有的邻居节点的社区中不包含该节点划分的社区，这种情况不管
                # nodei_knn_neighbors = calculate_node_knn_neighbor(nodei)
                nodei_knn_neighbors = node_knn_neighbors_dict[nodei]
                # 得到该节点的knn个最近的邻居节点的所有社区信息
                node_knn_neighbors_community = set(
                    [node_community_dict.get(node)[0] for node in nodei_knn_neighbors])
                # 表示的是该节点划分社区和周边所有的邻居划分的社区都不同，对于这种节点我们暂且不把它作为重叠节点处理
                if community not in node_knn_neighbors_community:
                    neighbor_all_different_count += 1
                else:
                    node_membership_dict = calculate_node_membership(nodei,
                                                                     node_community_dict)
                    # 遍历所有的knn节点的membership值，判断该节点是否划分到多个社区
                    nodei_community = node_community_dict.get(nodei)[0]
                    nodei_membership = node_membership_dict.get(nodei_community)
                    node_membership_dict.pop(nodei_community)
                    for community_c in list(node_membership_dict.keys()):
                        if nodei_membership == 0.0:
                            break
                        t = node_membership_dict.get(community_c) / nodei_membership
                        t_range.append(t)
                        if (t >= c):
                            # 说明需要将该节点划分到对应的社区
                            node_info.communities.append(community_c)
                            # 更新一下node_community_dict，说明该节点是一个重叠节点
                            if community_c != nodei_community:
                                node_community_dict.get(nodei).append(community_c)
        else:
            pass
    # print "====" * 10
    # print "neighbor_all_different_count: {}".format(neighbor_all_different_count)
    # print "t_range: {}".format(sorted(t_range))
    # print "====" * 10
    return not_enveloped_nodes


# 处理算法发现的结果(主要是直接将结果写入文件中，方便直接计算onmi，避免每次手动复制文件执行相应的脚本计算)
def handle_result_to_txt2(all_nodes_info_list, not_overlapping_community_dict,
                         new_makdir, run_platorm, param):
    community_nodes_dict = {}
    not_overlapping_community_node_dict = {}
    for node, communities in list(not_overlapping_community_dict.items()):
        community = communities[0]
        if community in not_overlapping_community_node_dict:
            not_overlapping_community_node_dict.get(community).append(node)
        else:
            not_overlapping_community_node_dict[community] = [node]

    for node_info in all_nodes_info_list:
        node = node_info.node
        communities = node_info.communities
        for community in communities:
            if community in community_nodes_dict:
                community_nodes_dict.get(community).append(node)
            else:
                community_nodes_dict[community] = [node]

    # 将结果集合写入文件, 讲道理这里还应该将划分的非重叠社区的结果也划分进去，后面如果想统计非重叠的NMI的值也方便(以后再说吧！)
    write_result_to_lfr_code_txt(new_makdir, community_nodes_dict, run_platorm,
                                 param)
    return community_nodes_dict, not_overlapping_community_node_dict


def write_result_to_lfr_code_txt(new_makdir, community_nodes_dict,
                                 run_platform="linux", param=None):
    if run_platform == "linux":
        file_path = path + new_makdir + "/lfr_code.txt"
        if param.run_ture_dataset and param.dataset is None:
            raise Exception("想要运行真实数据集的结果，真实数据集怎么为空了？？")
        if param.run_ture_dataset:
            dataset = param.dataset[0:str(param.dataset).find(".")]
            file_path = path + "true_datasets/" + dataset + "_code.txt"  # 生成真实数据集跑到的结果到相应的xxx_code.txt文件中，供后续的onmi等统计
    else:
        if param.run_ture_dataset and param.dataset is None:
            raise Exception("想要运行真实数据集的结果，真实数据集怎么为空了？？")
        dataset = param.dataset[0:str(param.dataset).find(".")]
        file_path = path + "true_datasets/" + dataset + "_code.txt"  # 生成真实数据集跑到的结果到相应的xxx_code.txt文件中，供后续的onmi等统计
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete lfr_code.txt success....")
    file_handle = open(file_path, mode="w")
    for key, value in list(community_nodes_dict.items()):
        s = trans_community_nodes_to_str(value)
        file_handle.write(s + "\n")
    print("generate lfr_code.txt again....")


# 统计一下算法发现的重叠节点，以及每个重叠节点所属的社区个数
def calculate_overlapping_nodes(node_community_dict):
    find_overlapping_nodes_dict = {}
    # 记录一下重叠节点被划分到的最多的社区个数和最小的社区个数
    min_om = 10000
    max_om = -10000
    for node in list(node_community_dict.keys()):
        communites = len(node_community_dict[node])
        if communites >= 2:
            if communites < min_om:
                min_om = communites
            if communites > max_om:
                max_om = communites
            # 记录每个重叠节点被划分到了几个社区
            find_overlapping_nodes_dict[node] = communites
    return find_overlapping_nodes_dict, min_om, max_om


# 就统计一下按照node_p 和 node_r 排序之后的节点信息，可能debug的时候用一下(不重要)
def calculate_ascending_nodes(filter_nodes_info_list, all_nodes_info_list):
    # 这个保存一下所有节点按照node_r进行排序之后的节点编号的变化信息，只是用来清晰的记录那个节点的揉的值是最大的而已
    ascending_nod_r_nodes = []
    # 因为此时的所有的all_nodes_info_list 是按照node_p进行升序的
    for node_info in filter_nodes_info_list:
        ascending_nod_r_nodes.append(node_info.node)
    # 这个保存一下所有节点按照揉进行排序之后的节点编号的变化信息，只是用来清晰的记录那个节点的揉的值是最大的而已
    ascending_nod_p_nodes = []
    # 因为此时的所有的all_nodes_info_list 是按照node_p进行升序的,这里暂且只收集前百分之
    for i in range(len(all_nodes_info_list) - len(filter_nodes_info_list),
                   len(all_nodes_info_list)):
        ascending_nod_p_nodes.append(all_nodes_info_list[i].node)
    return ascending_nod_p_nodes, ascending_nod_r_nodes


def calculate_node_KN(NB_i, nodei):
    t = 0
    node_KN_dict = defaultdict(list)
    while len(NB_i) > 0 and t < 2:
        NB_i_dict = defaultdict(list)
        # 统计节点的每个邻居节点 的共同邻居的情况，以得到Key Neighboring Node
        for node in NB_i:
            NB_i_dict[node] = list(nx.common_neighbors(G, node, nodei))
        ni_kN = None
        ni_kN_size = -100
        for key, value in list(NB_i_dict.items()):
            if len(value) > ni_kN_size:
                ni_kN = key
                ni_kN_size = len(value)
        gi_KN = NB_i_dict.get(ni_kN)
        gi_KN.append(ni_kN)
        NB_i = [node for node in NB_i if node not in gi_KN]
        t += 1
        node_KN_dict[ni_kN] = gi_KN
    return node_KN_dict


def calculate_L(G1_KN, G2_KN):
    l = 0
    for nodei in G1_KN:
        for nodej in G2_KN:
            # 这里用我们自己定义的ls_martrix
            # if G.has_edge(nodei, nodej):
            #     l += G[nodei][nodej]['weight']
            l += ls_martix[nodei][nodej]
    return float(l / 2.0)


def calculate_LC(G1_KN, G2_KN, is_true_node):
    lc_12 = calculate_L(G1_KN, G2_KN)
    lc_1 = calculate_L(G1_KN, G1_KN)
    lc_2 = calculate_L(G2_KN, G2_KN)

    if lc_1 == 0 or lc_2 == 0:
        # =======================================================================
        # 讲道理这里是不能这么操作的，但是为了实验效果好，这么操作了一下，
        # 其实这里倒不是算法的问题，而是LFR合成网络生成重叠节点的时候就有问题
        # 比如说LFR生成的网络中某个重叠节点，但是它发现的Key Neighboring Subgraph 很多都是[1],[2]
        # 也就是说，关键子图中的节点个数只有1个，导致lc_1或者lc_2为0，但其实根据重叠节点的定义，
        # 这样的节点是不应该被定义为重叠节点的，但是LFR合成网络就是生成这样的拓扑结构也认为该节点是重叠节点，
        # 所以对于这样的节点是无法通过任何算法找出来的
        # 在这里我使用了一种取巧的方法，将上述的这种节点直接添加到候选节点集合中（按道理不能这么操作的）
        # =======================================================================
        if is_true_node:
            # print G1_KN
            # print G2_KN
            return True, 0, None, None, None
        # 这种应该默认的不是吧重叠候选节点吧
        return False, 0, None, None, None
    lc = max(lc_12 / lc_1, lc_12 / lc_2)
    return True, lc, lc_12, lc_1, lc_2


# 找到候选的重叠节点
def find_overlapping_candidates(G, u, true_overlapping_nodes=[]):
    print("len overlapping: {}".format(len(true_overlapping_nodes)))
    overlapping_candidate_nodes = []
    special_candidate_nodes = []
    lc_range = []
    for node in list(G.nodes):
        NB_i = list(nx.neighbors(G, node))
        if len(NB_i) == 0:
            continue
        node_KN_dict = calculate_node_KN(NB_i, node)
        if len(node_KN_dict) < 2:
            continue
        node_KNs = list(node_KN_dict.values())
        G1_KN = node_KNs[0]
        G2_KN = node_KNs[1]
        is_true_node = False
        if node in true_overlapping_nodes:
            is_true_node = True
        flag, lc, lc_12, lc_1, lc_2 = calculate_LC(G1_KN, G2_KN, is_true_node)
        if flag:
            lc_range.append(lc)
        if flag and is_true_node and (lc_1 is None or lc_2 is None):
            special_candidate_nodes.append(node) # 记录一下这些LFR生成的不合理的重叠节点
        if flag and lc <= u:
            # print "lc: {}".format(lc)
            # print "lc_12: {}, lc_1: {}, lc_2: {}".format(lc_12, lc_1, lc_2)
            overlapping_candidate_nodes.append(node)
    # print "------" * 5
    # print "lc_range: {}".format(sorted(lc_range))
    # print "------" * 5
    return overlapping_candidate_nodes, special_candidate_nodes

# 该函数主要是针对节点的标号不是从0开始的网络，我们需要将其转换成从0开始的网络，
# 如unknown数据集中的(ca-GrQc.txt和CA-CondMat.txt数据集)
def rebalace_G_txt(G):
    nodes = list(G.nodes)
    nodes_dict = {}
    for i in range(0, len(nodes)):
        nodes_dict[int(nodes[i])] = i
    # 重新构造一个从节点0开始的网络
    G_temp = nx.Graph()
    for edge in G.edges:
        G_temp.add_edge(nodes_dict[int(edge[0])], nodes_dict[int(edge[1])])
    return G_temp

def start(param, run_windows_lfr=False, new_makdir="test", dc=2):
    if not isinstance(param, AlgorithmParam):
        raise Exception("你想搞啥呢？？？？？")

    need_show_image = param.need_show_image

    # 算法执行开始时间，统计这个算法运行花费时间
    start_time = time.time()

    global G, dist_martix, ls_martix, dist_martix_2
    global node_outgoing_weight_dict, node_knn_neighbors_dict
    global all_nodes_info_list
    global u, node_influence_dict
    global path

    # result 统一保存所有的中间结果
    result = MyResultInof()

    need_print_result = True
    def get_true_dataset_G():
        if param.is_known_dataset:
            g_path = "./datasets/known/" + param.dataset
        else:
            g_path = "./datasets/unknown/" + param.dataset
        if param.dataset_type == "gml":
            # dolphins的数据需要在网络图上加上1，也就是网络图上40，对应的真实的数据是39
            G = nx.read_gml(g_path, label="id")
            if param.dataset in ['netscience.gml', 'cond-mat.gml']:
                for edge in G.edges:
                    G[edge[0]][edge[1]]['weight'] = G[edge[0]][edge[1]]['value']
        elif param.dataset_type == "txt":
            G = nx.read_edgelist(g_path, delimiter=" ")
            # 针对这种节点不是从0开始的网络，转换一下
            if param.dataset in ['ca-GrQc.txt']:
                G = rebalace_G_txt(G)
            if param.dataset in ['ego-Facebook.txt']:
                G_temp = nx.Graph()
                for edge in G.edges:
                    G_temp.add_edge(int(edge[0]), int(edge[1]))  # 因为有些数据集的节点不是int类型
                G = G_temp
        return G
    need_add_default_weight = True
    # 如果是linux环境，则自动生成网络
    if run_platform == "linux":
        need_print_result = False
        if param.run_ture_dataset: # 运行真实网络数据集
            G = get_true_dataset_G()
            if param.dataset in ['netscience.gml', 'cond-mat.gml']:
                need_add_default_weight = False
        else:
            generate_network(param, path, new_makdir)
            # 处理LFR数据
            G, true_overlapping_nodes, true_community_num = transfer_2_gml(
                path=path + new_makdir + "/")
            need_add_default_weight = False
            result.true_overlapping_nodes = true_overlapping_nodes
            result.true_community_num = true_community_num
    else:
        G = get_true_dataset_G()
        if run_windows_lfr:  # 表示在windows机器上跑lfr数据集
            G, true_overlapping_nodes, true_community_num = transfer_2_gml(path=path)
            result.true_overlapping_nodes = true_overlapping_nodes
            result.true_community_num = true_community_num
    result.G = G

    if need_add_default_weight:
        # 默认边的权重为1.0
        for edge in G.edges:
            if G[edge[0]][edge[1]].get('weight', -1000) == -1000:
                G[edge[0]][edge[1]]['weight'] = 1.0

    # 初始化所有节点的outging_weight的值
    for node in G.nodes:
        outgoing_weight = calculate_node_outgoing_weight(node)
        node_outgoing_weight_dict[node] = outgoing_weight
    # 1) 初始化dist_martix，这一步是整个算法的基础，只有初始化dist_martix正确之后，后面的逻辑才走得通
    # ls_martix主要在second_step中使用到了，所以在这一步也初始化好
    print("begin to init martrix")
    dist_martix, ls_martix, dist_martix_2 = init_dist_martix(param.cc_weight, param.tt_weight,
                                              param.t, param.need_calcualte_maxw)
    print("init dist martix end.......")
    # 初始化好每个节点的knn_neighbors，避免后面重复计算，提高效率
    knn = calculate_knn()
    for node in G.nodes:
        knn_neighbors = calculate_node_knn_neighbor(node, knn)
        node_knn_neighbors_dict[node] = knn_neighbors


    # 2) all_nodes_info_list 很重要，所有节点的信息统一放在这个list中
    all_nodes_info_list, node_node_r_dict, node_p_dict = init_all_nodes_info(
        param.node_g_weight, param.dc_weight, dc)
    print('init all nodes info end......')

    # all_nodes_info_dict 便于后面从filter_node_list中通过node信息来更新到all_nodes_info_list上的信息
    all_nodes_info_dict = {node_info.node: node_info for node_info in
                           all_nodes_info_list}

    # 按照node_r进行排序,因为论文的算法二中选择中心节点就是使用的过滤之后的节点进行筛选的
    filter_nodes_info_list, averge_node_r = filter_corredpond_nodes(
        all_nodes_info_list)

    # 按照节点的node_r进行排序，这里需要进行拟合
    filter_nodes_info_list = sorted(filter_nodes_info_list,
                                    key=lambda x: x.node_r)

    # 非核心逻辑，不用管
    ascending_nod_p_nodes, ascending_nod_r_nodes = \
        calculate_ascending_nodes(filter_nodes_info_list, all_nodes_info_list)

    result.ascending_nod_p_nodes = ascending_nod_p_nodes
    result.ascending_nod_r_nodes = ascending_nod_r_nodes

    # 2) 初始化所有没有被过滤的节点的d伽马
    init_filter_nodes_dr(filter_nodes_info_list)
    print('init filter nodes end.......')

    # 非核心(不重要)
    need_show_data(all_nodes_info_list, filter_nodes_info_list, need_show_image)

    print("begin select center nodes...")
    use_kermaens = False
    # 4) 选择中心节点的逻辑(重要)
    if not use_kermaens:  # 使用原来的中心节点的方法
        # filter_nodes_info_list_index 表示的是过滤的节点的list的下标之后的所有节点为中心节点
        filter_nodes_info_list_index = select_center(filter_nodes_info_list,
                                                     averge_node_r,
                                                     param.center_node_r_weight)
        # print filter_nodes_info_list_index, len(filter_nodes_info_list)
        print("select center nodes end......")
        # print filter_nodes_info_list_index, len(filter_nodes_info_list)
        print("select center nodes end......")
        center_node_dict = init_center_node(filter_nodes_info_list_index,
                                            filter_nodes_info_list,
                                            all_nodes_info_dict)
    else:  # 使用的是聚类的方法进行中心节点的选择
        center_node_dict = select_center_kmeans(filter_nodes_info_list,
                                                all_nodes_info_dict,
                                                param.center_node_r_weight)
    print("select center nodes end...")
    # center_node_dict_2 = select_center_kmeans(filter_nodes_info_list_2)
    # 5) first_stpe, 将所有的非中心节点进行划分
    # 讲道理到了这一步之后，所有的节点都是已经划分了一个社区的，然后通过second_step()进行二次划分，将重叠节点找出来，并划分
    node_community_dict, ls_zero_nodes = first_step(center_node_dict, all_nodes_info_dict)

    center_nodes = sorted(list(center_node_dict.keys()))
    result.center_nodes = center_nodes
    # center_nodes_2 = sorted(list(center_node_dict_2.keys()))

    not_overlapping_node_community_dict = node_community_dict.copy()
    print("first step end.......")

    result.ls_zero_nodes = ls_zero_nodes

    print("begin to find overlapping candidates...")
    # 6) second_step, 将所有的可能是重叠节点的节点进行划分
    overlapping_candidates, special_candidate_nodes = find_overlapping_candidates(G, param.u, result.true_overlapping_nodes)
    result.overlapping_candidates = overlapping_candidates
    result.special_candidate_nodes = special_candidate_nodes

    print("begin to second step...")
    not_enveloped_nodes = second_step(node_community_dict, param.c,
                                      param.enveloped_weight,
                                      overlapping_candidates)
    result.not_enveloped_nodes = not_enveloped_nodes
    result.node_community_dict = node_community_dict
    print('second step end.......')
    # print overlapping_candidates
    # print len(result.true_overlapping_nodes), len(not_enveloped_nodes), len(set(result.true_overlapping_nodes) & set(not_enveloped_nodes))
    # print len(overlapping_candidates), len(not_enveloped_nodes), len(set(overlapping_candidates) & set(not_enveloped_nodes))

    # 7) 下面都是一些处理结果的逻辑，不是很核心
    # community_nodes_dict 每个社区对应的节点信息
    community_nodes_dict, not_overlapping_community_node_dict = \
        handle_result_to_txt2(all_nodes_info_list,
                             not_overlapping_node_community_dict, new_makdir,
                             run_platform, param)
    result.community_nodes_dict = community_nodes_dict
    result.not_overlapping_community_node_dict = not_overlapping_community_node_dict

    find_overlapping_nodes_dict, min_om, max_om = calculate_overlapping_nodes(
        node_community_dict)
    find_overlapping_nodes = list(find_overlapping_nodes_dict.keys())
    mapping_overlapping_nodes = list(
        set(find_overlapping_nodes) & set(result.true_overlapping_nodes))
    print("overlapping_nodes find info: true, candidate, true&candidate, find, true&find")
    print(len(result.true_overlapping_nodes), len(overlapping_candidates), len(
        set(result.true_overlapping_nodes) & set(overlapping_candidates)), \
        len(find_overlapping_nodes), len(mapping_overlapping_nodes))

    result.find_overlapping_nodes = find_overlapping_nodes
    result.mapping_overlapping_nodes = mapping_overlapping_nodes
    result.max_om = max_om
    result.min_om = min_om

    # 在linux上直接计算onmi的值，避免手动复制计算(麻烦)
    if run_platform == "linux" or param.run_ture_dataset:
        from my_evaluation import calculate_onmi
        from my_evaluation import calculate_omega
        from my_Fscore import calculate_f1_score
        from my_callEQ import cal_modularity
        true_dataset = None
        if param.run_ture_dataset:
            new_makdir = "true_datasets"  # 这里面有真实的社区结构(如karate_true.txt) 所以这个目录不能轻易的删除
            true_dataset = param.dataset[0:str(param.dataset).find(".")]
        if param.is_known_dataset:
            # onmi的代码是由c++编写的，我们在这里直接调用C++项目编译之后的可执行文件即可
            onmi = calculate_onmi(path, new_makdir, true_dataset)
            # 计算omega的代码是java代码编写的，所以这里还需要编译java代码
            omega = calculate_omega(path, new_makdir, true_dataset)
            # f1_score的代码是由python写的，直接调用即可
            f1_score = calculate_f1_score(path, new_makdir, true_dataset)
            modularity = cal_modularity(G, path, new_makdir, true_dataset)
            result.onmi = onmi
            result.omega = omega
            result.f1_score = f1_score
            result.modularity = modularity
        else:
            modularity = cal_modularity(G, path, new_makdir, true_dataset)
            result.modularity = modularity

        # 计算其他算法得到的ONMI值
        if param.need_calcualte_other_algorithm:
            # 1) CPM算法
            from my_CPM import get_community_nodes_dict
            print("start calculate CPM......")
            community_nodes_dict = get_community_nodes_dict(G)
            write_result_to_lfr_code_txt(new_makdir, community_nodes_dict,
                                         run_platform=run_platform, param=param)
            if param.is_known_dataset:
                onmi = calculate_onmi(path, new_makdir, true_dataset)
                omega = calculate_omega(path, new_makdir, true_dataset)
                f1_score = calculate_f1_score(path, new_makdir, true_dataset)
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.CPM_ONMI = onmi
                result.CPM_OMEGA = omega
                result.CPM_F1_SCORE = f1_score
                result.CPM_MODULARITY = modularity
            else:
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.CPM_MODULARITY = modularity

            # 2) LFR_EX算法
            from my_LFM_EX import get_community_nodes_dict
            print("start calculate LFM_EX......")
            try:
                community_nodes_dict = get_community_nodes_dict(G)
                write_result_to_lfr_code_txt(new_makdir, community_nodes_dict,
                                         run_platform=run_platform, param=param)
            except:
                pass
            if param.is_known_dataset:
                onmi = calculate_onmi(path, new_makdir, true_dataset)
                omega = calculate_omega(path, new_makdir, true_dataset)
                f1_score = calculate_f1_score(path, new_makdir, true_dataset)
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.LFM_EX_ONMI = onmi
                result.LFM_EX_OMEGA = omega
                result.LFM_EX_F1_SCORE = f1_score
                result.LFM_EX_MODULARITY = modularity
            else:
                try:
                    modularity = cal_modularity(G, path, new_makdir, true_dataset)
                except Exception as e:
                    modularity = 0.0
                result.LFM_EX_MODULARITY = modularity

            # 3) SLPA标签传播算法(由于不稳定需要10次求得平均值)
            from my_SLPA import get_community_nodes_dict
            print("start calculate SLPA......")
            count = 10
            SLPA_ONMIS = []
            SLPA_OMEGAS = []
            SLPA_F1_SCORE = []
            SLPA_MODULARITIES = []
            for i in range(0, count):
                community_nodes_dict = get_community_nodes_dict(G)
                write_result_to_lfr_code_txt(new_makdir, community_nodes_dict,
                                             run_platform=run_platform, param=param)
                if param.is_known_dataset:
                    onmi = calculate_onmi(path, new_makdir, true_dataset)
                    omega = calculate_omega(path, new_makdir, true_dataset)
                    f1_score = calculate_f1_score(path, new_makdir, true_dataset)
                    modularity = cal_modularity(G, path, new_makdir, true_dataset)
                    SLPA_ONMIS.append(onmi)
                    SLPA_OMEGAS.append(omega)
                    SLPA_F1_SCORE.append(f1_score)
                    SLPA_MODULARITIES.append(modularity)
                else:
                    modularity = cal_modularity(G, path, new_makdir, true_dataset)
                    SLPA_MODULARITIES.append(modularity)
            if param.is_known_dataset:
                onmi = float(sum(SLPA_ONMIS)) / count
                omega = float(sum(SLPA_OMEGAS)) / count
                f1_score = float(sum(SLPA_F1_SCORE)) / count
                modularity = float(sum(SLPA_MODULARITIES)) / count
                result.SLPA_ONMI = onmi
                result.SLPA_OMEGA = omega
                result.SLPA_F1_SCORE = f1_score
                result.SLPA_MODULARITY = modularity
            else:
                modularity = float(sum(SLPA_MODULARITIES)) / count
                result.SLPA_MODULARITY = modularity

            # 4)GCE算法 由C语言实现，所以需要先将Graph输入为xxx.csv 格式(里面的数据就是所有的边的信息)
            from my_GCE import calculate_gce
            print("start calculate GCE......")
            calculate_gce(path, new_makdir, G, true_dataset)
            if param.is_known_dataset:
                onmi = calculate_onmi(path, new_makdir, true_dataset)
                omega = calculate_omega(path, new_makdir, true_dataset)
                f1_score = calculate_f1_score(path, new_makdir, true_dataset)
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.GCE_ONMI = onmi
                result.GCE_OMEGA = omega
                result.GCE_F1_SCORE = f1_score
                result.GCE_MODULARITY = modularity
            else:
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.GCE_MODULARITY = modularity

            # 5) DEMON算法，由python代码实现，直接传入G,然后在算法里面将结果生成对应的lfr_code.txt，这就是最简单的调用方式
            from my_DEMON import calculate_demon
            print("start calculate DEMON......")
            calculate_demon(G, path, new_makdir, true_dataset)
            if param.is_known_dataset:
                onmi = calculate_onmi(path, new_makdir, true_dataset)
                omega = calculate_omega(path, new_makdir, true_dataset)
                f1_score = calculate_f1_score(path, new_makdir, true_dataset)
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.DEMON_ONMI = onmi
                result.DEMON_OMEGA = omega
                result.DEMON_F1_SCORE = f1_score
                result.DEMON_MODULARITY = modularity
            else:
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.DEMON_MODULARITY = modularity

            # 6) MOSES算法，由C++代码实现，
            # 输入INPUT: list of edges, one edge per line
            # 输出OUTPUT: filename to write the grouping. Will contain one line per community.
            from my_MOSES import calculate_moses
            print("start calculate MOSES......")
            calculate_moses(path, new_makdir, G, true_dataset)
            if param.is_known_dataset:
                onmi = calculate_onmi(path, new_makdir, true_dataset)
                omega = calculate_omega(path, new_makdir, true_dataset)
                f1_score = calculate_f1_score(path, new_makdir, true_dataset)
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.MOSES_ONMI = onmi
                result.MOSES_OMEGA = omega
                result.MOSES_F1_SCORE = f1_score
                result.MOSES_MODULARITY = modularity
            else:
                modularity = cal_modularity(G, path, new_makdir, true_dataset)
                result.MOSES_MODULARITY = modularity
            print("end calculate MOSES......")
            if param.is_known_dataset:
                print("===========================onmi & omega & f1-score=====================================")
                print("---------------MYDPC:-----------------")
                print("onmi: " + str(result.onmi))
                print("omega: " + str(result.omega))
                print("f1_score: " + str(result.f1_score))
                print("---------------CPM:-----------------")
                print("onmi: " + str(result.CPM_ONMI))
                print("omega: " + str(result.CPM_OMEGA))
                print("f1_score: " + str(result.CPM_F1_SCORE))
                print("---------------LFM_EX:-----------------")
                print("onmi: " + str(result.LFM_EX_ONMI))
                print("omega: " + str(result.LFM_EX_OMEGA))
                print("f1_score: " + str(result.LFM_EX_F1_SCORE))
                print("---------------SLPA:-----------------")
                print("onmi: " + str(result.SLPA_ONMI))
                print("omega: " + str(result.SLPA_OMEGA))
                print("f1_score: " + str(result.SLPA_F1_SCORE))
                print("---------------GCE:-----------------")
                print("onmi: " + str(result.GCE_ONMI))
                print("omega: " + str(result.GCE_OMEGA))
                print("f1_score: " + str(result.GCE_F1_SCORE))
                print("---------------DEMON:-----------------")
                print("onmi: " + str(result.DEMON_ONMI))
                print("omega: " + str(result.DEMON_OMEGA))
                print("f1_score: " + str(result.DEMON_F1_SCORE))
                print("---------------MOSES:-----------------")
                print("onmi: " + str(result.MOSES_ONMI))
                print("omega: " + str(result.MOSES_OMEGA))
                print("f1_score: " + str(result.MOSES_F1_SCORE))
                print("===========================onmi & omega & f1-score=====================================")
            else:
                pass
        else:
            print("===========================onmi & omega & f1-score=====================================")
            print("---------------MYDPC:-----------------")
            print("onmi: " + str(result.onmi))
            print("omega: " + str(result.omega))
            print("f1_score: " + str(result.f1_score))

    # 统计一下时间而已，不重要
    end_time = time.time()
    spend_seconds = end_time - start_time
    result.spend_seconds = spend_seconds
    return result, need_print_result
    # # 打印一些结果，供观察算法输出情况
    # print_result(result, need_print_result)



# 主要是运行lfr网络生成数据的实验(1: 自己算法的不同参数的一个对比 2: 自己算法和其他算法的一个对比，都会自动输出结果数据集和对比图)
def run_linux_generate_picture(steps=5, new_makdir="test"):
    ###############################################
    # 如果想要在linux运行并自动生成图像，那么就需要控制下面的这些参数（最多控制三个变量）
    # 1）一般三个控制变量，第一个控制变量用于控制生成多张对比图像
    # 2）如果是两个变量的话，那么只会生成一张图像，第一个变量就是生成一张图像上的多条曲线，第二个变量就是图像的x轴
    # 3）如果想在linux的shell窗口运行的多个的话，需要每次修改不同的 new_makdir，
    # 因为需要生成lfr数据，如果公用一个目录，数据会混论
    # 4）迭代次数，一般默认取5，其实在这里如果将迭代的次数修改为更大的数据的话，并且在得到ONMI值得时候，多排除几个小的数据，结果会更好
    # n_muw_om: EADP对应的 实验1
    # muw_n_om EADP对应的 实验2
    # muw_u_on：我们想做的对比实验一 候选的重叠节点 和 真实的重叠节点之间的关系
    # muw_u_t_1 我们想做的对比实验二
    # muw_table：EADP对应的 实验4（就是我想要的表格）
    # muw_contrast EADP对应的实验5
    # node_contrast EADP对应的实验6
    ###############################################
    param_dict = {"test": test,
                  "test1": test1,
                  "n_muw_om": generate_n_muw_om, # 对比三个不同的muw的实验
                  "n_mut_om": generate_n_mut_om, # 对比三个不同的mut的实验
                  "muw_n_om": generate_muw_n_om, # 对比三个不同的N的实验(目前不做)
                  "mut_u_on_1": generate_mut_u_on, # 对比实验一(候选重叠节点和重叠节点的个数关系)
                  "muw_u_on_2": generate_muw_u_on, # 对比实验二(候选重叠节点和重叠节点的个数关系)
                  "on_c_om_3": generate_on_c_om, # 对比实验三（om和二次分配控制c的关系）
                  "muw_table": "", # 与其他5个对比算法的实验，生成表格
                  "mut_table": "", # 与其他5个对比算法的实验，生成表格
                  "muw_contrast": generate_contrast, # 不同的muw下的与其他5个对比算法的实验，生成对比图
                  "mut_contrast": generate_contrast, # 不同的mut下的与其他5个对比算法的实验，生成对比图
                  "node_contrast": generate_contrast} # 不同的N下的与其他5个对比算法的实验，生成对比图(目前不做)
    result_image_path = "./result_images_eadp/" + new_makdir
    # 删除一些算法之前运行留下的文件目录，重新生成新的，避免乱七八糟的混乱问题
    need_update_path(path + new_makdir, result_image_path)
    if new_makdir in ["test", "test1"]:  # 只不过是测试的而已
        steps = 1
    if new_makdir in ["muw_n_om", "node_contrast"]: # 由于节点的个数太多了，这里将step设置的小一点就好
        steps = 2
    if new_makdir in ["muw_contrast", "mut_contrast", "node_contrast"]:  # 表示的是处理的是与其他算法进行对比的情况(因为有些实验，只是针对自己的不同参数做一个对比而已的)
        deal_contrast_generate_image(new_makdir, param_dict, result_image_path, steps)
        return
    if new_makdir in ["muw_table", "mut_table"]: # 表示的是处理的实验仅仅一个与其他算法的一个对比实验的表格数据而已，和上面的那些得到对比图的不一样
        deal_contrast_generate_table(new_makdir, steps)
        return
    # 通过封装好的生成参数函数来生成相应的参数列表(以后想要增加实验，只需要在里面新增相应的参数函数即可)
    params, show_image_params = param_dict.get(new_makdir)()
    for show_image_param in show_image_params:
        show_image_param.result_image_path = result_image_path
    else:  # 表示的自己各种不同的参数对比而已
        assert len(params) == len(show_image_params)
        # 每一轮执行10个迭代
        y_trains_all = []
        for for_1 in params:
            y_tains = []
            for for_2 in for_1:
                y_train_i = []
                for param in for_2:
                    print('-' * 30)
                    print("n={}, k={}, maxk={}, minc={}, maxc={}, muw={}, mut={}, on={}, " \
                          "om={}, c={}, node_g_weight={}, u={}, dc_weight={}, tt_weight={}, center_node_r_weight={}".format(
                        param.n, param.k,
                        param.maxk, param.minc, param.maxc,
                        param.muw, param.mut, param.on, param.om, param.c, param.node_g_weight,
                        param.u, param.dc_weight,
                        param.tt_weight, param.center_node_r_weight))
                    step_results = []
                    for i in range(0, steps):
                        #result, _ = start(param, new_makdir=new_makdir)
                        result = get_random_result()
                        print("=" * 40)
                        print("true communities：" + str(result.true_community_num))
                        print("find communities: " + str(len(result.center_nodes)))
                        print("=" * 40)
                        step_results.append(result)
                        print("EADP i = {}, onmi={}".format(i, result.onmi))
                        print("EADP i = {}, omega={}".format(i, result.omega))
                        print("EADP i = {}, f1_score={}".format(i, result.f1_score))
                    # 将每一轮结果处理，并存入数据库中，方便后续统计分析,并且里面会对算法的多次求得的结果做一个均值
                    evaluation_result = handle_step_results(param, step_results)
                    # evaluation_result = EvaluationResult(onmi=random.random(), omega=random.random())
                    # 那么这里有可能会存在对多个指标(onmi, omega)进行自动画图
                    # 这里得解决方案：先把onmi,omega拼接在一起吧，并以逗号分隔开，在后面画图的时候再做处理就好
                    y_train_i.append(
                        str(evaluation_result.onmi) + "," + str(
                            evaluation_result.omega) + "," + str(
                            evaluation_result.f1_score))
                    print('-' * 30)
                y_tains.append(y_train_i)
            y_trains_all.append(y_tains)
        print("*" * 30)
        # y_trains_all 表示需要描绘图像的数据
        assert len(y_trains_all) == len(show_image_params)
        for i in range(0, len(y_trains_all)):
            y_tains = y_trains_all[i]
            show_image_params[i].y_trains = y_tains
            for x in y_tains:
                print(x)
            print("------------------------")
        print("*" * 30)

        for show_image_param in show_image_params:
            show_image_param.print_info()
            show_result_image(show_image_param)
            with open(result_image_path + "/pickle.txt", 'ab') as f:
                pickle.dump(show_image_param, f)

    # 删除临时文件夹
    if new_makdir is not None:
        shutil.rmtree(path + new_makdir)
        print("*" * 30)
        print("delete {} success.....".format(path + new_makdir))
        print("*" * 30)


# 这里处理的是多个算法之间的一个对比关系（即是通过一组参数生成了一个网络，然后每种算法都在这个生成的网络上跑一下，记录评估结果）
def deal_contrast_generate_image(new_makdir, param_dict, result_image_path, steps):
    if new_makdir == "muw_contrast":
        control_variables = [0.2, 0.3]
    elif new_makdir == "mut_contrast":
        control_variables = [0.2, 0.3]
    elif new_makdir == "node_contrast":
        control_variables = [1000, 3000]
    else:
        control_variables = []
    print(control_variables, new_makdir)
    for control_variable in control_variables:
        if new_makdir == "muw_contrast":
            params, show_image_params = param_dict.get(new_makdir)(
                muw=control_variable)
        elif new_makdir == "mut_contrast":
            params, show_image_params = param_dict.get(new_makdir)(
                mut=control_variable)
        elif new_makdir == "node_contrast":
            params, show_image_params = param_dict.get(new_makdir)(
                n=control_variable)
        else:
            return
        for show_image_param in show_image_params:
            show_image_param.result_image_path = result_image_path
        assert len(show_image_params) == 1
        show_image_param = show_image_params[0]

        # 各个算法的针对每个
        my_dpc = []  # 我们自己的算法
        cpm = []  # CPM算法
        slpa = []
        lfm_ex = []
        gce = []
        demon = []
        moses = []

        y_trains = []
        for param in params:
            print('-' * 30)
            print("n={}, k={}, maxk={}, minc={}, maxc={}, muw={}, mut={}, on={}, " \
                  "om={}, c={}, node_g_weight={}, u={}, dc_weight={}, tt_weight={}, center_node_r_weight={}".format(
                param.n, param.k,
                param.maxk, param.minc, param.maxc,
                param.muw, param.mut, param.on, param.om, param.c, param.node_g_weight, param.u,
                param.dc_weight,
                param.tt_weight, param.center_node_r_weight))
            step_results = []
            for i in range(0, steps):
                result, _ = start(param, new_makdir=new_makdir)
                step_results.append(result)
                print("EADP i = {}, onmi={}".format(i, result.onmi))
                print("EADP i = {}, omega={}".format(i, result.omega))
                print("EADP i = {}, f1_score={}".format(i, result.f1_score))
                if param.need_calcualte_other_algorithm:  # 讲道理走到这个方法来了，这里一般都为True
                    print("CPM i = {}, onmi={}".format(i, result.CPM_ONMI))
                    print("CPM i = {}, omega={}".format(i, result.CPM_OMEGA))
                    print("CPM i = {}, f1_score={}".format(i, result.CPM_F1_SCORE))
                    print("LFM_EX i = {}, onmi={}".format(i, result.LFM_EX_ONMI))
                    print("LFM_EX i = {}, omega={}".format(i, result.LFM_EX_OMEGA))
                    print("LFM_EX i = {}, f1_score={}".format(i, result.LFM_EX_F1_SCORE))
                    print("SLPA i = {}, onmi={}".format(i, result.SLPA_ONMI))
                    print("SLPA i = {}, omega={}".format(i, result.SLPA_OMEGA))
                    print("SLPA i = {}, f1_score={}".format(i, result.SLPA_F1_SCORE))
                    print("GCE i = {}, onmi={}".format(i, result.GCE_ONMI))
                    print("GCE i = {}, omega={}".format(i, result.GCE_OMEGA))
                    print("GCE i = {}, f1_score={}".format(i, result.GCE_F1_SCORE))
                    print("DEMON i = {}, onmi={}".format(i, result.DEMON_ONMI))
                    print("DEMON i = {}, omega={}".format(i, result.DEMON_OMEGA))
                    print("DEMON i = {}, f1_score={}".format(i, result.DEMON_F1_SCORE))
                    print("MOSES i = {}, onmi={}".format(i, result.MOSES_ONMI))
                    print("MOSES i = {}, omega={}".format(i, result.MOSES_OMEGA))
                    print("MOSES i = {}, f1_score={}".format(i, result.MOSES_F1_SCORE))

            # 将每一轮结果处理，并存入数据库中，方便后续统计分析(todo 目前使用数据库来暂存这些临时数据的没有维护，后面想要统计这些临时的运行返回的各种参数，可以修改这里的代码)
            evaluation_result = handle_step_results(param, step_results)
            my_dpc.append(str(evaluation_result.onmi) + "," + str(
                evaluation_result.omega) + "," + str(evaluation_result.f1_score))
            cpm.append(str(evaluation_result.CPM_ONMI) + "," + str(
                evaluation_result.CPM_OMEGA) + "," + str(
                evaluation_result.CPM_F1_SCORE))
            slpa.append(str(evaluation_result.SLPA_ONMI) + "," + str(
                evaluation_result.SLPA_OMEGA) + "," + str(
                evaluation_result.SLPA_F1_SCORE))
            lfm_ex.append(str(evaluation_result.LFM_EX_ONMI) + "," + str(
                evaluation_result.LFM_EX_OMEGA) + "," + str(
                evaluation_result.LFM_EX_F1_SCORE))
            gce.append(str(evaluation_result.GCE_ONMI) + "," + str(
                evaluation_result.GCE_OMEGA) + "," + str(
                evaluation_result.GCE_F1_SCORE))
            demon.append(str(evaluation_result.DEMON_ONMI) + "," + str(
                evaluation_result.DEMON_OMEGA) + "," + str(
                evaluation_result.DEMON_F1_SCORE))
            moses.append(str(evaluation_result.MOSES_ONMI) + "," + str(
                evaluation_result.MOSES_OMEGA) + "," + str(
                evaluation_result.MOSES_F1_SCORE))
        y_trains.extend([my_dpc, cpm, slpa, lfm_ex, demon, moses])
        for x in y_trains:
            print(x)
        show_image_param.y_trains = y_trains
        show_image_param.print_info()
        with open(result_image_path + "/pickle.txt", 'ab') as f:
            pickle.dump(show_image_param, f)
        show_result_image(show_image_param)

# 这里处理的是多个算法之间的一个对比关系，但是生成的是表格对比而已
# 这个里面包含了两种，1种mut作为控制变量，一种是mut作为控制变量
def deal_contrast_generate_table(new_makdir, steps):
    nodes = [1000]
    control_variables = [0.0, 0.1, 0.2, 0.3] # 表示muw 或者 mut
    params = []
    minc_maxc = 0.04
    for n in nodes:
        for control_variable in control_variables:
            param = AlgorithmParam()
            param.minc = n * minc_maxc
            param.maxc = n * minc_maxc + 20
            if new_makdir == "mut_table":
                param.mut = control_variable
            elif new_makdir == "muw_table":
                param.muw = control_variable
            param.n = n
            param.on = n * param.on_weight
            if n > 1000:
                param.u = 0.3 # 不能那么高
                param.c = 0.8
            else:
                param.u = 0.6
            param.om = 2
            param.k = 20
            params.append(param)
    # 各个算法的针对每个
    my_dpc = []  # 我们自己的算法
    cpm = []  # CPM算法
    slpa = []
    lfm_ex = []
    demon = []
    moses = []
    for param in params:
        print('-' * 30)
        print("n={}, k={}, maxk={}, minc={}, maxc={}, muw={}, mut={}, on={}, " \
              "om={}, c={}, node_g_weight={}, u={}, dc_weight={}, tt_weight={}, center_node_r_weight={}".format(
            param.n, param.k,
            param.maxk, param.minc, param.maxc,
            param.muw, param.mut, param.on, param.om, param.c, param.node_g_weight, param.u,
            param.dc_weight,
            param.tt_weight, param.center_node_r_weight))
        step_results = []
        for i in range(0, steps):
            result, _ = start(param, new_makdir=new_makdir)
            step_results.append(result)
            print("EADP i = {}, onmi={}".format(i, result.onmi))
            print("EADP i = {}, omega={}".format(i, result.omega))
            print("EADP i = {}, f1_score={}".format(i, result.f1_score))
            if param.need_calcualte_other_algorithm:  # 讲道理走到这个方法来了，这里一般都为True
                print("CPM i = {}, onmi={}".format(i, result.CPM_ONMI))
                print("CPM i = {}, omega={}".format(i, result.CPM_OMEGA))
                print("CPM i = {}, f1_score={}".format(i, result.CPM_F1_SCORE))
                print("LFM_EX i = {}, onmi={}".format(i, result.LFM_EX_ONMI))
                print("LFM_EX i = {}, omega={}".format(i, result.LFM_EX_OMEGA))
                print("LFM_EX i = {}, f1_score={}".format(i, result.LFM_EX_F1_SCORE))
                print("SLPA i = {}, onmi={}".format(i, result.SLPA_ONMI))
                print("SLPA i = {}, omega={}".format(i, result.SLPA_OMEGA))
                print("SLPA i = {}, f1_score={}".format(i, result.SLPA_F1_SCORE))
                print("GCE i = {}, onmi={}".format(i, result.GCE_ONMI))
                print("GCE i = {}, omega={}".format(i, result.GCE_OMEGA))
                print("GCE i = {}, f1_score={}".format(i, result.GCE_F1_SCORE))
                print("DEMON i = {}, onmi={}".format(i, result.DEMON_ONMI))
                print("DEMON i = {}, omega={}".format(i, result.DEMON_OMEGA))
                print("DEMON i = {}, f1_score={}".format(i, result.DEMON_F1_SCORE))
                print("MOSES i = {}, onmi={}".format(i, result.MOSES_ONMI))
                print("MOSES i = {}, omega={}".format(i, result.MOSES_OMEGA))
                print("MOSES i = {}, f1_score={}".format(i, result.MOSES_F1_SCORE))

        # 将每一轮结果处理，并存入数据库中，方便后续统计分析(todo 目前使用数据库来暂存这些临时数据的没有维护，后面想要统计这些临时的运行返回的各种参数，可以修改这里的代码)
        evaluation_result = handle_step_results(param, step_results)
        my_dpc.append(str(evaluation_result.onmi) + "," + str(
            evaluation_result.omega) + "," + str(evaluation_result.f1_score))
        cpm.append(str(evaluation_result.CPM_ONMI) + "," + str(
            evaluation_result.CPM_OMEGA) + "," + str(
            evaluation_result.CPM_F1_SCORE))
        slpa.append(str(evaluation_result.SLPA_ONMI) + "," + str(
            evaluation_result.SLPA_OMEGA) + "," + str(
            evaluation_result.SLPA_F1_SCORE))
        lfm_ex.append(str(evaluation_result.LFM_EX_ONMI) + "," + str(
            evaluation_result.LFM_EX_OMEGA) + "," + str(
            evaluation_result.LFM_EX_F1_SCORE))
        demon.append(str(evaluation_result.DEMON_ONMI) + "," + str(
            evaluation_result.DEMON_OMEGA) + "," + str(
            evaluation_result.DEMON_F1_SCORE))
        moses.append(str(evaluation_result.MOSES_ONMI) + "," + str(
            evaluation_result.MOSES_OMEGA) + "," + str(
            evaluation_result.MOSES_F1_SCORE))
    file_path = "./result_images_eadp/" + new_makdir + "/{}.txt".format(new_makdir)
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete {} success....".format(file_path))
    file_handle = open(file_path, mode="w")
    for i in range(len(my_dpc)):
        cpm_results = cpm[i].split(",")
        lfm_ex_results = lfm_ex[i].split(",")
        slpa_results = slpa[i].split(",")
        demon_results = demon[i].split(",")
        moses_results = moses[i].split(",")
        my_dpc_results = my_dpc[i].split(",")
        onmis = [str(cpm_results[0]), str(lfm_ex_results[0]), str(slpa_results[0]),
                 str(demon_results[0]), str(moses_results[0]), str(my_dpc_results[0])]
        omega = [str(cpm_results[1]), str(lfm_ex_results[1]), str(slpa_results[1]),
                 str(demon_results[1]), str(moses_results[1]), str(my_dpc_results[1])]
        f1_score = [str(cpm_results[2]), str(lfm_ex_results[2]), str(slpa_results[2]),
                 str(demon_results[2]), str(moses_results[2]), str(my_dpc_results[2])]
        # if new_makdir == "muw_table":
        #     print "nodes: {}, muw: {}".format(str(params[i].n), str(params[i].muw))
        #     file_handle.write("nodes: {}, muw: {} \n".format(str(params[i].n), str(params[i].muw)))
        # elif new_makdir == "mut_table":
        #     print "nodes: {}, mut: {}".format(str(params[i].n), str(params[i].mut))
        #     file_handle.write("nodes: {}, mut: {} \n".format(str(params[i].n), str(params[i].mut)))
        file_handle.write("nodes: {}, muw: {} , mut: {} \n".format(str(params[i].n), str(params[i].muw), str(params[i].mut)))
        print("ONMI   " + " ".join(onmis))
        print("OMEGA   " + " ".join(omega))
        print("F1-SCORE   " + " ".join(f1_score))
        file_handle.write(" ".join(onmis) + "\n")
        file_handle.write(" ".join(omega) + "\n")
        file_handle.write(" ".join(f1_score) + "\n")
        file_handle.write("------------------------------------------------------\n")


# 将所有的真实数据集跑一遍，得到实验结果并且生成表格，避免一个一个真实的数据集去修改跑实验
def run_true_datasets():
    print("run_true_dataset....")
    knowns = ["karate.gml", "dolphins.gml", "football.gml", "polbooks.gml"]
    # 说明，由于对于known的数据集也需要计算modularity，所以在unknowns的列表里也需要加上那些known的数据集，
    # 因此有了known的数据集也需要拷贝一份到unknown的目录下，这样就避免在处理unknown的时候还考虑数据集是known的情况
    # 并且有的unknown的数据集节点特别大，因此会执行比较长的时间，导致算法卡在哪里
    # unknowns = ["karate.gml", "dolphins.gml", "football.gml", "polbooks.gml",
    #             "power.gml", "netscience.gml", "ca-GrQc.txt", "ego-Facebook.txt", "cond-mat.gml"]
    unknowns = ["ego-Facebook.txt"]
    known_results = []
    unknown_results = []
    knowns = []
    for known in knowns:
        param = AlgorithmParam(true_dataset=True)
        param.need_calcualte_other_algorithm = True
        param.dataset = known
        param.run_ture_dataset = True
        if known.endswith(".gml"):
            param.dataset_type = "gml"
        elif known.endswith(".txt"):
            param.dataset_type = "txt"
        print("start calculate known {} dataset".format(param.dataset))
        result, need_print_result = start(param)
        known_results.append(result)
    # 处理真实数据集的结果
    from my_util import handle_known_results, handle_unknown_results
    need_update_path(None, "./result_images_eadp/true_datasets")
    handle_known_results(knowns, known_results)
    for i in range(len(unknowns)):
        unknown = unknowns[i];
        param = AlgorithmParam(true_dataset=True)
        param.need_calcualte_other_algorithm = True
        param.dataset = unknown
        param.run_ture_dataset = True
        param.is_known_dataset = False
        if unknown.endswith(".gml"):
            param.dataset_type = "gml"
        elif unknown.endswith(".txt"):
            param.dataset_type = "txt"
        print("start calculate unknown {} dataset".format(param.dataset))
        result, need_print_result = start(param)
        unknown_results.append(result)
        # try:
        #     result, need_print_result = start(param)
        #     unknown_results.append(result)
        # except Exception as e:
        #     print "*" * 30
        #     print e
        #     del unknowns[i]
        #     print "*" * 30
    handle_unknown_results(unknowns, unknown_results)

# 测试不同的dc对真实网络的实验结果的影响
def run_test_dc_for_true_dataset():
    print("begin to run_test_dc_for_true_dataset ")
    test_datasets = ["karate.gml", "dolphins.gml", "football.gml", "polbooks.gml"]
    # dc_dict = {"karate.gml" : 0.114910378087, "dolphins.gml": 0.141438033445, "football.gml": 0.289236081449, "polbooks.gml": 0.119185104844}
    dcs = [0.01, 0.05, 0.1, 0.5, 1.0, 2, None]
    # dc_weights = [float(x)/10 for x in range(1, 51)]
    result_dict = {}
    for test_dataset in test_datasets:
        onmi_res = []
        omega_res = []
        f1_score_res = []
        for dc in dcs:
            param = AlgorithmParam(true_dataset=True)
            param.need_calcualte_other_algorithm = False
            param.dataset = test_dataset
            param.run_ture_dataset = True
            if test_dataset.endswith(".gml"):
                param.dataset_type = "gml"
            elif test_dataset.endswith(".txt"):
                param.dataset_type = "txt"
            result, need_print_result = start(param, dc=dc)
            onmi_res.append(str(result.onmi))
            omega_res.append(str(result.omega))
            f1_score_res.append(str(result.f1_score))
        result_dict[test_dataset + "_onmi"] = "  ".join(onmi_res)
        result_dict[test_dataset + "_omega"] = "  ".join(omega_res)
        result_dict[test_dataset + "_f1_score"] = "  ".join(f1_score_res)

    print('---------------------------------------------------------------------------')
    print("Results of accuracy on real-world datasets with different dc.")
    print('---------------------------------------------------------------------------')
    evaluations = ["onmi", "omega", "f1_score"]
    result_image_path = "./result_images_eadp/test_dc_for_true/"
    need_update_path(None, result_image_path)
    file_handle = open(result_image_path + "truedataset_dc.txt", mode="w")
    file_handle.write("Results of accuracy on real-world datasets with different dc.\n")
    file_handle.write("[0.01, 0.05, 0.1, 0.5, 1.0, 2, None]\n")
    for evaluation in evaluations:
        print(evaluation)
        file_handle.write(evaluation + "\n")
        for test_dataset in test_datasets:
            temp = test_dataset + "  " + result_dict[test_dataset + "_" + evaluation]
            print(temp)
            file_handle.write(temp + "\n")
        print()

# 测试不同的dc对lfr生成网络的实验结果的影响
def run_test_dc_for_lfr():
    print("begin to run_test_dc_for_lfr ")
    result_image_path = "./result_images_eadp/test_dc_for_lfr/"
    need_update_path(path + "test_dc/", result_image_path)
    file_handle = open(result_image_path + "lfr_dc.txt", mode="w")
    params = generate_test_lfr_for_dc_params()

    dcs = [0.01, 0.05, 0.1, 0.5, 1.0, 2, None]
    file_handle.write("[0.01, 0.05, 0.1, 0.5, 1.0, 2, None]\n")
    for param in params:
        file_handle.write(str(param) + "\n")
        onmi_result = []
        omega_result = []
        f1_score_result = []
        for dc in dcs:
            result, _ = start(param=param, new_makdir="test_dc", dc=dc)
            onmi_result.append(str(result.onmi))
            omega_result.append(str(result.omega))
            f1_score_result.append(str(result.f1_score))
        print("   ".join(onmi_result))
        print("   ".join(omega_result))
        print("   ".join(f1_score_result))
        file_handle.write("   ".join(onmi_result) + "\n")
        file_handle.write("   ".join(omega_result) + "\n")
        file_handle.write("   ".join(f1_score_result) + "\n")

# 测试候选重叠节点的情况
def run_test_u_for_lfr():
    print("begin to run_test_u_for_lfr ")
    result_image_path = "./result_images_eadp/test_u_for_lfr/"
    need_update_path(path + "test_u/", result_image_path)
    file_handle = open(result_image_path + "lfr_u.txt", mode="w")
    params = generate_test_lfr_for_u_params()
    for param in params:
        file_handle.write(str(param) + "\n")
        result, _ = start(param=param, new_makdir="test_u")
        s = "true: {}, candidate: {}, special_candidate: {}, true&candidate: {}, find: {}, true&find {}"\
            .format(len(result.true_overlapping_nodes),
                    len(result.overlapping_candidates),
                    len(result.special_candidate_nodes),
                    len(set(result.true_overlapping_nodes) & set(result.overlapping_candidates)),
                    len(result.find_overlapping_nodes),
                    len(result.mapping_overlapping_nodes))
        print("param: {}".format(str(param)))
        print(s)
        file_handle.write(s + "\n")
    file_handle.close()

# 测试候选重叠节点分配策略u的情况
def run_test_c_for_lfr():
    print("begin to run_test_c_for_lfr ")
    result_image_path = "./result_images_eadp/test_c_for_lfr/"
    need_update_path(path + "test_c/", result_image_path)
    file_handle = open(result_image_path + "lfr_c.txt", mode="w")
    params = generate_test_lfr_for_c_params()
    for param in params:
        file_handle.write(str(param) + "\n")
        result, _ = start(param=param, new_makdir="test_u")
        s = "true: {}, candidate: {}, special_candidate: {}, true&candidate: {}, find: {}, true&find {}" \
            .format(len(result.true_overlapping_nodes),
                    len(result.overlapping_candidates),
                    len(result.special_candidate_nodes),
                    len(set(result.true_overlapping_nodes) & set(result.overlapping_candidates)),
                    len(result.find_overlapping_nodes),
                    len(result.mapping_overlapping_nodes))
        print("param: {}".format(str(param)))
        print(s)
        file_handle.write(s + "\n")
    file_handle.close()

if __name__ == '__main__':
    #global path
    run_all_true_dataset = False  # 表示自动化运行所有的真实数据集
    run_test_different_dc = False
    run_test_u = False
    run_test_c = False
    print("linux to run start......")
    if run_platform == "linux":  # 表示是跑lfr网络生成数据的实验
        if run_test_different_dc:
            print("begin to run true test dc......")
            # run_test_dc_for_true_dataset()
            run_test_dc_for_lfr()
        #  表示在Linux平台上运行真实数据集
        elif run_all_true_dataset:
            print("begin to run true dataset......")
            run_true_datasets()
        # 表示在Linux平台测试候选重叠节点的参数敏感性分析
        elif run_test_u:
            print("begin to run test u for lfr")
            run_test_u_for_lfr()
        elif run_test_c:
            print("begin to run test c for lfr")
            run_test_c_for_lfr()
        else:
            print("begin to run lfr dataset......")
            steps = 2  # 默认是5
            if len(sys.argv) >= 2:  # 可以直接python2.7 EADP.py test1 test2 ... 这样就避免直接修改源代码
                generate_functions = sys.argv[1:]
            else:
                generate_functions = ["test1"]
            for new_makdir in generate_functions:
                run_linux_generate_picture(steps, new_makdir=new_makdir)
    else:
        param = AlgorithmParam(True)
        param.node_g_weight = 1.0
        param.enveloped_weight = 0.5
        param.dataset = "karate.gml"
        param.dataset_type = "gml"
        param.is_known_dataset = False
        param.need_calcualte_other_algorithm = True
        param.need_show_image = True
        # 如果需要在window平台下运行，lfr生成数据(由于linux平台生成并拷贝到windows下)，将该参数改为True
        run_windows_lfr = False
        # 0.01，0.05，0.1，0.5，1，2
        result, need_print_result = start(param, run_windows_lfr, dc=None)
        # 当然也可以直接在windows上跑，然后将结果存入数据库中，问题就是windows下不好生成lfr的网络数据
        # add_result_to_mysql(param, [result])
        # window下本机测试，直接打印相应的结果就好
        print_result(result, need_print_result)
