# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import time
from functools import wraps

##############################################################################
# 以下的代码是协助我们对网络的数据进行分析，避免每次重复去查找网络图
# 如从一个xxx.gml文件中分析得到节点的社区划分情况
##############################################################################


# 得到每个社区对应的节点集合
def get_community_with_nodes(G):
    node_groups = nx.get_node_attributes(G, 'value')
    comunity_node_dict = {}
    for node, comunity in list(node_groups.items()):
        if comunity in comunity_node_dict:
            comunity_node_dict.get(comunity).append(node)
        else:
            temp = []
            temp.append(node)
            comunity_node_dict[comunity] = temp
    return comunity_node_dict


# 得到每个节点的邻局节点的信息
def get_node_neighbors(community_a=[], community_b=[]):
    node_neighbors_dict = {}
    for node in community_a:
        neighbors = list(nx.neighbors(G, node))
        node_neighbors_dict[node] = neighbors
        sum_self = 0
        sum_another = 0
        for x in neighbors:
            if x in community_a:
                sum_self += 1
            if x in community_b:
                sum_another += 1
        if len(community_b) == 0:
            print(
                "节点： "
                + str(node)
                + " 邻居节点："
                + str(neighbors)
                + " "
                + str(len(neighbors))
                + " "
                + str(sum_self)
            )
        else:
            print(
                "节点："
                + str(node)
                + " 与自己社区相联系的节点个数: "
                + str(sum_self)
                + "/"
                + str(len(neighbors))
                + "  与b社区相联的节点个数： "
                + str(sum_another)
                + "/"
                + str(len(neighbors))
            )
    return node_neighbors_dict


# 得到每个节点的度，并且是否需要排序打印
def get_node_degress(nodes=[], is_need_sort=True):
    res = []
    for node in nodes:
        t = (node, nx.degree(G, node))
        res.append(t)
        if not is_need_sort:
            print("节点：" + str(t[0]) + "  度：" + str(t[1]))
    if is_need_sort:
        res = sorted(res, key=lambda x: x[1])
        for info in res:
            print("节点：" + str(info[0]) + "  度：" + str(info[1]))


def add_douhao(str=""):
    res = []
    for x in str.split(" "):
        res.append(int(x))
    return res


def sub_douhao(nodes=[]):
    res = ""
    for node in nodes:
        res = res + str(node) + " "
    return res


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


if __name__ == '__main__':
    G = nx.read_gml("./datasets/known/football.gml", label="id")
    comunity_node_dict = get_community_with_nodes(G)
    for community_number, comunity_nodes in list(comunity_node_dict.items()):
        print(sub_douhao(list(comunity_nodes)))

    print("=" * 30)
    G = nx.read_gml("./datasets/unknown/power.gml", label="id")
    print(len(list(G.nodes)))
    print(len(list(G.edges)))

    print("=" * 30)
    G = nx.read_gml("./datasets/unknown/netscience.gml", label="id")
    print(len(list(G.nodes)))
    print(len(list(G.edges)))

    print("=" * 30)
    G = nx.read_gml("./datasets/unknown/cond-mat.gml", label="id")
    print(len(list(G.nodes)))
    print(len(list(G.edges)))

    print("=" * 30)
    G = nx.read_edgelist("./datasets/unknown/ego-Facebook.txt", delimiter=" ")
    print(len(list(G.nodes)))
    print(len(list(G.edges)))

    print("=" * 30)
    G_temp = nx.read_edgelist("./datasets/unknown/ca-GrQc.txt", delimiter=" ")
    print(len(list(G_temp.nodes)))
    print(len(list(G_temp.edges)))
    G = rebalace_G_txt(G_temp)
    print(len(list(G.nodes)))
    print(len(list(G.edges)))

    G = nx.read_gml("./datasets/known/dolphins.gml", label="id")
    for edge in G.edges:
        print(str(edge[0]) + "\t" + str(edge[1]))
