# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2020/12/29
# -------------------------------------------------------------------------------

# 代码来源 https://github.com/RapidsAtHKUST/CommunityDetectionCodes/blob/master/Metrics/metrics/link_belong_modularity.py
import math

import numpy as np
import networkx.algorithms.community as nx_comm
from functools import reduce


class FuncTag:
    def __init__(self):
        pass

    exp_inv_mul_tag = 'exp_inv_mul'
    mul_tag = 'mul'
    min_tag = 'min'
    max_tag = 'max'


def get_coefficient_func(tag):
    if tag == FuncTag.exp_inv_mul_tag:
        return lambda l, r: 1.0 / reduce(
            lambda il, ir: il * ir,
            [1.0 + math.exp(0.8 - 1.5 * ele) for ele in [l, r]],
            1,
        )

    elif tag == FuncTag.mul_tag:
        return lambda l, r: l * r
    elif tag == FuncTag.min_tag:
        return min
    elif tag == FuncTag.max_tag:
        return max


def cal_modularity(G, path, new_makdir, true_dataset):
    if true_dataset is None:
        lfr_code = path + new_makdir + "/lfr_code.txt"
    else:
        lfr_code = path + new_makdir + "/" + true_dataset + "_code.txt"
    file_handle = open(lfr_code, "r")
    lines = file_handle.readlines()
    partitions = []
    for line in lines:
        partition = [int(node) for node in line.strip().split(" ")]
        partitions.append(partition)

    file_handle.close()
    # modularity = nx_comm.modularity(G, partitions)
    # return modularity
    return LinkBelongModularity(
        G, partitions, get_coefficient_func(FuncTag.exp_inv_mul_tag)
    ).calculate_modularity()


class LinkBelongModularity:
    PRECISION = 0.0001

    def __init__(self, input_graph, comm_result, coefficient_func):
        """
        :type input_graph: nx.Graph
        """
        self.comm_list = comm_result
        self.graph = input_graph
        self.coefficient_func = coefficient_func
        self.belong_weight_dict = {}
        self.in_degree_dict = {}
        self.out_degree_dict = {}

        def init_belong_weight_dict():
            belong_dict = {}
            for comm in comm_result:
                for mem in comm:
                    if mem not in belong_dict:
                        belong_dict[mem] = 0
                    belong_dict[mem] += 1
            for mem in belong_dict:
                self.belong_weight_dict[mem] = (
                    1.0 / belong_dict[mem] if belong_dict[mem] != 0 else 0
                )

        def init_degree_dicts():
            for vertex in self.graph.nodes():
                # since graph here studied are used in undirected manner
                self.in_degree_dict[vertex] = self.graph.degree(vertex)
                self.out_degree_dict[vertex] = self.graph.degree(vertex)
            return

        init_belong_weight_dict()
        init_degree_dicts()

    def calculate_modularity(self):
        modularity_val = 0
        vertex_num = self.graph.number_of_nodes()
        edge_num = self.graph.number_of_edges()
        for comm in self.comm_list:
            comm_size = len(comm)
            f_val_matrix = np.ndarray(shape=(comm_size, comm_size), dtype=float)
            f_val_matrix.fill(0)
            f_sum_in_vec = np.zeros(comm_size, dtype=float)
            f_sum_out_vec = np.zeros(comm_size, dtype=float)
            in_deg_vec = np.zeros(comm_size, dtype=float)
            out_deg_vec = np.zeros(comm_size, dtype=float)

            # calculate f_val_matrix, f_sum_in, f_sum_out
            for i in range(comm_size):
                src_mem = comm[i]
                in_deg_vec[i] = self.in_degree_dict[src_mem]
                out_deg_vec[i] = self.out_degree_dict[src_mem]
                for j in range(comm_size):
                    dst_mem = comm[j]
                    if i != j and self.graph.has_edge(src_mem, dst_mem):
                        f_val_matrix[i][j] = self.coefficient_func(
                            self.belong_weight_dict[src_mem],
                            self.belong_weight_dict[dst_mem],
                        )
                        f_sum_out_vec[i] += f_val_matrix[i][j]
                        f_sum_in_vec[j] += f_val_matrix[i][j]

            f_sum_in_vec /= vertex_num
            f_sum_out_vec /= vertex_num

            for i in range(comm_size):
                for j in range(comm_size):
                    if i != j and f_val_matrix[i][j] > LinkBelongModularity.PRECISION:
                        null_model_val = (
                            out_deg_vec[i]
                            * in_deg_vec[j]
                            * f_sum_out_vec[i]
                            * f_sum_in_vec[j]
                            / edge_num
                        )
                        modularity_val += f_val_matrix[i][j] - null_model_val
        modularity_val /= edge_num
        return float('%.6f' % float(modularity_val))


if __name__ == '__main__':
    import networkx as nx

    G = nx.read_gml("./datasets/known/karate.gml", label="id")
    # 1 2 3 4 5 6 7 8 9 11 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
    # 1 2 5 6 7 11 17 18 20 22
    # 25 26 32
    #
    # 4 8 9 10 13 14 15 16 19 21 23 26 27 28 29 31 33
    # 1 2 3 5 6 7 11 12 17 18 20 22 24 25 30 32 34
    #
    # 20 14 3 4 2 8 13 1 18 5 6 7 11 22 12 17
    # 16 27 15 23 10 30 34 31 21 33 9 19 29 24 28 32 26 25
    #
    line1 = "20 14 3 4 2 8 13 1 18 5 6 7 11 22 12 17"
    line2 = "16 27 15 23 10 30 34 31 21 33 9 19 29 24 28 32 26 25"
    lines = [line1, line2]
    partitions = []
    for line in lines:
        partition = [int(node) for node in line.split(" ")]
        partitions.append(partition)
    print(nx_comm.modularity(G, partitions))
