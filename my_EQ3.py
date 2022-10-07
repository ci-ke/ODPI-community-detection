from itertools import combinations
import collections
import numpy as np


def extended_modularity(graph, path, new_makdir, true_dataset, weight='weight', p=0.2):

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

    print(partitions)

    q = 0.0
    degrees = dict(graph.degree(weight=weight))
    m = sum(degrees.values())
    n = graph.number_of_nodes()
    alpha = {}

    for community in partitions:
        for nd in community:
            alpha[nd] = alpha.get(nd, 0) + 1

    def F(i, j, pr=p):
        return 1.0 / (
            (1.0 + np.exp(-(pr * (2.0 / alpha[i] - 1.0))))
            * (1.0 + np.exp(-(pr * (2.0 / alpha[j] - 1.0))))
        )

    for nd1, nd2 in combinations(graph.nodes, 2):
        if graph.has_edge(nd1, nd2):
            e = graph[nd1][nd2]
            wt = e.get(weight, 1)
        else:
            wt = 0
        for community in partitions:
            beta_out = 0.0
            for j in graph.nodes:
                if j in community:
                    continue
                if j == nd1:
                    continue
                if j == nd2:
                    continue
                beta_out = beta_out + F(nd1, j)
            beta_out = beta_out / n

            beta_in = 0.0
            for i in graph.nodes:
                if i in community:
                    continue
                if i == nd1:
                    continue
                if i == nd2:
                    continue
                beta_in = beta_in + F(i, nd2)
            beta_in = beta_in / n
        q = (
            q
            + F(nd1, nd2) * wt
            - float(beta_in * beta_out * degrees[nd1] * degrees[nd2] / m)
        )

    return q / m
    q = (
        q
        + F(nd1, nd2) * wt
        - float(beta_in * beta_out * degrees[nd1] * degrees[nd2] / m)
    )
    return q / m
