from itertools import combinations
import numpy as np


class EQ:
    @staticmethod
    def f(x, pr):
        return 2.0 * pr * x - pr

    @staticmethod
    def logistic(x, p):
        b = 1 + np.exp(-EQ.f(x, p))
        return 1.0 / b

    @staticmethod
    def logweight(i, j, alpha, p):
        return 1.0 / (EQ.logistic(alpha[i], p) * EQ.logistic(alpha[j], p))

    @staticmethod
    def extended_modularity(
        graph, path, new_makdir, true_dataset, weight='weight', p=30, func=None
    ):
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

        if func is None:
            func = EQ.logweight
        q = 0.0
        degrees = dict(graph.degree(weight=weight))
        m = sum(degrees.values())
        n = graph.number_of_nodes()

        alpha = {}
        beta = {}
        for nd in graph.nodes:
            alpha[nd] = 0

        for community in partitions:
            for nd in community:
                if nd not in alpha:
                    alpha[nd] = 0
                alpha[nd] += 1

        for nd in alpha:
            beta[nd] = alpha[nd] if alpha[nd] != 0 else 0

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
                    beta_out = beta_out + func(nd1, j, beta, p)
                beta_out = beta_out / m

                beta_in = 0.0
                for i in graph.nodes:
                    if i in community:
                        continue
                    if i == nd1:
                        continue
                    if i == nd2:
                        continue
                    beta_in = beta_in + func(i, nd2, beta, p)
                beta_in = beta_in / m

                q = (
                    q
                    + func(nd1, nd2, beta, p) * wt
                    - float(beta_in * beta_out * degrees[nd1] * degrees[nd2] / m)
                )

        print("q{}".format(q))
        print("m{}".format(m))
        return q / m
