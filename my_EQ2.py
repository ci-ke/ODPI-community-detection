import numpy as np
from itertools import combinations


def EQ_Newman(graph, path, new_makdir, true_dataset, weight='weight'):
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
    q = 0.0
    degrees = dict(graph.degree(weight=weight))

    U = np.zeros((graph.number_of_nodes(), len(partitions)))
    for k, nds in enumerate(partitions):
        U[[int(n) - 1 for n in nds], k] = 1
    U = U / U.sum(1, keepdims=True)
    m = np.sum([v for k, v in list(degrees.items())])

    for nd1, nd2 in combinations(graph.nodes, 2):
        if graph.has_edge(nd1, nd2):
            e = graph[nd1][nd2]
            wt = e.get(weight, 1)
        else:
            wt = 0

        q += (wt - degrees[nd1] * degrees[nd2] / (2 * m)) * np.dot(
            U[int(nd1) - 1], U[int(nd2) - 1]
        )
    # print "Qov{}".format(q / (2*m))
    return q / (2 * m)
