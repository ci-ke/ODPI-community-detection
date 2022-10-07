# -*- coding: utf-8 -*-#

import os
import subprocess
import time

from my_objects import AlgorithmParam
import networkx as nx
from collections import defaultdict
import math


def generate_network(param, path, new_makdir):
    # community_path = "/app/datasets/community.nmc"
    community_path = path + new_makdir + "/community.nmc"
    if os.path.exists(community_path):
        os.remove(community_path)
        print('*' * 30)
        print("delete community.nmc success...")
        print('*' * 30)
    # networkx_path = "/app/datasets/community.nse"
    networkx_path = path + new_makdir + "/community.nse"
    if os.path.exists(networkx_path):
        print('*' * 30)
        print("delete community.nse success...")
        print('*' * 30)
        os.remove(networkx_path)
    assert isinstance(param, AlgorithmParam)
    n = param.n
    k = param.k
    maxk = param.maxk
    minc = param.minc
    maxc = param.maxc
    mut = param.mut
    muw = param.muw
    on = param.on
    om = param.om
    args = "-N {n} -k {k} -maxk {maxk} -minc {minc} -maxc {maxc} -mut {mut} -muw {muw} -on {on}  -om {om} -name {name}".format(
        n=n,
        k=k,
        maxk=maxk,
        minc=minc,
        maxc=maxc,
        mut=mut,
        muw=muw,
        on=on,
        om=om,
        name=path + new_makdir + "/community",
    )
    print("begin generate network")
    subprocess.getoutput("{path}/benchmark {args}".format(path=path, args=args))
    while not os.path.exists(community_path):
        time.sleep(1)
        print("=" * 30)
        print("{path}/benchmark {args}".format(path=path, args=args))
        print("=" * 30)
    print("generate network success again...")


def calculate_onmi(path, new_makdir, true_dataset=None):
    if true_dataset is None:
        lfr_code = path + new_makdir + "/lfr_code.txt"
        lfr_true = path + new_makdir + "/lfr_true.txt"
    else:
        lfr_code = path + new_makdir + "/" + true_dataset + "_code.txt"
        lfr_true = path + new_makdir + "/" + true_dataset + "_true.txt"
    if not os.path.exists(lfr_true) or not os.path.exists(lfr_code):
        raise Exception("计算ONMI的时候，lfr_true.txt和lfr_code.txt文件不能为空!")
    res = subprocess.getoutput("{}/onmi/onmi {} {}".format(path, lfr_code, lfr_true))
    lines = res.splitlines(True)
    # 取得到的ONMI最大的结果
    onmis = []
    for line in lines:
        if line.strip().__contains__("NMI"):
            onmi = float(line.strip().split("\t")[1])
            onmis.append(onmi)
    return float(max(onmis))


def calculate_omega(path, new_makdir, true_dataset=None):
    if true_dataset is None:
        lfr_code = path + new_makdir + "/lfr_code.txt"
        lfr_true = path + new_makdir + "/lfr_true.txt"
    else:
        lfr_code = path + new_makdir + "/" + true_dataset + "_code.txt"
        lfr_true = path + new_makdir + "/" + true_dataset + "_true.txt"
    res = subprocess.getoutput(
        "java -classpath {}/omega/ OmegaIndex {} {}".format(path, lfr_code, lfr_true)
    )
    lines = res.splitlines(True)
    return float(lines[0].strip())


if __name__ == '__main__':
    from my_analysis_dataset import add_douhao

    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 3),
            (4, 6),
            (4, 8),
            (6, 7),
            (9, 11),
            (10, 13),
            (1, 2),
            (1, 3),
            (2, 4),
            (4, 5),
            (4, 7),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 8),
            (7, 8),
            (7, 10),
            (8, 9),
            (9, 10),
            (9, 12),
            (9, 13),
            (10, 11),
            (10, 12),
            (11, 12),
            (11, 13),
            (12, 13),
        ]
    )
    paration = [{0, 1, 2, 3}, {4, 5, 6, 7, 8}, {9, 10, 11, 12, 13}]

    G = nx.read_gml("./datasets/known/karate.gml", label="id")
    true_paration = [
        {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22},
        {9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34},
    ]

    G = nx.read_gml("./datasets/known/karate.gml", label="id")
    for edge in G.edges:
        print(edge[0], edge[1])
    a = "1 2 3 4 5 6 7 8 10 11 12 13 14 17 18 20 22"
    b = "9 10 15 16 19 21 23 24 25 26 27 28 29 30 31 32 33 34"
    c = add_douhao(a)
    d = add_douhao(b)
    paration = [c, d]
    # print cal_f_score(paration, paration)
