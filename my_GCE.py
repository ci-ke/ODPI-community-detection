# -*- coding: utf-8 -*-#

import subprocess
import os

# 调用GCE算法
def calculate_gce(path="", new_makdir='', G=None, true_dataset=None):
    # 1) 先生成GCE算法需要的网络边信息的csv文件
    if G is None:
        raise Exception("调用GCE算法，G不能为空")
    file_input = path + new_makdir + "/graph_input.csv"
    if os.path.exists(file_input):
        os.remove(file_input)
        print("delete {} success....".format(file_input))
        # os.mkdir(file_input)
    # 自己写入不知道为什么会是的commnads.getoutput不能得到正确的结果
    # file_input_handle = open(file_input, mode="w")
    # for edge in G.edges:
    #   file_input_handle.write(str(edge[0]) + " " + str(edge[1]) + "\n")
    import networkx as nx

    nx.write_edgelist(G, file_input)
    print("generate {} success...".format(file_input))

    # 2）调用GCE算法，得到结果，并将结果处理写入到lfr_code.txt文件中
    res = subprocess.getoutput("{}gce/gce {}".format(path, file_input))
    if not res.__contains__("Finished"):
        raise Exception("调用GCE算法出错")
    lines = res.splitlines(True)
    start = False
    if true_dataset is None:
        # 生产lfr_code.txt 用于最后计算ONMI的值
        file_path = path + new_makdir + "/lfr_code.txt"
    else:
        file_path = path + new_makdir + "/" + true_dataset + "_code.txt"
    if os.path.exists(file_path):
        os.remove(file_path)
        print("delete lfr_code.txt success....")
    file_handle = open(file_path, mode="w")
    for line in lines:
        if start and len(line.strip()) > 0:
            if not line.endswith("\n"):
                line = line + "\n"
            file_handle.write(line)
        if line.startswith("Finished"):
            start = True
    file_handle.flush()
    file_handle.close()
