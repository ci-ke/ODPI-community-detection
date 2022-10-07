# -*- coding: utf-8 -*-#

import subprocess
import os

# 调用GCE算法
def calculate_moses(path="", new_makdir='', G=None, true_dataset=None):
    # 1) 先生成MOSES算法需要的网络边信息的csv文件
    if G is None:
        raise Exception("调用MOSES算法，G不能为空")
    file_input = path + new_makdir + "/moses_edge.txt"
    if os.path.exists(file_input):
        os.remove(file_input)
        print("delete {} success....".format(file_input))
        # os.mkdir(file_input)
    import networkx as nx

    nx.write_edgelist(G, file_input, delimiter='\t')
    print("generate {} success...".format(file_input))

    # 2）调用MOSES算法，由于MOSES算法是直接将结果输出到指定的输出文件中，所以得到结果直接用于ONMI的计算等
    if true_dataset is None:
        # 生产lfr_code.txt 用于最后计算ONMI的值
        file_output = path + new_makdir + "/lfr_code.txt"
    else:
        file_output = path + new_makdir + "/" + true_dataset + "_code.txt"
    if os.path.exists(file_output):
        os.remove(file_output)
        print("delete lfr_code.txt success....")
    subprocess.getoutput("{}moses/moses {} {}".format(path, file_input, file_output))
