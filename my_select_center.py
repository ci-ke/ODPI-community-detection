# -*- coding: utf-8 -*-#

import numpy as np
from sklearn.cluster import KMeans

# data = np.array([0.8999, 0.1098, 0.1284, 0.1606, 0.0780, 0.1207, 0.0672, 1.0000, 0.0806, 0.0936]).reshape((10, 1))
# k = 3
# estimator = KMeans(n_clusters=k)
# estimator.fit(data)
# label_pred = estimator.labels_
#
# a = []
# b = []
# c = []
# for i in range(len(data)):
#      if label_pred[i] == 0:
#           a.append(data[i][0])
#      elif label_pred[i] == 1:
#           b.append(data[i][0])
#      else:
#           c.append(data[i][0])
# print a
# print b
# print c

# 通过kmeans算法进行二分或者三分聚类
def select_kmeans(node_info_list, k=2):
    bad_list = []
    midle_list = []
    best_list = []

    # 使用kmeans进行聚类
    all_node_r = [node_info.node_r for node_info in node_info_list]
    data = np.array(all_node_r).reshape((len(all_node_r), 1))
    estimator = KMeans(n_clusters=k)
    estimator.fit(data)
    label_pred = estimator.labels_
    a = []
    a_1 = []
    b = []
    b_1 = []
    c = []
    c_1 = []
    if k == 2:
        for i in range(len(node_info_list)):
            node_info = node_info_list[i]
            if label_pred[i] == 1:
                a.append(node_info.node_r)
                a_1.append(node_info)
            else:
                b.append(node_info.node_r)
                b_1.append(node_info)
        if max(a) < max(b):
            bad_list = a_1
            best_list = b_1
        else:
            bad_list = b_1
            best_list = a_1
    elif k == 3:
        for i in range(len(node_info_list)):
            node_info = node_info_list[i]
            node_label = node_info.node
            if label_pred[i] == 0:
                a.append(node_info.node_r)
                a_1.append(node_info)
            elif label_pred[i] == 1:
                b.append(node_info.node_r)
                b_1.append(node_info)
            else:
                c.append(node_info.node_r)
                c_1.append(node_info)
        if max(a) > max(b):
            if max(b) > max(c):
                best_list = a_1
                midle_list = b_1
                bad_list = c_1
            else:
                if max(a) > max(c):
                    best_list = a_1
                    midle_list = c_1
                    bad_list = b_1
                else:
                    best_list = c_1
                    midle_list = a_1
                    bad_list = b_1
        else:
            if max(b) < max(c):
                best_list = c_1
                midle_list = b_1
                bad_list = a_1
            else:
                if max(c) > max(a):
                    best_list = b_1
                    midle_list = c_1
                    bad_list = a_1
                else:
                    best_list = b_1
                    midle_list = a_1
                    bad_list = c_1
    else:
        raise Exception("目前算法只定义了二分或者三分算法")
    return bad_list, midle_list, best_list
