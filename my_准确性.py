# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2021/5/13
# -------------------------------------------------------------------------------
# -*- coding:utf-8 -*-
import sys
import importlib

importlib.reload(sys)
sys.setdefaultencoding('utf-8')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文


# ONMI的
CPM = [0.9851, 0.6236, 0.1761, 0.1761]
LFM = [0.8000, 0.7190, 0.6133, 0.4677]
SLPA = [0.9810, 0.9559, 0.9188, 0.7669]
DEMON = [0.9664, 0.8554, 0.7119, 0.5882]
MOSES = [0.8796, 0.8912, 0.8762, 0.8305]
EADP = [0.9449, 0.8927, 0.9571, 0.7992]
EDPC = [0.9838, 0.9378, 0.9217, 0.8661]
x = np.arange(4)
total_width, n = 0.6, 6
width = total_width / n
x = x - (total_width - width) / 2
jianju = width + 0.03
plt.bar(x, CPM, color='orange', width=width, label='CPM ')
plt.bar(x + jianju, LFM, color='chartreuse', width=width, label='LFM')
plt.bar(x + 2 * jianju, SLPA, color="indigo", width=width, label='SLPA')
plt.bar(x + 3 * jianju, DEMON, color="deepskyblue", width=width, label='DEMON')
plt.bar(x + 4 * jianju, MOSES, color="g", width=width, label='MOSES')
plt.bar(x + 5 * jianju, EADP, color="yellow", width=width, label='EADP')
plt.bar(x + 6 * jianju, ODPI, color="red", width=width, label='ODPI')
plt.xlabel("mut")
plt.ylabel("ONMI")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc=3,
    ncol=6,
    mode="expand",
    borderaxespad=0.0,
)
plt.xticks([0, 1, 2, 3], ['0.0', '0.1', '0.2', '0.3'])
plt.ylim(0, 1)
plt.savefig('./a')
plt.show()

# OMEGA
CPM = [0.9821, 0.2905, 0.0330, 0.0030]
LFM = [0.9003, 0.8444, 0.7565, 0.5730]
SLPA = [0.9920, 0.9350, 0.9039, 0.6463]
DEMON = [0.8744, 0.6961, 0.3172, 0.0360]
MOSES = [0.9327, 0.9256, 0.9246, 0.9261]
EADP = [0.9665, 0.9177, 0.8927, 0.8582]
EDPC = [0.9872, 0.9565, 0.9457, 0.9103]
x = np.arange(4)
total_width, n = 0.6, 6
width = total_width / n
x = x - (total_width - width) / 2
jianju = width + 0.03
plt.bar(x, CPM, color='orange', width=width, label='CPM ')
plt.bar(x + jianju, LFM, color='chartreuse', width=width, label='LFM')
plt.bar(x + 2 * jianju, SLPA, color="indigo", width=width, label='SLPA')
plt.bar(x + 3 * jianju, DEMON, color="deepskyblue", width=width, label='DEMON')
plt.bar(x + 4 * jianju, MOSES, color="g", width=width, label='MOSES')
plt.bar(x + 5 * jianju, EADP, color="yellow", width=width, label='EADP')
plt.bar(x + 6 * jianju, ODPI, color="red", width=width, label='ODPI')
plt.xlabel("mut")
plt.ylabel(r'$\Omega$ Index')
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc=3,
    ncol=6,
    mode="expand",
    borderaxespad=0.0,
)
plt.xticks([0, 1, 2, 3], ['0.0', '0.1', '0.2', '0.3'])
plt.ylim(0, 1)
plt.savefig('./b')
plt.show()


# F-Score
CPM = [0.9247, 0.3912, 0.0393, 0.0038]
LFM = [0.1045, 0.0530, 0.0370, 0.03890]
SLPA = [0.9506, 0.8788, 0.7837, 0.5081]
DEMON = [0.7143, 0.5356, 0.2526, 0.0591]
MOSES = [0.3580, 0.4396, 0.4670, 0.5415]
EADP = [0.8104, 0.7942, 0.7598, 0.5555]
EDPC = [0.9577, 0.8209, 0.7558, 0.6958]
x = np.arange(4)
total_width, n = 0.6, 6
width = total_width / n
x = x - (total_width - width) / 2
jianju = width + 0.03
plt.bar(x, CPM, color='orange', width=width, label='CPM ')
plt.bar(x + jianju, LFM, color='chartreuse', width=width, label='LFM')
plt.bar(x + 2 * jianju, SLPA, color="indigo", width=width, label='SLPA')
plt.bar(x + 3 * jianju, DEMON, color="deepskyblue", width=width, label='DEMON')
plt.bar(x + 4 * jianju, MOSES, color="g", width=width, label='MOSES')
plt.bar(x + 5 * jianju, EADP, color="yellow", width=width, label='EADP')
plt.bar(x + 6 * jianju, ODPI, color="red", width=width, label='ODPI')
plt.xlabel("mut")
plt.ylabel("F-Score")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc=3,
    ncol=6,
    mode="expand",
    borderaxespad=0.0,
)
plt.xticks([0, 1, 2, 3], ['0.0', '0.1', '0.2', '0.3'])
plt.ylim(0, 1)
plt.savefig('./c')
plt.show()
