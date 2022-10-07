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
dc_1 = [0.0500, 0.1698, 0.1671, 0.0100, 0.0003, 0.0642]
dc_2 = [0.7973, 0.1308, 0.0003, 0.5400, 0.2356, 0.2071]
dc_3 = [0.7973, 0.2495, 0.7329, 0.5400, 0.7485, 0.7498]
dc_4 = [0.3645, 0.5661, 0.7329, 0.5400, 0.9674, 0.9629]
dc_5 = [0.3645, 0.5667, 0.7329, 0.5400, 0.9658, 0.9773]
dc_6 = [0.8345, 0.4857, 0.7329, 0.5400, 0.9697, 0.9657]
dc_7 = [0.7973, 0.7224, 0.7329, 0.5400, 0.9537, 0.9814]
x = np.arange(6)
total_width, n = 0.6, 7
width = total_width / n
x = x - (total_width - width) / 2
jianju = width + 0.03
plt.bar(x, dc_1, color='orange', width=width, label='0.01 ')
plt.bar(x + jianju, dc_2, color='chartreuse', width=width, label='0.05')
plt.bar(x + 2 * jianju, dc_3, color="indigo", width=width, label='0.1')
plt.bar(x + 3 * jianju, dc_4, color="deepskyblue", width=width, label='0.5')
plt.bar(x + 4 * jianju, dc_5, color="g", width=width, label='1.0')
plt.bar(x + 5 * jianju, dc_6, color="y", width=width, label='2.0')
plt.bar(x + 6 * jianju, dc_7, color="red", width=width, label='Adaptive')
plt.xlabel(r"$d_c$")
plt.ylabel("ONMI")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc=3,
    ncol=6,
    mode="expand",
    borderaxespad=0.0,
)
plt.xticks(
    [0, 1, 2, 3, 4, 5], ['Dolphin', 'Football', 'Karate', 'Political', 'LFR-1', 'LFR-2']
)
plt.ylim(0, 1)
plt.savefig('./a')
plt.show()

# OMEGA
dc_1 = [0.0003, 0.0652, 0.1972, 0.0003, 0.0003, 0.0447]
dc_2 = [0.8731, 0.1452, 0.0003, 0.6697, 0.1888, 0.1543]
dc_3 = [0.8731, 0.2308, 0.7716, 0.6697, 0.7528, 0.7339]
dc_4 = [0.3631, 0.5788, 0.7716, 0.6697, 0.9722, 0.9700]
dc_5 = [0.3631, 0.5855, 0.7716, 0.6697, 0.9688, 0.9727]
dc_6 = [0.8935, 0.5106, 0.7716, 0.6697, 0.9751, 0.9689]
dc_7 = [0.8731, 0.7357, 0.7716, 0.6697, 0.9578, 0.9836]
x = np.arange(6)
total_width, n = 0.6, 7
width = total_width / n
x = x - (total_width - width) / 2
jianju = width + 0.03
plt.bar(x, dc_1, color='orange', width=width, label='0.01 ')
plt.bar(x + jianju, dc_2, color='chartreuse', width=width, label='0.05')
plt.bar(x + 2 * jianju, dc_3, color="indigo", width=width, label='0.1')
plt.bar(x + 3 * jianju, dc_4, color="deepskyblue", width=width, label='0.5')
plt.bar(x + 4 * jianju, dc_5, color="g", width=width, label='1.0')
plt.bar(x + 5 * jianju, dc_6, color="y", width=width, label='2.0')
plt.bar(x + 6 * jianju, dc_7, color="red", width=width, label='Adaptive')
plt.xlabel(r"$d_c$")
plt.ylabel(r'$\Omega$ Index')
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc=3,
    ncol=6,
    mode="expand",
    borderaxespad=0.0,
)
plt.xticks(
    [0, 1, 2, 3, 4, 5], ['Dolphin', 'Football', 'Karate', 'Political', 'LFR-1', 'LFR-2']
)
plt.ylim(0, 1)
plt.savefig('./b')
plt.show()

# F-Score的
dc_1 = [0.4050, 0.1966, 0.7350, 0.2133, 0.0070, 0.0177]
dc_2 = [0.9550, 0.1416, 0.3450, 0.5966, 0.1961, 0.1590]
dc_3 = [0.9700, 0.2758, 0.9400, 0.6033, 0.7552, 0.7580]
dc_4 = [0.2937, 0.5458, 0.9400, 0.5966, 0.9614, 0.9657]
dc_5 = [0.2937, 0.5416, 0.9400, 0.5966, 0.9652, 0.9633]
dc_6 = [0.9850, 0.4258, 0.9400, 0.5966, 0.9628, 0.9655]
dc_7 = [0.9700, 0.7475, 0.9400, 0.6033, 0.9625, 0.9500]
x = np.arange(6)
total_width, n = 0.6, 7
width = total_width / n
x = x - (total_width - width) / 2
jianju = width + 0.03
plt.bar(x, dc_1, color='orange', width=width, label='0.01 ')
plt.bar(x + jianju, dc_2, color='chartreuse', width=width, label='0.05')
plt.bar(x + 2 * jianju, dc_3, color="indigo", width=width, label='0.1')
plt.bar(x + 3 * jianju, dc_4, color="deepskyblue", width=width, label='0.5')
plt.bar(x + 4 * jianju, dc_5, color="g", width=width, label='1.0')
plt.bar(x + 5 * jianju, dc_6, color="y", width=width, label='2.0')
plt.bar(x + 6 * jianju, dc_7, color="red", width=width, label='Adaptive')
plt.xlabel(r"$d_c$")
plt.ylabel("F-Score")
plt.legend(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
    loc=3,
    ncol=6,
    mode="expand",
    borderaxespad=0.0,
)
plt.xticks(
    [0, 1, 2, 3, 4, 5], ['Dolphin', 'Football', 'Karate', 'Political', 'LFR-1', 'LFR-2']
)
plt.ylim(0, 1)
plt.savefig('./c')
plt.show()
