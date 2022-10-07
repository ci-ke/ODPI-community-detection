# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Author:       liuligang
# Date:         2021/5/14
# -------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0, 2 * (np.pi))  # numpy.linspace(开始，终值(含终值))，个数)

y1 = [math.exp(-(math.pow(t, 2) / 1)) for t in x]
y2 = [math.exp(-(math.pow(t, 2) / 2)) for t in x]
y3 = [math.exp(-t) for t in x]

# 画图
# plt.title('Compare cosx with sinx')  #标题
# plt.plot(x,y)
# 常见线的属性有：color,label,linewidth,linestyle,marker等
plt.plot(x, y1, color='red', label=r'$exp(-(\frac{d_{ij}} {d_c})^2)\{{d_c} = 1\}$')
plt.plot(
    x, y2, 'lime', label=r'$exp(-(\frac{d_{ij}} {d_c})^2)\{{d_c} = \sqrt{2}\}$'
)  #'b'指：color='blue'
plt.plot(x, y3, 'blue', label='$exp(-d_{ij})$')
plt.legend()  # 显示上面的label
plt.xlabel(r'${d_{ij}}$')
plt.ylabel(r'$f({d_{ij}})$')
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#            ncol=6, mode="expand", borderaxespad=0.)
plt.axis([0, 5, 0, 1])  # 设置坐标范围axis([xmin,xmax,ymin,ymax])
plt.savefig('./a')
plt.show()
