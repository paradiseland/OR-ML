# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/14 10:57
"""
# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/16 14:49
"""
from copy import deepcopy
import numpy as np

def Floyd(dis_matrix: np.ndarray):
    size = dis_matrix.shape[0]
    d = np.full((size, size), -1)
    v = np.full((size, size), fill_value=float('inf'))
    # --------initializing--------
    for i in range(size):
        for j in range(size):
            if dis_matrix[i, j] is not None and dis_matrix[i, j] != 0:
                v[i, j] = dis_matrix[i, j]
                d[i, j] = i + 1
    for i in range(size):
        v[i, i] = 0
    t = 0
    # ----------------------------
    while True:
        bo = deepcopy(v)
        t += 1
        if t > size:
            break
        for k in range(size):
            for l in range(size):
                if l == t-1:
                    continue
                last = bo[k, l]
                new = bo[k, t - 1] + bo[t - 1, l]
                v[k, l] = new if new < last else last
                if new < last:
                    v[k, l] = new
                    d[k, l] = t
                else:
                    pass
    return v

if __name__ == '__main__':
    data_matrix = [[0, None, None, 7],
          [20, 0, None, 2],
          [-3, None, 0, None],
          [7, None, 4, 0]]
    data_matrix = np.asarray(data_matrix)
    d = Floyd(data_matrix)
    print(d)
