# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/16 10:27
"""
import numpy as np


def bellman_ford(dis_matrix: np.ndarray):
    """
    realize this algorithm from OR book:Optimization in Operation Research.
    :return:
    """
    # ----------initializing---------
    length = dis_matrix.shape[0]

    d = [0 for i in range(length)]  # route
    v = [[] for i in range(length)]  # record of path length
    v[0].append(0)
    for i in range(1, 4):
        v[i].append(float('inf'))
    t = 0
    # --------------------------------
    while True:
        if t == length:
            break
        for k in range(length):
            tmp = np.asarray([v[i][t] for i in range(length)])
            # 选出进入k的路径，如1，第一列元素
            i: int
            tmp1 = []
            for i in range(length):
                if dis_matrix[i, k] is not None:
                    tmp1.append(tmp[i] + dis_matrix[i, k])

            new = min(tmp1)
            if new < v[k][t]:
                d[k] = np.where(tmp1 == np.min(tmp1))[0][0]+1
            v[k].append(new)

        t += 1
    return [i[-1] for i in v], d


if __name__ == '__main__':
    ma = [[0, 5, 8, None],
          [None, 0, -10, None],
          [8, None, 0, None],
          [None, 2, 3, 0]]
    dm = np.asarray(ma)
    print(bellman_ford(dm))
