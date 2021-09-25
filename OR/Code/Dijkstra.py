# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/16 16:07
"""

import numpy as np


def Dijkstra(dis_matrix: np.ndarray):
    size = dis_matrix.shape[0]
    # true:临时标签 false为永久标签
    labels_kind = np.ones(size, dtype=bool)
    v = np.full(size, fill_value=float('inf'))
    d = np.full(size, fill_value=-1)
    next_ever = 0  # next ever label
    labels_kind[next_ever] = False
    # --------initializing---------
    v[0] = 0
    # -----------------------------
    while True:
        if sum(labels_kind) == 0:
            break
        else:
            for i in range(size):
                if dis_matrix[next_ever, i] is not None:
                    if not labels_kind[i]:
                        continue
                    last = v[i]
                    new = v[next_ever] + dis_matrix[next_ever, i]
                    if new < last:
                        v[i] = new
                        d[i] = next_ever+1
            minn = float('inf')
            min_index = -1
            for idx, val in enumerate(v):
                if not labels_kind[idx]:
                    continue
                if val <= minn:
                    minn = val
                    min_index = idx
            next_ever = min_index
            labels_kind[next_ever] = False
    return v, d


if __name__ == '__main__':
    dis = [[0, 5, 20, None, 5],
          [None, 0, 12, None, 3],
          [None, 12, 0, None, None],
          [None, 4, None, 0, 6],
          [None, None, 2, None, 0]]
    dis = np.asarray(dis)
    print(Dijkstra(dis)[0])

