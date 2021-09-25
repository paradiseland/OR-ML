# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/20 14:59
"""

"""
选址问题
源：3个:  i = 1, 2, 3
启动成本: [400, 250, 300]
市场 5个: j = 1, 2, 3, 4, 5
市场需求: [75, 90, 81, 26, 57]
运输成本：
[[4, 7, 3, 12, 15], 
 [13, 11, 17, 9, 19], 
 [8, 12, 10, 7, 5]]
"""

import gurobipy as gp
from gurobipy import multidict, tupledict, tuplelist
import numpy as np
import scipy as sp

class Benders:
    """

    """
    pass


def P(d, setup_c, trans_c):
    model = gp.Model("P")
    d_sum = sum(d)
    source_size = len(setup_c)
    sink_size = len(d)
    x_index = tuplelist([(i, j) for i in range(1, source_size + 1) for j in range(1, sink_size + 1)])
    x = model.addVars(x_index, name='x', lb=0.0, vtype=gp.GRB.CONTINUOUS)
    y = model.addVars(range(1, source_size + 1), name='y', vtype=gp.GRB.BINARY)
    c_ij = tupledict(zip(x_index, [j for i in trans_c for j in i]))
    c_ij_y = tupledict(zip(range(1, 4), setup_cost))
    obj = x.prod(c_ij) + y.prod(c_ij_y)

    model.setObjective(obj, sense=gp.GRB.MINIMIZE)
    model.addConstrs((x.sum(i, '*') - d_sum * y[i] <= 0 for i in range(1, source_size + 1)), "selected_")
    model.addConstrs((x.sum('*', i + 1) == d[i] for i in range(sink_size)), "balance_")

    model.optimize()

    print(", ".join(["x[{},{}]:{:.2f}".format(i + 1, j + 1, x[i + 1, j + 1].X) for i in range(source_size) for j in
                     range(sink_size)]))
    print(", ".join(["y[{}]={}".format(i, y[i].X) for i in range(1, source_size + 1)]))
    print("obj:", model.objVal)


def BP_l(d, setup_c, trans_c, y_l):
    """
    固定一些整数变量来给出原问题的线性模型
    :param d: demand
    :param setup_c: setup cost
    :param trans_c: transport cost
    :param y_l: iteration l: y value
    :return: model
    """
    model = gp.Model("BP_l")
    source_size = len(setup_c)
    sink_size = len(d)

    x_index = tuplelist([(i, j) for i in range(1, source_size + 1) for j in range(1, sink_size + 1)])
    x = model.addVars(x_index, name='x', lb=0.0, vtype=gp.GRB.CONTINUOUS)
    y = y_l
    c_ij = tupledict(zip(x_index, [j for i in trans_c for j in i]))
    c_ij_y = tupledict(zip(range(1, 4), setup_cost))
    fy = [c_y_i * y_i for c_y_i, y_i in zip(c_ij_y.values(), y)]
    obj = x.prod(c_ij) + sum(fy)

    model.setObjective(obj, sense=gp.GRB.MINIMIZE)
    model.addConstrs((x.sum(i, '*') - sum(d) * y[i - 1] <= 0 for i in range(1, source_size + 1)), "selected_")
    model.addConstrs((x.sum('*', i + 1) == d[i] for i in range(sink_size)), "balance_")
    model.optimize()
    model.update()
    return model


def BD_l(BP_l: gp.Model):
    model = gp.Model("BD_l")
    constraint_size_BP_l = len(BP_l.getConstrs())
    var_size_BP_l = len(BP_l.getVars())
    fy = BP_l.getObjective().getConstant()
    b_Fy = [BP_l.getConstrs()[i].RHS for i in range(constraint_size_BP_l)]
    v = model.addMVar(shape=(constraint_size_BP_l), lb=0, vtype=gp.GRB.CONTINUOUS, name='v')
    for var in v[3:]:
        var.setAttr(gp.GRB.Attr.LB, -gp.GRB.INFINITY)
    c_ij = [BP_l.getObjective().getCoeff(i) for i in range(var_size_BP_l)]
    A = BP_l.getA().toarray().transpose()  # csr matrix

    model.setObjective(v @ (np.asarray(b_Fy) + np.asarray(fy)))
    model.addMConstrs(A=A, x=v, sense="<"*8, b=np.asarray(c_ij), name='c')
    model.optimize()
    print("z")


def solve_main():
    pass


def solve_dual():
    pass

def BM_l():
    pass

def benders():
    extreme_points = []
    extreme_directions = []
    l = 1
    z_0 = -float('inf')


if __name__ == '__main__':
    demand = [75, 90, 81, 26, 57]
    setup_cost = [400, 250, 300]
    transport_cost = [[4, 7, 3, 12, 15],
                      [13, 11, 17, 9, 19],
                      [8, 12, 10, 7, 5]]
    P(demand, setup_cost, transport_cost)
    y_0 = [1, 0, 0]
    BP = BP_l(demand, setup_cost, transport_cost, y_0)
    BD_l(BP)
