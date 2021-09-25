# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/17 21:39
"""

"""
problem:
下料问题
长度为11m的进料
需求：{2:22}, {3:17}, {6:13}, {7:9}
----------------------------------
思路：
切割组合：
决策变量：x_k 第k个方案使用的次数
min sum_{k in{K}} x_k
sum_
"""

import gurobipy as gp
import numpy as np
import scipy as sp


def ELP(tech_coes, rhs):
    model = gp.Model("ELP")
    x_size = len(tech_coes[0])
    x = model.addMVar(shape=x_size, vtype=gp.GRB.INTEGER, lb=0, name="x")
    model.setObjective(x.sum(), gp.GRB.MINIMIZE)
    matrix_A = np.asarray(tech_coes)
    rhs = np.asarray(rhs)
    model.addConstr(matrix_A @ x >= rhs, name='c')
    model.optimize()
    print(x.X)
    return model


def ELP_dual(tech_coes, rhs):
    """
    we must set the dual decide variables to be  continuous, which then get the corresponding dual solution
    :param tech_coes:
    :param rhs:
    :return:
    """
    model = gp.Model("ELP_dual")
    v_size = len(tech_coes)
    v = model.addMVar(shape=v_size, vtype=gp.GRB.CONTINUOUS, lb=0)
    obj = np.asarray(rhs)
    rhs = np.asarray([1] * len(tech_coes[1]))
    model.setObjective(obj @ v, gp.GRB.MAXIMIZE)
    matrix_A_T = np.asarray(tech_coes).transpose()
    model.addConstr(matrix_A_T @ v <= rhs)
    model.optimize()
    print(v.X)
    return model


def subP(res_of_dual: np.ndarray, h):
    """
    we opt a sub problem to get a new column to the primal problem.
    :return: a new column to be added into the technology coefficient matrix.
    """
    sub_p = gp.Model("sub-problem")
    h = np.asarray(h)
    a = sub_p.addMVar(shape=res_of_dual.shape[0], vtype=gp.GRB.INTEGER, lb=0)
    sub_p.setObjective(1 - res_of_dual @ a, gp.GRB.MINIMIZE)
    sum_ = sum([a_i * h_i for a_i, h_i in zip(a.vararr, h)])
    sub_p.addConstr(sum_ <= size)
    sub_p.optimize()
    if sub_p.objVal < -1e-3:
        return a.X
    else:
        return []


def column_generate(tech_coes, rhs, h):
    while True:
        RMP = ELP(tech_coes, rhs)
        Dual = ELP_dual(tech_coes, rhs)
        v = np.asarray([i.X for i in Dual.getVars()])
        sub = subP(v, h)
        if len(sub) > 0:
            tech_coes = np.c_[tech_coes, sub]
            print("add a new column:", sub)
            pass
        else:
            break
    print([i.X for i in RMP.getVars()])
    print(RMP.getA().toarray())
    print("opt=", RMP.objVal)


if __name__ == '__main__':
    size = 11
    h_i = [7, 6, 3, 2]
    A = np.asarray([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    b = np.asarray([9, 13, 17, 22])
    column_generate(A, b, h_i)
