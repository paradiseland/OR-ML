# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/24 14:23
"""
import gurobipy as gp
import numpy as np


class P:

    def __init__(self, a_list, b, obj_kind, c, sense, x_kind, x_lb, x_ub):
        self.model = gp.Model("P")
        self.A = a_list
        self.b = b
        self.obj_kind = obj_kind
        self.c = c
        # self.x = self.model.addVars(range(x_size))
        self.sense = sense
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.x_kind = x_kind

    def get_Dual(self):
        dual = gp.Model('Dual')
        A = self.A.transpose()
        if self.obj_kind == gp.GRB.MINIMIZE:
            obj_kind = gp.GRB.MAXIMIZE
        else:
            obj_kind = gp.GRB.MINIMIZE
        c = self.b
        b = self.c
        x = dual.addVars(self.A.shape[1], )
        x_size = len(x)
        x_kinds = [[] for i in range(x_size)]
        for i, se in enumerate(self.sense):
            x_kinds[i].append(gp.GRB.CONTINUOUS)
            if se == "<":
                x_kinds[i].append(-gp.GRB.INFINITY)  # lb
                x_kinds[i].append(0)  # ub
            elif se == ">":
                x_kinds[i].append(0)
                x_kinds[i].append(gp.GRB.INFINITY)
            else:
                x_kinds[i].append(-gp.GRB.INFINITY)
                x_kinds[i].append(gp.GRB.INFINITY)
        x_kind = [i[0] for i in x_kinds]
        x_lb = [i[1] for i in x_kinds]
        x_ub = [i[2] for i in x_kinds]
        sense = []
        for i in x:
            if i.lb == 0 and i.ub == gp.GRB.INFINITY:
                sense.append(gp.GRB.LESS_EQUAL)
            elif i.lb == - gp.GRB.INFINITY and i.ub == 0:
                sense.append(gp.GRB.GREATER_EQUAL)
            else:
                sense.append(gp.GRB.EQUAL)
        dual_model = P(x_kinds, b, obj_kind, c, sense, x_kind, x_lb, x_ub)
        return dual_model

    def get_model(self, name):
        model = gp.Model(name)
        x = model.addVars(len(self.c)).values()
        for i in range(len(x)):
            x[i].lb = self.x_lb[i]
            x[i].ub = self.x_ub[i]
            x[i].vType = self.x_kind[i]
        for row in range(self.A.shape[0]):
            model.addConstr(gp.LinExpr(self.A[row], x), self.sense[row], rhs=self.b[row],
                            name="c")
        model.setObjective(gp.LinExpr(self.c, x), self.obj_kind)
        return model


if __name__ == '__main__':

    p = P(np.asarray([[1, 2], [2, 1]]),
          np.asarray([2, 2]),
          gp.GRB.MINIMIZE,
          np.asarray([1, 1]),
          sense=[gp.GRB.GREATER_EQUAL, gp.GRB.GREATER_EQUAL],
          x_kind=[gp.GRB.CONTINUOUS, gp.GRB.CONTINUOUS],
          x_lb=[0, 0],
          x_ub=[gp.GRB.INFINITY, gp.GRB.INFINITY], )
    primal = p.get_model('p')
    primal.optimize()
    print([i.X for i in primal.getVars()])
