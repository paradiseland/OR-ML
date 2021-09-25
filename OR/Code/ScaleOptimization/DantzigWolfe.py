# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/24 17:07
"""
from ScaleOptimization.P import P
import numpy as np
import gurobipy as gp


class DantzigWolfe:
    """
    Realize a Dantzig-Wolfe decomposition, and give a instance.
    """

    def __init__(self, primal, kind, x_split):
        self.P = primal
        self.s_num = len(kind) - 1
        self.kind = kind
        self.sub_x = x_split
        # decision variables: lambda
        self.la_ = [[] for i in range(self.s_num)]
        self.mu_ = [[] for i in range(self.s_num)]
        self.delta_x = [[] for i in range(self.s_num)]
        self.x = [[] for i in range(self.s_num)]
        self.D_s = [[] for i in range(self.s_num)]
        self.P_s = [[] for i in range(self.s_num)]
        self.subs = self.subs_init()

    def iter(self):
        while True:
            lambda_, mu_, v_, q_ = self.solve_Px()
            for i in range(self.s_num):
                self.solve_sub(i)
        pass

    def solve_Px(self):
        """
        solve the main problem when x is determined
        x_size: number of extreme points
        delta_x_size: number of extreme directions 
        :return: 
        """""
        model = gp.Model("Px")
        lambda_ = []
        mu_ = []
        for i in range(self.s_num):
            lambda_.append(
                    model.addVars(range(len(self.x[i])), vtype=gp.GRB.CONTINUOUS, lb=0, name='lambda_{}'.format(i + 1)))
            mu_.append(
                    model.addVars(range(len(self.delta_x[i])), vtype=gp.GRB.CONTINUOUS, lb=0,
                                  name='mu_{}'.format(i + 1)))
        sum_constr_index = self.kind[0]
        sum_constrs = []
        for i in range(self.s_num):
            model.addConstr(gp.quicksum(lambda_[i]), sense=gp.GRB.EQUAL, rhs=1, name="s{}".format(i + 1))
        obj_x = []
        obj_delta_x = []
        tag = True
        for i in sum_constr_index:
            Sigma = gp.LinExpr(0)
            a_s = []
            a_s.append(self.P.A[i][:self.sub_x[1][0]])
            a_s.append(self.P.A[i][self.sub_x[1][0]:])
            for j in range(self.s_num):
                x = [x_ii * lambda_[j].values()[ind] for ind, x_i in enumerate(self.x[j]) for x_ii in x_i]
                if i == 0:
                    obj_x.append(x)
                delta_x = [delta_x_ii * mu_[j].values()[ind] for ind, delta_x_i in enumerate(self.delta_x[j]) for
                           delta_x_ii in delta_x_i]
                if delta_x:
                    if tag:
                        obj_x.append(x)
                        obj_delta_x.append(delta_x)
                        tag = False
                    Sigma += sum([a_s_i * x_i for a_s_i, x_i in zip(a_s[j], [x_i + delta_x_i if delta_x else 0 for
                                                                             x_i, delta_x_i in zip(x, delta_x)])])
                else:
                    Sigma += sum([a_s_i * x_i for a_s_i, x_i in zip(a_s[j], x)])
            model.addConstr(Sigma, sense=gp.GRB.LESS_EQUAL, rhs=self.P.b[i], name="Sigma_{}".format(i + 1))
        obj = gp.LinExpr(0)
        obj += sum([c_i*x_i for c_i, x_i in zip(self.P.c[:self.sub_x[1][0]], obj_x[0], )])
        obj += sum([c_i*x_i for c_i, x_i in zip(self.P.c[self.sub_x[1][0]:], obj_x[1], )])
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.optimize()
        lambda_x = [i[0].X for i in lambda_]
        if sum(mu_[0]) != 0:
            mu_x = [i[0].X for i in mu_]
        else:
            mu_x = []
        q_l = [model.getConstrs()[i].Pi for i in range(len(model.getConstrs())-self.s_num)]
        v_l = [model.getConstrs()[i].Pi for i in range(len(model.getConstrs())-self.s_num, len(model.getConstrs()))]
        print("lambda:", lambda_x)
        print("mu:", mu_x)
        print("dual variable q_l", q_l)
        print("dual variable v_l", v_l)
        print("mainP-obj:", model.objVal)
        return lambda_x, mu_x, v_l, q_l

    def subs_init(self):
        subs = [gp.Model("sub1"), gp.Model("sub2")]
        self.x[0].append([0, 22])
        self.x[1].append([0, 12])
        for i in range(self.s_num):
            model = subs[i]
            model.addMVar((2,), lb=0, vtype=gp.GRB.CONTINUOUS, name='x{}'.format(i))
            if i == 1:
                model.addMConstrs(self.P.A[, :2])
            else:
                model.addMConstrs()
        return subs

    def solve_sub(self, i):
        model = self.subs[i]
        model.reset()
        model.setObjective()
        model.addConstrs()


if __name__ == '__main__':
    p = P(a_list=np.asarray([[2.1, 2.1, .75, .75],
                             [.5, .5, .5, .5],
                             [1, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 1, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]]),
          b=np.asarray([60, 25, 22, 20, 12, 15, 25]),
          obj_kind=gp.GRB.MAXIMIZE,
          c=np.asarray([14, 8, 11, 7]),
          sense=[gp.GRB.LESS_EQUAL, gp.GRB.LESS_EQUAL, gp.GRB.GREATER_EQUAL, gp.GRB.LESS_EQUAL,
                 gp.GRB.GREATER_EQUAL, gp.GRB.LESS_EQUAL, gp.GRB.LESS_EQUAL],
          x_kind=[gp.GRB.CONTINUOUS, gp.GRB.CONTINUOUS],
          x_lb=[-gp.GRB.INFINITY, -gp.GRB.INFINITY, -gp.GRB.INFINITY, -gp.GRB.INFINITY],
          x_ub=[gp.GRB.INFINITY, gp.GRB.INFINITY, gp.GRB.INFINITY, gp.GRB.INFINITY])
    dw = DantzigWolfe(primal=p, kind=[[0, 1], [2, 3], [4, 6]], x_split=[2, [2]])
    dw.solve_Px()
