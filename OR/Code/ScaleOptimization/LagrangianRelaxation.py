# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/1/23 10:13
"""
import gurobipy as gp
import numpy as np
from P import P


class LagrangianRelaxation:
    """
    solve a 'min' problem by lagrangian relaxation.

    """

    def __init__(self, Primal, relax, lambda_0):
        self.P = Primal
        self.lambda_0 = lambda_0
        self.v_p = gp.GRB.INFINITY
        self.v_d = - gp.GRB.INFINITY
        self.select = relax
        self.init()

    def init(self):
        self.iter()

    def iter(self):
        l = 0
        new_lambda = self.lambda_0
        while True:

            Pv = self.solve_Pv(new_lambda)
            x_v = [i.X for i in Pv.getVars()]
            print(
                    "\n************************[{}]*************************"
                    "\nlambda={}\nx_v={}\nz(P_v)={:.2f}"
                    "\n*****************************************************"
                        .format(l, new_lambda, x_v, Pv.objVal))
            r_Rx = [self.P.b[i] - sum([r_i * x_i for r_i, x_i in zip(self.P.A[i], x_v)]) for i in self.select]
            if all([i <= 0 for i in r_Rx]) and all([v_i * r_Rx_i == 0 for v_i, r_Rx_i in zip(new_lambda, r_Rx)]):
                print("opt")
                break
            if Pv.objVal > self.v_d:
                self.v_d = Pv.objVal
            if Pv.objVal < self.v_d:
                break
            lambda_l_plus1 = 1 / (2 * (l + 1))
            new_new_lambda = [new_lambda[i] + lambda_l_plus1 * r_Rx[i] / (sum((j * j for j in r_Rx))) ** .5 for i in
                              range(len(new_lambda))]
            new_lambda = [i if i > 0 else 0 for i in new_new_lambda]
            l += 1

    def solve_Pv(self, lambd):
        model = gp.Model('Pv')
        x = model.addVars(len(self.P.c)).values()
        for i in range(len(x)):
            x[i].lb = self.P.x_lb[i]
            x[i].ub = self.P.x_ub[i]
            x[i].vType = self.P.x_kind[i]
        LR_penalty = gp.LinExpr(0)
        for row in range(self.P.A.shape[0]):
            if row in self.select:
                tmp = gp.LinExpr(self.P.A[row], x)

                LR_penalty += lambd[row] * (self.P.b[row] - tmp)
            else:
                model.addConstr(gp.LinExpr(self.P.A[row], x), self.P.sense[row], rhs=self.P.b[row],
                                name="c")
        model.setObjective(gp.LinExpr(self.P.c, x) + LR_penalty, self.P.obj_kind)
        model.optimize()
        return model


if __name__ == '__main__':
    primal = P(a_list=np.asarray([[5, 2], [2, 5], [8, 8]]),
               b=np.asarray([3, 3, 1]),
               obj_kind=gp.GRB.MINIMIZE,
               c=np.asarray([3, 2]),
               sense=[gp.GRB.GREATER_EQUAL, gp.GRB.GREATER_EQUAL, gp.GRB.GREATER_EQUAL],
               x_kind=[gp.GRB.INTEGER, gp.GRB.INTEGER],
               x_lb=[0, 0],
               x_ub=[1, 2])
    p_model = primal.get_model("P")
    p_model.optimize()
    print("opt_x=", [i.X for i in p_model.getVars()], "\n")
    lr = LagrangianRelaxation(primal, [0, 1], [0.5, 0.4])
