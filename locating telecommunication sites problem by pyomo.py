from random import seed
from random import randint
import numpy  as np
from pyomo.environ import *

I = 60
# number of citizens
J = 6
# number of centres
z1 = 320
# goal of benefit
z3 = 2500
# goal of wire usage
np.random.seed(1)
v = (np.random.randint(5,8,size=I))
# value of citizen i
d = (np.random.randint(40,60,size=(I,J)))        
# distance between citizen i and centre j

M = ConcreteModel()

bigM = 1000000
cm = 0.02
# price of using 1 metre of wire
ca = 1
# price of additional equipment

M.i = RangeSet(I)
M.j = RangeSet(J)
M.s = RangeSet(3)

M.x = Var(M.i, M.j, within = Binary)
# Is citizen i connect to centre j?
M.y = Var(M.j , within = Binary)
# Does centre j connect to more than 8 citizens?
M.p = Var(M.j , within = Binary)
# Is centre j active?
M.sp = Var(M.s , within = NonNegativeReals)
# splus
M.sm = Var(M.s , within = NonNegativeReals)
# sminus

M.z = Objective(expr = M.sm[1] + 3.5*M.sm[2] + cm*M.sp[3] , sense = minimize)

M.ST = ConstraintList()

#///////////////////////////////////////////////////////////////////////////////////////

for j in M.j:
    M.ST.add(sum(M.x[i,j] for i in M.i) <= 8 + bigM * M.y[j])
    M.ST.add(sum(M.x[i,j] for i in M.i) >= 8 - bigM * (1 - M.y[j]))
# for determining centres which connect to more than 8 citizen

#///////////////////////////////////////////////////////////////////////////////////////

for i in M.i:
    M.ST.add(sum(M.x[i,j] for j in M.j) <= 1)
# each citizen can connect to at most 1 centre

#///////////////////////////////////////////////////////////////////////////////////////

expr1 = sum(M.x[i,j]*v[i-1] for i in M.i for j in M.j)
expr2 = sum(M.x[i,j]*d[i-1,j-1] for i in M.i for j in M.j)
expr3 = sum(M.y[j] for j in M.j)
M.ST.add(expr1 - cm*expr2 - ca*expr3 + M.sm[1] - M.sp[1] == z1)
# for benefit goal

#///////////////////////////////////////////////////////////////////////////////////////

for j in M.j:
    M.ST.add((sum(M.x[i,j] for i in M.i))/bigM <= M.p[j])
    M.ST.add((sum(M.x[i,j] for i in M.i))*bigM >= M.p[j])
M.ST.add(sum(M.p[j] for j in M.j) + M.sm[2] - M.sp[2] == 5)  
# for at least 5 active centres
    
#///////////////////////////////////////////////////////////////////////////////////////

M.ST.add(sum(M.x[i,j]*d[i-1,j-1] for i in M.i for j in M.j) + M.sm[3] - M.sp[3] == z3)
# for wire usage goal

#///////////////////////////////////////////////////////////////////////////////////////                    

opt = SolverFactory('cplex')
results = opt.solve(M)
display(M)
print("Variables Printed Bellow:")
for i in M.i:
    for j in M.j:
        print("x[" , i , j , "]:" , M.x[i,j].value)
for j in M.j:
    print("y[" , j , "]:" , M.y[j].value)
    print("p[" , j , "]:" , M.p[j].value)
for s in M.s:
    print("sp[" , s , "]:" , M.sp[s].value)
    print("sm[" , s , "]:" , M.sm[s].value)
print("Optimal Value Printed Bellow:")
print(value(M.z))