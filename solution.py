# %%
from sympy import *
import numpy

x = Symbol('x')
r_ball = Symbol('r_ball')
yr = Symbol('yr')
xr = Symbol('xr')
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')
# %%
equation = Eq(sqrt(r_ball ** 2-(x-xr) ** 2)+yr, a*x ** 2 + b*x + c)

# %%
solve(equation, x)
# %%
