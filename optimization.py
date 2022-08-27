# %%
from solution import hit_rate

from scipy import optimize as opt
from scipy.optimize import Bounds
from noisyopt import minimizeCompass, minimizeSPSA

h_upper = 2
h_lower = 1
alpha_upper = 90
v0_upper = 20


def objective(x, n=100):
    # x in [0,1]^3
    # map to correct range
    h = h_lower + (h_upper - h_lower) * x[0]
    alpha = x[1]*alpha_upper
    v0 = x[2]*v0_upper

    print(h, alpha, v0)
    res = 1-hit_rate(h, alpha, v0, n, False)
    print(res)
    return res


bounds = [[0, 1], [1e-3, 1], [1e-3, 1]]

x0 = [1., 60.3/90, 7.112/20]  # h,alpha,v0

for n in [10000]:
    # if n <= 1000:
    res = minimizeCompass(lambda x: objective(x, n), x0, bounds=bounds,
                          paired=False, deltatol=0.01)
    # else:
    #     print('slsqp')
    # res = opt.minimize(lambda x: objective(x, n), x0, method='trust-constr',
    #                    options={'verbose': 1.}, bounds=bounds, tol=1e-9)
# %%
print(res)

# %%
