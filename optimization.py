#%%
from solution import trefferquote

from scipy import optimize as opt
from scipy.optimize import Bounds

def objective(x):
    h,alpha,v0 = x
    print(x)
    res = 1-trefferquote(h,alpha,v0,1000,False)
    print(res)
    return res


bounds = Bounds([1,2],[0,90],[0,10])

x0 = [1.5,60,7] # h,alpha,v0

res = opt.minimize(objective, x0, method='trust-constr', options={'verbose': 1})
#%%

