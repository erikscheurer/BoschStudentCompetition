# %%
from multiprocessing import Pool, freeze_support
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
from timeit import default_timer as timer

def same_signs(a, b):
    assert not math.isnan(a) and not math.isnan(b)
    return (a >= 0) == (b >= 0)

def plot_ring():
    r_ball = 0.765/(2*np.pi)
    x_board = 4.525
    y_board = 3.95
    x_ring = x_board-0.45
    y_ring = 3.05
    plt.xlim(0, 5)
    plt.ylim(0, 8)
    # ring
    plt.plot((x_ring),(y_ring), 'ro', markersize=2)
    xx = np.linspace(x_ring-r_ball+1e-7, x_ring+r_ball-1e-7, 100)
    def ring(x): return np.sqrt((r_ball)**2-(x-x_ring)**2) + y_ring
    rr = ring(xx)
    plt.plot(xx, rr, 'r--')
    # brett
    plt.plot([x_board]*2,(y_ring,y_board), 'r')
    plt.plot([x_board-r_ball]*2,(y_ring,y_board), 'r--')
    plt.plot([x_ring, x_board],[y_ring]*2, '-', linewidth=1, alpha=0.9, color='orange')

def plot_throw(f,x_lower=0,x_upper=5,line='b-'):
    plot_ring()
    xx = np.linspace(x_lower,x_upper, 1000)
    yy = f(xx)
    plt.plot(xx, yy, line, alpha=0.99, linewidth=1)
    # plt.xlim(3.8, 4.6)
    # plt.ylim(2.7, 4.6)

def plot(f, ring, x_board, y_lower, y_upper, x_ring, y_ring, sol, r_ball, y_board, e, v, v_parallel, v_bounced):
    xx = np.linspace(0, 5, 1000)
    rr = ring(xx)

    plt.figure(figsize=(5, 5))
    # brett
    y_brett = np.linspace(y_lower, y_upper)
    brett = plt.plot([x_board]*len(y_brett), y_brett, 'r')
    brett = plt.plot([x_board-r_ball]*len(y_brett), y_brett, 'r--')
    plt.plot(x_board-r_ball, y_board, 'rx')
    # ball
    plot_throw(f)
    # ring
    plt.plot(xx, rr, 'r--')
    plt.plot(x_ring, y_ring, 'ro', markersize=2)
    plt.plot(sol.x[0], f(sol.x[0]), 'x', markersize=10)

    plt.ylim(0, 10)
    plt.xlim(0, 5)

    v = v/np.linalg.norm(v)
    v_parallel = v_parallel/np.linalg.norm(v_parallel)
    v_bounced = v_bounced/np.linalg.norm(v_bounced)
    plt.arrow(sol.x[0], f(sol.x[0]), e[0], e[1])
    plt.arrow(sol.x[0], f(sol.x[0]), v[0], v[1], color='g')
    plt.arrow(sol.x[0], f(sol.x[0]), v_parallel[0], v_parallel[1], color='r')
    plt.arrow(sol.x[0], f(sol.x[0]), v_bounced[0], v_bounced[1], color='b')
    plt.pause(10000)

def get_sign_change_interval(f, a, b, vx, depth=2):
    if vx<0: # if we move left, check right first, then left since the ball is coming from the right
        a, b = b, a

    ya, yb = f(a), f(b)
    if not same_signs(ya, yb): # if a,b have sign change, return them
        return a, b

    m = (a+b)/2 # else, insert middle point and begin iterative search
    points = [a, m, b]
    values = [ya, f(m), yb]
    
    for k in range(depth):
        i = 1
        while i < len(values):
            if not same_signs(values[i-1], values[i]): # return interval if sign changes
                return points[i-1], points[i]
            else:
                points.insert(i, (points[i-1]+points[i])/2) # or insert middle point as new interval boundary candidate
                values.insert(i, f(points[i]))
                i += 1
            i += 1

    return None, None # if no sign change found, return None

def get_ring_collision(ring_left:float, ring_right:float, objective:Callable[[float],float], vx:float, eps=1e-8, tol=1e-5)->float:
    ring_left+=eps # for numeric stability
    ring_right-=eps

    a, b = get_sign_change_interval(objective, ring_left, ring_right, vx)
    if a is None:
        return False

    a, b = min(a,b), max(a,b) # change interval ordering back for optimize function
    return opt.brentq(objective, a, b, maxiter=100)

def throw_under_ring(f, x_ring, y_ring, r_ball, vx, ueps=1e-3):
    y_throw_ring_left = f(x_ring-r_ball)
    return vx > 0 and y_throw_ring_left < y_ring + ueps

def check_backboard(f, g, x0, vx, vy, x_board, r_ball, y_lower, y_upper, eps, x_plane, output=True):
    r = None
    goesover = False
    goesunder = False
    if vx > 0: # only check backboard if ball is moving to the right
        # interception of the ball with the backboard
        y_impact_board = f(x_board-r_ball)
        if y_upper < y_impact_board:
            if output:
                print('The ball goes over the backboard')
                plot_throw(f,x_lower=x0) # plot throw until the ball misses the backboard
            # plt.show()
            goesover = True
            # TODO Implement bounce on top of board
        elif y_lower > y_impact_board:
            # print('The ball goes under the backboard')
            goesunder = True
        else: # Cant hit backboard with negative x velocity
            if output:
                print('The ball hits the backboard')
                plot_throw(f, x_lower=x0, x_upper=x_board-r_ball) # plot throw until the ball hits the backboard
            # recursive call after hitting the backboard
            vy_impact_board = g*(x_board-r_ball-x0)/vx + vy
            r = simulate_throw(
                g=g,
                x0=x_board-r_ball,
                y0=y_impact_board,
                vx=-vx,
                vy=vy_impact_board,
                r_ball=r_ball,
                x_board=x_board,
                y_lower=y_lower,
                y_upper=y_upper,
                eps=eps,
                output=output
            )
    if goesover:
        return goesover, goesunder, x_plane
    else:
        return goesover, goesunder, r

def check_ring_collision(f, x0, vx, x_ring, y_ring, r_ball, eps=1e-8):
    def ring(x):
        assert x > x_ring-r_ball and x < x_ring+r_ball
        return np.sqrt(r_ball**2-(x-x_ring)**2) + y_ring
    def obj(x): return f(x)-ring(x)
    # x_spitze = -b/(2*a) # hoechster punkt der parabel

    if x_ring-r_ball + 2*eps < x0 < x_ring + r_ball - 2*eps: # if we bounced from the ring in the last recursion step
        if vx > 0: # we bounced to the right
            x_impact = get_ring_collision(x0,x_ring+r_ball,obj,vx)
            hitsring = bool(x_impact) # if return value is a number, hitsring is true
        else:# vx < 0, we bounced to the left
            # x_impact, sol = opt.bisect(obj, x_ring - r_ball + eps, x0,full_output=True)
            x_impact = get_ring_collision(x_ring-r_ball,x0,obj,vx)
            hitsring = bool(x_impact)
    else:
        x_impact = get_ring_collision(x_ring-r_ball,x_ring+r_ball,obj,vx)
        hitsring = bool(x_impact)

    return hitsring, x_impact

def ring_bounce(f, g, x0, y0, vx, vy, r_ball, x_board, y_lower, y_upper, eps, x_impact, x_ring, y_ring, output=True):
    y_impact = f(x_impact)

    e = np.array((x_ring-x_impact, y_ring-y_impact))
    e /= np.linalg.norm(e)
    # print('e = ', e)

    # vertical speed at impact
    vy_impact_board = g*(x_impact-x0)/vx + vy
    v = np.array((vx, vy_impact_board))
    # print('impact v: ', v)

    # velocity parallel to normed vector e
    v_parallel = np.dot(v, e)*e
    # print('parallel v: ', v_parallel)
    # elastic collision
    v_bounced = v - 2*v_parallel
    # print('bounced v: ', v_bounced)

    # move ball away from ring a bit to avoid infinite recursion
    x_impact -= eps*e[0]

    # plot(f, ring, x_board, y_lower, y_upper,x_ring, y_ring, sol, r_ball, y_board, e, v, v_parallel, v_bounced)
    if output:
        if vx > 0:
            plot_throw(f,x_upper=x_impact, x_lower=x0)
        else:
            plot_throw(f,x_lower=x_impact, x_upper=x0)
    # recursive call after hitting the ring
    return simulate_throw(
        g=g,
        x0=x_impact,
        y0=y_impact,
        vx=v_bounced[0],
        vy=v_bounced[1],
        r_ball=r_ball,
        x_board=x_board,
        y_lower=y_lower,
        y_upper=y_upper,
        eps=eps,
        output=output
    )

def simulate_throw(
    prev_throw='original',
    # general
    g=-9.81,
    # Ball paramaters
    x0=0,
    y0=2,
    vy=10,
    vx=2.18,
    r_ball=0.765/(2*np.pi),
    # board parameters
    x_board=4.525,#525,
    y_lower=3.05,
    y_upper=3.95,
    # numeric parameters
    eps=1e-8,
    ueps=1e-3,
    output = True
):
    if output:
        print('simulate_throw:')
        print('x0 = ', x0, '    y0 = ', y0, '   vx = ', vx, '   vy = ', vy, '   r_ball = ', r_ball)
    # helper parameters for parabola
    a = g/(2*vx**2)
    b = vy/vx-g*x0/vx**2
    c = g*x0**2/(2*vx**2)-vy/vx*x0 + y0

    # ring parameters
    x_ring = x_board-0.45
    y_ring = y_lower

    # define parabola
    def f(x): return a*x**2 + b*x + c
    def df(x): return 2*a*x + b

    hitsring = False
    goesover = False
    goesunder = False
    goesin = False

    x_spitze = -b/(2*a) # hoechster punkt der parabel
    y_spitze = f(x_spitze)

    # interception of the ball with 3.05m
    sqrt_part = b**2-4*a*(c-y_lower)
    if sqrt_part < 0 and y_spitze < y_lower:
        if output:
            print("The ball is thrown too low")
            plot_throw(f,x_lower=x0)
        return 0 # return 0 instead of x_plane as ball never hits plane
    x_plane = (-b - np.sign(vx) * np.sqrt(sqrt_part)) / (2*a) # ( das - vor sign ist nötig weil a negativ ist)
    assert x_plane is not None

    if not (x_ring - r_ball < x0 < x_ring + r_ball): # check if ball starts in ring area
        if throw_under_ring(f, x_ring, y_ring, r_ball, vx, ueps): # if the ball is under the ring, we can't hit anything
            if output:
                print("The ball is thrown too low")
                plot_throw(f,x_lower=x0)
            return x_plane

    # check if the ball hits the backboard, goes over it, or goes under it
    goesover, goesunder, r = check_backboard(f, g, x0, vx, vy, x_board, r_ball, y_lower, y_upper, eps, x_plane, output=output)
    if r is not None: # if ball hits, return result from recursive call, else return x_plane if the ball goes over the backboard
        return r
    
    # interception of the ball with the ring needs to be checked, as ball goes under the backboard or moves left
    hitsring, x_impact = check_ring_collision(f, x0, vx, x_ring, y_ring, r_ball)
    if hitsring:
        if output:
            print('The ball hits the ring')
        return ring_bounce(f, g, x0, y0, vx, vy, r_ball, x_board, y_lower, y_upper, eps, x_impact, x_ring, y_ring, output=output)

    # else:
    if goesunder or vx < 0:
        if x_plane < x_ring - r_ball: # check for airball
            if output:
                print('AIRBALL')
                if vx > 0:
                    plot_throw(f, x_lower=x0)
                else:
                    plot_throw(f, x_upper=x0)
            # plt.show()
            return x_plane
        elif x_plane > x_ring + r_ball - ueps: # check for basket
            if output:
                print('The ball goes in')
                if vx < 0:
                    plot_throw(f, x_upper=x0, x_lower = x_plane)
                else:
                    plot_throw(f, x_lower=x0, x_upper = x_plane) 
            # plt.show()
            return x_plane
    
    if np.abs(df(x_plane)) > 1e3:
        def ring(x):
            assert x > x_ring-r_ball and x < x_ring+r_ball
            return np.sqrt(r_ball**2-(x-x_ring)**2) + y_ring
        print("fail")
        return 0 #TODO: figure out how to deal with very steep parabolas


    # plot_throw(f)
    print("wtf happened")
    # plt.show()
    return 0
    # raise(Exception('How did we get here'))

def check_in_basket(x_plane, x_board=4.525, ring_durchmesser=0.45):
    a = x_board - ring_durchmesser < x_plane
    b = x_plane < x_board
    c = np.logical_and(a, b)
    return c.astype(int)

def mapfunc(x0, y0, vx, vy, r_ball, output):
    return simulate_throw(x0=x0, y0=y0, vx=vx, vy=vy, r_ball=r_ball, output=output)


def trefferquote(h,alpha,v0,n=100, output=True):
    
    hs = np.zeros(n)+h
    alphas = np.zeros(n)+alpha
    v0s = np.zeros(n)+v0
    
    h_rands = hs + np.random.uniform(-1,1, size=n)*.15
    alpha_rands = alphas + np.random.uniform(-1,1, size=n)*5
    alpha_rands = np.deg2rad(alpha_rands)
    v0_rands = (1 + np.random.uniform(-1,1, size=n)*.05)*v0

    x0s = np.cos(alpha_rands)*h_rands
    y0s = np.sin(alpha_rands)*h_rands
    vxs = np.cos(alpha_rands)*v0_rands
    vys = np.sin(alpha_rands)*v0_rands

    circ_balls = 0.765 + np.random.uniform(-1,1, size=n)*.015
    r_balls = circ_balls / (2*np.pi)

    # with Pool(8) as p:
    #     x_planes = p.starmap(mapfunc, zip(x0s, y0s, vxs, vys, r_balls, [output]*n))
    # x_planes = np.asarray(x_planes)

    x_planes = np.asarray(list(map(mapfunc, x0s, y0s, vxs, vys, r_balls, [output]*n)))

    in_baskets = check_in_basket(x_planes)
    hits = np.sum(in_baskets)

    if output:
        plt.show()
    print("trefferquote: ", hits/n)
    return hits/n

# %%
if __name__ == '__main__': # muss rein für multiprocessing
    freeze_support() # das anscheinend auch (keine ahnung was das ist) 

    np.random.seed(124587)#15) # seed 124587 yields error
    while True: # für zufällige input werte

        # sammle quoten für verschiedene anzahlen von würfen um zu gucken wie gut wir annähern müssen
        quoten = []
        ns = [100, 1000, 10000, 25000, 50000]
        h = np.random.rand()+1
        alpha = np.random.randint(45,75)
        v0 = (np.random.rand()*5)+5
        print("h: ", h, "alpha: ", alpha, "v0: ", v0)
        for n in ns:
            start = timer()
            quoten.append(trefferquote(h=h, alpha=alpha, v0=v0,n=n, output=False))
            if quoten[0]<.1: # kann gut sein dass die werte ganz scheiße sind, dann neue bevor man da rechenleistung verschwendet
                break
            end = timer()
            print(n, end-start)

        if len(quoten)>1: # plotte verschiedene trefferquoten über anzahl der iterationen 
            plt.show()
            plt.plot(ns, quoten, 'o-')
            plt.xscale('log')
            plt.show()

    exit()
    plt.show()

    # %% 
    # 0.6147413602474181 1.314172781056323 3.239501812265219 6.925294735574873 0.121860690103832
    print(simulate_throw(x0 = 0.6147413602474181, y0 = 1.314172781056323, vx = 3.239501812265219, vy = 6.925294735574873, r_ball = 0.121860690103832))
    plt.show()
    # for vx in np.linspace(3.3,4,100):
    #     print(vx, simulate_throw(vy=7,vx=vx))
    #     plt.show()
    # # # %%
    # plt.show()
