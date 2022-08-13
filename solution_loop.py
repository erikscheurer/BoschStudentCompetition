# %%
from cmath import sqrt
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

def get_ring_collision(ring_left:float, ring_right:float, objective:Callable[[float],float], vx:float, eps=1e-8, intervals=10)->float:
    ring_left+=eps # for numeric stability
    ring_right-=eps

    if vx<0: # if we move left, check right first, then left since the ball is coming from the right
        ring_left,ring_right = ring_right,ring_left

    # we need the midpoints if we have a ball crossing the ring twice. The ring is split into {intervals} pieces and we check for a sign change in each of them
    # we start in the direction the ball is moving.
    prev_right = None # to save one evaluation
    for i in range(intervals):
        curr_left= ((intervals-i)*ring_left+i*ring_right) / intervals
        curr_right= ((intervals-i-1)*ring_left+(i+1)*ring_right) / intervals
        y_left = prev_right or objective(curr_left) # if we have prev_right, then we have already evaluated the left side
        y_right = objective(curr_right)
        if y_left*y_right<0:
            return opt.brentq(objective, min(curr_left, curr_right), max(curr_left, curr_right)) # min and max is necessary if vx<0
        prev_right = y_right
    return False

def throw_under_ring(f, x_ring, y_ring, r_ball, vx, ueps=1e-3):
    y_throw_ring_left = f(x_ring-r_ball)
    y_throw_ring_right = f(x_ring+r_ball)
    return vx > 0 and y_throw_ring_left < y_ring + ueps and y_throw_ring_right < y_ring + ueps

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
            # recursive call after hitting the backboard
            plot_throw(f, x_lower=x0, x_upper=x_board-r_ball) # plot throw until the ball hits the backboard
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
    output = True
):
    if output:
        print('simulate_throw')
    # helper parameters for parabola
    a = g/(2*vx**2)
    b = vy/vx-g*x0/vx**2
    c = g*x0**2/(2*vx**2)-vy/vx*x0 + y0

    # ring parameters
    x_ring = x_board-0.45
    y_ring = y_lower

    # define parabola
    def f(x): return a*x**2 + b*x + c

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
        return x0 # return x0 instead of x_plane as ball never hits plane
    x_plane = (-b - np.sign(vx) * np.sqrt(sqrt_part)) / (2*a) # ( das - vor sign ist nötig weil a negativ ist)
    assert x_plane is not None

    if not (x_ring - r_ball < x0 < x_ring + r_ball): # check if ball starts in ring area
        if throw_under_ring(f, x_ring, y_ring, r_ball, vx): # if the ball is under the ring, we can't hit anything
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
        elif x_plane > x_ring + r_ball: # check for basket
            if output:
                print('The ball goes in')
            # goesin = True
            if vx < 0:
                plot_throw(f, x_upper=x0, x_lower = x_plane)
            else:
                plot_throw(f, x_lower=x0, x_upper = x_plane) 
            # plt.show()
            return x_plane
    
    plot_throw(f)
    plt.title("wtf happened")
    plt.show()
    raise(Exception('How did we get here'))

def check_in_basket(x_plane, x_board=4.525, ring_durchmesser=0.45):
    return x_board-ring_durchmesser<x_plane<x_board

def trefferquote(h,alpha,v0,n=100, output=True):
    hits = 0
    for i in range(n): #TODO: replace loop with map
        if output:
            print("\nNew throw\n")
    
        h_rand = h + np.random.uniform(-1,1)*.15
        alpha_rand = alpha + np.random.uniform(-1, 1)*5
        alpha_rand = np.deg2rad(alpha_rand)
        v0_rand = (1+np.random.uniform(-1,1)*.05)*v0

        x0 = np.cos(alpha_rand)*h_rand
        y0 = np.sin(alpha_rand)*h_rand
        vx = np.cos(alpha_rand)*v0_rand
        vy = np.sin(alpha_rand)*v0_rand

        circ_ball = 0.765 + np.random.uniform(-1,1)*.015
        r_ball = circ_ball / (2*np.pi)

        if output:
            print(x0, y0, vx, vy, r_ball)
        x_plane = simulate_throw(x0=x0, y0=y0, vx=vx, vy=vy, r_ball=r_ball, output=output)
        in_basket = check_in_basket(x_plane)
        if output:
            print('x_plane = ', x_plane)
            print(in_basket)
        if in_basket:
            hits += 1
    if output:
        plt.show()
    print("trefferquote: ", hits/n)

# %%
start = timer()
trefferquote(h=1.5, alpha=60, v0=7.5,n=50, output=False)
end = timer()
print("50", end-start)

start = timer()
trefferquote(h=1.5, alpha=60, v0=7.5,n=500, output=False)
end = timer()
print("500", end-start)

start = timer()
trefferquote(h=1.5, alpha=60, v0=7.5, n=5000, output=False)
end = timer()
print("5000", end-start)
exit()


# %% 
# 0.6147413602474181 1.314172781056323 3.239501812265219 6.925294735574873 0.121860690103832
print(simulate_throw(x0 = 0.6147413602474181, y0 = 1.314172781056323, vx = 3.239501812265219, vy = 6.925294735574873, r_ball = 0.121860690103832))
plt.show()
# for vx in np.linspace(3.3,4,100):
#     print(vx, simulate_throw(vy=7,vx=vx))
#     plt.show()
# # # %%
# plt.show()
