# %%
import os
#import multiprocessing
#from multiprocessing import Pool, freeze_support
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import Axes3D
np.seterr(all="ignore")


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
    plt.plot((x_ring), (y_ring), 'ro', markersize=2)
    xx = np.linspace(x_ring-r_ball+1e-7, x_ring+r_ball-1e-7, 100)
    def ring(x): return np.sqrt((r_ball)**2-(x-x_ring)**2) + y_ring
    rr = ring(xx)
    plt.plot(xx, rr, 'r--')
    # backboard
    plt.plot([x_board]*2, (y_ring, y_board), 'r')
    plt.plot([x_board-r_ball]*2, (y_ring, y_board), 'r--')
    plt.plot([x_ring, x_board], [y_ring]*2, '-',
             linewidth=1, alpha=0.9, color='orange')


def plot_throw(f, x_lower=0, x_upper=5, line='b-'):
    plot_ring()
    xx = np.linspace(x_lower, x_upper, 1000)
    yy = f(xx)
    plt.plot(xx, yy, line, alpha=0.99, linewidth=1)
    # plt.xlim(3.8, 4.6)
    # plt.ylim(2.7, 4.6)


def plot(f, ring, x_board, y_lower, y_upper, x_ring, y_ring, sol, r_ball, y_board, e, v, v_parallel, v_bounced):
    xx = np.linspace(0, 5, 1000)
    rr = ring(xx)

    plt.figure(figsize=(5, 5))
    # backboard
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
    if vx < 0:  # if we move left, check right first, then left since the ball is coming from the right
        a, b = b, a

    ya, yb = f(a), f(b)
    if not same_signs(ya, yb):  # if a,b have sign change, return them
        return a, b

    m = (a+b)/2  # else, insert middle point and begin iterative search
    points = [a, m, b]
    values = [ya, f(m), yb]

    for k in range(depth):
        i = 1
        while i < len(values):
            # return interval if sign changes
            if not same_signs(values[i-1], values[i]):
                return points[i-1], points[i]
            else:
                # or insert middle point as new interval boundary candidate
                points.insert(i, (points[i-1]+points[i])/2)
                values.insert(i, f(points[i]))
                i += 1
            i += 1

    return None, None  # if no sign change found, return None


def get_ring_collision(ring_left: float, ring_right: float, objective: Callable[[float], float], vx: float, eps=1e-8, tol=1e-5) -> float:
    ring_left += eps  # for numeric stability
    ring_right -= eps

    a, b = get_sign_change_interval(objective, ring_left, ring_right, vx)
    if a is None:
        return False

    # change interval ordering back for optimize function
    a, b = min(a, b), max(a, b)
    return opt.brentq(objective, a, b, maxiter=100)


def throw_under_ring(f, x_ring, y_ring, r_ball, vx, ueps=1e-3):
    y_throw_ring_left = f(x_ring-r_ball)
    return vx > 0 and y_throw_ring_left < y_ring + ueps


def check_ring_collision(f, x0, vx, x_ring, y_ring, r_ball, eps=1e-8):
    def ring(x):
        assert x > x_ring-r_ball and x < x_ring+r_ball
        return np.sqrt(r_ball**2-(x-x_ring)**2) + y_ring

    def obj(x): return f(x)-ring(x)

    # if we bounced from the ring in the last recursion step
    if x_ring-r_ball + 2*eps < x0 < x_ring + r_ball - 2*eps:
        if vx > 0:  # we bounced to the right
            x_impact = get_ring_collision(x0, x_ring+r_ball, obj, vx)
            # if return value is a number, hitsring is true
            hitsring = bool(x_impact)
        else:  # vx < 0, we bounced to the left
            # x_impact, sol = opt.bisect(obj, x_ring - r_ball + eps, x0,full_output=True)
            x_impact = get_ring_collision(x_ring-r_ball, x0, obj, vx)
            hitsring = bool(x_impact)
    else:
        x_impact = get_ring_collision(x_ring-r_ball, x_ring+r_ball, obj, vx)
        hitsring = bool(x_impact)

    return hitsring, x_impact


def simulate_throw(
    # general parameters
    g = 9.81, # gravity acceleration [m/s^2]
    rho = 1.204, # density of air [kg/m^3]
    # ball paramaters
    r_ball = 0.765/(2*np.pi), # radius of the ball [m]
    m_ball = 0.609, # mass of the ball [kg]
    cw = 0.47, # drag coefficient of a sphere [-]
    # throw parameters
    x0 = 0, # x coordinate of start point [m]
    y0 = 2, # y coordinate of start point [m]
    vx = 2.18, # x component of throwing velocity [m/s]
    vy = 10, # y component of throwing velocity [m/s]
    # board parameters
    x_board = 4.525, # x coordinate of board position [m]
    y_lower = 3.05, # y coordinate of board position (bottom) [m]
    y_upper = 3.95, # y coordinate of board position (top) [m]
    d_ring = 0.45, # diameter of the ring [m]
    # other parameters
    eps = 1e-8, # used for the bounces to avoid infinite recursion
    ueps = 1e-3, # used for the bounces to avoid infinite recursion
    output = False, # if debug information should be printed
    plot = False # if the throw should be plotted
):
    if output:
        print(f'simulate throw: x0 = {x0:.4f}, y0 = {y0:.4f}, vx = {vx:.4f}, vy = {vy:.4f}, r_ball = {r_ball:.4f}, m_ball = {m_ball:.4f}')
    # ring parameters
    x_ring = x_board - d_ring
    y_ring = y_lower

    #########################################################################
    # 1. Define the analytic solution of the ball throw with air resistance #
    #########################################################################

    # The air resistance force is modeled as Newton friction
    # and is given by F = 0.5 * cw * r * A * rho * v^2 = k * v^2
    k = 0.5 * cw * r_ball**2 * np.pi * rho
    v0 = np.sqrt(vx**2 + vy**2)

    # For the derivation of the analytic solution of the initial value problem we followed
    # https://matheplanet.com/default3.html?call=article.php?sid=735
    # The analytic solution is only defined for vx >= 0, hence for vx < 0 a transformation is used
    vx_ = np.abs(vx)
    sgn = vx / vx_
    sin_alpha = vy / v0
    cos_alpha = vx_ / v0
    # For convenience we introduced an auxiliary quantity c
    c = np.arctan(np.sqrt(k/(m_ball*g))*v0*sin_alpha)
    # mapping from time to x position:
    t2x = lambda t: x0 + sgn * m_ball/k*np.log(k*v0*cos_alpha/m_ball*t + 1)
    # mapping from x position to time:
    x2t = lambda x: m_ball/(k*v0*cos_alpha)*(np.exp(k/m_ball*x) - 1)
    # mapping from time to y position (upward throw):
    t2y_up = lambda t: y0 + m_ball/k*(np.log(np.cos(np.sqrt(k*g/m_ball)*t - c)) - np.log(np.cos(c)))
    # mapping from time to y position (downward throw):
    t2y_down = lambda t: y0 + m_ball/k*(-np.log(np.cosh(np.sqrt(k*g/m_ball)*t - c)) - np.log(np.cos(c)))

    # calculate the peak of the ball throw
    t_peak = np.sqrt(m_ball / (k*g)) * c
    t_peak = 0 if t_peak < 0 else t_peak
    x_peak = t2x(t_peak)

    # The final solution is composed of the two functions t2y_up (before peak) and t2y_down (after peak)
    ft = lambda t: np.where(t < t_peak, t2y_up(t), t2y_down(t))
    # Transformation to account for vx < 0
    f = lambda x: ft(x2t(sgn * (x - x0)))
    # Derivatives that are needed to compute the velocity at the bounces:
    dy_up = lambda x: m_ball/k*np.sqrt(k*g/m_ball)*np.tan(c - x2t(x)*np.sqrt(g*k/m_ball))
    dy_down = lambda x: m_ball/k*np.sqrt(k*g/m_ball)*np.tanh(c - x2t(x)*np.sqrt(g*k/m_ball))
    dx_ = lambda x: m_ball*v0*cos_alpha/(k*x2t(x)*v0*cos_alpha+m_ball)
    dy_ = lambda x: np.where(x < x_peak, dy_up(x), dy_down(x))
    dy = lambda x: dy_(sgn * (x - x0))
    dx = lambda x: sgn * dx_(sgn * (x - x0))
    y_peak = f(x_peak)

    #######################################################################
    # 2. Compute the intersection of the throw with 3.05m plane and floor #
    #######################################################################
    hitsring = False
    goesover = False
    goesunder = False
    goesin = False

    if y_peak < y_lower:
        if output:
            print("The ball is thrown too low")
            plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
        return 0  # return 0 instead of x_plane as ball never hits plane
    
    # compute intersection points with the 3.05m plane
    x_plane = None
    t1_plane = np.sqrt(m_ball/(g*k))*(np.arccos(np.cos(c)*np.exp((k/m_ball)*(y_lower - y0))) + c)
    t2_plane = np.sqrt(m_ball/(g*k))*(np.arccosh(np.exp(-np.log(np.cos(c))-k/m_ball*(y_lower - y0))) + c)
    if t1_plane is not None and t1_plane >= 0 and vx <= 0:
        x_plane = t2x(t1_plane)
    else:
        if t2_plane is not None:
            x_plane = t2x(t2_plane)
        else:
            try:
                print('fallback: optimization')
                x_plane = opt.newton(lambda x: f(x) - y_lower, x0=(x0+3*x_ring)/4)
            except:
                x_plane = None
                if output:
                    print("The ball is thrown too low")
                if plot:
                    plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
                return 0  # return 0 instead of x_plane as ball never hits plane

    # compute intersection points with the floor
    y_floor = 0.0
    x_floor = None
    t1_floor = np.sqrt(m_ball/(g*k))*(np.arccos(np.cos(c)*np.exp((k/m_ball)*(y_floor - y0))) + c)
    t2_floor = np.sqrt(m_ball/(g*k))*(np.arccosh(np.exp(-np.log(np.cos(c))-k/m_ball*(y_floor - y0))) + c)
    if t1_floor is not None and t1_floor >= 0 and vx <= 0:
        x_floor = t2x(t1_floor)
    else:
        if t2_floor is not None:
            x_floor = t2x(t2_floor)
        else:
            try:
                print('fallback: optimization')
                x_floor = opt.newton(lambda x: f(x) - y_floor, x0=(x0+3*x_ring)/4)
            except:
                x_floor = None

    assert x_plane is not None

    ######################################################################
    # 3. Check for collision of the throw with ring, backboard or basket #
    ######################################################################
    # check if ball starts in ring area
    if not (x_ring - r_ball < x0 < x_ring + r_ball):
        # if the ball is under the ring, we can't hit anything
        if throw_under_ring(f, x_ring, y_ring, r_ball, vx, ueps):
            if output:
                print("The ball is thrown too low")
            if plot:
                plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
            return x_plane

    # check if the ball hits the backboard, goes over it, or goes under it
    r = None
    goesover = False
    goesunder = False
    if vx > 0:  # only check backboard if ball is moving to the right
        # interception of the ball with the backboard
        y_impact_board = f(x_board-r_ball)
        if y_upper < y_impact_board:
            if output:
                print('The ball goes over the backboard')
            if plot:
                # plot throw until the ball hits the floor
                plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
            goesover = True
        elif y_lower > y_impact_board:
            goesunder = True
        else:  # Cant hit backboard with negative x velocity
            if output:
                print('The ball hits the backboard')
            if plot:
                # plot throw until the ball hits the backboard
                plot_throw(f, x_lower=x0, x_upper=x_board-r_ball)
            # recursive call after hitting the backboard
            vy_impact_board = dy(x_board)
            vx_impact_board = dx(x_board)
            r = simulate_throw(g = g, rho = rho, r_ball = r_ball, m_ball = m_ball, cw = cw,
                x0 = x_board-r_ball, y0 = y_impact_board, vx = -vx_impact_board, vy = vy_impact_board,
                x_board = x_board, y_lower = y_lower, y_upper = y_upper, d_ring = d_ring,
                eps = eps, ueps = ueps, output = output, plot = plot)
    if goesover:
        r = x_plane
    if r is not None:  # if ball hits, return result from recursive call, else return x_plane if the ball goes over the backboard
        return r

    # interception of the ball with the ring needs to be checked, as ball goes under the backboard or moves left
    hitsring, x_impact = check_ring_collision(
        f, x0, vx, x_ring, y_ring, r_ball)
    if hitsring:
        if output:
            print('The ball hits the ring')
        y_impact = f(x_impact)
        # e is the distance vector from impact location to the ring
        e = np.array((x_ring-x_impact, y_ring-y_impact))
        e /= np.linalg.norm(e)

        # compute velocity at impact
        v = np.array((dx(x_impact), dy(x_impact)))

        # velocity parallel to normed vector e
        v_parallel = np.dot(v, e)*e
        # elastic collision
        # (Note that this is only an approximation as a real basketball would loose energy here)
        v_bounced = v - 2*v_parallel
        # move ball away from ring a bit to avoid infinite recursion
        x_impact -= eps*e[0]

        if plot:
            if vx > 0:
                plot_throw(f, x_upper=x_impact, x_lower=x0)
            else:
                plot_throw(f, x_lower=x_impact, x_upper=x0)
        # recursive call after hitting the ring
        return simulate_throw(g = g, rho = rho, r_ball = r_ball, m_ball = m_ball, cw = cw,
                x0 = x_impact, y0 = y_impact, vx = v_bounced[0], vy = v_bounced[1],
                x_board = x_board, y_lower = y_lower, y_upper = y_upper, d_ring = d_ring,
                eps = eps, ueps = ueps, output = output, plot = plot)
    # else:
    if goesunder or vx < 0:
        if x_plane < x_ring - r_ball:  # check for airball (i.e. ball hits nothing)
            if output:
                print('AIRBALL')
            if plot:
                if vx > 0:
                    plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
                else:
                    plot_throw(f, x_upper=x0, x_lower=x_floor, line='m-')
            return x_plane
        elif x_board - r_ball > x_plane > x_ring + r_ball - ueps:  # check for basket
            goesin = True
            if output:
                print('The ball goes in')
            if plot:
                if vx < 0:
                    plot_throw(f, x_upper=x0, x_lower=x_plane, line='g-')
                else:
                    plot_throw(f, x_lower=x0, x_upper=x_plane, line='g-')
            return x_plane

    if np.abs(dy(x_plane)) > 1e3:
        if output:
            print("fail")
        return 0  # TODO: figure out how to deal with very steep parabolas

    if plot:
        if vx > 0:
            plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
        else:
            plot_throw(f, x_upper=x0, x_lower=x_floor, line='m-')
    return 0  # return 0 instead of x_plane as ball never hits plane


def check_in_basket(x_plane, x_board=4.525, d_ring=0.45):
    a = x_board - d_ring < x_plane
    b = x_plane < x_board
    c = np.logical_and(a, b)
    return c.astype(int)


def mapfunc(x0, y0, vx, vy, r_ball, m_ball, output, plot):
    x_plane = simulate_throw(x0=x0, y0=y0, vx=vx, vy=vy, r_ball=r_ball, m_ball=m_ball, output=output, plot=plot)
    if output:
        print('-'*20)
    return x_plane


def hit_rate(h, alpha, v0, n=100, output=False, plot=False, conv=False):
    hs = np.zeros(n)+h
    alphas = np.zeros(n)+alpha
    v0s = np.zeros(n)+v0

    h_rands = hs + np.random.uniform(-1, 1, size=n) * 0.15
    alpha_rands = alphas + np.random.uniform(-1, 1, size=n) * 5
    alpha_rands = np.deg2rad(alpha_rands)
    v0_rands = (1 + np.random.uniform(-1, 1, size=n)*.05) * v0

    x0s = np.cos(alpha_rands) * h_rands
    y0s = np.sin(alpha_rands) * h_rands
    vxs = np.cos(alpha_rands) * v0_rands
    vys = np.sin(alpha_rands) * v0_rands

    circ_balls = 0.765 + np.random.uniform(-1, 1, size=n) * 0.015
    r_balls = circ_balls / (2*np.pi)

    m_balls = 0.609 + np.random.uniform(-1, 1, size=n) * 0.015

    """
    if plot:
        x_planes = np.asarray(list(map(mapfunc, x0s, y0s, vxs, vys, r_balls, m_balls, [output]*n, [plot]*n)))
    else:
        available_cores = 4#multiprocessing.cpu_count()
        with Pool(available_cores) as p:
            x_planes = np.asarray(p.starmap(mapfunc, zip(x0s, y0s, vxs, vys, r_balls, m_balls, [output]*n, [plot]*n)))
    """
    x_planes = np.asarray(
        list(map(mapfunc, x0s, y0s, vxs, vys, r_balls, m_balls, [output]*n, [plot]*n)))

    in_baskets = check_in_basket(x_planes)
    hits = np.sum(in_baskets)
    rate_convergence = np.cumsum(in_baskets) / np.arange(1, n+1)

    if output:
        print("hit rate: ", hits/n)
    if conv:
        return hits/n, rate_convergence
    else:
        return hits/n


def plot3d_h_alpha(v0=7):  # fixed v0 for now
    rates = []
    hs = np.linspace(1, 2, 10)
    alphas = np.linspace(30, 90, 30)
    n = 100  # number of simulations per combination
    start = timer()
    for h in hs:  # for random input values
        rates.append([])
        for alpha in alphas:
            rates[-1].append(hit_rate(h=h, alpha=alpha, v0=v0, n=n, output=False))

    end = timer()
    print("Time: ", end-start)

    if len(rates) > 1:  # plot the hit reates vs. number of iterations
        plt.show()
        ax = plt.axes(projection="3d")
        hs, alphas = np.meshgrid(hs, alphas)
        ax.plot_surface(hs, alphas, np.array(rates).T, cmap='viridis', edgecolor='none')
        ax.set_xlabel('h [m]')
        ax.set_ylabel(r'\alpha [°]')
        ax.set_zlabel('hit rate [-]')
        plt.show()

def plot3d_v0_alpha(h=2.0, n=500):  # fixed h for now
    rates = []
    v0s = np.linspace(7, 9.5, 15)
    alphas = np.linspace(55, 75, 16)
    start = timer()
    for v0 in v0s:  # for random input values
        rates.append([])
        for alpha in alphas:
            rates[-1].append(hit_rate(h=h, alpha=alpha, v0=v0, n=n, output=False))

    end = timer()
    print("Time: ", end-start)

    if len(rates) > 1:  # plot the hit reates vs. number of iterations
        plt.show()
        ax = plt.axes(projection="3d")
        hs, alphas = np.meshgrid(v0s, alphas)
        ax.plot_surface(hs, alphas, np.array(rates).T,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$v_0$ [m/s]')
        ax.set_ylabel(r'$\alpha$ [°]')
        ax.set_zlabel('hit rate [-]')
        plt.show()
    print(rates)

def plot3d_v0_alpha_fine(h=2.0, n=5000):  # fixed h for now
    rates = []
    v0s = np.linspace(7.3,7.5,9)#np.linspace(8, 9.5, 15)#np.linspace(7.1, 7.8, 15) - 1,2
    alphas = np.linspace(60.5,61.5,11)#np.linspace(55, 75, 16)#np.linspace(58, 64, 16)
    start = timer()
    for v0 in v0s:  # für zufällige input werte
        rates.append([])
        for alpha in alphas:
            rates[-1].append(hit_rate(h=h, alpha=alpha, v0=v0, n=n, output=False, plot=False))

    end = timer()
    print("Time: ", end-start)

    if len(rates) > 1:  # plot the hit reates vs. number of iterations
        plt.show()
        ax = plt.axes(projection="3d")
        hs, alphas = np.meshgrid(v0s, alphas)
        ax.plot_surface(hs, alphas, np.array(rates).T,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$v_0$ [m/s]')
        ax.set_ylabel(r'$\alpha$ [°]')
        ax.set_zlabel('hit rate [-]')
        plt.show()
    print(rates)

def plot3d_v0_alpha_fine2(h=2.0, n=5000):  # fixed h for now
    rates = []
    v0s = np.linspace(7.35,7.4,11)
    alphas = np.linspace(60.5,60.7,11)
    start = timer()
    for v0 in v0s:  # für zufällige input werte
        rates.append([])
        for alpha in alphas:
            rates[-1].append(hit_rate(h=h, alpha=alpha, v0=v0, n=n, output=False, plot=False))

    end = timer()
    print("Time: ", end-start)

    if len(rates) > 1:  # plotte verschiedene trefferquoten über anzahl der iterationen
        plt.show()
        ax = plt.axes(projection="3d")
        hs, alphas = np.meshgrid(v0s, alphas)
        ax.plot_surface(hs, alphas, np.array(rates).T,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel(r'$v_0$ [m/s]')
        ax.set_ylabel(r'$\alpha$ [°]')
        ax.set_zlabel('hit rate [-]')
        plt.show()
    print(rates)


def korbwurf(
    abweichung_wurfarmhoehe = 0.0,
    abweichung_abwurfwinkel = 0.0, 
    abweichung_beschleunigung = 0.0,
    abweichung_geschwindigkeit = 0.0,
    ballradius = 0.765/(2*np.pi),
    ballgewicht = 0.609
    ):
    best_h = 2.0
    best_alpha = 60.68
    best_velocity = 7.37
    h = best_h + abweichung_wurfarmhoehe
    alpha = best_alpha + abweichung_abwurfwinkel
    v0 = best_velocity + abweichung_geschwindigkeit
    rad_alpha = np.deg2rad(alpha)
    return simulate_throw(
        r_ball = ballradius,
        m_ball = ballgewicht,
        x0 = h * np.cos(rad_alpha),
        y0 = h * np.sin(rad_alpha),
        vx = v0 * np.cos(rad_alpha),
        vy = v0 * np.sin(rad_alpha),
        output = False,
        plot = False
    )


# %%
if __name__ == '__main__':
    #freeze_support()
    # Example call of `korbwurf` function:
    pos = korbwurf(0, 0, 0, 0, ballradius=0.765/(2*np.pi), ballgewicht=0.609)
    print(pos)

    # Plot the corresponding throw (without uncertainties)
    h, alpha, v0 = 2.0, 60.68, 7.37 # optimal parameters
    rad_alpha = np.deg2rad(alpha)
    fig, ax = plt.subplots()
    simulate_throw(x0 = h * np.cos(rad_alpha), y0 = h * np.sin(rad_alpha),
        vx = v0 * np.cos(rad_alpha), vy = v0 * np.sin(rad_alpha),
        output = False, plot = True
    )
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0,5), ylim=(1,5))
    fig.tight_layout()
    plt.show()

    # Plot 100 throws with uncertainties (using the same configuration)
    fig, ax = plt.subplots()
    print('hit rate for 100 samples:', hit_rate(h, alpha, v0, n=100, output=False, plot=True))
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0,5), ylim=(1,5))
    fig.tight_layout()
    plt.show()
    
    # Calculate the hit rate for a different count of uncertainty samples
    # This can take a while on slow computers
    print('hit rate for 1000 samples:', hit_rate(h, alpha, v0, n=1000, output=False, plot=False))
    print('hit rate for 10000 samples:', hit_rate(h, alpha, v0, n=10000, output=False, plot=False))
    # After 100000 samples the hit rate should be converged to ~0.46-0.47
    rate, rate_conv = hit_rate(h, alpha, v0, n=100000, output=False, plot=False, conv=True)
    print('hit rate for 100000 samples:', rate)
    plt.plot(np.arange(1, len(rate_conv)+1)[1000:], rate_conv[1000:])
    plt.title('Hit rate convergence')
    plt.xlabel('samples')
    plt.ylabel('hit rate')
    plt.show()
    exit()
