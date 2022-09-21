
#####################################################################
#                          General Idea                             #
#####################################################################

# We have a robot with arm length h that throws a ball at the angle alpha with a velocity v:
#
#                        |
#                      🗑️
#         🏀             
#        /
#       /
#      /
#     /
#   🤖|<--   4.525m   -->|

# The idea is now, that the ball has a radius and is a perfect circle. 
# We can then calculate the intersection of the ball with the ring and the backboard by just tracking the center of the ball. 
# Intersection happens by adding a "buffer" of the radius of the ball and bouncing the center of the ball off this buffer.

# We also incorporate air resistance for a more accurate model.

#####################################################################
#                            Authors                                #
#####################################################################

# David Gekeler, Erik Scheurer, Julius Herb, Niklas Hornischer


#####################################################################
#                             Usage                                 #
#####################################################################

# At the bottom of the file, some examples are given.
# You can also import the `korbwurf` function. 
# `hitrate` computes an average for given parameters.
# `simulate_throw` is the core of the simulation that also enables plotting.

#####################################################################
#                             Code                                  #
#####################################################################

#import os
#import multiprocessing
#from multiprocessing import Pool, freeze_support
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math
np.seterr(all="ignore")


def same_signs(a, b):
    assert not math.isnan(a) and not math.isnan(b)
    return (a >= 0) == (b >= 0)


def plot_ring():
    """Plots the ring and the circle around it where the ball bounces off"""

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
    """Plot a section between bounces with a given function f"""
    plot_ring()
    xx = np.linspace(x_lower, x_upper, 1000)
    yy = f(xx)
    plt.plot(xx, yy, line, alpha=0.99, linewidth=1)
    # plt.xlim(3.8, 4.6)
    # plt.ylim(2.7, 4.6)


def get_sign_change_interval(f, a, b, vx, depth=2):
    """Returns the interval where the sign of f changes. 

    We need this for the Ring Collision. If there is no sign change, at the outer bounds of the interval, there may be no or multiple interceptions with the ring collisionbox.
    The brentq algorithm only works if there is a sign change in the interval.
    """
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
    """Returns the x coordinate of the ring collision point """
    ring_left += eps  # for numeric stability
    ring_right -= eps

    a, b = get_sign_change_interval(objective, ring_left, ring_right, vx)
    if a is None:
        return False

    # change interval ordering back for optimize function
    a, b = min(a, b), max(a, b)
    return opt.brentq(objective, a, b, maxiter=100)


def throw_under_ring(f, x_ring, y_ring, r_ball, vx, ueps=1e-3):
    """Checks if the ball is thrown under the ring."""
    y_throw_ring_left = f(x_ring-r_ball)
    return vx > 0 and y_throw_ring_left < y_ring + ueps


def check_ring_collision(f, x0, vx, x_ring, y_ring, r_ball, eps=1e-8):
    """Checks if the ball collides with the ring and return the x coordinate of the collision point"""
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
    g=9.81,  # gravity acceleration [m/s^2]
    rho=1.204,  # density of air [kg/m^3]
    # ball parameters
    r_ball=0.765/(2*np.pi),  # radius of the ball [m]
    m_ball=0.609,  # mass of the ball [kg]
    cw=0.47,  # drag coefficient of a sphere [-]
    # throw parameters
    x0=0,  # x coordinate of start point [m]
    y0=2,  # y coordinate of start point [m]
    vx=2.18,  # x component of throwing velocity [m/s]
    vy=10,  # y component of throwing velocity [m/s]
    # board parameters
    x_board=4.525,  # x coordinate of board position [m]
    y_lower=3.05,  # y coordinate of board position (bottom) [m]
    y_upper=3.95,  # y coordinate of board position (top) [m]
    d_ring=0.45,  # diameter of the ring [m]
    # other parameters
    eps=1e-8,  # used for the bounces to avoid infinite recursion
    ueps=1e-3,  # used for the bounces to avoid infinite recursion
    output=False,  # if debug information should be printed
    plot=False  # if the throw should be plotted
):
    if output:
        print(f'simulate throw: x0 = {x0:.4f}, y0 = {y0:.4f}, vx = {vx:.4f}, vy = {vy:.4f}, r_ball = {r_ball:.4f}, m_ball = {m_ball:.4f}')
    # ring parameters
    x_ring = x_board - d_ring
    y_ring = y_lower

    ###########################################################################
    # 1. Define the analytical solution of the ball throw with air resistance #
    ###########################################################################

    # The air resistance force is modeled based on Newton friction
    # and is given by F = 0.5 * cw * A * rho * v^2 = k * v^2
    k = 0.5 * cw * r_ball**2 * np.pi * rho
    v0 = np.sqrt(vx**2 + vy**2)

    # We have derived the analytical solution of the initial value problem and checked it against
    # a solution by Andreas Lindner (https://www.geogebra.org/m/S4EyHaFa)
    # The analytical solution is only defined for vx >= 0 and x0 = 0, hence for the general case
    # the affine transformation x -> sgn * (x - x0) is employed, where sgn indicates the sign of vx
    vx_ = np.abs(vx)
    sgn = np.sign(vx)
    sin_alpha = vy / v0
    cos_alpha = vx_ / v0
    # Mapping from time to x position (in consideration of the transformation):
    t2x = lambda t: x0 + sgn * m_ball / k * np.log(k * v0 * cos_alpha / m_ball * t + 1)
    # Mapping from x position to time (in consideration of the transformation):
    x2t = lambda x: m_ball / (k * v0 * cos_alpha) * (np.exp(k / m_ball * sgn * (x - x0)) - 1)
    # For convenience we introduce some auxiliary quantities
    v_inf = np.sqrt((m_ball * g) / k)
    c = np.arctan(sin_alpha * v0 / v_inf)
    d = np.sqrt((k * g) / m_ball)
    # Mapping from time to y position (upward throw before peak):
    t2y_up = lambda t: y0 + m_ball / k * (np.log(np.cos(d * t - c)) - np.log(np.cos(c)))
    # Mapping from time to y position (downward throw after peak):
    t2y_down = lambda t: y0 + m_ball / k*(-np.log(np.cosh(d * t - c)) - np.log(np.cos(c)))

    # Calculate the time at which the ball is at the peak
    t_peak = c / d
    # Special case in which the ball flies directly downwards:
    t_peak = np.inf if t_peak < 0 else t_peak

    # The solution is composed of the two functions t2y_up (before peak) and t2y_down (after peak)
    t2y = lambda t: np.where(t < t_peak, t2y_up(t), t2y_down(t))
    # Mapping from x position to y position, i.e. the ball throw trajectory
    f = lambda x: t2y(x2t(x))
    # Derivatives that are needed to compute the velocity at the bounces:
    t2dy_up = lambda t: m_ball / k * d * np.tan(c - d * t)
    t2dy_down = lambda t: m_ball / k * d * np.tanh(c - d * t)
    t2dx = lambda t: m_ball * v0 * cos_alpha / (k * t * v0 * cos_alpha + m_ball)
    t2dy = lambda t: np.where(t < t_peak, t2dy_up(t), t2dy_down(t))
    dy = lambda x: t2dy(x2t(x))
    # We account for the affine transformation by applying the chain rule
    dx = lambda x: sgn * t2dx(x2t(x))

    #######################################################################
    # 2. Compute the intersection of the throw with 3.05m plane and floor #
    #######################################################################

    hitsring = False
    goesover = False
    goesunder = False

    # Compute intersection points with the 3.05m plane
    x_plane = None
    # The two intersection points have been calculated analytically
    t1_plane = (np.arccos(np.cos(c)*np.exp((k/m_ball)*(y_lower - y0))) + c) / d
    t2_plane = (np.arccosh(np.exp(-np.log(np.cos(c))-k/m_ball*(y_lower - y0))) + c) / d
    if t1_plane is not None and t1_plane >= 0 and vx <= 0:
        x_plane = t2x(t1_plane)
    else:
        if t2_plane is not None:
            x_plane = t2x(t2_plane)
        else: # no intersection with the 3.05m plane is found
            x_plane = None
            if output:
                print("The ball is thrown too low")
            if plot:
                plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
            return 0  # return 0 instead of x_plane as ball never hits plane

    # compute intersection points with the floor
    y_floor = 0.0
    x_floor = None
    # The two intersection points have been calculated analytically
    t1_floor = (np.arccos(np.cos(c)*np.exp((k/m_ball)*(y_floor - y0))) + c) / d
    t2_floor = (np.arccosh(np.exp(-np.log(np.cos(c))-k/m_ball*(y_floor - y0))) + c) / d
    if t1_floor is not None and t1_floor >= 0 and vx <= 0:
        x_floor = t2x(t1_floor)
    else:
        x_floor = t2x(t2_floor)

    ######################################################################
    # 3. Check for collision of the throw with ring, backboard or basket #
    ######################################################################

    # check if ball starts in ring area (for recursive calls after ring collision)
    if not (x_ring - r_ball < x0 < x_ring + r_ball):
        # if the ball comes down before the ring, we can't hit anything
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
            r = simulate_throw(g=g, rho=rho, r_ball=r_ball, m_ball=m_ball, cw=cw,
                               x0=x_board-r_ball, y0=y_impact_board, vx=-vx_impact_board, vy=vy_impact_board,
                               x_board=x_board, y_lower=y_lower, y_upper=y_upper, d_ring=d_ring,
                               eps=eps, ueps=ueps, output=output, plot=plot)
    if goesover:
        r = x_plane
    if r is not None:  # if ball hits, return result from recursive call, else return x_plane if the ball goes over the backboard
        return r

    # interception of the ball with the ring needs to be checked, as ball goes under the backboard or moves left
    hitsring, x_impact = check_ring_collision(f, x0, vx, x_ring, y_ring, r_ball)
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
        # (Note that this is only an approximation as a real basketball would lose energy here)
        v_bounced = v - 2*v_parallel
        # move ball away from ring a bit to avoid infinite recursion
        x_impact -= eps*e[0]

        if plot:
            if vx > 0:
                plot_throw(f, x_upper=x_impact, x_lower=x0)
            else:
                plot_throw(f, x_lower=x_impact, x_upper=x0)
        # recursive call after hitting the ring
        return simulate_throw(g=g, rho=rho, r_ball=r_ball, m_ball=m_ball, cw=cw,
                              x0=x_impact, y0=y_impact, vx=v_bounced[0], vy=v_bounced[1],
                              x_board=x_board, y_lower=y_lower, y_upper=y_upper, d_ring=d_ring,
                              eps=eps, ueps=ueps, output=output, plot=plot)
    # else:
    if goesunder or vx < 0:
        # check for airball (i.e. ball hits nothing)
        if x_plane < x_ring - r_ball:
            if output:
                print('The ball hits nothing')
            if plot:
                if vx > 0:
                    plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
                else:
                    plot_throw(f, x_upper=x0, x_lower=x_floor, line='m-')
            return x_plane
        elif x_board - r_ball > x_plane > x_ring + r_ball - ueps:  # check if ball is in the basket
            goesin = True
            if output:
                print('The ball goes in')
            if plot:
                if vx < 0:
                    plot_throw(f, x_upper=x0, x_lower=x_plane, line='g-')
                else:
                    plot_throw(f, x_lower=x0, x_upper=x_plane, line='g-')
            return x_plane

    if plot:
        if vx > 0:
            plot_throw(f, x_lower=x0, x_upper=x_floor, line='m-')
        else:
            plot_throw(f, x_upper=x0, x_lower=x_floor, line='m-')
    return 0  # return 0 instead of x_plane as ball never hits plane


def check_in_basket(x_plane, x_board=4.525, d_ring=0.45):
    """Check if the x coordinate at the 3.05m height is in the basket."""
    a = x_board - d_ring < x_plane
    b = x_plane < x_board
    c = np.logical_and(a, b)
    return c.astype(int)


def mapfunc(x0, y0, vx, vy, r_ball, m_ball, output, plot):
    """Wrapper function for simulate_throw to be used with map."""
    x_plane = simulate_throw(x0=x0, y0=y0, vx=vx, vy=vy, r_ball=r_ball, m_ball=m_ball, output=output, plot=plot)
    if output:
        print('-'*20)
    return x_plane


def hit_rate(h, alpha, v0, n=100, output=False, plot=False, conv=False):
    """Compute the hit rate for a given height h, angle alpha and starting velocity v0, by sampling n throws."""
    hs = np.zeros(n) + h
    alphas = np.zeros(n) + alpha

    h_rands = hs + np.random.uniform(-1, 1, size=n) * 0.15
    alpha_rands = alphas + np.random.uniform(-1, 1, size=n) * 5
    alpha_rands = np.deg2rad(alpha_rands)
    v0_rands = (1 + np.random.uniform(-1, 1, size=n) * 0.05) * v0

    x0s = np.cos(alpha_rands) * h_rands
    y0s = np.sin(alpha_rands) * h_rands
    vxs = np.cos(alpha_rands) * v0_rands
    vys = np.sin(alpha_rands) * v0_rands

    circ_balls = 0.765 + np.random.uniform(-1, 1, size=n) * 0.015
    r_balls = circ_balls / (2*np.pi)

    m_balls = 0.609 + np.random.uniform(-1, 1, size=n) * 0.015

    """ # multiprocessing (not necessary)
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
    hit_rate = hits / n
    rate_convergence = np.cumsum(in_baskets) / np.arange(1, n+1)

    if conv:
        return hit_rate, rate_convergence
    else:
        return hit_rate


def korbwurf(
    abweichung_wurfarmhoehe=0.0,
    abweichung_abwurfwinkel=0.0,
    abweichung_beschleunigung=0.0,
    abweichung_geschwindigkeit=0.0,
    ballradius=0.765/(2*np.pi),
    ballgewicht=0.609
):
    """Simulate a basketball throw. The deviation of the throw height, the throw angle,
    the acceleration and the velocity from their respective optimal values can be set."""
    best_h = 2.0                # optimal throw height
    best_alpha = 60.68          # optimal throw angle
    best_velocity = 7.37        # optimal velocity
    h = best_h + abweichung_wurfarmhoehe
    alpha = best_alpha + abweichung_abwurfwinkel    # calculation of parameters with added deviation, as specified in the function arguments
    v0 = best_velocity + abweichung_geschwindigkeit
    rad_alpha = np.deg2rad(alpha)
    return np.array(simulate_throw( # return the x coordinate of the ball at a height of 3.05m for the given parameters
        r_ball=ballradius,
        m_ball=ballgewicht,
        x0=h * np.cos(rad_alpha),
        y0=h * np.sin(rad_alpha),
        vx=v0 * np.cos(rad_alpha),
        vy=v0 * np.sin(rad_alpha),
        output=False,
        plot=False
    ))


# %%
if __name__ == '__main__':

    # Example call of `korbwurf` function (without uncertainties)
    pos = korbwurf(0, 0, 0, 0, ballradius=0.765/(2*np.pi), ballgewicht=0.609)
    print(pos)

    # Plot the corresponding throw (without uncertainties)
    h, alpha, v0 = 2.0, 60.68, 7.37  # optimal parameters
    rad_alpha = np.deg2rad(alpha)
    fig, ax = plt.subplots()
    simulate_throw(x0=h * np.cos(rad_alpha), y0=h * np.sin(rad_alpha),
                   vx=v0 * np.cos(rad_alpha), vy=v0 * np.sin(rad_alpha),
                   output=False, plot=True
                   )
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, 5), ylim=(1, 5))
    ax.set_title('Throw with optimal parameters (without uncertainties)')
    fig.tight_layout()
    plt.show()

    # Plot throw with uncertainties (using the same configuration)
    fig, ax = plt.subplots()
    hit_rate(h, alpha, v0, n=1, output=True, plot=True)
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, 5), ylim=(1, 5))
    ax.set_title('Throw with optimal parameters (with uncertainties)')
    fig.tight_layout()
    plt.show()

    # Plot 100 throws with uncertainties (using the same configuration)
    fig, ax = plt.subplots()
    print('hit rate for 100 samples:', hit_rate(h, alpha, v0, n=100, output=False, plot=True))
    ax.set_aspect('equal', 'box')
    ax.set(xlim=(0, 5), ylim=(1, 5))
    ax.set_title('100 throws with optimal parameters (with uncertainties)')
    fig.tight_layout()
    plt.show()

    # Calculate the hit rate for a different count of uncertainty samples
    print('hit rate for 1000 samples:', hit_rate(h, alpha, v0, n=1000, output=False, plot=False))
    # After 20000 samples the hit rate should be converged to ~0.47
    print('Computing the hit rate for 20000 samples')
    print('This can take a while on slow computers...')
    rate, rate_conv = hit_rate(h, alpha, v0, n=20000, output=False, plot=False, conv=True)
    print('hit rate for 20000 samples:', rate)
    plt.plot(np.arange(1, len(rate_conv)+1)[100:], rate_conv[100:])
    plt.title('Hit rate convergence')
    plt.xlabel('samples')
    plt.ylabel('hit rate')
    plt.show()
    exit()
