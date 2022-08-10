# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math

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
    xx = np.linspace(3.3, 4.3, 1000)
    def ring(x): return np.sqrt((r_ball)**2-(x-x_ring)**2) + y_ring
    rr = ring(xx)
    plt.plot(xx, rr, 'r--')
    # brett
    plt.plot([x_board]*2,(y_ring,y_board), 'r')
    plt.plot([x_board-r_ball]*2,(y_ring,y_board), 'r--')
    plt.plot([x_ring, x_board],[y_ring]*2, '-', linewidth=1, alpha=0.9, color='orange')

def plot_throw(f,x_lower=0,x_upper=5):
    plot_ring()
    xx = np.linspace(x_lower,x_upper, 1000)
    yy = f(xx)
    plt.plot(xx, yy, 'b-', alpha=1.0, linewidth=1)

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

def simulate_throw(
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
    eps=1e-8
):
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

    # interception of the ball with 3.05m 
    x_plane = (-b - np.sign(vx) * np.sqrt(b**2-4*a*(c-y_lower))) / (2*a) # ( das - vor sign ist nötig weil a negativ ist)
    assert x_plane is not None

    y_throw_ring_left = f(x_ring-r_ball)
    y_throw_ring_right = f(x_ring+r_ball)
    if vx > 0 and y_throw_ring_left < y_ring and y_throw_ring_right < y_ring: # check if the ball goes under the ring
        print("The ball is thrown too low")
        plot_throw(f,x_lower=x0)
        # plt.show()
        return x_plane    

    if vx > 0: # only check backboard if ball is moving to the right
        # interception of the ball with the backboard
        y_board = f(x_board-r_ball)
        if y_upper < y_board:
            print('The ball goes over the backboard')
            plot_throw(f,x_lower=x0) # plot throw until the ball misses the backboard
            # plt.show()
            goesover = True
            # TODO Implement bounce on top of board
            return x_plane
        elif y_lower > y_board:
            # print('The ball goes under the backboard')
            goesunder = True
        else: # Cant hit backboard with negative x velocity
            print('The ball hits the backboard')
            # recursive call after hitting the backboard
            plot_throw(f, x_lower=x0, x_upper=x_board-r_ball) # plot throw until the ball hits the backboard
            vy_impact = g*(x_board-r_ball-x0)/vx + vy
            r = simulate_throw(
                g=g,
                x0=x_board-r_ball,
                y0=y_board,
                vy=vy_impact,
                vx=-vx,
                r_ball=r_ball,
                x_board=x_board,
                y_lower=y_lower,
                y_upper=y_upper,
                eps=eps
            )
            return r

    
    # interception of the ball with the ring
    def ring(x): return np.sqrt(r_ball**2-(x-x_ring)**2) + y_ring
    def obj(x): return f(x)-ring(x)
    x_spitze = -b/(2*a) # hoechster punkt der parabel

    if (vx < 0 and x_spitze > x_ring - r_ball) or (vx>0 and x_spitze < x_ring + r_ball): # only consider ball hitting the ring after highest point of parabola
        if x_ring-r_ball < x0 < x_ring + r_ball: # if we bounced from the ring in the last recursion step
            if vx > 0 and not same_signs(obj(x_ring + r_ball - eps), obj(x0)): # wenn die Vorzeichen die selben sind, dann gibt es keine Kollision
                x_impact, sol = opt.bisect(obj, x0, x_ring+r_ball -eps,full_output=True)
                hitsring = sol.converged
            elif vx < 0 and not same_signs(obj(x_ring-r_ball + eps), obj(x0)):
                x_impact, sol = opt.bisect(obj, x_ring - r_ball + eps, x0,full_output=True)
                hitsring = sol.converged
            else: # wir sind abgeprallt und kein vorzeichenwechsel=> keine kollision mit ring
                hitsring=False
        else:
            if not same_signs(obj(x_ring - r_ball + eps), obj(x_ring + r_ball - eps)): # This should be the default for the first call
                x_impact, sol = opt.bisect(obj, x_ring-r_ball+eps, x_ring+r_ball-eps, full_output=True)
                hitsring = sol.converged
            else: # Noch eine Fallunterscheidung für wenn man den kreis um ring zweimal schneidet
                if vx < 0 and not same_signs(obj(x_ring + r_ball - eps), obj(x_ring)): # wenn die Vorzeichen die selben sind, dann gibt es keine Kollision
                    x_impact, sol = opt.bisect(obj, x_ring, x_ring+r_ball - eps,full_output=True)
                    hitsring = sol.converged
                elif vx > 0 and not same_signs(obj(x_ring-r_ball+eps), obj(x_ring)):
                    x_impact, sol = opt.bisect(obj, x_ring - r_ball + eps, x_ring,full_output=True)
                    hitsring = sol.converged
                else: # wir sind abgeprallt und kein vorzeichenwechsel=> keine kollision mit ring
                    hitsring=False
    
    if hitsring and ((vx>0 and x_impact > x_spitze) or (vx<0 and x_impact < x_spitze)): # if the ball hits the ring after the highest point of the parabola
        print('The ball hits the ring')
        # x_impact = sol.x[0]
        y_impact = f(x_impact)

        e = np.array((x_ring-x_impact, y_ring-y_impact))
        e /= np.linalg.norm(e)
        # print('e = ', e)

        # vertical speed at impact
        vy_impact = g*(x_impact-x0)/vx + vy
        v = np.array((vx, vy_impact))
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
            eps=eps
        )

    # else:
    
    if goesunder or vx < 0:
        if x_plane < x_ring - r_ball:
            print('AIRBALL')
            if vx > 0:
                plot_throw(f, x_lower=x0)
            else:
                plot_throw(f, x_upper=x0)
            # plt.show()
            return x_plane
        elif x_plane > x_ring + r_ball:
            print('The ball goes in')
            goesin = True
            if vx < 0:
                plot_throw(f, x_upper=x0, x_lower = x_plane)
            else:
                plot_throw(f, x_lower=x0, x_upper = x_plane) 
            # plt.show()
            return x_plane
    raise(Exception('How did we get here'))

def check_in_basket(x_plane, x_board=4.525, ring_durchmesser=0.45):
    return x_board-ring_durchmesser<x_plane<x_board

def trefferquote(h,alpha,v0,n=100):
    hits = 0
    for i in range(n):
    
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

        x_plane = simulate_throw(x0=x0, y0=y0, vx=vx, vy=vy, r_ball=r_ball)
        print(x0, y0, vx, vy, r_ball)
        print('x_plane = ', x_plane)
        in_basket = check_in_basket(x_plane)
        print(in_basket)
        if in_basket:
            hits += 1
        plt.show()
    print("trefferquote: ", hits/n)
trefferquote(h=1.5, alpha=70, v0=9,n=100)
exit()
# %% 
# 0.5397157485117868 1.5104108734552943 2.8928826226119035 8.095819662241084 0.12317991825296011
print(simulate_throw(x0=0.54, y0=1.51, vy=8.096, vx=2.893, r_ball=0.123))
plt.show()
for vx in np.linspace(2.07,2.55,100):
    print(vx, simulate_throw(vy=10,vx=vx))
    plt.show()
# %%

# plt.show()
