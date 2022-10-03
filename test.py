# Use ODEINT to solve the differential equations defined by the vector field
from scipy.integrate import odeint
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def ode(xi, t, p):
    """
    Defines the ODE system for the ball throw with air resistance

    Arguments:
        xi :  vector of the state variables:
                  xi = [x,y,vx,vy]
        t :  time
        p :  vector of the parameters:
                  p = [k,g,m]
    """
    x, y, vx, vy = xi
    k, g, m = p

    # Create f = (x',y',vx',vy'):
    f = [vx,
         vy,
         -k/m*vx-k/m*vx*np.sqrt(vx**2+vy**2),
         -g-k/m*vy*np.sqrt(vx**2+vy**2)]
    return f

def ode_decoupled(xi, t, p):
    """
    Defines the decoupled ODE system for the ball throw with air resistance

    Arguments:
        xi :  vector of the state variables:
                  xi = [x,y,vx,vy]
        t :  time
        p :  vector of the parameters:
                  p = [k,g,m]
    """
    x, y, vx, vy = xi
    k, g, m = p
    t_peak = np.sqrt(k/(m*g))
    # Create f = (x',y',vx',vy'):
    f = [vx,
         vy,
         -k/m*vx**2,
         np.where(vy > 0,-g-k/m*vy**2,-g+k/m*vy**2)]
    return f

# general parameters
g=9.81  # gravity acceleration [m/s^2]
rho=1.204 # 1.2  # density of air [kg/m^3]
# ball parameters
r_ball=0.765/(2*np.pi)  #0.2 radius of the ball [m]
m_ball=0.609  #0.5 mass of the ball [kg]
cw=0.47  # 0.5drag coefficient of a sphere [-]

# Initial conditions
alpha = 60.68#60
v0 = 7.37#10
x0=0  # x coordinate of start point [m]
y0=2  # y coordinate of start point [m]
vx=v0*np.cos(np.deg2rad(alpha))  # x component of throwing velocity [m/s]
vy=v0*np.sin(np.deg2rad(alpha))  # y component of throwing velocity [m/s]

###########################################################################
# 1. Define the analytical solution of the ball throw with air resistance #
###########################################################################

# The air resistance force is modeled based on Newton friction
# and is given by F = 0.5 * cw * A * rho * v^2 = k * v^2
k = 0.5 * cw * r_ball**2 * np.pi * rho
v0 = np.sqrt(vx**2 + vy**2)

# We have derived the analytical solution of the initial value problem and checked it against
# a solution by Andreas Lindner (https://www.geogebra.org/m/S4EyHaFa)
# The analytical solution is only well-defined for vx >= 0 and x0 = 0, hence for the
# general case, the affine transformation x -> sgn * (x - x0), vx -> sgn * vx is employed,
# where sgn indicates the sign of vx
sgn = np.sign(vx)
sin_alpha = vy / v0
cos_alpha = sgn * vx / v0
# Mapping from time to x position (in consideration of the affine transformation):
t2x = lambda t: x0 + sgn * m_ball / k * np.log(k * v0 * cos_alpha / m_ball * t + 1)
# Mapping from x position to time (in consideration of the affine transformation):
x2t = lambda x: m_ball / (k * v0 * cos_alpha) * (np.exp(k / m_ball * sgn * (x - x0)) - 1)
# For convenience we introduce some auxiliary quantities:
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
t2y = lambda t: np.where(True, t2y_up(t), t2y_down(t))
# Mapping from x position to y position, i.e. the ball throw trajectory
f = lambda x: t2y(x2t(x))
# Derivatives that are needed to compute the velocity at the bounces:
t2dy_up = lambda t: m_ball / k * d * np.tan(c - d * t)
t2dy_down = lambda t: m_ball / k * d * np.tanh(c - d * t)
# Time derivative of the x position, i.e. horizontal velocity
t2dx = lambda t: m_ball * v0 * cos_alpha / (k * t * v0 * cos_alpha + m_ball)
# Time derivative of the y position, i.e. vertical velocity
t2dy = lambda t: np.where(t < t_peak, t2dy_up(t), t2dy_down(t))
# Horizontal velocity at position x
# We account for the affine transformation by applying the chain rule
dx = lambda x: sgn * t2dx(x2t(x))
# Vertical velocity at position x
dy = lambda x: t2dy(x2t(x))

# Without air resistance:
pa = -g/(2*vx**2)
pb = vy/vx+g*x0/vx**2
pc = -g*x0**2/(2*vx**2)-vy/vx*x0 + y0
# define parabola
def fp(x): return pa*x**2 + pb*x + pc
def dfp(x): return 2*pa*x + pb

# ODE solver parameters
abserr = 1.0e-16
relerr = 1.0e-12
stoptime = 1.5
numpoints = 1000

# Create the time samples for the output of the ODE solver.
# I use a large number of points, only because I want to make
# a plot of the solution that looks nice.
t = np.array([stoptime * float(i) / (numpoints - 1) for i in range(numpoints)])

# Pack up the parameters and initial conditions:
p = [k, g, m_ball]
w0 = [x0, y0, vx, vy]

# Call the ODE solver.
ode_sol = odeint(ode, w0, t, args=(p,),
              atol=abserr, rtol=relerr)
ode_sol = np.array(ode_sol)
x = ode_sol[:,0]
y = ode_sol[:,1]
oded_sol = odeint(ode_decoupled, w0, t, args=(p,),
              atol=abserr, rtol=relerr)
oded_sol = np.array(oded_sol)
xd = oded_sol[:,0]
yd = oded_sol[:,1]
plt.figure(1, figsize=(6, 4.5))

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
lw = 1

#plt.plot(t, x, 'b', linewidth=lw)
#plt.plot(t, t2x(t), linewidth=lw)
#plt.plot(t, y, 'b', linewidth=lw)
#plt.plot(t, t2y(t), linewidth=lw)
plt.plot(x, y, linewidth=lw)
plt.plot(xd, yd, linewidth=lw)
plt.plot(x, f(x), linewidth=lw)
plt.plot(x, fp(x), linewidth=lw)
plt.plot(x, oded_sol[:,3], linewidth=lw)

plt.legend((r'with air resistance (RK)', r'with air resistance (decoupled, RK)', r'with air resistance (decoupled, analytic)', r'without air resistance', 'vy'))
plt.title('Ball throw with air resistance')
plt.show()
