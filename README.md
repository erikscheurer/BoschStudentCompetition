Repository for the Bosch Basketball Students Competition 2022
von David Gekeler, Erik Scheurer und Julius Herb.

# Requirements
Numpy, Scipy, Matplotlib

# Overview

[`solution.py`](solution.py):
Contains our basketball throwing simulation, along with the monte carlo estimation code.

[`test.py`](test.py):
Contains a comparison of the different possible air resistance solutions

[`optimization.py`](optimization.py):
Contains the code for the grid search we used to find the optimum, along with another optimization method from the noisyopt package, which we did not end up using.



# Problem Statement and Idea


We have a robot with arm length $h$ that throws a ball at the angle alpha with a velocity $v$:

<img src="https://user-images.githubusercontent.com/84399192/204150058-08498ff0-1fe9-4ba0-b1b0-4ad7d5e4ba4d.png" width="450" />

 The idea is now, that the ball has a certain radius and is assumed to be a perfect circle.
 We can then calculate the intersection of the ball with the ring and the backboard by just tracking the center of the ball. 
 Intersection happens by adding a "buffer" of the radius of the ball and bouncing the center of the ball of this buffer.

 According to the specifications, the ball is modeled as a rigid body without considering angular momentum.
 In addition, the collisions are assumed to be perfectly elastic and the ideal law of reflection is applied.
 We also incorporate air resistance based on the drag equation for a more accurate model.

 Multiprocessing is available to parallelise the computation of the average hit rate and can be enabled manually.
 Please note that the support of multiprocessing depends on the available software and hardware setup.


# Usage                                 


At the bottom [`solution.py`](solution.py), some example runs are given.
You can also import the `korbwurf` function.
`simulate_throw` is the core of the simulation that also enables plotting.
`hit_rate` computes the average hit rate for given parameters.

The examples at the bottom of the file are also used to create the following plots

# Results
We get the following optimal parameters:

| Parameter | Value |
| --- | --- |
| $h$ | $2m$ |
| $v$ | $7.37 \frac{m}{s}$ |
| $\alpha$ | $60.68°$ |
| **Hit Rate** | $0.47$ |

The following plot shows one throw with the optimal parameters:

<img src="https://user-images.githubusercontent.com/84399192/204149306-0d7695e8-adab-4431-90ad-128d9c966ac5.png" width="450"/>

Including uncertainties, we see different colors for each segment of the throw. Green means, the ball goes in the hoop, blue means, it bounces of the ring or the backboard. Magenta means, the ball is out of bounds.

<img src="https://user-images.githubusercontent.com/84399192/204149304-04ad94b3-690a-4060-8b5e-a5acda075d4a.png"  width="450"/>

# Optimization

The following plot shows the optimization landscape:

<img src="https://user-images.githubusercontent.com/84399192/204149307-e56f6711-ce26-4fd2-897c-cbc92ba1c69f.png" width="450"/>

Convergence can be seen in the following plot:

<img src="https://user-images.githubusercontent.com/84399192/204149309-fdc59d5c-3430-4e4c-a960-602a06e105e4.png" width="450"/>
