import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# (x-h)**2 + (y-k)**2 = a**2 
def f(x, h, k, a):
    within_sqrt = a**2 - (x-h)**2
    within_sqrt = np.where(within_sqrt > 0, within_sqrt, 100) #100 used as artificial value to ensure the error is high when we would get an out of bounds error.
    return within_sqrt**.5 + k


deltas = np.loadtxt("data/classroom_data.csv", delimiter=",")
points = [np.array([0.0,0.0])]
for delta in deltas:
    points.append(points[-1] + delta)
points = np.array(points)


params, cov = sp.optimize.curve_fit(f, points[:,0], points[:,1], [70, 0, 75])

fig, ax = plt.subplots()
ax.plot(points[:,0], points[:,1])
ax.plot(points[:,0], f(points[:,0], *params))
ax.set_title(f"h={params[0]:.2e}, k={params[1]:.2e}, a={params[2]:.2e}")
fig.savefig("plots/classroom_curve.svg")
