import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import odeint
import time

# Commencement de l'odyssée spatiale
start_time = time.time()

# Paramètres cosmiques
Nb = 20
ti = 0
tf = 10
Nt = 500
t_step = (tf - ti) / Nt
t_array = np.linspace(ti, tf, Nt)
G = 6.674e-11 * (6e24 * (365 * 24 * 60 * 60) ** 2) / (1.496e11) ** 3
M = np.random.uniform(low=6e24, high=2e30, size=Nb) / 6e24

# Conditions initiales pour un ballet céleste
init_pos = np.random.uniform(low=-7, high=7, size=3 * Nb)
init_vel = np.random.uniform(low=-5, high=5, size=3 * Nb)
init_cond = np.append(init_pos, init_vel)

# La chorégraphie céleste commence ici
def N_Body(Yk, t):
    Rk = Yk[:3 * Nb].reshape((3, Nb))
    Sk = Yk[3 * Nb:]

    accelerations = np.zeros_like(Rk)
    for i in range(Nb):
        for j in range(Nb):
            if j != i:
                r_ij = Rk[:, i] - Rk[:, j]
                distance_ij = np.linalg.norm(r_ij)
                accelerations[:, i] += (
                    -G * M[j] * r_ij / (distance_ij ** 3 + 1e-1)
                )

    return np.append(Sk, accelerations.flatten())

# La trajectoire des étoiles en 2D
trajectory = odeint(N_Body, init_cond, t_array)

# Animation en 2D pour impressionner l'observateur
plt.style.use('dark_background')
fig = plt.figure(facecolor='black')
ax = plt.axes()
ax.axis('square')
ax.set_xlim(np.min(init_pos) - 10, np.max(init_pos) + 10)
ax.set_ylim(np.min(init_pos) - 10, np.max(init_pos) + 10)
plt.xlabel("X (AU)")
plt.ylabel("Y (AU)")

trail = 50

def Animate2D(frame):
    for i, line in enumerate(lines, 0):
        line.set_data(
            trajectory[frame:max(1, frame - trail):-1, 3 * i],
            trajectory[frame:max(1, frame - trail):-1, 3 * i + 1]
        )
    plt.title("Problème à " + str(Nb) + " corps à t = " + str(round(t_array[frame], 1)) + " ans")
    return lines

lines = [ax.plot([], [], "o-", markersize=3, markevery=10000, lw=1)[0] for _ in range(Nb)]

animation_2d = animation.FuncAnimation(fig, Animate2D, frames=len(t_array), interval=30, blit=False)

# Le spectacle prend fin, le rideau tombe
print("%s seconds" % (time.time() - start_time))
plt.show()
