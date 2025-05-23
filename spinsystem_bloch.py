import numpy as np
import scipy.linalg as sp 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from qutip import mesolve, sigmax, sigmay, sigmaz, Qobj, Bloch


# time/steps
dt = 0.1
nsteps = 1000
times = np.linspace(0,(nsteps-1)*dt,nsteps)


# spin-1/2 system Hamiltonian
tunneling_rate = 2*np.pi * 0.1
H_1spin = tunneling_rate * sigmaz()

# damping rate / jump operator
gamma_1spin = 0.05
L_1spin = sigmax()
L_1spin_y = sigmay()

# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

rho0_1spin = np.outer(spin_up,spin_up.conj())
# print(rho0_1spin)




e_ops=[sigmax(), sigmay(), sigmaz()]

# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
result_qutip_Liouv = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops)

# with Jump operator -> Lindblad equation
result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
result_qutip_y = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_y))
# -------------------------------------------------




# Extract x, y, z expectation values

expt_x_liouv = result_qutip_Liouv.expect[0]
expt_y_liouv = result_qutip_Liouv.expect[1]
expt_z_liouv = result_qutip_Liouv.expect[2]

expt_x = result_qutip.expect[0]
expt_y = result_qutip.expect[1]
expt_z = result_qutip.expect[2]

expt_x_y = result_qutip_y.expect[0]
expt_y_y = result_qutip_y.expect[1]
expt_z_y = result_qutip_y.expect[2]


# --------------static-bloch-figure-------------------
# b = Bloch(figsize=[6.5,7])
# b.font_size = 16
# b.add_points([expt_x, expt_y, expt_z])
# b.add_points([expt_x_liouv, expt_y_liouv, expt_z_liouv])
# # b.add_points([expt_x_y, expt_y_y, expt_z_y])

# b.render()
# plt.show()
# exit()
# -------------------------------------------------

# ----------------animation------------------------
fig = plt.figure(figsize=(6, 6))
b = Bloch(fig=fig)

def update(i):
    b.clear()
    b.add_points([expt_x[:i+1], expt_y[:i+1], expt_z[:i+1]])
    b.add_points([expt_x_liouv[:i+1], expt_y_liouv[:i+1], expt_z_liouv[:i+1]])
    b.render()

# Build animation
ani = FuncAnimation(fig, update, frames=len(times), interval=100, repeat=False)

# save if wanted
ani.save("bloch_animation.gif", writer=PillowWriter(fps=20))

plt.show()

