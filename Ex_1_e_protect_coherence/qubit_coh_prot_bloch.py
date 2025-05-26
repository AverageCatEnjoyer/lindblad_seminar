import numpy as np
import scipy.linalg as sp 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from qutip import mesolve, sigmax, sigmay, sigmaz, sigmam, sigmap, Qobj, Bloch


# time/steps
dt = 0.1
nsteps = 250
times = np.linspace(0,(nsteps-1)*dt,nsteps)


# spin-1/2 system Hamiltonian
hbar_omega_half = 1/2
H_0 = hbar_omega_half * sigmaz()

# driving Hamiltonian for coherence protection
A = np.array([1.5 , 0.0 , 0.0]) # A_i \in [0,2\pi] WHY 2pi
H_d = hbar_omega_half * (A[0] * sigmax() + A[1] * sigmay() + A[2] * sigmaz())

# total Hamiltonian
H = H_0 + H_d

# damping rate / jump operator
gamma_1spin = 0.25
L_1spin = sigmap()

# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

init_state = 1/np.sqrt(2) * (spin_up + spin_down)
rho0_1spin = np.outer(init_state,init_state.conj())
# print(rho0_1spin)
# exit()



e_ops=[sigmax(), sigmay(), sigmaz()]

# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
result_qutip_Liouv = mesolve(Qobj(H_0), Qobj(rho0_1spin), times, e_ops = e_ops)

# with Jump operator -> Lindblad equation

# without H_d
result_qutip_0 = mesolve(Qobj(H_0), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
# with H_d
result_qutip = mesolve(Qobj(H), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
# -------------------------------------------------




# Extract x, y, z expectation values
expt_x_liouv = result_qutip_Liouv.expect[0]
expt_y_liouv = result_qutip_Liouv.expect[1]
expt_z_liouv = result_qutip_Liouv.expect[2]

expt_x_0 = result_qutip_0.expect[0]
expt_y_0 = result_qutip_0.expect[1]
expt_z_0 = result_qutip_0.expect[2]

expt_x = result_qutip.expect[0]
expt_y = result_qutip.expect[1]
expt_z = result_qutip.expect[2]

# --------------static-bloch-figure-------------------
# b = Bloch(figsize=[6.5,7])
# b.font_size = 16
# b.add_points([expt_x, expt_y, expt_z])
# b.add_points([expt_x_liouv, expt_y_liouv, expt_z_liouv])
# b.add_points([expt_x_0, expt_y_0, expt_z_0])

# b.render()
# plt.show()
# exit()
# -------------------------------------------------

# ----------------animation------------------------
fig = plt.figure(figsize=(6, 6))
b = Bloch(fig=fig)

def update(i):
    b.clear()
    b.add_points([expt_x_0[:i+1], expt_y_0[:i+1], expt_z_0[:i+1]])
    b.add_points([expt_x[:i+1], expt_y[:i+1], expt_z[:i+1]])
    # b.add_points([expt_x_liouv[:i+1], expt_y_liouv[:i+1], expt_z_liouv[:i+1]])
    b.render()

# Build animation
ani = FuncAnimation(fig, update, frames=len(times), interval=100, repeat=False)

# save if wanted
# ani.save("qubit_prot_coh_bloch_animation.gif", writer=PillowWriter(fps=20))

plt.show()

