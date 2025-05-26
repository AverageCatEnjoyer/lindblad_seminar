import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from qutip import sigmax, sigmay, sigmaz, sigmap, sigmam, mesolve, Qobj, Bloch



def bloch_vector(rho):
    r_x = np.real(np.trace(rho @ sigmax().full()))
    r_y = np.real(np.trace(rho @ sigmay().full()))
    r_z = np.real(np.trace(rho @ sigmaz().full()))
    return np.array([r_x, r_y, r_z])


# time/steps
dt = 0.1
nsteps = 1000
times = np.linspace(0,(nsteps-1)*dt,nsteps)


# spin-1/2 system Hamiltonian
tunneling_rate = 2*np.pi * 0.1
H_1spin = tunneling_rate * sigmaz()

# damping rate / Jump operators
gamma_1spin = 0.05

L_1spin = sigmaz() #pure dephasing
L_1spin_2 = sigmap() #bath at temp of up-state


# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

init_state = (spin_up + spin_down)/np.sqrt(2)
rho0_1spin = np.outer(init_state,init_state.conj())

rho0_1spin_2 = np.array([[0.75,0.3],[0.3,0.25]] ,dtype=np.float64)

init_vec = bloch_vector(rho0_1spin)
init_vec_2 = bloch_vector(rho0_1spin_2)



e_ops=[sigmax(), sigmay(), sigmaz()]

# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
result_qutip_Liouv = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops)
result_qutip_Liouv_2 = mesolve(Qobj(H_1spin), Qobj(rho0_1spin_2), times, e_ops = e_ops)

# with Jump operator -> Lindblad equation
result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
result_qutip_2 = mesolve(Qobj(H_1spin), Qobj(rho0_1spin_2), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_2))
# -------------------------------------------------


# Extract x, y, z expectation values

expt_x_Liouv = result_qutip_Liouv.expect[0]
expt_y_Liouv = result_qutip_Liouv.expect[1]
expt_z_Liouv = result_qutip_Liouv.expect[2]

expt_x_Liouv_2 = result_qutip_Liouv_2.expect[0]
expt_y_Liouv_2 = result_qutip_Liouv_2.expect[1]
expt_z_Liouv_2 = result_qutip_Liouv_2.expect[2]

expt_x = result_qutip.expect[0]
expt_y = result_qutip.expect[1]
expt_z = result_qutip.expect[2]

expt_x_2 = result_qutip_2.expect[0]
expt_y_2 = result_qutip_2.expect[1]
expt_z_2 = result_qutip_2.expect[2]


# --------------static-bloch-figure-------------------
# fig = plt.figure(figsize=(6.5,7))
# b = Bloch(fig=fig)
# b.font_size = 16
# b.vector_color = ['red','blue']

# b.add_vectors(init_vec)
# b.add_vectors(init_vec_2)

# # b.add_points([expt_x_Liouv, expt_y_Liouv, expt_z_Liouv],colors='red')
# # b.add_points([expt_x_Liouv_2, expt_y_Liouv_2, expt_z_Liouv_2],colors='blue')

# b.add_points([expt_x, expt_y, expt_z],colors='red')
# b.add_points([expt_x_2, expt_y_2, expt_z_2],colors='blue')

# b.render()
# plt.show()
# exit()
# -------------------------------------------------

# ----------------animation------------------------
fig = plt.figure(figsize=(6, 6))
b = Bloch(fig=fig)
# b.point_marker = ['o']
def update(i):
    b.clear()
    b.vector_color = ['red','blue','grey','grey']

# ----------------------------------
    # Liouville
    # # # points
    # b.add_points([expt_x_Liouv[:i+1], expt_y_Liouv[:i+1], expt_z_Liouv[:i+1]],colors='red')
    # b.add_points([expt_x_Liouv_2[:i+1], expt_y_Liouv_2[:i+1], expt_z_Liouv_2[:i+1]],colors='blue')
    # # vectors
    # b.add_vectors([expt_x_Liouv[i], expt_y_Liouv[i], expt_z_Liouv[i]])
    # b.add_vectors([expt_x_Liouv_2[i], expt_y_Liouv_2[i], expt_z_Liouv_2[i]])
    # b.add_vectors(init_vec)
    # b.add_vectors(init_vec_2)
# ----------------------------------

# ----------------------------------
    # Lindblad
    # points
    b.add_points([expt_x[:i+1], expt_y[:i+1], expt_z[:i+1]],colors='red')
    b.add_points([expt_x_2[:i+1], expt_y_2[:i+1], expt_z_2[:i+1]],colors='blue')
    # vectors
    b.add_vectors([expt_x[i], expt_y[i], expt_z[i]])
    b.add_vectors([expt_x_2[i], expt_y_2[i], expt_z_2[i]])
    b.add_vectors(init_vec)
    b.add_vectors(init_vec_2)
# ----------------------------------

    b.render()


# Build animation
ani = FuncAnimation(fig, update, frames=len(times), interval=10, repeat=False)

# save if wanted
# ani.save("bloch_animation.gif", writer=PillowWriter(fps=20))

plt.show()
