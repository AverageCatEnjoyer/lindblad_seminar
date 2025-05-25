import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from qutip import sigmax, sigmay, sigmaz, sigmap, sigmam, mesolve, Qobj, Bloch

# time/steps
dt = 0.1
nsteps = 1000
times = np.linspace(0,(nsteps-1)*dt,nsteps)


# spin-1/2 system Hamiltonian
tunneling_rate = 2*np.pi * 0.1
H_1spin = tunneling_rate * sigmaz()

# damping rate / jump operator
gamma_1spin = 0.05

# competing
a=1
L_1spin_w = sigmap()
L_1spin_y = 2*a*sigmap() + a*sigmam()
L_1spin = a*sigmap() + a*sigmam()
L_1spin_v = 0.5*a*sigmap() + a*sigmam()
L_1spin_z = sigmam()


# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

# init_state = spin_up
init_state = (spin_up + spin_down)/np.sqrt(2)

rho0_1spin = np.outer(init_state,init_state.conj())
# print(rho0_1spin)




e_ops=[sigmax(), sigmay(), sigmaz()]

# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
# result_qutip_Liouv = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops)

# with Jump operator -> Lindblad equation
# result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
# result_qutip_y = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_y))
# -------------------------------------------------

# competing
result_qutip_w = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_w))
result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
result_qutip_y = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_y))
result_qutip_z = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_z))
result_qutip_v = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = e_ops, c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_v))



# Extract x, y, z expectation values

expt_x_w = result_qutip_w.expect[0]
expt_y_w = result_qutip_w.expect[1]
expt_z_w = result_qutip_w.expect[2]

expt_x = result_qutip.expect[0]
expt_y = result_qutip.expect[1]
expt_z = result_qutip.expect[2]

expt_x_y = result_qutip_y.expect[0]
expt_y_y = result_qutip_y.expect[1]
expt_z_y = result_qutip_y.expect[2]

expt_x_z = result_qutip_z.expect[0]
expt_y_z = result_qutip_z.expect[1]
expt_z_z = result_qutip_z.expect[2]

expt_x_v = result_qutip_v.expect[0]
expt_y_v = result_qutip_v.expect[1]
expt_z_v = result_qutip_v.expect[2]

# --------------static-bloch-figure-------------------
# fig = plt.figure(figsize=(6.5,7))
# b = Bloch(fig=fig)
# b.font_size = 16
# b.add_points([expt_x_z, expt_y_z, expt_z_z],colors='blue')
# b.add_points([expt_x_v, expt_y_v, expt_z_v],colors='purple')
# b.add_points([expt_x, expt_y, expt_z],colors='orange')
# b.add_points([expt_x_y, expt_y_y, expt_z_y],colors='brown')
# b.add_points([expt_x_w, expt_y_w, expt_z_w],colors='r')

# b.render()
# plt.show()
# exit()
# -------------------------------------------------

# ----------------animation------------------------
fig = plt.figure(figsize=(6, 6))
b = Bloch(fig=fig)
b.point_marker = ['o']
def update(i):
    b.clear()
    b.add_points([expt_x_z[:i+1], expt_y_z[:i+1], expt_z_z[:i+1]],colors='blue')
    b.add_points([expt_x_v[:i+1], expt_y_v[:i+1], expt_z_v[:i+1]],colors='purple')
    b.add_points([expt_x[:i+1], expt_y[:i+1], expt_z[:i+1]],colors='orange')
    b.add_points([expt_x_y[:i+1], expt_y_y[:i+1], expt_z_y[:i+1]],colors='brown')
    b.add_points([expt_x_w[:i+1], expt_y_w[:i+1], expt_z_w[:i+1]],colors='r')
    b.render()

# Build animation
ani = FuncAnimation(fig, update, frames=len(times), interval=10, repeat=False)

# save if wanted
# ani.save("bloch_animation.gif", writer=PillowWriter(fps=20))

plt.show()
