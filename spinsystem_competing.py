import numpy as np
import scipy.linalg as sp 
import matplotlib.pyplot as plt
import qiskit
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator
from qutip import *
# from qutip import mesolve, Qobj


# time/steps
dt = 0.1
nsteps = 1000
times = np.linspace(0,(nsteps-1)*dt,nsteps)

# Pauli matrices
sigmax = np.array([[0,1],[1,0]],dtype=np.complex128)
sigmay = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
sigmaz = np.array([[1,0],[0,-1]],dtype=np.complex128)
ident = np.eye(2,dtype=np.complex128)
sigmap = (sigmax + 1j*sigmay)/2
sigmam = (sigmax - 1j*sigmay)/2

# spin-1/2 system Hamiltonian
tunneling_rate = 2*np.pi * 0.1
H_1spin = tunneling_rate * sigmaz

# damping rate / jump operator
gamma_1spin = 0.05

# dissipation
# L_1spin = sigmap.copy()
# L_1spin_y = sigmam.copy()


# competing
a=1
b=1
L_1spin_w = sigmap.copy()
L_1spin = a*sigmap.copy() + b*sigmam.copy()
L_1spin_y = 2*a*sigmap.copy() + b*sigmam.copy()
L_1spin_v = 0.5*a*sigmap.copy() + b*sigmam.copy()
L_1spin_z = sigmam.copy()


# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

rho0_1spin = np.outer(spin_up,spin_up.conj())
# rho0_1spin = np.outer(spin_down,spin_down.conj())
# print(rho0_1spin)






# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
# result_qutip_Liouv = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz))

# with Jump operator -> Lindblad equation
# result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
# result_qutip_y = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_y))


# competing
result_qutip_w = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_w))
result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
result_qutip_y = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_y))
result_qutip_z = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_z))
result_qutip_v = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_v))

# print(result_qutip_Liouv.expect[0])
# exit()
# -------------------------------------------------

# ---------------plotting--------------------------
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize=(9,6))
# ax.plot(times,result_qutip_Liouv.expect[0],linestyle=':',alpha=0.8,c='green',label=r'Liouville, $L = 0$',zorder=20)
# ax.plot(times,result_qutip.expect[0],c='red',label=r'Lindblad, $L \propto \sigma_x$')
# ax.plot(times,result_qutip_y.expect[0],c='orange',label=r'Lindblad, $L \propto \sigma_y$')



# competing
ax.plot(times,result_qutip_w.expect[0],c='red',label=r'$\beta = 0$')

ax.plot(times,result_qutip_z.expect[0],label=r'$\alpha=0$')

ax.plot(times,result_qutip_y.expect[0],c='brown',label=r'$\alpha = \frac{1}{2}\beta$')

ax.plot(times,result_qutip.expect[0],c='orange',label=r'$\alpha = \beta$')

ax.plot(times,result_qutip_v.expect[0],c='purple',label=r'$\alpha = 2\beta$')



ax.set_title(r'$H \propto \sigma_z$')
ax.set_xlabel('Time')
ax.set_ylabel(r'$\sigma_z$')
ax.legend(loc='lower left')
plt.show()
# -------------------------------------------------

