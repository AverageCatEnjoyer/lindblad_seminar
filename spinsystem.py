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
nsteps = 250
times = np.linspace(0,(nsteps-1)*dt,nsteps)

# Pauli matrices
sigmax = np.array([[0,1],[1,0]],dtype=np.complex128)
sigmay = np.array([[1,0],[0,-1]],dtype=np.complex128)
sigmaz = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
ident = np.eye(2,dtype=np.complex128)

# spin-1/2 system Hamiltonian
tunneling_rate = 2*np.pi * 0.1
H_1spin = tunneling_rate * sigmax

# damping rate / jump operator
gamma_1spin = 0.05
L_1spin = sigmax.copy()
L_1spin_y = sigmay.copy()

# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

rho0_1spin = np.outer(spin_up,spin_up.conj())
# print(rho0_1spin)






# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
result_qutip_Liouv = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz))

# with Jump operator -> Lindblad equation
result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin))
result_qutip_y = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(L_1spin_y))

# print(result_qutip_Liouv.expect[0])
# exit()
# -------------------------------------------------

# ---------------plotting--------------------------
plt.rcParams['font.size'] = '16'
fig, ax = plt.subplots(figsize=(9,6))
ax.plot(times,result_qutip_Liouv.expect[0],alpha=0.3,linestyle='--',label=r'Liouville, $L = 0$')

ax.plot(times,result_qutip.expect[0],label=r'Lindblad, $L \propto \sigma_x$')

ax.plot(times,result_qutip_y.expect[0],label=r'Lindblad, $L \propto \sigma_y$')

ax.set_title(r'$H \propto \sigma_x$')
ax.set_xlabel('Time')
ax.set_ylabel(r'$\sigma_z$')
ax.legend()
plt.show()
# -------------------------------------------------

