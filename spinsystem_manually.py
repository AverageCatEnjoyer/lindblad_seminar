import numpy as np
import scipy.linalg as sp 
import matplotlib.pyplot as plt
import qiskit
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.quantum_info.operators import Operator
import qutip as qt
from qutip import mesolve, Qobj

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
H_1spin = 2*np.pi * 0.1 * sigmax

# jump operator / damping rate
gamma_1spin = 0.05
L_1spin = sigmax.copy()

# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

rho0_1spin = np.outer(spin_up,spin_up.conj())
# print(rho0_1spin)


# ----------------manually-------------------------
Nsystem = H_1spin.shape[0] #dimension of Hilbert space
vec_rho0_1spin = rho0_1spin.reshape(Nsystem**2)
# print(vec_rho0_1spin)

# derivation matrix
ident_h = np.eye(Nsystem, dtype=np.complex128)
Amat = -1j * (np.kron(H_1spin,ident_h))- np.kron(ident_h,H_1spin.T)
# print(Amat)
# print(np.shape(Amat))

# lindblad term
Amat += (2.0*(np.kron(L_1spin,L_1spin.conj())) - np.kron(ident_h,L_1spin.T@L_1spin.conj()) - np.kron(L_1spin.T.conj()@L_1spin,ident_h))*0.5*gamma_1spin
# print(Amat)

results_manually = []
for i in range(nsteps):
    Gt = sp.expm(Amat*dt*i)
    vec_rhot_1spin = Gt@vec_rho0_1spin
    rhot_1spin = vec_rhot_1spin.reshape(Nsystem,Nsystem)

    # expval of sigmaz
    results_manually.append(np.trace(rhot_1spin@sigmaz))
# -------------------------------------------------




# ----------------Qutip---------------------------
# inputs for mesolve [Hamiltonian , density matrix , time , jump operator , expectation values for observables]

# not give jump operator:Liouville equation
result_qutip_Liouv = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz))

# not give jump operator:Liouville equation
result_qutip = mesolve(Qobj(H_1spin), Qobj(rho0_1spin), times, e_ops = Qobj(sigmaz), c_ops = np.sqrt(gamma_1spin)*Qobj(sigmax))

# print(result_qutip_Liouv.expect[0])
# exit()
# -------------------------------------------------

# ---------------plotting--------------------------
fig, ax = plt.subplots()
ax.plot(times,result_qutip.expect[0])
ax.plot(times,result_qutip_Liouv.expect[0])
ax.plot(times,results_manually)
plt.show()
# -------------------------------------------------
