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
dt = 0.01
nsteps = 2500
times = np.linspace(0,(nsteps-1)*dt,nsteps)

# system Hamiltonian parameters
nsite = 3 #amount of spins
ndvr = 2**nsite #Hilbertspace dimension
OMEGA_i = [0.65,1.0,1.0]
Jix = [0.75,1.0]
Jiy = [0.75,1.0]
Jiz = [0.0,0.0]

# damping rates
Gamma1 = [1/30.0] * nsite
Gamma2 = [1/30.0] * nsite

# Lindbladian yet to fill
Lindbladian = []
Lindbladian_qobj = []

# spin states
spin_up = np.array([1,0],dtype=np.float64)
spin_down = np.array([0,1],dtype=np.float64)

# Pauli matrices
sigmax = np.array([[0,1],[1,0]],dtype=np.complex128)
sigmay = np.array([[1,0],[0,-1]],dtype=np.complex128)
sigmaz = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
ident = np.eye(2,dtype=np.complex128)

sigmap = (sigmax + 1j*sigmay)/2
sigmam = (sigmax - 1j*sigmay)/2
sigma2 = sigmap@sigmam

# spin chain Hamiltonian
H_diag = np.zeros((ndvr,ndvr),dtype=np.complex128) #diagonal part
for i in range(nsite):
    tmp = 1.0
    for k in range(nsite):
        if(i==k):
            tmp = np.kron(tmp,sigmaz)
        else:
            tmp = np.kron(tmp,ident)
    H_diag += OMEGA_i[i]*tmp

H_coup = np.zeros((ndvr,ndvr),dtype=np.complex128) #off-diagonal (coupling) part
XX= np.kron(sigmax,sigmax)
YY= np.kron(sigmay,sigmay)
ZZ= np.kron(sigmaz,sigmaz)
for i in range(nsite-1):
    coup_tmp = Jix[i]*XX + Jiy[i]*YY + Jiz[i]*ZZ
    tmp = 1.0
    for k in range(nsite-1):
        if (i==k):
            tmp = np.kron(tmp,coup_tmp)
        else:
            tmp = np.kron(tmp,ident.copy())
    H_coup += tmp
H = H_diag - 0.5 * H_coup
H_qobj = Qobj(H)
print(H_coup.real)
exit()

# fill Lindbladian
for isite in range(nsite):
    # Lindbladian type 1
    res = 1.0
    for j in range(nsite):
        if (j==isite):
            res = np.kron(res,sigmam) * np.sqrt(Gamma1[isite])
        else:
            res = np.kron(res,ident)
    Lindbladian.append(res)
    Lindbladian_qobj.append(Qobj(res))

    # Lindbladian type 2
    res = 1.0
    for j in range(nsite):
        if(j==isite):
            res = np.kron(res,sigma2) * np.sqrt(Gamma2[isite])
        else:
            res = np.kron(res,ident)
    Lindbladian.append(res)
    Lindbladian_qobj.append(Qobj(res))
# print(Lindbladian_qobj)
# exit()


# initial state (up,down,down)
INIT_STATE = spin_up.copy() 
for i in range(nsite-1):
    INIT_STATE = np.kron(INIT_STATE,spin_down)
INIT_STATE_Qobj = Qobj(INIT_STATE)

DENSITY_MAT = np.zeros((ndvr,ndvr),dtype=np.complex128)
DENSITY_MAT = np.outer(INIT_STATE,INIT_STATE.conj())
DENSITY_MAT_Qobj = Qobj(DENSITY_MAT)
# print(DENSITY_MAT_Qobj)


# ----------------Qutip---------------------------
# Liouville (pure spin chain system)
result_liouv = mesolve(H_qobj,DENSITY_MAT_Qobj,times,e_ops=[])
# Lindblad equation
result = mesolve(H_qobj,DENSITY_MAT_Qobj,times,c_ops=Lindbladian_qobj,e_ops=[])

# calculate overlap with initial state
overlaps_liouv = []
for state in result_liouv.states:
    overlap = abs(INIT_STATE_Qobj.overlap(state))**2
    overlaps_liouv.append(overlap)

overlaps = []
for state in result.states:
    overlap = abs(INIT_STATE_Qobj.overlap(state))**2
    overlaps.append(overlap)
# -------------------------------------------------


# ---------------plotting--------------------------
# plt.rcParams['font.size'] = '16'
# fig, ax = plt.subplots(figsize=(9,6))
# ax.plot(times,overlaps_liouv,alpha=0.3,linestyle='--',label='Liouville')

# ax.plot(times,overlaps,label='Lindblad')


# ax.set_xlabel('Time')
# ax.set_ylabel('Initial state overlap')
# ax.legend()
# plt.show()
# exit()
# -------------------------------------------------


# -------------------------------------------------
qt.plot_fock_distribution(result.states[-1])
plt.title("Spin Chain")
plt.show()
# -------------------------------------------------
