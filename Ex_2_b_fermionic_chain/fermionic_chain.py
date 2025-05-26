import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

plt.rcParams['font.size'] = '16'

# System parameters
N = 3 # Number of sites
t = 1  # Hopping amplitude
mu = 0 # Chemical potential
F = 0 # Slope chemical potential rampage
U = 0.5 # Interaction strength
gamma_loss = 1  # Dissipation rate (electron loss)
gamma_gain = 1  # Pumping rate (electron creation)

# Function to create site-specific operators
def annihilate(site, N):
    """Creates an annihilation operator at a specific site."""
    op_list = [qt.qeye(2) for _ in range(N)]
    op_list[site] = qt.destroy(2)
    return qt.tensor(op_list)

def create(site, N):
    """Creates a creation operator at a specific site."""
    op_list = [qt.qeye(2) for _ in range(N)]
    op_list[site] = qt.create(2)
    return qt.tensor(op_list)


# Construct the Hamiltonian
H = 0
# Hopping term
H = sum(
    -t * (create(i, N) * annihilate(i+1, N) + create(i+1, N) * annihilate(i, N))
        for i in range(N-1))
# Chemical potential term
H += sum(
    (-mu-F * (i - (N - 1) // 2)) * (create(i, N) * annihilate(i, N))
        for i in range(N))
# Interacting term
H += sum(
    U * (create(i, N) * annihilate(i, N) * create(i+1, N) * annihilate(i+1, N))
        for i in range(N-1))

# Lindblad operators (electron loss and gain at each site)
L = []

# for i in range(N):
#    L.append(np.sqrt(gamma_loss) * create(i, N))  # Electron loss
#    L.append(np.sqrt(gamma_gain) * annihilate(i, N))      # Electron gain

L.append(np.sqrt(gamma_gain) * create(0, N))      # Electron gain
L.append(np.sqrt(gamma_loss) * annihilate(N-1, N))  # Electron loss




# Initial State-----------------------------------------------------------

# # Occupied positions (can be any two neighboring sites)
# occupied_sites = [2, 3]

# # Build the list of basis states
# state_list = [qt.basis(2, 1) if i in occupied_sites else qt.basis(2, 0) for i in range(N)]

# # Tensor product to build the many-body state
# psi = qt.tensor(state_list)

# rho0 = psi * psi.dag()

# old------
# Initial state (vacuum state)
rho0 = qt.tensor([qt.basis(2, 0) * qt.basis(2, 0).dag() for _ in range(N)])

# ------------------------------------------------------------------------

# Time evolution
tmax = 100
t_list = np.linspace(0, tmax, 10*tmax)

# Solve the master equation
result = qt.mesolve(H, rho0, t_list, L, [])

# anticommutator = create(3, N) * annihilate(3, N) + annihilate(3, N) * create(3, N)
# print(anticommutator)

# commutator = - create(0, N) * annihilate(0, N) + annihilate(0, N) * create(0, N)
# print(commutator)


# ------------------------------------------------------------------------
# Plot results
# The Fock states in QuTiP are ordered in binary counting order, meaning that each Fock state corresponds to a binary representation of occupation numbers

max_dist = np.max(np.abs(result.states[-1].full()))
qt.plot_fock_distribution(result.states[-1])
plt.title(f"Fermionic Chain of length {N}")
plt.ylim([0,(1.01)*max_dist])
# plt.show()
# exit()
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
occupations = [qt.expect(create(i, N) * annihilate(i, N), result.states[-1]) for i in range(N)]

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(range(N), occupations, 'o-')
ax.set_title(f"Fermionic Chain of length {N}")
ax.set_xlabel("Site Index")
ax.set_ylabel(f"Fermion Occupation")
# plt.show()
# exit()
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# Create number operator for the first and last sites
n1 = create(0, N) * annihilate(0, N)
nN = create(N-1, N) * annihilate(N-1, N)

# Compute the incoming current at each time step
curr_in = [gamma_gain*(1-qt.expect(n1, state)) for state in result.states]
curr_out = [gamma_loss*(qt.expect(nN, state)) for state in result.states]

# Plot occupation vs. time
fig, ax1 = plt.subplots(figsize=(9,6))

ax1.plot(t_list, curr_in, label=r"$I_{\text{in}}$")
ax1.plot(t_list, curr_out, label=r"$I_{\text{out}}$")
plt.xscale("log")
plt.xlabel("Time")
plt.ylabel(r"Current $I$")
plt.title(f"Current for N={N}")
plt.ylim(0, gamma_gain)
plt.legend()
plt.grid()
plt.show()





























# import numpy as np
# import scipy.linalg as sp
# import matplotlib.pyplot as plt
# import qiskit
# from qiskit import *
# from qiskit_aer import AerSimulator
# from qiskit.quantum_info.operators import Operator
# import qutip as qt
# from qutip import mesolve, Qobj


# # time/steps
# dt = 0.01
# nsteps = 2500
# times = np.linspace(0,(nsteps-1)*dt,nsteps)

# # system Hamiltonian parameters
# nsite = 3 #amount of spins
# ndvr = 2**nsite #Hilbertspace dimension
# OMEGA_i = [0.65,1.0,1.0]
# Jix = [0.75,1.0]
# Jiy = [0.75,1.0]
# Jiz = [0.0,0.0]

# # damping rates
# Gamma1 = [1/30.0] * nsite
# Gamma2 = [1/30.0] * nsite

# # Lindbladian yet to fill
# Lindbladian = []
# Lindbladian_qobj = []

# # spin states
# spin_up = np.array([1,0],dtype=np.float64)
# spin_down = np.array([0,1],dtype=np.float64)

# # Pauli matrices
# sigmax = np.array([[0,1],[1,0]],dtype=np.complex128)
# sigmay = np.array([[1,0],[0,-1]],dtype=np.complex128)
# sigmaz = np.array([[0,-1j],[1j,0]],dtype=np.complex128)
# ident = np.eye(2,dtype=np.complex128)

# sigmap = (sigmax + 1j*sigmay)/2
# sigmam = (sigmax - 1j*sigmay)/2
# sigma2 = sigmap@sigmam

# # spin chain Hamiltonian
# H_diag = np.zeros((ndvr,ndvr),dtype=np.complex128) #diagonal part
# for i in range(nsite):
#     tmp = 1.0
#     for k in range(nsite):
#         if(i==k):
#             tmp = np.kron(tmp,sigmaz)
#         else:
#             tmp = np.kron(tmp,ident)
#     H_diag += OMEGA_i[i]*tmp

# H_coup = np.zeros((ndvr,ndvr),dtype=np.complex128) #off-diagonal (coupling) part
# XX= np.kron(sigmax,sigmax)
# YY= np.kron(sigmay,sigmay)
# ZZ= np.kron(sigmaz,sigmaz)
# for i in range(nsite-1):
#     coup_tmp = Jix[i]*XX + Jiy[i]*YY + Jiz[i]*ZZ
#     tmp = 1.0
#     for k in range(nsite-1):
#         if (i==k):
#             tmp = np.kron(tmp,coup_tmp)
#         else:
#             tmp = np.kron(tmp,ident.copy())
#     H_coup += tmp
# H = H_diag - 0.5 * H_coup
# H_qobj = Qobj(H)
# # print(H_qobj)


# # fill Lindbladian
# for isite in range(nsite):
#     # Lindbladian type 1
#     res = 1.0
#     for j in range(nsite):
#         if (j==isite):
#             res = np.kron(res,sigmam) * np.sqrt(Gamma1[isite])
#         else:
#             res = np.kron(res,ident)
#     Lindbladian.append(res)
#     Lindbladian_qobj.append(Qobj(res))

#     # Lindbladian type 2
#     res = 1.0
#     for j in range(nsite):
#         if(j==isite):
#             res = np.kron(res,sigma2) * np.sqrt(Gamma2[isite])
#         else:
#             res = np.kron(res,ident)
#     Lindbladian.append(res)
#     Lindbladian_qobj.append(Qobj(res))
# # print(Lindbladian_qobj)
# # exit()


# # initial state (up,down,down)
# INIT_STATE = spin_up.copy()
# for i in range(nsite-1):
#     INIT_STATE = np.kron(INIT_STATE,spin_down)
# INIT_STATE_Qobj = Qobj(INIT_STATE)

# DENSITY_MAT = np.zeros((ndvr,ndvr),dtype=np.complex128)
# DENSITY_MAT = np.outer(INIT_STATE,INIT_STATE.conj())
# DENSITY_MAT_Qobj = Qobj(DENSITY_MAT)
# # print(DENSITY_MAT_Qobj)


# # ----------------Qutip---------------------------
# # Liouville (pure spin chain system)
# result_liouv = mesolve(H_qobj,DENSITY_MAT_Qobj,times,e_ops=[])
# # Lindblad equation
# result = mesolve(H_qobj,DENSITY_MAT_Qobj,times,c_ops=Lindbladian_qobj,e_ops=[])

# # calculate overlap with initial state
# overlaps_liouv = []
# for state in result_liouv.states:
#     overlap = abs(INIT_STATE_Qobj.overlap(state))**2
#     overlaps_liouv.append(overlap)

# overlaps = []
# for state in result.states:
#     overlap = abs(INIT_STATE_Qobj.overlap(state))**2
#     overlaps.append(overlap)
# # -------------------------------------------------


# # ---------------plotting--------------------------
# plt.rcParams['font.size'] = '16'
# fig, ax = plt.subplots(figsize=(9,6))
# ax.plot(times,overlaps_liouv,alpha=0.3,linestyle='--',label='Liouville')

# ax.plot(times,overlaps,label='Lindblad')


# ax.set_xlabel('Time')
# ax.set_ylabel('Initial state overlap')
# ax.legend()
# plt.show()
# # -----------------------------------------------
