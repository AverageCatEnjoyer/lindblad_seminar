import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
# from Ipython.display import display,display_latex

# System parameters
N = 4  # Number of sites
t = 1.0  # Hopping amplitude
mu = 0 # Chemical potential
F = 0 # Slope chemical potential rampage
U = 0.5  # Interaction strength
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

#for i in range(N):
#    L.append(np.sqrt(gamma_loss) * create_annihilation_operator(i, N))  # Electron loss
#    L.append(np.sqrt(gamma_gain) * create_creation_operator(i, N))      # Electron gain

L.append(np.sqrt(gamma_gain) * create(0, N))      # Electron gain
L.append(np.sqrt(gamma_loss) * annihilate(N-1, N))  # Electron loss

# Initial state (vacuum state)
rho0 = qt.tensor([qt.basis(2, 0) * qt.basis(2, 0).dag() for _ in range(N)])

# Time evolution
tmax = 100
t_list = np.linspace(0, tmax, 10*tmax)

# Solve the master equation
result = qt.mesolve(H, rho0, t_list, L, [])

anticommutator = create(0, N) * annihilate(0, N) + annihilate(0, N) * create(0, N)
# print(anticommutator)

commutator = - create(0, N) * annihilate(0, N) + annihilate(0, N) * create(0, N)
# print(commutator)

# Plot results
# The Fock states in QuTiP are ordered in binary counting order, meaning that each Fock state corresponds
# to a binary representation of occupation numbers
qt.plot_fock_distribution(result.states[-1])


occupations = [qt.expect(create(i, N) * annihilate(i, N), result.states[-1]) for i in range(N)]
plt.plot(range(N), occupations, 'o-')
# plt.xlabel("Site Index")
# plt.ylabel("Electron Occupation")
plt.show()