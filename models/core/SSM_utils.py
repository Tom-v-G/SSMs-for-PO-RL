import jax
import jax.numpy as jnp
from jax.nn.initializers import lecun_normal

# Discretization Methods

def ZOH(Lambda, B, delta):
    Lambda_bar = jnp.exp(Lambda * delta)
    B_bar = jnp.diagflat(1 / Lambda) @ jnp.diagflat(Lambda_bar - 1) @ B
    return Lambda_bar, B_bar

def GBT(A, B, delta):
    I = jnp.identity(A.shape[0])
    inv = jnp.linalg.inv(I - 0.5 * delta * A)
    A = inv @ (I + 0.5 * delta * A)
    B = delta * inv @ B
    return A, B

# Associative Operators

@jax.vmap
def associative_operator(q_i: tuple, q_j: tuple):
    """
    q_i: (A, B @ u_i) 
    q_j: (A, B @ u_j)
    Assume A is diagonal -> use A*A instead of A @ A
    """
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j

    return (
        (A_j * A_i), # A
        (A_j * Bu_i + Bu_j) # Bu
    )

@jax.vmap
def associative_operator_reset(q_i: tuple, q_j: tuple):
    """
    q_i: (A, B @ u_i, d_i) 
    q_j: (A, B @ u_j, d_j)
    Assume A is diagonal -> use A*A instead of A @ A
    """
    A_i, Bu_i, d_i = q_i
    A_j, Bu_j, d_j = q_j

    return (
        (A_j * A_i) * (1-d_j) + A_j * d_j, # A
        (A_j * Bu_i + Bu_j) * (1-d_j) + (Bu_j * d_j), # Bu
        jnp.array(d_i * (1 - d_j) + d_j, dtype=jnp.bool) # d
    )

# HiPPO matrix initialization

def create_DPLR_HiPPO(N):
    # Normal HiPPO
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = -1 * (jnp.tril(A) - jnp.diag(jnp.arange(N)))

    #NPLR HiPPO
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)

    # DPLR hiPPO
    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)
    
    # Set SSM parameters
    Lambda = Lambda_real + 1j* Lambda_imag

    return Lambda, B, V

def init_VinvB(rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
         Args:
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H)
     """
    B = lecun_normal()(rng, shape)
    VinvB = Vinv @ B
    return VinvB

def init_CV(rng, shape, V):
    """ Initialize C_tilde=CV. First samples C. Then compute CV.
         Args:
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H, P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H, P)
     """
    C = lecun_normal()(rng, shape)
    return C @ V

