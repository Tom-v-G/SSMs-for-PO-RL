import equinox as eqx
import jax
import jax.numpy as jnp

from models.core.SSM_utils import ZOH, GBT, associative_operator, associative_operator_reset, create_DPLR_HiPPO, init_VinvB, init_CV


class S5SSM(eqx.Module):
    Lambda: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    deltas: jax.Array
    conj_symm: bool
    discretizer: str

    def __init__(self, key, N=8, H=4, dt_log_bounds=(-3, -1), conj_symmetry=True, HiPPO_B=False, use_feedthrough=True, discretizer="ZOH"):
        r"""
        Implementation of the S5 model, assuming single channel (M=1)
        N: Hidden state size 
        H amount of features
        delta_bounds: tuple containing $\Delta t_{\text{min}}$ and $\Delta t_\text{max}}
        sequence_length: Length of the sequence fed to the model (L) 

        Let N be the hidden state dimension, U the input signal dimension. We assume the output signal also has dimension U
        initialises with
        Lambda  :   (N, N)
        B       :   (N, H)
        C       :   (H, N)
        D       :   (H,)
        """
        key, B_key, C_key, D_key = jax.random.split(key, 4)
        
        if conj_symmetry:
            assert (N % 2 == 0) # Hidden state length should be even for conj. symmetry
            self.conj_symm = True
            block_size = N // 2
        else:
            self.conj_symm = False
            block_size = N

        self.deltas = jnp.logspace(dt_log_bounds[0], dt_log_bounds[1], block_size)
        Lambda, B, V = create_DPLR_HiPPO(block_size)
        self.Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        V_inv = V.conj().T
        
        if HiPPO_B:
            self.B = jnp.stack([V_inv @ B for i in range(H)], axis=1) # Stack H copies of B
        else:
            self.B = init_VinvB(B_key, (block_size, H), V_inv)
        
        self.C = init_CV(C_key, (H, block_size), V)

        if use_feedthrough:
            self.D = jax.random.normal(D_key, (H))
        else:
            self.D = jnp.zeros(shape=(H))

        self.discretizer = discretizer

        # jax.debug.print(f"{self.Lambda.shape=}")
        # jax.debug.print(f"{V.shape=}")
        # jax.debug.print(f"{self.B.shape=}")
        # jax.debug.print(f"{self.C.shape=}")
        # jax.debug.print(f"{self.D.shape=}")

    def __call__(self, u, x, d):
        """
        u : (L, H)
        """
        if self.discretizer == "ZOH":
            Lambda, B = ZOH(self.Lambda, self.B, self.deltas)
        elif self.discretizer == "GBT":
            Lambda, B = GBT(self.Lambda, self.B, self.deltas)
        

        # Create associative operator tuples
        Lambdas = Lambda * jnp.ones((u.shape[0], Lambda.shape[0]), dtype=jnp.complex64)
        
        Bu = jax.vmap(lambda u_t: B @ u_t)(u)

        # Account for previous hidden states '
        
        Lambdas = jnp.insert(Lambdas, 0, jnp.ones(self.Lambda.shape, dtype=jnp.complex64), axis=0)
        Bu = jnp.insert(Bu, 0, x, axis=0)

        if d is not None:
            d = jnp.insert(d, 0, jnp.zeros(1, dtype=jnp.bool), axis=0) # (done signal is irrelevant, can be 0)
            _, xs, _ = jax.lax.associative_scan(associative_operator_reset, (Lambdas, Bu, d))
        else:
            _, xs = jax.lax.associative_scan(associative_operator, (Lambdas, Bu))
        
        xs = xs[1:]
        
        if self.conj_symm:
            # vmap over time
            y = jax.vmap(lambda x_t, u_t: 2*(self.C @ x_t).real + (self.D * u_t))(xs, u)
        else:
            y = jax.vmap(lambda x_t, u_t: (self.C @ x_t).real + (self.D * u_t))(xs, u)
        
        x = xs[-1] # new hidden states

        return y, x 

class ResidualS5BlockRL(eqx.Module):
    """
        Implementation of a S5 layer with gelu activation, dropout, layer normalisation, and a residual connection.
    """
    
    SSM: S5SSM
    layernorm: eqx.nn.LayerNorm
    inference: bool
    p_dropout: float = eqx.field(static=True)
    GLU_activation: bool
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, 
                 key,
                 p_dropout=0.2, 
                 use_layernorm=True,
                 GLU_activation=True,
                 **kwargs
        ):
        ssmkey, linear1key, linear2key = jax.random.split(key, 3)
        self.SSM = S5SSM(key=ssmkey, **kwargs)

        if use_layernorm:
            self.layernorm = eqx.nn.LayerNorm(shape=(kwargs["H"]))
        else:
            self.layernorm = eqx.nn.Identity()

        self.inference = False
        self.p_dropout = p_dropout
        self.GLU_activation = GLU_activation
        self.linear1 = eqx.nn.Linear(kwargs["H"], kwargs["H"], key=linear1key)
        self.linear2 = eqx.nn.Linear(kwargs["H"], kwargs["H"], key=linear2key)


    def __call__(self, u, x, d, key, inference=None, testing=False):
        """
        args: 
            u   : (L, H) Input sequence of length L
            x   : (N)    Hidden state, length N
            d   : (L)    Done signals
            key : (,)    PNRNG key
        output:
            y: (L, H) Output sequence of length L
            x: (N) new hidden state
        """
        # SSM call 
        y, x = self.SSM(u, x, d)

        if self.GLU_activation:
            y = jax.nn.gelu(y)
            y = jax.vmap(self.linear1)(y) * jax.nn.sigmoid(jax.vmap(self.linear2)(y))
        
        else:
            y = jax.nn.gelu(y)

        # Dropout per feature channel
        def _dropout(h):
            q = 1 - jax.lax.stop_gradient(self.p_dropout)
            mask = jax.random.bernoulli(key, q, h.shape[0])
            h = jax.vmap(lambda row, mask_row: jnp.where(mask_row, row / q, jax.lax.stop_gradient(jnp.zeros(h.shape[-1]))), in_axes=(0,0))(h, mask)
            return h
        
        if inference is None:
            inference = self.inference
        if self.p_dropout == 0:
            inference = True

        if not inference:
            y = _dropout(y)

        # Residual connection
        y = y + u 

        # Layer normalisation
        y = jax.vmap(self.layernorm)(y)

        return y , x 