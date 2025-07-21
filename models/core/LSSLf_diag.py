import equinox as eqx
import jax
import jax.numpy as jnp

from models.core.SSM_utils import ZOH, GBT, associative_operator, associative_operator_reset, create_DPLR_HiPPO, init_VinvB, init_CV


class LSSLfDiagSSM(eqx.Module):
    Lambdas: jax.Array
    B_mats: jax.Array
    C_mats: jax.Array
    D_mats: jax.Array

    conj_symm: bool

    def __init__(self, key, N=128, H=128, dt_log_bounds=(-3, -1),  conj_symmetry=True, HiPPO_B=False, use_feedthrough=True, discretizer="ZOH"):
        r"""
        Implementation of the LSSL-f model, assuming single channel (M=1)
        N: Hidden state size
        H amount of features
        delta_bounds: tuple containing $\Delta t_{\text{min}}$ and $\Delta t_\text{max}}
        sequence_length: Length of the sequence fed to the model (L)

        initialises with
        Lambdas: Diagonalized state matrix for each feature;    size (H, N)
        B_mats: B matrices for each feature;                    size (H, H)
        C_mats: C matrices for each feature;                    size (H, N)
        D_mats: D matrices for each feature;                    size (H,)

        Note that only C and D are learned
        """
        key, B_key, C_key, D_key = jax.random.split(key, 4)

        if conj_symmetry:
            assert (N % 2 == 0) # Hidden state length should be even for conj. symmetry
            self.conj_symm = True
            block_size = N // 2
        else:
            self.conj_symm = False
            block_size = N

        Lambda, B, V = create_DPLR_HiPPO(block_size)
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        V_inv = V.conj().T

        deltas = jnp.logspace(dt_log_bounds[0], dt_log_bounds[1], H)
        if HiPPO_B:
            B_mat = V_inv @ B
        else:
            B_mat = init_VinvB(B_key, (block_size, 1), V_inv)
            B_mat = jnp.squeeze(B_mat)
        
        if discretizer == "ZOH":
            f = ZOH
        elif discretizer == "GBT":
            f = GBT
        else:
            raise Exception('Unknown discretizer')

        self.Lambdas, self.B_mats = jax.vmap(lambda dt: f(Lambda, B_mat, dt))(deltas)

        self.C_mats = init_CV(C_key, (H, block_size), V)

        if use_feedthrough:
            self.D_mats = jax.random.normal(D_key, (H))
        else:
            self.D_mats = jnp.zeros(shape=(H))

        # jax.debug.breakpoint()

    def __call__(self, u, x, d):
        """
        args:
            u: (H, L) H input sequences of length L
            x: (H, N) H hidden states, length N
            d: (L) done signals
        output:
            y: (H, L) H output sequences
            x: (H, N) H new hidden states
        parameters:
            Lambdas: (H, N) (H diagonal matrices)
            B_mats: (H, N) (H vectors of length N)
            C_mats: (H, N) (H output matrices)
            D_mats: ()

        """
        def kernel_op(Lambda, B, C, D, u, x, d):
            # Create associative operator tuples
            Lambdas = Lambda * jnp.ones((u.shape[0], Lambda.shape[0]), dtype=jnp.complex64)
            Bu = jax.vmap(lambda u_t: B * u_t)(u) # note: * instead of @ since u_t is a scalar

            # Account for previous hidden states '
            Lambdas = jnp.insert(Lambdas, 0, jnp.ones(Lambda.shape, dtype=jnp.complex64), axis=0)
            Bu = jnp.insert(Bu, 0, x, axis=0)

            if d is not None:
                d = jnp.insert(d, 0, jnp.zeros(1, dtype=jnp.bool), axis=0) # (done signal is irrelevant, can be 0)
                _, xs = jax.lax.associative_scan(associative_operator_reset, (Lambdas, Bu, d))
            else:
                _, xs = jax.lax.associative_scan(associative_operator, (Lambdas, Bu))
            
            xs = xs[1:]
        
            if self.conj_symm:
                # vmap over time
                y = jax.vmap(lambda x_t, u_t: 2*(C @ x_t).real + (D * u_t))(xs, u)
            else:
                y = jax.vmap(lambda x_t, u_t: (C @ x_t).real + (D * u_t))(xs, u)
            
            x = xs[-1] # new hidden states

            return y, x
        
        vmap_k = jax.vmap(lambda Lambda, B, C, D, u_H, x_H: kernel_op(Lambda, B, C, D, u_H, x_H, d))
        y, x = vmap_k(self.Lambdas, self.B_mats, self.C_mats, self.D_mats, u, x)

        return y, x


class ResidualLSSLfDiagBlock(eqx.Module):
    SSM: LSSLfDiagSSM
    linear: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    inference: bool
    p_dropout: float = eqx.field(static=True)

    def __init__(
        self,
        key,
        p_dropout=0.2, 
        use_layernorm=True,
        **kwargs
    ):
        """
        Implementation of a LSSLf layer with gelu activation, dropout, layer normalisation, and a residual connection.
        """

        LSSLkey, linearkey = jax.random.split(key, 2)

        self.SSM = LSSLfDiagSSM(key=LSSLkey,**kwargs)
        self.linear = eqx.nn.Linear(kwargs["H"], kwargs["H"], key=linearkey)
        if use_layernorm:
            self.layernorm = eqx.nn.LayerNorm(shape=(kwargs["H"]))
        else:
            self.layernorm = eqx.nn.Identity()
        self.inference = False
        self.p_dropout = p_dropout

    def __call__(self, u, x, d, key, inference=None):
        """
        args:
            u   : (H, L) Input sequence of length L
            x   : (H, N)    Hidden state, length N
            d   : (L)    Done signals
            key : (,)    PNRNG key
        output:
            y: (H, L) Output sequence of length L
            x: (H, N) new hidden state
        """
        # SSM call
        y, x = self.SSM(u, x, d)
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

        # Layer normalisation (map over sequence length)
        y = jax.vmap(self.layernorm, in_axes=1, out_axes=1)(y)

        return y, x  # residual connection



