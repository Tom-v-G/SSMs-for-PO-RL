import equinox as eqx
import jax
import jax.numpy as jnp

from models.core.SSM_utils import associative_operator_reset, associative_operator

# @jax.vmap
# def GRU_associative_operator_reset(q_i: tuple, q_j: tuple):
#     """
#     q_i: ((1-z_i), z_i*h^{~}_i, d_i) 
#     q_j: ((1-z_j), z_j*h^{~}_j, d_j)
#     """
#     q_i_0, q_i_1, d_i = q_i
#     q_j_0, q_j_1, d_j = q_j


#     jax.debug.breakpoint()

#     return (
#         (q_i_0 * q_j_0) * (1 - d_j) + q_j_0 * d_j, # (1- z_t) * h_{t-1}
#         (q_j_0 * q_i_1 + q_j_1) * (1-d_j) + q_j_1 * d_j,  # (1-z_t * h_t-1 + z_t * h^{~}_t)
#         jnp.array(d_i * (1 - d_j) + d_j, dtype=jnp.bool) # d
#     )

class minGRU(eqx.Module):
    hidden_proj: eqx.nn.Linear
    update_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    
    def __init__(self, key, N=8, **kwargs):
        """
        Note: feature dimension is equal to hidden state dimension for GRU models. 
        """
        z_key, h_key, output_key = jax.random.split(key, 3)

        self.update_proj = eqx.nn.Linear(N, N, key=z_key)
        self.hidden_proj = eqx.nn.Linear(N, N, key=h_key)
        self.output_proj = eqx.nn.Linear(N, N, key=output_key)

    def __call__(self, u, x, d=None):
        """
        u: (L, H)
        x: (N)
        d: (L)
        """
        
        # Create update and hidden values based on input
        z = jax.nn.sigmoid(
            jax.vmap(self.update_proj)(u)
        )

        h_sim = jax.vmap(self.hidden_proj)(u)

        # Create scan tuples
        
        zs = jnp.insert(1 - z, 0, jnp.ones_like(z.shape[0]), axis=0)
        xs = jnp.insert(z * h_sim, 0, x, axis=0) # Account for previous hidden states 

        # jax.debug.breakpoint()

        if d is not None:
            d = jnp.insert(d, 0, jnp.zeros(1, dtype=jnp.bool), axis=0) # (done signal is irrelevant, can be 0)
            _, xs, _ = jax.lax.associative_scan(associative_operator_reset, (zs, xs, d))
        else:
            _, xs = jax.lax.associative_scan(associative_operator, (zs, xs))
            raise NotImplementedError
        
        # return output, last hidden state
        return xs[1:], xs[-1]
    

class ResidualminGRUblockRL(eqx.Module):

    GRU: minGRU
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

        GRUkey, linear1key, linear2key = jax.random.split(key, 3)
        self.GRU = minGRU(key=GRUkey, **kwargs)

        if use_layernorm:
            self.layernorm = eqx.nn.LayerNorm(shape=(kwargs["N"]))
        else:
            self.layernorm = eqx.nn.Identity()

        self.inference = False
        self.p_dropout = p_dropout
        self.GLU_activation = GLU_activation
        self.linear1 = eqx.nn.Linear(kwargs["N"], kwargs["N"], key=linear1key)
        self.linear2 = eqx.nn.Linear(kwargs["N"], kwargs["N"], key=linear2key)


    def __call__(self, u, x, d, key, inference=None, testing=False):
        """
        args: 
            u   : (L, N) Input sequence of length L
            x   : (N)    Hidden state, length N
            d   : (L)    Done signals
            key : (,)    PNRNG key
        output:
            y: (L, N) Output sequence of length L
            x: (N) new hidden state
        """
        # SSM call 
        y, x = self.GRU(u, x, d)

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











