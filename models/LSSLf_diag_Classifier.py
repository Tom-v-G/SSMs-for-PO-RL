import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
import optax  # https://github.com/deepmind/optax
import distrax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import chex

from models.core.LSSLf_diag import ResidualLSSLfDiagBlock

class Broadcast(eqx.Module):
    dim: int = eqx.field(static=True)

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return jnp.broadcast_to(x, (self.dim, *x.shape))


class LSSLfDiagClassifier(eqx.Module):
    broadcast: Broadcast
    residuallayers: list
    linear: eqx.nn.Linear

    num_res_layers: int
    N: int
    H: int

    def __init__(
        self,
        key,
        num_blocks: int,
        residualblockparams: dict
    ):
        
        key, linearkey = jax.random.split(key, 2)
        reskeys = jax.random.split(key, num_blocks)
        self.broadcast = Broadcast(residualblockparams.H)
        self.residuallayers = [ResidualLSSLfDiagBlock(reskey, **residualblockparams) for reskey in reskeys]
        self.linear = eqx.nn.Linear(residualblockparams.H, 10, key=linearkey)
        self.num_res_layers = num_blocks
        if residualblockparams.conj_symmetry:
            self.N = residualblockparams.N // 2
        else:
            self.N = residualblockparams.N
        self.H = residualblockparams.H


    def __call__(self, u, key):
        """
        input:
            u: input sequence
        output:
            y: classification
        """

        # Broadcasting
        y = self.broadcast(u) # shape: (H, L)

        # Computing SSM blocks
        x = self.initialize_hidden_state() # shape (num_res_layers, H, N)
        d = None

        # jax.debug.breakpoint()
        # breakpoint()

        for index, layer in enumerate(self.residuallayers):
            layerkey, key = jax.random.split(key)
            y, _ = layer(y, x[index], d, layerkey)  # (H, L)

        y = jnp.mean(y, axis=1)  # Meanpooling (H,)
        y = self.linear(y)  # (10)

        # jax.debug.breakpoint()
        return jax.nn.log_softmax(y)
    
    def initialize_hidden_state(self):
        return jnp.zeros(shape=(self.num_res_layers, self.H, self.N), dtype=jnp.complex64)

     

def create_LSSLf_Diag_filter_spec(model, p_dropout=0.2, use_layernorm=True, **kernelparameters):
    is_res_block = lambda x: isinstance(x, ResidualLSSLfDiagBlock)

    def _create_ResidualLSSLBlock_filter_spec(res_model):
        filter_spec = jtu.tree_map(lambda _: False, res_model)
        filter_spec = eqx.tree_at(
            lambda tree: (tree.SSM.C_mats, tree.SSM.D_mats),
            filter_spec,
            replace=(True, kernelparameters['use_feedthrough']),
        )
        filter_spec = eqx.tree_at(
            lambda m: (m.linear.weight, m.linear.bias),
            filter_spec,
            replace=(True, True),
        )
        filter_spec = eqx.tree_at(
            lambda m: (m.layernorm.weight, m.layernorm.bias),
            filter_spec,
            replace=(True, True),
        )
        return filter_spec

    temp = eqx.filter_eval_shape(
        ResidualLSSLfDiagBlock,
        jax.random.PRNGKey(0),
        p_dropout,
        use_layernorm,
        **kernelparameters
    )
    res_filter_spec = _create_ResidualLSSLBlock_filter_spec(temp)

    filter_spec = jtu.tree_map(lambda _: False, model)
    for idx, layer in enumerate(filter_spec.residuallayers):
        if is_res_block(layer):
            filter_spec.residuallayers[idx] = res_filter_spec
    filter_spec = eqx.tree_at(
        lambda m: (m.linear.weight, m.linear.bias), filter_spec, replace=(True, True)
    )
    return filter_spec