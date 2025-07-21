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

from models.core.S5 import ResidualS5BlockRL

class S5Classifier(eqx.Module):
    """S5 network for classification"""

    encoder: list
    residuallayers: list
    linear: eqx.nn.Linear

    num_res_layers: int
    N: int
    def __init__(self, 
                 key,
                 num_blocks: int,
                 residualblockparams: dict,
                 ):
        
        key, encoder_1_key, encoder_2_key, linearkey = jax.random.split(key, 4)
        reskeys = jax.random.split(key, num_blocks)

        self.encoder = [
            eqx.nn.Linear("scalar", residualblockparams.H, key=encoder_1_key),
        ]

        self.residuallayers = [ResidualS5BlockRL(reskey, **residualblockparams) for reskey in reskeys]

        self.linear = eqx.nn.Linear(residualblockparams.H, 10, key=linearkey)
        self.num_res_layers = num_blocks
        if residualblockparams.conj_symmetry:
            self.N = residualblockparams.N // 2
        else:
            self.N = residualblockparams.N

    def __call__(self, u, key, testing=False):
        """
        input:
            u: input sequence   (L,)
            key: PRNG key       ()
        output:
            y:  classification  (,)
        """

        # Encoding
        y = u
        for layer in self.encoder:
            # breakpoint()
            y = jax.vmap(layer)(y) #vmap over sequence length
        # y: (L, H)

        # Compute SSM blocks
        x = self.initialize_hidden_state()
        d = None
 
        for index, layer in enumerate(self.residuallayers):
            layerkey, key = jax.random.split(key)
            y, _ = layer(y, x[index], d, layerkey, testing=testing) # (L, H) 
        
        y = jnp.mean(y, axis=0) # Meanpooling along sequence length
        
        if testing:
            jax.debug.breakpoint(ordered=True)

        # FF layers
        y = self.linear(y)
        
        if testing:
            jax.debug.breakpoint(ordered=True)

        return jax.nn.log_softmax(y)
    
    def initialize_hidden_state(self):
        return jnp.zeros(shape=(self.num_res_layers, self.N), dtype=jnp.complex64)

    
