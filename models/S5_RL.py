import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import distrax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import chex

import time
import sys
import logging
import json
import os
from functools import partial
from typing import NamedTuple, List, Tuple

from models.core.S5 import ResidualS5BlockRL
from models.core.ActorCritic_layers import ActorNetwork, CriticNetwork

class S5ActorCriticNetwork(eqx.Module):
    """S5 network with actor and critic head"""

    encoder: list
    residuallayers: list
    actor: ActorNetwork
    critic: CriticNetwork

    num_res_layers: int
    N: int

    def __init__(self, 
                 key,
                 obs_features: int,  
                 num_actions: int,
                 num_blocks: int,
                 residualblockparams: dict,
                 actor_hiddenlayers: List[int],
                 critic_hiddenlayers:List[int] 
                 ):
        
        key, encoder_1_key, encoder_2_key, actor_key, critic_key = jax.random.split(key, 5)
        reskeys = jax.random.split(key, num_blocks)

        self.encoder = [
            eqx.nn.Linear(obs_features, residualblockparams.H // 2, key=encoder_1_key),
            jax.nn.relu,
            eqx.nn.Linear(residualblockparams.H // 2, residualblockparams.H, key=encoder_2_key),
            jax.nn.relu
        ]

        self.residuallayers = [ResidualS5BlockRL(reskey, **residualblockparams) for reskey in reskeys]

        self.actor = ActorNetwork(actor_key, residualblockparams.H, actor_hiddenlayers, num_actions)
        self.critic = CriticNetwork(critic_key, residualblockparams.H, critic_hiddenlayers)

        self.num_res_layers = num_blocks
        self.N = residualblockparams.N // 2 if residualblockparams.conj_symmetry else residualblockparams.N

    def __call__(self, u, x, d, key, testing=False):
        """
        input:
            u: input sequence   (L, H_u)
            x: hidden state     (res_layers, N)
            d: Done signal      (L)
            key: PRNG key       ()
        output:
            action  : Categorical distribution    (L, num_actions)
            value   : Value of current state      (L,)
            x       : new hidden state            (N)
        """
        # Encoding
        u = jnp.array(u, ndmin=2) # necessary when sequence length is 1
        d = jnp.array(d, ndmin=1) # necessary when sequence length is 1
        y = u

        if testing:
            jax.debug.breakpoint(ordered=True)

        for layer in self.encoder:
            y = jax.vmap(layer)(y) #vmap over sequence length

        if testing:
            jax.debug.breakpoint(ordered=True)

        # Compute SSM blocks
        new_hidden_states = []
        for index, layer in enumerate(self.residuallayers):
            layerkey, key = jax.random.split(key)
            y, x_new = layer(y, x[index], d, layerkey, testing=testing) # (L, H) 
            new_hidden_states.append(x_new)
        
        x = jnp.stack(new_hidden_states)
        # y = jnp.mean(y, axis=0) # Meanpooling along sequence length
        
        
        if testing:
            jax.debug.breakpoint(ordered=True)

        # FF layers
        action = jax.vmap(self.actor)(y)
        value = jax.vmap(self.critic)(y)

        if testing:
            jax.debug.breakpoint(ordered=True)

        return action, value, x

    def initialize_hidden_state(self, num_envs):
        return jnp.zeros(shape=(num_envs, self.num_res_layers, self.N), dtype=jnp.complex64)