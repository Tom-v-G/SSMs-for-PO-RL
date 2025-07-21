import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import distrax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

from typing import List
import time
import sys
import logging
import json
import os
from functools import partial

from models.core.ActorCritic_layers import ActorNetwork, CriticNetwork


# global logger
logger = logging.getLogger(__name__)

class GRUActorCriticNetwork(eqx.Module):
    hidden_size: int
    GRUcell : eqx.nn.GRUCell
    actor : eqx.nn.Linear
    critic: eqx.nn.Linear

    def __init__(self, 
                 key,
                 obs_features: int,  
                 num_actions: int,
                 num_blocks: int,
                 residualblockparams: dict,
                 actor_hiddenlayers: List[int],
                 critic_hiddenlayers:List[int] 
                 ):
        ckey, actor_key, critic_key = jax.random.split(key, 3)
        # TODO: allow for multiple GRU cells 
        # ckeys = jax.random.split(ckey, num_blocks)
        self.hidden_size = residualblockparams.N
        self.GRUcell = eqx.nn.GRUCell(obs_features, self.hidden_size, key=key)
        self.actor = ActorNetwork(actor_key, residualblockparams.N, actor_hiddenlayers, num_actions)
        self.critic = CriticNetwork(critic_key, residualblockparams.N, critic_hiddenlayers)

    def __call__(self, u, x, d, key):
        """
        inputs: 
            u: (H, L) Input sequence of length L
            x: (N,) Hidden state length N
            d: (H,)   Done signals
        """
        u = jnp.array(u, ndmin=2)
        def _f(hidden_state, input):
            return self.GRUcell(input, hidden_state), self.GRUcell(input, hidden_state)
        
        x, y = jax.lax.scan(_f, x, u) # y contains stacked hidden states for action/value mapping
        # FF layers
        action = jax.vmap(self.actor)(y)
        value = jax.vmap(self.critic)(y)
        # action = self.actor(x)
        # value = self.critic(x)
        return action, value, x
    
    def initialize_hidden_state(self, num_envs):
        return jnp.zeros(shape=(num_envs, self.hidden_size))