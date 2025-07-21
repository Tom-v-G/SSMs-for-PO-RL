import jax
import equinox as eqx
from typing import List
import jax.numpy as jnp

from models.core.ActorCritic_layers import ActorNetwork, CriticNetwork

class FFActorCriticNetwork(eqx.Module):
    actor: ActorNetwork
    critic: CriticNetwork

    def __init__(self, 
                 key, 
                 obs_features, 
                 num_actions,
                 actor_hiddenlayers: List[int], 
                 critic_hiddenlayers: List[int]
                ):
        
        actor_key, critic_key = jax.random.split(key, 2)
        self.actor = ActorNetwork(actor_key, obs_features, actor_hiddenlayers, num_actions)
        self.critic = CriticNetwork(critic_key, obs_features, critic_hiddenlayers)
    
    def __call__(self, obs, _, _1, _2):
        obs = jnp.array(obs, ndmin=2) # ensure value
        action_dist = jax.vmap(self.actor)(obs)
        value = jax.vmap(self.critic)(obs)
        return action_dist, value, _
    
    # TODO: simplify
    def initialize_hidden_state(self, num_envs):
        """
        For compatibility reasons
        """
        return jnp.zeros(shape=(num_envs), dtype=jnp.int32)
