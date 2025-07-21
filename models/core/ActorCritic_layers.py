import jax
import equinox as eqx
from typing import List
import distrax
import jax.numpy as jnp

class ActorNetwork(eqx.Module):
    """Actor network"""

    layers: list

    def __init__(self, 
                 key, 
                 in_shape: int, 
                 hidden_features: List[int], 
                 num_actions: int
                 ):
        keys = jax.random.split(key, len(hidden_features))
        self.layers = [
            eqx.nn.Linear(in_shape, hidden_features[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_features[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_features[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_features[-1], num_actions, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return distrax.Categorical(logits=self.layers[-1](x))
    
class CriticNetwork(eqx.Module):
    """
        Critic network with a single output
        Used for example to output V when given a state
        or Q when given a state and action
    """
    layers: list

    def __init__(self, 
                 key, 
                 in_shape: int, 
                 hidden_layers: List[int]
                 ):
        
        keys = jax.random.split(key, len(hidden_layers))
        self.layers = [ # init with first layer
            eqx.nn.Linear(in_shape, hidden_layers[0], key=keys[0])
        ]
        for i, feature in enumerate(hidden_layers[:-1]):
            self.layers.append(eqx.nn.Linear(feature, hidden_layers[i+1], key=keys[i]))
        self.layers.append(eqx.nn.Linear(hidden_layers[-1], 1, key=keys[-1]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return jnp.squeeze(self.layers[-1](x), axis=-1)
    
