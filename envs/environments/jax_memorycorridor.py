import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

@struct.dataclass
class EnvState:
    curr_depth: int
    max_depth: int
    path: jnp.ndarray

@struct.dataclass
class EnvParams:
    pass

class MemoryCorridor(environment.Environment):
    def __init__(self, num_doors=3, length=50):
        super().__init__()
        self.num_doors = num_doors
        self.length = length
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        terminated = jnp.logical_not(jnp.equal(action, state.path[state.curr_depth]))
        
        done = jnp.equal(state.curr_depth, self.length - 1)
        
        reward = jnp.where(terminated, 0.0, 1.0)
        on_max_depth = jnp.equal(state.curr_depth, state.max_depth)
        max_depth = jnp.where(on_max_depth, 
                              state.max_depth + 1,
                              state.max_depth)
        curr_depth = jnp.where(on_max_depth,
                               0,
                               state.curr_depth + 1)
        
        new_state = EnvState(
            curr_depth=jnp.int32(curr_depth),
            max_depth=jnp.int32(max_depth),
            path = state.path
        )
        obs = self.get_obs(new_state)

        return obs, new_state, reward, jnp.logical_or(terminated, done), {}
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""

        path = jax.random.randint(key, shape=(self.length), minval=0, maxval=self.num_doors)
        state = EnvState(
            curr_depth=jnp.int32(0),
            max_depth=jnp.int32(0),
            path=path,
        )
        obs = self.get_obs(state)
        return obs, state
    
    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state. Shows last door (when there) or -1"""
        on_max_depth = jnp.equal(state.curr_depth, state.max_depth)
        obs = jnp.where(
            on_max_depth,
            state.path[state.max_depth],
            jnp.int32(-1)
        )
        return obs

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.num_doors)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.array(0), jnp.array(self.num_doors), shape=(1,), dtype=jnp.float32)
    
class MemoryCorridorEasy(MemoryCorridor):

    def __init__(self, num_doors=3, length=50):
        super().__init__(num_doors, length)
    
    def get_obs(self, state: EnvState) -> chex.Array:
        """Returns observation from the state. Tuple containing (curr_depth, max_depth, last_door)"""
        obs = jnp.array([state.curr_depth, state.max_depth, state.path[state.max_depth]])
        return obs
    
    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.array([0, 0, 0]), jnp.array([self.length, self.length, self.num_doors]), shape=(3,), dtype=jnp.float32)
    