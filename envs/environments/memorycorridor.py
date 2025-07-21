import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
import pygame
import sys

class MemoryCorridorEnv(gym.Env):
    metadata = {"render_modes": ["terminal", "graphic"]}

    def __init__(self, render_mode=None, num_doors=3, verbose=0, seed=None):
        super().__init__()
        self.seed(seed)
        self.num_doors = num_doors

        # Observation space is the index of the door that is currently the correct door
        # or num_doors if the correct door is hidden,
        self.observation_space = Discrete(self.num_doors + 1)

        # We have num_doors actions, corresponding to "open door 0", "open door 1", "open door 2" etc.
        self.action_space = Discrete(self.num_doors)

        # Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.pygame_initialized = False

        self.verbose = verbose

    def _get_obs(self):
        # Show correct door if it is the last of the sequence, else all doors are hidden (num_doors)
        obs = self.num_doors

        if self._on_last_door:
            obs = self._final_correct_door

        return obs

    def _get_info(self):
        return {
            "successfull_opens": self._successfull_opens,
            "depth": self._depth,
            "max_depth": len(self._correct_door_path),
        }

    def _compute_reward(self):
        if self._terminated:
            return 0
        return 1

    def seed(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)

    def reset(self):
        # Generate a new set of self.num_doors doors and the sequence of the correct doors
        self._correct_door_path = [np.random.randint(self.num_doors)]
        self._depth = 1
        self._successfull_opens = 0
        self._terminated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        assert action in self.action_space, "Invalid action: %s" % action

        if action != self._correct_door_path[self._depth - 1]:
            self._terminated = True
            if self.verbose > 0:
                print("=== GAME OVER ===")

        elif self._on_last_door:
            # Last door of sequence, generate a new final door and start from beginning of sequence
            self._correct_door_path.append(np.random.randint(self.num_doors))
            self._depth = 1
            self._successfull_opens += 1
            if self.verbose > 0:
                print("=== DOOR OPENED - NEW SEQUENCE ===")

        else:
            self._depth += 1
            self._successfull_opens += 1
            if self.verbose > 0:
                print("=== DOOR OPENED ===")

        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._terminated
        info = self._get_info()

        return observation, reward, terminated, False, info

    @property
    def _final_correct_door(self):
        return self._correct_door_path[-1]

    @property
    def _on_last_door(self):
        return self._depth == len(self._correct_door_path)


def test():
    from edugym.envs.interactive import play_env, play_env_terminal

    render_mode = "graphic"  # 'terminal', 'none'
    env = MemoryCorridorEnv(render_mode=render_mode)

    if render_mode == "graphic":
        play_env(
            env,
            "0=First Door, 1=Second Door, 2=Third Door",
            {pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2},
        )
    elif render_mode == "terminal":
        play_env_terminal(
            env, "1=First Door, 2=Second Door, 3=Third Door", {"1": 0, "2": 1, "3": 2}
        )


if __name__ == "__main__":
    env = MemoryCorridorEnv(verbose=True)
    obs, info = env.reset()
    while(True):
        print(obs)
        print(info)
        action = int(input())
        obs, _, _, _, info = env.step(action)