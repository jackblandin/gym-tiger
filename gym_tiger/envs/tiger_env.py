# Core modules
import logging.config

# 3rd party modules
import gym
from gym import spaces
import numpy as np


ACTION_LISTEN = 0
ACTION_OPEN_LEFT = 1
ACTION_OPEN_RIGHT = 2
ACTION_MAP = {
    ACTION_LISTEN: 'LISTEN',
    ACTION_OPEN_LEFT: 'OPEN_LEFT',
    ACTION_OPEN_RIGHT: 'OPEN_RIGHT',
}


class TigerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("TigerEnv - Version {}".format(self.__version__))

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

        self.curr_step = -1

        self.reset()

        # Define what the agent can do: LISTEN, OPEN_LEFT, OPEN_RIGHT
        self.action_space = spaces.Discrete(2)

        # Define what the agent can observe: GROWL_LEFT, GROWL_RIGHT
        self.observation_space = spaces.Discrete(1)

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        done = self.left_door_open or self.right_door_open
        if done:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        # Recompute done since action may have modified it
        done = self.left_door_open or self.right_door_open
        reward = self._get_reward()
        ob = self._get_obs()
        return ob, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.left_door_open = False
        self.right_door_open = False
        # TODO fix this random update
        self.tiger_left = np.random.randint(0, 2)
        self.tiger_right = 1 - self.tiger_left
        return [-1, -1]

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def translate_obs(self, obs):
        """
        Method created by JDB to easily interpet the obs. in plain English.
        """
        if obs[0] == 1:
            return 'GROWL_LEFT'
        elif obs[1] == 1:
            return 'GROWL_RIGHT'
        else:
            return 'NONE'

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        if action == ACTION_OPEN_LEFT:
            self.left_door_open = True
        elif action == ACTION_OPEN_RIGHT:
            self.right_door_open = True
        elif action == ACTION_LISTEN:
            pass
        else:
            raise ValueError('Invalid action ', action)

    def _get_reward(self):
        if not (self.left_door_open or self.right_door_open):
            return -1
        if self.left_door_open:
            if self.tiger_left:
                return -100
            else:
                return 10
        if self.right_door_open:
            if self.tiger_right:
                return -100
            else:
                return 10
        raise ValueError('Unreachable state reached.')

    def _get_obs(self):
        last_action = self.action_episode_memory[self.curr_episode][-1]
        if last_action == ACTION_LISTEN:
            # Return accurate observation
            if np.random.rand() < .85:
                if self.tiger_left:
                    return [1, 0]
                else:
                    return [0, 1]
            # Return inaccurate observation
            else:
                if self.tiger_left:
                    return [0, 1]
                else:
                    return [1, 0]
        return [-1, -1]
