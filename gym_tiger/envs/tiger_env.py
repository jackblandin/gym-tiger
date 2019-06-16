# Core modules
import logging.config

# 3rd party modules
import gym
from gym import spaces
import numpy as np


OBS_GROWL_LEFT = [1, 0, 0, 0]
OBS_GROWL_RIGHT = [0, 1, 0, 0]
OBS_START = [0, 0, 1, 0]

ACTION_OPEN_LEFT = 0
ACTION_OPEN_RIGHT = 1
ACTION_LISTEN = 2
ACTION_MAP = {
    ACTION_OPEN_LEFT: 'OPEN_LEFT',
    ACTION_OPEN_RIGHT: 'OPEN_RIGHT',
    ACTION_LISTEN: 'LISTEN',
}


class TigerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, reward_tiger=-100, reward_gold=10, reward_listen=-1,
                 obs_accuracy=.85, max_steps_per_episode=100):
        """
        OpenAI Gym environment for the partially observable Tiger game.

        Parameters
        ----------
        reward_tiger : numeric
            Reward for opening the door with the tiger.
        reward_gold : numeric
            Reward for opening the door with the gold.
        reward_listen : numeric
            Reward for taking the listen action.
        obs_accuracy : numeric
            Number b/w 0 and 1. The accuracy of the growl. I.e. obs_accuracy of
            1 means that a GROWL_LEFT implies TIGER_LEFT 100% of the time.
        max_steps_per_episode : int, default=100
            Maximum allowed steps per episode. This will define how long an
            episode lasts, since the Tiger game does not end otherwise.

        Attributes
        ----------
        curr_episode : int
            Current episode as a count.
        action_episode_memory : list<int>
            History of actions taken in episode.
        curr_step : int
            Current timestep in episode, as a count.
        action_space : gym.spaces.Discrete
            Action space.
        observation_space : gym.spaces.Discrete
            Observation space.
        """
        self.reward_tiger = reward_tiger
        self.reward_gold = reward_gold
        self.reward_listen = reward_listen
        self.obs_accuracy = obs_accuracy
        self.max_steps_per_episode = max_steps_per_episode

        self.__version__ = "0.1.0"
        logging.info("TigerEnv - Version {}".format(self.__version__))

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

        self.curr_step = -1

        self.reset()

        # Define what the agent can do: LISTEN, OPEN_LEFT, OPEN_RIGHT
        self.action_space = spaces.Discrete(3)

        # Define what agent can observe: GROWL_LEFT, GROWL_RIGHT, START
        self.observation_space = spaces.Discrete(3)

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : list
                A list of ones or zeros which together represent the state of
                the environment.
            reward : float
                Amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                Whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : dict
                Diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """
        done = self.curr_step >= self.max_steps_per_episode
        if done:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        # Recompute done since action may have modified it
        done = self.curr_step >= self.max_steps_per_episode
        reward = self._get_reward()
        ob = self._get_obs()
        # Perform resets that happen after each timestep
        self._step_reset()
        return ob, reward, done, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        object
            The initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.left_door_open = False
        self.right_door_open = False
        self.tiger_left = np.random.randint(0, 2)
        self.tiger_right = 1 - self.tiger_left
        return OBS_START

    def render(self, mode='human'):
        return

    def close(self):
        pass

    def translate_obs(self, obs):
        """
        Method created by JDB to easily interpet the obs. in plain English.

        Parameters
        ----------
        obs : list or array-like
            The observation to be translated.

        Returns
        -------
        str
            A representation of the observation in English.
        """
        if obs[0] == 1:
            return 'GROWL_LEFT'
        elif obs[1] == 1:
            return 'GROWL_RIGHT'
        elif obs[2] == 1:
            return 'START'
        else:
            raise ValueError('Invalid observation: '.format(obs))

    def translate_action(self, action):
        """
        Method created by JDB to easily interpet the action in plain English.

        Parameters
        ----------
        action : int
            The action to be translated.

        Returns
        -------
        str
            A representation of the action in English.
        """
        return ACTION_MAP[action]

    def _take_action(self, action):
        """
        How to change the environment when taking an action.

        Parameters
        ----------
        action : int
            Action.

        Returns
        -------
        None
        """
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
        """
        Obtain the reward for the current state of the environment.

        Returns
        -------
        float
            Reward.
        """
        if not (self.left_door_open or self.right_door_open):
            return self.reward_listen
        if self.left_door_open:
            if self.tiger_left:
                return self.reward_tiger
            else:
                return self.reward_gold
        if self.right_door_open:
            if self.tiger_right:
                return self.reward_tiger
            else:
                return self.reward_gold
        raise ValueError('Unreachable state reached.')

    def _get_obs(self):
        """
        Obtain the observation for the current state of the environment.

        Returns
        -------
        list
            Observation.
        """
        last_action = self.action_episode_memory[self.curr_episode][-1]
        if last_action != ACTION_LISTEN:
            # Return accurate observation, but this won't be informative, since
            # the tiger will be reset afterwards.
            if self.tiger_left:
                return OBS_GROWL_LEFT
            else:
                return OBS_GROWL_RIGHT
        # Return accurate observation
        if np.random.rand() < self.obs_accuracy:
            if self.tiger_left:
                return OBS_GROWL_LEFT
            else:
                return OBS_GROWL_RIGHT
        # Return inaccurate observation
        else:
            if self.tiger_left:
                return OBS_GROWL_RIGHT
            else:
                return OBS_GROWL_LEFT

    def _step_reset(self):
        """
        Performs resets that happen after each timestep.

        Returns
        -------
        None
        """
        # Make sure doors are closed
        self.left_door_open = False
        self.right_door_open = False
        self.tiger_left = np.random.randint(0, 2)
        self.tiger_right = 1 - self.tiger_left

