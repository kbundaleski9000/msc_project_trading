from pettingzoo import AECEnv
import matplotlib.pyplot as plt
import numpy as np
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import torch
from scipy.optimize import brentq
import functools
import math
import gymnasium as gym
from gymnasium import spaces
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, SAC
from gym.wrappers import FlattenObservation
from typing import Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import random
from scipy.optimize import brentq
import torch
import torch.nn as nn
import torch.optim as optim


def equilibrium_insider_action(v, mean_v, beta):
    return beta * (v - mean_v)

def equilibrium_market_maker_action(y, mean_v, lambda_):
    return mean_v + lambda_ * y


class KyleOnePeriodAEC(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "KyleOnePeriodAEC",
    }

    def __init__(self, sigma_v=2.0, sigma_u=1.0, p0=0.5, gamma=0.9, action_scaling=5, T=1):
        super().__init__()
        self.sigma_v_initial = sigma_v
        self.sigma_u_initial = sigma_u
        self.p0_initial = p0
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.p0 = p0
        self.timestep = 0
        self.gamma = gamma
        self.T = T
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)


        self.possible_agents = ["insider", "market_maker"]
        self.agents = self.possible_agents[:]
        self.render_mode = None
        self.rewards = {a: 0 for a in self.agents}
        self.agent_selector = agent_selector(self.agents)

        self.action_spaces = {
            "insider": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            "market_maker": spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        }

        self.observation_spaces = {
            "insider": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
            "market_maker": spaces.Box(low=-10, high=10, shape=(3,), dtype=np.float32),
        }
        self.action_scaling = action_scaling

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.timestep = 0
        self.sigma_v = self.sigma_v_initial
        self.sigma_u = self.sigma_u_initial
        self.p0 = self.p0_initial
        self.optimal_x = 0
        self.optimal_y = 0
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)

        # Sample true value
        self.v = torch.normal(self.p0, self.sigma_v, size=(1,)).item()
        self.u = torch.normal(0, self.sigma_u, size=(1,)).item()
        self.p = 0
        self.y = 0

        # Set first agent
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self.agent_selector.next()

    def solve_kyle_model(self):
        """
        Calculates the multi-period Kyle model equilibrium using a two-pass algorithm.

        Args:
            N (int): The total number of trading periods.
            sigma2_v (float): The initial variance of the asset's value (Sigma_0).
            sigma2_u (float): The variance of the noise trader's orders.

        Returns:
            dict: A dictionary containing the numpy arrays for each equilibrium
                parameter sequence: 'Sigma', 'Lambda', 'Beta', and 'Alpha'.
        """
        # Helper for readability
        sigma_u = np.sqrt(self.sigma_u)

        # --- ⏪ PASS 1: BACKWARD ITERATION for h_n ---
        # h_n = Sigma_n / Sigma_{n-1}
        h = np.zeros(self.T)
        h[-1] = 0.5  # Terminal condition for the last period (n=N)

        # Iterate backwards from n = N-1 down to n=1
        for n in reversed(range(self.T - 1)):
            # Calculate theta_{n+1} from the known h_{n+1}
            h_next = h[n + 1]
            theta_next = np.sqrt(h_next / (1 - h_next))

            # Define the function whose root h_n we need to find
            # f(h_n) = theta_{n+1} - (2*h_n - 1) / (h_n * sqrt(1-h_n)) = 0
            def root_func(h_n):
                if h_n <= 0.5 or h_n >= 1.0: # Numerical stability
                    return np.inf
                return theta_next - (2 * h_n - 1) / (h_n * np.sqrt(1 - h_n))

            # Find the root h_n in the interval (0.5, 1.0)
            # brentq is a robust and efficient choice here.
            h[n] = brentq(root_func, 0.5 + 1e-9, 1.0 - 1e-9)

        # --- ⏩ PASS 2: FORWARD ITERATION for equilibrium parameters ---
        Sigma = np.zeros(self.T + 1)
        Lambda = np.zeros(self.T)
        Beta = np.zeros(self.T)
        Alpha = np.zeros(self.T)

        Sigma[0] = self.sigma_v_initial # Initial condition

        # Iterate forward from n=1 to N
        for n in range(self.T):
            # Update variance using the pre-calculated h_n
            Sigma[n + 1] = h[n] * Sigma[n]

            # Calculate the equilibrium parameters for period n
            Lambda[n] = np.sqrt((1 - h[n]) * Sigma[n + 1]) / sigma_u
            Beta[n] = (np.sqrt(1 - h[n]) * sigma_u) / np.sqrt(Sigma[n + 1])
            Alpha[n] = (sigma_u / (2 * np.sqrt(Sigma[n]))) * np.sqrt(h[n] / (1 - h[n]))
        
        self.beta = Beta
        self.lmbd = Lambda
        self.alpha = Alpha
        self.h = h
        self.Sigma = Sigma

        self.delta[self.T-1] = 0  # Terminal condition
        
        for n in reversed(range(1, self.T-1)):
            self.delta[n-1] = self.delta[n] + self.alpha[n] * ( self.lmbd[n] ** 2 ) * self.sigma_u


    def observe_insider(self):
        return torch.tensor([self.v, self.p0, self.beta[self.timestep]], dtype=torch.float32)

    def observe_market_maker(self):
        return torch.tensor([self.y, self.p0, self.lmbd[self.timestep]], dtype=torch.float32)

    def observe(self, agent):
        if agent == "insider":
            return self.observe_insider()
        elif agent == "market_maker":
            return self.observe_market_maker()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection

        # Store the action
        if agent == "insider":


            self.insider_action = action[0] * self.action_scaling
            self.u = torch.normal(0, self.sigma_u, size=(1,)).item()
            self.y = self.insider_action + self.u

        elif agent == "market_maker":
            self.market_maker_action = action[0] * self.action_scaling


            insider_profit = (self.v - self.market_maker_action) * self.insider_action 
            market_maker_loss = - (self.market_maker_action - self.v) ** 2

            # Assign rewards
            self.rewards = {
                "insider": insider_profit,
                "market_maker": market_maker_loss
            }

            self.p0 = equilibrium_market_maker_action(self.y, self.p0, self.lmbd[self.timestep])

            # Update timestep
            self.timestep += 1
            if self.timestep >= self.T:
                self.terminations = {a: True for a in self.agents}

            # Update cumulative rewards
            for a in self.agents:
                self._cumulative_rewards[a] += self.rewards[a]

        # Get next agent
        self.agent_selection = self.agent_selector.next()

    def render(self):
        print(f"Step: {self.timestep}")
        print(f"True value v: {self.v:.2f}")
        print(f"Order flow y: {self.y:.2f}")
        if hasattr(self, 'rewards'):
            print(f"Agents' rewards: {self._cumulative_rewards}")

    def close(self):
        pass


class KyleModelEnvInsider(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, sigma_v=2.0, sigma_u=1.0, p0=0.5, T=25, gamma=0.95, action_scaling = 5):
        super(KyleModelEnvInsider, self).__init__()
        
        self.action_scaling = action_scaling
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.p0 = p0
        self.T = T
        self.u = 0.0
        self.gamma = gamma
        self.actual_actions = []
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.h = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.solve_kyle_model()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'v': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'p': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_v': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_u': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'time': spaces.Box(low=0, high=self.T, shape=(1,), dtype=np.float32)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.v = np.random.normal(self.p0, self.sigma_v)  # True asset value
        self.p = self.p0
        self.t = 0
        self.solve_kyle_model()

        obs = self._get_insider_obs()
        info = {}
        return obs, info  # Gymnasium requires (obs, info)

    def _get_insider_obs(self):
        return {
            'v': np.array([self.v], dtype=np.float32),
            'p': np.array([self.p], dtype=np.float32),
            'sigma_v': np.array([self.sigma_v], dtype=np.float32),
            'sigma_u': np.array([self.sigma_u], dtype=np.float32),
            'time': np.array([self.t/self.T], dtype=np.float32)
        }
    
    def solve_kyle_model(self):
        """
        Calculates the multi-period Kyle model equilibrium using a two-pass algorithm.

        Args:
            N (int): The total number of trading periods.
            sigma2_v (float): The initial variance of the asset's value (Sigma_0).
            sigma2_u (float): The variance of the noise trader's orders.

        Returns:
            dict: A dictionary containing the numpy arrays for each equilibrium
                parameter sequence: 'Sigma', 'Lambda', 'Beta', and 'Alpha'.
        """
        # Helper for readability
        sigma_u = np.sqrt(self.sigma_u)

        # --- ⏪ PASS 1: BACKWARD ITERATION for h_n ---
        # h_n = Sigma_n / Sigma_{n-1}
        h = np.zeros(self.T)
        h[-1] = 0.5  # Terminal condition for the last period (n=N)

        # Iterate backwards from n = N-1 down to n=1
        for n in reversed(range(self.T - 1)):
            # Calculate theta_{n+1} from the known h_{n+1}
            h_next = h[n + 1]
            theta_next = np.sqrt(h_next / (1 - h_next))

            # Define the function whose root h_n we need to find
            # f(h_n) = theta_{n+1} - (2*h_n - 1) / (h_n * sqrt(1-h_n)) = 0
            def root_func(h_n):
                if h_n <= 0.5 or h_n >= 1.0: # Numerical stability
                    return np.inf
                return theta_next - (2 * h_n - 1) / (h_n * np.sqrt(1 - h_n))

            # Find the root h_n in the interval (0.5, 1.0)
            # brentq is a robust and efficient choice here.
            h[n] = brentq(root_func, 0.5 + 1e-9, 1.0 - 1e-9)

        # --- ⏩ PASS 2: FORWARD ITERATION for equilibrium parameters ---
        Sigma = np.zeros(self.T + 1)
        Lambda = np.zeros(self.T)
        Beta = np.zeros(self.T)
        Alpha = np.zeros(self.T)

        Sigma[0] = self.sigma_v # Initial condition

        # Iterate forward from n=1 to N
        for n in range(self.T):
            # Update variance using the pre-calculated h_n
            Sigma[n + 1] = h[n] * Sigma[n]

            # Calculate the equilibrium parameters for period n
            Lambda[n] = np.sqrt((1 - h[n]) * Sigma[n + 1]) / sigma_u
            Beta[n] = (np.sqrt(1 - h[n]) * sigma_u) / np.sqrt(Sigma[n + 1])
            Alpha[n] = (sigma_u / (2 * np.sqrt(Sigma[n]))) * np.sqrt(h[n] / (1 - h[n]))
        
        self.beta = Beta
        self.lmbd = Lambda
        self.alpha = Alpha
        self.h = h
        self.Sigma = Sigma

        self.delta[self.T-1] = 0  # Terminal condition
        
        for n in reversed(range(1, self.T-1)):
            self.delta[n-1] = self.delta[n] + self.alpha[n] * ( self.lmbd[n] ** 2 ) * self.sigma_u

    def step(self, action):
        """Execute one step"""
        if self.t >= self.T:
            raise ValueError("Episode has already ended")

        scaled_action = action * self.action_scaling
        self.actual_actions.append(scaled_action[0])

        x = scaled_action[0]
        self.u = np.random.normal(0, self.sigma_u)
        y = x + self.u
        p_new = self.p + self.lmbd[self.t] * y


        insider_profit = (self.v - p_new) * x

        # Update state
        self.p = p_new
        self.t += 1
            

        terminated = (self.t >= self.T)  # natural end
        truncated = False  # No time/truncation logic here

        obs = self._get_insider_obs()
        info = {}

        return obs, insider_profit, terminated, truncated, info



class KyleModelEnvInsiderHistoryPrice(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, sigma_v=2.0, sigma_u=1.0, p0=0.5, T=25, gamma=0.95, action_scaling = 5):
        super(KyleModelEnvInsiderHistoryPrice, self).__init__()
        
        self.action_scaling = action_scaling
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.p0 = p0
        self.T = T
        self.u = 0.0
        self.gamma = gamma
        self.actual_actions = []
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.h = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.history_price = np.zeros(self.T, dtype=np.float32)
        self.history_price[0] = self.p0
        self.history_action = np.zeros(self.T, dtype=np.float32)
        self.solve_kyle_model()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'v': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'history_price': spaces.Box(low=-np.inf, high=np.inf, shape=(self.T,), dtype=np.float32),
            'history_action': spaces.Box(low=-np.inf, high=np.inf, shape=(self.T,), dtype=np.float32),
            'sigma_v': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_u': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'time': spaces.Box(low=0, high=self.T, shape=(1,), dtype=np.float32)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.v = np.random.normal(self.p0, self.sigma_v)  # True asset value
        self.p = self.p0
        self.t = 0
        self.solve_kyle_model()
        self.history_price = np.zeros(self.T, dtype=np.float32)
        self.history_action = np.zeros(self.T, dtype=np.float32)
        self.history_price[0] = self.p

        obs = self._get_insider_obs()
        info = {}
        return obs, info  # Gymnasium requires (obs, info)

    def _get_insider_obs(self):
        return {
            'v': np.array([self.v], dtype=np.float32),
            'history_price': self.history_price.reshape(-1,),
            'history_action': self.history_action.reshape(-1,),
            'sigma_v': np.array([self.sigma_v], dtype=np.float32),
            'sigma_u': np.array([self.sigma_u], dtype=np.float32),
            'time': np.array([self.t/self.T], dtype=np.float32)
        }
    
    def solve_kyle_model(self):
        """
        Calculates the multi-period Kyle model equilibrium using a two-pass algorithm.

        Args:
            N (int): The total number of trading periods.
            sigma2_v (float): The initial variance of the asset's value (Sigma_0).
            sigma2_u (float): The variance of the noise trader's orders.

        Returns:
            dict: A dictionary containing the numpy arrays for each equilibrium
                parameter sequence: 'Sigma', 'Lambda', 'Beta', and 'Alpha'.
        """
        # Helper for readability
        sigma_u = np.sqrt(self.sigma_u)

        # --- ⏪ PASS 1: BACKWARD ITERATION for h_n ---
        # h_n = Sigma_n / Sigma_{n-1}
        h = np.zeros(self.T)
        h[-1] = 0.5  # Terminal condition for the last period (n=N)

        # Iterate backwards from n = N-1 down to n=1
        for n in reversed(range(self.T - 1)):
            # Calculate theta_{n+1} from the known h_{n+1}
            h_next = h[n + 1]
            theta_next = np.sqrt(h_next / (1 - h_next))

            # Define the function whose root h_n we need to find
            # f(h_n) = theta_{n+1} - (2*h_n - 1) / (h_n * sqrt(1-h_n)) = 0
            def root_func(h_n):
                if h_n <= 0.5 or h_n >= 1.0: # Numerical stability
                    return np.inf
                return theta_next - (2 * h_n - 1) / (h_n * np.sqrt(1 - h_n))

            # Find the root h_n in the interval (0.5, 1.0)
            # brentq is a robust and efficient choice here.
            h[n] = brentq(root_func, 0.5 + 1e-9, 1.0 - 1e-9)

        # --- ⏩ PASS 2: FORWARD ITERATION for equilibrium parameters ---
        Sigma = np.zeros(self.T + 1)
        Lambda = np.zeros(self.T)
        Beta = np.zeros(self.T)
        Alpha = np.zeros(self.T)

        Sigma[0] = self.sigma_v # Initial condition

        # Iterate forward from n=1 to N
        for n in range(self.T):
            # Update variance using the pre-calculated h_n
            Sigma[n + 1] = h[n] * Sigma[n]

            # Calculate the equilibrium parameters for period n
            Lambda[n] = np.sqrt((1 - h[n]) * Sigma[n + 1]) / sigma_u
            Beta[n] = (np.sqrt(1 - h[n]) * sigma_u) / np.sqrt(Sigma[n + 1])
            Alpha[n] = (sigma_u / (2 * np.sqrt(Sigma[n]))) * np.sqrt(h[n] / (1 - h[n]))
        
        self.beta = Beta
        self.lmbd = Lambda
        self.alpha = Alpha
        self.h = h
        self.Sigma = Sigma

        self.delta[self.T-1] = 0  # Terminal condition
        
        for n in reversed(range(1, self.T-1)):
            self.delta[n-1] = self.delta[n] + self.alpha[n] * ( self.lmbd[n] ** 2 ) * self.sigma_u

    def step(self, action):
        """Execute one step"""
        if self.t >= self.T:
            raise ValueError("Episode has already ended")

        scaled_action = action * self.action_scaling
        self.actual_actions.append(scaled_action[0])

        x = scaled_action[0]
        self.u = np.random.normal(0, self.sigma_u)
        y = x + self.u
        p_new = self.p + self.lmbd[self.t] * y


        insider_profit = (self.v - p_new) * x

        self.history_action[self.t] = scaled_action
        # Update state
        self.p = p_new
        self.t += 1

        if self.t < self.T:
            self.history_price[self.t] = self.p
            

        terminated = (self.t >= self.T)  # natural end
        truncated = False  # No time/truncation logic here

        obs = self._get_insider_obs()
        info = {}

        return obs, insider_profit, terminated, truncated, info


class KyleModelEnvInsiderHistoryPrice(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, sigma_v=2.0, sigma_u=1.0, p0=0.5, T=25, gamma=0.95, action_scaling = 5):
        super(KyleModelEnvInsiderHistoryPrice, self).__init__()
        
        self.action_scaling = action_scaling
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.p0 = p0
        self.T = T
        self.u = 0.0
        self.gamma = gamma
        self.actual_actions = []
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.h = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.history_price = np.zeros(self.T, dtype=np.float32)
        self.history_price[0] = self.p0
        self.history_action = np.zeros(self.T, dtype=np.float32)
        self.solve_kyle_model()

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'v': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'history_price': spaces.Box(low=-np.inf, high=np.inf, shape=(self.T,), dtype=np.float32),
            'sigma_v': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_u': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'time': spaces.Box(low=0, high=self.T, shape=(1,), dtype=np.float32)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        self.v = np.random.normal(self.p0, self.sigma_v)  # True asset value
        self.p = self.p0
        self.t = 0
        self.solve_kyle_model()
        self.history_price = np.zeros(self.T, dtype=np.float32)
        self.history_action = np.zeros(self.T, dtype=np.float32)
        self.history_price[0] = self.p

        obs = self._get_insider_obs()
        info = {}
        return obs, info  # Gymnasium requires (obs, info)

    def _get_insider_obs(self):
        return {
            'v': np.array([self.v], dtype=np.float32),
            'history_price': self.history_price.reshape(-1,),
            'sigma_v': np.array([self.sigma_v], dtype=np.float32),
            'sigma_u': np.array([self.sigma_u], dtype=np.float32),
            'time': np.array([self.t/self.T], dtype=np.float32)
        }
    
    def solve_kyle_model(self):
        """
        Calculates the multi-period Kyle model equilibrium using a two-pass algorithm.

        Args:
            N (int): The total number of trading periods.
            sigma2_v (float): The initial variance of the asset's value (Sigma_0).
            sigma2_u (float): The variance of the noise trader's orders.

        Returns:
            dict: A dictionary containing the numpy arrays for each equilibrium
                parameter sequence: 'Sigma', 'Lambda', 'Beta', and 'Alpha'.
        """
        # Helper for readability
        sigma_u = np.sqrt(self.sigma_u)

        # --- ⏪ PASS 1: BACKWARD ITERATION for h_n ---
        # h_n = Sigma_n / Sigma_{n-1}
        h = np.zeros(self.T)
        h[-1] = 0.5  # Terminal condition for the last period (n=N)

        # Iterate backwards from n = N-1 down to n=1
        for n in reversed(range(self.T - 1)):
            # Calculate theta_{n+1} from the known h_{n+1}
            h_next = h[n + 1]
            theta_next = np.sqrt(h_next / (1 - h_next))

            # Define the function whose root h_n we need to find
            # f(h_n) = theta_{n+1} - (2*h_n - 1) / (h_n * sqrt(1-h_n)) = 0
            def root_func(h_n):
                if h_n <= 0.5 or h_n >= 1.0: # Numerical stability
                    return np.inf
                return theta_next - (2 * h_n - 1) / (h_n * np.sqrt(1 - h_n))

            # Find the root h_n in the interval (0.5, 1.0)
            # brentq is a robust and efficient choice here.
            h[n] = brentq(root_func, 0.5 + 1e-9, 1.0 - 1e-9)

        # --- ⏩ PASS 2: FORWARD ITERATION for equilibrium parameters ---
        Sigma = np.zeros(self.T + 1)
        Lambda = np.zeros(self.T)
        Beta = np.zeros(self.T)
        Alpha = np.zeros(self.T)

        Sigma[0] = self.sigma_v # Initial condition

        # Iterate forward from n=1 to N
        for n in range(self.T):
            # Update variance using the pre-calculated h_n
            Sigma[n + 1] = h[n] * Sigma[n]

            # Calculate the equilibrium parameters for period n
            Lambda[n] = np.sqrt((1 - h[n]) * Sigma[n + 1]) / sigma_u
            Beta[n] = (np.sqrt(1 - h[n]) * sigma_u) / np.sqrt(Sigma[n + 1])
            Alpha[n] = (sigma_u / (2 * np.sqrt(Sigma[n]))) * np.sqrt(h[n] / (1 - h[n]))
        
        self.beta = Beta
        self.lmbd = Lambda
        self.alpha = Alpha
        self.h = h
        self.Sigma = Sigma

        self.delta[self.T-1] = 0  # Terminal condition
        
        for n in reversed(range(1, self.T-1)):
            self.delta[n-1] = self.delta[n] + self.alpha[n] * ( self.lmbd[n] ** 2 ) * self.sigma_u

    def step(self, action):
        """Execute one step"""
        if self.t >= self.T:
            raise ValueError("Episode has already ended")

        scaled_action = action * self.action_scaling
        self.actual_actions.append(scaled_action[0])

        x = scaled_action[0]
        self.u = np.random.normal(0, self.sigma_u)
        y = x + self.u
        p_new = self.p + self.lmbd[self.t] * y


        insider_profit = (self.v - p_new) * x

        self.history_action[self.t] = scaled_action
        # Update state
        self.p = p_new
        self.t += 1

        if self.t < self.T:
            self.history_price[self.t] = self.p
            

        terminated = (self.t >= self.T)  # natural end
        truncated = False  # No time/truncation logic here

        obs = self._get_insider_obs()
        info = {}

        return obs, insider_profit, terminated, truncated, info


class KyleModelEnvMM(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, sigma_v=2.0, sigma_u=1.0, p0=0.5, T=25, gamma=0.9, action_scaling = 5):
        super(KyleModelEnvMM, self).__init__()

        self.action_scaling = action_scaling
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.p0 = p0
        self.T = T
        self.gamma = gamma
        self.actual_actions = []
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.h = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.solve_kyle_model()

        # Define action space (Market Maker's price guess)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space for the market maker
        self.observation_space = spaces.Dict({
            'y': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'p': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_v': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_u': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'time': spaces.Box(low=0, high=self.T, shape=(1,), dtype=np.float32)
        })

    def solve_kyle_model(self):
        """
        Calculates the multi-period Kyle model equilibrium using a two-pass algorithm.

        Args:
            N (int): The total number of trading periods.
            sigma2_v (float): The initial variance of the asset's value (Sigma_0).
            sigma2_u (float): The variance of the noise trader's orders.

        Returns:
            dict: A dictionary containing the numpy arrays for each equilibrium
                parameter sequence: 'Sigma', 'Lambda', 'Beta', and 'Alpha'.
        """
        # Helper for readability
        sigma_u = np.sqrt(self.sigma_u)

        # --- ⏪ PASS 1: BACKWARD ITERATION for h_n ---
        # h_n = Sigma_n / Sigma_{n-1}
        h = np.zeros(self.T)
        h[-1] = 0.5  # Terminal condition for the last period (n=N)

        # Iterate backwards from n = N-1 down to n=1
        for n in reversed(range(self.T - 1)):
            # Calculate theta_{n+1} from the known h_{n+1}
            h_next = h[n + 1]
            theta_next = np.sqrt(h_next / (1 - h_next))

            # Define the function whose root h_n we need to find
            # f(h_n) = theta_{n+1} - (2*h_n - 1) / (h_n * sqrt(1-h_n)) = 0
            def root_func(h_n):
                if h_n <= 0.5 or h_n >= 1.0: # Numerical stability
                    return np.inf
                return theta_next - (2 * h_n - 1) / (h_n * np.sqrt(1 - h_n))

            # Find the root h_n in the interval (0.5, 1.0)
            # brentq is a robust and efficient choice here.
            h[n] = brentq(root_func, 0.5 + 1e-9, 1.0 - 1e-9)

        # --- ⏩ PASS 2: FORWARD ITERATION for equilibrium parameters ---
        Sigma = np.zeros(self.T + 1)
        Lambda = np.zeros(self.T)
        Beta = np.zeros(self.T)
        Alpha = np.zeros(self.T)

        Sigma[0] = self.sigma_v # Initial condition

        # Iterate forward from n=1 to N
        for n in range(self.T):
            # Update variance using the pre-calculated h_n
            Sigma[n + 1] = h[n] * Sigma[n]

            # Calculate the equilibrium parameters for period n
            Lambda[n] = np.sqrt((1 - h[n]) * Sigma[n + 1]) / sigma_u
            Beta[n] = (np.sqrt(1 - h[n]) * sigma_u) / np.sqrt(Sigma[n + 1])
            Alpha[n] = (sigma_u / (2 * np.sqrt(Sigma[n]))) * np.sqrt(h[n] / (1 - h[n]))
        
        self.beta = Beta
        self.lmbd = Lambda
        self.alpha = Alpha
        self.h = h
        self.Sigma = Sigma

        self.delta[self.T-1] = 0  # Terminal condition
        
        for n in reversed(range(1, self.T-1)):
            self.delta[n-1] = self.delta[n] + self.alpha[n] * ( self.lmbd[n] ** 2 ) * self.sigma_u

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # True asset value is drawn from N(p0, sigma_v^2)
        self.v = np.random.normal(self.p0, self.sigma_v)

        # Initialize state variables
        self.p = self.p0
        self.t = 0

        # Generate initial observation
        self.x = self.beta[self.t] * (self.v - self.p)
        self.u = np.random.normal(0, self.sigma_u)
        self.y = self.x + self.u

        obs = self._get_mm_obs()
        info = {}
        return obs, info  # Gymnasium requires (obs, info)

    def _get_mm_obs(self) -> Dict[str, np.ndarray]:
        return {
            'y': np.array([self.y], dtype=np.float32),
            'p': np.array([self.p], dtype=np.float32),
            'sigma_v': np.array([self.sigma_v], dtype=np.float32),
            'sigma_u': np.array([self.sigma_u], dtype=np.float32),
            'time': np.array([self.t/self.T], dtype=np.float32)
        }

    def step(self, action):
        """Execute one step of the environment"""
        if self.t >= self.T:
            raise ValueError("Episode has already ended")

        # Scale PPO's normalized action to market price guess
        scaled_action = action[0] * self.action_scaling
        self.actual_actions.append(scaled_action)

        # Market maker's loss function (negative MSE against true value)
        mm_loss = - (scaled_action - self.v)**2
        self.p = scaled_action

        # Update timestep
        self.t += 1

        if self.t < self.T:
            self.x = self.beta[self.t] * (self.v - self.p)
            self.u = np.random.normal(0, self.sigma_u)
            self.y = self.x + self.u

        terminated = (self.t >= self.T)  # natural termination
        truncated = False  # no time truncation here

        obs = self._get_mm_obs()
        info = {}

        return obs, mm_loss, terminated, truncated, info


class KyleModelEnvMMHistory(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, sigma_v=2.0, sigma_u=1.0, p0=0.5, T=25, gamma=0.9, action_scaling = 5):
        super(KyleModelEnvMMHistory, self).__init__()

        self.action_scaling = action_scaling
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.p0 = p0
        self.T = T
        self.gamma = gamma
        self.actual_actions = []
        self.beta = np.zeros(self.T)
        self.lmbd = np.zeros(self.T)
        self.alpha = np.zeros(self.T)
        self.h = np.zeros(self.T)
        self.Sigma = np.zeros(self.T)
        self.delta = np.zeros(self.T)
        self.solve_kyle_model()

        # Define action space (Market Maker's price guess)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space for the market maker
        self.observation_space = spaces.Dict({
            'v': spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            'history_y': spaces.Box(low=-np.inf, high=np.inf, shape=(self.T,), dtype=np.float32),
            'sigma_v': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'sigma_u': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'time': spaces.Box(low=0, high=self.T, shape=(1,), dtype=np.float32)
        })

    def solve_kyle_model(self):
        """
        Calculates the multi-period Kyle model equilibrium using a two-pass algorithm.

        Args:
            N (int): The total number of trading periods.
            sigma2_v (float): The initial variance of the asset's value (Sigma_0).
            sigma2_u (float): The variance of the noise trader's orders.

        Returns:
            dict: A dictionary containing the numpy arrays for each equilibrium
                parameter sequence: 'Sigma', 'Lambda', 'Beta', and 'Alpha'.
        """
        # Helper for readability
        sigma_u = np.sqrt(self.sigma_u)

        # --- ⏪ PASS 1: BACKWARD ITERATION for h_n ---
        # h_n = Sigma_n / Sigma_{n-1}
        h = np.zeros(self.T)
        h[-1] = 0.5  # Terminal condition for the last period (n=N)

        # Iterate backwards from n = N-1 down to n=1
        for n in reversed(range(self.T - 1)):
            # Calculate theta_{n+1} from the known h_{n+1}
            h_next = h[n + 1]
            theta_next = np.sqrt(h_next / (1 - h_next))

            # Define the function whose root h_n we need to find
            # f(h_n) = theta_{n+1} - (2*h_n - 1) / (h_n * sqrt(1-h_n)) = 0
            def root_func(h_n):
                if h_n <= 0.5 or h_n >= 1.0: # Numerical stability
                    return np.inf
                return theta_next - (2 * h_n - 1) / (h_n * np.sqrt(1 - h_n))

            # Find the root h_n in the interval (0.5, 1.0)
            # brentq is a robust and efficient choice here.
            h[n] = brentq(root_func, 0.5 + 1e-9, 1.0 - 1e-9)

        # --- ⏩ PASS 2: FORWARD ITERATION for equilibrium parameters ---
        Sigma = np.zeros(self.T + 1)
        Lambda = np.zeros(self.T)
        Beta = np.zeros(self.T)
        Alpha = np.zeros(self.T)

        Sigma[0] = self.sigma_v # Initial condition

        # Iterate forward from n=1 to N
        for n in range(self.T):
            # Update variance using the pre-calculated h_n
            Sigma[n + 1] = h[n] * Sigma[n]

            # Calculate the equilibrium parameters for period n
            Lambda[n] = np.sqrt((1 - h[n]) * Sigma[n + 1]) / sigma_u
            Beta[n] = (np.sqrt(1 - h[n]) * sigma_u) / np.sqrt(Sigma[n + 1])
            Alpha[n] = (sigma_u / (2 * np.sqrt(Sigma[n]))) * np.sqrt(h[n] / (1 - h[n]))
        
        self.beta = Beta
        self.lmbd = Lambda
        self.alpha = Alpha
        self.h = h
        self.Sigma = Sigma

        self.delta[self.T-1] = 0  # Terminal condition
        
        for n in reversed(range(1, self.T-1)):
            self.delta[n-1] = self.delta[n] + self.alpha[n] * ( self.lmbd[n] ** 2 ) * self.sigma_u

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # True asset value is drawn from N(p0, sigma_v^2)
        self.v = np.random.normal(self.p0, self.sigma_v)

        # Initialize state variables
        self.p = self.p0
        self.t = 0

        # Generate initial observation
        self.x = self.beta[self.t] * (self.v - self.p)
        self.u = np.random.normal(0, self.sigma_u)
        self.y = self.x + self.u

        obs = self._get_mm_obs()
        info = {}
        return obs, info  # Gymnasium requires (obs, info)

    def _get_mm_obs(self) -> Dict[str, np.ndarray]:
        return {
            'y': np.array([self.y], dtype=np.float32),
            'p': np.array([self.p], dtype=np.float32),
            'sigma_v': np.array([self.sigma_v], dtype=np.float32),
            'sigma_u': np.array([self.sigma_u], dtype=np.float32),
            'time': np.array([self.t/self.T], dtype=np.float32)
        }

    def step(self, action):
        """Execute one step of the environment"""
        if self.t >= self.T:
            raise ValueError("Episode has already ended")

        # Scale PPO's normalized action to market price guess
        scaled_action = action[0] * self.action_scaling
        self.actual_actions.append(scaled_action)

        # Market maker's loss function (negative MSE against true value)
        mm_loss = - (scaled_action - self.v)**2
        self.p = scaled_action

        # Update timestep
        self.t += 1

        if self.t < self.T:
            self.x = self.beta[self.t] * (self.v - self.p)
            self.u = np.random.normal(0, self.sigma_u)
            self.y = self.x + self.u

        terminated = (self.t >= self.T)  # natural termination
        truncated = False  # no time truncation here

        obs = self._get_mm_obs()
        info = {}

        return obs, mm_loss, terminated, truncated, info