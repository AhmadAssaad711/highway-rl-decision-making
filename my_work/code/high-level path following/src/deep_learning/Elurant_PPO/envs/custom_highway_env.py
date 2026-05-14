import gymnasium as gym
import numpy as np
import highway_env


class CustomHighwayEnv(gym.Env):
    def __init__(self, render_mode=None, beta: float = 0.005):
        super().__init__()

        self.render_mode = render_mode

        config = {
            "observation": {
                "type": "Kinematics",
                "features": ["x", "y", "vx", "vy"],
                "vehicles_count": 10,
                "normalize": True,
                "absolute": False,
            },

            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True
            },

            # Traffic setup
            "vehicles_count": 60,
            "vehicles_density": 2.0,
            "ego_spacing": 1.0,

            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",

            "duration": 60,
            "policy_frequency": 5,
            "offroad_terminal": True,
        }

        self.env = gym.make(
            "highway-v0",
            render_mode=render_mode,
            config=config
        )

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.v_target = 25.0
        self.beta = float(beta)  # reward scale

    # -----------------------------------
    # RESET
    # -----------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    # -----------------------------------
    # STEP
    # -----------------------------------
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, _, terminated, truncated, info = self.env.step(action)

        reward = self.compute_reward(obs, terminated, truncated, info)

        # Optional logging (VERY useful)
        v = obs[0][2] * self.v_target
        info["speed"] = v
        info["reward"] = reward

        return obs, reward, terminated, truncated, info

    # -----------------------------------
    # REWARD (ONLY 2 TERMS)
    # -----------------------------------
    def compute_reward(self, obs, terminated, truncated, info):

        # 1) HARD PENALTY: collision / off-road only
        if terminated:
            return -1.0
        if truncated:
            return 0.0
        
        # 2) VELOCITY REWARD (positive only)
        v = obs[0][2] * self.v_target
        r_speed = np.clip(v / self.v_target, 0.0, 1.0)

        reward = self.beta * r_speed

        return reward

    # -----------------------------------
    # RENDER / CLOSE
    # -----------------------------------
    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
