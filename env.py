import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class MouseEnvContinuous(gym.Env):
    def __init__(self, target=(0.8, 0.8), step_size=0.02, max_steps=200):
        super(MouseEnvContinuous, self).__init__()
        
        # self.mouse_pos = np.array([0.5, 0.5])
        self.mouse_pos = None
        # self.target = np.array(target)
        self.target = None
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps = 0
        self.path = []

        # Continuous action space: dx, dy between -1 and 1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: mouse_x, mouse_y, target_x, target_y
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self):
        self.mouse_pos = np.array([0.5, 0.5])
        self.target = np.random.uniform(low=0.1, high=0.9, size=(2,))
        self.steps = 0
        self.path = [self.mouse_pos.copy()]
        return np.concatenate([self.mouse_pos, self.target])

    def step(self, action):
        dx, dy = action * self.step_size
        self.mouse_pos += np.array([dx, dy])
        self.mouse_pos = np.clip(self.mouse_pos, 0, 1)
        self.path.append(self.mouse_pos.copy())

        dist = np.linalg.norm(self.mouse_pos - self.target)
        reward = -dist

        self.steps += 1
        done = False
        truncated = False

        if dist < 0.02 or self.steps >= self.max_steps:
            done = True
            if dist < 0.02:
                reward += 10

        obs = np.concatenate([self.mouse_pos, self.target])
        info = {}

        return obs, reward, done, info

    def render(self):
        plt.figure(figsize=(5, 5))
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], marker="o")
        plt.scatter(*self.target, color='red', label="Target")
        plt.title("Mouse path")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # 'dones' tells which envs ended in this step
        done_array = self.locals.get("dones")
        infos = self.locals.get("infos")
        for done, info in zip(done_array, infos):
            if done and "episode" in info:
                # 'episode' info dict has 'r' = episode reward
                self.episode_rewards.append(info["episode"]["r"])
        return True

# def main():
#     env = MouseEnvContinuous()

#     model = PPO('MlpPolicy', env, verbose=1)
#     model.learn(total_timesteps=200000)

#     # Test the trained agent
#     obs = env.reset()
#     done = False
#     truncated = False
#     # while not done and not truncated:
#     while not done:
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)

#     env.render()

def main():
    env = MouseEnvContinuous()

    model = PPO('MlpPolicy', env, verbose=1)
    callback = RewardLoggerCallback()

    model.learn(total_timesteps=200000, callback=callback)

    # Plot convergence (episode rewards)
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Convergence")
    plt.show()

    # Test the trained agent
    for _ in range (5):
        obs = env.reset()
        done = False
        truncated = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

        env.render()

if __name__ == "__main__":
    main()