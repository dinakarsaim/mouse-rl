import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
from robotic_detection_algo import analyze_mouse_path
import keyboard

screen_w, screen_h = pyautogui.size()

def env_to_screen(x, y):
    return int((x / 10) * screen_w), int((y / 10) * screen_h)

def move_real_mouse(x, y):
    screen_x, screen_y = env_to_screen(x, y)
    pyautogui.moveTo(screen_x, screen_y, duration=0.01)  # smooth move

class MouseEnvContinuous(gym.Env):
    def __init__(self, target=(0.8, 0.8), step_size=0.2, max_steps=200):
        super(MouseEnvContinuous, self).__init__()
        
        # self.mouse_pos = np.array([0.5, 0.5])
        self.mouse_pos = None
        # self.target = np.array(target)
        self.target = None
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps = 0
        self.path = []
        self.start_time = None
        self.points = []

        # Continuous action space: dx, dy between -1 and 1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation: mouse_x, mouse_y, target_x, target_y
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(5,), dtype=np.float32)

    def reset(self):
        # self.mouse_pos = np.array([5.0, 5.0])
        self.mouse_pos = np.random.uniform(low=1, high=9, size=(2,))
        self.target = np.random.uniform(low=0, high=10, size=(2,))
        self.steps = 0
        self.path = [self.mouse_pos.copy()]

        self.points = []
        self.start_time = time.time()

        self.points.append((0, self.mouse_pos[0], self.mouse_pos[1]))
        return np.concatenate([self.mouse_pos, self.target, [0.0]])

    def step(self, action):
        # dx, dy = action * self.step_size
        dx, dy, pause_flag_raw = action
        pause_flag = 1 if pause_flag_raw >= 0 else 0


        noise = np.random.normal(0, 0.5, size=2)
        noisy_dxdy = np.clip(np.array([dx, dy]) + noise, -1, 1)
        # if pause_flag > 0:
        #     # Pause → no movement
        #     dx, dy = 0.0, 0.0
        # else:
        #     # Normal move scaled by step_size
        #     dx, dy = dx * self.step_size, dy * self.step_size

        # self.mouse_pos += np.array([dx, dy])
        if pause_flag == 1:
        # Pause → no movement
            move = np.array([0.0, 0.0])
        else:
            # Normal move scaled by step_size
            move = noisy_dxdy * self.step_size

        self.mouse_pos += move
        self.mouse_pos = np.clip(self.mouse_pos, 0, 10)
        self.path.append(self.mouse_pos.copy())

        dist = np.linalg.norm(self.mouse_pos - self.target)
        reward = -dist

        self.steps += 1
        done = False

        current_time_ms = int((time.time() - self.start_time)*1000)
        self.points.append((current_time_ms, self.mouse_pos[0], self.mouse_pos[1]))
        
        # Calculate robotic score only at episode end
        if dist < 0.2 or self.steps >= self.max_steps:
            done = True
        
        # Base reward (closer is better)
        reward = -dist

        if done:
            scores = analyze_mouse_path(self.points)
            robotic_score = scores.get("robotic_score", 0)
            jitter = scores.get("jitter_magnitude", 0)     
            pause = scores.get("pause_count", 0)
            # Penalize robotic_score (scale as needed)
            reward -= robotic_score * 7  # tune this coefficient

            if 0.02 < jitter < 0.2:
                reward += 5

            # Encourage some small pauses (e.g., 1-3 pauses)
            if 1 <= pause <= 3:
                reward += 5
            
            if dist < 0.2:
                reward += 10
                if pause_flag == 1:
                    reward += 5
        
        obs = np.concatenate([self.mouse_pos, self.target, [float(pause_flag)]])
        info = {}
        return obs, reward, done, info
    

    def render(self):
        plt.figure(figsize=(10, 8))
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], marker="o")
        plt.scatter(*self.target, color='red', label="Target")
        plt.title("Mouse path")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
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

    model.learn(total_timesteps=250000, callback=callback)

    # Plot convergence (episode rewards)
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Convergence")
    plt.show()

    # Test the trained agent
    for _ in range (1):
        obs = env.reset()
        done = False
        truncated = False
        while not done:
            if keyboard.is_pressed("esc"):  # Emergency stop
                print("Stopped by user")
                break
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            move_real_mouse(env.mouse_pos[0], env.mouse_pos[1])

        env.render()

        metrics = analyze_mouse_path(env.points)
        print("Robotic Detection Metrics:")
        for key, val in metrics.items():
            print(f"  {key}: {val}")
        print()

if __name__ == "__main__":
    main()