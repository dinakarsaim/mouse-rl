import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
from robotic_detection_algo import analyze_mouse_path
# NOTE: no pyautogui/keyboard imports at top level

class MouseEnvContinuous(gym.Env):
    def __init__(self, step_size=0.2, max_steps=200):
        super(MouseEnvContinuous, self).__init__()
        self.mouse_pos = None
        self.target = None
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps = 0
        self.path = []
        self.start_time = None
        self.points = []

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(5,), dtype=np.float32)

    def reset(self):
        self.mouse_pos = np.random.uniform(low=1, high=9, size=(2,))
        self.target = np.random.uniform(low=0, high=10, size=(2,))
        self.steps = 0
        self.path = [self.mouse_pos.copy()]
        self.points = []
        self.start_time = time.time()
        self.points.append((0, self.mouse_pos[0], self.mouse_pos[1]))
        return np.concatenate([self.mouse_pos, self.target, [0.0]])

    def step(self, action):
        dx, dy, pause_flag_raw = action
        pause_flag = 1 if pause_flag_raw >= 0 else 0

        noise = np.random.normal(0, 0.5, size=2)
        noisy_dxdy = np.clip(np.array([dx, dy]) + noise, -1, 1)

        if pause_flag == 1:
            move = np.array([0.0, 0.0])
        else:
            move = noisy_dxdy * self.step_size

        self.mouse_pos += move
        self.mouse_pos = np.clip(self.mouse_pos, 0, 10)
        self.path.append(self.mouse_pos.copy())

        dist = np.linalg.norm(self.mouse_pos - self.target)
        reward = -dist

        self.steps += 1
        done = False

        current_time_ms = int((time.time() - self.start_time) * 1000)
        self.points.append((current_time_ms, self.mouse_pos[0], self.mouse_pos[1]))

        if dist < 0.2 or self.steps >= self.max_steps:
            done = True

        if done:
            scores = analyze_mouse_path(self.points)
            robotic_score = scores.get("robotic_score", 0)
            jitter = scores.get("jitter_magnitude", 0)
            pause = scores.get("pause_count", 0)

            reward -= robotic_score * 7
            if 0.02 < jitter < 0.2:
                reward += 5
            if 1 <= pause <= 3:
                reward += 5
            if dist < 0.2:
                reward += 10
                if pause_flag == 1:
                    reward += 5

        obs = np.concatenate([self.mouse_pos, self.target, [float(pause_flag)]])
        info = {}
        return obs, reward, done, info

    def render(self, show=True):
        plt.figure(figsize=(10, 8))
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], marker="o")
        plt.scatter(*self.target, label="Target")
        plt.title("Mouse path")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.legend()
        if show:
            plt.show()

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")
        infos = self.locals.get("infos")
        for done, info in zip(done_array, infos):
            if done and "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

def train_only(total_timesteps=250000):
    env = MouseEnvContinuous()
    model = PPO('MlpPolicy', env, verbose=1)
    callback = RewardLoggerCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("mouse_model.zip")
    plt.figure()
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Convergence")
    plt.show()
    return model

def test_with_real_mouse(model, episodes=1):
    # Import here to avoid touching GUI during training
    import pyautogui
    pyautogui.PAUSE = 0.01  # throttle a bit
    # optional: disable failsafe if needed (moving to corner raises exception)
    # pyautogui.FAILSAFE = False

    screen_w, screen_h = pyautogui.size()

    def env_to_screen(x, y):
        return int((x / 10.0) * screen_w), int((y / 10.0) * screen_h)

    def move_real_mouse(x, y):
        sx, sy = env_to_screen(x, y)
        try:
            pyautogui.moveTo(sx, sy, duration=0.01)
            time.sleep(1)
            print('Moving mouse')
        except pyautogui.FailSafeException:
            print("PyAutoGUI failsafe triggered. Stopping test.")
            raise KeyboardInterrupt

    env = MouseEnvContinuous()

    for _ in range(episodes):
        obs = env.reset()
        done = False
        try:
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                move_real_mouse(env.mouse_pos[0], env.mouse_pos[1])
                # small sleep so we don't saturate GUI event loop
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("Test interrupted by user (Ctrl+C).")
            break

        # Render AFTER movement is done
        env.render(show=True)

        metrics = analyze_mouse_path(env.points)
        print("Robotic Detection Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    # Toggle these flags
    TRAIN = False
    TEST = True

    model = None
    if TRAIN:
        model = train_only(total_timesteps=250000)
    if TEST:
        # If you didn't just train, load your model here instead of using the returned one.
        if model is None:
            model = PPO.load("mouse_model.zip")
        test_with_real_mouse(model, episodes=1)