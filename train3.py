import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
from robotic_detection_algo import analyze_mouse_path
import pyautogui  # now imported here so we can get screen size

# Get screen dimensions
SCREEN_W, SCREEN_H = pyautogui.size()

class MouseEnvContinuous(gym.Env):
    def __init__(self, step_size=50, max_steps=200):  # step_size in pixels
        super(MouseEnvContinuous, self).__init__()
        self.mouse_pos = None
        self.target = None
        self.step_size = step_size
        self.max_steps = max_steps
        self.steps = 0
        self.path = []
        self.start_time = None
        self.points = []

        # Action: dx, dy, pause_flag
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # Observation: mouse_x, mouse_y, target_x, target_y, pause_flag
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([SCREEN_W, SCREEN_H, SCREEN_W, SCREEN_H, 1], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self):
        self.mouse_pos = np.random.uniform(low=[100, 100], high=[SCREEN_W-100, SCREEN_H-100])
        self.target = np.random.uniform(low=[0, 0], high=[SCREEN_W, SCREEN_H])
        self.steps = 0
        self.path = [self.mouse_pos.copy()]
        self.points = []
        self.start_time = time.time()
        self.points.append((0, self.mouse_pos[0], self.mouse_pos[1]))
        return np.concatenate([self.mouse_pos, self.target, [0.0]])

    def step(self, action):
        dx, dy, pause_flag_raw = action
        pause_flag = 1 if pause_flag_raw >= 0 else 0

        noise = np.random.normal(0, 5, size=2)  # noise in pixels
        noisy_dxdy = np.clip(np.array([dx, dy]) + noise / self.step_size, -1, 1)

        if pause_flag == 1:
            move = np.array([0.0, 0.0])
        else:
            move = noisy_dxdy * self.step_size

        self.mouse_pos += move
        self.mouse_pos = np.clip(self.mouse_pos, [0, 0], [SCREEN_W, SCREEN_H])
        self.path.append(self.mouse_pos.copy())

        dist = np.linalg.norm(self.mouse_pos - self.target)
        reward = -dist

        self.steps += 1
        done = False

        current_time_ms = int((time.time() - self.start_time) * 1000)
        self.points.append((current_time_ms, self.mouse_pos[0], self.mouse_pos[1]))

        if dist < 20 or self.steps >= self.max_steps:  # 20 pixels = success radius
            done = True

        if done:
            scores = analyze_mouse_path(self.points)
            robotic_score = scores.get("robotic_score", 0)
            jitter = scores.get("jitter_magnitude", 0)
            pause = scores.get("pause_count", 0)

            reward -= robotic_score * 7
            if 2 < jitter < 20:  # jitter in pixels
                reward += 5
            if 1 <= pause <= 3:
                reward += 5
            if dist < 20:
                reward += 10
                if pause_flag == 1:
                    reward += 5

        obs = np.concatenate([self.mouse_pos, self.target, [float(pause_flag)]])
        return obs, reward, done, {}

    def render(self, show=True):
        plt.figure(figsize=(10, 8))
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], marker="o")
        plt.scatter(*self.target, label="Target", c="red")
        plt.title("Mouse path")
        plt.xlim(0, SCREEN_W)
        plt.ylim(0, SCREEN_H)
        plt.gca().invert_yaxis()  # y-axis flipped for screen coords
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
    model.save("mouse_model_screen.zip")
    plt.figure()
    plt.plot(callback.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Training Convergence")
    plt.show()
    return model


def test_with_real_mouse(model, episodes=1):
    import pyautogui
    pyautogui.PAUSE = 0.01
        
    def smooth_move_to(x_target, y_target, duration=0.3, steps=40):
        x_start, y_start = pyautogui.position()
        dx = (x_target - x_start) / steps
        dy = (y_target - y_start) / steps
        for i in range(steps):
            pyautogui.moveTo(x_start + dx * (i + 1), y_start + dy * (i + 1))
            time.sleep(duration / steps)

    def move_real_mouse(x, y):
        try:
            smooth_move_to(int(x), int(y), duration=0.3, steps=40)
            # time.sleep(1)  # short gap so you can see motion
        except pyautogui.FailSafeException:
            print("PyAutoGUI failsafe triggered.")
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
        except KeyboardInterrupt:
            print("Test interrupted.")
            break

        env.render(show=True)
        metrics = analyze_mouse_path(env.points)
        print("Robotic Detection Metrics:")
        for key, val in metrics.items():
            print(f"{key}: {val}")


if __name__ == "__main__":
    TRAIN = False
    TEST = True

    model = None
    if TRAIN:
        model = train_only(total_timesteps=250000)
    if TEST:
        if model is None:
            model = PPO.load("mouse_model_screen.zip")
        test_with_real_mouse(model, episodes=1)