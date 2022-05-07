import random
from collections import deque
import gym
import gym_super_mario_bros
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from gym.spaces import Box
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from torch import nn
from torchvision import transforms

class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=1, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)

def create_wrap_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrames(env, 4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.layers_outsize = self._get_conv_out(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.layers_outsize, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        return self.fc(self.layers(input))

    def _get_conv_out(self, shape):
        out = self.layers(torch.zeros(1, *shape[:3]))
        return int(np.prod(out.size()))


class Agent:
    def __init__(self,
                 action_dim,
                 observation_dim,
                 save_directory,
                 continue_training,
                 checkpoint_period):

        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.save_directory = save_directory
        self.checkpoint_period = checkpoint_period
        self.exploration_rate_decay = 0.99
        self.exploration_rate_min = 0.03
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 0.0002
        self.loss = torch.nn.MSELoss()
        self.current_episode_reward = 0.0

        if not continue_training:
            self.model = Model(self.observation_dim, self.action_dim).cuda()
            self.exploration_rate = 1.0
            self.current_step = 0
            self.memory = deque(maxlen=30000)
            self.episode = 0
            self.episode_rewards = []
            self.average_episode_rewards = []

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # todo cpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def log_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def log_period(self, episode, epsilon, step):
        self.average_episode_rewards.append(np.round(np.mean(self.episode_rewards[-self.checkpoint_period:]), 3))
        print(f"Episode {episode} - Step {step} - Epsilon {epsilon} - Mean Reward {self.average_episode_rewards[-1]}")
        self.draw_graph()

    def draw_graph(self):
        plt.title("Training graph")
        plt.xlabel("Episodes (x10)")
        plt.ylabel("Average Reward")
        plt.plot(self.average_episode_rewards)
        plt.savefig(self.save_directory + "/graphs/graph.png")
        plt.clf()

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((torch.tensor(state.__array__()), torch.tensor(next_state.__array__()),
                            torch.tensor([action]), torch.tensor([reward]), torch.tensor([done])))

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward

        if self.batch_size > len(self.memory):
            return

        state, next_state, action, reward, done = self.recall()
        q_estimate = self.model(state.cuda())[np.arange(0, self.batch_size), action.cuda()]
        with torch.no_grad():
            best_action = torch.argmax(self.model(next_state.cuda()), dim=1)
            next_q = self.model(next_state.cuda())[np.arange(0, self.batch_size), best_action]
            q_target = (reward.cuda() + (1 - done.cuda().float()) * self.gamma * next_q).float()
        loss = self.loss(q_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()  # gradients computation
        self.optimizer.step()  # error backpropagation

    def recall(self):
        state, next_state, action, reward, done = map(torch.stack, zip(*random.sample(self.memory, self.batch_size)))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.model(torch.tensor(state.__array__()).cuda().unsqueeze(0))
            action = torch.argmax(action_values, dim=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1
        return action

    def save_checkpoint(self):
        with open(self.save_directory + "/data.dat", "wb") as f:
            pickle.dump(self, f)


    def next_episode(self):
        self.episode += 1


def run(continue_training):
    checkpoint_period = 10
    save_directory = "data"
    env = create_wrap_env()

    if continue_training:
        with open(save_directory + "/data.dat", "rb") as f:
            agent = pickle.load(f)
            agent.action_dim = env.action_space.n
            agent.observation_dim = env.observation_space.shape
    else:
        agent = Agent(action_dim=env.action_space.n,
                      observation_dim=env.observation_space.shape,
                      save_directory=save_directory,
                      continue_training=continue_training,
                      checkpoint_period=checkpoint_period)
    while True:
#    for i in range(1000):
        state = env.reset()
        while True:
            action = agent.act(state)
            env.render()
            next_state, reward, done, info = env.step(action)
            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay(reward)
            state = next_state
            if done:
                agent.next_episode()
                agent.log_episode()
                if agent.episode % 100 == 0:
                    agent.save_checkpoint()
                if agent.episode % checkpoint_period == 0:
                    agent.log_period(episode=agent.episode, epsilon=agent.exploration_rate, step=agent.current_step)
                break


def main():
    run(continue_training=False)


if __name__ == "__main__":
    main()
