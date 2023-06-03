from gym.wrappers import Monitor
from IPython.display import HTML
from IPython import display as ipythondisplay
import uuid
import io
import glob
import base64
import matplotlib.pyplot as plt
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym.wrappers import FrameStack
from gym.wrappers import Monitor
#from gym.wrappers import Monitor
from IPython.display import HTML
from pyvirtualdisplay import Display
import numpy as np
import torch

def show_video():
    """
    Utility function to enable video recording of gym environment and displaying it
    To enable video, just do "env = wrap_env(env)""
    """
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
    else: 
        print("Could not find video")
    
     

def wrap_env(env):
    """
    Wrapper del ambiente donde definimos un Monitor que guarda la visualizacion como un archivo de video.
    """

    env = Monitor(env, './video', force=True)
    return env

   
def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.gcf()

def show_obs(obs):  
    for i in  range(0,4):
      plt.imshow(obs[i])
      plt.title("Observation " + str(i))
      plt.axis('off')

      #display.clear_output(wait=True)
      #display.display(plt.gcf())
      plt.show()


from torchvision import transforms as T

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class MyJoypadSpace(JoypadSpace):
    def __init__(self, env, actions):
      super().__init__(env, actions)

    def step(self, action):      
      next_state, reward, done, info = super().step(action)
      return next_state[:], reward, done, info

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


# Apply Wrappers to environment
def make_env(env_name):
  env = gym_super_mario_bros.make(env_name)

  env = SkipFrame(env, skip=4)
  env = GrayScaleObservation(env)
  env = ResizeObservation(env, shape=84)
  env = FrameStack(env, num_stack=4)
  # Limit the action-space to
  #   0. walk right
  #   1. jump right
  env = MyJoypadSpace(env, [["right"], ["right", "A"]])
  return env
