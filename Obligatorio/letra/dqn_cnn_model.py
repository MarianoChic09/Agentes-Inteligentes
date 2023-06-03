import torch.nn as nn
import torch.nn.functional as F

class DQN_CNN_Model(nn.Module):
    def __init__(self,  env_inputs, n_actions):
      super(DQN_CNN_Model, self).__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
      self.bn1 = nn.BatchNorm2d(16)
      self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
      self.bn2 = nn.BatchNorm2d(32)
      self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
      self.bn3 = nn.BatchNorm2d(32)

      def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
      
      w = env_inputs[1]
      h = env_inputs[2]

      convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
      convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
      linear_input_size = convw * convh * 32
      self.head = nn.Linear(linear_input_size, n_actions)


    def forward(self, env_input):
        env_input = F.relu(self.bn1(self.conv1(env_input)))
        env_input = F.relu(self.bn2(self.conv2(env_input)))
        env_input = F.relu(self.bn3(self.conv3(env_input)))
        return self.head(env_input.view(env_input.size(0), -1))