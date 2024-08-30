import gym
import random
import torch
import numpy as np
from collections import namedtuple, deque
from circuit_fl import Circuit_manager

class Agent:
    def __init__(self
                , qdqn
                , action_space
                , greedy
                , greedy_decay
                , greedy_min
                , structure_batch
                , cm:Circuit_manager
                , seed=1234):
        self.qdqn = qdqn
        self.action_space = action_space
        self.greedy = greedy
        self.greedy_decay = greedy_decay
        self.greedy_min = greedy_min
        self.structure_batch = structure_batch
        self.cm = cm
        random.seed(seed)
        np.random.seed(seed)

    def __call__(self, state, prob, device=torch.device('cpu')):
        if random.uniform(0, 1) > self.greedy:
            action = self.get_action(state, prob, device)
        else:
            action = self.get_random_action()
            # print(f'random action: {action}')

        return action

    def preset_byprob(self, prob):
        return np.array([np.random.choice(np.arange(prob.size()[1])
            , p=np.array(prob[i].detach().cpu().clone().numpy())) for i in range(prob.size()[0])])
    
    def get_random_action(self):
        return self.action_space.sample()

    def get_action(self, state, prob=[], device=torch.device('cpu')):
        # if not isinstance(state, torch.Tensor):
        #     state = torch.tensor(np.array([state]))
        # state = state.to(device)
        state = torch.from_numpy(np.array(state)).to(device)
        # print(f"state: {state}, state shape: {state.shape}")

        # activate eval model
        if len(prob)>0:
            qvs = []
            for _ in range(self.structure_batch):
                sample_struc = self.preset_byprob(prob)
                self.cm.set_current_sampled_struc(sample_struc) # set struc before circuit running
                # print(f"** -- sample_struc -- **: {sample_struc_ts}")
                # print(f"** -- inputs: {inputs}, shape: {inputs.shape} -- **")
                qvs.append(self.qdqn.eval()(state))
    #         print(f"qvs: {qvs}")
    #         qvs: [tensor([[44.4233, 51.4572]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[45.3093, 42.8838]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[78.1133, 77.9809]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[22.1962, 25.8107]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[53.7998, 32.3309]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[31.0641, 69.0260]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[74.1404, 
    # 89.8707]], dtype=torch.float64, grad_fn=<MulBackward0>), tensor([[84.3737, 78.0760]], dtype=torch.float64), tensor([[53.7998, 32.3309]], dtype=torch.float64, 
    # grad_fn=<MulBackward0>), tensor([[29.6943, 45.1197]], dtype=torch.float64, grad_fn=<MulBackward0>)]
            q_value = torch.mean(torch.stack(qvs), dim=0)
        else:
            q_value = self.qdqn.eval()(state)
        # print(f"q_values: {q_value}") # q_values: tensor([[51.6914, 54.4887]], dtype=torch.float64, grad_fn=<MeanBackward1>)
        _, action = torch.max(q_value, 0)
        # print(action)
        
        return int(action.item())

    def get_greedy(self):
        return self.greedy
    
    # TODO: add argument for step by decaying greedy 
    # e.g. greedy = max(greedy min, greedy min + (greedy init - greedy min)*(decay^step))
    def update_greedy(self):
        self.greedy = max(self.greedy_min, self.greedy*self.greedy_decay)
        return self.greedy

Record = namedtuple('Record', ('state', 'action', 'reward', 'new_state', 'done'))

class Memory:
    def __init__(self, memory_size=int(1e5), random_seed=1234):
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        random.seed(random_seed)

    def append(self, *args):
        self.memory.append(Record(*args))
    
    def sample(self, batch_size, device):
        minibatch = random.sample(self.memory, batch_size)

        states, actions, rewards, new_states, dones = zip(*minibatch) # * for destructure tuple
        states = torch.from_numpy(np.array(states)).to(device)
        actions = torch.from_numpy(np.array(actions)).to(device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(device)
        dones = torch.from_numpy(np.array(dones, dtype=np.int32)).to(device)
        new_states = torch.from_numpy(np.array(new_states)).to(device)

        return states, actions, rewards, new_states, dones

    def __len__(self):
        return len(self.memory)


class BinaryObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(BinaryObservation, self).__init__(env)
        self.bits = int(np.ceil(np.log2(env.observation_space.n)))
        self.observation_space = gym.spaces.MultiBinary(self.bits)

    def observation(self, obs):
        binary = map(float, f'{obs:0{self.bits}b}')
        return np.array(list(binary))