import sys
sys.path.append('../d4rl-atari')
import gym
import d4rl_atari
import torch
from torch.utils import data
import numpy as np


# the resulting sequences are overlapping
class Atari(data.Dataset):
    def __init__(self,
                 transform = None,
                 seq_len = 4,
                 num_seq = 6,
                 downsample = 3,
                 epsilon = 5,
                 dataset = None,
                 stack = True,
                 n_channels = 5,
                 return_actions = False,
                 overlapping = True):
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.downsample = downsample
        self.epsilon = epsilon 

        # if the returned frames are not overlapping, we must return actions
        
        
        self.overlapping = overlapping
        self.return_actions = return_actions

        # create the env
        assert dataset is not None

        # get the dataset
        self.dataset = dataset
        
        self.observations = dataset['observations']
        self.terminals = dataset['terminals']
        self.actions = dataset['actions']
        
        traj_start = [0]
        self.terminals = self.dataset['terminals']
        for i, done in enumerate(self.terminals):
            if done and i != len(self.terminals) - 1:
                traj_start.append(i + 1)


        print(f'total trajactories : {len(traj_start)}')
        self.num_frames = len(self.terminals)

        # we still need to discard too short trajs
        if self.overlapping:
            self.frames_needed = self.seq_len
        else:
            self.frames_needed = (self.num_seq - 1) * self.seq_len + 1
        
        # filter out too short trajs
        self.starting_point = []
        for i in range(len(traj_start) - 1):
            start_cur = traj_start[i]
            end_cur = traj_start[i + 1] - 1
            if end_cur - self.frames_needed >= start_cur:
                self.starting_point.append([start_cur, end_cur]) 
        

        print(f'total feasible trajactories : {len(self.starting_point)}')
        print(f'filter out {len(traj_start) - len(self.starting_point)} short trajactories')
        # each time we pick a traj, and sample 8 seqs from it
        
    def __len__(self):
        # drop the last traj
        return len(self.starting_point) - 1
    

    # the behaviour of the idx_sampler decides if the frames are overlapping or not
    def idx_sampler(self, start, end):
        # return shape (num_seq, seq_len, H, W)
        
        # how many frames needed?
        '''
        chang here if we want to change overlapping/non-overlapping

        Note that each frames in the dataset is of (4, 84, 84), which means they are already stacked
        '''
        if self.overlapping:
            #print('overlapping branch')
            
            frames_needed = self.num_seq
            #print(f'what we need is : num_seq == {self.num_seq} and seq_len = {self.seq_len}')
            try:
                seq_start = np.random.choice(range(start, end - frames_needed))
            except:
                print(start, end-frames_needed)
            frames = torch.Tensor(self.observations[seq_start:seq_start+self.num_seq])
            (num_seq, seq_len, H, W) = frames.size()
            #print(num_seq, seq_len)
            frames = frames.contiguous().view(num_seq, 1, seq_len, H, W)
            
            
            # here the action is just integers (because the frames are overlapped)
            actions = torch.Tensor(self.actions[seq_start:seq_start+self.num_seq])
            #print(f'what we get is : shape = {frames.shape}')

        else :
            
            #print('non-overlapping branch')
            frames_needed = (self.num_seq - 1) * self.seq_len + 1
            # seq_start = np.random.choice(range(start, end-frames_needed))
            try:
                seq_start = np.random.choice(range(start, end - frames_needed))
            except:
                print(start, end, frames_needed,)
            #seq_start = 0
            # pick non-overlapped frames
            frames = torch.Tensor(self.observations[seq_start:seq_start+frames_needed:self.seq_len])
            indices = torch.IntTensor([np.arange(i-3, i+1) for i in np.arange(seq_start, seq_start+frames_needed, self.seq_len)])
            #print("indices", np.arange(seq_start, seq_start+frames_needed, self.seq_len))
            
            # need to check if actions are making problems : when the game starts, there are no enough frames 
            # indices = torch.clip(indices, min=0, max=self.num_frames-1)
            # non_sense_action = indices < 0
            
            #print(indices)
            # need to return all actions
            actions = torch.Tensor(self.actions[indices])
            # give actins that does not make sense a 19
            # actions[non_sense_action] = 19
            
#             actions = torch.Tensor(self.actions[seq_start:seq_start+self.num_seq:self.seq_len])
            (num_seq, seq_len, H, W) = frames.size()
            frames = frames.contiguous().view(num_seq, 1, seq_len, H, W)

            #print(frames.shape)
            #print(actions.shape)


        if self.return_actions:
            return frames, actions

        else:
            return frames
    
    def __getitem__(self, index):
        # eligible start frame of the first seq
        
        start = self.starting_point[index][0]
        
        # eligible end frame of the last seq
        end = self.starting_point[index][1]
        
        return self.idx_sampler(start, end)       