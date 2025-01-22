import numpy as np


class ReplayBuffer:
    def __init__(self,buffer_size,nA,nS):
        self.buffer_size = buffer_size
        self.index = 0
        self.buffer_full = False
        self.buffer_s = np.zeros((buffer_size,nS))
        self.buffer_s_ = np.zeros((buffer_size,nS))
        self.buffer_a = np.zeros((buffer_size,1))
        self.buffer_r = np.zeros((buffer_size,1))
        self.buffer_t = np.zeros((buffer_size,1))
       

    def add(self,s,a,r,s_,t):
        s = np.array(s)
        a = np.array(a)
        r = np.array(r)
        s_ = np.array(s_)
        t = np.array(t)
        self.buffer_s[self.index]=s
        self.buffer_s_[self.index]=s_
        self.buffer_a[self.index]=a
        self.buffer_r[self.index]=r
        self.buffer_t[self.index]=t

        self.index+=1
        if self.index==self.buffer_size:
            self.index=0
            self.buffer_full=True
    
    def batch(self,batch_size=64):
        indices = np.random.choice(self.buffer_size, size=batch_size, replace=False)
        b_s =  np.array([self.buffer_s[i] for i in indices])
        b_s_ = np.array([self.buffer_s_[i] for i in indices])
        b_a = np.array([self.buffer_a[i] for i in indices])
        b_r = np.array([self.buffer_r[i] for i in indices])
        b_t = np.array([self.buffer_t[i] for i in indices])
        

        return b_s,b_a,b_s_,b_r,b_t
       
        