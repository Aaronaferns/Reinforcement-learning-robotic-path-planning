import numpy as np

class Buffer:
    def __init__(self,nS,nA,buffer_size=50000,useHindsight=False):
        self.useHindsight=useHindsight
        self.buffer_size=buffer_size
        self.g=np.empty((buffer_size,nS))
        self.s=np.empty((buffer_size,nS))
        self.a=np.empty((buffer_size,nA))
        self.s_=np.empty((buffer_size,nS))
        self.r = np.empty((buffer_size,1))
        self.done = np.empty((buffer_size,1))
        
        self.R=np.empty((buffer_size,nS))
        self.O = np.empty((buffer_size,nS+1))
        self.size = 0
        self.idx = -1
        self.idx_o=-1
        self.size_o=0
        self.idx_r=-1
        self.size_r=0
        self.full=False
        self.k=2
    
    def add(self,trajectory_memory):
        # self.idx+=1
        # if self.idx==self.buffer_size-1: self.full=True
        # if self.idx>=self.buffer_size:
        #     self.idx=0
        # if self.size!=self.buffer_size:
        #     self.size+=1
        # self.buff[self.idx,:]=np.array(data)
        # if self.useHindsight:
        #     self.idx+=1
        #     if self.idx==self.buffer_size-1: self.full=True
        #     if self.idx>=self.buffer_size:
        #         self.idx=0
        #     if self.size!=self.buffer_size:
        #         self.size+=1
        #     dg,s,a,r,s_=data
            
            
        #     self.buff[self.idx,:]
        
        tm=trajectory_memory #its a numpy array where (tlen,g,s,a,r,s_)
        
        len_tm=len(tm)
        for t in range(len_tm):
            print(t)
            self.idx+=1
            if self.idx==self.buffer_size-1: self.full=True
            if self.idx>=self.buffer_size:
                self.idx=0
            if self.size!=self.buffer_size:
                self.size+=1
            g,s,a,r,s_,done=tm[t]
            self.s[self.idx]=s
            self.s_[self.idx]=s_
            self.a[self.idx]=a
            self.r[self.idx]=r
            self.g[self.idx]=g
            self.done[self.idx]=done

            if self.useHindsight:
                
                #use future strategy
                future_range=np.arange(t+1,len_tm)
                if len(future_range) == 0:
                    continue
                future_idxs = np.random.choice(future_range,size=self.k,replace=True)
                for f_idx in future_idxs:
                    self.idx+=1
                    if self.idx==self.buffer_size-1: self.full=True
                    if self.idx>=self.buffer_size:
                        self.idx=0
                    if self.size!=self.buffer_size:
                        self.size+=1
                    _,dg,_,_,_=tm[f_idx]
                    self.s[self.idx]=s
                    self.s_[self.idx]=s_
                    self.a[self.idx]=a
                    if np.array_equal(dg,s_): new_r=0
                    else: new_r = -1
                    self.r[self.idx]=new_r
                    self.g[self.idx]=dg
                    self.done[self.idx]=done
            
        self.add_R(s)
    def get_batch(self,batch_size):
        indices = np.random.choice(np.arange(0,self.size),size=batch_size,replace=True)
        return self.g[indices],self.s[indices],self.a[indices],self.r[indices],self.s_[indices],self.done[indices]
        
    
    def add_O(self,outcome):
        self.idx_o+=1
        if self.idx_o>=self.buffer_size:
            self.idx_o=0
        if self.size_o!=self.buffer_size:
            self.size_o+=1
        self.O[self.idx_o,:-1]=outcome[0]
        self.O[self.idx_o,-1]=outcome[1]
    def get_batch_o(self,batch_size):
        success = self.O[:self.size_o, -1].astype(int)
        success_idx = np.where(success == 1)[0]
        failure_idx = np.where(success == 0)[0]
        half = batch_size // 2
        sampled_success = np.random.choice(success_idx, size=half, replace=len(success_idx) < half)
        sampled_failure = np.random.choice(failure_idx, size=half, replace=len(failure_idx) < half)
        indices = np.concatenate([sampled_success, sampled_failure])
        np.random.shuffle(indices)
        return self.O[indices,:-1],self.O[indices,-1]
    
    def add_R(self,s):
        self.idx_r+=1
        if self.idx_r>=self.buffer_size:
            self.idx_r=0
        if self.size_r!=self.buffer_size:
            self.size_r+=1
        self.R[self.idx_r,:]=s
    def get_batch_r(self,batch_size):
        indices = np.random.choice(self.R[:self.size_r],size=self.size_r-1,replace=False)
        return self.R[indices]
        
       
        
def main():
    trajectory = [([1,2],[0,1],0,0,[0,2]),([1,2],[0,1],0,0,[0,55]) ,([1,2],[0,1],0,0,[0,2]),([1,2],[0,1],0,0,[0,5]) ,([1,2],[0,1],0,0,[0,88]) ]
    replay_buffer = Buffer(2,1,100,True)
    replay_buffer.add(trajectory)
    
    for i in range(replay_buffer.size):
        print(replay_buffer.s[i])
        print(replay_buffer.s_[i])
        print(replay_buffer.a[i])
        print(replay_buffer.r[i])
        print(replay_buffer.g[i])
    
    
if __name__=="__main__":
    main()
        
        