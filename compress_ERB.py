import pickle
from expreplay import ExpReplay 
from expreplay import ReplayMemory
import cv2
import os
import numpy as np
import shutil
from sklearn import preprocessing
#./google-cloud-sdk/bin/gcloud compute scp --recurse /raid/home/thomaszhen/Distributed_System/Agent1 marine_shell_00001:
direct='/raid/home/thomaszheng/compression/'

def run_train(ROOT,FILES,LOG_DIR,ERB_PATH=None,PARAM_FILE=None):

    s='/home/thomaszheng/Desktop/TestRun/MARINE-SHELL-main/DQN.py --gpu 1 --task train --files '+FILES+' --logDir '+LOG_DIR+' --lr 0.001 --max_epochs '+str(1)+' --agents 1 '
    if ERB_PATH is not None:
        s=s+'--prev_exp '+ERB_PATH+' '
    if PARAM_FILE is not None:
        s=s+'--load '+PARAM_FILE
    os.system(s)
    return
def unpack_ERB(f,rate=4):
    with (open(f, "rb")) as openfile:
        obj=pickle.load(openfile)
        
    max_size = obj.max_size
    state_shape = obj.state_shape
    history_len = obj.history_len
    agents=obj.agents

    state=obj.state
    action=obj.action
    reward=obj.reward
    isOver=obj.isOver

    _curr_pos=obj._curr_pos
    _curr_size = obj._curr_size
    _hist = obj._hist
    weights=obj.weights
    print(reward.shape)
    
    new_state=[]
    new_action=[]
    new_reward=[]
    for i in range(reward.shape[0]):
        mask=[]
        for index,weight in enumerate(weights[i]):
            mask+=[index for j in range(weight)]
        #print(mask)
        new_size=len(mask)
        new_state.append(state[i,mask,:,:,:])
        new_action.append(action[i,mask])
        new_reward.append(reward[i,mask])
    new_state=np.array(new_state)
    new_action=np.array(new_action)
    new_reward=np.array(new_reward)
    obj.max_size=new_size
    print(new_state.shape)
    obj.isOver=isOver[0:new_size]
    obj._curr_pos=0
    obj._curr_size=new_size
    obj._hist.clear()
    obj.state=new_state
    obj.reward=new_reward
    obj.action=new_action

    temp=f.split('.obj')[0]
    temp+='_unpacked.obj'
    filehandler = open(temp,"wb")
    pickle.dump(obj,filehandler)
    filehandler.close()
    print('output ERB')
    return True

    
def k_means_weight(f,rate):
    with (open(f, "rb")) as openfile:
        obj=pickle.load(openfile)
        
    max_size = obj.max_size
    print(max_size)
    state_shape = obj.state_shape
    history_len = obj.history_len
    agents=obj.agents

    state=obj.state
    action=obj.action
    reward=obj.reward
    isOver=obj.isOver
    print(state.shape)

    _curr_pos=obj._curr_pos
    _curr_size = obj._curr_size
    _hist = obj._hist
    
    
    from sklearn.cluster import KMeans
    print('begin cluster')
    new_state=[]
    new_action=[]
    new_reward=[]
    all_weights=[]
    for i in range(reward.shape[0]):
        data=reward[i].reshape(-1,1)
        kmeans = KMeans(n_clusters=int(max_size/rate), n_init=1, random_state=0,verbose=0).fit(data)
        centers = np.array(kmeans.cluster_centers_)
        from collections import Counter, defaultdict
        c=Counter(kmeans.labels_)
        weights=np.array(list(c.values()))
        #print(weights)
        all_weights.append(weights)
        #print('complete cluster')
        from sklearn.metrics import pairwise_distances_argmin_min
        print(kmeans.cluster_centers_.shape)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)
   
        mask=closest
        #print(mask)
        if True:
            new_state.append(state[i,mask,:,:,:])
            new_action.append(action[i,mask])
            new_reward.append(reward[i,mask])
    new_state=np.array(new_state)
    new_action=np.array(new_action)
    new_reward=np.array(new_reward)
    #print(new_reward.shape)
    obj.max_size=int(max_size/rate)
    obj.isOver=isOver[0:int(max_size/5)]
    obj._curr_pos=0
    obj._curr_size=int(max_size/rate)
    obj._hist.clear()
    obj.state=new_state
    obj.reward=new_reward
    obj.action=new_action
    obj.weights=np.array(all_weights)

    temp=f.split('.obj')[0]
    temp+='_weight.obj'
    filehandler = open(temp,"wb")
    pickle.dump(obj,filehandler)
    filehandler.close()
    print('output ERB')
    return True

import numpy as np
import scipy.interpolate as interpolate
def inverse_transform_sampling(data, n_bins=1, n_samples=1000):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = np.random.rand(n_samples)
    return inv_cdf(r)
    
def inv_cdf(f,rate):
    with (open(f, "rb")) as openfile:
        obj=pickle.load(openfile)
        
    max_size = obj.max_size
    print(max_size)
    state_shape = obj.state_shape
    history_len = obj.history_len
    agents=obj.agents

    state=obj.state
    action=obj.action
    reward=obj.reward
    isOver=obj.isOver
    print(state.shape)

    _curr_pos=obj._curr_pos
    _curr_size = obj._curr_size
    _hist = obj._hist
    
    
    from sklearn.cluster import KMeans
    print('begin sampling')
    new_state=[]
    new_action=[]
    new_reward=[]
    all_weights=[]
    for i in range(reward.shape[0]):
        data=reward[i].reshape(-1,1)
        iv=inverse_transform_sampling(data,1,int(max_size/rate))
        iv=np.array([iv]).T
        from sklearn.metrics import pairwise_distances_argmin_min
        closest, _ = pairwise_distances_argmin_min(iv, data)
   
        mask=closest
        #print(mask)
        if True:
            new_state.append(state[i,mask,:,:,:])
            new_action.append(action[i,mask])
            new_reward.append(reward[i,mask])
    new_state=np.array(new_state)
    new_action=np.array(new_action)
    new_reward=np.array(new_reward)
    #print(new_reward.shape)
    obj.max_size=int(max_size/rate)
    obj.isOver=isOver[0:int(max_size/5)]
    obj._curr_pos=0
    obj._curr_size=int(max_size/rate)
    obj._hist.clear()
    obj.state=new_state
    obj.reward=new_reward
    obj.action=new_action

    temp=f.split('.obj')[0]
    temp+='_inv.obj'
    filehandler = open(temp,"wb")
    pickle.dump(obj,filehandler)
    filehandler.close()
    print('output ERB')
    return True


if __name__ == "__main__":
    for i in range(1,2):
        k_means_weight('/raid/home/thomaszheng/Full_body/AgentMcore/train_previous/MoreRound_5task0/dev/Experience_Replay_Buffer_weight.obj' ,4)


