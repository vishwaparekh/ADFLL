import os
import pickle
import logging
import glob
import shutil
import sys    
import datetime
import numpy as np
from csv import writer
import sys
sys.path.insert(1, '/home/thomaszheng/Desktop/TestRun/MARINE-SHELL-main/')
import compress_ERB
compression='1'
agent='Mcore'

epoch=4
iterations=str(epoch*25000)

train_sets = [dI for dI in os.listdir('/raid/home/thomaszheng/Distributed_System/train_sets') if os.path.isdir(os.path.join('/raid/home/thomaszheng/Distributed_System/train_sets',dI))]
train_sets = ['/raid/home/thomaszheng/Distributed_System/train_sets/'+i for i in train_sets]
print(len(train_sets))
import random
random.Random(4).shuffle(train_sets)
train_sets=train_sets[:10]
session=0

#parameters for running DQN.py
cwd = os.getcwd()
ROOT=r"{}".format(cwd+"/train/")
FILES=train_sets[0]+"/train_image_paths.txt "+train_sets[0]+"/train_label_paths.txt"
LOG_DIR=ROOT+compression+"xRound"+str(session)+"_logs/"
PARAM_FILE=LOG_DIR+"dev/model-"+iterations


#function call DQN.py with parameters
def run_train(ROOT,FILES,LOG_DIR,ERB_PATH=None,PARAM_FILE=None):

	s='/home/thomaszheng/Desktop/TestRun/MARINE-SHELL-main/DQN.py --gpu 3 --task train --files '+FILES+' --logDir '+LOG_DIR+' --lr 0.001 --max_epochs '+str(epoch)+' --agents 1 '
	if ERB_PATH is not None:
		s=s+'--prev_exp '+ERB_PATH+' '
	if PARAM_FILE is not None:
		s=s+'--load '+PARAM_FILE
	os.system(s)
	return
	

#some good to have global variables
root=r"{}".format(cwd+'/')
train_dir_files=os.listdir(root+'train')

#run basic train for the first time
first_time=True
if first_time:
	run_train(ROOT,FILES,LOG_DIR)
first_time=False


while True:
	#save computing power
	import time
	time.sleep(3)
	
	#if a training session is done (used model-iteration to check)
	model_path=r"{}".format(LOG_DIR+'dev/model-'+str((session+1)*int(iterations))+'.index')
	print(model_path)
	if os.path.exists(model_path):
		obj_path=one+'/Experience_Replay_Buffer.obj'
		#compress_ERB.k_means_weight(obj_path,10)
		temp=obj_path.split('.obj')[0]
		temp+='_weight.obj'
		#compress_ERB.unpack_ERB(temp,10)
		temp=temp.split('.obj')[0]
		temp+='_unpacked.obj'
		temp=obj_path

		#update parameters for DQN.py
		ERB_PATH=[ROOT+compression+'xRound'+str(i)+'_logs/dev/Experience_Replay_Buffer_sampling.obj' for i in range(0,session)]
		ERB_PATH=' '.join(ERB_PATH)
		#move model param file to Used_ERB
		PARAM_FILE=model_path

				
		FILES=train_sets[session]+"/train_image_paths.txt "+train_sets[session]+"/train_label_paths.txt"
		
		#train again with new information
		LOG_DIR=ROOT+compression+"xRound"+str(session)+"_logs/"
		run_train(ROOT,FILES,LOG_DIR,ERB_PATH,PARAM_FILE) 

