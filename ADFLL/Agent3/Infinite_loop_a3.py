import os
import pickle
import logging
import glob
import shutil
import sys    
import datetime
import numpy as np
from csv import writer

file_name =  os.path.basename(sys.argv[0])
DQN_PATH='/home/thomaszheng/Desktop/TestRun/MARINE-SHELL-main/DQN.py'
total_agents=4
agent=file_name.split('.')[0][-1]
hub=str(1)
hub_dir='/raid/home/thomaszheng/Distributed_System/Agent1/receive'
print(agent)
epoch=4
iterations=str(epoch*25000)

#parameters for running DQN.py
cwd = os.getcwd()
ROOT=r"{}".format(cwd+"/train/")
FILES=ROOT+"train_image_paths.txt "+ROOT+"train_label_paths.txt"
LOG_DIR=ROOT+"Agent"+agent+"_logs/"
PARAM_FILE=LOG_DIR+"dev/model-"+iterations


#function call DQN.py with parameters
def run_train(ROOT,FILES,LOG_DIR,ERB_PATH=None,PARAM_FILE=None):
	#TODO: CHANGE THIS
	s=DQN_PATH+' --gpu '+str(0)+' --task train --files '+FILES+' --logDir '+LOG_DIR+' --lr 0.001 --max_epochs '+str(epoch)+' --agents 1 '
	if ERB_PATH is not None:
		s=s+'--prev_exp '+ERB_PATH+' '
	if PARAM_FILE is not None:
		s=s+'--load '+PARAM_FILE
	os.system(s)
	return
	
#get pathology and modality:
def get_path_mod():
	fline=open(ROOT+"train_image_paths.txt").readline().rstrip()
	mod=fline.split('.')[0]
	mod=mod.split('_')[-1]
	if 'LGG' in fline:
		path='LGG'
	else:
		path='HGG'
	return mod,path
	
def check_erb(agent_num):
	files=glob.glob(hub_dir+'/*.obj')
	files=[i.split('/')[-1] for i in files]
	print(files)
	if len(files) == 0:

		return False
	else:

		#make directory to store already used ERB files
		isExist = os.path.exists(root+'Used_ERB/Files')
		if not isExist:
			os.makedirs(root+'Used_ERB/Files')
	
		#move all already used ERB files to another folder
		Used_files = glob.glob(root+'Used_ERB/*.obj')
		for i in Used_files:
			shutil.move(i,root+'Used_ERB/Files')
			
		#get the files that haven't been used from receive folder
		Used_files=[i.split('/')[-1] for i in Used_files]
		files_to_use=list(set(files).difference(set(Used_files)))
		files_to_use=[hub_dir+'/'+i for i in files_to_use if ('Agent'+agent_num) not in i]
		#move these files to Used_ERB
		
		if len(files_to_use) >0:
			for i in files_to_use:
				shutil.copy(i,root+'Used_ERB')
			return True
		else:
			return False
			

#some good to have global variables
root=r"{}".format(cwd+'/')
train_dir_files=os.listdir(root+'train')
session=0
need_new_session=False
file_transfer_done=False
iterations_done=int(iterations)

#create log
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=root+'Agent'+agent+'_log.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logging.info('Infinite loop start')

#create csv file
col_names=['ERB_id','Modality','Landmark','Pathology','Agents_with_this_ERB','time_started']
update=[]

#run basic train for the first time
first_time=True
if first_time:
	run_train(ROOT,FILES,LOG_DIR)
else:
	ERB=glob.glob(root+'Used_ERB/Files/Experience_Replay_Buffer_Agent'+agent+'*')
	ERB=[int(i.split('.')[0][-1]) for i in ERB]
	session=max(ERB)


while True:
	#save computing power
	import time
	time.sleep(3)
	
	#if a training session is done (used model-iteration to check)
	model_path=r"{}".format(root+'train/Agent'+agent+'_logs/dev/model-'+str(iterations_done)+'.index')
	if os.path.exists(model_path):
		obj_path=r"{}".format(root+'train/Agent'+agent+'_logs/dev/Experience_Replay_Buffer.obj')
		#as soon as the session is done, send files to other agents (check using ERB file, move it to Used_ERB so this if statement does not repeat)
		if os.path.exists(obj_path):
			session=session+1
			logging.info('session '+str(session)+' ended')
			
			#remove old ERB
			old_ERB = [files for files in os.listdir(root+'Used_ERB') if os.path.isfile(os.path.join(root+'Used_ERB', files))]
			for i in old_ERB:
				os.remove(root+'Used_ERB/'+i)
			
			#update csv
			if session==1:
				mod,path=get_path_mod()
				update=[root+'Used_ERB/Experience_Replay_Buffer_Agent'+agent+'_'+str(session)+'.obj',mod,'Right_Top_Ventricle',path,[1,2,3,4],datetime.datetime.now()]
				np.savetxt('Agent'+agent+'.csv',[p for p in zip(col_names, update)], delimiter=',', fmt='%s')
			else:
				mod,path=get_path_mod()
				update=[root+'Used_ERB/Experience_Replay_Buffer_Agent'+agent+'_'+str(session)+'.obj',mod,'Right_Top_Ventricle',path,[1,2,3,4],datetime.datetime.now()]
				with open('Agent'+agent+'.csv', 'a') as f_object:
					writer_object = writer(f_object)
					writer_object.writerow(update)
					f_object.close()
			
			shutil.move(obj_path,update[0])
			
			#TODO: CHANGE THIS
			parent_dir=os.path.dirname(root)
			parent_dir=os.path.dirname(parent_dir)
			shutil.copy(update[0], parent_dir+'/Agent'+hub+'/receive/')
			
			
			
		#TODO: CHANGE THIS
        #When there is ERB coming from other agents
        
		if check_erb(agent):

			
			
			#When there are new images in the incoming_images folder
			if len(os.listdir(root+'incoming_images'))!=0:
				logging.info('new session can be started')
				
				#TODO: CHANGE THIS
				#move received ERB to Used_ERB folder
				erb_files=glob.glob(root+'Used_ERB/Experience*.obj')
				str_erb=' '.join(erb_files)
				#update parameters for DQN.py
				ERB_PATH=root+'Used_ERB/Files/Experience_Replay_Buffer_Agent'+agent+'_'+str(session)+'.obj'+" "+str_erb
				
				#move model param file to Used_ERB
				model_param=glob.glob(root+'train/Agent'+agent+'_logs/dev/model-'+str(iterations_done)+'*')
				for files in model_param:
					shutil.copy(files,root+'Used_ERB/Files')
				PARAM_FILE=root+'Used_ERB/Files/model-'+str(iterations_done)
				need_new_session=True
				
				#empty train folder
				old_train_files=glob.glob(root+'train/train_*.txt')
				for i in old_train_files:
					os.remove(i)
			
				#move new images to train folder
				shutil.move(root+'incoming_images/train_image_paths.txt',root+'train/train_image_paths.txt')
				shutil.move(root+'incoming_images/train_label_paths.txt',root+'train/train_label_paths.txt')
		
		if need_new_session:
			#train again with new information
			#delete files in the logs folder for future use
			files = glob.glob(root+'train/Agent'+agent+'_logs/dev/*') #VP# I am assuming this deletes all the files in the dev folder - ERBs and trained models. (Good)
			for f in files:
				os.remove(f)
			shutil.rmtree(root+'train/Agent'+agent+'_logs/dev')
			shutil.rmtree(root+'train/Agent'+agent+'_logs')
			logging.info('new session started')
			need_new_session=False
			iterations_done=iterations_done+int(iterations)
			run_train(ROOT,FILES,LOG_DIR,ERB_PATH,PARAM_FILE) 

