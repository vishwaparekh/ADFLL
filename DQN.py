#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Amir Alansary <amiralansary@gmail.com>
# Modified: Athanasios Vlontzos <athanasiosvlontzos@gmail.com>
# Modified: Alex Bocchieri <abocchi2@jhu.edu>
# Modified: Vishwa Parekh <vishwaparekh@gmail.com>
# Modified: Shuhao Lai <slai16@jhu.edu>

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)

import os
import argparse

import tensorflow as tf
from medical import MedicalPlayer, FrameStack
from tensorpack.input_source import QueueInput
from tensorpack_medical.models.conv3d import Conv3D
from common import Evaluator, play_n_episodes
from DQNModel import Model3D as DQNModel
from expreplay import ExpReplay
from ERBSaver import ERBSaver
import pickle
from tensorpack import (PredictConfig, OfflinePredictor, get_model_loader,
                        logger, TrainConfig, ModelSaver, PeriodicTrigger,
                        ScheduledHyperParamSetter, ObjAttrParam,
                        HumanHyperParamSetter, argscope, RunOp, FullyConnected, PReLU, SimpleTrainer,
                        launch_train_with_config)


###############################################################################
# BATCH SIZE USED IN NATURE PAPER IS 32 - MEDICAL IS 256
BATCH_SIZE = 48 #32
# BREAKOUT (84,84) - MEDICAL 2D (60,60) - MEDICAL 3D (26,26,26)
IMAGE_SIZE = (15, 15, 9)#(23,23,9)#(45, 45, 11)#(11, 11, 9)
# how many frames to keep
# in other words, how many observations the network can see
FRAME_HISTORY = 4
# the frequency of updating the target network
UPDATE_FREQ = 4
# DISCOUNT FACTOR - NATURE (0.99) - MEDICAL (0.9)
GAMMA = 0.9 #0.99
# REPLAY MEMORY SIZE - NATURE (1e6) - MEDICAL (1e5 view-patches)
MEMORY_SIZE = 4000#5#6   # to debug on bedale use 1e4
# consume at least 1e6 * 27 * 27 * 27 bytes
INIT_MEMORY_SIZE = MEMORY_SIZE // 20 #5e4
# each epoch is 100k played frames
# 10000 5000 // UPDATE_FREQ * 10
# 20000 10000 // UPDATE_FREQ * 10
# 16660 10000 // UPDATE_FREQ * 10 freq=6
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ * 10
# num training epochs in between model evaluations
EPOCHS_PER_EVAL = 2
# the number of episodes to run during evaluation
EVAL_EPISODE = 50

###############################################################################

def get_player(directory=None, files_list= None, viz=False,
               task='play', saveGif=False, saveVideo=False,agents=2,reward_strategy=1):
    # in atari paper, max_num_frames = 30000
    env = MedicalPlayer(directory=directory, screen_dims=IMAGE_SIZE,
                        viz=viz, saveGif=saveGif, saveVideo=saveVideo,
                        task=task, files_list=files_list,agents=agents, max_num_frames=1500,reward_strategy=reward_strategy)
    if (task != 'train'):
        # in training, env will be decorated by ExpReplay, and history
        # is taken care of in expreplay buffer
        # otherwise, FrameStack modifies self.step to save observations into a queue
        env = FrameStack(env, FRAME_HISTORY,agents=agents)
    return env

###############################################################################

class Model(DQNModel):
    def __init__(self, agents=2, lr=1e-4):
        super(Model, self).__init__(IMAGE_SIZE, FRAME_HISTORY, METHOD,
                                    NUM_ACTIONS, GAMMA, agents, lr)

    def _get_DQN_prediction(self, images):
        """ image: [0,255]

        :returns predicted Q values"""
        # normalize image values to [0, 1]

        agents = len(images)

        Q_list = []

        with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

            for i in range(0, agents):
                images[i] = images[i] / 255.0
                with argscope(Conv3D, nl=PReLU.symbolic_function, use_bias=True):

                    if i == 0:
                        conv_0 = tf.layers.conv3d(images[i], name='conv0',
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                        conv_1 = tf.layers.conv3d(max_pool_0, name='conv1',
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                        conv_2 = tf.layers.conv3d(max_pool1, name='conv2',
                                                  filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                        conv3 = tf.layers.conv3d(max_pool2, name='conv3',
                                                 filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same',
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                                                 bias_initializer=tf.zeros_initializer())
                    else:
                        conv_0 = tf.layers.conv3d(images[i], name='conv0', reuse=True,
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool_0 = tf.layers.max_pooling3d(conv_0, 2, 2, name='max_pool0')
                        conv_1 = tf.layers.conv3d(max_pool_0, name='conv1', reuse=True,
                                                  filters=32, kernel_size=[5, 5, 5], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool1 = tf.layers.max_pooling3d(conv_1, 2, 2, name='max_pool1')
                        conv_2 = tf.layers.conv3d(max_pool1, name='conv2', reuse=True,
                                                  filters=64, kernel_size=[4, 4, 4], strides=[1, 1, 1], padding='same',
                                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                      2.0),
                                                  bias_initializer=tf.zeros_initializer())
                        max_pool2 = tf.layers.max_pooling3d(conv_2, 2, 2, name='max_pool2')
                        conv3 = tf.layers.conv3d(max_pool2, name='conv3', reuse=True,
                                                 filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same',
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
                                                 bias_initializer=tf.zeros_initializer())

                ### now for the dense layers##
                if 'Dueling' not in self.method:
                    fc0 = FullyConnected('fc0_{}'.format(i), conv3, 512, activation=tf.nn.relu)
                    fc1 = FullyConnected('fc1_{}'.format(i), fc0, 256, activation=tf.nn.relu)
                    fc2 = FullyConnected('fc2_{}'.format(i), fc1, 128, activation=tf.nn.relu)
                    Q = FullyConnected('fct_{}'.format(i), fc2, self.num_actions, nl=tf.identity)
                    Q_list.append(tf.identity(Q, name='Qvalue_{}'.format(i)))



                else:
                    fc0 = FullyConnected('fc0V_{}'.format(i), conv3, 512, activation=tf.nn.relu)
                    fc1 = FullyConnected('fc1V_{}'.format(i), fc0, 256, activation=tf.nn.relu)
                    fc2 = FullyConnected('fc2V_{}'.format(i), fc1, 128, activation=tf.nn.relu)
                    V = FullyConnected('fctV_{}'.format(i), fc2, 1, nl=tf.identity)

                    fcA0 = FullyConnected('fc0V_{}'.format(i), conv3, 512, activation=tf.nn.relu)
                    fcA1 = FullyConnected('fc1V_{}'.format(i), fcA0, 256, activation=tf.nn.relu)
                    fcA2 = FullyConnected('fc2V_{}'.format(i), fcA1, 128, activation=tf.nn.relu)
                    A = FullyConnected('fctV_{}'.format(i), fcA2, self.num_actions, nl=tf.identity)

                    Q = tf.add(A, V - tf.reduce_mean(A, 1, keepdims=True))
                    Q_list.append(tf.identity(Q, name='Qvalue_{}'.format(i)))

        return Q_list

###############################################################################

def get_config(files_list, input_names=['state_1','state_2'],
             output_names=['Qvalue_1','Qvalue_2'],agents=2,reward_strategy=1,prev_exp=None):
    """This is only used during training."""
    expreplay = ExpReplay(
        predictor_io_names=(input_names, output_names),
        player=get_player(task='train', files_list=files_list,agents=agents,reward_strategy=reward_strategy),
        state_shape=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY,
        agents=agents,
        prev_exp=prev_exp
    )
    callbacks = [
        ModelSaver(),
        PeriodicTrigger(
            RunOp(DQNModel.update_target_param, verbose=True),
            # update target network every 10k steps
            every_k_steps=10000 // UPDATE_FREQ),
        expreplay,
        # These other epochs 60 >= will not be reached
        ScheduledHyperParamSetter('learning_rate', [(0, args.lr), (60, 4e-4), (100, 2e-4)]),
        ScheduledHyperParamSetter(
            ObjAttrParam(expreplay, 'exploration'),
            # 1->0.1 in the first million steps
            [(0, 1), (10, 0.1), (320, 0.01)],
            interp='linear'),
        PeriodicTrigger(
            Evaluator(nr_eval=EVAL_EPISODE, input_names=input_names,
                      output_names=output_names, files_list=files_list,
                      get_player_fn=get_player, agents=agents, reward_strategy=reward_strategy),
            every_k_epochs=EPOCHS_PER_EVAL),
        HumanHyperParamSetter('learning_rate')
    ]

    if not args.no_erb:
        erbSaver = ERBSaver(expreplay)
        callbacks.append(PeriodicTrigger(erbSaver, every_k_epochs=1))

    return TrainConfig(
        # dataflow=expreplay,
        data=QueueInput(expreplay),
        model=Model(agents=agents, lr=args.lr),
        callbacks=callbacks,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=args.max_epochs,
    )



###############################################################################
###############################################################################


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform. Must load a pretrained model if task is "play" or "eval"',
                        choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling','DuelingDouble'],
                        default='DQN')
    parser.add_argument('--files', type=argparse.FileType('r'), nargs='+', default= ('/vol/biomedic2/aa16914/shared/thanos/data/cardiac_train_files.txt', '/vol/biomedic2/aa16914/shared/thanos/data/cardiac_train_landmarks.txt'),
                        help="""Filepath to the text file that comtains list of images.
                                Each line of this file is a full path to an image scan.
                                For (task == train or eval) there should be two input files ['images', 'landmarks']""")
    # parser.add_argument('--files', type=argparse.FileType('r'), nargs='+', default=(
    #     '/vol/medic01/users/aa16914/projects/tensorpack-medical-gitlab/examples/LandmarkDetection/DQN/data/fetal_brain_us_yuanwei_miccai_2018/train_list.txt','a'),
    #                     help="""Filepath to the text file that comtains list of images.
    #                                     Each line of this file is a full path to an image scan.
    #                                     For (task == train or eval) there should be two input files ['images', 'landmarks']""")
    #
    parser.add_argument('--saveGif', help='save gif image of the game', action='store_true', default=False)
    parser.add_argument('--saveVideo', help='save video of the game', action='store_true', default=False)
    parser.add_argument('--logDir', help='store logs in this directory during training', default='train_log')
    parser.add_argument('--name', help='name of current experiment for logs', default='dev')
    parser.add_argument('--lr', type=float, help='Learning rate to use while training', default=1e-4)
    parser.add_argument('--max_epochs', type=int, help='The maximum number of epochs to train for', default=3)
    parser.add_argument('--agents', help='Number of agents to train together', default=2)
    # For reward strategy: 1=_calc_reward, 2=_calc_reward_geometric, 3=_distance_to_other_agents, 4=_distance_to_other_agents_and_line,
    # 5=_distance_to_other_agents_and_line_no_point, 6=_calc_reward_geometric_generalized
    parser.add_argument('--reward_strategy', help='Which reward strategy you want? 1 is simple, 2 is line based, 3 is agent based',default=1)
    parser.add_argument('--prev_exp', type=argparse.FileType('rb'), nargs='+', help='Add any previous experience you want to use to avoid catastrophic forgetting', default=None)
    parser.add_argument('--no_erb', action='store_true', help='Flag specifying to NOT save ERB for lifelong learning')
    parser.add_argument('--iter',default=2000)

    args = parser.parse_args()
    STEPS_PER_EPOCH=args.iter
    if args.prev_exp is None:
        prev_exp = args.prev_exp
    else:
        prev_exp = []
        if isinstance (args.prev_exp, list):
            for idx in range(len(args.prev_exp)):
                filehandler = open(args.prev_exp[idx].name, 'rb') 
                prev_exp.append(pickle.load(filehandler))
        else:
            filehandler = open(args.prev_exp.name, 'rb') 
            prev_exp.append(pickle.load(filehandler))
        #prev_exp._curr_pos=0
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # check input files
    if args.task == 'play':
        error_message = """Wrong input files {} for {} task - should be 1 \'images.txt\' """.format(len(args.files), args.task)
        assert len(args.files) == 1
    else:
        error_message = """Wrong input files {} for {} task - should be 2 [\'images.txt\', \'landmarks.txt\'] """.format(len(args.files), args.task)
        assert len(args.files) == 2, (error_message)

    args.agents=int(args.agents)

    METHOD = args.algo
    # load files into env to set num_actions, num_validation_files
    for idx in range(len(args.files)):
      args.files[idx] = args.files[idx].name
    init_player = MedicalPlayer(files_list=args.files,
                                screen_dims=IMAGE_SIZE,
                                task='train',agents=args.agents,reward_strategy=args.reward_strategy)
    NUM_ACTIONS = init_player.action_space.n
    num_files = init_player.files.num_files

  ##########################################################
    #initialize states and Qvalues for the various agents
    state_names=[]
    qvalue_names=[]
    for i in range (0,args.agents):
        state_names.append('state_{}'.format(i))
        qvalue_names.append('Qvalue_{}'.format(i))


############################################################


    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(agents=args.agents, lr=args.lr),
            session_init=get_model_loader(args.load),
            input_names=state_names,
            output_names=qvalue_names))
        # demo pretrained model one episode at a time
        if args.task == 'play':
            play_n_episodes(get_player(directory=args.logDir,
                                       files_list=args.files, viz=0.01,
                                       saveGif=args.saveGif,
                                       saveVideo=args.saveVideo,
                                       task='play',agents=args.agents,reward_strategy=args.reward_strategy),
                            pred, num_files, agents=args.agents)
        # run episodes in parallel and evaluate pretrained model
        elif args.task == 'eval':
            play_n_episodes(get_player(directory=args.logDir,
                                       files_list=args.files, viz=0.01,
                                       saveGif=args.saveGif,
                                       saveVideo=args.saveVideo,
                                       task='eval',agents=args.agents,reward_strategy=args.reward_strategy),
                            pred, num_files, agents=args.agents)
            
    else:  # train model
        logger_dir = os.path.join(args.logDir, args.name)
        logger.set_logger_dir(logger_dir)
        config = get_config(args.files, input_names=state_names,
             output_names=qvalue_names,agents=args.agents,reward_strategy=args.reward_strategy,prev_exp = prev_exp)
        if args.load:  # resume training from a saved checkpoint
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
