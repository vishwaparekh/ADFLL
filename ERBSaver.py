# -*- coding: utf-8 -*-
# File: ERBSaver.py
# Author: Vishwa Parekh <vishwaparekh@gmail.com>

import os
from datetime import datetime

from tensorpack import Callback
from tensorpack.compat import tfv1 as tf
from tensorpack.utils import fs,logger
#from tensorpack.utils.fs import normpath
import pickle

class ERBSaver(Callback):
    """
    Save the model once triggered.
    """

    def __init__(self, expreplay,
                 checkpoint_dir=None):
        """
        Args:
            max_to_keep (int): the same as in ``tf.train.Saver``.
            keep_checkpoint_every_n_hours (float): the same as in ``tf.train.Saver``.
                Note that "keep" does not mean "create", but means "don't delete".
            checkpoint_dir (str): Defaults to ``logger.get_logger_dir()``.
            var_collections (str or list of str): collection of the variables (or list of collections) to save.
        """
        
        self.expreplay = expreplay.mem
        if checkpoint_dir is None:
            checkpoint_dir = logger.get_logger_dir()
        if checkpoint_dir is not None:
            if not tf.gfile.IsDirectory(checkpoint_dir):  # v2: tf.io.gfile.isdir
                tf.gfile.MakeDirs(checkpoint_dir)  # v2: tf.io.gfile.makedirs
        # If None, allow it to be init, but fail later if used
        # For example, if chief_only=True, it can still be safely initialized
        # in non-chief workers which don't have logger dir
        #self.checkpoint_dir = normpath(checkpoint_dir) if checkpoint_dir is not None else checkpoint_dir
        self.checkpoint_dir = checkpoint_dir


    def _trigger(self):
        try:
            filehandler = open(self.checkpoint_dir + '/Experience_Replay_Buffer.obj','wb')
            pickle.dump(self.expreplay, filehandler)
            logger.info("ERB saved ")
        except (IOError, tf.errors.PermissionDeniedError,
                tf.errors.ResourceExhaustedError,
                tf.errors.AlreadyExistsError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ERBSaver!")

