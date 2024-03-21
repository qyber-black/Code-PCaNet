# pcanet/cfg.py - PCaNet - config
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2024 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import matplotlib.pyplot as plt
import json

class Cfg:
  # Default configuration - do not overwrite here but set alternatives in file
  # These are static variables for the class, accessed via the class. No object
  # of this class should be used; all methods are static.
  #
  # Change these values in ROOT_PATH/cfg.json (generated after first run; overwrites
  # defaults here) or ~/config/mrsnet.json (not generated; overwrites cfg.json and
  # defaults here).
  val = {
    'path_root': None,
    'figsize': (26.67,15.0),
    'default_screen_dpi': 96,
    'screen_dpi': None,
    'image_dpi': [300],
    'base_learning_rate': 1e-4, # Learning rate for batch size 16 (scaled linearly
                                # with batch_size - https://arxiv.org/abs/1706.02677)
    'beta1': 0.9, # Adam beta1
    'beta2': 0.999, # Adam beta2
    'epsilon': 1e-8, # Adam epsilon
    'logits_calibration_percentage': 0.1, # Percentage of train data used for calibration only
    'py_seed': None, # Global python seed
    'tf_seed': None, # Global tensorflow seed
    'tf_deterministic': False # Make tensorflow deterministic
  }
  # Development flags for extra functionalities and test (not relevant for use).
  # These are set via the environment varaible MRSNET_DEV (colon separated list),
  # but all in use should be in the comments here for reference:
  # * selectgpo_optimise_noload: do not load existing datapoints for GPO model selection
  # * spectrum_set_phase_correct: show phase correction effect
  dev_flags = set()
  # check_dataset_export - Test if exporting dataset to tensors is correct in train
  # flag_plots - Show some test graphs for the activated checks
  # feature_selectgpo_optimse_noload - do not load existing results for SelectGPO
  file = os.path.expanduser(os.path.join('~','.config','mrsnet.json'))

  @staticmethod
  def init(bin_path):
    # Root path of mrsnet
    Cfg.val["path_root"] = os.path.dirname(bin_path)
    # Load cfg file - data folders and other Cfg values can be overwritten by config file
    # We first load ROOT/cfg.json, if it exists, then the user config file
    root_cfg_file = os.path.join(Cfg.val["path_root"],'cfg.json')
    root_cfg_vals = {}
    for fc in [root_cfg_file, Cfg.file]:
      if os.path.isfile(fc):
        with open(fc, "r") as fp:
          js = json.load(fp)
          if fc == root_cfg_file:
            root_cfg_vals = js
          for k in js.keys():
            if k in Cfg.val:
              Cfg.val[k] = js[k]
            else:
              if fc != root_cfg_file: # We fix this here later
                raise Exception(f"Unknown config file entry {k} in {fc}")
    # Setup plot defaults
    if Cfg.val["screen_dpi"] == None:
      Cfg.val["screen_dpi"] = Cfg._screen_dpi()
    plt.rcParams["figure.figsize"] = Cfg.val['figsize']
    # Store configs in ROOT/cfg.json if it does not exist
    changed = False
    del_keys = []
    for k in root_cfg_vals.keys(): # Do not store paths and remove old values
      if k[0:5] == 'path_' or k not in Cfg.val:
        del_keys.append(k)
        changed = True
    for k in del_keys:
      del root_cfg_vals[k]
    for k in Cfg.val: # Add any new values (except paths)
      if k[0:5] != 'path_' and k not in root_cfg_vals:
        root_cfg_vals[k] = Cfg.val[k]
        changed = True
    if changed:
      with open(root_cfg_file, "w") as fp:
        print(json.dumps(root_cfg_vals, indent=2, sort_keys=True), file=fp)
    # Dev flags
    if 'OCANET_DEV' in os.environ:
      for f in os.environ['PCANET_DEV'].split(":"):
        Cfg.dev_flags.add(f)
    # Set seeds
    Cfg.set_seeds()

  @staticmethod
  def dev(flag):
    return flag in Cfg.dev_flags

  @staticmethod
  def _screen_dpi():
    # DPI for plots on screen
    try:
      from screeninfo import get_monitors
    except ModuleNotFoundError:
      return Cfg.val['default_screen_dpi']
    try:
      m = get_monitors()[0]
    except:
      return Cfg.val['default_screen_dpi']
    from math import hypot
    try:
      dpi = hypot(m.width, m.height) / hypot(m.width_mm, m.height_mm) * 25.4
      return dpi # set in cfg.json if this is not working
    except:
      return Cfg.val['default_screen_dpi']

  @staticmethod
  def py_seed(seed):
    # Initialize seeds for python libraries with stochastic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

  @staticmethod
  def tf_seed(seed, deterministic):
    # Initialise global seed and determinism for tensorflow
    if deterministic:
      # Deterministic operations in tensorflow
      os.environ['TF_DETERMINISTIC_OPS'] = '1'
      os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    import tensorflow as tf
    tf.random.set_seed(seed)
    if deterministic:
      # Deterministic operations in tensorflow
      tf.config.threading.set_inter_op_parallelism_threads(1)
      tf.config.threading.set_intra_op_parallelism_threads(1)
      tf.config.experimental.enable_op_determinism()

  @staticmethod
  def set_seeds(pseed="cfg", tseed="cfg", tdet="cfg"):
    # Set all seeds using Cfg values if not specified
    if pseed == "cfg":
      pseed = Cfg.val['py_seed']
    if tseed == "cfg":
      tseed = Cfg.val['tf_seed']
    if tdet == "cfg":
      tdet = Cfg.val['tf_deterministic']
    Cfg.py_seed(pseed)
    Cfg.tf_seed(tseed, tdet)
