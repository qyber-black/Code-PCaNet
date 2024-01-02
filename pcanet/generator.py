# generator.py - dataset sequence generators for tensorflow
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import glob
import psutil
import numpy as np
import cv2 as cv
from tensorflow.keras.utils import Sequence

class SeqClassifyFiles(Sequence):
  # Keras data sequence from filenames
  # May want to consider tf.data for speed, but harder to keep track of pids, etc.

  def __init__(self, Xfiles, Y, ds, tags, batch_size, size, shuffle=True, bits=8):
    super(SeqClassifyFiles, self).__init__()
    self.Xfiles = Xfiles
    self.Y = Y
    self.dataset = ds
    self.tags = tags
    self.batch_size = batch_size
    self.dim = (size,size)
    self.shuffle = shuffle
    self.bits = bits # Assume 8-bit images by default; treat them in general as images
    self.max_val = 2**bits-1
    self.idx = np.arange(0,len(self.Xfiles))
    tag = self.tags[self.Y[0]]
    pid = self.Xfiles[0].split(":")[0]
    self.num_channels = self.dataset[(tag,pid)][0].shape[0]
    self.cache = [None] * len(self.idx)
    self.cache_warn = False
    self.on_epoch_end() # Run shuffle on first, too

  def __len__(self):
    # Number of batches per epoch
    batches = len(self.Xfiles) // self.batch_size
    if batches * self.batch_size < len(self.Xfiles):
      batches += 1
    return batches

  def __getitem__(self, index):
    # Generate one batch
    batch_idx = self.idx[index*self.batch_size:min((index+1)*self.batch_size,len(self.Xfiles))]
    X = np.empty((len(batch_idx), self.num_channels, *self.dim, 1), dtype=np.float64)
    Y = np.empty(len(batch_idx), dtype=int)
    for l in range(0,len(batch_idx)):
      if self.cache[batch_idx[l]] is None:
        pid = self.Xfiles[batch_idx[l]].split(":")
        data = self.dataset[(self.tags[self.Y[batch_idx[l]]],pid[0])][int(pid[1])] * self.max_val
        for ch in range(0,data.shape[0]):
          X[l,ch,:,:,0] = np.clip(cv.resize(data[ch,:,:], self.dim, interpolation=cv.INTER_AREA), 0, self.max_val)
        if psutil.virtual_memory().percent < 95:
          if self.cache_warn:
            print(f"***Warning: memory available: {psutil.virtual_memory().percent}%; restarting cache***")
            self.cache_warn = False
          self.cache[batch_idx[l]] = np.copy(X[l,:,:,:,0])
        elif not self.cache_warn:
          print(f"***Warning: memory low: {psutil.virtual_memory().percent}%; stopping cache***")
          self.cache_warn = True
      else:
        X[l,:,:,:,0] = np.copy(self.cache[batch_idx[l]])
      Y[l] = self.Y[batch_idx[l]]
    return X, Y

  def on_epoch_end(self):
    # Shuffle indices, if requested
    if self.shuffle == True:
      np.random.shuffle(self.idx)

  def disable_shuffle(self):
    self.shuffle = False
    self.idx = np.arange(len(self.Xfiles))

  def enable_shuffle(self):
    self.shuffle = True
    self.idx = np.arange(len(self.Xfiles))
    self.on_epoch_end()

  def get_all(self):
    bs = self.batch_size
    self.batch_size = len(self.Xfiles)
    X, Y = self[0]
    self.batch_size = bs
    return X, Y

  def get_Y(self):
    return np.array([self.Y[p] for p in self.idx])

  def get_P(self):
    return np.array([self.Xfiles[p] for p in self.idx])
