# dataset.py - raw, classification and segmentation datasets
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import joblib
import cv2 as cv

from pcanet.cfg import Cfg

class Dataset:

  def __init__(self, folder, channels, shape, verbose=0):
    # Create dataset from folder, assumed to be crated by qdicom-utilities;
    # see details on structure there: FOLDER/PID/PID-SESSION-SCAN-SLICE-CHANNEL.npy

    self.folder = folder
    self.channels = channels
    self.verbose = verbose

    # Find all slices and channels
    if verbose > 0:
      print(f"# Loading dataset {self.folder}")

    # Find all slice folders
    files = sorted(glob.glob(os.path.join(folder,'*','*.npy')))
    self.slices = {}
    found_channels = []
    for s in files:
      fn = os.path.basename(s).split("-")
      fn[-1] = fn[-1].split(".")[0]
      p = "-".join(fn[0:4])
      if p not in self.slices:
        self.slices[p] = os.path.dirname(s)
      c = "-".join(fn[4:])
      if c not in found_channels:
        found_channels.append(c)
    for c in self.channels:
      if c not in found_channels:
        raise Exception(f"Channel {c} not found in dataset {self.folder}")

    if verbose > 0:
      print(f"  Slices: {len(self.slices)}")
      print(f"  Channels: {', '.join([c for c in self.channels])}")

    # Get resolution (all need to be the same; guaranteed by create_dataset from qdicom-utilities)
    if shape is None:
      self.shape = None
    elif len(shape) != 2 or (shape[0] == 0 and shape[1] == 0):
      sample = np.load(files[0])
      self.shape = sample.shape
    if verbose > 0:
      if self.shape is None:
        print("  Height, Width: keep original shape")
      else:
        print(f"  Height, Width: {', '.join([str(s) for s in self.shape])}")

  def __iter__(self):
    # Iterator via slices dictionary iterate: init
    self._slice_iter = iter(self.slices)
    return self

  def __next__(self):
    # Iterator via slices dictionary iterate: next
    return next(self._slice_iter)

  def __getitem__(self, pid):
    # Load slice
    slice_data = None
    # Load per slice, over channels
    for ch_idx, ch in enumerate(self.channels):
      fn = os.path.join(self.slices[pid],pid+"-"+ch+".npy")
      # If it does not exist, leave it at 0 (specifically for masks); otherwise load
      if not os.path.isfile(fn) and ch[0:5] == "dwi_c":
        # Check if we find another dwi_c channel (b-values can vary for computed dwi)
        # Manually listed alternatives instead of searching - we know these are alternatives in the datasets
        for v in ["1400", "1000", "1600"]:
          fn = os.path.join(self.slices[pid],pid+"-dwi_c-"+v+".npy")
          if os.path.isfile(fn):
            break
      if os.path.isfile(fn):
        data = np.load(fn)
        # If shape is not specified, first channel determines shape; otherwise scale to specified shape
        if slice_data is None:
          if self.shape is None:
            slice_data = np.zeros((len(self.channels), *data.shape), dtype=np.float64)
          else:
            slice_data = np.zeros((len(self.channels), self.shape), dtype=np.float64)
        if data.shape != slice_data.shape:
          # Scale to standard shape of dataset (either because we rescale or shapes do not match)
          data = cv.resize(data, slice_data.shape[1:], interpolation=cv.INTER_AREA)
          if ch[0:6] == "pirads" or ch == "suspicious" or ch == "normal":
            # Round masks to integer
            data = np.floor(data+0.5)
        slice_data[ch_idx,:,:] = data[:,:]
    return slice_data

class SegmentationDataset:

  def __init__(self, folder, channels=None, verbose=0):
    # Create dataset from folder, assumed to be crated by augmentation:
    # FOLDER_FACTOR_CHANNEL1_CHANNEL2_.../PID/AID.npy

    self.folder = folder
    self.channels = channels
    self.verbose = verbose

    if self.verbose > 0:
      print(f"# Loading {self.folder}")

    # Recover factor and channels from folder name
    fs = os.path.basename(self.folder).split("_")
    self.factor = int(fs[1])
    chs = fs[2:-1]
    found_channels = []
    skip = False
    for l in range(0,len(chs)):
      if skip:
        skip = False
      else:
        if chs[l] == "dwi" and l+1 < len(chs) and (chs[l+1][0:2] == "c-" or chs[l+1][0:3] == "qc-"):
          found_channels.append(chs[l]+"_"+chs[l+1])
          skip = True
        elif chs[l] == "adc" and l+1 < len(chs) and (chs[l+1] == "r" or chs[l+1][0] == "q"):
          found_channels.append(chs[l]+"_"+chs[l+1])
          skip = True
        elif (chs[l] == "normal" or chs[l] == "suspicious") and l+1 < len(chs) and chs[l+1] == "sq":
          found_channels.append(chs[l]+"_"+chs[l+1])
          skip = True
        else:
          found_channels.append(chs[l])
    # Channels to use and get channel selection index
    if self.channels is not None:
      self._ch_idx = -np.ones(len(self.channels), dtype=int)
      for l in range(0,len(self.channels)):
        if self.channels[l] not in channels:
          raise Exception(f"Channel {self.channels[l]} not in dataset")
        self._ch_idx[l] = found_channels.index(self.channels[l])
    else:
      self.channels = found_channels
      self._ch_idx = np.arange(0,len(self.channels))

    # Find all slice folders containing augmented sets of slices
    self.slices = {}
    for f in sorted(glob.glob(os.path.join(self.folder,'*'))):
      self.slices[os.path.basename(f)] = f

    # Get resolution (all need to be the same; guaranteed by create_dataset from qdicom-utilities)
    sample = np.load(os.path.join(self.slices[next(iter(self.slices))], "000.npy"))
    self.shape = sample.shape

    if self.verbose > 0:
      print(f"  Augmentation factor: {self.factor}")
      print(f"  Slices: {len(self.slices)}")
      print(f"  Channels: {', '.join(self.channels)}")
      print(f"  Channels, Height, Width: {', '.join([str(s) for s in self.shape])}")

  def __iter__(self):
    # Iterator via slices dictionary iterate: init
    self._slice_iter = iter(self.slices)
    return self

  def __next__(self):
    # Iterator via slices dictionary iterate: next
    return next(self._slice_iter)

  def __getitem__(self, pid):
    # Load slices of one augmentation set
    slices = None
    files = sorted(glob.glob(os.path.join(self.slices[pid],'*.npy')))
    for n, f in enumerate(files):
      orig = np.load(f)[self._ch_idx,:,:]
      if slices is None:
        slices = np.zeros((self.factor, *orig.shape), dtype=np.float64)
      slices[n,:,:,:] = orig
    return slices

  def aug_set_size(self, pid):
    return len(glob.glob(os.path.join(self.slices[pid],'*.npy')))

  @staticmethod
  def save_segment(slices, pid, path, verbose=0):
    # Save numpy array of augmented slices [aid,channel,height,width] for pid to path
    fn = os.path.join(path,pid)
    os.makedirs(fn, exist_ok=True)
    for aid in range(0,slices.shape[0]):
      if verbose > 3:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,slices.shape[1])
        fig.suptitle(fn)
        for l in range(0,slices.shape[1]):
          ax[l].imshow(slices[aid,l,:,:], cmap='gray')
        plt.show()
      np.save(os.path.join(fn, f"{aid:03d}.npy"), slices[aid,:,:,:], allow_pickle=False)

class ClassifyDataset:

  def __init__(self, folder, channels=None, verbose=0):
    # Create dataset from folder, assumed to be crated by augmentation:
    # FOLDER_FACTOR_CHANNEL1_CHANNEL2_.../TAG/PID/AID.npy

    self.folder = folder
    self.channels = channels
    self.verbose = verbose

    if self.verbose > 0:
      print(f"# Loading {self.folder}")

    # Recover factor and channels from folder name
    fs = os.path.basename(self.folder).split("_")
    self.factor = int(fs[1])
    chs = fs[2:-1]
    found_channels = []
    skip = False
    for l in range(0,len(chs)):
      if skip:
        skip = False
      else:
        if chs[l] == "dwi" and l+1 < len(chs) and (chs[l+1][0:2] == "c-" or chs[l+1][0:3] == "qc-"):
          found_channels.append(chs[l]+"_"+chs[l+1])
          skip = True
        elif chs[l] == "adc" and l+1 < len(chs) and (chs[l+1] == "r" or chs[l+1][0] == "q"):
          found_channels.append(chs[l]+"_"+chs[l+1])
          skip = True
        elif (chs[l] == "normal" or chs[l] == "suspicious") and l+1 < len(chs) and chs[l+1] == "sq":
          found_channels.append(chs[l]+"_"+chs[l+1])
          skip = True
        else:
          found_channels.append(chs[l])
    # Channels to use and get channel selection index
    if self.channels is not None:
      self._ch_idx = -np.ones(len(self.channels), dtype=int)
      for l in range(0,len(self.channels)):
        if self.channels[l] not in channels:
          raise Exception(f"Channel {self.channels[l]} not in dataset")
        self._ch_idx[l] = found_channels.index(self.channels[l])
    else:
      self.channels = found_channels
      self._ch_idx = np.arange(0,len(self.channels))

    # Find all slice folders containing augmented sets of slices
    self.slices = {}
    self.idx = []
    for t in sorted(glob.glob(os.path.join(self.folder,'*'))):
      tag = os.path.basename(t)
      self.slices[tag] = {}
      for f in sorted(glob.glob(os.path.join(t,'*'))):
        pid = os.path.basename(f)
        self.slices[tag][pid] = f
        self.idx.append((tag,pid))

    if self.verbose > 0:
      print(f"  Augmentation factor: {self.factor}")
      for t in self.slices:
        print(f"    {t}: {len(self.slices[t])}")
      print(f"  Channels: {', '.join(self.channels)}")

  def __iter__(self):
    # Iterator via slices dictionary iterate: init
    self._slice_iter = iter(self.idx)
    return self

  def __next__(self):
    # Iterator via slices dictionary iterate: next
    return next(self._slice_iter)

  def __getitem__(self, idx):
    # Load slices of one augmentation set
    slices = []
    try:
      files = sorted(glob.glob(os.path.join(self.slices[idx[0]][idx[1]],'*.npy')))
    except:
      raise Exception(f"Could not find index {idx}")
    slices = [np.load(f)[self._ch_idx,:,:] for f in files]
    return slices

  def aug_set_size(self, idx):
    return len(glob.glob(os.path.join(self.slices[idx[0]][idx[1]],'*.npy')))

  @staticmethod
  def save_classify(crops, pid, path):
    # Save crop[TAG][CROP-ID][channel,height,width] for pid to path
    for tag in crops.keys():
      if len(crops[tag]) > 0:
        fn = os.path.join(path,tag,pid)
        os.makedirs(fn, exist_ok=True)
        for aid, crop in enumerate(crops[tag]):
          np.save(os.path.join(fn, f"{aid:03d}.npy"), crop, allow_pickle=False)
