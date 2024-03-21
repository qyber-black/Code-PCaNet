# texture.py - Texture features
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import joblib
import numpy as np
import scipy.stats as sps
from scipy import interpolate
import cv2 as cv
import mahotas as mt
import mahotas.interpolate as mt_interp # Avoids partial import problem with lbp later on in some strange scenarios
import tensorflow.keras as keras

from pcanet.pcnn import get_pcnn

def texture_features(slice, size, first_order=True, haralick=True, lbp=True,
                     bits=8, normalise=False):
  # Compute features
  # Consider 12 bits, but much more expensive due to haralick features
  slice_features = []
  slice_img = []
  max_val = 2**bits-1
  for chn in range(0,slice.shape[0]):
    data = np.copy(slice[chn,:,:]) * max_val
    if normalise: # Normalise per channel; probably not a good idea
      top, bot = np.amax(data,axis=None), np.amin(data,axis=None)
      if (top - bot) >= 1.0:
        data = (data - bot) * (max_val / (top - bot))
      else:
        data = np.full_like(data, max_val//2)
    # Resize
    if size > 0:
      data = np.clip(cv.resize(data, (size,size), interpolation=cv.INTER_AREA), 0, max_val)
    # Features
    slice_features.append(np.nan_to_num(compute_texture_features(data, first_order, haralick, lbp), copy=False))
    slice_img.append(data)
  return (slice_features, slice_img)

def compute_texture_features(data, first_order, haralick, lbp):
  # Compute texture features for one channel
  # Note, Fehr, et al [2015] takes intensities from exact ROI to classify texture

  # 1st order features: moments of the intensity volume histogram (mean, SD,
  #                     skewness, and kurtosis) computed from the structure ROI.
  #                     [Fehr, et al 2015]
  if first_order:
    mean = data.mean(axis=None)
    sd = data.std(axis=None)
    skewness = sps.skew(data, axis=None)
    kurtosis = sps.kurtosis(data, axis=None)
    first_order_val = np.array([mean, sd, skewness, kurtosis])
  else:
    first_order_val = np.array([])

  # 2nd order features: Haralick features using the gray level co-occurrence
  #                     matrix (GLCM) with 128 bins, consisting of energy,
  #                     entropy, correlation, homogeneity, and contrast.
  #                     [Fehr, et al 2015]
  #                     There are 14 Harlick features per direction, usually
  #                     the four up,down,left,right.
  #                     [Le, et al 2017] says 88 features? (44 per modality?)
  if haralick:
    try:
      haralick_val = mt.features.haralick(data.astype(int), compute_14th_feature=True)
      # 13 per 4 directions (14th often skipped, but added here for selection; unclear why it
      # is supposed to be unstable).
      # We take mean over directions to avoid direction dependency and std to
      # represent their anisotropy
      haralick_val = np.concatenate([haralick_val.mean(axis=0),haralick_val.std(axis=0)])
    except:
      # If haralick fails (it can)
      haralick_val = np.zeros(28)
  else:
    haralick_val = np.array([])

  # Local Binary Pattern features
  if lbp:
    # Note, radius and points could be chosen more carefully/tested
    lbp_val = mt.features.lbp(data.astype(int),radius=1,points=8,ignore_zeros=False)
  else:
    lbp_val = np.array([])

  return np.concatenate([first_order_val, haralick_val, lbp_val])

def texture_feature_names(channels, inp_features, size):
  # Get names of features for channels and input features (output maps indices to feature names)

  # We need to reconstruct the shapes, so this is complex
  # First get the texture features and collect CNN feature names
  cnn_features = []
  first_order = False
  haralick = False
  lbp = False
  for inf in inp_features.split("_"):
    inf = inf.lower()
    if inf == "first":
      first_order = True
    elif inf == "haralick":
      haralick = True
    elif inf == "lbp":
      lbp = True
    else:
      cnn_features.append(inf)
  # Now add CNN features
  cnn_names = []
  cnn_len = []
  for cf in cnn_features:
    model, _ = get_pcnn(cf, size)
    cnn_names.append(cf)
    cnn_len.append(model.output_shape[-1])

  # Construct feature names
  feature_names = []
  for ch in channels:
    if first_order:
      feature_names.append(ch+"-mean")
      feature_names.append(ch+"-std")
      feature_names.append(ch+"-skew")
      feature_names.append(ch+"-kurtosis")
    if haralick:
      for l in range(1,15):
        feature_names.append(ch+f"-haralick{l:02d}-mean")
      for l in range(1,15):
        feature_names.append(ch+f"-haralick{l:02d}-std")
    if lbp:
      for l in range(0,36):
        feature_names.append(ch+f"-lbp-{l:02d}")
  for ch in channels:
    for c in range(0,len(cnn_names)):
      for l in range(1,cnn_len+1):
        feature_names.append(ch+f"-{cnn_names[c]}-{l}")

  return feature_names

def process_aug_set(set, pid, size, first_order=True, haralick=True, lbp=True, norm_slices=False):
  # Compute texture features for all slices in a single set of augmentations
  set_features = {}
  for slice in range(0,len(set)):
    set_features[pid+":"+str(slice)] = texture_features(set[slice], size,
                                                        first_order=first_order,
                                                        haralick=haralick,
                                                        lbp=lbp,
                                                        normalise=norm_slices)
  return set_features

def compute_features(ds, inp_features, size, norm_slices=False, parallel=True, verbose=0):
  # Compute features for dataset
  if verbose > 0:
    print("# Computing features")

  # Features
  cnn_features = []
  first_order, haralick, lbp = False, False, False
  for inf in inp_features.split("_"):
    inf = inf.lower()
    if inf == "first":
      first_order = True
    elif inf == "haralick":
      haralick = True
    elif inf == "lbp":
      lbp = True
    else:
      cnn_features.append(inf)

  # Feature lengths and CNN feature models
  txtfeature_len = 0
  if first_order:
    txtfeature_len += 4
  if haralick:
    txtfeature_len += 28
  if lbp:
    txtfeature_len += 36
  models = []
  preprocess = []
  cnnfeature_len = 0
  for cf in cnn_features:
    model, pre = get_pcnn(cf, size)
    models.append(model)
    preprocess.append(pre)
    cnnfeature_len += model.output_shape[-1]
  if len(cnn_features) > 0:
    w = np.array([[[[[1, 1, 1]]]]], dtype=np.float32)

  features = {}
  total = 0
  for tag in ds.slices:
    if verbose > 0:
      print(f"  {tag} data")
    # Texture features
    if parallel:
      res = joblib.Parallel(n_jobs=-1, prefer="threads", verbose=10*verbose)\
              (joblib.delayed(process_aug_set)(ds[(tag,pid)], pid, size,
                                               first_order=first_order,
                                               haralick=haralick,
                                               lbp=lbp)
                 for pid in ds.slices[tag])
    else:
      res = [process_aug_set(ds[(tag,pid)], pid, size,
                             first_order=first_order,
                             haralick=haralick,
                             lbp=lbp)
             for pid in ds.slices[tag]]
    feature_total = np.sum([len(a) for a in res])

    # CNN features
    if len(cnn_features) > 0:
      fpid = next(iter(res[0]))
      for chn in range(0,len(ds.channels)):
        # Setup input for tag and channel
        X = np.empty((feature_total,*res[0][fpid][1][chn].shape,1))
        cnt = 0
        for l in range(0,len(res)):
          for p in res[l]:
            X[cnt,:,:,0] = res[l][p][1][chn]
            cnt += 1
        # Compute features over all CNN models and append to features in res
        for m in range(0,len(models)):
          y = keras.layers.Conv2D(3, (1,1), padding='same', use_bias=False, weights=w)(X)
          y = np.array(keras.layers.Flatten()(models[m](preprocess[m](y))))
          cnt = 0
          for l in range(0,len(res)):
            for pid in res[l]:
              res[l][pid][0].append(y[cnt])
              cnt += 1
    # Extract features from res
    features[tag] = {}
    for l in range(0,len(res)):
      for pid in res[l]:
        features[tag][pid] = np.concatenate(res[l][pid][0])

    if verbose > 0:
      print(f"    {feature_total}")
    total += feature_total

  tag = next(iter(features))
  pid = next(iter(features[tag]))
  feature_len = features[tag][pid].shape[0]
  if verbose > 0:
    print(f"  Samples: {total}")
    print(f"  Features: {feature_len}")

  # Features to vector
  X = np.zeros((total,feature_len),dtype=np.float64)
  Y = np.zeros(total,dtype=np.int32)
  P = []
  cnt = 0
  for tagn, tag in enumerate(features):
    for pid in features[tag]:
      X[cnt,:] = features[tag][pid]
      Y[cnt] = tagn
      P.append(pid)
      cnt += 1

  return X, Y, P
