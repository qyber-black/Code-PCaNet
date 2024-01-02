# augment.py - dataset augmentation
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import scipy as sp
import scipy.interpolate
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import joblib

import pcanet.dataset as dataset

class Augment:

  def __init__(self, masks, factor, parallel=True, verbose=0):
    self.augs = []
    self.masks = masks
    self.factor = factor
    self.parallel = parallel
    self.verbose = verbose
    self.rng = np.random.default_rng()

  def add(self, prob, func):
    # Add augmentation function (expected from this class), applied with given probability
    self.augs.append((prob,func))
    return self

  def add_default_classify(self):
    # Default augmentation for classification
    self.add(0.5, lambda s: self.contrast(s, contrast=0.2, bright=0.1))
    self.add(0.5, lambda s: self.blur(s, sigma_max=1))
    self.add(0.5, lambda s: self.gaussian_noise(s, mean_max=0.0, sigma_max=0.1))
    self.add(0.5, lambda s: self.shear(s, shear_max=0.25)) # !!
    self.add(0.5, lambda s: self.elastic_deform(s, alpha=2, sigma=0.08))
    self.add(0.5, self.flip_horizontal)
    self.add(0.5, self.flip_vertical)
    self.add(0.5, lambda s: self.rotate(s, angle=90))
    self.add(0.5, lambda s: self.scale(s, scale_per=0.1))
    return self

  def add_default_segment(self):
    # Default augmentation for segmentation
    self.add(1.5, lambda s: self.contrast(s, contrast=0.2, bright=0.1))
    self.add(1.5, lambda s: self.blur(s, sigma_max=1))
    self.add(1.5, lambda s: self.gaussian_noise(s, mean_max=0.0, sigma_max=0.1))
    self.add(1.5, lambda s: self.shear(s,shear_max=0.25))
    self.add(1.5, lambda s: self.elastic_deform(s, alpha=2, sigma=0.08))
    self.add(1.5, self.flip_vertical)
    self.add(1.5, lambda s: self.rotate(s,angle=30))
    self.add(1.5, lambda s: self.scale(s,scale_per=0.1))
    self.add(1.5, lambda s: self.translate(s,trans_per=0.1))
    return self

  def _augment_slice(self, ds, pid, post_process):
    # Augment one slice, including original
    orig = ds[pid]
    data = np.zeros((self.factor,*orig.shape), dtype=np.float64)
    data[0,:,:,:] = orig
    size = np.prod(data.shape[1:])
    # Store original mask values to be able to map them back
    orig_mask_vals = []
    for midx in self.masks:
      orig_mask_vals.append(set(np.unique(orig[midx,:,:])))
    # Augment images
    for n in range (1,self.factor):
      rep = True
      rep_cnt = 0
      while rep:
        # Apply augmentations with probability
        data[n,:,:,:] = np.copy(data[0,:,:,:])
        for a in self.augs:
          if self.rng.random() < a[0]:
            data[n,:,:,:] = a[1](data[n,:,:,:])
        # Make sure it is sufficiently different from the others
        rep = False
        for ss in range(0,n-1):
          if (np.sum(np.abs(data[ss,:,:,:] - data[n,:,:,:])) / size) < 1e-4:
            rep = True
            rep_cnt += 1
            if rep_cnt > 10000: # Fail-safe
              raise Exception(f"Repetition failed to create different image for {pid}")
            break
      if len(self.masks) > 0:
        # Map masks to original values if changed due to augmentation
        data[n,self.masks,:,:] = np.floor(data[n,self.masks,:,:]+0.5)
        for cnt, midx in enumerate(self.masks):
          mask_vals = set(np.unique(data[n,midx,:,:]))
          if mask_vals != orig_mask_vals[cnt]:
            data_mask = data[n,midx,:,:]
            orig_mask_vals[cnt] = list(orig_mask_vals[cnt])
            for val in mask_vals.difference(orig_mask_vals[cnt]):
              new_val_idx = np.argmin(np.abs(orig_mask_vals[cnt] - val) + 0.5)
              val_loc = (data_mask == val)
              data_mask[val_loc] = orig_mask_vals[cnt][new_val_idx]
      if self.verbose > 3:
        f,a = plt.subplots(data.shape[1],3)
        if len(a.shape) == 1:
          a.reshape[1,a.shape[1]]
        f.suptitle(f"{pid} augmentation {n+1}")
        for chidx in range(0,data.shape[1]):
          a[chidx,0].imshow(data[0,chidx,:,:],cmap='gray',vmin=0)
          a[chidx,1].imshow(data[n,chidx,:,:],cmap='gray',vmin=0)
          a[chidx,2].imshow(data[n,chidx,:,:] - data[0,chidx,:,:],cmap='gray',vmin=0)
        plt.show()
    if post_process is not None:
      # Any further processing and saving of whole augmentation set given by function
      post_process(data, pid)

  def apply(self, ds, post_process):
    # Apply augmentation to dataset
    if self.parallel:
      joblib.Parallel(n_jobs=-1, prefer="threads", verbose=10*self.verbose) \
          (joblib.delayed(self._augment_slice)(ds, p, post_process) for p in ds)
    else:
      for p in ds:
        self._augment_slice(ds, p, post_process)

  def contrast(self, slice, contrast, bright):
    for idx in range(0,slice.shape[0]):
      if idx not in self.masks: # Do not apply to masks
        lo, hi = np.min(slice[idx,:,:]), np.max(slice[idx,:,:])
        r = hi - lo
        f = 1.0 + self.rng.uniform(-contrast,contrast)
        b = np.clip(self.rng.uniform(-bright,bright) * r, lo, hi)
        slice[idx,:,:] = np.clip(f*slice[idx,:,:] + b, lo, hi)
    return slice

  def gaussian_noise(self, slice, mean_max, sigma_max):
    for idx in range(0,slice.shape[0]):
      if idx not in self.masks: # do not apply to masks
        lo, hi = np.min(slice[idx,:,:]), np.max(slice[idx,:,:])
        r = hi - lo
        slice[idx,:,:] = np.clip(slice[idx,:,:] + 
                                 self.rng.normal(self.rng.random()*mean_max*r,
                                                 self.rng.random()*sigma_max*r,
                                                 (1,slice.shape[1],slice.shape[2])),
                                 lo, hi)
    return slice

  def blur(self, slice, sigma_max):
    return ndi.gaussian_filter(slice, [0, self.rng.random()*sigma_max, np.random.random()*sigma_max])

  def flip_horizontal(self, slice):
    return np.flip(slice, axis=1)

  def flip_vertical(self, slice):
    return np.flip(slice, axis=2)

  def rotate(self, slice, angle):
    return ndi.rotate(slice, angle=self.rng.uniform(-1.0,1.0)*angle,
                      axes=(2,1), reshape=False, mode="grid-constant", cval=0.0)

  def translate(self, slice_data, trans_per):
    shift_max = [trans_per*slice_data.shape[1],trans_per*slice_data.shape[2]]
    shift = [int(self.rng.uniform(-shift_max[0],shift_max[0])+0.5),
             int(self.rng.uniform(-shift_max[1],shift_max[1])+0.5)]
    slice_data = np.roll(slice_data, shift, axis=[1, 2])
    if shift[0] < 0:
      slice_data[:,(slice_data.shape[1]+shift[0]-1):slice_data.shape[1],:] = 0.0
    elif shift[0] > 0:
      slice_data[:,0:shift[0],:] = 0.0
    if shift[1] < 0:
      slice_data[:,:,(slice_data.shape[2]+shift[1]-1):slice_data.shape[2]] = 0.0
    elif shift[1] > 0:
      slice_data[:,:,0:shift[1]] = 0.0
    return slice_data

  def scale(self, slice, scale_per):
    xs = 1.0 + self.rng.uniform(-scale_per,scale_per)
    ys = 1.0 + self.rng.uniform(-scale_per,scale_per)
    return ndi.interpolation.affine_transform(slice,
                                              np.array([[1, 0, 0, 0],
                                                        [0,xs, 0, 0],
                                                        [0, 0,ys, 0],
                                                        [0, 0, 0, 1]]))

  def shear(self, slice, shear_max):
    shear = self.rng.uniform(-shear_max,shear_max)
    if self.rng.random() < 0.5:
      sx, sy = shear, 0.0
      ox, oy = -shear*slice.shape[1]/2.0, 0.0
    else:
      sx, sy = 0.0, shear
      ox, oy = 0.0, -shear*slice.shape[2]/2.0
    return ndi.interpolation.affine_transform(slice,
                                              np.array([[1, 0, 0, 0],
                                                        [0, 1,sx,ox],
                                                        [0,sy, 1,oy],
                                                        [0, 0, 0, 1]]))

  def distort(self, slice, xs, ys):
    xs = self.rng.random() * xs # in [0,0.1]
    ys = self.rng.random() * ys # not larger than image size
    func = np.sin if self.rng.random() < 0.5 else np.cos
    vert = np.random.random() < 0.5
    for i in range(slice.shape[2 if vert else 1]):
      delta = int(ys * func(np.pi * i * xs))
      if vert:
        slice[:,:,i] = np.roll(slice[:,:,i], delta, axis=1)
      else:
        slice[:,i,:] = np.roll(slice[:,i,:], delta, axis=1)
    return slice

  def elastic_deform(self, slice, alpha, sigma):
    # Random elastic deformations
    # PY Simard, D Steinkraus, JC Platt. Best Practices for Convolutional Neural
    # Networks applied to Visual Document Analysis. Int Conf Document Analysis
    # and Recognition, 958-963, 2003.
    alpha, sigma = slice.shape[1] * alpha, slice.shape[1] * sigma
    dx = ndi.gaussian_filter((self.rng.random(slice.shape[1:3]) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha
    dy = ndi.gaussian_filter((self.rng.random(slice.shape[1:3]) * 2 - 1), sigma,
                             mode="constant", cval=0.) * alpha
    # Coordinates
    coords = np.arange(slice.shape[2]), np.arange(slice.shape[1])
    # Sample points
    x, y = np.meshgrid(np.arange(slice.shape[2]), np.arange(slice.shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    for c in range(0,slice.shape[0]):
      intrp = sp.interpolate.RegularGridInterpolator(coords, slice[c,:,:], method="linear",
                                                     bounds_error=False, fill_value=0.0)
      slice[c,:,:] = intrp(indices).reshape(slice.shape[1:3])
    return slice
