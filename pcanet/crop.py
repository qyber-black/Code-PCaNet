# crop.py - cropping to masks
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import scipy.ndimage as ndi

def crop_classify(slices, masks, channels, make_square=False, verbose=0):
  # Crop to masks from slices[aid,chn,height,width] for channels given by masks
  crops = {channels[m]: [] for m in masks}
  for mask in masks:
    for aid in range(0,slices.shape[0]):
      slice_data = slices[aid,:,:,:]
      # Find components
      labelled, components = ndi.label(slice_data[mask,:,:], structure=np.ones((3,3),dtype=np.int32))
      if components == 0:
        continue
      indices = np.indices(slice_data.shape[1:3]).T[:,:,[1,0]]

      # Draw rectangles of regions on top of each other to merge
      rects = np.zeros((slice_data.shape[1:3]),dtype=np.float64)
      for l in range(0,components):
        idx = indices[labelled == l+1]
        idx_range = (np.amin(idx, axis=0), np.amax(idx, axis=0))
        rects[idx_range[0][0]:idx_range[1][0]+1,idx_range[0][1]:idx_range[1][1]+1] = 1.0

      # Extract bounding boxes from rectangles image and crop (plot if requested)
      labelled, components = ndi.label(rects, structure=np.ones((3,3),dtype=np.int32))
      if components == 0: # Shouldn't happen?
        continue
      if verbose > 3:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,slice_data.shape[0])
        fig.suptitle("Classification components")
        for l in range(0,slice_data.shape[0]):
          ax[0,l].imshow(slice_data[l,:,:], cmap='gray')
      for l in range(0,components):
        idx = indices[labelled == l+1]
        idx_range = (np.amin(idx, axis=0), np.amax(idx, axis=0))
        if make_square:
          # Find minimum centered bounding square, if requested
          # Note, we assume we are not close to the edge of the image and the image
          # is large enough (applies to our datasets); otherwise this will fail  (on
          # purpose here).
          lengths = (idx_range[1][0]+1-idx_range[0][0],idx_range[1][1]+1-idx_range[0][1])
          if lengths[0] > lengths[1]:
            xtend = lengths[0] - lengths[1]
            xmin = xtend // 2
            idx_range[0][1] -= xmin
            idx_range[1][1] += xtend - xmin
            if idx_range[0][1] < 0 or idx_range[1][1] >= slice_data.shape[2]:
              raise Exception("Cannot make square as we are at the edge of the image")
          elif lengths[1] > lengths[0]:
            xtend = lengths[1] - lengths[0]
            xmin = xtend // 2
            idx_range[0][0] -= xmin
            idx_range[1][0] += xtend - xmin
            if idx_range[0][0] < 0 or idx_range[1][0] >= slice_data.shape[1]:
              raise Exception("Cannot make square as we are at the edge of the image")
          lengths = (idx_range[1][0]+1-idx_range[0][0],idx_range[1][1]+1-idx_range[0][1])
          if lengths[0] != lengths[1]:
            raise Exception("Crop was not made square")
        crop = np.copy(slice_data[:,idx_range[0][0]:idx_range[1][0]+1,idx_range[0][1]:idx_range[1][1]+1])
        if np.min(crop.shape[1:]) > 3 and np.sum(crop[mask,:,:]) > 6.0:
          # At least 6 pixels in tagged component with minimum 3 pixels along each axis
          crops[channels[mask]].append(crop)
          if verbose > 3 and l < slice_data.shape[0]:
            ax[1,l].imshow(crop[mask,:,:], cmap='gray')
      if verbose > 3:
        plt.show()

  return crops

def crop_classify_pirads(slices, thresholds, channels, make_square=False, verbose=0):
  # Crop to masks from slices[id,chn,height,width] from pirads channel with normal/suspicious threshold
  crops = { }
  if "pirads" not in channels:
    raise Exception("No PIRADS mask provided")
  mask = channels.index("pirads")
  last_idx = len(thresholds)-1
  for tidx, _ in enumerate(thresholds):
    for aid in range(0,slices.shape[0]):
      slice_data = np.copy(slices[aid,:,:,:])
      # Create class based on thresholds
      if tidx == last_idx:
        slice_data[mask,:,:] = (slice_data[mask,:,:] >= thresholds[tidx]).astype(int)
      else:
        slice_data[mask,:,:] = ((slice_data[mask,:,:] >= thresholds[tidx]) & 
                                (slice_data[mask,:,:] < thresholds[tidx+1])).astype(int)
      # Find components
      labelled, components = ndi.label(slice_data[mask,:,:], structure=np.ones((3,3),dtype=np.int32))
      if components == 0:
        continue
      indices = np.indices(slice_data.shape[1:3]).T[:,:,[1,0]]

      # Draw rectangles of regions on top of each other to merge
      rects = np.zeros((slice_data.shape[1:3]),dtype=np.float64)
      for l in range(0,components):
        idx = indices[labelled == l+1]
        idx_range = (np.amin(idx, axis=0), np.amax(idx, axis=0))
        rects[idx_range[0][0]:idx_range[1][0]+1,idx_range[0][1]:idx_range[1][1]+1] = 1.0

      # Extract bounding boxes from rectangles image and crop (plot if requested)
      labelled, components = ndi.label(rects, structure=np.ones((3,3),dtype=np.int32))
      if components == 0: # Shouldn't happen?
        continue
      if verbose > 3:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,slice_data.shape[0])
        fig.suptitle(f"Classification components for pirads >= {thresholds[tidx]}" + 
                     (f" & pirads < {thresholds[tidx+1]}" if tidx < len(thresholds)-1 else ""))
        for l in range(0,slice_data.shape[0]):
          ax[0,l].imshow(slice_data[l,:,:], cmap='gray', vmin=0)
          ax[0,l].set_title(channels[l])
      for l in range(0,components):
        idx = indices[labelled == l+1]
        idx_range = (np.amin(idx, axis=0), np.amax(idx, axis=0))
        if make_square:
          # Find minimum centered bounding square, if requested
          # Note, we assume we are not close to the edge of the image and the image
          # is large enough (applies to our datasets); otherwise this will fail  (on
          # purpose here).
          lengths = (idx_range[1][0]+1-idx_range[0][0],idx_range[1][1]+1-idx_range[0][1])
          if lengths[0] > lengths[1]:
            xtend = lengths[0] - lengths[1]
            xmin = xtend // 2
            idx_range[0][1] -= xmin
            idx_range[1][1] += xtend - xmin
            if idx_range[0][1] < 0 or idx_range[1][1] >= slice_data.shape[2]:
              raise Exception("Cannot make square as we are at the edge of the image")
          elif lengths[1] > lengths[0]:
            xtend = lengths[1] - lengths[0]
            xmin = xtend // 2
            idx_range[0][0] -= xmin
            idx_range[1][0] += xtend - xmin
            if idx_range[0][0] < 0 or idx_range[1][0] >= slice_data.shape[1]:
              raise Exception("Cannot make square as we are at the edge of the image")
          lengths = (idx_range[1][0]+1-idx_range[0][0],idx_range[1][1]+1-idx_range[0][1])
          if lengths[0] != lengths[1]:
            raise Exception("Crop was not made square")
        crop = np.copy(slice_data[:,idx_range[0][0]:idx_range[1][0]+1,idx_range[0][1]:idx_range[1][1]+1])
        if np.min(crop.shape[1:]) > 3 and np.sum(crop[mask,:,:]) > 6.0:
          # At least 6 pixels in tagged component with minimum 3 pixels along each axis
          cclass = f"pirads{thresholds[tidx]}"
          if cclass not in crops:
            crops[cclass] = []
          crops[cclass].append(crop)
          if verbose > 3 and l < slice_data.shape[0]:
            ax[1,l].imshow(crop[mask,:,:], cmap='gray')
      if verbose > 3:
        plt.show()

  return crops

def segment_pirads(slices, thresholds, channels, verbose=0):
  # Create segmentation masks for pirads score with thresholds
  if "pirads" not in channels:
    raise Exception("No PIRADS mask provided")
  mask = channels.index("pirads")
  out_shape = list(slices.shape)
  out_shape[1] += len(thresholds)-1 # At least two masks for one threshold 
  slice_data = np.zeros(out_shape)
  mask_end = mask + len(thresholds)
  for aid in range(0,slices.shape[0]):
    slice_data[aid,:mask,:,:] = np.copy(slices[aid,:mask,:,:])
    slice_data[aid,mask_end:,:,:] = np.copy(slices[aid,mask+1:,:,:])
    # Generate mask for pirads thresholds
    for idx in range(mask,mask_end):
      l = idx-mask
      if idx == mask_end-1:
        slice_data[aid,idx,:,:] = (slices[aid,mask,:,:] >= thresholds[l]).astype(int)
      else:
        slice_data[aid,idx,:,:] = ((slices[aid,mask,:,:] >= thresholds[l]) & 
                                   (slices[aid,mask,:,:] < thresholds[l+1])).astype(int)
    if verbose > 3:
      import matplotlib.pyplot as plt
      _, ax = plt.subplots(1,slice_data.shape[1])
      for l in range(0,slice_data.shape[1]):
        ax[l].imshow(slice_data[aid,l,:,:], cmap='gray', vmin=0)
        if l < mask:
          ax[l].set_title(channels[l])
        elif l >= mask_end:
          ax[l].set_title(channels[l-len(thresholds)-1])
        else:
          ax[l].set_title(f"pirads{thresholds[l-mask]}")
      plt.show()

  return slice_data
