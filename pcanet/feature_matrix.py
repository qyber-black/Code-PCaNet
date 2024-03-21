#!/usr/bin/env python3
#
# feature_matrix.py - extract features used in classifiers and plot with accuracy
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import glob
import json
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

def feature_map(path,pattern,labels=['Accuracy', 'AUC', 'F1', 'Sensitivity', 'Specificity']):
  features = set() # Set of features used over all models
  model_features = {} # Dict mapping models to feature list
  model_train_metrics = {} # Dict [mode][metric] listing train metrics for model
  model_test_metrics = {} # Dict [mode][metric] listing test metrics for model
  model_test_perf = [] # List of (model,average_accuracy) pairs for sorting

  # Find all model training.json files
  training_files = sorted(glob.glob(os.path.join(path,pattern,'training.json')))

  for tf in training_files:
    # Load model training.json file
    with open(tf, 'r') as f:
      train_data = json.load(f)
    # Model name is folder name in training.json file path
    model = os.path.basename(os.path.dirname(tf))
    # Add features used by model to set of all used features
    features.update(train_data['classifier']['feature_names'])
    # Store features used by model
    model_features[model] = train_data['classifier']['feature_names']
    # Store test/train metrics for model
    model_train_metrics[model] = {}
    model_test_metrics[model] = {}
    for metric in labels:
      model_train_metrics[model][metric] = [f[metric] for f in train_data['train']]
      model_test_metrics[model][metric] = [f[metric] for f in train_data['test']]
    # Store (model,average_performance) in list
    model_test_perf.append((model,np.mean(model_test_metrics[model][labels[0]])))

  # Sort feature names
  features = sorted(features)

  # Sort models according to average performance
  model_test_perf.sort(key=itemgetter(1))

  # Create array feature_map[model_idx,feature_idx) where model_idx
  # is sorted by performance (index in model_test_perf)
  feature_map = np.zeros((len(model_test_perf)+1,len(features)),dtype=np.float64)
  for model_idx in range(0,len(model_test_perf)):
    for model_feature in model_features[model_test_perf[model_idx][0]]:
      feature_idx = features.index(model_feature)
      feature_map[model_idx+1,feature_idx] = 1
  feature_map[0,:] = np.sum(feature_map[1:,:],axis=0)
  feature_map[0,:] /= np.max(feature_map[0,:])

  fig,ax = plt.subplots(1,2,figsize=(30.0,10.0),sharey=True)

  # Plot map
  ax[0].pcolor(feature_map,edgecolors='k',linewidths=1,snap=True)
  # y-axis labels are model names
  ax[0].set_yticks(0.5+np.arange(0,feature_map.shape[0]))
  model_labels = ["Count"]
  model_labels.extend([model_test_perf[idx][0] for idx in range(0,len(model_test_perf))])
  ax[0].set_yticklabels(model_labels)
  # x-axis labels are feature names
  ax[0].set_xticks(0.5+np.arange(0,feature_map.shape[1]))
  ax[0].set_xticklabels(features,rotation=90)
  ax[0].set_title("Feature Map")

  # Plot performance aligned with map
  xmin = 1.0
  num_labels = 2*len(labels)
  bar_height = 1.0/num_labels
  for p, metric in enumerate(labels):
    # Test metrics
    bar_pos = 0.5-p/num_labels-1/(2*num_labels)
    avgs = [0]
    avgs.extend([np.mean(model_test_metrics[model_test_perf[m][0]][metric]) 
                 for m in range(0,len(model_test_perf))])
    ax[1].barh(0.5+bar_pos+np.arange(0,feature_map.shape[0]), avgs, height=bar_height).set_label(metric+" (Test)")
    for m in range(0,len(model_test_perf)):
      xmin = np.min([xmin,
                     np.min(model_test_metrics[model_test_perf[m][0]][metric])*0.999])
      ax[1].scatter(model_test_metrics[model_test_perf[m][0]][metric], 
                    [1.5+bar_pos+m]*len(model_test_metrics[model_test_perf[m][0]][metric]),
                    s=1, c='k')
    #  Train metrics
    bar_pos = p/num_labels+1/(2*num_labels)
    avgs = [0]
    avgs.extend([np.mean(model_train_metrics[model_test_perf[m][0]][metric]) 
                 for m in range(0,len(model_test_perf))])
    ax[1].barh(0.5-bar_pos+np.arange(0,feature_map.shape[0]), avgs, height=bar_height).set_label(metric+" (Train)")
    for m in range(0,len(model_test_perf)):
      xmin = np.min([xmin,
                     np.min(model_train_metrics[model_test_perf[m][0]][metric])*0.99])
      ax[1].scatter(model_train_metrics[model_test_perf[m][0]][metric], 
                    [1.5-bar_pos+m]*len(model_train_metrics[model_test_perf[m][0]][metric]),
                    s=1, c='k')
  for m in range(0,len(model_test_perf)):
    ax[1].plot([0.0,1.0],[1.0+m,1.0+m],color='k',linewidth=1)
    ax[1].plot([0.0,1.0],[1.5+m,1.5+m],color='grey',linewidth=1)
  ax[1].legend(loc="lower center", ncol=num_labels)
  ax[1].set_xlim([xmin,1])
  ax[1].set_xlabel("")
  ax[1].set_title("Metrics")

  fig.tight_layout()

  return fig, ax

if __name__ == '__main__':
  labels = ['Accuracy', 'AUC']
  feature_map('classify', 'prostatex01v1-svmsbfs:*', labels)
  plt.show()
  feature_map('classify', 'swpca01v1-svmsbfs:*', labels)
  plt.show()
  feature_map('classify', 'swpx01v1-svmsbfs:*', labels)
  plt.show()

  feature_map('classify', 'prostatex01v1-rfcsbfs:*', labels)
  plt.show()
  feature_map('classify', 'swpca01v1-rfcsbfs:*', labels)
  plt.show()
  feature_map('classify', 'swpx01v1-rfcsbfs:*', labels)
  plt.show()
