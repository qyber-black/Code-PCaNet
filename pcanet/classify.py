# classify.py - unified classifier framework
#
# SPDX-FileCopyrightText: Copyright (C) 2022-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from pcanet.cfg import Cfg

class Classify:

  def __init__(self, classifier, parallel=True, verbose=0):
    self.classifier = classifier
    self.parallel = parallel
    self.verbose = verbose

    self.rng = np.random.default_rng()

    self.X = None
    self.Y = None
    self.P = None
    self.data = None
    self.tags = None
    self.fold = None
    self.k_folds = None
    self.predictions = None

  def preprocess(self, ds, folds, balance=False):
    # Preprocess data to vectors for training
    self.X, self.Y, self.P = self.classifier.preprocess(ds, self.parallel)
    self.tags = [t for t in ds.slices.keys()]
    self.data = ds.folder

    if balance:
      # Balance dataset
      # Not ideal, as we computed/preprocessed them all already, but ensures
      # balance independent of classifier preprocessing (e.g. if it splits samples, etc)
      if self.verbose > 0:
        print("# Balance")
      idx_tag = {}
      for tn, t in enumerate(self.tags):
        idx_tag[t] = [p for p in range(0,len(self.Y)) if self.Y[p] == tn]
      sz_min = np.min([len(idx_tag[t]) for t in self.tags])
      if isinstance(self.X, list):
        sel_x = [True] * len(self.X)
      else:
        sel_x = [True] * self.X.shape[0]
      for tn, t in enumerate(self.tags):
        sz_orig = len(idx_tag[t]) // ds.factor
        if len(idx_tag[t]) > sz_min:
          tf = np.floor(sz_min / sz_orig + 0.5)
          cnt = len(idx_tag[t])
          for l in idx_tag[t]:
            if int(self.P[l].split(":")[-1]) >= tf:
              sel_x[l] = False
              cnt -= 1
          if self.verbose > 0:
            print(f"  {t} - {tf} x {sz_orig}: {len(idx_tag[t])} -> {cnt}")
        else:
          if self.verbose > 0:
            print(f"  {t} - {ds.factor} x {sz_orig}: {len(idx_tag[t])}")
      if isinstance(self.X, list):
        self.X = [v[0] for v in zip(self.X,sel_x) if v[1]]
      else:
        self.X = self.X[sel_x]
      self.Y = self.Y[sel_x]
      self.P = [self.P[l] for l in range(0,len(self.P)) if sel_x[l]]

    # K-fold split per tag over augmentation sets
    if self.verbose > 0:
      print(f"# {folds}-fold split")

    self.fold = np.zeros(len(self.X) if isinstance(self.X, list) else self.X.shape[0], dtype=int)
    if folds > 1:
      idx_tag = {}
      for tn, t in enumerate(self.tags):
        idx_tag[t] = [p for p in range(0,len(self.Y)) if self.Y[p] == tn]
      self.k_folds = folds
      for tn, t in enumerate(self.tags):
        if self.verbose > 0:
          print(f"  {t}")
        pids = {}
        # Find augmentation sets for tag
        for p in idx_tag[t]:
          if self.Y[p] != tn:
            raise Exception("Tag index vs. Y value mismatch")
          pid = self.P[p].split(":")[0]
          if pid not in pids:
            pids[pid] = [p]
          else:
            pids[pid].append(p)
        # Patients need to go into the same fold (pids[pid] are the
        # sample indices of the slices and for 3D annotations we've
        # got multiple slices per patient; does not affect 2D-only
        # annotations as the split is the same)
        patients = {}
        for pid in pids:
          patient = pid.split("-")[0]
          if patient not in patients:
            patients[patient] = [pid]
          else:
            patients[patient].append(pid)
        # Put patients (or slices in case of 2D) into folds
        # This does not particularly maintain the balance between folds, but is quick
        # and if patients have about the same amount of slices (e.g. 1 for the 2D cases)
        # then it does not make a huge difference. We could try to optimize the split
        # such that numbers match, but usually this will still generate at least one
        # quite ill-balanced fold (based on results from the pvc segmentation using an
        # optimized stratified split).
        idx = [patient for patient in patients]
        self.rng.shuffle(idx)
        for l, patient in enumerate(idx):
          f = l % self.k_folds
          for pid in patients[patient]:
            self.fold[pids[pid]] = f
      # Fold sizes
      if self.verbose > 0:
        for f in range(self.k_folds):
          cnt = {}
          for tn, t in enumerate(self.tags):
            cnt[t] = len([l for l in idx_tag[t] if self.fold[l] == f])
          print(f"  Fold {f}: {', '.join([t+': '+str(cnt[t]) for t in cnt])}")
    else:
      self.k_folds = 1


    # Shuffle data
    if self.verbose > 0:
      print("# Shuffle dataset")
    idx = np.arange(0, len(self.X) if isinstance(self.X, list) else self.X.shape[0])
    self.rng.shuffle(idx)
    if isinstance(self.X, list):
      self.X = [self.X[idx[l]] for l in range(0,idx.shape[0])]
    else:
      self.X = self.X[idx]
    self.Y = self.Y[idx]
    self.fold = self.fold[idx]
    self.P = [self.P[idx[l]] for l in range(0,idx.shape[0])]
    
    # Plot dataset and features
    if self.verbose > 3:
      from matplotlib.widgets import Slider
      fig = plt.figure()
      fig.suptitle(f"Sample 0: {self.P[0]} - {self.tags[self.Y[0]]} - Fold {self.fold[0]}")
      slices = ds[(self.tags[self.Y[0]],self.P[0].split(":")[0])]
      aid = int(self.P[0].split(":")[1])
      grid = plt.GridSpec(2, slices[0].shape[0], wspace=0.4, hspace=0.3)
      plt.subplots_adjust(bottom=0.25)
      axslice = plt.axes([0.25,0.1,0.65,0.03])
      slider = Slider(ax=axslice, label='Slice', valmin=1,
                      valmax=len(self.X) if isinstance(self.X, list) else self.X.shape[0],
                      valinit=0, valstep=1)
      aximg = []
      for chn in range(0,slices[aid].shape[0]):
        ax = fig.add_subplot(grid[0,chn])
        aximg.append(ax.imshow(slices[aid][chn,:,:], cmap='gray'))
        ax.set_title(ds.channels[chn])
      if isinstance(self.X, np.ndarray) and len(self.X.shape) == 2: # Only show features if they are 1D vectors
        ax = fig.add_subplot(grid[1,:])
        if not hasattr(self.classifier.estimator, "pca") or \
           'selected_features' not in self.classifier.estimator.pca or \
           self.classifier.estimator.pca['selected_features'] is None:
          col = 'green'
        else:
          col = ['red' if l in self.classifier.estimator.pca['selected_features'] else 'green' for l in range(0,self.X.shape[1])]
        aximg.append(ax.bar(np.arange(self.X.shape[1]), self.X[0,:], align='center', alpha=0.5, color=col))
        ax.set_ylim([np.min(self.X),np.max(self.X)])
      def update_data(num):
        nonlocal self, slices, aid, aximg, fig
        num = int(num) - 1
        fig.suptitle(f"Sample {num}: {self.P[num]} - {self.tags[self.Y[num]]} - Fold {self.fold[num]}")
        slices = ds[(self.tags[self.Y[num]],self.P[num].split(":")[0])]
        aid = int(self.P[num].split(":")[1])
        for chn in range(0,slices[aid].shape[0]):
          aximg[chn].set_data(slices[aid][chn,:,:])
          aximg[chn].set_extent([0,slices[aid].shape[1],0,slices[aid].shape[2]])
          aximg[chn].set_norm(None)
        if isinstance(self.X, np.ndarray) and len(self.X.shape) == 2:
          for r, h in zip(aximg[-1],self.X[num]):
            r.set_height(h)
      slider.on_changed(update_data)
      fig.set_dpi(Cfg.val['screen_dpi'])
      plt.show(block=True)

  def fit(self):
    if self.verbose > 0:
      print("# Training")
    self.predictions = self.classifier.fit(self.X, self.Y, self.P, self.fold, self.parallel)

  def evaluate(self, data_ext=None):
    # Evaluate model across folds
    if self.verbose > 0:
      print("# Evaluating...")

    if self.predictions is None:
      # Evaluate on data, if not trained
      self.predictions = self.classifier.predict(self.X, self.Y, self.P, self.parallel)
      title = self.classifier.title() + "\n\nDataset: " + self.data
      dataset_name = os.path.basename(self.data)
      if data_ext is not None:
        dataset_name += "_"+data_ext
    else:
      title = self.classifier.title() + "\n\nTraining"
      dataset_name = "training"

    # Setup plot (before prediction to get title right)
    num_cols = len(self.predictions) + 1 + len(self.predictions)
    fig = plt.figure()
    grid = plt.GridSpec(self.k_folds,num_cols, figure=fig)
    fig.suptitle(title)

    # Collect metrics from predictions and create plot
    metrics = {}
    roc_ax = []
    for setn, set in enumerate(self.predictions):
      if self.verbose > 0:
        print(f"  Analysing {set}")
      metrics[set] = []

      for fold in range(0,len(self.predictions[set])):
        if setn == 0:
          roc_ax.append(fig.add_subplot(grid[fold,len(self.predictions)]))
        if self.verbose > 0:
          print(f"    Fold {fold}")
        res = self.predictions[set][fold]
        # Values
        num_classes = len(res[3][0])
        if num_classes == 2:
          tp_idx = [k for k in range(0,len(res[0])) if res[0][k] == 1 and res[1][k] == 1]
          tn_idx = [k for k in range(0,len(res[0])) if res[0][k] == 0 and res[1][k] == 0]
          fp_idx = [k for k in range(0,len(res[0])) if res[0][k] == 1 and res[1][k] == 0]
          fn_idx = [k for k in range(0,len(res[0])) if res[0][k] == 0 and res[1][k] == 1]
          fpr, tpr, thres, auc = res[3][0][1], res[3][1][1], res[3][2][1], res[3][3][1]
          tp, tn, fp, fn = len(tp_idx), len(tn_idx), len(fp_idx), len(fn_idx)
          fp_idx = sorted([res[2][l] for l in fp_idx])
          fn_idx = sorted([res[2][l] for l in fn_idx])
        else:
          tp_idx, tn_idx, fp_idx, fn_idx = [], [], [], []
          pid_fp, pid_fn = [], []
          tp, tn, fp, fn = 0, 0, 0, 0
          for l in range(num_classes):
            tp_idx.append(sorted([res[2][k] for k in range(0,len(res[0])) if res[0][k] == l and res[1][k] == l]))
            tn_idx.append(sorted([res[2][k] for k in range(0,len(res[0])) if res[0][k] != l and res[1][k] != l]))
            fp_idx.append(sorted([res[2][k] for k in range(0,len(res[0])) if res[0][k] == l and res[1][k] != l]))
            fn_idx.append(sorted([res[2][k] for k in range(0,len(res[0])) if res[0][k] != l and res[1][k] == l]))
            tp += len(tp_idx[-1])
            tn += len(tn_idx[-1])
            fp += len(fp_idx[-1])
            fn += len(fn_idx[-1])
          fpr, tpr, thres, auc = res[3]
        metrics[set].append({
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Accuracy': (tp+tn) / (tp + tn + fp + fn),
            'Sensitivity':  tp / (tp + fn),
            'Specificity': tn / (tn + fp),
            'F1': 2*tp / (2*tp + fp + fn),
            'FPR': fpr,
            'TPR': tpr,
            'Threshold': thres,
            'PID_FP': fp_idx,
            'PID_FN': fn_idx
          })
        if num_classes > 2:
          if 'macro' in auc:
            metrics[set][-1]['AUCmacro'] = auc['macro']
          if 'micro' in auc:
            metrics[set][-1]['AUCmicro'] = auc['micro']
        else:
          metrics[set][-1]['AUC'] = auc
        # Confusion matrix
        ax = fig.add_subplot(grid[fold,setn])
        if num_classes == 2:
          cm = np.array([[tp,fp],[fn,tn]])
        else:
          cm = np.ndarray((num_classes,num_classes))
          for l in range(num_classes):
            for k in range(num_classes):
              cm[l,k] = len([m for m in range(0,len(res[0])) if res[0][m] == k and res[1][m] == l])
        cmd = ConfusionMatrixDisplay(cm, display_labels=reversed(self.tags))
        cmd.plot(ax=ax)
        ax.set_yticklabels(reversed(self.tags))
        if setn == 0:
          ax.set_ylabel(f"Fold {fold+1}\n\nTrue label")
        if fold == 0:
          ax.set_title(f"{set.capitalize()} Confusion Matrix")
        if fold < len(self.predictions[set])-1:
          ax.set_xlabel("")

        # ROC curves
        if num_classes > 2:
          # ROC curve using OvR macro-average
          fpr_grid = np.linspace(0.0,1.0,1000)
          mean_tpr = np.zeros_like(fpr_grid)
          for l in range(num_classes):
            mean_tpr += np.interp(fpr_grid, fpr[l], tpr[l])
          mean_tpr /= num_classes
          if 'macro' in auc:
            roc_ax[fold].plot(fpr_grid, mean_tpr, label=set+f"- AUCmacro: {auc['macro']:.4f}")
          elif 'micro' in auc:
            roc_ax[fold].plot(fpr_grid, mean_tpr, label=set+f"- AUCmacro: {auc['micro']:.4f}")
          else:
            roc_ax[fold].plot(fpr_grid, mean_tpr, label=set)
        else:
          roc_ax[fold].plot(fpr, tpr, label=set+f" - AUC: {auc:.4f}")
        if fold ==len(self.predictions[set])-1:
          roc_ax[fold].set_xlabel("False positive rate")
        roc_ax[fold].set_ylabel("True positive rate")
        roc_ax[fold].legend(loc="best")
        if fold == 0:
          roc_ax[fold].set_title("ROC Curves")

      # Plot scores over folds
      ax = fig.add_subplot(grid[:,len(self.predictions)+1+setn])
      twd = 0.9
      wd = 2*twd/5
      bars = []
      keys = []
      for m in ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'AUC', 'AUCmacro', 'AUCmicro']:
        if m in metrics[set][0]:
          vals = [metrics[set][l][m] for l in range(0, len(metrics[set]))]
          mean = np.mean(vals)
          std = np.std(vals)
          bars.append(ax.bar(np.arange(1,len(metrics[set])+1)-wd,vals,width=twd/5*0.9, align='center', label=m))
          keys.append(f"{m}: {mean:.4f}$\sigma${std:.4f}")
          wd -= twd/5
      ax.legend(bars, keys, loc="lower center")
      ax.set_title(f"{set.capitalize()} Metrics")
      ax.set_xlabel("Fold")
      ax.set_ylabel("Metric")
      ax.autoscale(tight=True)

    # Complete plots and data
    if len(self.classifier.model) > 0:
      result_file = os.path.join(self.classifier.model,dataset_name)
      for dpi in Cfg.val['image_dpi']:
        plt.savefig(f"{result_file}@{dpi}.png", dpi=dpi)
      # Store metrics in json
      metrics['classifier'] = self.classifier.json()
      with open(result_file+".json", "w") as fp:
        print(json.dumps(metrics, indent=2, sort_keys=True, cls=NpEncoder), file=fp)
    if self.verbose > 1:
      fig.set_dpi(Cfg.val['screen_dpi'])
      plt.show(block=True)
    plt.close()

    return metrics

  def predict(self, data_ext=None):
    # Predict labels for data
    if self.verbose > 0:
      print("# Predicting...")
    pred, data_Y, P, _ = self.classifier.predict(self.X, self.Y, self.P, False, self.parallel)["data"][0]
    pred = np.array(pred).flatten()
    res = { P[l]: (pred[l], data_Y[l]) for l in range(0,len(pred)) } # res[PID] = (PREDICTION,DATASET-LABEL)}
    return res

class NpEncoder(json.JSONEncoder):
  # Encode numpy arrays as list into json file; rest as usual
  def default(self, data):
    if isinstance(data, np.ndarray):
      return data.tolist()
    return json.JSONEncoder.default(self, data)
