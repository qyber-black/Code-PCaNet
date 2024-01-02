# classify_rfc.py - RFC classifier
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2021-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

import pcanet.dataset as dataset
import pcanet.texture as texture

class RFCClassifier:

  def __init__(self, model, n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf,
               inp_features, size, channels, standardise, normalise, std_scaler=None, norm_scaler=None,
               data=None, classifier=None, parallel=True, verbose=0):
    # Setup classifier and either load (if it exists) or prepare for training
    self.model = model
    self.verbose = verbose
    try:
      max_features = float(max_features)
      if max_features == int(max_features):
        max_features = int(max_features)
    except:
      pass
    min_samples_split = float(min_samples_split)
    if min_samples_split == int(min_samples_split):
      min_samples_split = int(min_samples_split)
    min_samples_leaf = float(min_samples_leaf)
    if min_samples_leaf == int(min_samples_leaf):
      min_samples_leaf = int(min_samples_leaf)
    if classifier is None:
      if max_depth == 0:
        max_depth = None
      max_features_v = 'sqrt' if max_features == 'auto' else max_features # auto replaced by sqrt
      self.estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                              max_features=max_features_v, min_samples_split=min_samples_split,
                                              min_samples_leaf=min_samples_leaf,
                                              n_jobs=-1 if parallel else 1, verbose=(verbose>2))
    else:
      self.estimator = classifier
    self.estimator.pca = {
      'classifier': 'RFCClassifier',
      'n_estimators': n_estimators,
      'max_depth': max_depth,
      'max_features': max_features,
      'min_samples_split': min_samples_split,
      'min_samples_leaf': min_samples_leaf,
      'inp_features': inp_features,
      'patch_size': size,
      'channels': channels,
      'standardise': standardise,
      'standardise_scaler': std_scaler,
      'normalise': normalise,
      'normalise_scaler': norm_scaler,
      'data': data
    }
    if self.verbose > 0:
      print(self)

  def __str__(self):
    if self.estimator.pca['standardise_scaler'] != None:
      ss = self.estimator.pca['standardise_scaler'].get_params(deep=True)
      if hasattr(self.estimator.pca['standardise_scaler'], "center_"):
        ss["center"] = self.estimator.pca['standardise_scaler'].center_
      if hasattr(self.estimator.pca['standardise_scaler'], "scale_"):
        ss["scale"] = self.estimator.pca['standardise_scaler'].scale_
    else:
      ss = None
    if self.estimator.pca['normalise_scaler'] != None:
      ns = self.estimator.pca['normalise_scaler'].get_params(deep=True)
      if hasattr(self.estimator.pca['normalise_scaler'], "min_"):
        ns["min"] = self.estimator.pca['normalise_scaler'].min_
      if hasattr(self.estimator.pca['normalise_scaler'], "scale_"):
        ns["scale"] = self.estimator.pca['normalise_scaler'].scale_
    else:
      ns = None
    return "# RFC Classifier:\n" + \
           f"  N-estimators: {self.estimator.pca['n_estimators']}\n" + \
           f"  Max depth: {self.estimator.pca['max_depth']}\n" + \
           f"  Max features: {self.estimator.pca['max_features']}\n" + \
           f"  Min samples split: {self.estimator.pca['min_samples_split']}\n" + \
           f"  Min samples leaf: {self.estimator.pca['min_samples_leaf']}\n" + \
           f"  In features: {self.estimator.pca['inp_features']}\n" + \
           f"  Patch size: {self.estimator.pca['patch_size']}\n" + \
           f"  Channels: {self.estimator.pca['channels']}\n" + \
           f"  Standardise: {self.estimator.pca['standardise']}\n" + \
            ("" if ss is None else ("\n".join([f"    {p}: {ss[p]}" for p in ss])+"\n")) + \
           f"  Normalise: {self.estimator.pca['normalise']}\n" + \
            ("" if ns is None else ("\n".join([f"    {p}: {ns[p]}" for p in ns])+"\n")) + \
           f"  Train data: {self.estimator.pca['data']}"

  def json(self):
    if self.estimator.pca['standardise_scaler'] != None:
      ss = self.estimator.pca['standardise_scaler'].get_params(deep=True)
      if hasattr(self.estimator.pca['standardise_scaler'], "center_"):
        ss["center"] = self.estimator.pca['standardise_scaler'].center_
      if hasattr(self.estimator.pca['standardise_scaler'], "scale_"):
        ss["scale"] = self.estimator.pca['standardise_scaler'].scale_
    else:
      ss = {}
    if self.estimator.pca['normalise_scaler'] != None:
      ns = self.estimator.pca['normalise_scaler'].get_params(deep=True)
      if hasattr(self.estimator.pca['normalise_scaler'], "min_"):
        ns["min"] = self.estimator.pca['normalise_scaler'].min_
      if hasattr(self.estimator.pca['normalise_scaler'], "scale_"):
        ns["scale"] = self.estimator.pca['normalise_scaler'].scale_
    else:
      ns = {}
    return {
        'classifier': "RFCClassifier",
        'n_estimators': self.estimator.pca['n_estimators'],
        'max-depth': self.estimator.pca['max_depth'],
        'max_features': self.estimator.pca['max_features'],
        'min_samples_split': self.estimator.pca['min_samples_split'],
        'min_samples_leaf': self.estimator.pca['min_samples_leaf'],
        'inp_features': self.estimator.pca['inp_features'],
        'patch_size': self.estimator.pca['patch_size'],
        'channels': self.estimator.pca['channels'],
        'standardise': self.estimator.pca['standardise'],
        'standardise_scaler': ss,
        'normalise': self.estimator.pca['normalise'],
        'normalise_scaler': ns,
        'train_data': self.estimator.pca['data']
      }

  def title(self):
    return "RFC - " + \
           f"N-estimators: {self.estimator.pca['n_estimators']}, " + \
           f"Max depth: {self.estimator.pca['max_depth']}, " + \
           f"Max feat.: {self.estimator.pca['max_features']}, " + \
           f"Min split: {self.estimator.pca['min_samples_split']}, " + \
           f"Min leaf: {self.estimator.pca['min_samples_leaf']}, " + \
           f"In features: {self.estimator.pca['inp_features']}, " + \
           f"Patch size: {self.estimator.pca['patch_size']}, " + \
           f"Channels: {self.estimator.pca['channels']}, " + \
           f"Std.: {self.estimator.pca['standardise']}, " + \
           f"Norm.: {self.estimator.pca['normalise']}, " + \
           f"Train data: {self.estimator.pca['data']}"

  def preprocess(self, ds, parallel=True):
    if self.estimator.pca['data'] is None:
      self.estimator.pca['data'] = ds.folder
    X, Y, P = texture.compute_features(ds, self.estimator.pca['inp_features'],
                                       self.estimator.pca['patch_size'],
                                       parallel=parallel, verbose=self.verbose)
    return X, Y, P

  def fit(self, X, Y, P, folds, parallel=True):
    # Standardise/Normalise features
    if self.estimator.pca['standardise']:
      if self.verbose > 0:
        print("  Standardise")
      scaler = RobustScaler()
      X = scaler.fit_transform(X)
      self.estimator.pca['standardise_scaler'] = scaler
    if self.estimator.pca['normalise']:
      if self.verbose > 0:
        print("  Normalise")
      scaler = MinMaxScaler()
      X = scaler.fit_transform(X)
      self.estimator.pca['normalise_scaler'] = scaler

    # Train classifier on data input X, classes Y with folds
    num_classes = np.unique(Y).shape[0]
    if self.verbose > 0:
      print(f"# Cross validation for RFC classifier training: {num_classes} classes")
    predictions = { "train": [], "test": [] }
    k_folds = np.max(folds)+1
    for fold in range(0,k_folds):
      if self.verbose > 0:
        print(f"  Fold {fold}")
      idx_train = [k for k in range(0, X.shape[0]) if folds[k] != fold]
      X_train = X[idx_train,:]
      Y_train = Y[idx_train]
      P_train = [P[idx] for idx in idx_train]
      idx_test = [k for k in range(0, X.shape[0]) if folds[k] == fold]
      X_test = X[idx_test,:]
      Y_test = Y[idx_test]
      P_test = [P[idx] for idx in idx_test]
      if self.verbose > 0:
        print(f"    Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
      self.estimator.fit(X_train, Y_train)
      scores_train = self.estimator.predict_proba(X_train)
      scores_test = self.estimator.predict_proba(X_test)
      fpr_train, tpr_train, auc_train, thres_train = {}, {}, {}, {}
      fpr_test, tpr_test, auc_test, thres_test = {}, {}, {}, {}
      for l in range(num_classes):
        fpr_train[l], tpr_train[l], thres_train[l] = roc_curve(Y_train == l, scores_train[:,l])
        auc_train[l] = auc(fpr_train[l], tpr_train[l])
        fpr_test[l], tpr_test[l], thres_test[l] = roc_curve(Y_test == l, scores_test[:,l])
        auc_test[l] = auc(fpr_test[l], tpr_test[l])
      if num_classes > 2:
        # AUC micro-average
        auc_test["micro"] = roc_auc_score(Y_test, scores_test, multi_class="ovr", average="micro")
        auc_train["micro"] = roc_auc_score(Y_train, scores_train, multi_class="ovr", average="micro")
        # AUC macro-average (all classes treated equally a-priori vs. micro-average)
        auc_test["macro"] = roc_auc_score(Y_test, scores_test, multi_class="ovr", average="macro")
        auc_train["macro"] = roc_auc_score(Y_train, scores_train, multi_class="ovr", average="macro")
      predictions["train"].append((self.estimator.predict(X_train),Y_train,P_train,
                                  (fpr_train, tpr_train, thres_train, auc_train)))
      predictions["test"].append((self.estimator.predict(X_test),Y_test,P_test,
                                 (fpr_test, tpr_test, thres_test, auc_test)))

    # Final fit to all data
    self.estimator.fit(X, Y)

    if len(self.model) > 0:
      os.makedirs(self.model, exist_ok=True)
      joblib.dump(self.estimator, os.path.join(self.model,"rfc_classifier.joblib"))

    return predictions

  def predict(self, X, Y=None, P=None, evaluate=True, parallel=True):
    # Predictions
    if self.verbose > 0:
      print(f"# Predicting classes of {X.shape[0]} samples")
    if self.estimator.pca['standardise']:
      X = self.estimator.pca['standardise_scaler'].transform(X)
    if self.estimator.pca['normalise']:
      X = self.estimator.pca['normalise_scaler'].transform(X)
    Y_pred = self.estimator.predict(X)
    if Y is not None and evaluate:
      scores = self.estimator.predict_proba(X)
      fpr, tpr, aucv, thres = {}, {}, {}, {}
      num_classes = np.unique(Y).shape[0]
      for l in range(num_classes):
        fpr[l], tpr[l], thres[l] = roc_curve(Y == l, scores[:,l])
        aucv[l] = auc(fpr[l], tpr[l])
      if num_classes > 2:
        # AUC micro-average
        aucv["micro"] = roc_auc_score(Y, scores, multi_class="ovr", average="micro")
        # AUC macro-average (all classes treated equally a-priori vs. micro-average)
        aucv["macro"] = roc_auc_score(Y, scores, multi_class="ovr", average="macro")
      roc_data = (fpr, tpr, thres, aucv)
    else:
      roc_data = None
    return { "data": [(Y_pred,Y,P,roc_data)] }

  @staticmethod
  def load(model, verbose=0):
    fn = os.path.join(model,"rfc_classifier.joblib")
    if not os.path.exists(fn):
      raise Exception(f"Cannot load {fn}")
    if verbose > 0:
      print(f"# Loading RFCClassifier: {model}")
    classifier = joblib.load(fn)
    if not hasattr(classifier, 'pca') or \
       classifier.pca['classifier'] != 'RFCClassifier':
      raise Exception("Not a RFCClassifier")
    return RFCClassifier(model=model, n_estimators=classifier.pca['n_estimators'],
                         max_depth=classifier.pca['max_depth'],
                         max_features=classifier.pca['max_features'],
                         min_samples_split=classifier.pca['min_samples_split'],
                         min_samples_leaf=classifier.pca['min_samples_leaf'],
                         size=classifier.pca['patch_size'],
                         inp_features=classifier.pca['inp_features'],
                         channels=classifier.pca['channels'],
                         standardise=classifier.pca['standardise'],
                         normalise=classifier.pca['normalise'],
                         std_scaler=classifier.pca['standardise_scaler'],
                         norm_scaler=classifier.pca['normalise_scaler'],
                         data=classifier.pca['data'],
                         classifier=classifier,
                         verbose=verbose)
