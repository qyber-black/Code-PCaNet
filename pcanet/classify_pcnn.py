# classify_pcnn.py - Pretrained CNN classifier
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2021-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model

from pcanet.cfg import Cfg
from pcanet.generator import SeqClassifyFiles
from pcanet.pcnn import get_pcnn
import pcanet.dataset as dataset
import pcanet.texture as texture

class PCNNClassifier:

  def __init__(self, model, type, fc, do, rgb_train, merge, ft_layers, ft_epochs, batch_size,
               epochs, loss, size, channels, temperature=None, data=None, classifier=None, verbose=0):
    # Setup classifier and either load (if it exists) or prepare for training
    self.model = model
    self.verbose = verbose
    if model == "CNN" and ft_layers != 0:
      raise Exception("CNN model is trained in first phase, so fine tuning not possible")
    if classifier is None:
      self.estimator = None
    else:
      self.estimator = classifier
    self.pca = {
      'classifier': 'PCNNClassifier',
      'type': type,
      'fc': fc,
      'do': do,
      'rgb_train': rgb_train,
      'merge': merge,
      'ft_layers': ft_layers,
      'batch_size': batch_size,
      'epochs': epochs,
      'loss': loss,
      'ft_epochs': ft_epochs,
      'patch_size': size,
      'temperature': temperature,
      'channels': channels,
      'data': data
    }
    if self.verbose > 0:
      print(self)

  def __str__(self):
    return "# PCNN Classifier:\n" + \
           f"  Type: {self.pca['type']}\n" + \
           f"  FC: {self.pca['fc']}\n" + \
           f"  DO: {self.pca['do']}\n" + \
           f"  RGB train: {self.pca['rgb_train']}\n" + \
           f"  Merge: {self.pca['merge']}\n" + \
           f"  Finetune: {self.pca['ft_layers']}\n" + \
           f"  Batch size: {self.pca['batch_size']}\n" + \
           f"  Epochs: {self.pca['epochs']}\n" + \
           f"  Loss: {self.pca['loss']}\n" + \
           f"  FT epochs: {self.pca['ft_epochs']}\n" + \
           f"  Patch size: {self.pca['patch_size']}\n" + \
           f"  T: {self.pca['temperature']}\n" + \
           f"  Channels: {self.pca['channels']}\n" + \
           f"  Train data: {self.pca['data']}"

  def json(self):
    return {
        'classifier': "PCNNClassifier",
        'type': self.pca['type'],
        'fc': self.pca['fc'],
        'do': self.pca['do'],
        'rgb_train': self.pca['rgb_train'],
        'merge': self.pca['merge'],
        'ft_layers': self.pca['ft_layers'],
        'batch_size': self.pca['batch_size'],
        'epochs': self.pca['epochs'],
        'loss': self.pca['loss'],
        'ft_epochs': self.pca['ft_epochs'],
        'patch_size': self.pca['patch_size'],
        'temperature': self.pca['temperature'],
        'channels': self.pca['channels'],
        'train_data': self.pca['data']
      }

  def title(self):
    return "PCNN - " + \
           f"Type: {self.pca['type']}, " + \
           f"FC: {self.pca['fc']}, " + \
           f"DO: {self.pca['do']}, " + \
           f"RBG train: {self.pca['rgb_train']}, " + \
           f"Merge: {self.pca['merge']}, " + \
           f"Finetune: {self.pca['ft_layers']}, " + \
           f"Batch size: {self.pca['batch_size']}, " + \
           f"Epochs: {self.pca['epochs']}, " + \
           f"Loss: {self.pca['loss']}, " + \
           f"FT epochs: {self.pca['ft_epochs']}, " + \
           f"Patch size: {self.pca['patch_size']}, " + \
           f"Channels: {self.pca['channels']}, " + \
           f"T: {self.pca['temperature'] if self.pca['temperature'] is not None else -1.0:.4f},\n" + \
           f"Train data: {self.pca['data']}"

  def _construct(self, type, fc, do, rgb_train, merge, chs, size):
    # Construct network

    input = Input(shape=(chs,size,size,1))

    def make_path(type, rgb_train, c, input):
      # Put this in a closure to create each path, if not strange things can
      # happen (all paths produce the same value)

      X = Lambda(lambda x : x[:,c,:,:,:])(input)

      if type.lower() == "cnn":
        # Custom, trainable CNN; VGG like
        X = Conv2D(64,  (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(64,  (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(64,  (3,3), strides=1, padding="same", activation='relu')(X)
        X = MaxPool2D(pool_size=(2,2), strides=(2,2))(X)
        X = Conv2D(128, (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(128, (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(128, (3,3), strides=1, padding="same", activation='relu')(X)
        X = MaxPool2D(pool_size=(2,2), strides=(2,2))(X)
        X = Conv2D(256, (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(256, (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(256, (3,3), strides=1, padding="same", activation='relu')(X)
        X = MaxPool2D(pool_size=(2,2), strides=(2,2))(X)
        X = Conv2D(512, (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(512, (3,3), strides=1, padding="same", activation='relu')(X)
        X = Conv2D(512, (3,3), strides=1, padding="same", activation='relu')(X)
        X = MaxPool2D(pool_size=(2,2), strides=(2,2))(X)
        cnn = [] # Note, fine-tuning won't work and not needed
      else:
        # Each input channel to 3 channels via convolution (trainable or just copy)
        if rgb_train:
          X = Conv2D(3,(3,3),padding='same')(X)
        else:
          # w = np.array([[[[[1, 1, 1]]]]], dtype=np.float32)
          w_init = tf.keras.initializers.Constant(1.)
          X = Conv2D(3,(1,1),padding='same',use_bias=False,kernel_initializer=w_init,trainable=False)(X)

        # Pretrained CNN
        cnn, preprocess = get_pcnn(type.lower(),size)
        cnn._name = cnn._name+f"_{c}"
        cnn.trainable = False
        cnns.append(cnn)
        X = Lambda(preprocess, name=f"preprocess_{c}")(X)
        X = cnn(X)

      if merge == "wavg" or merge == "wadd" or merge == "wconc":
        # Weights per feature
        X = WeighFeatures()(X)

      return X, cnn

    # Construct path per channel
    zs = [None] * chs
    cnns = [None] * chs
    for c in range(0,chs):
      zs[c], cnns[c] = make_path(type, rgb_train, c, input)

    # Merge
    if len(zs) > 1:
      if merge == "conc" or merge == "wconc":
        z = concatenate(zs)
      elif merge == "add" or merge == "wadd":
        z = Add()(zs)
      elif merge == "avg" or merge == "wavg":
        z = Average()(zs)
      else:
        raise Exception(f"Unknown ensemble merge {merge}")
    else:
      z = zs[0]

    # Classify
    z = Flatten()(z)
    if fc > 0:
      z = Dense(fc, activation="relu")(z)
      if do > 0.0:
        z = Dropout(do)(z)
      elif do < 0.0:
        z = BatchNormalization()(z)
      z = Dense(fc, activation="relu")(z)
      if do > 0.0:
        z = Dropout(do)(z)
      elif do < 0.0:
        z = BatchNormalization()(z)
    z = Dense(1)(z)

    return Model(inputs=input, outputs=z), cnns

  def preprocess(self, ds, parallel=True):
    if self.verbose > 0:
      print("# Load")
    if self.pca['data'] is None:
      self.pca['data'] = ds.folder

    # Get slices
    total =np.sum(ds.aug_set_size(l) for l in ds)
    X = []
    Y = np.zeros(total,dtype=np.int32)
    P = []
    cnt = 0
    for tn, t in enumerate(ds.slices):
      if self.verbose > 0:
        print(f"  {t} data")
      feature_total = 0
      for p in ds.slices[t]:
        for l in range(0,ds.aug_set_size((t,p))):
          X.append(p+":"+str(l))
          Y[cnt] = tn
          P.append(p+":"+str(l))
          cnt += 1
        feature_total += 1
      if self.verbose > 0:
        print(f"    {feature_total}")

    self.dataset = ds
    self.tags = [t for t in ds.slices]

    return X, Y, P

  def fit(self, X, Y, P, folds, parallel=True):
    # Train classifier on data input X, classes Y with folds
    if self.verbose > 0:
      print("# Cross validation for PCNN classifier training")
    learning_rate = Cfg.val['base_learning_rate'] * self.pca['batch_size'] / 16.0

    num_classes = np.unique(Y).shape[0]
    if num_classes != 2:
      raise Exception("PCNN classifiers are only created for two classes")

    predictions = { "train": [], "test": [] }
    k_folds = np.max(folds)+1
    history = []
    finetune_history = []
    for fold in range(0,k_folds):
      if self.verbose > 0:
        print(f"## Fold {fold}")
      # Get data
      if self.pca['loss'] == 'bclc' or self.pca['loss'] == 'bfclc':
        train_idx = [k for k in range(0, len(X)) if folds[k] != fold]
        np.random.default_rng().shuffle(train_idx)
        split = int(len(train_idx) * Cfg.val['logits_calibration_percentage'] + 0.5)
        cal_idx = train_idx[:split]
        train_idx = train_idx[split:]
      else:
        train_idx = [k for k in range(0, len(X)) if folds[k] != fold]
      train_data = SeqClassifyFiles([X[k] for k in train_idx],
                                    [Y[k] for k in train_idx],
                                    self.dataset, self.tags,
                                    batch_size=self.pca['batch_size'],
                                    size=self.pca['patch_size'],
                                    shuffle=True, bits=8)
      test_idx = [k for k in range(0, len(X)) if folds[k] == fold]
      test_data = SeqClassifyFiles([X[k] for k in test_idx],
                                   [Y[k] for k in test_idx],
                                   self.dataset, self.tags,
                                   batch_size=self.pca['batch_size'],
                                   size=self.pca['patch_size'],
                                   shuffle=True, bits=8)
      if self.verbose > 0:
        print(f"  Train set: {len(train_data.Xfiles)}, Test set: {len(test_data.Xfiles)}")

      # Construct model
      self.estimator, cnns = self._construct(self.pca['type'], self.pca['fc'], self.pca['do'],
                                             self.pca['rgb_train'], self.pca['merge'],
                                             len(self.pca['channels']), self.pca['patch_size'])
      if fold == 0:
        # Only do this once for the architecture
        if len(self.model) > 0:
          os.makedirs(self.model, exist_ok=True)
          # Save architecture png
          for dpi in Cfg.val['image_dpi']:
            keras.utils.plot_model(self.estimator,
                                   to_file=os.path.join(self.model,'architecture@'+str(dpi)+'.png'),
                                   show_shapes=True,
                                   show_dtype=True,
                                   show_layer_names=True,
                                   rankdir='TB',
                                   expand_nested=True,
                                   dpi=dpi)
        if self.verbose > 0:
          self.estimator.summary()
      optimiser = keras.optimizers.Adam(learning_rate=learning_rate,
                                        beta_1=Cfg.val['beta1'],
                                        beta_2=Cfg.val['beta2'],
                                        epsilon=Cfg.val['epsilon'])
      if self.pca['loss'] == 'bcl' or self.pca['loss'] == 'bclc':
        loss = keras.losses.BinaryCrossentropy(from_logits=True)
      elif self.pca['loss'] == 'bc':
        loss = keras.losses.BinaryCrossentropy(from_logits=False)
      elif self.pca['loss'] == 'bfcl' or self.pca['loss'] == 'bfclc':
        loss = keras.losses.BinaryFocalCrossentropy(from_logits=True)
      elif self.pca['loss'] == 'bfc':
        loss = keras.losses.BinaryFocalCrossentropy(from_logits=False)
      elif self.pca['loss'] == 'hinge':
        loss = keras.losses.Hinge()
      elif self.pca['loss'] == 'kld':
        loss = keras.losses.KLDivergence()
      else:
        raise Exception(f"Unknown loss {self.pca['loss']}")
      self.estimator.compile(optimizer=optimiser,
                             loss=loss,
                             metrics=[keras.metrics.binary_accuracy])
      callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                 min_delta=1e-8,
                                                 patience=50,
                                                 mode='min',
                                                 verbose=(self.verbose > 0),
                                                 restore_best_weights=True)]

      # Train
      hist = self.estimator.fit(train_data, validation_data=test_data, epochs=self.pca['epochs'],
                                callbacks=callbacks, verbose=(self.verbose > 0)*2)
      history.append(hist.history)

      if self.pca['ft_layers'] != 0:
        # Fine-tune
        if self.verbose > 0:
          if self.pca['ft_layers'] < 0:
            print("### Finetune all layers of CNNs")
          else:
            print(f"### Finetune last {self.pca['ft_layers']} layers of CNNs")
        if self.pca['ft_layers'] < 0:
          for cnn in cnns:
            cnn.trainable = True # fine tune entire model
            for l in cnn.layers:
              if isinstance(l,keras.layers.BatchNormalization):
                # Don't fine-tune batch normalisation
                l.trainable = False
        else:
          for cnn in cnns:
            for n, l in enumerate(cnn.layers):
              if isinstance(l,keras.layers.BatchNormalization):
                # Don't fine-tune batch normalisation
                l.trainable = False
              elif n >= len(cnn.layers) - self.pca['ft_layers']: # Fine-tune last layers
                l.trainable = True
              else:
                l.trainable = False
        optimiser = keras.optimizers.Adam(learning_rate=learning_rate/10.0,
                                          beta_1=Cfg.val['beta1'],
                                          beta_2=Cfg.val['beta2'],
                                          epsilon=Cfg.val['epsilon'])
        if self.pca['loss'] == 'bcl':
          loss = keras.losses.BinaryCrossentropy(from_logits=True)
        elif self.pca['loss'] == 'bc':
          loss = keras.losses.BinaryCrossentropy(from_logits=False)
        elif self.pca['loss'] == 'bfcl':
          loss = keras.losses.BinaryFocalCrossentropy(from_logits=True)
        elif self.pca['loss'] == 'bfc':
          loss = keras.losses.BinaryFocalCrossentropy(from_logits=False)
        elif self.pca['loss'] == 'hinge':
          loss = keras.losses.Hinge()
        elif self.pca['loss'] == 'kld':
          loss = keras.losses.KLDivergence()
        else:
          raise Exception(f"Unknown loss {self.pca['loss']}")
        self.estimator.compile(optimizer=optimiser,
                               loss=loss,
                               metrics=[keras.metrics.binary_accuracy])
        callbacks = [keras.callbacks.EarlyStopping(monitor='loss',
                                                   min_delta=1e-8,
                                                   patience=50,
                                                   mode='min',
                                                   verbose=(self.verbose > 0),
                                                   restore_best_weights=True)]
        hist = self.estimator.fit(train_data, validation_data=test_data, epochs=self.pca['ft_epochs'],
                                  verbose=(self.verbose > 0)*2, callbacks=callbacks)
        finetune_history.append(hist.history)

      # Collect predictions on train and test data
      train_data.disable_shuffle()
      train_score = self.estimator.predict(train_data, verbose=(self.verbose>0)*2)
      test_data.disable_shuffle()
      test_score = self.estimator.predict(test_data, verbose=(self.verbose>0)*2)

      # Logits need sigmoid 
      if self.pca['loss'] == 'bclc' or self.pca['loss'] == 'bfclc':
        # Logits with calibration

        # Data
        test_true_y = np.expand_dims(test_data.get_Y(),axis=-1).astype(np.float32)
        train_true_y = np.expand_dims(train_data.get_Y(),axis=-1).astype(np.float32)
        train_score_pre = tf.nn.sigmoid(train_score)
        test_score_pre = tf.nn.sigmoid(test_score)
        # Calibration dataset (not used for anything else; selected above from train dataset)
        cal_data = SeqClassifyFiles([X[k] for k in cal_idx],
                                    [Y[k] for k in cal_idx],
                                    self.dataset, self.tags,
                                    batch_size=self.pca['batch_size'],
                                    size=self.pca['patch_size'],
                                    shuffle=False, bits=8)
        cal_true_y = np.expand_dims(cal_data.get_Y(),axis=-1).astype(np.float32)
        cal_score = self.estimator.predict(cal_data, verbose=(self.verbose>0)*2)
        cal_score_pre = tf.nn.sigmoid(cal_score)

        # Reliability before calibration
        test_ece_pre, test_acc_pre, test_conf_pre = PCNNClassifier._ece(test_score_pre, test_true_y)
        cal_ece_pre, cal_acc_pre, cal_conf_pre = PCNNClassifier._ece(cal_score_pre, cal_true_y)
        if self.verbose > 0:
          print("# Calibration")
          print(f"  ECE before calibration: test - {test_ece_pre}; cal - {cal_ece_pre}")

        # Calibrate
        temp, hist = PCNNClassifier._temperature_calibration(train_score_pre, train_true_y,
                                                             cal_score_pre, cal_true_y,
                                                             test_score_pre, test_true_y)
        self.pca['temperature'] = np.float64(temp)

        # Reliability after calibration
        train_score = tf.nn.sigmoid(train_score / temp)
        test_score = tf.nn.sigmoid(test_score / temp)
        cal_score = tf.nn.sigmoid(cal_score / temp)
        test_ece_post, test_acc_post, test_conf_post = PCNNClassifier._ece(test_score, test_true_y)
        cal_ece_post, cal_acc_post, cal_conf_post = PCNNClassifier._ece(cal_score, cal_true_y)
        if self.verbose > 0:
          print(f"  Calilbration temperature: {temp}")
          print(f"  ECE after calibration: test - {test_ece_post}; cal - {cal_ece_post}")

        # Reliability diagrams
        fig, ax = plt.subplots(1,3)
        ax[0].plot([0,1],[0,1],linestyle='--')
        ax[0].plot(test_conf_pre, test_acc_pre, marker='.', label=f'before calibration, ECE: {test_ece_pre}')
        ax[0].plot(test_conf_post, test_acc_post, marker='.', label=f'after calibration, ECE: {test_ece_post}')
        ax[0].legend()
        ax[0].set_title('Test Dataset Reliability Diagram')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_xlabel('Confidence')

        ax[1].plot([0,1],[0,1],linestyle='--')
        ax[1].plot(cal_conf_pre, cal_acc_pre, marker='.', label=f'before calibration, ECE: {cal_ece_pre}')
        ax[1].plot(cal_conf_post, cal_acc_post, marker='.', label=f'after calibration, ECE: {cal_ece_post}')
        ax[1].legend()
        ax[1].set_title('Calibration Dataset Reliability Diagram')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('Confidence')

        ax[2].plot(hist[:,0], label='training calibration loss')
        ax[2].plot(hist[:,1], label='validation calibration loss')
        ax[2].plot(hist[:,2], label='test calibration loss')
        ax2 = ax[2].twinx()
        ax2.plot(hist[:,3], label='calibration temperature', color='red')
        ax[2].legend()
        ax[2].set_title('Calibration History')
        ax[2].set_ylabel('Calibration Cost')
        ax2.set_ylabel('Temperature')
        ax[2].set_xlabel('Calibration Epoch')

        for dpi in Cfg.val['image_dpi']:
          plt.savefig(os.path.join(self.model,f"calibration_{fold}@{dpi}.png"), dpi=dpi)
        if self.verbose > 1:
          fig.set_dpi(Cfg.val['screen_dpi'])
          plt.show(block=True)
        plt.close()

      elif self.pca['loss'] == 'bcl' or self.pca['loss'] == 'bfcl':
        # Logits without calibration

        train_score = tf.nn.sigmoid(train_score)
        test_score = tf.nn.sigmoid(test_score)

      # Predictions on Train
      train_pred = tf.where(train_score < 0.5, 0, 1)
      train_Y = train_data.get_Y()
      train_P = train_data.get_P()
      fpr_train, tpr_train, thres_train = roc_curve(train_Y, train_score)
      auc_train = roc_auc_score(train_Y, train_score)

      # Predictions on Test
      test_pred = tf.where(test_score < 0.5, 0, 1)
      test_Y = test_data.get_Y()
      test_P = test_data.get_P()
      fpr_test, tpr_test, thres_test = roc_curve(test_Y, test_score)
      auc_test = roc_auc_score(test_Y, test_score)
      predictions["train"].append((train_pred, train_Y, train_P,
                                   (fpr_train, tpr_train, thres_train, auc_train)))
      predictions["test"].append((test_pred, test_Y, test_P,
                                  (fpr_test, tpr_test, thres_test, auc_test)))
      if len(self.model) > 0:
        # Save model
        self.estimator.save(os.path.join(self.model,f"pcnn_tf_{fold}"))
        with open(os.path.join(self.model, f"pcnn_tf_{fold}" , "pcanet.json"), 'w') as f:
          print(json.dumps(self.pca, indent=2, sort_keys=True), file=f)

    if len(self.model) > 0:
      # Histories plot
      fig, axes = plt.subplots(k_folds,2)
      for fold in range(0,k_folds):
        x = np.arange(0,len(history[fold]['binary_accuracy']))
        axes[fold,0].plot(x, history[fold]['binary_accuracy'],
                          label=str(f"Accuracy; best: {np.max(history[fold]['binary_accuracy']):.4f}"))
        axes[fold,0].plot(x, history[fold]['val_binary_accuracy'],
                          label=str(f"Val. Accuracy; best: {np.max(history[fold]['val_binary_accuracy']):.4f}"))
        axes[fold,1].plot(x, history[fold]['loss'],
                          label=str(f"Loss; best: {np.min(history[fold]['loss']):.4f}"))
        axes[fold,1].plot(x, history[fold]['val_loss'],
                          label=str(f"Val. Loss; best: {np.min(history[fold]['val_loss']):.4f}"))
        if len(finetune_history) > 0:
          x = len(history[fold]['binary_accuracy']) + np.arange(0,len(finetune_history[fold]['binary_accuracy']))
          axes[fold,0].plot(x, finetune_history[fold]['binary_accuracy'],
                            label=str(f"FT Accuracy; best: {np.max(finetune_history[fold]['binary_accuracy']):.4f}"))
          axes[fold,0].plot(x, finetune_history[fold]['val_binary_accuracy'],
                            label=str(f"FT Val. Accuracy; best: {np.max(finetune_history[fold]['val_binary_accuracy']):.4f}"))
          axes[fold,1].plot(x, finetune_history[fold]['loss'],
                            label=str(f"FT Loss; best: {np.min(finetune_history[fold]['loss']):.4f}"))
          axes[fold,1].plot(x, finetune_history[fold]['val_loss'],
                            label=str(f"FT Val. Loss; best: {np.min(finetune_history[fold]['val_loss']):.4f}"))
        axes[fold,0].set_ylabel(f"Fold {fold}\n\nAccuracy")
        axes[fold,0].legend(loc='lower right')
        if self.pca['loss'] == 'bcl' or self.pca['loss'] == 'bc':
          axes[fold,1].set_ylabel("Binary Crossentropy")
        elif self.pca['loss'] == 'bfcl' or self.pca['loss'] == 'bfc':
          axes[fold,1].set_ylabel("Binary Focal Crossentropy")
        elif self.pca['loss'] == 'hinge':
          axes[fold,1].set_ylabel("Hinge Loss")
        elif self.pca['loss'] == 'kld':
          axes[fold,1].set_ylabel("KLD Loss")
        else:
          raise Exception(f"Unknown loss {self.pca['loss']}")
        axes[fold,1].legend(loc='upper right')

      for dpi in Cfg.val['image_dpi']:
        plt.savefig(os.path.join(self.model,f"history@{dpi}.png"), dpi=dpi)
      if self.verbose > 1:
        fig.set_dpi(Cfg.val['screen_dpi'])
        plt.show(block=True)
      plt.close()

    return predictions

  def predict(self, X, Y=None, P=None, evaluate=True, parallel=True):
    # Predict only
    if self.verbose > 0:
      print(f"# Predicting classes of {len(X)} samples")
    data = SeqClassifyFiles(X, Y, self.dataset, self.tags,
                            batch_size = self.pca['batch_size'],
                            size = self.pca['patch_size'],
                            shuffle=False, bits=8)
    score = self.estimator.predict(data, verbose=(self.verbose>0)*2)
    if self.pca['loss'] == 'bcl' or self.pca['loss'] == 'bfcl':
      score = tf.nn.sigmoid(score)
    elif self.pca['loss'] == 'bclc' or self.pca['loss'] == 'bfclc':
      score = tf.nn.sigmoid(score/self.pca['temperature'])
    pred = tf.where(score < 0.5, 0, 1)
    P = data.get_P()
    if Y is not None and evaluate:
      data_Y = data.get_Y()
      fpr, tpr, thres = roc_curve(data_Y, score)
      auc = roc_auc_score(data_Y, score)
      roc_data = (fpr,tpr,thres,auc)
    else:
      roc_data = None
      data_Y = Y
    return { "data": [(pred,data_Y,P,roc_data)] }

  @staticmethod
  def _ece(score, true_y):
    # Expected calibration error
    # See https://towardsdatascience.com/neural-network-calibration-using-pytorch-c44b7221a61
    n_bins = 10
    bins = []
    for l in range(0,n_bins):
      bins.append([])
    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    ece = 0
    # Histogram
    for l in range(0,score.shape[0]):
      bins[int(np.clip(score[l,0].numpy()*n_bins, 0, n_bins-1))].append(l)
    # Compute accuracy, confidence and ece (yes, there is a faster way)
    b = 0
    for l in range(0,n_bins):
      if len(bins[l]) > 0:
        for k in bins[l]:
          p = np.clip(score[k,0].numpy(), 0.0, 1.0)
          p_y = int(p+0.5)
          t_y = int(true_y[k,0])
          if  p_y == t_y:
            acc[b] += 1
          if t_y == 1:
            conf[b] += p
          else:
            conf[b] += 1-p
        acc[b] /= len(bins[l])
        conf[b] /= len(bins[l])
        ece += len(bins[l]) * np.abs(acc[b] - conf[b])
        b += 1
    ece /= score.shape[0]
    idx = np.argsort(conf[0:b],axis=0)
    return ece, acc[idx], conf[idx]

  @staticmethod
  def _temperature_calibration(train_x, train_y, cal_x, cal_y, test_x, test_y, epochs=100000):
    # Temperature calibration
    # See https://github.com/stellargraph/stellargraph/blob/develop/stellargraph/calibration.py

    def loss(T, x, y):
      return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.multiply(x,1.0/T),
                                                                    labels=y))
     

    def grad(T, x, y):
      with tf.GradientTape() as tape:
        l = loss(T, x, y)
      return l, tape.gradient(l, T)

    T = tf.Variable(tf.ones(shape=(1,)))

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    history = []
    for epoch in range(epochs):
      train_loss, grads = grad(T, train_x, train_y)
      optimizer.apply_gradients(zip([grads], [T]))
      cal_loss = loss(T, cal_x, cal_y)
      test_loss = loss(T, test_x, test_y)
      if len(history) > 0 and cal_loss > history[-1][1]:
        break
      history.append([train_loss, cal_loss, test_loss, T.numpy()[0]])

    history = np.asarray(history)
    return history[-1, -1], history

  @staticmethod
  def load(model, fold=0, verbose=0):
    fn = os.path.join(model,f"pcnn_tf_{fold}")
    if not os.path.exists(fn):
      raise Exception(f"Cannot load {fn}")
    if verbose > 0:
      print(f"# Loading PCNNClassifier: {model}")
    classifier = load_model(fn)
    with open(os.path.join(fn, "pcanet.json"), 'r') as f:
      pca = json.load(f)
    if pca['classifier'] != 'PCNNClassifier':
      raise Exception("Not a PCNNClassifier")
    return PCNNClassifier(model=model, type=pca['type'],
                          fc=pca['fc'],
                          do=pca['do'],
                          rgb_train=pca['rgb_train'],
                          merge=pca['merge'],
                          ft_layers=pca['ft_layers'],
                          batch_size=pca['batch_size'],
                          epochs=pca['epochs'],
                          loss=pca['loss'],
                          ft_epochs=pca['ft_epochs'],
                          size=pca['patch_size'],
                          channels=pca['channels'],
                          data=pca['data'],
                          temperature=pca['temperature'],
                          classifier=classifier,
                          verbose=verbose)

class WeighFeatures(Layer):
  # Keras layer - multiply each element of the tensor with a trainable factor

  def __init__(self, **kwargs):
    super(WeighFeatures, self).__init__(**kwargs)

  def build(self, input_shape):
    self.input_spec = InputSpec(ndim=len(input_shape), axes=dict(list(enumerate(input_shape[1:], start=1))))
    self.factors = self.add_weight(name='kernel', shape=input_shape[1:], initializer='ones', trainable=True)
    super(WeighFeatures, self).build(input_shape)

  def call(self, x):
    return tf.math.multiply(x, self.factors)

  def compute_output_shape(self, input_shape):
    return (input_shape)
