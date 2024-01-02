# pcnn.py - Pretrained CNN architectures
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
# SPDX-FileCopyrightText: Copyright (C) 2021-2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

def get_pcnn(model, size):
  import tensorflow.keras as keras
  model = model.lower()
  if model == "vgg16":
    return keras.applications.VGG16(input_shape=(size,size,3),
                                    include_top=False,
                                    weights='imagenet'), \
           keras.applications.vgg16.preprocess_input
  elif model == "resnet50":
    return keras.applications.ResNet50(input_shape=(size,size,3),
                                       include_top=False,
                                       weights='imagenet'), \
           keras.applications.resnet50.preprocess_input
  elif model == "inceptionv3":
    return keras.applications.InceptionV3(input_shape=(size,size,3),
                                          include_top=(size,size,3),
                                          weights='imagenet'), \
           keras.applications.inception_v3.preprocess_input
  elif model == "mobilenetv2":
    return keras.applications.MobileNetV2(input_shape=(size,size,3),
                                          include_top=False,
                                          weights='imagenet'), \
           keras.applications.mobilenet_v2.preprocess_input
  elif model == "efficientnetv2s":
    return keras.applications.EfficientNetV2S(input_shape=(size,size,3),
                                              include_top=False,
                                              weights='imagenet'), \
           keras.applications.efficientnet_v2.preprocess_input
  elif model == "efficientnetv2m":
    return keras.applications.EfficientNetV2M(input_shape=(size,size,3),
                                              include_top=False,
                                              weights='imagenet'), \
           keras.applications.efficientnet_v2.preprocess_input
  elif model == "efficientnetv2l":
    return keras.applications.EfficientNetV2L(input_shape=(size,size,3),
                                              include_top=False,
                                              weights='imagenet'), \
           keras.applications.efficientnet_v2.preprocess_input
  raise Exception(f"Unknown network {model}")

def get_pcnn_names():
  return ["VGG16", "InceptionV3", "ResNet50", "MobileNetV2", "EfficientNetV2S",
          "EfficientNetV2M", "EfficientNetV2L"]
