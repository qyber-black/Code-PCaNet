# pcanet/__init__.py - PCaNet - init package
#
# SPDX-FileCopyrightText: Copyright (C) 2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later

__all__ = ["augment", "cfg", "classify", "classify_pcnn", "classify_rfc", "classify_rfcsbfs",
           "classify_svm", "classify_svmsbfs", "crop", "dataset", "generator", "pcnn", "texture"]

version_info = (0,0,1)
__version__ = '.'.join(map(str, version_info))
