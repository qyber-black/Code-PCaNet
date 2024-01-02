# PCaNet

> SPDX-FileCopyrightText: Copyright (C) 2021-2023 Asmail Muftah <MuftahA@cardiff.ac.uk>, PhD student at Cardiff University
> SPDX-FileCopyrightText: Copyright (C) 2020-2023 Frank C Langbein <frank@langbein.org>, Cardiff University  
> SPDX-FileCopyrightText: Copyright (C) 2020-2022 S Shermer <lw1660@gmail.com>, Swansea University
> SPDX-License-Identifier: AGPL-3.0-or-later

Code for classification and segmentation of prostate cancer.

The main functions are in the pcanet module. We provide jupyter notebook and
command line interfaces, as needed/used.

# Install Instructions (Linux)

These instructions are for Linux. Note that the code has only been tested on
Linux and may not work on another platform.

1. Clone the repository:
   ```
   git clone git@qyber.black:ca/pca/code-pcanet.git pcanet
   ```
   Check the clone url, as it may be different if you use a different
   repository, e.g. from a mirror or alternative versions for development, etc.
2. Navigate to the directory:
   ```
   cd pcanet
   ```
   Make sure to select a branch or tag with `git checkout BRANCH_OR_TAG` for a
   specific version instead of the main branch.
3. Update submodules:
   ```
   git submodule update --init --recursive
   ```
4. Install the requirements:
   ```
   pip3 install -r requirements.txt
   ```
   Note that the requirements may need additional libraries, etc. to be
   installed on you system that pip does not add automatically. Also note,
   that some packages (in particular mahotas), may not be available at
   the requested version if you are use a more recent version of python
   (python3.10) - these pacakges can still be installed from source; obtain
   the sources and checkout the version requested and run `python3
   setup.py install --prefix=~/.local` or similar.

# Command Line Interface

* `pca.py` is the command line interface to the code for some functionalities. See
  `--help` option for further information.
  * `augment` sub-command creates augmented datasets for classification and segmentation.
    Classification includes extracting patches cropped to a rectangular region.
    Default augmentation options are set in pcanet/augment.py for the two modes. The
    original sample is always included. If the augmentation factor is 1, then only the
    original sample is included.
  * `classify` trains various classifiers on the datasets. If the model specified already
     exists it only evaluates it on the dataset given.
  * `predict` runs a given classifier on a dataset to predict results.
  * `view` shows augmentation dataset content.

# Jupyter Notebooks

None directly included in the package, but see submodules that may have some to analyse
the results, etc.
