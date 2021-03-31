.. image:: https://rieslab.de/assets/img/decode_microtubules.png

DECODE Manual
=============
This is the documentation of the DECODE Deep Learning for Superresolution Localization Microscopy.

..
   _THIS IS A COPY OF THE README.MD PART. DO NOT EDIT THIS DIRECTLY. EDIT README AND COPY

DECODE is a Python and `Pytorch <http://pytorch.org/>`__ based deep
learning tool for single molecule localization microscopy (SMLM). It has
high accuracy for a large range of imaging modalities and conditions. On
the public `SMLM 2016 <http://bigwww.epfl.ch/smlm/challenge2016/>`__
software benchmark competition, it
`outperformed <http://bigwww.epfl.ch/smlm/challenge2016/leaderboard.html>`__
all other fitters on 12 out of 12 data-sets when comparing both
detection accuracy and localization error, often by a substantial
margin. DECODE enables live-cell SMLM data with reduced light exposure
in just 3 seconds and to image microtubules at ultra-high labeling
density.

DECODE works by training a DEep COntext DEpendent (DECODE) neural
network to detect and localize emitters at sub-pixel resolution.
Notably, DECODE also predict detection and localization uncertainties,
which can be used to generate superior super-resolution reconstructions.

Get Started
###########
To try out DECODE we recommend to first have a look at the Google Colab notebooks.

DECODE on Google Colab
""""""""""""""""""""""
Our notebooks below comprise training a model, fitting experimental data and exporting the fitted localizations.

* `Training a DECODE model <https://colab.research.google.com/drive/1uQ7w1zaqpy9EIjUdaLyte99FJIhJ6N8E?usp=sharing>`_
* `Fitting high-density data <https://colab.research.google.com/drive/1HAvJUL29vVuCHMZHMbU9jxd4fbLIPdhZ?usp=sharing>`_

DECODE on your machine
""""""""""""""""""""""
The installation is described in detail here `installation instructions. <installation.html>`__

Once you have installed DECODE on your local machine, please follow our
`Tutorial. <tutorial.html>`__

Video tutorial
###############
As part of the virtual `I2K 2020 <https://www.janelia.org/you-janelia/conferences/from-images-to-knowledge-with-imagej-friends>`__ conference we organized a workshop on DECODE. Please find the video below.

*DECODE is being actively developed, therefore the exact commands might differ from those shown in the video.*

.. raw:: html

   <p style="text-align:center"><iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/zoWsj3FCUJs" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></p>


Contents
########

.. toctree::
   :maxdepth: 0

   installation
   tutorial
   data
   logging
   faq

.. toctree::
   :maxdepth: 1
   :caption: DECODE API

   decode
