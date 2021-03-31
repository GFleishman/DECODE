# DECODE
[![Gateway Test](https://github.com/TuragaLab/DECODE/actions/workflows/test_gateway.yml/badge.svg)](https://github.com/TuragaLab/DECODE/actions/workflows/test_gateway.yml)
[![Unit Tests](https://github.com/TuragaLab/DECODE/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/TuragaLab/DECODE/actions/workflows/unit_tests.yml)
[![Docs](https://readthedocs.org/projects/decode/badge/?version=master)](https://decode.readthedocs.io/en/master/?badge=master)

DECODE is a Python and [Pytorch](http://pytorch.org/) based deep learning tool for single molecule 
localization microscopy (SMLM). It has high accuracy for a large range of imaging modalities and 
conditions. 
On the public [SMLM 2016](http://bigwww.epfl.ch/smlm/challenge2016/) software benchmark competition,
it [outperformed](http://bigwww.epfl.ch/smlm/challenge2016/leaderboard.html) all other fitters on 
12 out of 12 data-sets when comparing both detection accuracy and localization error, often by a 
substantial margin. DECODE enables live-cell SMLM data with reduced light exposure in just 3 
seconds and to image microtubules at ultra-high labeling density.

DECODE works by training a DEep COntext DEpendent (DECODE) neural network to detect and localize 
emitters at sub-pixel resolution. Notably, DECODE also predict detection and localization 
uncertainties, which can be used to generate superior super-resolution reconstructions.

## Getting started

The easiest way to try out the algorithm is to have a look at the Google Colab Notebooks we provide
for training our algorithm and fitting experimental data. For installation instructions and further 
information please **refer to our** [**docs**](https://decode.readthedocs.io).
You can find these here:

- [Documentation](https://decode.readthedocs.io)
- DECODE Training (**NEW: v0.10**) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uQ7w1zaqpy9EIjUdaLyte99FJIhJ6N8E?usp=sharing)
- DECODE Fitting (**NEW: v0.10**) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HAvJUL29vVuCHMZHMbU9jxd4fbLIPdhZ?usp=sharing)

## Local Installation

Details about the installation can be found in the [documentation](https://decode.readthedocs.io).

## Video Tutorial
As part of the virtual [I2K 2020](https://www.janelia.org/you-janelia/conferences/from-images-to-knowledge-with-imagej-friends) conference we organized a workshop on DECODE.
Please find the video below.
*DECODE is being actively developed, therefore the exact commands might differ from those shown in the video.*

[![DECODE Video Tutorial](https://img.youtube.com/vi/zoWsj3FCUJs/0.jpg)](https://www.youtube.com/watch?v=zoWsj3FCUJs)

## Paper
This is the *official* implementation of the [preprint](https://www.biorxiv.org/content/10.1101/2020.10.26.355164v1).

Artur Speiser*, Lucas-Raphael Müller*, Ulf Matti, Christopher J. Obara, Wesley R. Legant, Anna Kreshuk, Jakob H. Macke†, Jonas Ries†, and Srinivas C. Turaga†, **Deep learning enables fast and dense single-molecule localization with high accuracy**, [*bioRxiv* 2020.10.26.355164](https://www.biorxiv.org/content/10.1101/2020.10.26.355164v1).

### Data availability
The data referred to in our paper can be accessed at the following locations:
- Fig 3: Can be downloaded from the SMLM 2016 challenge [website](http://bigwww.epfl.ch/smlm/challenge2016/)
- Fig 4: [here](https://oc.embl.de/index.php/s/SFM6Pc8RetX09pJ)
- Fig 5: By request from the authors Wesley R Legant, Lin Shao, Jonathan B Grimm, Timothy A Brown, Daniel E Milkie, Brian B Avants, Luke D Lavis & Eric Betzig, [**High-density three-dimensional localization microscopy across large volumes**](https://www.nature.com/articles/nmeth.3797), _Nature Methods_, *13*, pages 359–365 (2016).

## Contributors
If you want to get in touch, the best way to get your questions answered is our [**GitHub discussions page**](https://github.com/TuragaLab/DECODE/discussions)
- Artur Speiser ([@aspeiser](https://github.com/ASpeiser), arturspeiser@gmail.com)
- Lucas-Raphael Müller ([@haydnspass](https://github.com/Haydnspass), lucasraphaelmueller@gmail.com)

### Acknowledgements
- Don Olbris ([@olbris](https://github.com/olbris), olbrisd@janelia.hhmi.org) for help with python packaging.
