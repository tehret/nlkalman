IMPLEMENTATION OF THE VIDEO DENOISING ALGORITHM NON-LOCAL KALMAN
================================================================

* Author    : EHRET Thibaud <ehret.thibaud@gmail.com>
* Copyright : (C) 2019 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+, see gpl.txt

OVERVIEW
--------

This source code provides an implementation of NL-Kalman developped in "Thibaud Ehret,
Jean-Michel Morel, Pablo Arias, Non-Local Kalman: a recursive video denoising algorithm
ICIP 2018". Video examples are available on the 
[webpage of the article](https://tehret.github.io/nlkalman)

In order to reproduce the results from the ICIP article a post-processing step consisting
of applying [DCT denoising](http://www.ipol.im/pub/art/2017/201/) is necessary.

The [DCT denoising](http://www.ipol.im/pub/art/2017/201/) and the [TVL1 optical flow](http://www.ipol.im/pub/art/2013/26/) codes are provided inside this repo in order to centralize
all code necessary to reproduce the results. Shell script linking the different parts are 
also provided.

This code is part of an [IPOL](http://www.ipol.im/) publication. Plase cite it
if you use this code as part of your research. (The article is not already published 
at this time)

COMPILATION
-----------

The code is compilable on Unix/Linux and hopefully on Mac OS (not tested!). 

**Compilation:** requires the cmake and make programs.

**Dependencies:** BLAS, LAPACK and OpenMP [optional]. 
For image i/o we use [Enric Meinhardt's iio](https://github.com/mnhrdt/iio),
which requires libpng, libtiff and libjpeg.
 
Compile the source code using make.

UNIX/LINUX/MAC:
```
$ mkdir build; cd build
$ cmake ..
$ make
```

Binaries will be created in `build/bin folder`.

NOTE: By default, the code is compiled with OpenMP multithreaded
parallelization enabled (if your system supports it). 
The number of threads used by the code is defined in `nlkalman/nlkalman.h`.

USAGE
-----

The following commands have to be run from the current folder:

List all available options:</br>
```
$ ./nlkalman --help
```

While being a video denoising algorithm, the method takes as input the frames of the video 
and not an actual video. The frames can be extracted using ffmpeg on linux. For example: 
```
$ ffmpeg -i video.mp4 video/i%04d.png
```

There is five mandatory input arguments:
* `-i` the input sequence
* `-of` the optical flow corresponding to the input sequence (backward)
* `-f` the index of the first frame
* `-l` the index of the last frame
* `-sigma` the standard deviation of the noise

When providing a sequence that is already noisy the option `-add_noise` should be set to false.

All path should be given using the C standard. For example to reference to the following video:
* video/i0000.png
* video/i0001.png
* video/i0002.png
* video/i0003.png
* video/i0004.png
* video/i0005.png
* video/i0006.png
* video/i0007.png
* video/i0008.png
* video/i0009.png

The command for denoising with a noise standard deviation of 20 should be 
```
$ ./nlkalman -i video/i%04d.png -f 0 -l 9 -sigma 20 -of video/flow_%04d.flo
```

-----

FILES
-----

This project contains the following source files:
```
    main function:               src/main_nlkalman.cpp
    command line parsing:        src/cmd_option.h
    nlkalman implementation:     src/nlkalman/nlkalman.h
                                 src/nlkalman/nlkalman.cpp
    parameters container:        src/nlkalman/nlkParams.h
    Matrix operations:           src/nlkalman/LibMatrix.h
                                 src/nlkalman/LibMatrix.cpp
    image i/o:                   src/nlkalman/iio.h
                                 src/nlkalman/iio.c
    image container:             src/nlkalman/LibImages.h
                                 src/nlkalman/LibImages.cpp
    image container:             src/nlkalman/LibVideoT.hpp
                                 src/nlkalman/LibVideoT.cpp
    random number generator:     src/nlkalman/mt19937ar.h
                                 src/nlkalman/mt19937ar.c
    utilities functions:         src/nlkalman/Utilities.h
                                 src/nlkalman/Utilities.cpp
    Parametric trans. functions: src/nlkalman/parametric_transformation.h
                                 src/nlkalman/parametric_transformation.cpp
                                 src/nlkalman/parametric_utils.h
                                 src/nlkalman/parametric_utils.cpp
```

ABOUT THIS FILE
---------------

Copyright 2019 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.
