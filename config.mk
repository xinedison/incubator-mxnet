#-------------------------------------------------------------------------------
#  Template configuration for compiling mxnet
#
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory of mxnet. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ cp make/config.mk .
#
#  Next modify the according entries, and then compile by
#
#  $ make
#
#  or build in parallel with 8 threads
#
#  $ make -j8
#-------------------------------------------------------------------------------

#---------------------
# choice of compiler
#--------------------

export CC = gcc
export CXX = g++
export NVCC = nvcc arch=sm_50

# whether compile with debug
DEBUG = 0

# the additional link flags you want to add
ADD_LDFLAGS = -L/usr/local/cudnn-v5/lib64 -Wl,-R'/usr/local/cuda-8.0.61/lib64',-R'/usr/local/cudnn-v5/lib64',-R'/usr/local/opencv-2.4.13/lib' 

# the additional compile flags you want to add
ADD_CFLAGS = -I/usr/local/cuda-8.0.61/include -I/usr/local/cudnn-v5/include -I/usr/include/openblas 

#---------------------------------------------
# matrix computation libraries for CPU/GPU
#---------------------------------------------

# whether use CUDA during compile
USE_CUDA = 1

# add the path to CUDA libary to link and compile flag
# if you have already add them to enviroment variable, leave it as NONE
# USE_CUDA_PATH = /usr/local/cuda
USE_CUDA_PATH = /usr/local/cuda-8.0.61

# whether use CUDNN R3 library
USE_CUDNN = 1

USE_CUDNN_PATH = /usr/local/cudnn-v5/
 
# whether use opencv during compilation
# you can disable it, however, you will not able to use
# imbin iterator
USE_OPENCV = 1

# use openmp for parallelization
USE_OPENMP = 1

# choose the version of blas you want to use
# can be: mkl, blas, atlas, openblas
USE_STATIC_MKL = NONE
USE_BLAS = openblas

# add path to intel libary, you may need it for MKL, if you did not add the path
# to enviroment variable
USE_INTEL_PATH = NONE

# If use MKL, choose static link automaticly to allow python wrapper
ifeq ($(USE_BLAS), mkl)
	USE_STATIC_MKL = 1
endif

#----------------------------
# distributed computing
#----------------------------

# whether or not to enable mullti-machine supporting
USE_DIST_KVSTORE = 1

# whether or not allow to read and write HDFS directly. If yes, then hadoop is
# required
USE_HDFS = 0

# path to libjvm.so. required if USE_HDFS=1
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

# whether or not allow to read and write AWS S3 directly. If yes, then
# libcurl4-openssl-dev is required, it can be installed on Ubuntu by
# sudo apt-get install -y libcurl4-openssl-dev
USE_S3 = 0
