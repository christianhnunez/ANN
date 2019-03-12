# ANN

Zihao's ANN Code; It has API's which can be readily used
Required packages: keras, numpy, scipy, root_numpy, ROOT


## Suggested Pakcage Installation Method (Linux)

* ROOT installation

follow closely https://root.cern.ch/building-root; Suggested to use ROOT 5.34/36


* other package installation

install conda:

```sh
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
chmod +777 Miniconda2-latest-Linux-x86_64.sh
./Miniconda2-latest-Linux-x86_64.sh
```

create an virtual environment

```sh
conda create -n envname python
```

activate the virtual environment

```sh
conda activate envname
```

install packages within the virtual environment (suggested to use python2.7)

```sh
python2.7 -m pip install numpy
python2.7 -m pip install scipy
python2.7 -m pip install sklearn
python2.7 -m pip install keras
python2.7 -m pip install tensorflow
python2.7 -m pip install root_numpy
```

