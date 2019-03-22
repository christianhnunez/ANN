# ANN

Zihao's ANN Code; It has API's which can be readily used
Required packages: keras, numpy, scipy, root_numpy, ROOT


## Installation Method with conda (Linux, suggested)

install conda, follow instruction closely and make sure you have a few GB space available:
```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Open a new terminal and create an virtual environment:
```sh
conda create -n myrootenv python=3.7 root -c conda-forge
conda activate myrootenv
conda config --env --add channels conda-forge
```

Install all packages under conda:
```sh
pip install numpy
pip install scipy
pip install sklearn
pip install keras
pip install theano
pip install tensorflow
pip install root_numpy
```

The method has been tested on lxplus. After installation, each time opening a new window, do:
```sh
conda activate myrootenv
```

## Testing installation (on lxplus)
```sh
python test.py
```

The output should be something like:
```sh
[ 438607.6   159969.81  160083.84 ... 1384873.9  1270115.9  1194152.6 ]
<keras.layers.core.Dense object at 0x7f3ba6843630>
```

If the test fails complaining errors about tensorflow, try to switch from tensorflow to theano by changing the backend in the keras.json file
https://keras.io/backend/#switching-from-one-backend-to-another