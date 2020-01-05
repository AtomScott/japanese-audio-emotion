export PYTHONPATH="/usr/local/lib/python3.5/dist-packages/dlib-19.17.99-py3.5-linux-x86_64.egg:$PYTHONPATH"

export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_PATH/include:$CPATH
export LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
source jvaer-venv/bin/activate
