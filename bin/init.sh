conda create -n KAG_IC_NEU python=3.8
conda activate KAG_IC_NEU

conda install -c conda-forge cudatoolkit=11.0.0 cudnn=8.0.0

pip install -r ./requirements.txt

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
