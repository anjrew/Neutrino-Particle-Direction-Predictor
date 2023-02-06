conda create -n KAG_IC_NEU python=3.8
conda activate KAG_IC_NEU
pip install -r ./requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
