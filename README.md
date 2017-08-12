# python version
3.6.1

# install package
pip install -r requirements.txt

# create directory
mkdir data
mkdir predict_data
mkdir source
mkdir training_set

# clean data
python clean/training_set.py

# tensorflow
cd predict
python tensor_d3.py