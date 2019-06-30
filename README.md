# ganabi

**For the baseline project, use Python 2.7 and Tensorflow 1.14.**

### Getting Started:
```
fork/clone repo into your home folder
cd ~/ganabi/hanabi-env
cmake .
make
cd ..
source /data1/shared/venvg2/bin/activate # use venvg for python 3 
mkdir data # FIXME: should get created if it doesn't exist
python create_data.py
```

## Phase One

### Description
In the first project, we will be creating a baseline agent that is aimed to be simple but representative. The model will be limited to using only MLP, and we will try to obtain better result by applying fundamental machine learning techniques.

### Run the project
```bash
% Latest code at branch james_setup

python run_experiment.py -newrun --mode="naive_mlp" --configpath="./naive_mlp.config.gin"
```

### Learning Focus
Here is a list of techniques that we might want to get familiar with. Feel free to add more
- Tensorflow 2.0 framework
- Keras API
- Tensorflow Dataset API
- Cross Validation
- Regularization
- Effective Matplotlib Data Visualization

### Contributor
Chu-Hung Cheng
Soumil Shekdar