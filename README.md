# ganabi

Because DeepMind wrote their Rainbow agents in Py 2.7 and tf 1.x, the data creation script, which interfaces with that code, uses Py 2.7 and tf 1.x. However, once the data is produced, we only use Py 3.6 and tf 2.0 for building and training our models.

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

### Setup
I would highly recommend setting up the aliases to reduce time typing up the commands 

In ~/.bash_aliases
```bash
% Create Data Relies on Python2
alias cganabi='source path-to-virtualenv-python2/bin/activate && cd /path-to-ganabi/ganabi && python create_data.py -newnpy -newrun --mode="naive_mlp" --configpath="./naive_mlp.config.gin"'


% Training Relies on Python3
alias rganabi='source path-to-virtualenv-python3/bin/activate && cd /path-to-ganabi/ganabi && python run_experiment.py 
```

Now you can train by performing
```bash
cganabi
rganabi
```

FYI, structure of ./data after running cganabi looks like the following
```bash
├── test
│   ├── rainbow_agent_1
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_2
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_3
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_4
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_5
│   │   ├── act
│   │   └── obs
│   └── rainbow_agent_6
│       ├── act
│       └── obs
├── train
│   ├── rainbow_agent_1
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_2
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_3
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_4
│   │   ├── act
│   │   └── obs
│   ├── rainbow_agent_5
│   │   ├── act
│   │   └── obs
│   └── rainbow_agent_6
│       ├── act
│       └── obs
└── validation
    ├── rainbow_agent_1
    │   ├── act
    │   └── obs
    ├── rainbow_agent_2
    │   ├── act
    │   └── obs
    ├── rainbow_agent_3
    │   ├── act
    │   └── obs
    ├── rainbow_agent_4
    │   ├── act
    │   └── obs
    ├── rainbow_agent_5
    │   ├── act
    │   └── obs
    └── rainbow_agent_6
        ├── act
        └── obs
```
### Current Issues
1. The backend of the rainbow_agent relies on python2, so its very difficult to integrate the entire data pipeline with the trainer code.
2. Refactoring the code, architechture is a must. Presenting some ideas over here.
    1. In general, the pipeline should be somewhat similar to
    ```bash
    Parse Args -> Resolve Directories -> Parse Gin Configs -> 
    Resolve Dataset -> Resolve DataGenerator -> Start Trainer
    ```
    2. Have a neater argument Manager. Limit the users flexibility for the exchange of more organized code base. Argument such as ckpt_dir can assume to be locate at uniform location 
    3. Break Gin Config into multiple files
    4. Instead of storing trival "utils" files, create a Kit Package where numerous of subkits exist. For instance, some architecture look similar to  
    ```
    Kits
    ├── ArgsKit
    │   ├── ...
    │   └── ...
    ├── DirsKit
    │   ├── ...
    │   └── ...
    ├── DatasetKit
    │   ├── ...
    │   └── ...
    ├── ....
    │   ├── ...
    │   └──...
    ├main.py % entry of the project
    ``` 