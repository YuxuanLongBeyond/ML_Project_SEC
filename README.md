# ML Project Road Segmentation

To make sure that the software requirements are satisfied, under the top directory in the project, please run:  
`pip install -r requirements.txt`  
Please also make sure to run the scripts in Python3. It is suggested to run in GPU environment, nevertheless running in CPU environment is also allowed.  

## A Quick Start
To test our model, please first download the trained weights in:  
https://drive.google.com/drive/folders/1uqSavfAJ8LMrAo178em-3EKrhEFNCfH1?usp=sharing  
There are three weights in this drive: 'weights0' for LinkNet, 'weights1' for D-LinkNet, 'weights2' for D-LinkNet+. Note that these weights are all reproducible.  
For convenience, please only download 'weights2' and put it into folder 'parameters'. Downloading all the weights and putting them into folder 'parameters' can enable test in ensemble mode (not recommended).  
The second step is to put the training and test data into folder 'data', i.e. data/training and data/test_set_images.  
The directory structure should now become:  
```
ML_Project_SEC
│   ...                   // Python scripts
│
└───data                  // store training and test data
│   │
│   └───training
│       │   
│       └───groundtruth
│       │   ...
│       └───images
│       │   ...
│       
│   └───test_set_images
│       │   ...
│       
└───parameters            // stored trained weights
    │   ...
```

Finally, as the parameters in 'run.py' are already set by default to enable test, please **directly run** 'run.py' to reproduce the official submission file (the final project submission), where the F1 score should be 0.921. 'submission.csv' will be produced, and a new folder 'output' will be created to store the predictions (i.e. binary masks).  
The best result can be reproduced by downloading the best weights (only for D-LinkNet+) from:  
https://drive.google.com/drive/folders/1PvoWi6j6ZpECoewhw__p8qnkNGVGs5hf?usp=sharing  
Note this best weights are not reproducible from training, since its random seeds are unknown unfortunately. To test it, please replace this weights in folder 'parameters', and run 'run.py' again, so that F1 score of 0.922 can be reached.  

## Training
To reproduce the trained weights, here we introduce several parameters in 'run.py':  
`train_flag`: the boolean flag to enable training, by default it is set to False.  
`test_flag`: the boolean flag to enable test, by default it is set to True.  
`model_choice`: if `model_choice = 2` (by default), D-LinkNet+ is chosen; if `model_choice = 1`, D-LinkNet is chosen; if `model_choice = 0`, LinkNet is chosen.  
To reproduce the weights trained on D-LinkNet+ (we use them to make final submission), please **only** change `train_flag` and set it as `train_flag = True`. To reproduce other weights, please modify the value of `model_choice`. It is not necessary to change any other training parameters, since they are already to default values.  
After training, the weights will be saved in folder 'parameters'. Please be **careful** that the weights can be overwritten.
