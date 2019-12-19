# ML Project Road Segmentation

To make sure that the software requirements are satisfied, under the top directory in the project, please run:  
`pip install -r requirements.txt`  
Please also make sure to run the scripts in Python3.  

## A Quick Start
To test our model, please first download the trained weights in:  
https://drive.google.com/drive/folders/1uqSavfAJ8LMrAo178em-3EKrhEFNCfH1?usp=sharing  
There are three weights in this drive: 'weights0' for LinkNet, 'weights1' for D-LinkNet, 'weights2' for D-LinkNet+.  
As to enable ensemble, please put all three downloaded weights into the folder 'parameters'.  
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
    │   weights0
    │   weights1
    │   weights2
```

Finally, please run 'run.py' to reproduce the submission file. 
