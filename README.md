# Code-Mixed-Hate-Speech-Detection
## Introduction
This repository contains the classifiers to detect hate speech in code mixed hinglish text. It contains codes for training different classifiers: Naive Bayes, Random Forest and
CNN-1D architecture proposed in https://arxiv.org/pdf/1811.05145.pdf 

## Detection

Follow the below steps to use the trained model.

```
pip install -r /content/Code-Mixed-Hate-Speech-Detection/requirements.txt
cd /content/Code-Mixed-Hate-Speech-Detection/detect
python detect.py 
```
It will take some time to load the model. After that, the below interface can be used to input the text.
```
2023-04-28 13:02:52.066523: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-04-28 13:02:53.396334: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
--------Initialising Model---------------
6367
(None, 6367, 200)
(None, 6366, 64)
(None, 6365, 64)
(None, 6364, 64)
(None, 6367, 200)
(None, 6366, 64)
(None, 6365, 64)
(None, 6364, 64)
---------------Loaded Model-------------------------
Text to classify (Press q to exit) : tumko padne likhne bhi nahi aata
CAG
Text to classify (Press q to exit) : itna dumb kaise ho sakthe ho
OAG
Text to classify (Press q to exit) : how is the weather today
NAG
Text to classify (Press q to exit) : have a great day
NAG
Text to classify (Press q to exit) : q
```

