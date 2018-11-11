### Network Performance Evaluation 

#####  Network (V1) - Configs

```python
NUM_CLASSES = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
SEED = 42
DROP_OUT = False
L2_REGULARIZE = False
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_RATE = 0.95
```
Observation :: 
**Final validation error  6.9%**
```
Step 0 of 1000
Mini-batch loss: 4.57231 Error: 88.00000 Learning rate: 0.01000
Validation error: 91.0%
Step 100 of 1000
Mini-batch loss: 0.11781 Error: 2.00000 Learning rate: 0.01000
Validation error: 15.7%
Step 200 of 1000
Mini-batch loss: 0.23938 Error: 8.00000 Learning rate: 0.01000
Validation error: 11.0%
Step 300 of 1000
Mini-batch loss: 0.05344 Error: 2.00000 Learning rate: 0.01000
Validation error: 9.5%
Step 400 of 1000
Mini-batch loss: 0.08195 Error: 2.00000 Learning rate: 0.01000
Validation error: 8.3%
Step 500 of 1000
Mini-batch loss: 0.01743 Error: 0.00000 Learning rate: 0.01000
Validation error: 8.5%
Step 600 of 1000
Mini-batch loss: 0.22797 Error: 4.00000 Learning rate: 0.01000
Validation error: 9.1%
Step 700 of 1000
Mini-batch loss: 0.07143 Error: 2.00000 Learning rate: 0.01000
Validation error: 8.4%
Step 800 of 1000
Mini-batch loss: 0.02019 Error: 0.00000 Learning rate: 0.01000
Validation error: 7.9%
Step 900 of 1000
Mini-batch loss: 0.04057 Error: 2.00000 Learning rate: 0.01000
Validation error: 6.9%

```


#####  Network (V2) - Configs

```python
NUM_CLASSES = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
SEED = 42
DROP_OUT = True
L2_REGULARIZE = False
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_RATE = 0.95
```
Observation ::
**Final validation error  2.3%**
```
Mini-batch loss: 4.57231 Error: 88.00000 Learning rate: 0.01000
Validation error: 91.9%
Step 100 of 1000
Mini-batch loss: 0.11781 Error: 2.00000 Learning rate: 0.01000
Validation error: 6.0%
Step 200 of 1000
Mini-batch loss: 0.23938 Error: 8.00000 Learning rate: 0.01000
Validation error: 4.0%
Step 300 of 1000
Mini-batch loss: 0.05344 Error: 2.00000 Learning rate: 0.01000
Validation error: 3.3%
Step 400 of 1000
Mini-batch loss: 0.08195 Error: 2.00000 Learning rate: 0.01000
Validation error: 3.1%
Step 500 of 1000
Mini-batch loss: 0.01743 Error: 0.00000 Learning rate: 0.01000
Validation error: 2.8%
Step 600 of 1000
Mini-batch loss: 0.22797 Error: 4.00000 Learning rate: 0.01000
Validation error: 2.8%
Step 700 of 1000
Mini-batch loss: 0.07143 Error: 2.00000 Learning rate: 0.01000
Validation error: 3.2%
Step 800 of 1000
Mini-batch loss: 0.02019 Error: 0.00000 Learning rate: 0.01000
Validation error: 2.9%
Step 900 of 1000
Mini-batch loss: 0.04057 Error: 2.00000 Learning rate: 0.01000
Validation error: 2.3%
```

#####  Network (V3) - Configs 

```python
NUM_CLASSES = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
SEED = 42
DROP_OUT = False
L2_REGULARIZE = True
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_RATE = 0.95
```
Observation ::
**Final validation error  2.1%**
```
Step 0 of 1000
Mini-batch loss: 7.68296 Error: 88.00000 Learning rate: 0.01000
2018-11-11 22:04:39.014691: W tensorflow/core/framework/allocator.cc:122] Allocation of 1003520000 exceeds 10% of system memory.
2018-11-11 22:04:40.608435: W tensorflow/core/framework/allocator.cc:122] Allocation of 501760000 exceeds 10% of system memory.
Validation error: 91.9%
Step 100 of 1000
Mini-batch loss: 3.18507 Error: 2.00000 Learning rate: 0.01000
2018-11-11 22:04:55.915515: W tensorflow/core/framework/allocator.cc:122] Allocation of 1003520000 exceeds 10% of system memory.
2018-11-11 22:04:57.255567: W tensorflow/core/framework/allocator.cc:122] Allocation of 501760000 exceeds 10% of system memory.
Validation error: 5.8%
Step 200 of 1000
Mini-batch loss: 3.32836 Error: 10.00000 Learning rate: 0.01000
2018-11-11 22:05:08.916096: W tensorflow/core/framework/allocator.cc:122] Allocation of 1003520000 exceeds 10% of system memory.
Validation error: 3.9%
Step 300 of 1000
Mini-batch loss: 3.09006 Error: 2.00000 Learning rate: 0.01000
Validation error: 3.1%
Step 400 of 1000
Mini-batch loss: 3.08653 Error: 2.00000 Learning rate: 0.01000
Validation error: 3.3%
Step 500 of 1000
Mini-batch loss: 2.99207 Error: 2.00000 Learning rate: 0.01000
Validation error: 2.9%
Step 600 of 1000
Mini-batch loss: 3.14293 Error: 2.00000 Learning rate: 0.01000
Validation error: 2.7%
Step 700 of 1000
Mini-batch loss: 3.00302 Error: 6.00000 Learning rate: 0.01000
Validation error: 2.9%
Step 800 of 1000
Mini-batch loss: 2.89098 Error: 0.00000 Learning rate: 0.01000
Validation error: 2.8%
Step 900 of 1000
Mini-batch loss: 2.87223 Error: 0.00000 Learning rate: 0.01000
Validation error: 2.1%
```

#####  Network (V4) - Configs 

```python
NUM_CLASSES = 10
VALIDATION_SIZE = 5000
BATCH_SIZE = 60
NUM_CHANNELS = 1
SEED = 42
DROP_OUT = True
L2_REGULARIZE = True
LEARNING_RATE = 0.01
MOMENTUM = 0.9
DECAY_RATE = 0.95
```
Observation ::
**Validation error: 1.9%**
```
Step 0 of 916
Mini-batch loss: 7.71249 Error: 91.66667 Learning rate: 0.01000
Validation error: 88.9%
Step 100 of 916
Mini-batch loss: 3.28715 Error: 8.33333 Learning rate: 0.01000
Validation error: 5.8%
Step 200 of 916
Mini-batch loss: 3.30949 Error: 8.33333 Learning rate: 0.01000
Validation error: 3.6%
Step 300 of 916
Mini-batch loss: 3.15385 Error: 3.33333 Learning rate: 0.01000
Validation error: 3.1%
Step 400 of 916
Mini-batch loss: 3.08212 Error: 1.66667 Learning rate: 0.01000
Validation error: 2.7%
Step 500 of 916
Mini-batch loss: 3.02827 Error: 1.66667 Learning rate: 0.01000
Validation error: 2.2%
Step 600 of 916
Mini-batch loss: 3.03260 Error: 5.00000 Learning rate: 0.01000
Validation error: 1.9%
Step 700 of 916
Mini-batch loss: 3.16032 Error: 6.66667 Learning rate: 0.01000
Validation error: 2.2%
Step 800 of 916
Mini-batch loss: 3.06246 Error: 3.33333 Learning rate: 0.01000
Validation error: 2.0%
Step 900 of 916
Mini-batch loss: 2.85098 Error: 0.00000 Learning rate: 0.01000
Validation error: 1.9%
```
The error seems to have gone down. Let's evaluate the results using the test set.

To help identify rare mispredictions, we'll include the raw count of each (prediction, label) pair in the confusion matrix.

![][logo]

[logo]: ../docs/images/v1_score.png?raw=true "Logo Title Text 2"
