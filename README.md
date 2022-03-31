# A Lightweight Implementation of Random Shuffling (LIRS) with TFRecords
This repo contains the source code for a full-range random shuffling implementation with TFRecord file format.  
It enables programmers to take advantage of the fast random access property in SSD to perform efficient DNN training.  
With a higher degree of randomness, the model converges faster and achieves higher testing accuracy.  
LIRS is part of our ISPASS 2021 paper [“Analyzing the Interplay Between Random Shuffling and Storage Devices for Efficient Machine Learning.”](https://ieeexplore.ieee.org/document/9408217).  

You can find a list of all functions we implement [in this PDF](LIRS-TFRecord%20Implementation%20-%20Function%20List.pdf). 

## How to use
Our implementation enables programmers to train a DNN model using randomly shuffled training data stored in TFRecord file format. There are two steps.   
* Constructing a TFRecord file from raw data.  
* Training the model with the constructed TFRecord file using the random shuffling function we provided.  
  
We use training a model with the MNIST dataset as an example to show how to perform these two steps. 
* Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and follow [this document](mnist%20to%20sparse%20with%20padding%20next-fit%20class.ipynb) to construct a TFRecord file from the raw data. 
* After constructing the TFRecord file, start training with the following command.
```console
foo@bar:~$ python mnist_sparse_class.py [Use Page_Aware=1 or not=0] [TFRecord dir.] [OffsetTable dir.]
```
