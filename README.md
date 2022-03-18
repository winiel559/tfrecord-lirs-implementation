# tfrecord-lirs-implementation
This is an implementation to enable programmers to achieve full-range random shuffling with TFRecord format.
With a higher degree of randomness, the model converges faster and achieves higher testing accuracy.

You can find a list of all functions we implement [in this PDF](tfrecord-lirs-implementation.pdf).

## How to use
There are two steps to use a TFRecord file:  
* Constructing a TFRecord file from raw data.  
* Using the TFRecord file to train the model.  
  
Using our implementation in both steps enables full-range random shuffling with TFRecord format.  
  
We show an example of either step.  
* The [first example](mnist%20to%20sparse%20with%20padding%20next-fit%20class.ipynb) shows how to take raw data and contruct a TFRecord file. Before using it, you need to download [the MNIST dataset](http://yann.lecun.com/exdb/mnist/).
* After constructing the TFRecord file, start training with the following command.
```console
foo@bar:~$ python mnist_sparse_class.py [Use Page_Aware=1 or not=0] [TFRecord dir.] [OffsetTable dir.]
```
