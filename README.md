# tfrecord-lirs-implementation
This is an implementation to enable programmers to achieve full-range random shuffling with TFRecord format.
With a higher degree of randomness, the model converges faster and achieves higher testing accuracy.

You can find a list of all functions we implement [in this PDF](https://github.com/winiel559/tfrecord-lirs-implementation/blob/main/tfrecord-lirs-implementation.pdf).

## How to use
There are two steps to use a TFRecord file:  
* Constructing a TFRecord file from raw data.   
* Using the TFRecord file to train the model.  
Using our implementation in both steps enables full-range random shuffling with TFRecord format.  
  
We show an example of either step.  
* The construction of TFRecord is a little cumbersome, so it's shown step-by-step in the example. Before using it, you need to download [the MNIST dataset](http://yann.lecun.com/exdb/mnist/).
* After constructing the TFRecord file, you can start training by the following command.
```console
foo@bar:~$ python mnist_sparse_class.py [] [] []
```
