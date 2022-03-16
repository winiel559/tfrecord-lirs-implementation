# tfrecord-lirs-implementation
This is an implementation to enable programmers to achieve full-range random shuffling with TFRecord format.
With a higher degree of randomness, the model converges faster and achieves higher testing accuracy.

You can find a list of all functions we implement [in this PDF](https://github.com/winiel559/tfrecord-lirs-implementation/blob/main/tfrecord-lirs-implementation.pdf).

## How to use
Generally speaking, we need to construct a TFRecord file from raw data first.   
After that, we can use the TFRecord file to train the model.  
Using our implementation in both parts enables full-range random shuffling with TFRecord format.  
  
We show an example of either part.  
The construction of TFRecord is a little cumbersome

