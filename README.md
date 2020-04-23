# Notes on implementation of Cross Entropy Loss

This is a reference note for myself if I ever want to recall the formulas and the implementations.


1) Cross Entropy Loss for Binary classification

##### ___loss = label * (-1) * log(pred) + (1 - label) * (-1) * log(1 - pred)___

Here "label" can be either 0 or 1 and "pred" can be a probability value between 0 to 1 - any real value. The loss is a scalar value.

So, for label=1, the loss will be
##### ___loss_label_1 = label * (-1) * log(pred) + 0___
##### ___loss_label_1 = - log(pred)___

Same way for label=0, the loss will be
##### ___loss_label_0 = 0 + (1 - label) * (-1) * log(1 - pred)___
##### ___loss_label_0 = -log(1-pred)___

So, the first equation actually handles both the cases.


2) Softmax Cross Entropy Loss for Binary Classification

##### ___softmax_logits = softmax(logits)___
##### ___loss_softmax_cross = label * (-1) * log(softmax_logits) + (1 - label) * (-1) * log(1 - softmax_logits)___

Here, because the logits are softmaxed, they contain the probability of being a positive class.


3) Softmax Cross Entropy Loss for Multi Class Classification
##### ___softmax_logits = softmax(logits)___
##### ___loss_softmax_cross_multi = sum(label * (-1) * log(softmax_logits))___

Here, labels and logits both are a vector / single column array. E.g. for 10 classes it is [10,] array.
The sum represents the sum across the dimension depicted by num_of_classes which is in this case last dims. So, the loss is a scalar value.


4) Weighted Softmax Cross Entropy Loss for Multi Class Classification
##### ___softmax_logits = softmax(logits)___
##### ___loss_softmax_cross_multi = sum(cls_weight * label * (-1) * log(softmax_logits))___

Here, labels, logits and cls_weights all are having same shape - a vector / single column array. E.g. for 10 classes it is [10,] array.
The loss is a scalar value.
The cls_weight has a separate weight for each class.


5) Sigmoid Cross Entropy Loss
The sigmoid cross entropy is same as softmax cross entropy except for the fact that instead of softmax, we apply sigmoid function on logits before feeding them.



### Notes on implementation part:
Now that we know the mathematical formula, we can implement the loss function by ourselves.
But when Tensorflow has made a wheel then why to reinvent the same wheel.
The Tensorflow provides several APIs for the cross entropy losses.

##### ___tf.nn.sigmoid_cross_entropy_with_logits(label, logit)___
##### ___tf.losses.sigmoid_cross_entropy(label, logit)___
##### ___tf.nn.softmax_cross_entropy_with_logits(label, logit)___
##### ___tf.losses.softmax_cross_entropy(label, logit)___

Now the key difference between tf.nn and tf.losses - is that tf.nn is low level api and tf.losses is high level API.

Now let's take an example to see how these two APIs differ computing the same function and resulting in different shapes.
a) E.g. Label as well as Logits are having shape: [Batch x num_classes] 

___tf.nn.sigmoid_cross_entropy_with_logits___ will compute the loss as per the above formulas. So at first the shape remains intact. Then it sums across the last axis - which represents num_classes. And then returns the final tensor which is having shape [Batch] in our case. If we want the single value of loss per batch then we can take mean or sum of all the values across rest of the dimensions.

On the other hand, ___tf.losses.sigmoid_cross_entropy___ will compute the loss as per the same above formula. But, it returns a single scalar value for the loss. It first takes the sum across last axis which is num_class axis, and then take the mean of all the values across all axis. That's why it returns a single value.

b) E.g. Let's take a tough shape which I have faced during the implementation of Yolo Loss function.
Label and Logit both have shape [batch x 13 x 13 x 3 x 20]

Now, ___tf.losses.sigmoid_cross_entropy___ will give us single value and the loss for a batch of 64 is in the range of 0.0038 which is very low because it takes sum across last axis and takes mean across rest of the axis.
In case of tf.nn.sigmoid_cross_entropy_with_logits - it gives me a tensor having shape [b x 13 x 13 x 3] by only taking sum across last axis. Now i can multiply this with another [b x 13 x 13 x 3] shaped tensor.


After these examples you can use the API which suit your needs.


*** ___If you want to use cls_weights then you have to implement the cross entropy loss from scratch as per formula given in (3) because there are one APIs in Tensorflow for that which is tf.nn.weighted_cross_entropy_with_logits but it is deprecated.___





### How to compute the cls_weight for imbalance class problem?


1. First count the freq of each class:
	E.g. [cls_1: 1000, cls_2: 2300, cls_3: 20310, cls_4: 6700]
2. Then take the ration (sum_of_all_freqs / cls_freq) for all classes
	E.g. [cls_1: 30.31, cls_2: 13.17, cls_3: 1.492, cls_4: 4.523]
3. Take log of it.
	E.g. [cls_1: 3.41, cls_2: 2.57, cls_3: 0.40, cls_4: 1.50]
4. Now take the max between the value and 1.0 to clip lower values at 1.0
	E.g. [cls_1: 3.41, cls_2: 2.57, cls_3: 1.00, cls_4: 1.50]
4a.[Optional] Take the max value from 3 and divide all values with that.
	E.g. [cls_1: 1.00, cls_2: 0.755, cls_3: 0.117, cls_4: 0.442]

Now, these are your class weights. As you can see the cls_3 which has highest occurrence is given least weight and cls_1 which has the least freq. is given highest weight.
As per "https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits?hl=en", these weights greater than 1 which decrease false negative counts which improves recall.
And if we want to reduce false positive and improve precision than use weights computed in 4a. -- again as per the above link.

