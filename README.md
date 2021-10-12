# Classifiers

## 1. 2D classifier

### Dataset Class
Initialisation : We have two data files : train.npz and valid.npz for training and validation.
Using the split argument specifying the one we want to use, we update the path to the corresponding file and set the samples and annotations numpy arrays.

Get Item : Given an index we want the corresponding sample and annotation which we can directly get from the arrays and return as a tensor.

### LinearClassifier Class
Initialisation : We are at first using only a single linear layer in our sequential call. Since our input is a 2D value the linear layer has two in_features. Since the output is the probability of a given 2D point being part of cluster 1, the linear layer has only one out_feature.

### Training
We can implement the training loop in the run_training_epoch function from train.py. There are four steps for each training step:
1. Clear the gradients from the previous step using optimizer.zero grad()
2. Forward pass of the network to obtain the predictions
3. Compute the loss (binary cross entropy)
4. Backwards pass on the loss using loss.backward() to obtain the gradients followed by one step of gradient descent (optimization) using using optimizer.step().