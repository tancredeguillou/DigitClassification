# Classifiers

## 1. 2D classifier

### Dataset class
Initialisation : We have two data files : train.npz and valid.npz for training and validation.
Using the split argument specifying the one we want to use, we update the path to the corresponding file and set the samples and annotations numpy arrays.

Get Item : Given an index we want the corresponding sample and annotation which we can directly get from the arrays and return as a tensor.

### LinearClassifier class
Initialisation : We are at first using only a single linear layer in our sequential call. Since our input is a 2D value the linear layer has two in_features. Since the output is the probability of a given 2D point being part of cluster 1, the linear layer has only one out_feature.

### Training
We can implement the training loop in the run_training_epoch function from train.py. There are four steps for each training step:
1. Clear the gradients from the previous step using optimizer.zero grad()
2. Forward pass of the network to obtain the predictions
3. Compute the loss (binary cross entropy)
4. Backwards pass on the loss using loss.backward() to obtain the gradients followed by one step of gradient descent (optimization) using using optimizer.step().

### Non linearly separable data : MLP
networks.py implements a Multi-Layer-Perceptron for non linearly separable data. Since MLPs are designed to approximate any continuous function, they can solve problems which are not linearly separable.

### Feature transform of data
Since our data has a circular shape, we might want to change the coordinate system of our model to make our classes linearly separable. We can logically transform them with polar coordinates (r, θ). Since data represents two circles, by having r as horizontal axis and θ as vertical axis we will have two straight vertical lines for the corresponding radius.

## 2. MNIST Classifier

### Data normalization
We want pixel value to range in [-1, 1] instead of [0, 255]. Since the values are stored as uint8, i.e cannot be negative, we first have to consider them as float by changing the type.

### Training loop
The training loop is done in the same way except for the loss function which is a cross entropy instead of binary. That is beacause we don't have only two classes anymore.

### MLP
One linear prediction layer : 91.85% accuracy.
One hidden linear layer of dimension 32 and one final linear prediction layer : 94.88% accuracy.

### Convolutional network
Three convolutional layers with ReLu and max pooling followed by one last adaptive max pooling and a simple classifier linear prediction layer.
Accuracy obtained : 98.08%.