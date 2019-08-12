# ConvNet-CIFAR-10
Design:
The program design consists of a CNN class which contains all the methods like forward, backward, train_step. Layer class which contains Convolution layer, Max pooling, Flatten and Fully connected. Loss contains L2 Regularization and softmax loss. cnn_final contains the main function. Predict function The cifar 10 data is taken in as batches. make_CNN creates a Network and sgd_momentum functions is called for training the neural network. It takes the network, training set, learning rate and iterations, training_set, mini_batch size, test dat. Then, the Conv layer takes these parameters, which is then goes to ReLu layer and Maxpool layer which happens for two time and then it goes to a fully connected layer.

Simulation:
The program is being run on the reduced data set (1000 training and 100 test) as running the complete data set was taking around 3 hours for just 6 iterations.
