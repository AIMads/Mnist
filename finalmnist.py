import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Gets the dataset from tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Puts all the data on the gpu for training this is an optimaztion for bigger datasets
tf.contrib.data.prefetch_to_device("gpu:0", buffer_size = None)
tf.reset_default_graph()
#Defines the size for the hidden layers
n_nodes_hl1 = 700
n_nodes_hl2 = 700
n_nodes_hl3 = 700

#Defines the amount of classes, in this model there is 10, which are the numbers from 0 to 9
n_classes = 10

#This is done too devide the dataset into smaller bits for less powerfull computers
batch_size = 100

#Defines two constants for the graph
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#This function is the feed forward neural network model, the weights is how certain the model is on a descision, the biases is to
#secure that no matter what a neuron will fire, by adding this value
def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}

# This is where the feed forward flow happens, the values that start as random values, multiplied and then the bias is added 
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    
    return output
# This function is taking the neural network model that has been build, and now trains it

def train_neural_network(x):
    #The prediciton is what our model tell us, at start this will be random, but as we optimize that will change
    prediction = neural_network_model(x)
    #The cost is a value that describes how wrong the result is, and it is what needs too be minimized
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    #Too optimize the cost, the following function is used, i have used gradientdescent, but there are many others that can be used
    # for example "AdamOptimizer"
    optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    # The amount of epochs tells how many times the feed forward network shall be used
    hm_epochs = 12
    saver = tf.train.Saver()
    #Too use tensorflows tools it is needed to activate a session 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    # For each epoch and each batch the optimizer and cost function will be runned on the batch of data
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
    #This will give a loss for each epoch that at the start will get a lot better and slowly getting smaller and maybe even worse
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
    #correctAnswer will say how many right answers the model got, where y is the answer
        correctAnswer = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    #accuracy is done out from some math
        accuracy = tf.reduce_mean(tf.cast(correctAnswer, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)

        
train_neural_network(x)

