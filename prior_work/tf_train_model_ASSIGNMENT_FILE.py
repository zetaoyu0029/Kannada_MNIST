import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class build_train:
    def __init__(self):
        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
        self.save_dir = self.rootPath + '/tf_model'                           

    def build_train_network(self, network):

        ############### MNIST DATA #########################################
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     
        ############### END OF MNIST DATA ##################################

        ############### CONSTRUCT NEURAL NETWORK MODEL HERE ################

        # MODEL
        # INPUT MUST BE 784 array in order be able to train on MNIST
        # INPUT PLACEHOLDERS MUST BE NAME AS name='ph_x' AND name='ph_y_'
        '''
        Follow following format for defining placeholders:
        x = tf.placeholder(data_type, array_shape, name='ph_x')
        y_ = tf.placeholder(data_type, array_shape, name='ph_y_')
        '''
        # OUTPUT VECTOR y MUST BE LENGTH 10, EACH OUTPUT NEURON CORRESPONDS TO A DIGIT 0-9
        x = tf.placeholder(tf.float32, [None, 784], name='ph_x')	#inputs
        y_ = tf.placeholder(tf.float32, [None, 10], name='ph_y_')	#labels

        W1 = tf.Variable(tf.zeros([784, 10]), name='W1')			#weights
        b1 = tf.Variable(tf.zeros([10]), name='b1')					#bias
        
        y = tf.nn.softmax(tf.matmul(x, W1) + b1, name='op_y')       #computation



        # LOSS FUNCTION, PREDICTION FUNCTION, ACCURACY FUNCTIONS
        # MAKE SURE ACCURCY FUNCTION IS NAMED ---name='op_accuracy'----
        '''
        EXAMPLE OF NAMING ACCURACY FUNCTION:
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='op_accuracy')
        '''
        
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]), name='op_loss')       #loss function for probability distribution

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1), name='op_pred')          #prediction and accuracy function
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='op_accuracy')

        ############# END OF NEURAL NETWORK MODEL ##########################

        ############# CONSTRUCT TRAINING FUNCTION ##########################

        # TRAINING FUNCTION SHOULD USE YOUR LOSS FUNCTION TO OPTIMIZE THE MODEL PARAMETERS
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name='op_train')

        ############# END OF TRAINING FUNCTION #############################


        ############# CONSTRUCT TRAINING SESSION ###########################
        saver = tf.train.Saver()                                            
        sess = tf.InteractiveSession()                                      
        sess.run(tf.global_variables_initializer())                         

        train_eval = []
        val_eval = []
        test_eval = []
        time = []

        for i in range(500):
            print("Training iteration: " + str(i))
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) 
            if i % 100 == 1:
                print("Accuracy train:")
                batch_xs, batch_ys = mnist.train.next_batch(100) 
                out1 = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                train_eval.append(out1)
                print(out1)      #print the accuracy 
                
                print("Validation train:")
                batch_xs, batch_ys = mnist.validation.next_batch(100)
                out2 = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                val_eval.append(out2)
                print(out2)      #print the accuracy 
                
                print("Test train:")
                batch_xs, batch_ys = mnist.test.next_batch(100) 
                out3 = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
                test_eval.append(out3)
                print(out3)      #print the accuracy 

                time.append(i)


        ############# END OF TRAINING SESSION ##############################

        ############# SAVE MODEL ###########################################

        print(self.rootPath)
        print(self.save_dir)
        saver.save(sess, save_path=self.save_dir, global_step=network)     
        print('Model Saved')                                               
        sess.close()                                                        
        ############# END OF SAVE MODEL ####################################

        ############# OUTPUT ACCURACY PLOT ################################

        plt.plot(time,train_eval, "b", time, val_eval, "r", time, test_eval, "g")
        # legend
        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.title("Traning Accuracy Evaluation")
        plt.show()

        ############# END OF ACCURACY PLOT ################################


