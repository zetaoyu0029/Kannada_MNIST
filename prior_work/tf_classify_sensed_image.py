import tensorflow as tf
import os
# from tensorflow.examples.tutorials.mnist import input_data

class evaluate_model:
    def __init__(self):
        self.rootPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))   
        self.save_dir = self.rootPath + '/tf_model'                             
        
    def evaluate_model(self,model_version,input):

        # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        filename = self.save_dir + '-' + str(model_version) + '.meta'           
        filename2 = self.save_dir + '-' + str(model_version)                    
        print('Opening Model: ' + str(filename))                                
        saver = tf.train.import_meta_graph(filename)                            
        sess = tf.InteractiveSession()                                          

        # INITIALIZE GLOBAL VARIABLES
        sess.run(tf.global_variables_initializer())                             

        # RESTORE MODEL LOSS OP AND INPUT PLACEHOLDERS
        saver.restore(sess, filename2)                                          
        graph = tf.get_default_graph()                                          


        # accuracy = graph.get_tensor_by_name('op_accuracy:0')
        x = graph.get_tensor_by_name('ph_x:0')
        # y_ = graph.get_tensor_by_name('ph_y_:0')
        y = graph.get_tensor_by_name('op_y:0')
        output = tf.arg_max(y,1)


        print('Classifying Image...')
        output = sess.run(output, feed_dict={x: input})
        sess.close()

        return output


