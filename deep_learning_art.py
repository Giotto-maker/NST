#!/usr/bin/env python3
# reference : https://github.com/gemaatienza/Deep-Learning-Coursera/blob/master/4.%20Convolutional%20Neural%20Networks/Art%20Generation%20with%20Neural%20Style%20Transfer%20-%20v2.ipynb

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys 
import imageio
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import matplotlib.pyplot as plt 
from PIL import Image
from matplotlib.pyplot import imshow 

from nst_utils import *

# load the pre-trained model :
#   The model is stored in a python dictionary where each variable name is the key 
#   and the corresponding value is a tensor containing that variable's value
model = load_vgg_model("pretrained_model/imagenet-vgg-verydeep-19.mat")
print("MODEL OVERVIEW:\n")
for (layer_name, layer_tensor) in model.items() :
    print("layer_name = " +str(layer_name)+ " , layer_tensor = " +str(tf.compat.v1.shape(layer_tensor)))
print()

##################################################################################################################
#   COMPUTE CONTENT LOSS (in respect to a chosen layer OF C and G)
def compute_content_cost(a_C,a_G):
    """ 
    Computes the content cost
    
    Arguments:
    a_C --> tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G --> tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content --> scalar that you compute using equation 1 above.
    """

    # retrive dimensions from a_G layer (same of a_C layer)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # unroll a_G and a_C from a 3D volume to 2D volume (in order to speed up calculations and prepare for style loss)
    a_C_unrolled = tf.reshape(a_C, [n_H*n_W , n_C])
    a_G_unrolled = tf.reshape(a_G, [n_H*n_W , n_C])
    # compute cost (i.e distance between a_G and a_C)
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))

    return (1/(4*n_H*n_W*n_C)) * J_content

tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as test:
    tf.compat.v1.set_random_seed(1)
    a_C = tf.compat.v1.random_normal([1,4,4,3], mean=1, stddev=4)
    a_G = tf.compat.v1.random_normal([1,4,4,3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J CONTENT COMPUTATION:\n")
    print("J_content = " +str(J_content.eval()))
print()

#  COMPUTE STYLE LOSS 

# I.  Gram Matrix computation
def gram_matrix(A):
    """
    Argument:
    A --> matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA --> Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A,A,transpose_b=True)
    return GA


tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as test:
    tf.compat.v1.set_random_seed(1)
    A = tf.compat.v1.random_normal([3, 2*1] , mean=1, stddev=4)
    GA = gram_matrix(A)
    print("GRAN MATRIX COMPUTATION:\n")
    print("GA = " +str(GA.eval()))
print()


# II.  Style loss for layer
def compute_layer_style_cost(a_S,a_G):
    """
    Arguments:
    a_S --> tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G --> tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer --> tensor representing a scalar value, style cost defined above by equation (2)
    """

    # retrive dimensions from a_G
    m,n_H,n_W,n_C = a_G.get_shape().as_list()
    # unroll tensors from 3D to 2D
    a_S = tf.transpose(tf.reshape(a_S,[n_H*n_W , n_C]))
    a_G = tf.transpose(tf.reshape(a_G,[n_H*n_W , n_C]))
    # compute gran matrices for both S and G tensors
    GS = gram_matrix(a_S) # shape (nc x nc)
    GG = gram_matrix(a_G) # shape (nc x nc)
    # compute loss for this layer
    J_style_layer_loss = tf.reduce_sum(tf.square(tf.subtract(GS,GG))) # note that reduce_sum performs all sums until the tensor is a scalar (sums over the channels here)

    return (1/((2*n_C*n_H*n_W)**2)) * J_style_layer_loss 


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as test:
    tf.compat.v1.set_random_seed(1)
    a_S = tf.compat.v1.random_normal([1,4,4,3], mean=1 , stddev=4)
    a_G = tf.compat.v1.random_normal([1,4,4,3], mean=1 , stddev=4)
    J_style_layer_loss = compute_layer_style_cost(a_S,a_G)
    print("J STYLE COMPUTATION:\n")
    print("J_style_layer_loss = " +str(J_style_layer_loss.eval()))
print()

# III. Style loss for each layer
def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model --> our tensorflow model
    STYLE_LAYERS --> A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style --> tensor representing a scalar value, style cost defined above by equation (2)
    """

    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        # select the output tensor of the currently selected layer
        out = model[layer_name]
        #   NOTE : 
        #   -   this function will be called inside an open session
        #   -   the image input for the model will be defined before this function is called
        a_S = sess.run(out)   # select hidden layer from the style image
        a_G = out             # select hidden layer from the generated image  

        # compute style cost by comparing the layers 
        J_style_layer = compute_layer_style_cost(a_S,a_G)
        # add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer
    
    return J_style


# IV.  Total style loss
def total_cost(J_content, J_style, alpha = 10 , beta = 40):
    J = (alpha*J_content)+(beta*J_style)
    return J 


# NOTE: this is not the session that will call compute style cost
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    print("TOTAL LOSS COMPUTATION:\n")
    J = total_cost(J_content, J_style)
    print("J = " + str(J))


#  SOLVING THE OPTIMIZATION PROBLEM :
    # NOTE : Unlike a regular session, the "INTERACTIVE SESSION" installs itself as the default session 
    #        to build a graph. This allows you to run variables without constantly 
    #        needing to refer to the session object, which simplifies the code

# reset the graph
tf.compat.v1.reset_default_graph()
# start the Interactive Session
sess = tf.compat.v1.InteractiveSession()
# load reshape and normalize the content image
content_image = imageio.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)
# load reshape and normalize the style image
style_image = imageio.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)
# generate noisy image to start with ( this will help the content of the "generated" image more rapidly match the content of the "content" image)
generated_image = generate_noise_image(content_image)  # this image is still slightly correlated to the content one
imshow(generated_image[0])
plt.show()


#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

model = load_vgg_model("pretrained_model/imagenet-vgg-verydeep-19.mat")

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

### START CODE HERE ### (1 line)
J = total_cost(J_content, J_style, alpha = 10, beta = 40)
### END CODE HERE ###

# define optimizer (1 line)
optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
# <Gema> Define the tensorflow optimizer. Use an AdamOptimizer.

# define train_step (1 line)
train_step = optimizer.minimize(J) 
# <Gema> This computes the backpropagation by passing through the tensorflow graph in the reverse order. From cost to inputs

def model_nn(sess, input_image, num_iterations = 200):
    
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.compat.v1.global_variables_initializer()) # <Gema> Run the initialization. Now all variables are initialized.
    ### END CODE HERE ###
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image)) # <Gema> Assign the input of the model to be the initial image
    ### END CODE HERE ###
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        _ =sess.run(train_step) # Run the session to execute the "optimizer" and the "cost"
        # <Gema> When coding, we often use _ as a "throwaway" variable to store values that we won't need to use later
        ### END CODE HERE ###
        
        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


model_nn(sess, generated_image)

