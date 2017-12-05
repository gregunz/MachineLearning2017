import tensorflow as tf
import numpy as np

def F1_score(labels, predictions):
    #P, update_op1 = tf.contrib.metrics.streaming_precision(predictions, labels)
    P, update_op1 = tf.metrics.precision(predictions, labels)
    #R, update_op2 = tf.contrib.metrics.streaming_recall(predictions, labels)
    R, update_op2 = tf.metrics.recall(predictions, labels)
    eps = 1e-5; #To prevent division by zero
    return (2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))

def mean_F1_score(labels, predictions):
    F1_scores = tf.constant(0.0)
    ops = tf.group()
    for i in range(predictions.shape[0]):
        pred = predictions[i]
        lab = labels[i]
        pred = tf.reshape(pred, [-1])
        lab = tf.reshape(lab, [-1])
        F1, update_op = F1_score(lab, pred)
        ops = tf.group(ops, update_op)
        F1_scores +=F1

    # the return must be a tuple (metric_value, update_op)
    # TODO change 10.0 by a variable
    return (F1_scores / 10.0 , ops)



#DEFINE THE BASELINE MODEL
def baseline_model_fn(features, labels, mode, params):

    h= 400
    w= 400

    # Input layer
    conv1 = tf.layers.conv2d(inputs=features["x"],
                            filters= 8,
                            kernel_size=[5,5],
                            padding="same",
                            activation=tf.nn.tanh)
    """
    # Pooling layer 1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                   pool_size=[2,2],
                                   strides=2)
    #Add dropout
    # Convolution layer 2
    conv2 = tf.layers.conv2d(inputs=conv1,
                            filters=1,
                            kernel_size=[8,8],
                            padding="same",
                            activation= tf.nn.tanh)
    

    # Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                   pool_size=[2,2],
                                   strides=2)


    de_conv1 = tf.layers.conv2d_transpose(inputs=conv2,
                                         filters= 1,
                                         kernel_size=[16,16],
                                         padding= "same",
                                         activation= tf.nn.tanh)
    """
    logits = tf.layers.conv2d_transpose(inputs=conv1,
                                         filters= 1,
                                         kernel_size=[5,5],
                                         padding= "same")
    
    logits = tf.reshape(logits, [-1, h*w])
    predictions = tf.sigmoid(logits)
    predictions = tf.reshape(predictions, [-1, h, w])
    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        print(predictions.shape)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    labels = tf.reshape(labels, [-1, h*w])
    print(labels.shape)
    print(logits.shape)
    
    cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(labels= labels, logits= logits)

    #labels = tf.one_hot(tf.reshape(labels, [-1]), 2)
    loss = tf.reduce_sum(cross_entropies)
    tf.summary.scalar('Loss',loss)
    
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(loss=loss, 
                                  global_step=tf.train.get_global_step())
    
    #cast label to boolean
    labels = labels > 0.5

    eval_metric_ops = {
        #"loss": loss,
        "mean_F1-Score": mean_F1_score(labels=labels, predictions=predictions)
    }


    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)
