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


#Implementation of U-Net without the copy and crop https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
#DEFINE THE BASELINE MODEL
def baseline_model_fn(features, labels, mode, params):

    h= int(features["x"].shape[1])
    w= int(features["x"].shape[1])
    #print(features["x"].shape)
    with tf.device('/gpu:1'):
        # Input layer
        conv1 = tf.layers.conv2d(inputs=features["x"],
                                filters= 64,
                                kernel_size=[3,3],
                                padding="same",
                                activation=tf.nn.tanh)
        conv2 = tf.layers.conv2d(inputs=conv1,
                            filters= 64,
                            kernel_size=[3,3],
                            padding="same",
                            activation=tf.nn.tanh)

        # Pooling layer 1
        pool1 = tf.layers.max_pooling2d(inputs=conv2,
                                       pool_size=[2,2],
                                       strides=2)
        # Convolution layer 2
        conv3 = tf.layers.conv2d(inputs=pool1,
                                filters=128,
                                kernel_size=[3,3],
                                padding="same",
                                activation= tf.nn.tanh)
        conv4 = tf.layers.conv2d(inputs=conv3,
                            filters=128,
                            kernel_size=[3,3],
                            padding="same",
                            activation= tf.nn.tanh) 

        # Pooling layer 2
        pool2 = tf.layers.max_pooling2d(inputs=conv4,
                                       pool_size=[2,2],
                                       strides=2)
        conv5 = tf.layers.conv2d(inputs=pool2,
                            filters=256,
                            kernel_size=[3,3],
                            padding="same",
                            activation= tf.nn.tanh) 
        conv6 = tf.layers.conv2d(inputs=conv5,
                        filters=256,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)

        # Pooling layer 3
        pool3 = tf.layers.max_pooling2d(inputs=conv6,
                                       pool_size=[2,2],
                                       strides=2)
        conv7 = tf.layers.conv2d(inputs=pool3,
                        filters=512,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)
        conv8 = tf.layers.conv2d(inputs=conv7,
                        filters=512,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)

        # Pooling layer 4
        pool4 = tf.layers.max_pooling2d(inputs=conv8,
                                       pool_size=[2,2],
                                       strides=2)

        conv9 = tf.layers.conv2d(inputs=pool4,
                        filters=1024,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)
        conv10 = tf.layers.conv2d(inputs=conv9,
                        filters=1024,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)

        #Up sampling (like inverse of pooling)
        de_conv1 = tf.layers.conv2d_transpose(inputs=conv10,
                                             filters= 512,
                                             kernel_size=[3,3],
                                             strides=(2, 2),
                                             padding= "same",
                                             activation= tf.nn.tanh)

        #concatenate de_conv1 with conv8
        concat1 = tf.concat([conv8, de_conv1], axis= 3) 
        conv11 = tf.layers.conv2d(inputs=concat1,
                        filters=512,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)
        conv12 = tf.layers.conv2d(inputs=conv11,
                        filters=512,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)


        #Up sampling 2
        de_conv2 = tf.layers.conv2d_transpose(inputs=conv12,
                                             filters= 256,
                                             kernel_size=[3,3],
                                             strides=(2, 2),
                                             padding= "same",
                                             activation= tf.nn.tanh)
        concat2 = tf.concat([conv6 , de_conv2], axis=3)

        conv13 = tf.layers.conv2d(inputs=concat2,
                        filters=256,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)
        conv14 = tf.layers.conv2d(inputs=conv13,
                        filters=256,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)

        #Up sampling 3
        de_conv3 = tf.layers.conv2d_transpose(inputs=conv14,
                                             filters= 128,
                                             kernel_size=[3,3],
                                             strides=(2, 2),
                                             padding= "same",
                                             activation= tf.nn.tanh)
        concat3 = tf.concat([conv4, de_conv3], axis=3)
        conv15 = tf.layers.conv2d(inputs=concat3,
                        filters=128,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)
        conv16 = tf.layers.conv2d(inputs=conv15,
                        filters=128,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)


        #Up sampling 4
        de_conv4 = tf.layers.conv2d_transpose(inputs=conv16,
                                             filters= 64,
                                             kernel_size=[3,3],
                                             strides=(2, 2),
                                             padding= "same",
                                             activation= tf.nn.tanh)
        concat4 = tf.concat([conv2, de_conv4], axis=3)
        conv17 = tf.layers.conv2d(inputs=concat4,
                        filters=64,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)
        conv18 = tf.layers.conv2d(inputs=conv17,
                        filters=64,
                        kernel_size=[3,3],
                        padding="same",
                        activation= tf.nn.tanh)

        logits = tf.layers.conv2d(inputs=conv18,
                        filters=1,
                        kernel_size=[3,3],
                        padding="same")   
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.image("Prediction", tf.sigmoid(logits))
            tf.summary.image("label", tf.expand_dims(labels, 3))

        #print(logits.shape)    
        logits = tf.reshape(logits, [-1, h*w])
        predictions = tf.sigmoid(logits)
        predictions = tf.reshape(predictions, [-1, h, w])

        # Provide an estimator spec for `ModeKeys.PREDICT`.
        if mode == tf.estimator.ModeKeys.PREDICT:
            print(predictions.shape)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        labels = tf.reshape(labels, [-1, h*w])
        #print(labels.shape)
        #print(logits.shape)

        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(labels= labels, logits= logits)

        #labels = tf.one_hot(tf.reshape(labels, [-1]), 2)
        #print("--------")
        #print(cross_entropies.shape)
        loss = tf.reduce_sum(cross_entropies) #/ tf.constant(logits.shape[0], dtype=tf.float32)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('Loss',loss)
        else:
            #evaluate
            tf.summary.scalar('Loss test set',loss)

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=params["learning_rate"])
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss=loss, 
                                      global_step=tf.train.get_global_step())
        """
        #cast label to boolean
        labels = labels > 0.5

        eval_metric_ops = {
            #"loss": loss,
            "mean_F1-Score": mean_F1_score(labels=labels, predictions=predictions)
        }
        """

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)#, eval_metric_ops=eval_metric_ops)
