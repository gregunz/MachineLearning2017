
# coding: utf-8

# In[3]:


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from helpers import *
from ourHelpers import *
from models import *


# In[4]:


X, Y = load_training_dataset()
plt.imshow(concatenate_images(X[4], Y[4]))
plt.show()


# # Load the model
# 
# The model is saved inside `model_dir` so the training can be stop at anytime and restart form the same directory

# In[5]:


config = tf.estimator.RunConfig()
config._save_summary_steps = 20
model_params = {"learning_rate": 0.01}
my_estimator = tf.estimator.Estimator(model_fn=baseline_model_fn, model_dir="../model_dir",
                                      params=model_params,config=config)


# # Training
# Start or continue training from the model saved at `model_dir`

# In[4]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X},
    y=Y,
    batch_size=10,
    queue_capacity = 100,  # Important to avoid OOM Error
    num_epochs=None,
    shuffle=True)

#Train
my_estimator.train(input_fn=train_input_fn, steps=10000)


# ## Prediction on the training set

# In[8]:


predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X},
    num_epochs=1,
    batch_size= 10,
    queue_capacity = 10,
    shuffle=False)

predictions = [p for p in my_estimator.predict(input_fn=predict_input_fn)]


# In[13]:


pred = np.array([p>0.5 for p in predictions])
Y_bin = np.array([y>0.5 for y in Y])
sk_mean_F1_score(Y_bin, pred)


# In[9]:


for i, p in enumerate(predictions):
    print(np.sum(p>0.5))
    plt.imshow(concatenate_images(X[i], np.array(p>0.5)))
    #plt.imshow(p)
    plt.show()


# # Reference:
# 
# How to create your own [estimator](https://www.tensorflow.org/extend/estimators) 
