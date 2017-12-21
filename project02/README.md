# [Road Segmentation Project][kaggle]

Kaggle competition between EPFL Machine Learning course (CS433) students

## How to prepare your environment
- images should be but in the 'data' folder in the root of the projects containing the training and testing images
inside their respective folders ('training' and 'test_set_images' folders as on Kaggle)
- [weights.hdf5][weights] should be put in the 'src' folder (for the pre-trained model).


## Dependencies
- numpy, natsort, opencv-python, imutils, matplotlib, pillow, tensorflow, keras, telepyth

all those dependencies can be easily installed with pip

python version: 3.5.3 on Linux x86_64 4.4.0-103-generic

- note_1: training was done with GPU on FloydHub (see Standard GPU : Tesla K80 - 12 GB Memory - 61 GB RAM),
it took around 3h (1 per epoch)
- note_2: predictions on FloydHub took around 20 minutes with the GPU (with pre-trained weights)
- note_3: FloydHub default keras environment had all the dependencies already installed except: imutils, telepyth, natsort

## How to run
- As simple as:

`
python run.py
`

note: the LOAD_WEIGHTS variable determines whether we should load weights or do the full training.


## Output
- The run.py will create an output folder containing another folder the model name. Inside this folder,
 there will be a config.json (sums up the configuration used), model_summary.json (model description),
  the submission_xxx.csv and the weights of the model.

## File structure

- #### abstract_pipeline.py
Define an abstract class representing an abstract pipeline which makes it easier to create multiple model and still run 
the same code, thanks to object oriented programming. (class is abstract because the model is not implemented)

This abstract pipeline implements loading data, loading model, training model, computing predictions, creating 
submission.

- #### unet.py
A class implementing the abstract pipeline with a U-net architecture model.

- #### run.py
A script which runs the whole pipeline from training to creating a submission.

- #### default_config.json
The default configuration of the pipeline

- #### helpers.py
Helpers which contains general use functions e.g. to load images, the image pre-processing pipeline, config handling...

- #### helpers_image.py
Helpers which contains image related functions e.g. to apply operations (e.g. rotations or gamma correction), 
to transform images to patches and back, etc.

- #### helpers_keras.py
Helpers which contains keras related functions e.g. metrics such as f1, recall and precision

- #### helpers_submission.py
Helpers which contains submission related functions to go from masks (predictions) to a submission file

[kaggle]: https://www.kaggle.com/c/epfml17-segmentation
[weights]: https://drive.google.com/open?id=1-RqtAF-T3o_7mIZQVtmQ-UGmf01SW46F