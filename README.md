## SeeFood
This is a Streamlit based application that provides basic information about Desi Food (origin, ingredients &amp; preparation time).


## Data & Modeling Process

### Data

This uses the Indian Food Image Dataset that consists of 80 categories that can be found on Kaggle(https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset). Apart from the images that are already present, we have also augmented the dataset using different transformations (rotation, channel shift, vertical & horizontal flip , shearing). 90% of the data was used for training, while 10% was used for validation.

### Modeling Procedure

1. The pixels of each image are normalized to be in the range 0-1, by dividing the images by 255.

2. These normalized images are split into batches of 128 and fed into a MobileNet Network, where we train the model for 150 epochs.

3. We use Adam optimizer with a learning rate of 0.003

4. The model weights are then automatically stored in the file `training_1/cp.ckpt`.

