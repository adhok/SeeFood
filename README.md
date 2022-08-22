## SeeFood (Indian Food Edition 🇮🇳)


This is a Streamlit based application that provides basic information about Desi Food (origin, ingredients &amp; preparation time).


## Data & Modeling Process

### Data

This uses the Indian Food Image Dataset that consists of 80 categories that can be found on Kaggle(https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset). Apart from the images that are already present, we have also augmented the dataset using different transformations (rotation, channel shift, vertical & horizontal flip , shearing). You can recreate this by running the code ```python augment.py``` on your terminal. 90% of the data was used for training, while 10% was used for validation.

### Modeling Procedure

1. The pixels of each image are normalized to be in the range 0-1, by dividing the images by 255.

2. These normalized images are split into batches of 128 and fed into a MobileNet Network, where we train the model for 150 epochs.

3. We use Adam optimizer with a learning rate of 0.003

4. The model weights are then automatically stored in the file `training_1/cp.ckpt`.

5. The training can be done by running the command ```python train.py```

## The other Python Files and what they mean

1. ```streamlit_app.py``` -> This contains the front end function of the app, such as layout , text color and the drag and drop functionality. The app can be invoked using the command ```streamlit run streamlit_app.py```.

2. ```data_prep_for_prediction.py``` -> This contains the code that maps all 80 types of Indian Food to its prep time, region of origin and ingredients.

3. ```inference.py``` -> This uses the model weights file created in the folder ```training_1/``` and outputs the desired results(Food Type, Region of Origin, Prep time and Ingredients)  based on the input image.





