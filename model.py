import tensorflow as tf
from tensorflow import keras


def model_def():
    
    
    






    class mobile_net_unit(keras.layers.Layer):
        def __init__(self,filters_1,filters_2,size_1,size_2,strides_1,strides_2,**kwargs):
            super().__init__(**kwargs)

            self.main_layers = [

                keras.layers.DepthwiseConv2D(kernel_size = size_1,strides = strides_1,padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Conv2D(filters=filters_2,kernel_size=size_2,strides=strides_2,padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu')


            ]

        def call(self,inputs):
            Z = inputs
            for layer  in self.main_layers:

                Z = layer(Z)


            return(Z)

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32,3,strides=2,input_shape=[224,224,3]))

    ## Filter and stride pairs

    for i in zip([32,64,128,256],[1,2]*4):

        if i[0] == 128 or i[0]==256:

            model.add(mobile_net_unit(filters_1 = i[0],filters_2 = i[0],strides_1 = i[1],strides_2 = 1,size_1 = 3, size_2 = 1)),
            model.add(mobile_net_unit(filters_1 = i[0],filters_2 = i[0]*2,strides_1 = i[1],strides_2 = 1,size_1 = 3, size_2 = 1))



        else:

            model.add(mobile_net_unit(filters_1 = i[0],filters_2 = i[0]*2,strides_1 = i[1],strides_2 = 1,size_1 = 3, size_2 = 1))




    for i in range(5):

        model.add(mobile_net_unit(filters_1 = 512,filters_2 = 512,strides_1 = 1,strides_2 = 1,size_1 = 3, size_2 = 1))

    model.add(mobile_net_unit(filters_1 = 512,filters_2 = 1024,strides_1 = 2,strides_2 = 1,size_1 = 3, size_2 = 1))

    model.add(mobile_net_unit(filters_1 = 1024,filters_2 = 1024,strides_1 = 1,strides_2 = 1,size_1 = 3, size_2 = 1))

    model.add(keras.layers.AvgPool2D(pool_size=7,strides=1,padding='valid'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(80, activation = 'softmax'))

    model.build()
    model.summary()
    return(model)




