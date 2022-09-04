import tensorflow as tf
from tensorflow import keras


def model_def():
    

    base_model =tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        
        
    weights='imagenet',
        
    include_preprocessing = False,
    

    include_top=False
    )

    avg = keras.layers.GlobalMaxPool2D()(base_model.output)

    output = keras.layers.Dense(85,activation='softmax')(avg)

    model = keras.Model(inputs = base_model.input,outputs = output)



    






#     class ReLU6(keras.layers.Layer):
    
#         def __init__(self,**kwargs):
#             super().__init__(**kwargs)


#         def call(self,inputs):

#             Z = inputs

#             return tf.nn.relu6(Z)

        
        
        
        
#     class bottleneck_residual_block(keras.layers.Layer):
#         def __init__(self,filters_1,strides_1,expansion_factor,**kwargs):
#             super().__init__(**kwargs)

#             self.main_layers = [

#                 keras.layers.Conv2D(filters=filters_1*expansion_factor,kernel_size=1,strides=strides_1,padding='same'),
#                 keras.layers.BatchNormalization(),
#                 ReLU6(),
#                 keras.layers.DepthwiseConv2D(depth_multiplier = expansion_factor,kernel_size = 3,strides = 1,padding='same'),
#                 keras.layers.BatchNormalization(),
#                 ReLU6(),
#                 keras.layers.Conv2D(filters=filters_1,kernel_size=1,strides=1,padding='same'),
#                 keras.layers.BatchNormalization()







#             ]
#             self.skip_layer = [

#                  keras.layers.Conv2D(filters=filters_1,kernel_size=1,strides=strides_1,padding='same'),


#             ]

#         def call(self,inputs):
#             Z = inputs
#             Z1 = inputs

#             for layer in self.skip_layer:
#                 Z1 = layer(Z1)

#             for layer  in self.main_layers:

#                 Z = layer(Z)

#             print(Z.shape)
#             return(Z1+Z)
        
        
        
#     class bsb_2(keras.layers.Layer):
#         def __init__(self,filters_1,strides_1,expansion_factor,**kwargs):
#             super().__init__(**kwargs)

#             self.main_layers = [

#                 keras.layers.Conv2D(filters=filters_1*expansion_factor,kernel_size=1,strides=strides_1,padding='same'),
#                 keras.layers.BatchNormalization(),
#                 ReLU6(),
#                 keras.layers.DepthwiseConv2D(depth_multiplier = expansion_factor,kernel_size = 3,strides = 1,padding='same'),
#                 keras.layers.BatchNormalization(),
#                 ReLU6(),
#                 keras.layers.Conv2D(filters=filters_1,kernel_size=1,strides=1,padding='same'),
#                 keras.layers.BatchNormalization()







#             ]
#             self.skip_layer = [



#             ]

#         def call(self,inputs):
#             Z = inputs
#             Z1 = inputs
#             for layer  in self.main_layers:

#                 Z = layer(Z)

#             print(Z.shape)
#             return(Z)

#     model = keras.models.Sequential()

#     model.add(keras.layers.Conv2D(32,2,strides=2,input_shape=[224,224,3]))
#     model.add(bottleneck_residual_block(filters_1=16,strides_1=1,expansion_factor=1))

#     model.add(bsb_2(filters_1=24,strides_1=2,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=24,strides_1=1,expansion_factor=6))

#     model.add(bsb_2(filters_1=32,strides_1=2,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=32,strides_1=1,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=32,strides_1=1,expansion_factor=6))


#     model.add(bsb_2(filters_1=64,strides_1=2,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=64,strides_1=1,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=64,strides_1=1,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=64,strides_1=1,expansion_factor=6))




#     model.add(bottleneck_residual_block(filters_1=96,strides_1=1,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=96,strides_1=1,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=96,strides_1=1,expansion_factor=6))



#     model.add(bsb_2(filters_1=160,strides_1=2,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=160,strides_1=1,expansion_factor=6))
#     model.add(bottleneck_residual_block(filters_1=160,strides_1=1,expansion_factor=6))

#     model.add(bottleneck_residual_block(filters_1=320,strides_1=1,expansion_factor=6))
#     model.add(keras.layers.Conv2D(1280,kernel_size=1,strides=1,activation='relu'))

#     model.add(keras.layers.BatchNormalization())

#     model.add(keras.layers.AvgPool2D(pool_size=7,strides=7,padding='same'))

#     model.add(keras.layers.Conv2D(1280,kernel_size=1,strides=1,padding='same'))

#     model.add(keras.layers.Flatten())

#     model.add(keras.layers.Dense(85,activation = 'softmax'))

    for layer in base_model.layers[:-2]:
        layer.trainable = False

    #model.build()
    model.summary()
    return(model)




