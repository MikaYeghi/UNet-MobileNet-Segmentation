from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, UpSampling2D, Concatenate
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
from tensorflow.keras.applications import MobileNetV2

def model(image_size=256, num_of_classes=21):
    inputs = Input(shape=(image_size, image_size, 3), name="input_image")                                           # define the input
    
    # Pre-trained encoder
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)                   # define the encoder (MobileNetV2)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]    # skip connection layers (concatenated with the upsampling path)
    encoder_output = encoder.get_layer("block_13_expand_relu").output                                               # (16, 16, 196)
    
    # Decoder
    f = [32, 48, 64, 96, 128]           # channels
    x = encoder_output                  # encoder output
    
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    
    x = Conv2D(num_of_classes, (1, 1), padding="same")(x)       # output layer has NUM_OF_CLASSES neurons
    x = Activation("softmax")(x)                                # softmax activation instead of sigmoid
    
    model = Model(inputs, x)
    return model