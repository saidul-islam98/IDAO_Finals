from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

target_height = 144
target_width = 144

def load_model(weight_path):
    inputs = Input(shape=(target_height, target_width, 1))

    x = Conv2D(64, (3,3), activation='relu', input_shape=(target_height,target_width,1))(inputs)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), activation='relu', input_shape=(target_height,target_width,1))(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), activation='relu', input_shape=(target_height,target_width,1))(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), activation='relu', input_shape=(target_height,target_width,1))(x)
    x = MaxPooling2D(2,2)(x)
    '''
    x = Conv2D(64, (3,3), activation='relu', input_shape=(target_height,target_width,1))(x)
    x = MaxPooling2D(2,2)(x)
    '''

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    class_output = Dense(1, activation='sigmoid', name='class')(x)
    energy_output = Dense(1, name='energy')(x)

    model = Model(inputs=inputs, outputs=[class_output, energy_output])

    # checkpoint_filepath = root+'weights.hdf5'
    model.load_weights(weight_path)

    return model