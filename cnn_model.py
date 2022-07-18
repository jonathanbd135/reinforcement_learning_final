from keras.models import  Model
from keras.layers import  Dense
from keras.layers import Conv2D,  MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras import backend as K


def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


num_of_action = 5 #TODO defune by the number of actions
discount_factor = 0.9 #TODO use and add as argument and consider decing


def cnn_model(model_input, loss='mse'):
    print('Building model...')
    layer = model_input
    conv_1 = Conv2D(filters=32, kernel_size=(3,3),
                    padding='same', activation='relu', name='conv_1')(layer)
    conv_2 = Conv2D(filters=64, kernel_size=(2,2), strides=2,
                    padding='same', activation='relu', name='conv_2')(conv_1)
    conv_3 = Conv2D(filters=128, kernel_size=(3,3),
                    padding='same', activation='relu', name='conv_3')(conv_2)
    pool_1 = MaxPooling2D((2, 2))(conv_3)

    flatten1 = Flatten()(pool_1)
    layer_1 = Dense(256, activation='relu')(flatten1)
    layer_2 = Dense(128, activation='relu')(layer_1)
    layer_3 = Dense(64, activation='relu')(layer_2)
    layer_4 = Dense(32, activation='relu')(layer_3)
    output = Dense(num_of_action, name='preds')(layer_4)

    model_output = output
    model = Model(model_input, model_output)
    opt = RMSprop(learning_rate=0.001)  # Optimizer
    if loss=='huber':
        loss = huber_loss
    model.compile(
            loss=loss,
            optimizer=opt,
            metrics=['accuracy']
        )
    return model
