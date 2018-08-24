from src.params import Params
from src.optimizer import OptimizerGA

def generate_nmist_dataset():
    import keras
    from keras.datasets import mnist
    inputs = 784 # Images for 28x28
    outputs =  10 # clasess
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, outputs)
    y_test = keras.utils.to_categorical(y_test, outputs)
    return (x_train, y_train), (x_test, y_test)

variable_params = {
    "hidden_layers": range(1, 5),
    "neurons": [2 ** x for x in range(6, 12)],
    "dropout": [ x / 10 for x in range(1, 10)],
    "activation_functions": ['relu', 'tanh', 'sigmoid'],
}

def generate_model_ann(params):
    '''Create a simple neural network model with the input params'''

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import RMSprop
    
    inputs = 784 # Images for 28x28
    outputs =  10 # clasess

    model = Sequential()
    model.add(Dense(params['neurons'], activation=params['activation_functions'], input_shape=(inputs,)))
    model.add(Dropout(params['dropout']))
    for layer in range(1, params['hidden_layers']):
        model.add(Dense(params['neurons'], activation=params['activation_functions']))
        model.add(Dropout(params['dropout']))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])
    return model


# Sample
# Sample
def sample_nmist(iterations, epochs, path):
    train, test = generate_nmist_dataset()
    params = Params(variable_params)
    print(params.optimize_params)

    print(len(train))

    optimizer = OptimizerGA(train, test, params, generate_model_ann)
    optimizer.verbose_train = 0 
    optimizer.epochs_train = epochs 
    optimizer.generate_population(10)
    for i in range(0, iterations):
        print("=> Generación ", i)
        optimizer.evolve(i == 0)
        print(optimizer.population_score())
    print(optimizer.population)
    optimizer.population_save(path)
