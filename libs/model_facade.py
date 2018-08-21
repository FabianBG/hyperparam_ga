
class ModelFacade():

    def __init__(self):
        self.custom_score = self.basic_score

    def load(self, model_function, params):
        return model_function(params)

    def set_score_function(self, score_function):
        self.custom_score = score_function

    def train(self, model, train, test, batch_size=8, epochs=1, verbose=1):
        '''Train a model with a set of specific params'''
        x_train, y_train = train
        x_test, y_test = test
        return model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        validation_data=(x_test, y_test))
    
    def basic_score(self, history):
        acc = history.history['acc'][-1]
        loss = 1 - history.history['loss'][-1]
        acc = acc * 0.5
        loss = loss * 0.5
        return acc + loss
    
    def model_score(self, history):
        return self.custom_score(history)