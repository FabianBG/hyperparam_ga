import random

class Params():
    def __init__(self, optimize_params):
        '''Definir modelo de parámetros'''
        self.optimize_params = optimize_params

    def get_rand_value(self, param):
        '''Choose a random value for a param from template of params'''
        array = self.optimize_params[param]
        secure_random = random.SystemRandom()
        return secure_random.choice(array)

    def generate_random_params(self):
        '''Generate a set of random parameters'''
        params = {}
        for param in self.optimize_params:
            params[param] = self.get_rand_value(param)
        return params