import random
from libs.params import Params
from libs.model_facade import ModelFacade
import json

class OptimizerGA():
    
    def __init__(self, train, test, params_model:Params, model_function):
        self.params = params_model
        self.train = train
        self.test = test
        self.population = []
        self.model_function = model_function
        self.model_generator = ModelFacade()
        self.verbose_train = 0
        self.batch_train = 64
        self.epochs_train = 1

    def population_score(self):
        scores = []
        for p in self.population:
            scores.append(p["score"])
        return scores

    def population_save(self, file="population.txt"):
        text_file = open(file, "w")
        for p in self.population:
            text_file.write("Score: {}, Acc: {} \n Params: \n".format(p["score"], p["history"].history["acc"]))
            text_file.write(json.dumps(p["params"]))
            text_file.write("\n")
        text_file.close()

    def generate_population(self, size):
        population = []
        for i in range(0, size):
            params = self.params.generate_random_params()
            model = self.model_generator.load(self.model_function, params)
            population.append({
                "params": params,
                "model": model
            })
        self.population = population
        return population

    def mutate(self, model_params, prob=0.5):
        for param in model_params:
            if prob > random.random():
                print("Mutate param", param)
                model_params[param] = self.params.get_rand_value(param) 
        return model_params

    def generate_child(self, father_params, mother_params, balance=0.5):
        child = {}
        for param in father_params:
            if balance > random.random():
                value = father_params[param]
            else:
                value = mother_params[param]
            child[param] = value
        return child

    def train_population(self, train, test, epochs=1, batch_size=32, verbose=0):
        x_train, y_train = train
        x_test, y_test = test
        for value in self.population:
            if "history" in value.keys():
                print("Modelo entrenado previamente.")
                continue
            model = value["model"]
            value["history"] = model.fit(x_train, y_train,
                            batch_size=self.batch_train,
                            epochs=self.epochs_train,
                            verbose=self.verbose_train,
                            validation_data=(x_test, y_test))
            value["score"] = self.model_generator.model_score(value["history"])

    def evolve(self, initial_train, best_prune=0.3, random_variety=0.1, mutation_factor=0.3):
        # Initial Train population
        if initial_train:
            self.train_population(self.train, self.test)
        # Sort by score
        self.population = sorted(self.population, key=lambda model: model["score"], reverse=True)
        size = len(self.population)
        prune = int(size * best_prune)
        variety = int(size * random_variety)
        # Choose the best
        best_poplation = self.population[:prune]
        # Add random values for variety
        best_poplation = best_poplation + random.sample(self.population[prune:], variety)
        # Re-populate generating childs
        
        n_childs = size - prune - variety
        childs = []
        for _ in range(0, n_childs):
            father = random.choice(best_poplation)
            mother = random.choice(best_poplation)
            # Generate child
            child = self.generate_child(father["params"], mother["params"])
            # Mutation probability
            if mutation_factor > random.random():
                child = self.mutate(child)
            model = self.model_generator.load(self.model_function, child)
            history = model.fit(self.train[0], self.train[1],
                        batch_size=self.batch_train,
                        epochs=self.epochs_train,
                        verbose=self.verbose_train,
                        validation_data=self.test)
            child = self.generate_child(father["params"], mother["params"])
            childs.append({
                "params": child,
                "model": model,
                "history": history,
                "score":  self.model_generator.model_score(history)
            })
            # Random mutation
        self.population = best_poplation + childs
        return best_poplation + childs