import pandas as pd
import random
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from functions import (
    load_datasets, 
    train_test_lr,
    recreate_datasets_with_emas,
    translate_target_id, 
    outputs_location)

class DatasetsOptimisation(FloatProblem):
    
    def __init__(self, target_id, test_metric, random_seed, data_folder, results_folder):
        super().__init__()

        '''
        Datasets Optimisation. Objective is to minimise training set size and test metric
        :param target_id: target temperature to model
        :param test_metric: test metric to minimise
        :param random_seed: seed to use in experiment
        :param data_folder: location of datasets
        :param results_folder: folder to save results
        '''
        
        #parameters
        self.target_id = target_id
        self.test_metric = test_metric
        self.random_seed = random_seed
        self.data_folder = data_folder
        self.results_folder = results_folder

        #fixed params
        self.number_of_variables = 20
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.number_of_bits = 20
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]        
        self.obj_labels = ['Test Metric', 'Training Set Size']       

        random.seed(self.random_seed)    
        self.datasets = load_datasets(self.data_folder)        
        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]   
         
        self.file_path = outputs_location(
            self.target_id, 
            self.test_metric, 
            self.random_seed, 
            self.results_folder)
        self.all_solutions = pd.DataFrame(columns=[i for i in range(self.number_of_variables)] + [self.test_metric, "size"])
    
    def create_solution(self) -> FloatSolution:
        '''create solution'''
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)

        new_solution.variables = [
            random.uniform(self.lower_bound[i], 
            self.upper_bound[i]) for i in range(self.number_of_variables)]
        return new_solution
    
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        training_data = test_data = pd.DataFrame()

        for position, value in enumerate(solution.variables):
            if value >= 0.5:                
                training_data = pd.concat([training_data, self.datasets[position]])
            else:
                test_data = pd.concat([test_data, self.datasets[position]])
        
        if training_data.shape[0] >1 and test_data.shape[0]>1:            
            test_error = train_test_lr(training_data, test_data, self.target_id, self.test_metric)                    
            solution.objectives[0] = test_error
            solution.objectives[1] = training_data.shape[0]               
        else:            
            solution.objectives[0] = 10000 #dummy
            solution.objectives[1] = 250000 #dummy
        
        evaluated_solution = [i for i in solution.variables] + [solution.objectives[0], solution.objectives[1]]   
        self.all_solutions.loc[len(self.all_solutions.index)] = evaluated_solution    
        return solution

    def get_name(self) -> str:
        return 'Thermal Modelling' 

class DatasetsEMAOptimisation(FloatProblem):
    
    def __init__(self, target_id, test_metric, random_seed, data_folder, results_folder):
        super().__init__()
        self.target_id = target_id
        self.test_metric = test_metric
        self.random_seed = random_seed
        self.data_folder = data_folder
        self.results_folder = results_folder

        self.number_of_variables =30
        self.number_of_objectives =2
        self.number_of_constraints =0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]        
        self.obj_labels = ['Test Metric', 'Training Set Size']       

        random.seed(self.random_seed)    
        self.datasets = load_datasets(self.data_folder)
        
        self.lower_bound= [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound= [1.0 for _ in range(self.number_of_variables)]    
        self.file_path = outputs_location(
            self.target_id, 
            self.test_metric, 
            self.random_seed, 
            self.results_folder)
        self.all_solutions = pd.DataFrame(
            columns=[i for i in range(self.number_of_variables)] + [self.test_metric, "size"])

    def create_solution(self) -> FloatSolution:
        '''create solution'''
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)

        new_solution.variables = [
            random.uniform(self.lower_bound[i], 
            self.upper_bound[i]) for i in range(self.number_of_variables)]
        return new_solution
    
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        
        training_data = test_data = pd.DataFrame()
        training_data, test_data = recreate_datasets_with_emas(solution, self.datasets)

        if training_data.shape[0] >1 and test_data.shape[0]>1:
            test_error = train_test_lr(training_data, test_data, self.target_id, self.test_metric)
            solution.objectives[0] = test_error
            solution.objectives[1] = training_data.shape[0]               
        else:            
            solution.objectives[0] = 10000 #dummy
            solution.objectives[1] = 250000 #dummy
        
        evaluated_solution = [i for i in solution.variables] + [solution.objectives[0], solution.objectives[1]]   
        self.all_solutions.loc[len(self.all_solutions.index)] = evaluated_solution    
        return solution

    def get_name(self) -> str:
        return 'Thermal Modelling' 

class MinimaxOptimisation(FloatProblem):

    def __init__(self, target_id, test_metric, random_seed, data_folder, results_folder):
        super().__init__()
        self.target_id = target_id
        self.test_metric = test_metric
        self.random_seed = random_seed
        self.data_folder = data_folder
        self.results_folder = results_folder
        
        self.number_of_variables =30
        self.number_of_objectives =2
        self.number_of_constraints =0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]        
        self.obj_labels = ['Test Metric', 'Training Set Size']       

        self.target_component = translate_target_id(self.target_id)
        self.target_columns= [i for i in range(6)]
        
        random.seed(self.random_seed)    
        self.datasets = load_datasets(self.data_folder)        
        self.lower_bound= [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound= [1.0 for _ in range(self.number_of_variables)]    
        self.file_path = outputs_location(
            self.target_id, 
            self.test_metric, 
            self.random_seed, 
            self.results_folder)
        self.all_solutions = pd.DataFrame(
            columns=[i for i in range(self.number_of_variables)] + 
                ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"] + 
                [self.test_metric, "size", "max_temp"])
        
    def create_solution(self) -> FloatSolution:
        '''create solution'''
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = [
            random.uniform(self.lower_bound[i], 
            self.upper_bound[i]) for i in range(self.number_of_variables)]
        return new_solution      
    
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        training_data = test_data = pd.DataFrame
        training_data, test_data = recreate_datasets_with_emas(solution, self.datasets)

        if training_data.shape[0] >1 and test_data.shape[0]>1:
            measures = []

            for target in self.target_columns:
                test_error = train_test_lr(training_data, test_data, target, self.test_metric)
                measures.append(test_error)
                
            solution.objectives[0] = max(measures)
            solution.objectives[1] = training_data.shape[0]
            target_with_max = translate_target_id(measures.index(max(measures)))
            
        else:
            solution.objectives[0] = 10000 #dummy
            solution.objectives[1] = 250000 #dummy
            measures = [10 for _ in range(6)] #dummy
            target_with_max = "none" #dummy
        
        evaluated_solution = [i for i in solution.variables] + [i for i in measures] + [solution.objectives[0], solution.objectives[1], target_with_max]   
        self.all_solutions.loc[len(self.all_solutions.index)] = evaluated_solution    
        return solution

    def get_name(self) -> str:
        return 'Thermal Modelling'
