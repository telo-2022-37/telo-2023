
import pandas as pd
import datetime
import random
import os
import csv

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from jmetal.core.solution import FloatSolution


# Data processing functions
def load_datasets(data_folder:str, random=True):
    datasets = []
    dataset_ids = ["{:02}".format(i) for i in range(1,20+1)] #create dataset Ids from 01 to 20
    # kind_id = "Rnd" if random else "Ord"
    for dataset_id in dataset_ids:
        # data = pd.read_csv(f"{data_folder}/allData_m{IDs}_Y_X13_{kind_id}.csv")
        data = pd.read_csv(f"{data_folder}/DS{dataset_id}.csv")
        datasets.append(data)
    return datasets

def prepare_data(data:pd.DataFrame, target_id:int, shuffle=True):
    """Set X and y columns for the dataset
    X runs from column 6 to end. This is according to the thermal datasets.
    y is defined by user
    Returns X and y
    """
    data.sample(frac=1) if shuffle==True else data
    X = data.iloc[:,6:data.shape[1]]
    y = data.iloc[:,[target_id]]  
    return X, y

def _map_emas_and_weights(n_datasets, n_variables):
    weight_positions = [i for i in range(n_datasets, n_variables)] #20-29
    weights = [0.001*2**i for i in range(n_variables - n_datasets)]
    dictionary_of_emas_and_weights = {weight_positions[i]: weights[i] for i in range(len(weight_positions))}
    return dictionary_of_emas_and_weights

def _add_columns_based_on_emas(data, selected_weights, dictionary_of_emas_and_weights):
    pd.set_option("mode.chained_assignment", None) # suppress SettingWithCopyWarning
    new_data = data.iloc[:,0:12]
    for weight in selected_weights:
        for col_name in data.columns:        
            if str(dictionary_of_emas_and_weights[weight]) in col_name:
                new_data[col_name] = data.loc[:,col_name]
    return new_data

def recreate_datasets_with_emas(solution, datasets):
    this_solution = solution.variables if type(solution)==FloatSolution else solution
    dictionary = _map_emas_and_weights(len(datasets), len(this_solution))
    selected_weights =[]
    all_training_data = all_test_data = pd.DataFrame()

    for position, value in enumerate(this_solution):
        if position < 20:       
            if value >= 0.5:
                all_training_data = pd.concat([all_training_data, datasets[position]])            
            else:
                all_test_data = pd.concat([all_test_data, datasets[position]])
        else:        
            if value >= 0.5:
                selected_weights.append(position)

    training_data = _add_columns_based_on_emas(all_training_data, selected_weights, dictionary)
    test_data = _add_columns_based_on_emas(all_test_data, selected_weights, dictionary)
    return training_data, test_data

# Machine learning functions

def standard_scaler(X_train, X_test):
    column_names = X_train.columns # added on 20/02/2023
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=column_names)   # added on 20/02/2023
    X_test = pd.DataFrame(X_test, columns=column_names) # added in 20/02/2023
    return X_train, X_test

def train_model(algorithm_to_train, X_train, y_train):
    trained_model = algorithm_to_train.fit(X_train, y_train) # modified on 20/02/2023
    return trained_model

def test_model(model_to_test, X_test):
    predicted_y = model_to_test.predict(X_test)    
    return predicted_y

def evaluate_model(test_metric:str, y_test, predicted_y):
    if test_metric == "mse" or test_metric == "MSE":
        return metrics.mean_squared_error(y_test, predicted_y) 
    elif test_metric == "mae" or test_metric == "MAE":
        return metrics.mean_absolute_error(y_test, predicted_y) 
    elif test_metric == "r2" or test_metric == "R2":
        return metrics.r2_score(y_test, predicted_y) 

def train_test_lr(train_data, test_data, target_id, test_metric):
    X_train, y_train = prepare_data(train_data, target_id)
    X_test, y_test = prepare_data(test_data, target_id)
    X_train, X_test = standard_scaler(X_train, X_test)
    model = LinearRegression()
    model = train_model(model, X_train, y_train)
    y_pred = test_model(model, X_test)
    metric = evaluate_model(test_metric, y_test, y_pred)
    return metric

def train_test_evaluate(training_data, test_data, target_id, test_metric):
    X_train, y_train, = prepare_data(training_data, target_id)
    X_test, y_test = prepare_data(test_data, target_id)
    X_train, X_test = standard_scaler(X_train, X_test)
    algorithm = LinearRegression()
    trained_model = train_model(algorithm, X_train, y_train)    
    y_pred = test_model(trained_model, X_test)   
    metric = evaluate_model(test_metric, y_test, y_pred)
    return trained_model, metric

# Outputs Management

def project_seeds(seed=1234, k=5):
    random.seed(seed)
    return [random.randint(1111, 9999) for _ in range(k)]

def _time_now():
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    date_time = datetime.datetime.fromtimestamp(ts, tz=None)
    return date_time.strftime("%d-%m-%Y_%H-%M")

def outputs_location(target_id, test_metric, random_seed, results_folder):
    folder= f'{results_folder}/{target_id}/{test_metric}_{_time_now()}/{test_metric}_seed_{random_seed}'
    if not os.path.exists(folder):
       os.makedirs(folder)
    return folder

def write_front_tofile(self, front):
    '''write front to csv file'''    
    file_name= f'{self.file_path}/{self.target_id}_{self.test_metric}_Nondominated_solutions.csv'    
    with open(file_name, 'w') as f:
        for item in front:
            solu = [i for i in item.variables]
            solu.append(item.objectives[0])
            solu.append(item.objectives[1]) 
            
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(solu)

def write_solutions_tofile(self):
    self.all_solutions.to_csv(f"{self.file_path}/{self.target_id}_{self.test_metric}_All_solutions.csv", index=False)

def translate_target_id(target_id):
    if target_id == 0:
        return "Winding Temperature"
    if target_id == 1:
        return "Inner Bearing Temperature"
    if target_id == 2:
        return "Outer Bearing Temperature"
    if target_id == 3:
        return "Steel Temperature"
    if target_id == 4:
        return "Flange Temperature"
    if target_id == 5:
        return "Rotor Temperature"
    if target_id == 10:
        return "Minimax Optimisation"
