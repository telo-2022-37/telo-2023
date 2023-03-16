from jmetal.algorithm.multiobjective import NSGAII
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.solution import get_non_dominated_solutions
from jmetal.util.observer import ProgressBarObserver
from multiprocessing import Pool
from timebudget import timebudget
from optimisation import MinimaxOptimisation
from functions import project_seeds, write_front_tofile, write_solutions_tofile

def the_optimisation(rseed):
    evaluations = 500
    pop_size = 200

    moop3 = MinimaxOptimisation(
        target_id=0,
        test_metric="MAE",
        random_seed=rseed,
        data_folder='data',
        results_folder='results/moop3/')

    solver = NSGAII(
        problem = moop3,
        population_size = pop_size,
        offspring_population_size = pop_size,
        mutation = PolynomialMutation(probability=1.0/moop3.number_of_variables, distribution_index=20),
        crossover = SBXCrossover(probability=0.8, distribution_index=20), 
        termination_criterion = StoppingByEvaluations(max_evaluations=evaluations))

    #Run Algorithm
    progress_bar = ProgressBarObserver(max=evaluations)
    solver.observable.register(progress_bar)
    solver.run()

    #get front and write to file
    front = get_non_dominated_solutions(solver.get_result())
    write_front_tofile(moop3, front)

    # write all solutions to csv file
    write_solutions_tofile(moop3)

@timebudget
def run_the_optimisation(the_optimisation, random_seeds, pool):
    pool.map(the_optimisation, random_seeds)

if __name__ == "__main__":

    n_repeatitions = 5
    random_seeds = project_seeds(1234, n_repeatitions)

    process_pool = Pool(n_repeatitions)
    run_the_optimisation(the_optimisation, random_seeds, process_pool)
