#!/usr/bin/python3
import os
import sys
import argparse
from itertools import product
import subprocess
from pathlib import Path
import numpy as np

# Define the arguments
parser = argparse.ArgumentParser(description='Run experiments for the random graph.')
parser.add_argument(
        "--experiment", required=True, choices=['agents', 'density', 'values'], type=str, help="Name of the experiment"
    )
parser.add_argument(
        "--solver", required=True, type=str, help="Name of the algorithm (or 'all')"
    )
parser.add_argument(
		"--overwrite", required=False, default=False, action='store_true', help="Overwrite all results"
	)
args = parser.parse_args()

# Set the arguments
experiment = args.experiment
overwrite = args.overwrite
solver = args.solver

# 
available_solver_list = ['problem', 'dpop', 'afdpop', 'dbay', 'duct', 'sdgibbs']
if solver == 'all':
    available_solver_list = ['dpop', 'afdpop', 'duct', 'sdgibbs']
    solver_list = available_solver_list
else:
    if not solver in available_solver_list:
        print(f'Solver not in the available solver list:\n{solver} -> {available_solver_list}')
        sys.exit(1)
    solver_list = [solver]

relation_list = ['bimodal_bird']

if experiment == 'agents':
    agent_list = [3, 4, 5, 6, 7, 8, 9, 10]
    density_list = [0.2]
    # if solver == 'dbay':
    #     value_list = [15]
    # else:
    #     value_list = [50]
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif experiment == 'density':
    agent_list = [6]
    density_list = list(np.arange(start=0.1, stop=0.40+0.02, step=0.02))
    # if solver == 'dbay':
    #     value_list = [15]
    # else:
    #     value_list = [50]
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
elif experiment == 'values':
    agent_list = [6]
    density_list = [0.2]
    # if solver == 'dbay':
    #     value_list = list(np.arange(start=4, stop=15+1, step=1))
    # else:
    #     value_list = list(np.arange(start=4, stop=50+1, step=1))
    seed_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
else:
    print('This should never be reached (due to argparser choices)')
    sys.exit(1)

dcop_cli_path = Path.home() / "pyDcop/pydcop/dcop_cli.py"
data_directory = Path.home() / "pyDcop_out"

for relation, seed_number, agent_count, graph_density in product(relation_list, seed_list, agent_list, density_list):
    try:
        # create the problem
        problem_name = f'continuous_random_{relation}_{agent_count}_{graph_density:3.2f}_{seed_number}'
        print(f'============= {problem_name} =============')

        directory_problem = f'{data_directory}/{problem_name}'
        if not os.path.isdir(directory_problem):
            os.makedirs(directory_problem, exist_ok=True)
            print(f'Created the problem directory: {directory_problem}')
        filename_problem = Path(f"{directory_problem}/{problem_name}.yaml")

        if not filename_problem.exists() or overwrite:
            args_generate = [
                "python3",
                f"{dcop_cli_path}", 
                "--output",
                f"{filename_problem}",
                "generate",
                "cdcop_random_bird",
                "--agent_count",
                f"{agent_count}",  
                "--graph_density",
                f"{graph_density}",
                "--random_seed",
                f"{seed_number}",
            ]
            subprocess.run(args_generate, check=True)
        else:
            print(f'Problem file already exits: {filename_problem}')

    except Exception as e:
        print(e)
