import pypsa
import numpy as np
import pandas as pd
import glob
import sys
import os
import logging
import gurobipy as gp
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process energy data from a specified folder.")
parser.add_argument("scenario", type=str, help="Scenario to take base in.")
parser.add_argument("mod", type=str, help="Modification chosen.")
parser.add_argument("--memlimit", type=str, default=None, help="Set memory limit (optional).")
args = parser.parse_args()

# File paths
cwd = os.getcwd()
memlimit = args.memlimit
scenario = args.scenario
mod = args.mod
data_path = f"thesis/data/{scenario}_{mod}/"
log_path = f"thesis/logs/{scenario}_{mod}/"
network_path = f"thesis/networks/mod/{scenario}_{mod}.nc"
output_path = f"thesis/networks/solved/{scenario}_{mod}_solved.nc"

# Ensure existence of folders
os.makedirs(log_path, exist_ok=True)

# Set up logging
log_file_path = os.path.join(log_path, "network_solver.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[
    logging.FileHandler(log_file_path, mode="w"),  # 'w' to overwrite each run
    logging.StreamHandler(sys.stdout)              # Print to console as well
])
logger = logging.getLogger()


n = pypsa.Network(network_path)


os.environ["GRB_LICENSE_FILE"] = "/work/thesis/gurobi.lic"
kwargs = {
    "threads": 0,
    "method": 2,  # barrier
    "crossover": 0,
    "BarConvTol": 1.e-5,
    "FeasibilityTol": 1.e-4,
    "OptimalityTol": 1.e-4,
    "Seed": 123,
    "AggFill": 0,
    "PreDual": 0,
    "GURO_PAR_BARDENSETHRESH": 200
}

# Only add memlimit if it's provided
if memlimit:
    kwargs["memlimit"] = int(memlimit)  # Ensure it's an integer

n.optimize(solver_name='gurobi', keep_files=False, solver_options=kwargs)

n.export_to_netcdf(output_path)
logger.info(f"Solved and saved modified network")
logger.info("Script completed.")

