import pypsa
import numpy as np
import pandas as pd
import glob
import sys
import os
import logging
import argparse

from numpy.lib.stride_tricks import sliding_window_view

# Set up argument parser
parser = argparse.ArgumentParser(description="Process energy data from a specified folder.")
parser.add_argument("scenario", type=str, help="Scenario to take base in.")
parser.add_argument("mod", type=str, help="Modification chosen.")
args = parser.parse_args()

# File paths
scenario = args.scenario
mod = args.mod
data_path = f"data/{scenario}_{mod}/"
log_path = f"logs/{scenario}_{mod}/"
network_path = f"networks/base/{scenario}.nc"
output_path = f"networks/mod/{scenario}_{mod}.nc"

# Ensure existence of folders
os.makedirs(log_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

# Set up logging
log_file_path = os.path.join(log_path, "load_data_transformer.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[
    logging.FileHandler(log_file_path, mode="w"),  # Overwrite log file each run
    logging.StreamHandler(sys.stdout)              # Print to console as well
])
logger = logging.getLogger()

# Load network file
n = pypsa.Network(network_path)
logger.info(f"Loaded network file from {network_path}")

# apply load increase factor
load_factors = pd.read_csv(f"{data_path}load_factors.csv") # load increase factors
# Ensure column names match (assuming 'Node' and 'Factor' in load_factors)
load_factors.set_index("node", inplace=True)  # Set Node as index for easy mapping

# Apply load increase factor
n.loads["p_set"] *= n.loads["bus"].map(load_factors["factor"])

logger.info("Applied load increase factors to network loads.")

# For no DSM
if mod == "no":
    logger.info("No modifications made only load factor application.")
    n.export_to_netcdf(output_path)
    logger.info(f"Exported network file to {output_path}")
else:
    # Load data
    DSM_data = pd.read_csv(f"{data_path}DSM_data.csv") # DSM data dataframe
    DSM_data = DSM_data.set_index("Category")
    load_profiles = np.load(f"{data_path}load_profiles.npy", allow_pickle=True).item() # Load profiles dictionary
    load_factors = pd.read_csv(f"{data_path}load_factors.csv") # load increase factors
    load_share = pd.read_csv(f"{data_path}load_shares.csv", index_col=0) # load shares csv
    # reordering load_share to match network data
    load_share = load_share.reindex(columns=n.loads.index)
    logger.info(f"Loaded all data files")

    ### Create DSM Load Dataframe
    nodes = n.loads.index # Extract all node names
    categories = DSM_data.index # Extract all DSM categories
    DSM_loads = pd.DataFrame(index=n.snapshots, columns=pd.MultiIndex.from_product([nodes, categories])) # Create MultiIndex DataFrame: (time, (node, category))
    total_loads = n.loads_t.p_set.loc[n.snapshots]  # Get total loads for all nodes, Shape: (timesteps, nodes) 

    # Iterate over DSM categories and apply vectorized operations
    for j in categories:
        logger.info(f"Load profile calculation for sector {j} in all nodes")
        if j in ['industry', 'hydrolysis']:
            # Convert to NumPy array to avoid dimension mismatch
            mean_loads = total_loads.mean(axis=0).values  # Shape: (nodes,)
            load_share_values = load_share.loc[j].values  # Shape: (nodes,)
            
            # Perform element-wise multiplication
            DSM_loads.loc[:, (nodes, j)] = mean_loads * load_share_values

        elif j in ['sanitaryHeating', 'transport']:
            daily_profile = pd.Series(load_profiles[j], index=np.arange(24))  # 24-hour profile
            daily_loads = total_loads.resample("D").sum()  # Aggregate to daily total
            # Expand daily values across 24 hours
            expanded_load = np.repeat(daily_loads.values, 24, axis=0)[:len(total_loads)]
            num = np.tile(daily_profile.values, len(daily_loads))[:len(total_loads)]
            num = num[:, None]  # Reshape from (8760,) to (8760,1)
            # Compute DSM load (vectorized)
            DSM_loads.loc[:, (nodes, j)] = load_share.loc[j].values * expanded_load * num

        if j == 'residentialTertiary':
            # Get first weekday for shifting weekly profile
            start_day_of_week = total_loads.index[0].weekday()
            shifted_profile = np.roll(load_profiles[j], -start_day_of_week * 24)
            weekly_profile = pd.Series(shifted_profile, index=np.arange(168))  # 168-hour profile
            weekly_loads = total_loads.resample("W").sum()
            # Expand weekly values across 168 hours
            expanded_load = np.repeat(weekly_loads.values, 168, axis=0)[:len(total_loads)]
            # Fix shape mismatch
            num = np.tile(weekly_profile.values, len(weekly_loads))[:len(total_loads)]
            num = num[:, None]  # Reshape from (8760,) to (8760,1)
            # Perform element-wise multiplication
            DSM_loads.loc[:, (nodes, j)] = load_share.loc[j].values * expanded_load * num

    # Define DSM categories to subtract (excluding spaceHeating)
    filtered_categories = ['hydrolysis', 'industry', 'residentialTertiary', 'sanitaryHeating', 'transport']
    # Compute total DSM load excluding 'spaceHeating'
    total_dsm_load = DSM_loads.loc[:, (nodes, filtered_categories)].T.groupby(level=0).sum().T
    # Create an iterator for all (node, 'spaceHeating') tuples
    space_heating_cols = [(node, 'spaceHeating') for node in nodes]
    # Assign values all at once
    DSM_loads.loc[:, space_heating_cols] = np.maximum(0, (total_loads.values - total_dsm_load.values))
    # Export
    # To save the DataFrame
    DSM_loads.to_pickle(f'data/{scenario}_{mod}/DSM_loads.pkl')
    
    # Reassign network loads
    logger.info(f"Reapply DSM adjusted load")
    n.loads_t.p_set = DSM_loads.groupby(level=0, axis=1).sum()

    ### Add DSM Infrastructure To Network
    n_nodes = n.buses.index
    for i in n_nodes:
        logger.info(f"Add DSM infrastructure for node {i}")
        x_loc = n.buses.x[f"{i}"]
        y_loc = n.buses.y[f"{i}"]
        for j in DSM_data.index:
            n.add("Bus", f"{i}_DSM_bus_{j}", carrier="AC", x=x_loc , y=y_loc)
            total_load = n.loads_t.p_set.loc[:, i] # full load of given node
            L_sched = DSM_loads.loc[:, (i, j)] # original load of the given sector
            L_max = L_sched.max() # maximum load
            L_flex = L_sched*DSM_data["flex_share"].loc[j] # flexible load
            delta_t = int(DSM_data['tau'].loc[j]) # timeframe of flexibility
            max_frac = DSM_data['max_fraction'].loc[j]

            # Approach 2:
            p_max = L_max*max_frac-L_sched # (Lambda*s_inc-L_flex)
            p_max = p_max.where(p_max > 0, 0)
            p_min = -L_flex #(L_flex-Lambda*s_dec) assuming s_dec=0
            p_min = p_min.where(p_min < 0, 0)

            # E calc
            arr = L_flex.values
            nn = len(L_sched)
            wrapped_arr = np.concatenate([arr, arr])
            # For e_max (forward window)
            windows_max = sliding_window_view(wrapped_arr, delta_t)[:nn]
            e_max = pd.Series(np.sum(windows_max, axis=1), index=L_sched.index, name='e_max')
            # For e_min (backward window)
            windows_min = sliding_window_view(wrapped_arr, delta_t)[nn-delta_t:2*nn-delta_t]
            e_min = pd.Series(-np.sum(windows_min, axis=1), index=L_sched.index, name='e_min')
            e_min.iloc[0] = -1e-6 # forcing first E-value to close 0 to force cyclic around 0

            # Nom calc
            p_nom = max(max(abs(p_max)), max(abs(p_min))) # can result in division by 0
            if p_nom == 0:
                logger.info(f"p_nom = 0 for mod: {mod}, node: {i} and sector: {j}")
                p_nom=L_max*DSM_data["flex_share"].loc[j]
            e_nom = max(max(abs(e_max)), max(abs(e_min)))
            #e_nom = max(max(abs(e_max)), abs(min(abs(e_min))))

            if j in ["hydrolysis", "industry"]:
                e_max_pu = 1
                e_min_pu = -1
            else:
                e_max_pu = (e_max/e_nom).astype(float).fillna(0)
                e_min_pu = (e_min/e_nom).astype(float).fillna(0)
            
            # Add DSM links
            n.add(
                "Link",
                f"{i}_DSM_link_{j}",
                bus0=f"{i}",
                bus1=f"{i}_DSM_bus_{j}",
                carrier="AC",
                p_nom=p_nom,
                p_max_pu = (p_max/p_nom).astype(float).fillna(0), 
                p_min_pu = (p_min/p_nom).astype(float).fillna(0),
                efficiency=1,
                marginal_cost=0,
                overwrite=True,
            )
            # Add DSM Store
            n.add(
                "Store",
                name=f"{i}_DSM_store_{j}",
                bus=f"{i}_DSM_bus_{j}",
                carrier="AC",
                e_nom=e_nom,
                e_max_pu = e_max_pu,
                e_min_pu = e_min_pu,
                e_cyclic=True,
                overwrite=True,
            )

    n.export_to_netcdf(output_path)
    logger.info(f"Exported network file to {output_path}")

logger.info("Script completed.")
