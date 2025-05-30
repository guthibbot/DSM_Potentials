import numpy as np
import pandas as pd
import glob
import sys
import os
import logging
import argparse
import re

# Set up argument parser
parser = argparse.ArgumentParser(description="Process energy data from a specified folder.")
parser.add_argument("scenario", type=str, help="Scenario to take base in.")
parser.add_argument("mod", type=str, help="Modification chosen.")
args = parser.parse_args()

# File paths
scenario = args.scenario
mod = args.mod
log_path = f"logs/{scenario}_{mod}/"
output_path = f"data/{scenario}_{mod}/"

# Ensure existence of folders
os.makedirs(log_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# Set up logging
log_file_path = os.path.join(log_path, "load_profile_creator.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[
    logging.FileHandler(log_file_path, mode="w"),  # Overwrite log file each run
    logging.StreamHandler(sys.stdout)              # Print to console as well
])
logger = logging.getLogger()

# transport
transport_24 = np.array([16, 15, 14, 12.5, 9.5, 2.5, 3, 4, 5.5, 6, 5.5, 5, 4.5, 4, 3, 2, 2.5, 4, 6, 8, 10, 17, 16.5, 16])
lp_transport = transport_24/transport_24.sum()
# sanitaryHeating
sanitaryHeating_48 = np.array([
    0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2, 1.9, 4, 5, 6.8, 7.2, 7.5,
    7.2, 6.9, 5.2, 4.3, 3, 2.5, 2.1, 2, 1.9, 1.8, 1.8, 1.8, 1.8, 1.8, 1.9,
    2, 2.3, 2.5, 2.8, 4, 4.5, 5, 5.2, 4.9, 4.3, 3.8, 2.9, 2.4, 2, 1, 0.3,
    0.1, 0.05, 0])
sanitaryHeating_24 = [np.mean(sanitaryHeating_48[i:i+2]) for i in range(0, 48, 2)]
lp_sanitaryHeating = sanitaryHeating_24/np.sum(sanitaryHeating_24)
# industry
lp_industry = np.ones(24)/np.ones(24).sum()
# hydrolysis
lp_hydrolysis = np.ones(24)/np.ones(24).sum()
# residentialTertiary
residentialTertiary_168 = np.array([
    26, 25.5, 25.5, 28, 29, 31, 35, 37, 40, 41.5, 43, 43.5, 41.5, 41, 43, 42.5, 40, 37, 35, 33, 31, 29, 28, 26,
    25, 26, 26, 28, 29, 31, 33, 37, 39, 41, 41.5, 41, 39.5, 39, 41, 40, 38.5, 37, 35, 33, 31, 29, 28.5, 28,
    28, 29, 30, 32, 33, 35, 38, 39, 40, 42, 43.5, 44, 43, 42, 44.5, 43, 42, 40, 38, 35, 32, 31, 30, 29,
    29, 29, 30, 32, 33, 35, 39, 41, 42.5, 43.5, 44, 43.5, 42.5, 43, 44.5, 42.5, 41.5, 39.5, 38, 35, 32, 30, 29, 27.5,
    28, 29, 30, 32, 34, 37, 38, 39.5, 41, 42, 43, 43.5, 43, 42, 41.5, 43, 41, 39, 36, 34, 30, 28, 27, 26,
    26, 25.5, 24, 26, 28, 30, 32, 34, 35, 36, 35, 34, 33.5, 34, 35, 36, 34, 32, 31.5, 28, 27, 26.5, 26, 25,
    22, 21.5, 22, 23, 25, 27, 29, 31, 32, 33, 32.5, 31.5, 31, 31.5, 32, 33.5, 33, 32, 31.5, 30, 29, 28, 27, 26
])
lp_residentialTertiary = residentialTertiary_168/residentialTertiary_168.sum()

load_profiles = {
    'hydrolysis': lp_hydrolysis,
    'industry': lp_industry,
    'residentialTertiary': lp_residentialTertiary,
    'sanitaryHeating': lp_sanitaryHeating,
    'spaceHeating': 1,
    'transport': lp_transport
}
logger.info(f"Created load profile dictionary")

output_filename_load_profiles = os.path.join(output_path, "load_profiles.npy")
np.save(output_filename_load_profiles, load_profiles, allow_pickle=True)
logger.info(f"Saved load profiles to {output_filename_load_profiles}")

## DSM data
DSM_data = pd.DataFrame(data=
    {'Category': ['hydrolysis', 'industry', 'residentialTertiary', 'sanitaryHeating', 'spaceHeating', 'transport'],
    'max_fraction': [1.2, 1.1, 1.0, 2.0, 1.1, 1.0],
    'tau': [72, 4, 4, 12, 4, 24],
    'flex_share': [0.8, 0.4, 0.1, 0.5, 0.5, 0.5]
    } 
).set_index("Category")

# Define default mod actions
mod_multipliers = {
    "no": None,   # Skip saving
    "real": 1.0,  # No change
}

# Apply transformations based on mod
if mod in mod_multipliers:
    multiplier = mod_multipliers[mod]
    if multiplier is not None:
        DSM_data *= multiplier  # Apply global multiplier
elif mod == "full":
    DSM_data["max_fraction"] = 10
    DSM_data["tau"] = 168
    DSM_data["flex_share"] = 1
elif mod in ["max_fraction", "tau", "flex_share"]:
    # Set only the specific column if mod is one of them
    DSM_data[mod] = {"max_fraction": 10, "tau": 168, "flex_share": 1}[mod]
else:
    logger.error(f"Unknown mod '{mod}', exiting.")
    sys.exit(1)

# Save DSM data unless 'no' mod is used
if mod != "no":
    output_filename_DSM_data = os.path.join(output_path, "DSM_data.csv")
    DSM_data.to_csv(output_filename_DSM_data)
    logger.info(f"Saved modified DSM data to {output_filename_DSM_data}")
else:
    logger.info("Skipping DSM data saving as per 'no' mod")

logger.info("Script completed.")


# elif "_" in mod:
#     # Parse column-specific modification (e.g., "flex_share_1.5")
#     match = re.match(r"(max_fraction|tau|flex_share)_(\d+(\.\d+)?)", mod)
#     if match:
#         column, value = match[1], float(match[2])
#         DSM_data[column] *= value
#     else:
#         logger.error(f"Unknown mod '{mod}', exiting.")
#         sys.exit(1)
# else:
#     logger.error(f"Unknown mod '{mod}', exiting.")
#     sys.exit(1)

# # DSM data
# DSM_data = pd.DataFrame(data=
#     {'Category': ['hydrolysis', 'industry', 'residentialTertiary', 'sanitaryHeating', 'spaceHeating', 'transport'],
#     'max_fraction': [1.2, 1.1, 1.0, 2.0, 1.1, 1.0],
#     'tau': [72, 4, 4, 12, 4, 24],
#     'flex_share': [0.8, 0.4, 0.1, 0.5, 0.5, 0.5]
#     } 
# ).set_index("Category")
# logger.info(f"Created DSM data dataframe")

# output_filename_DSM_data = os.path.join(output_directory, "DSM_data.csv")
# DSM_data.to_csv(output_filename_DSM_data)
# logger.info(f"Saved DSM data to {output_filename_DSM_data}")
