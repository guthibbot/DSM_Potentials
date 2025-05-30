import pandas as pd
import pypsa
import glob
import sys
import os
import logging
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Process energy data from a specified folder.")
parser.add_argument("scenario", type=str, help="Scenario to take base in.")
parser.add_argument("mod", type=str, help="Modification chosen.")
args = parser.parse_args()

# File paths
scenario = args.scenario
mod = args.mod
data_path = f"data/CFE_unc-ren_2050/"
log_path = f"logs/{scenario}_{mod}/"
network_path = f"networks/base/{scenario}.nc"
output_path = f"data/{scenario}_{mod}/"

# Ensure existence of folders
os.makedirs(log_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# Set up logging
log_file_path = os.path.join(log_path, "load_data_transformer.log")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", handlers=[
    logging.FileHandler(log_file_path, mode="w"),  # Overwrite log file each run
    logging.StreamHandler(sys.stdout)              # Print to console as well
])
logger = logging.getLogger()

# Process each csv file in the folder
elec_files = glob.glob(os.path.join(data_path, "*_elec.csv"))
h2_files = glob.glob(os.path.join(data_path, "*_h2.csv"))

scenario = "unconstrained renewables"
final_result_df = pd.DataFrame()

# Loop through electricity files
for i, elec_file in enumerate(elec_files):
    logger.info(f"Processing {elec_file}...")

    # Read the electricity file
    df = pd.read_csv(elec_file, delimiter='\t', encoding='utf-16')

    # Convert the values by removing commas and changing to float
    df[scenario] = df[scenario].replace({',': ''}, regex=True).astype(float)

    # Summing values according to the rules
    transport = df[df['Sector'] == 'transport'][scenario].sum()
    industry = df[df['Sector'] == 'industry'][scenario].sum()
    sanitary = df[df['Subsector (group)'].isin(['res water heating', 'ser water heating'])][scenario].sum()
    space_heating = df[df['Subsector (group)'].isin(['res space heating', 'ser space heating'])][scenario].sum()
    tertiary_residential = df[
        ((df['Sector'] == 'residential') & (~df['Subsector (group)'].isin(['res water heating', 'res space heating']))) |
        ((df['Sector'] == 'tertiary') & (~df['Subsector (group)'].isin(['ser water heating', 'ser space heating'])))
    ][scenario].sum()

    country = os.path.basename(elec_file).replace('_elec.csv', '')
    country_label = f"{country}0"

    # Create new DataFrame with the results
    result_df = pd.DataFrame({
        'Category': ['transport', 'industry', 'sanitaryHeating', 'spaceHeating', 'residentialTertiary'],
        country_label: [transport, industry, sanitary, space_heating, tertiary_residential]
    })

    # Convert ktoe to MWh
    result_df[country_label] = result_df[country_label] * 11630

    # Handle hydrolysis share for corresponding h2 file
    h2_file = os.path.join(data_path, f"{country}_h2.csv")
    df_h2 = pd.read_csv(h2_file, delimiter='\t', encoding='utf-16')
    h2_share = df_h2.loc[df_h2.iloc[:, 1] == scenario, "Hydrogen TWh"].values[0]
    h2_share = float(h2_share) * 1e6  # Convert to MWh

    # Add hydrolysis row
    hydrolysis_df = pd.DataFrame({'Category': ['hydrolysis'], country_label: [h2_share]})
    result_df = pd.concat([result_df, hydrolysis_df], ignore_index=True)

    # Append the result of this country to the final result DataFrame
    if final_result_df.empty:
        final_result_df = result_df  # Initialize with the first country's results
    else:
        final_result_df = pd.merge(final_result_df, result_df, on='Category', how='outer')

    # Rename nodes to match PyPSA clusters
    rename_dict = {"SE0": "SE1", "GB0": "GB2", "FI0": "FI1", "IE0": "IE3"}
    final_result_df.rename(columns=rename_dict, inplace=True)

# Copy data for missing nodes
node_mappings = {
    "AL0": "GR0", "ME0": "HR0", "BA0": "HR0", "IT4": "IT0", "FR5": "IT0", "ES6": "ES0",
    "DK1": "DK0", "GB3": "IE3", "XK0": "HU0", "RS0": "HU0", "MK0": "HU0", "CH0": "AT0"
}
for new_node, source_node in node_mappings.items():
    if source_node in final_result_df.columns:
        final_result_df[new_node] = final_result_df[source_node]
    else:
        logger.warning(f"Source node {source_node} for {new_node} not found in dataframe!")

# Special case: NO0 = (DK1 + SE1) / 2
if "DK1" in final_result_df.columns and "SE1" in final_result_df.columns:
    final_result_df["NO1"] = (final_result_df["DK1"] + final_result_df["SE1"]) / 2
else:
    logger.warning("DK1 or SE1 missing, cannot compute NO0.")

# Calculate load shares
final_result_df.set_index('Category', inplace=True)
final_result_df.columns = [f"{col} 0" for col in final_result_df.columns]

# Load PyPSA network
n = pypsa.Network(network_path)
final_result_df = final_result_df.reindex(columns=n.loads.index)  # Reset index to match PyPSA network nodes
final_result_df.iloc[:, 1:] = final_result_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Compute load shares
load_shares = final_result_df.div(final_result_df.sum(axis=0), axis=1)

# Compute load increase factor for 2050
load_factors = final_result_df.sum(axis=0).div(n.loads_t.p_set.sum())  # Factor of load increase to 2050

# Adjust node mappings
node_mappings_adjusted = {f"{new_node} 0": f"{source_node} 0" for new_node, source_node in node_mappings.items()} #add " 0" to node map
for new_node, source_node in node_mappings_adjusted.items(): # Overwrite values for the new nodes using the source nodes
    if source_node in load_factors.index: # Ensure source node exists
        load_factors[new_node] = load_factors[source_node]  # Copy value
    else:
        logger.warning(f"Source node {source_node} for {new_node} not found in load_factors!")

load_factors = load_factors.rename_axis("node").reset_index(name="factor") # renaming columns

# Save results
output_filename_shares = os.path.join(output_path, "load_shares.csv")
load_shares.to_csv(output_filename_shares, index=True)
logger.info(f"Saved handled data to {output_filename_shares}")

output_filename_factors = os.path.join(output_path, "load_factors.csv")
load_factors.to_csv(output_filename_factors, index=False)
logger.info(f"Saved handled data to {output_filename_factors}")

logger.info("Script completed.")
