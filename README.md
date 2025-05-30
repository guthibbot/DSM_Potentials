# Energy Systems Modelling: Extreme Events in Highly Renewable Networks
This repository contains the materials and scripts for a project investigating demand side management (DSM) in a fully renewable energy systems using the [PyPSA](https://pypsa.org/) framework.
The study analyses DSM utilization and network composition under varying DSM parameter configurations.

## Project Overview
With the growing adoption of renewable energy systems, implementing DSM into them holds great potential. But for that a better understanding of DSM potentials and how to model it is also important. This project implements DSM into the modelling framework PyPSA using a storage analogy. Six different scenarios are investigated with varying amounts of DSM added to the system as well as a sensitivity analysis of the DSM parameters.

## Repository Structure
The repository is organized as follows:
├── data/ # data from external datasets and folders for storing processing data
├── figures/ # Contains all figures generated during the project
├── networks/ # Networks created and used in the project
│ ├── base/ # base Pypsa network
│ └── mod/ # modified networks based on scenarios
│ └── solved/ # solved modified networks
├── report/ # Folder containing final project report in PDF format
├── scripts/ # Python scripts used to set up and simulate various scenarios
│ ├── load_data_transformer.py
│ ├── load_profile_creator.py
│ └── network_creator.py #
│ └── network_solver.py
├── notebooks/ # folder containing jupyter files used for plotting
├── env.yml # Python environment file
└── README.md # Project documentation (this file)


## Key Scenarios
The study investigates six key scenarios to evaluate the system's response to various configurations of DSM:
1. **Base Scenario**: Baseline network created with PyPSA-EUR.
2. **Real Scenario**: Implemented DSM based on scenario projections.
3. **Full Scenario**: All DSM parameters set to extreme high.
4. **Max_fraction**: DSM parameters like in "real" except max fraction extreme high.
5. **Tau**: DSM parametes like in "real" except tau extreme high.
6. **Flex_share**: DSM parameters like "real" except flexible share extreme high.

## Usage Instructions
1. Clone the repository:
      git clone https://github.com/guthibbot/DSM_potentials.git
      cd DSM_potentials
2. Install the python environment:
      mamba env create -f env.yml
4. Choose the scripts to run:
      python scripts/EXAMPLE.py

## Results
The project outputs include:
- **Figures**: illustrating system dynamics and extreme events (located in the figures/ folder).
- **Networks**: Solved networks with modified configurations.
- **Report**: Comprehensive findings and analyses in the project report.
- **References**: Key references include: Kleinhans et al. (2014), "Towards a systematic characterization of the potential of demand side management"

## Acknowledgments
This project was conducted as part of the Master Thesis at Aarhus University.
Special thanks are extended to Professor Alexander Kies at Aarhus University for his supervision of this project.

