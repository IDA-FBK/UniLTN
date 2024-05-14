# UniLTN

**UniLTN - Code for the paper:**

Enhancing Logical Tensor Networks: Integrating Uninorm-Based Fuzzy Operators for Complex Reasoning

## Project Structure

The project structure is organized as follows:

- **data**: Directory for storing datasets and loading them.
- **experiments**: Directory for experiment configurations and results.
  - **configurations**: Subdirectory for experiment configuration files categorized by example (e.g., binary classification and parent_ancestor).
  - **results**: Subdirectory for storing experiment results categorized by example.
  - **operators.py**: Python module for defining operators used in experiments.
  - **plots.py**: Python module for generating plots to visualize experiment results.
  - **get_plots_from_csv.py**: Python module for generating plots from csv files of multilabel and mnist_single_digit_addition examples.
- **ltn**: Directory for logic tensor networks (LTN) related code.
- **models**: Directory containing modules for model construction and utilities.
  - **axioms.py**: Python module defining axioms for logic tensor networks.
  - **commons.py**: Python module containing train functions used across examples.
  - **evaluate.py**: Python module containing evaluation functions for a subset of examples.
  - **models.py**: Python module for defining models.
  - **steps.py**: Python module for defining training and test steps.
  - **utils.py**: Utility functions.
- **main.py**: Main script to execute the project.
- **utils.py**: General utility functions.
- **uniltn.yml**: YAML file specifying the conda environment required to run the project.

## Running the Code

### 1. Set up the Project Environment

- First, ensure you have Conda installed. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Then, create the project environment using the provided fuzzyltn_env.yml file:
  ```bash
  conda env create -f uniltn_env.yml
  ```

- And activate it:

  ```bash
  conda activate uniltn
  ```

### 2. Set up Configuration

- Ensure you have a configuration file located in the `experiments/configurations/<example_name>/` directory.

- This file should contain experiment settings such as seed, epochs and optimizer.

### 3. Command-line Arguments

- The main script accepts command-line arguments to specify dataset, path to configuration file, and directory to store results.

- Use the following command-line arguments:
  - `-example`: Specify the example to run (default is "binaryc").
  - `-path_to_conf`: Provide path to configuration file (default is `./experiments/configurations/<example_name>/ltn/conf-00.json`).
  - `-path_to_results`: Define directory where results will be saved (default is `./experiments/results/<example_name>/ltn/`).

### 4. Run the Main Script

- Run the main script:
  ```bash
  python ./main.py -path_to_conf ./experiments/configurations/<example_name>/ltn/conf-00.json -path_to_results ./experiments/results/<example_name>/ltn/ -example <example_name>

  ```

### 5. Get plots for Multilabel and Single Digit Addition

- Move inside **experiments** folder
- Run the `get_plots_from_csv.py` script:
  ```bash
  python ./get_plots_from_csv.py -example <example_name> -path_to_results <path_to_results> -metric <metric>  -exp_id <exp_id>

  ```
  - `-example_name`: name of the example to plot (i.e, multilabel, mnist_single_digit_addition, default is `multilabel`).
  - `-path_to_results`: path containing the csv files (default is `multilabel`).
  - `-exp_id`: identifier for exp (default is `seed-0_epochs500_lr0.001`).
  - `-metric`: Metric to plot (i.e., Loss, Satisfiability, Accuracy, default is `Satisfiability`).

## Authors
- Paulo Vitor De Campos Souza: pdecampossouza@fbk.eu
- Gianluca Apriceno: apriceno@fbk.eu
- Mauro Dragoni: dragoni@fbk.eu
## License
For open source projects, say how it is licensed.


