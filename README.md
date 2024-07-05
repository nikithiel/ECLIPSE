# ECLIPSE
Modeling Extracorporeal Life Support using Parameter Selection and Estimation

## Overview
ECLIPSE is a tool designed for modeling Extracorporeal Life Support (ECLS) with a focus on parameter selection and estimation. The main script to execute is [main.py](main.py).

## How to Use
### 1) Setup
Ensure you have all the required dependencies installed. Refer to [requirements.txt](requirements.txt) for the list of necessary packages. Code runs on **Python 3.12.2**.

### 2) Configuration
Modify the [runtime.txt](runtime.txt) file to specify the type of run. The options include:
  
**p** for normal production run\
**s** for Sensitivity Analysis\
**f** for Parameter Fitting

### 3) Execution
Run the main.py script: `python main.py`

The script initializes pytrees and data for the model and launches the corresponding run based on your configuration in [runtime.txt](runtime.txt).

### 4) Patient Data
Store and manage patient-specific parameters in the `Patients` directory. You can use files like [patient_1_data.txt](Patients/patient_1_data.txt) to input data for individual patients. Before you can perform patient specific simulations. You need to run the parameter fitting that creates [result_optimization_patient_1.txt](Optimization/Results/result_optimization_patient_1.txt) in the `Optimization` folder.

### 5) Plotting
Change [ListOfPlots.txt](ListOfPlots.txt) to define desired plots.

Checkout [model_plot.py](model_plot.py) and function [plot_results()](https://github.com/nikithiel/ECLIPSE/blob/abdbb8ea3022b728042b89fffccf02397cc97db0/model_plot.py#L102) for currently available plotting capabilities.
  
### Directory Structure
- **main.py:** Main script to run the model.
- **Analyses:** Scripts for convergence study of Global sensitivity analysis and other analyses.
- **Optimization:** Contains inputs for parameter optimization and corresponding results.
- **Parameters:** Initial parameters for models and other configurations.
- **Patients:** Directory for storing patient-specific data.
- **Plots:** Directory for storing generated plots.
- **Sensitivity:** Inputs for Sensitivity analysis and corresponding results.
<br>
If you have any questions, do not hesitate to open an issue or send an email to: thiel@ame.rwth-aachen.de or neidlin@ame.rwth-aachen.de
