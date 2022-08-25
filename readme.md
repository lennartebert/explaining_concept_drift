# A Generic Approach for Explaining Drifts in Business Processes using Process Mining - Protoype and Experiments

This repository has the source code for the Master's thesis "A Generic Approach for Explaining Drifts in Business Processes using Process Mining" by Lennart Ebert (2022).

## How to get the code running?

### 1. Install Python Dependencies
Python dependencies are defined in the `requirements.txt`.

Here are two options for installing the dependencies:
1. Install Python 3.10 locally and use `pip install -r requirements.txt`
2. Use Anaconda for installing the dependencies in a virtual environment:
```sh
    conda create --name concept_drift python=3.10.4 pip
    conda activate concept_drift
    pip install -r requirements.txt
```

### 2. Download the Datasets
Follow the instructions in the `notebook 01_Import and Preprocess Data` to download and unpack the datasets.

### 3. Download Apromore ProDrift 2.5
Go to https://apromore.com/research-lab/ and download ProDrift 2.5.

Unpack the ZIP file to a location of your choice. If you unpack it to the folder `ProDrift2.5/ProDrift2.5.jar` in 
the project directory, no changes need to be made. If placed at other locations, the path needs to be specified 
whenever a `processdrift.explanation.drift_detection.ProDriftDD()` drift detector is created.

## Project Files
The drift explanation approach is implemented in the package `processdrift/explanation`. Docstrings are provided in each module.

The following notebooks are provided:
1. `01_Import and Preprocess Data.ipynb`: Help with import and preprocessing of data used.
2. `10_Synthetic Attribute Data Generator.ipynb`: Add generated attribute data to any XES event log.
3. `20_Visualize Trace Attributes.ipynb`: Visualize trace attribute data without any drift detection.
4. `30_Generic Approach for Explaining Drifts_Implementation for One Dataset.ipynb`: Apply the developed approach to an event log of your choice.
5. `40_Synthetic Dataset_Generation.ipynb`: Generate the synthetic dataset used in the thesis and two additional ones.
6. `41_Synthetic Dataset_Run Experiments.ipynb`: Run the drift explanation approach on the generated dataset.
7. `42_Synthetic Dataset_Analyze Results.ipynb`: Analyze the results from the experiments on the generated dataset.
8. `50_Real World Dataset_BPI Challenge 2015.ipynb`: Analysis of the BPIC 2015 dataset.
9. `60_Real World Dataset_BPI Challenge 2018.ipynb`: Analysis of the BPIC 2018 dataset.
