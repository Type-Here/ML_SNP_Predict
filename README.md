# GeneInsight: p53 Mutation Classifier
## AI-based Prediction of TP53 Mutation Pathogenicity

## **Project Overview**

This university project, created for the *Fundamentals of AI exam*, aims to classify mutations in the **TP53** and **HRAS** genes based on their pathogenicity.

Using neural networks, it analyzes the impact of single nucleotide polymorphisms (SNPs) and predicts whether mutations are likely to be:
- Benign
- Pathogenic
- Uncertain

## Setup Instructions

1. Prerequisites

Ensure you have Python 3.10+ and Mamba or Conda installed.

Ensure you have Qt installed on your system if you plan to run the application outside of the Conda environment. 
You can install it using your system's package manager or by downloading it from the official Qt website.  
  
For example, on Ubuntu, you can install it with:
```sh
sudo apt-get install qt5-default
```

2. Setting Up the Environment

After downloading the project, open its root directory in a terminal.
The environment was tested using Mamba but should work with Conda as well.

**Option 1: Minimal Environment**

This option sets up the basic environment for running the project:

```sh
conda env create -f environment.yml
conda activate fia_env
python tensor.py  # Installs TensorFlow and Keras -- 
```

Using `python tensor.py` should install TensorFlow GPU version if a correct environment is detected. CPU one otherwise.

**Option 2: Full Environment**

This option installs all dependencies, including TensorFlow (CPU) and Keras:

```sh
conda env create -f full_env.yml
conda activate fia_env
```

3. Running the Application

You can run the project in two ways:

- Directly with Python:
```sh
python UI/main.py
```

- Using the Shell Script if your system runs bash:
```sh
./run.sh
```

The GUI will launch, allowing you to interact with the model and make predictions on TP53 or HRAS mutations.

## Replicating the Experiment

To replicate the results:

- Load a Model:
    If not present, a prompt will ask you to train a new one.
    The dataset will be automatically downloaded when running the project for the first time.

- Predict a Mutation:
    Choose a valid DNA position to insert a SNP (1 - to gene length), a Reference Nucleotide will be loaded.
    Choose a mutation (A, T, C, G) and click on `Predict`.

- The Model will load the output.

## The Training Process

When not already present, a new model can be automatically trained. The operations performed include:

- Preprocess the data:
    The pipeline cleans, extracts features, encodes, scales, and balances the dataset.

- Train the model:
    The training process uses the neural network defined with embedding layers and dense layers (v2 model).

- Evaluate the model:
    Evaluation metrics include accuracy, recall, precision, and F1-score.
    Cross-validation is performed using a 10-times 10-fold scheme.
    Stats and plots are saved.

- Retrain the model:
    Retrained with the entire dataset to perform future predictions.

- Save the model:
    Saves it as a `.h5` and `.keras` format file.

## Dependencies

Core Dependencies:

- Python 3.10+
- TensorFlow 2.18.0
- Keras
- PyQt5
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Matplotlib

For a full list, see `full_env.yml`.

## License

This project is licensed under the GPLv3 License.
