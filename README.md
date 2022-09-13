# MLE School 2022 - Material for Hands-On III

# Introduction to artificial neural networks: classifying handwritten numbers using Python and Tensorflow/Keras.

The goal of this workshop is to provide an insight into deep learning on the use case of detecting handwritten numbers. After an introduction to the basics of neural networks, a workflow for machine learning tasks is solved in groups. Therefore, Jupyter notebooks are provided to guide the course through ANN development with Tensorflow/Keras. Finally, the trained networks will be applied for the detection and recognition of handwritten text on images.


## Getting started

To participate, a Laptop with an installation of [*Anaconda*](https://www.anaconda.com/) is the most useful choice. So please install it according to the instructions on their website.

Alternatively, a participation in Google Colab is possible. 

[**COLAB Engl.**](https://colab.research.google.com/github/BenBol/MLE-School/blob/main/Workshop_english.ipynb)

[**COLAB Germ.**](https://colab.research.google.com/github/BenBol/MLE-School/blob/main/Workshop_Germ.ipynb)

For local participation, download the material by opening a **Terminal** (**Anaconda Promt** on Windows) and copy the material in a folder. 

```bash
git clone https://github.com/BenBol/MLE-School.git
```

Navigate in the folder
```bash
cd MLE-School
```
and create a new environment 
```bash
conda env create -f environment.yaml
```
or for an **Apple Silicon Mac**
```bash
conda env create -f environment_M1_Mac.yaml
```
Use the following command to activate the workshop environment.

```bash
conda activate MLE-Hands-On-III
```

## Removing the data

After the course, the environment can be deleted with
```bash
conda remove --name MLE-Hands-on-III  --all
conda clean --all
``` 