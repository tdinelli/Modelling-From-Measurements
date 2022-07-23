# Modelling From Measurements

This repository contains the homework of the course "Modelling from measurements" organized by Professor [Alberto Berizzi](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ricerche/RicercaPerDocentiPublic.do?evn_didattica=evento&k_doc=14853&polij_device_category=DESKTOP&__pj0=0&__pj1=161107224bf5e306682c8834e636702f) and helded at Politecnico di Milano by professor [J. Nathan Kutz](https://faculty.washington.edu/kutz/) in May 2022.

## Repository

The repository is organized as follow:

- [Data](Data/)\
Contains all the useful data to run the Notebooks for the assignement.

- [Homework](Homework/)\
Contains the text of the homework and the final report produced.

- [Materials](Materials/)\
Some useful scripts in python and MATLAB to produce data and explore different things for the homework.

- [Notebooks](Notebooks/)\
Jupyter Notebooks of the homework. This folder contains also:
    - [Lotka Volterra](Notebooks/Utility/lotkavolterra.py) Python function to compute the Lotka Volterra sytem of equations.
    - [DMD Functions](Notebooks/Utility/FunctionsDMD.py) Python implementation of some useful functions for computing DMD and reconstruction. 
    - [BOP-DMD](Notebooks/Utility/PythonBOPDMD/) Python implementation of Bagging Optimized DMD taken from https://github.com/kunert/py-optDMD and slightly modified to match Python3 requirements.
    - [Lorenz](Notebooks/Utility/Lorenz.py) Python function describing Lorenz system of equations.
    - [Kuramoto-Sivashinsky](Notebooks/Utility/KuraSiva.py) Python function for computing Kuramoto-Sivashinsky equation.
    - [Reaction Diffusion](Notebooks/Utility/Reaction_Diffusion.py) Python Function for the computation of Lambda-Omega reaction diffusion system.

## External packages

- [Numpy](https://numpy.org)
- [Pandas](https://pandas.pydata.org)
- [Matplotlib](https://matplotlib.org)
- [Scipy](https://scipy.org)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Keras](https://keras.io)
- [PySINDY](https://pysindy.readthedocs.io/en/latest/)
- [Tabulate](https://pypi.org/project/tabulate/)
- [Science Plots](https://github.com/garrettj403/SciencePlots)

If you would like to reproduce the results keep in mind that in order to use pySINDY package you should use python3.7.
