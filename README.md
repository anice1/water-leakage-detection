## Abstract

### Leak size and Leak location optimization using the steady state approach
Leak localization involves the process of determining the possible location from which the leak is occuring. This process was carried on the python jupyter notebook development environment. The steps involved can be summarised in the following:
* Import all necessary libraries (pre written code for easy references of prebuilt methods), these libraries include: wntr, PyGAD, numpy, pandas, matplotlib etc.
* Automate temporary folder creation for storing optimization outputs for future analysis using python.
* Setup custom python classes for easier operations. The classes created were: Config class and the WaterLeakModel class
* Select and add leak to a node for optimization
* Define the custom fitness function
* Optimise and evaluate results
