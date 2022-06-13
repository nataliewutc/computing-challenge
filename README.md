# Computing challenge - Machine learning 

## Aim of project 
This project uses real-world data to build, validate and analyse the results of a classifier to predict the crystalline structure of perovskites, depending on various features such as the atomic radii and electronegativity of their components. 

## Background on perovskites 
Perovskite are a class of materials with an ideal composition given by the formula ABX3, where A and B are two differently sized cations, and X is an anion. Perovskites have interesting properties that make them a strong candidate for the development of various applications related to solar energy conversion. Moreover, due to their compositional flexibility (i.e., potential to deviate from their ideal composition introducing, e.g., vacant sites), distortion of the cation configuration, and mixed valence state electronic structure, perovskites have a rather large design space that can be exploited for their optimisation.

A total of 73 elements have been identified in ABO3 perovskites, leading to numerous oxides of the perovskite type. However, which exact crystalline structure one obtains changes depending on features such as electronegativity, ionic radius, valence, and bond lengths of A-O and B-O pairs and so on. Notably, different crystalline structures lead to different performance and, depending on the specific application for which they are deployed, one would like to be able to tune the crystalline structure at will.

## The dataset 
The dataset in the file Crystal_structure.csv provided for the Computing Challenge consists of 4,165 ABO3 perovskite-type oxides. Each observation is described by 13 feature columns and 1 class column which identifies the crystalline structure to be either a cubic, tetragonal, orthorhombic, and rhombohedral structure.

## Remarks 
The main.py file consists of the full code whilst the 3 jupyter notebooks divides the code into 3 stages, with explanations and comments provided in between each cell for clarity so the reasoning could be followed. 