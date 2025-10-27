<<<<<<< HEAD:README.md
#kdiehl
# Welcome to AISCO-code
=======
##################################################################################################

# This README.md file is under Reconstruction. 

# most important changes:
    src/ folder repplaces SDF_Prototype/ folder
    tests/ folder replaces SDF_Tests/ folder
| **Euler compatibility**                       | **Purpose: We 

# src/
    **Main source code for 3D SDF generation and optimization**

    DummySphere
        
    
# tests/
    **Integration tests for the SDF generation code**


#################################################
PRE:
An Assembly CAD-file
POST:
A 3D SDF representation of an Aeroshell
PPOST:
A converted Mesh representation of the Aeroshell
##################################################################################################
CAD Assembly → [Assembler] → SDF Network → [Training] → Trained SDF → [Mesh Conversion] → 3D Mesh
##################################################################################################

#aalthaus

This SDF_Prototype folder serves as a week long experiment where @aalthaus @eschaerer @chuber test the concept of a 3D SDF generator


Some flaws that still need to be adressed: 

| Transition from flat to src/ folder structure | Purpose: Get used to real deep learning workflow and keep structure cleaner. For detailed advantages: https://medium.com/@adityaghadge99/python-project-structure-why-the-src-layout-beats-flat-folders-and-how-to-use-my-free-template-808844d16f35 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Euler compatibility**                       | **Purpose: We need to get our code running on Euler ASAP to keep the transition headaches minimal --> helps us later**                                                                                                                               |
| **Discuss solution regarding environments**   | **Avoid compatibility hell later on, discuss on Monday 20.10. with Kolja how to do**



# 3D SDF Vehicle Shape Optimization

This project implements a neural signed distance field (SDF) approach for optimizing 3D vehicle shapes, specifically targeting aerodynamic efficiency and solar collection area for solar race cars.

## Overview

The system uses a neural network to represent a 3D shape as a signed distance field, then optimizes the shape to:
- **Minimize frontal area** (reduce aerodynamic drag)
- **Maximize top surface area** (increase solar panel area)
- **Satisfy size constraints** (fit within specified bounding boxes)
- **Maintain valid SDF properties** (Eikonal constraint)
- **Ensure smooth surfaces** (manufacturability)

## Files Description

### Core Modules
- `sdf_network.py` - 3D neural SDF network with skip connections and mixed activations
- `loss_functions.py` - Comprehensive loss functions for multi-objective optimization
- `visualization.py` - Advanced visualization tools for 3D shapes and training progress
- `main_training.py` - Complete training pipeline and experiment management

### Utility Files
- `demo.py` - Quick demo script for testing (reduced iterations)
- `requirements.txt` - Python package dependencies
- `README.md` - This documentation file

## Installation

1. Install Python 3.7+ and pip
2. Install required packages:
```bash
pip install -r requirements.txt
>>>>>>> WP4_SDF_Development:subprojects/WP4-Aeroshell/000_README.md
```
git clone git@github.com:IDEALLab/AISCO-code.git

```
# Quick installation guide
Linux/Windows symply run `bash ./setup_conda.sh`
Windows symply run `python ./setup_conda.bat`
*a venv version for Euler will be discussed with ffelten*

the script will create a python 3.11.6 environment and most used dependencies. You can manually install workpage specific dependencies

# Code Structure
every subproject/workpage uses its own folder. They are all children of the AISCO folder
every folder contains at least
- README file (only the very necessary technical instructions)
- data folder
- code files
(- archive folder)

There is also a test folder. Store your test cases here


# Naming Conventions
folders use capital letters and underscores, e.g.: WP1_CFD_AI
files only use small letters and underscroes, e.g.: dummy_2.py

classes use capitals for every first letter and do not use underscores, e.g.: ExampleClass
functions use captials for every first letter besides the very first and do not use underscores, e.g.: exampleFunction

# Branch Convention
main

only rigorously tested code is added to the main branch. If a new merge is done, always do a pull_request, following the PULL_REQUEST_TEMPLATE

only one person works per definition. Put you tag (#eth_kuerzel) in every function you edit. Ask the original editor before editig their functions!

`def test(x: int) -> str:
    #kdiehl
    """Beispiel-Funktion: gibt eine stringbasierte Bestätigung zurück.

    Args:
        x: eine Ganzzahl (wird hier nur zu Demonstrationszwecken angezeigt)

    Returns:
        Eine Bestätigungszeichenkette mit dem übergebenen Wert.
    """`

# Coding
- always define the input and output of a functions, e.g. `def test(x: int) -> str:`
- call functions via `from AISCO.subprojects.WP5.dummy_2 import test`
    - this should also work for folders above or in parallel to yours
- store generated data in the data folder within your subfolder/WP
    - will be synced via gDrive or NAS later on
