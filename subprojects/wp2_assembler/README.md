# WP2 Assembler

Welcome to AISCO's Assembler!

Here the skeleton structure and the outer bounding box of the solar car is/will be generated.
These assemblies will be rules and regulations (RnR) compliant.
WP4, the Aeroshell Optimizer, uses our product to tell the NN where to build and where not to build an aeroshell (outer aerodynamic hull of the car).

## Overview

This workpackage contains different versions of the assembler tool that generates and validates solar car chassis assemblies according to competition regulations.

## Available Versions

- **version2/**: Current stable version - Random assembly generation with regulation validation

## Quick Start

Navigate to the version you want to use:
```bash
cd version2
```

See the version-specific README for setup and usage instructions.

## Project Structure
```
wp2assembler/
├── README.md           # This file
├── __init__.py
├──archive
|   ├── version1/       # Discontinued version
|   └── version2/       # Discontinued functions from version 2
├── version2/           # Current implementation
└── version3/           # Future versions...
```

## Requirements

- Python 3.10+
- See version-specific requirements

## Development

Each version is self-contained with its own dependencies and configuration. When starting a new version, copy the structure from the previous version and increment the version number.
