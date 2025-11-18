Welcome to the Assembler from AISCO!

This is the second version of the Assembler.
Goals:
Create new type of asssembly - like the previous assembler with primitive components, just there is no chassis.
It will generate the assemblies more randomly to have a high variance, these are then to be checked by a validator for RnR-compliance.

# Possible File structure from Claude:

version2/
├── __init__.py
├── README.md
│
├── components/              # Your main assembly components
│   ├── __init__.py
│   ├── nbz.py               # No-build-zone functions
│   ├── wheel.py             # Chassis generation
│   ├── seat.py              # Skeleton structure
│   ├── bounding_box.py      # Bounding box generation
│   └── ...                  # more to come
│
├── utils/                   # Helper utilities
│   ├── __init__.py
│   ├── geometry.py          # Geometric calculations
│   ├── validators.py        # Rule & regulation validators
│   ├── converters.py        # Format conversions
│   └── visualization.py     # Display/show helpers
│
├── models/                  # Data structures
│   ├── __init__.py
│   └── assembly.py          # Assembly data model
│
├── config/                  # Configuration
│   └── regulations.yaml     # RnR specifications
│
├── tests/                   # Unit tests
│   ├── test_components/
│   └── test_utils/
│
└── examples/                # Examples & notebooks
    └── manualTesting.ipynb


# Key Principles
## Components (components/)
Purpose: Domain-specific business logic, main functional units
Contains: Classes and functions that represent your module's core features
Example: assembler.py, chassis.py, nbz.py (no-build-zone)
Naming: Descriptive, domain-specific names
## Utils (utils/)
Purpose: Generic, reusable helper functions
Contains: Functions that could be used across different components
Example: File readers, data validators, converters, formatters
Naming: Functionality-based (e.g., io.py, validation.py, converters.py)

# Best Practices
1. __init__.py files: Use them to expose public APIs
2. Keep utils generic: If a function is specific to one component, put it in that component file
3. Avoid circular imports: Utils shouldn't import from components; components can import from utils
4. Single responsibility: Each file should have one clear purpose
5. Follow your coding guidelines: Use type hints and documentation as specified in your Code Guidelines.md
