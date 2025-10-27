#kdiehl
# Welcome to AISCO-code
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
folders and files use small letters and underscores, e.g.: wp1_cfd_ai


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
- call functions via `from AISCO.subprojects.wp5.dummy_2 import test`
    - this should also work for folders above or in parallel to yours
- store generated data in the data folder within your subfolder/WP
    - will be synced via gDrive or NAS later on
