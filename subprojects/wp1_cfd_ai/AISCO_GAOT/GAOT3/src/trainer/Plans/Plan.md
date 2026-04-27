Plan for restructuring the entire Training Loop.

High-Level Vision:
- This document describes how the entire training-process shall be restructured.
- The new Code shall do Mathematically exactly the same as the old process, which will be verified with AISCO-Code\AISCO_code\subprojects\wp1_cfd_ai\AISCO_GAOT\GAOT3\tests\test_gaot2_gaot3_equivalence.py
- The new code shall be much better structured and easier to read for a human.

The following Files shall be (re-) implemented (more files can be added, if necessary)
In the GAOT3-folder: 
- main.py -> loads the config, does all changes necessary and calls the training_orchestrator
In the GAOT3.src.trainer-folder:
- training_orchestrator.py -> has the epoch-loop and calls most other classes
- optimiser.py -> defines optimiser behaviour
- WandB_logger.py -> collects and loggs Values and images to WandB
- loss_fn.py -> Calculates the loss-function
- validation_calculator.py -> Does all calculation and image-generation for validation
- Data_handler.py -> loads the data and processes it. e.g. random downsampling


The following Style-rules should be obayed during in all new Files:
- Every new File should only contain one class (except the Main-file) with public and private methods (starting with one _) All methods that are not called from outside this class should be per default private
- No new File shall be longer than 250 lines
- The main_config gets imported once in the main file as a dict and only be modified in the main file. Every new implemented class shall have the attribute .config: dict with the config loaded and modified in the main file. No values from the config shall be saved anywhere as a variable/attribute  etc. separatly, but they shall always be accessed directly from the .config
- Every Method shall have a short descriptive name that nicely summarizes what this method is doing.
- All variables and attributes shall have short descriptive Names
- Only write comments with """comment""" at the start of each document. This comment at the start of each Document should briefly describe, how this class shall be called and with what arguments.