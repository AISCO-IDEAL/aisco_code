#kdiehl, #eschaerer
# Welcome to AISCO_code
You only find technical infos to get you coding here. All conceptual information can be found in `01 Knowledge Base/00 IT Tutorials / AISCO_code` on Obsidian.

Lets clone the AISCO_code Repo via SSH:
`git clone git@github.com:IDEALLab/AISCO_code.git`

## VS Code Plugins
- install `python`
- install `Git graph`
	- click on `Git Graph` at the bottom left to open the plugin
- install `ruff`
- install `mypy type checker`

## Important VS Code settings
- open settings `ctrl + ,` -> enter `git blame` -> enable
	- shows who edited what line of code => no tags necessary
- open settings `ctrl + ,` -> enter `format on save` -> enable
	- activates ruff, to make your code look nice and work well
	- reduces ruff errors later on

## hide unnecessary folders
- open VS Code
- `strg + , ` to open settings
- search for `hide folders`
- click on `add pattern`
- add folder name starting with `**/`

# Environment Installation Guide
Linux (not Euler)/Mac run `bash ./setup_conda.sh`
Windows run `python ./setup_conda.bat`

Euler environment guide:
1. `deactivate` any environment that might be active
2. run `chmod +x setup_conda.sh setup_venv.sh` to gain access rights for running the scripts
3. run `~/AISCO_code/setup_venv.sh`
4. for AISCO, please activate myenv in the `cd ~/AISCO_code` folder:
`source aisco_env/bin/activate`

these scripts will create a python 3.11.6 environment and most used dependencies. You can manually install workpage specific dependencies, by commanding `python3 -m pip install '.[<wpX_name>S]'"`. Replace `<wpX_name>` with the workpage name, you want to use.

## Environment Activation Guide (do this every time you open a new terminal)
for AISCO, please activate myenv in the `cd ~/AISCO_code/` folder:
`source aisco_env/bin/activate`
