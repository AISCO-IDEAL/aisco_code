#kdiehl, #eschaerer
# Welcome to AISCO-code
You only find technical infos to get you coding here. All conceptual information can be found in `01 Knowledge Base/00 IT Tutorials / AISCO-code` on Obsidian.

Lets clone the AISCO-code Repo via SSH:
`git clone git@github.com:IDEALLab/AISCO-code.git`

### VS Code Plugins
- install `python`
- install `Git graph`
	- click on `Git Graph` at the bottom left to open the plugin
- install `ruff`
- install `mypy type checker`

### hide unnecessary folders
- open VS Code
- `strg + , ` to open settings
- search for `hide folders`
- click on `add pattern`
- add folder name starting with `**/`

# Environment Installation Guide
Linux/Mac simply run `bash ./setup_conda.sh`
Windows symply run `python ./setup_conda.bat`
Euler environment guide:
1. `deactivate` any environment that might be active
2.
3. run `./setup_venv.sh`
4. enjoy!

these scripts will create a python 3.11.6 environment and most used dependencies. You can manually install workpage specific dependencies, by commanding `pip install [<wpX_name>]`. Replace `<wpX_name>` with the workpage name, you want to use.
