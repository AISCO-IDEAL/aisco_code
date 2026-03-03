# WP7 GUI

## Status
We asssume that we get a decent PC by April.
1. We focus on getting the code to run on Matthew's PC
2. We implement the most basic GUI possible.

## Run code on Matthew's PC
### 1. Get Python 3.11.6 (down from 3.12.3) in a virtual env
	# 1. Install Build Dependencies
	sudo apt update
	sudo apt install -y build-essential libssl-dev zlib1g-dev \
	libbz2-dev libreadline-dev libsqlite3-dev curl \
	libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
	# 2. Install pyenv
	curl https://pyenv.run | bash
	# 3. Update your Path
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init -)"' >> ~/.bashrc
	source ~/.bashrc
	# 4. Install and Set 3.11.6
	pyenv install 3.11.6
	pyenv local 3.11.6
	# Verify
	python --version
	# Should output: Python 3.11.6
	# Step 1: Initialize the Virtual Env
	cd ~/AISCO_code
	python -m venv .aisco-pc
	source .aisco-pc/bin/activate

	# you can list all folders like this:
	ls
	# include files explicitly:
	ls -la
	# you can remove stuff like this:
	rm <name> 
	rm -rf <name> # if folder includes a lot of file

	# Install the dependencies
	# Step 2: Install the Monorepo
	pip install --upgrade pip
	pip install "wandb[media]>=0.25.0"
	pip install -e ".[all]"

	# Now, in your VS Code Terminal (on Matthew's PC), install git:
	https://git-scm.com/install/windows