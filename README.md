# hmc_switched_systems
This repo contains research code into the application of HMC to switched systems. The root contains folders which correspond to different systems models. Within each there are python scripts that run PyStan for sampling as well as variations on the theme.

## Virtual environment
The scripts have been developed in Python 3.8.6 with `venv` to create a virtual environment. One can create a virtual environment (in a subfolder called `.venv`) by running the following:
```
python venv -m ./.venv
``` 
Use `python` or `python3` depending on your own set up.
## Requirements
A list of requirements is given in requirements.txt. The `jax` and `jaxlib` packages need to be installed together, separately to the other package requirements. The requirements can be installed via:
```
sh setup_packages.sh
```
Or by running each line of the aforementioned `.sh`.
