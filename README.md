# hmc_switched_systems
This repo contains research code into the application of HMC to switched systems. The root contains folders which correspond to different systems models. Within each there are python scripts that run PyStan for sampling as well as variations on the theme.

### MSD Model
This model is a mass-spring-damper system expressed as a two-state linear state space model. The control inputs are forces applied to the mass, and the states are the mass position and velocity. There is a measurement of the position and a measurement of the acceleration of the mass.

### Furata Pendulum
This model is a furata or rotary pendulum model that is based on the [Quanser QUBE Servo 2](https://www.quanser.com/products/qube-servo-2/). The model is a four-state non-linear state space model, and the system evolution is simulated as RK4 integration of the state gradient with time. The states are the base and pendulum angles and corresponding angular velocities. The control input is the base arm motor voltage. There are encoder measurements of both arm angles.

## Virtual environment
The scripts have been developed in Python 3.8.6 on Ubuntu with `venv` to create a virtual environment. One can create a virtual environment (in a subfolder called `.venv`) by running the following:
```
python venv -m ./.venv
``` 
Use `python` or `python3` depending on your own set up. Once the virtual environment is created, it can be activated by:
```
source ./.venv/bin/activate
```
Deactivate by:
```
deactivate
```
While the virtual environment is activated, packages are installed locally in the `.venv` folder. Python will only see the packages in the virtual environment.
## Requirements
A list of requirements is given in requirements.txt. The requirements can be installed via:
```
pip install -r requirements.txt
```
