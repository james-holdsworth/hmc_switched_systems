# data_to_mpc


Project requirements listed in requirements.txt

to create a new requirements file the following command can be run

pip freeze > requirements.txt

Note that this will add everything on your python path as == requirements

To install these requirements run

pip install -r requirements.txt

# Notes

The jaxlib version in the current requirements.txt will be for cuda 11.0, if you are using a different version of cuda then you will need to install a different version of jaxlib, and installing jaxlib correctly also cannot be completed through pip.

CUDA 11.2 does not have a ready-made version of Jaxlib,

Additionally, Python version should be less than 3.9.0 -- there is an issue in multiprocessing which can break PyStan.

# Installing CUDA, cuDNN and jax/jaxlib on Linux

[Nvidia CUDA toolkit documentation and install links](https://developer.nvidia.com/cuda-toolkit-archive)

[Nvidia cuDNN download (requires Nvidia account)](https://developer.nvidia.com/CUDnn)

[Nvidia cuDNN install guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)

[Jax install guide](https://github.com/google/jax#installation)

The instructions beyond here are my installation process to put CUDA 11.1.0, cuDNN v8, and Python+Jax on Ubuntu 20.04 LTS. Don't start with a distribution that comes with the Nvidia proprietary driver preinstalled (you will have to remove it) and I didn't install using a package manager because that just installed CUDA 11.2 (which didn't have a ready-made Jax version). I didn't install 11.0 Update 1 because there is a bug in the bundled driver that stops it from installing on the 20.04 LTS kernel, so I just moved to 11.1.0.

Following only these instructions will get you there but keep your phone handy to read this page.

## Install prerequisites 
Remove the Nvidia proprietary driver if it is already installed. Install Python < 3.9.0 by any means. Check that `gcc` is installed with `gcc --version`. If not, probably just install all of `build-essentials` by way of `sudo apt-get install build-essentials`. Also, to make sure the kernel headers are there, there is no harm in running `sudo apt-get install linux-headers-$(uname -r)`.

## Install CUDA toolkit via runfile
To install CUDA using the runfile method first download the runfile after selecting a version from [this page](https://developer.nvidia.com/cuda-toolkit-archive) (the runfile is the same for all Linux distros). This command gets 11.1.0:
```shell
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
```
I just did this in ~. It is a reasonably large download which can be deleted later. Running this file won't work until the graphics driver is unloaded.

Next, disable Nouveau which is the open source Nvidia driver. To check if Nouveau is loaded or not, do `lsmod | grep nouveau` - no output means it is not loaded (handy). To disable Nouveau on Ubuntu or Debian: 
```shell
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```
Put into the new file:
```shell
blacklist nouveau
options nouveau modeset=0
```
`Ctrl+X` to exit, `y` to write new buffer, `Enter` to save. Then run:
```shell
sudo update-initramfs -u
```
Nothing will happen until reboot. Reboot to check - no Nouveau means the screen scaling should be terrible, and there should be no output from `lsmod | grep nouveau`. 

Once Nouveau is disabled, reboot into console mode/text only so that no video driver is loaded at all. For distros using `systemd` including Ubuntu >= 15.04 this can be achieved by setting:
```shell
sudo systemctl set-default multi-user.target
```
If this worked, `systemctl get-default` should return `multi-user.target`. Reboot!

After reboot, the GUI will not be loaded. Log in and navigate to the directory where the runfile was downloaded, then run the runfile. For the 11.1.0 runfile:
```shell
sudo sh cuda_11.1.0_455.23.05_linux.run
```
It will appear to do nothing for a bit, then an EULA will appear. It's probably poorly scaled, type `accept` and hit enter to dismiss. The next menu should be some installation settings, enter the Toolkit settings and exert your will if you want to stop it from making desktop shortcuts. Install everything as the samples are useful for validation. 

It will print nothing for a while again and should succeed. If it doesn't say on the last line that it has failed then it has succeeded. 

After install these lines need to be added to `~/.bashrc` or equivalent. The version number is in the path, so change it if installing a different version. Run:
```shell
echo 'export PATH=/usr/local/cuda-11.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
```
To check if it worked, run `source ~/.bashrc`, then `nvcc --version` should work and print the installed version. At this point the install is finished, so set the GUI as default again:
```shell
sudo systemctl set-default graphical.target
```
And `sudo reboot` to return to the land of the sane.

There is some additional verification to do [here](https://docs.nvidia.com/cuda/archive/11.1.0/cuda-installation-guide-linux/index.html#verify-installation).

## Install cuDNN library via .tgz
Go to the [download site](https://developer.nvidia.com/CUDnn), log in, and choose cuDNN for 11.1, 11.1 and 11.2 and click on "cuDNN Library for Linux (x86_64)" to get a tarball. After that is downloaded unpack it:
```shell
tar -xzvf cudnn-x.x-linux-x64-v8.x.x.x.tgz
```
Replace the x's with the appropriate version numbers. In the unpacked directory, run: 
```shell
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## Install Jax
```shell
pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.59+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Change `cuda111` to match the current CUDA version. If you visit the [URL in the command](https://storage.googleapis.com/jax-releases/jax_releases.html) there is a list of available versions.