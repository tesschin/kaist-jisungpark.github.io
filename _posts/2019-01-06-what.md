---
published: true
title: Nvidia Driver and Cuda9.0 Installation
classes: wide
categories:
  - Tools
tags:
  - gpu
---

Tested hardware and OS configuration:
* OS: Ubuntu 16.04 LTS
* NVIDIA Graphic Card: Quadro M1000M
* Cuda Version: 9.0
* Graphic Card Driver Version: 410.xx
* Disable `secure  boot` in BIOS setting

The recommended way to install the Nvidia driver and Cuda is using `.run` files since the `run` files provide flexibility for configuration. You can get rid of the login loop mess-up caused by the bundled `opengl` libs in Nvidia drivers with care.

Let's start with the `dkms(Dynamic Kernel Module Support)` package. This is a super useful package when you install drivers. Sometimes your laptop may have the latest hardware that the ubuntu does not support. You have to install the corresponding drivers by building from source codes. The issue is that if the system updates the kernel automatically, you basically lose your manually-installed drivers for the hardware. You have to re-install it. But with `dkms` package, you don't need to worry about this problem anymore. The rebuild of the modules is handled automatically when a kernel is upgraded.

## Preliminary
1.	Install `dkms` via `apt-get`
```bash
sudo apt-get install dkms
```
2.  Install the kernel header with   
```bash
sudo apt-get install linux-headers-$(uname -r)
```
3.	Download the nvidia driver according to your graphic card model from [nvidia-website](https://www.nvidia.com/Download/index.aspx?lang=en-us) and the corresponding cuda-toolkit from [here](https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=runfilelocal). They look like something like `NVIDIA-Linux-x86_64-xxx.xx.run` and `cuda_9.0.176_384.81_linux.run`. Make them executable by
```bash
chmod +x NVIDIA-Linux-x86_64-410.93.run
chmod +x cuda_9.0.176_384.81_linux.run
```
You may download the four patches for the cuda9.0 as well.
```bash
chmod +x cuda_9.0.176.1_linux.run
chmod +x cuda_9.0.176.2_linux.run
chmod +x cuda_9.0.176.3_linux.run
chmod +x cuda_9.0.176.4_linux.run
```
4.	Blacklist the nouveau. The nouveau coming with ubuntu systems will affect the installation of nvidia drivers. Blacklist it by
```bash
# create blacklist file for nouveau
sudo touch /etc/modprobe.d/blacklist-nouveau.conf
# write the content
sudo bash -c "echo 'blacklist nouveau
options nouveau modeset=0' > /etc/modprobe.d/blacklist-nouveau.conf"
# update the blacklist
sudo update-initramfs -u
```
If you get nothing output in the terminal by `lsmod | grep nouveau`, you are good to go.
5. Purge the nvidia driver installed via PPA by
```bash
sudo apt-get purge nvidia*
```

## Installation
1. Kill the x-server by
```bash
sudo service lightdm stop
```   
2. Login to the system from tty by `alt+ctrl+F1`, login with your user name and password.
3. Navigate to the directory of downloaded `run` files, install graphic driver in headless mode:
  ```bash
  sudo ./NVIDIA-Linux-x86_64-410.93.run -no-opengl-files
  ```  
4.	Install cuda9.0
```bash
sudo ./cuda_9.0.176_384.81_linux.run --no-opengl-libs
```
During the installation,
  * **accept** the EULA conditions
  * say **NO** to installing the nvidia drivers
  * say **YES** to installing cuda toolkit
  * say **YES** to installing cuda samples
  * say **YES** to creating a symbolic link for cuda
  * say **NO** to rebuilding any Xserver configuration with nvidia

5. Set the env variables for cuda in `~/.bashrc`
```bash
sudo bash -c "echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc"
sudo bash -c "echo '/usr/local/cuda/lib64/' > /etc/ld.so.conf.d/cuda.conf"
source ~/.bashrc
sudo ldconfig
```
**NOTE**: Be careful with `>>` and `>`, the symbol `>` will overwrite the file. You don't want to overwrite your `.bashrc` file.  
Reboot or get back to the window mode by:
```bash
sudo service lightdm start
```
You should get similar outputs as below by `nvidia-smi` command:  
  ```bash
  Mon Jan  7 21:00:08 2019       
  +-----------------------------------------------------------------------------+
  | NVIDIA-SMI 410.93       Driver Version: 410.93       CUDA Version: 10.0     |
  |-------------------------------+----------------------+----------------------+
  | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
  | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
  |===============================+======================+======================|
  |   0  Quadro M1000M       Off  | 00000000:01:00.0 Off |                  N/A |
  | N/A   51C    P0    N/A /  N/A |      0MiB /  2004MiB |      0%      Default |
  +-------------------------------+----------------------+----------------------+
  +-----------------------------------------------------------------------------+
  | Processes:                                                       GPU Memory |
  |  GPU       PID   Type   Process name                             Usage      |
  |=============================================================================|
  |  No running processes found                                                 |
  +-----------------------------------------------------------------------------+
```
6. Apply the patches if you need them (**optional**):
```bash
sudo ./cuda_9.0.176.1_linux.run  # accept and Enter
sudo ./cuda_9.0.176.2_linux.run  # accept and Enter
sudo ./cuda_9.0.176.3_linux.run  # accept and Enter
sudo ./cuda_9.0.176.4_linux.run  # accept and Enter  
```

## Cuda Testing with Samples
  * install compiler
  ```bash
  sudo apt-get install -y gcc build-essential
  ```
  * compiling
  ```bash
  cd ~/NVIDIA_CUDA-9.0_Samples
  make -j8
  ```
  * testing
  ```bash
  bin/x86_64/linux/release/deviceQuery # test 1
  bin/x86_64/linux/release/bandwidthTest # test 2
  ```
  * you should get something like this:
  ```bash
  bin/x86_64/linux/release/deviceQuery Starting...
  CUDA Device Query (Runtime API) version (CUDART static linking)
  Detected 1 CUDA Capable device(s)
  Device 0: "Quadro M1000M"
  CUDA Driver Version / Runtime Version          10.0 / 9.0
  CUDA Capability Major/Minor version number:    5.0
  Total amount of global memory:                 2004 MBytes (2101870592 bytes)
  ( 4) Multiprocessors, (128) CUDA Cores/MP:     512 CUDA Cores
  GPU Max Clock rate:                            1072 MHz (1.07 GHz)
  Memory Clock rate:                             2505 Mhz
  Memory Bus Width:                              128-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
     deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 10.0, CUDA Runtime Version = 9.0, NumDevs = 1
     Result = PASS
  ```

## Troubleshooting
If you get stuck in the login loop by accidently installing nvidia driver with the bundled opengl library, uninstall the nvidia driver and cuda library by
```bash
sudo /usr/bin/nvidia-uninstall
sudo /usr/local/cuda-9.0/bin/uninstall_cuda_9.0.pl
```
and then reboot. You are back to normal.


## Install cuDNN  
Pretty easy!
* Join the [NVIDIA Developer Program](https://developer.nvidia.com/accelerated-computing-developer) and get the permission to download the cuDNN.
* Download the cuDNN from [here](https://developer.nvidia.com/rdp/cudnn-download), select the deb version that matches cuda9.0:   
  * cuDNN Runtime Library for Ubuntu16.04 (Deb):  `libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb`  
  * cuDNN Developer Library for Ubuntu16.04 (Deb):  `libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb`  
  * cuDNN Code Samples and User Guide for Ubuntu16.04 (Deb): `libcudnn7-doc_7.4.2.24-1+cuda9.0_amd64.deb`
* Install cuDNN by   
```bash
sudo dpkg -i libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.4.2.24-1+cuda9.0_amd64.deb
```
* Testing
```
cd /usr/src/cudnn_samples_v7/mnistCUDNN
sudo make -j8
./mnistCUDNN
```
You should get something like this if installed successfully   
```bash
cudnnGetVersion() : 7402 , CUDNN_VERSION from cudnn.h : 7402 (7.4.2)
Host compiler version : GCC 5.5.0
There are 1 CUDA capable devices on your machine :
device 0 : sms  4  Capabilities 5.0, SmClock 1071.5 Mhz, MemSize (Mb) 2004, MemClock 2505.0 Mhz, Ecc=0, boardGroupID=0
Using device 0
Testing single precision
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.029920 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 1.930336 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 2.276768 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 2.370656 time requiring 203008 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 2: 17.794369 time requiring 57600 memory
Resulting weights from Softmax:
0.0000000 0.9999399 0.0000000 0.0000000 0.0000561 0.0000000 0.0000012 0.0000017 0.0000010 0.0000000
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 0.9999288 0.0000000 0.0000711 0.0000000 0.0000000 0.0000000 0.0000000
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 0.9999820 0.0000154 0.0000000 0.0000012 0.0000006
Result of classification: 1 3 5
Test passed!
Testing half precision (math in single precision)
Loading image data/one_28x28.pgm
Performing forward propagation ...
Testing cudnnGetConvolutionForwardAlgorithm ...
Fastest algorithm is Algo 1
Testing cudnnFindConvolutionForwardAlgorithm ...
^^^^ CUDNN_STATUS_SUCCESS for Algo 0: 0.025600 time requiring 0 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 1: 0.033376 time requiring 3464 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 4: 0.170336 time requiring 207360 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 7: 0.340768 time requiring 2057744 memory
^^^^ CUDNN_STATUS_SUCCESS for Algo 5: 0.551552 time requiring 203008 memory
Resulting weights from Softmax:
0.0000001 1.0000000 0.0000001 0.0000000 0.0000563 0.0000001 0.0000012 0.0000017 0.0000010 0.0000001
Loading image data/three_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000000 0.0000000 1.0000000 0.0000000 0.0000714 0.0000000 0.0000000 0.0000000 0.0000000
Loading image data/five_28x28.pgm
Performing forward propagation ...
Resulting weights from Softmax:
0.0000000 0.0000008 0.0000000 0.0000002 0.0000000 1.0000000 0.0000154 0.0000000 0.0000012 0.0000006
Result of classification: 1 3 5
Test passed!
```



## Install tensorflow-gpu
Take the python3 from the system as an example:
```bash
sudo pip3 install --upgrade tensorflow-gpu
```
### Test
```bash
python3
>>> import tensorflow as tf
>>> sess = \
tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

The output should be something like this:
```bash
2019-01-07 22:54:32.219774: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-01-07 22:54:32.697484: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-01-07 22:54:32.697992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
name: Quadro M1000M major: 5 minor: 0 memoryClockRate(GHz): 1.0715
pciBusID: 0000:01:00.0
totalMemory: 1.96GiB freeMemory: 1.92GiB
2019-01-07 22:54:32.698012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-07 22:58:26.675975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-07 22:58:26.676021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
2019-01-07 22:58:26.676036: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
2019-01-07 22:58:26.676508: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 1683 MB memory) -> physical GPU (device: 0, name: Quadro M1000M, pci bus id: 0000:01:00.0, compute capability: 5.0)
Device mapping:
/job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
/job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Quadro M1000M, pci bus id: 0000:01:00.0, compute capability: 5.0
2019-01-07 22:58:26.678076: I tensorflow/core/common_runtime/direct_session.cc:307] Device mapping:
/job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device
/job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Quadro M1000M, pci bus id: 0000:01:00.0, compute capability: 5.0
```
