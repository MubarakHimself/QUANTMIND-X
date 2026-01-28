---
title: Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU
url: https://www.mql5.com/en/articles/13498
categories: Trading, Trading Systems, Expert Advisors, Machine Learning
relevance_score: 0
scraped_at: 2026-01-24T13:28:36.984413
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/13498&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082885081229430994)

MetaTrader 5 / Trading


Table of contents:

- [Introduction](https://www.mql5.com/en/articles/13498#para1)
- [Preparations](https://www.mql5.com/en/articles/13498#para2)
- [Hardware Configuration](https://www.mql5.com/en/articles/13498#para3)
- [Development Environment](https://www.mql5.com/en/articles/13498#para4)
- [Training the Model](https://www.mql5.com/en/articles/13498#para5)
- [Summary](https://www.mql5.com/en/articles/13498#para6)

### Introduction

In the previous article, we briefly discussed how to create datasets for large language models and demonstrated how to train a language model using only a CPU with a simple example. However, we did not test the model because, in real terms, it was only a pre-trained model. In this article, we continue our discussion on model training, this time using GPUs to accelerate the process. It’s important to note that, as a demonstration example, this model is still not powerful enough, so we won’t cover model testing in this article. Testing will be addressed in subsequent articles.

We previously covered the setup of CUDA acceleration environments in the second part of this series. Now, we’ll focus on using AMD graphics cards to accelerate training, which serves as a supplement to that previous article. Currently, setting up an NVIDIA graphics card environment is relatively straightforward, while configuring an environment for AMD cards may present various challenges. In this article, we’ll provide solutions to common issues, allowing you to smoothly accelerate training for your own financial language model using AMD graphics cards. If you’re using NVIDIA graphics cards, don’t worry—the training methods are the same. As long as you’ve already set up the CUDA environment, you can follow the training instructions provided in this article without needing to focus on the specific configuration steps for AMD cards.

Are you ready to go?

### Preparations

In a previous article in this series, we covered the setup of NVIDIA environments. However, we shouldn’t overlook the fact that some users are working with AMD hardware. Therefore, we’ll also discuss setting up the ROCm (Radeon Open Compute) acceleration computing environment. As of now, there’s no ROCm-supported PyTorch version available for Windows, and WSL (Windows Subsystem for Linux) doesn’t yet support ROCm either. So, if you want to use ROCm for training large models, Linux (specifically Ubuntu) is currently the viable option (On a positive note, next release ROCm may add WSL support in the future, but the current latest version is 6.1.2, and I don’t expect the first supported version to be optimal.). During my usage, I encountered several issues related to AMD hardware that I hadn’t faced with NVIDIA. I’ll share some solutions that I believe could be helpful for AMD users. Although it might be a bit cumbersome, remember that we’re saving money, right? Ah-ha…

So, if you want to accelerate training for large language models using AMD graphics cards, here are the necessary preparations:

**1\. Install Ubuntu**

You’ll need to install an Ubuntu system. Setting up a dual-boot system with Windows and Ubuntu on the same PC is feasible and recommended. We might need to access data from the Windows system, so having both OS options is useful. Detailed tutorials for setting up a dual-boot system are widely available online. Alternatively, you can deploy your AMD device as a remote Ubuntu host and use it as a remote development environment. You can then connect to the remote host via SSH from your Windows system. This approach is similar to using WSL on Windows, with the key difference being that AMD graphics cards don’t support WSL. I’ve asked ROCm developers about this, and they mentioned that WSL support is currently in development (possibly available in the next version). However, keep in mind that this solution might require you to have multiple computers. Personally, I’m using this method.

**2\. Check Hardware Configuration**

Be cautious about your hardware configuration. For instance, I encountered frequent driver crashes with my 7900XTX graphics card. Initially, I used an 850W power supply, which should have been sufficient for a single GPU. However, I didn’t consider that driver crashes might be related to power supply issues. After trying various solutions without success, a friend suggested upgrading to a 1250W power supply, and everything ran smoothly.

So, here’s a sincere piece of advice for all AMD hardware users: Make sure your power supply can handle your devices.

**3\. ROCM and HIP**

If you prefer a less complicated environment setup, you can use AMD’s official Docker image. Simply pull the ROCm official container image using Docker. Most of the necessary environment configurations are already set up in the Docker image, which is why I recommend this approach.

**4\. Hardware Settings**

During my device usage, I noticed that the optimization of the AMDGPU driver under Ubuntu isn’t as good. Take my 7900XTX as an example: Although it can output 350W of power, the default limit was set to 303W. Additionally, even under full load, the fan speed was less than 45%, and the core temperature remained at 100°C. For true large model training, which might run continuously for several hours or even days, maintaining such high temperatures is risky for the device.

Feeling tempted to switch hardware devices now? Trust me, there’s more to come…

### Hardware Configuration

**1\. Driver setting**

So far, the amdgpu driver in Ubuntu provides some external interfaces for GPU settings. The configuration files are located in "/sys/class/drm/card0/device/hwmon/hwmon2/". Note that "hwmon2" may vary; it could be "hwmon1" or another value. Please adjust it according to your specific device.I recommend adjusting the fan speed and power appropriately, but avoid modifying other settings.

Let’s illustrate how to set the fan speed. These file operations require administrator privileges, otherwise, you won’t be able to make changes. I recommend using the root user for modifications:

- First, enable pwm1:echo "1" > /sys/class/drm/card0/device/hwmon/hwmon2/pwm1\_enable
- Apply the setting: echo "c" > /sys/class/drm/card0/device/hwmon/hwmon2/pwm1\_enable

Set the fan speed to 128.The value can range from 0 to 255, where 255 corresponds to the maximum fan speed.

- Adjust it based on your needs: echo "128" > /sys/class/drm/card0/device/hwmon/hwmon2/pwm1
- Apply the setting: echo "c" > /sys/class/drm/card0/device/hwmon/hwmon2/pwm1

Adjust the power limit.For example, if your 7900XTX has a maximum power of 350W and defaults to 303W:

- Modify it to 330W: echo "330000000" > /sys/class/drm/card0/device/hwmon/hwmon2/power1\_cap
- Apply the setting: echo "c" > /sys/class/drm/card0/device/hwmon/hwmon2/power1\_cap

**Note:**

> 1\. To revert settings, use echo "r" > followed by the relevant path (e.g., echo "r" > /sys/class/drm/card0/device/hwmon/hwmon2/pwm1\_enable).
>
> 2\. Avoid setting fan speeds simultaneously through both "pwm1" and "fan\[1-\*\]\_target" interfaces, as the latter will override the former.

**2\. ROCM-SMI Settings**

Below are commonly used commands for rocm-smi:

usage: rocm-smi \[-h\] \[-V\] \[-d DEVICE \[DEVICE ...\]\] \[--alldevices\] \[--showhw\] \[-a\] \[-i\] \[-v\] \[-e \[EVENT ...\]\]

                \[--showdriverversion\] \[--showtempgraph\] \[--showfwinfo \[BLOCK ...\]\] \[--showmclkrange\] \[--showmemvendor\]

                \[--showsclkrange\] \[--showproductname\] \[--showserial\] \[--showuniqueid\] \[--showvoltagerange\] \[--showbus\]

                \[--showpagesinfo\] \[--showpendingpages\] \[--showretiredpages\] \[--showunreservablepages\] \[-f\] \[-P\] \[-t\]

                \[-u\] \[--showmemuse\] \[--showvoltage\] \[-b\] \[-c\] \[-g\] \[-l\] \[-M\] \[-m\] \[-o\] \[-p\] \[-S\] \[-s\]

                \[--showmeminfo TYPE \[TYPE ...\]\] \[--showpids \[VERBOSE\]\] \[--showpidgpus \[SHOWPIDGPUS ...\]\]

                \[--showreplaycount\] \[--showrasinfo \[SHOWRASINFO ...\]\] \[--showvc\] \[--showxgmierr\] \[--showtopo\]

                \[--showtopoaccess\] \[--showtopoweight\] \[--showtopohops\] \[--showtopotype\] \[--showtoponuma\]

                \[--showenergycounter\] \[--shownodesbw\] \[--showcomputepartition\] \[--showmemorypartition\] \[-r\]

                \[--resetfans\] \[--resetprofile\] \[--resetpoweroverdrive\] \[--resetxgmierr\] \[--resetperfdeterminism\]

                \[--resetcomputepartition\] \[--resetmemorypartition\] \[--setclock TYPE LEVEL\] \[--setsclk LEVEL \[LEVEL ...\]\]

                \[--setmclk LEVEL \[LEVEL ...\]\] \[--setpcie LEVEL \[LEVEL ...\]\] \[--setslevel SCLKLEVEL SCLK SVOLT\]

                \[--setmlevel MCLKLEVEL MCLK MVOLT\] \[--setvc POINT SCLK SVOLT\] \[--setsrange SCLKMIN SCLKMAX\]

                \[--setextremum min\|max sclk\|mclk CLK\] \[--setmrange MCLKMIN MCLKMAX\] \[--setfan LEVEL\]

                \[--setperflevel LEVEL\] \[--setoverdrive %\] \[--setmemoverdrive %\] \[--setpoweroverdrive WATTS\]

                \[--setprofile SETPROFILE\] \[--setperfdeterminism SCLK\]

                \[--setcomputepartition {CPX,SPX,DPX,TPX,QPX,cpx,spx,dpx,tpx,qpx}\]

                \[--setmemorypartition {NPS1,NPS2,NPS4,NPS8,nps1,nps2,nps4,nps8}\] \[--rasenable BLOCK ERRTYPE\]

                \[--rasdisable BLOCK ERRTYPE\] \[--rasinject BLOCK\] \[--gpureset\] \[--load FILE \| --save FILE\]

                \[--autorespond RESPONSE\] \[--loglevel LEVEL\] \[--json\] \[--csv\]

Set options:

  --setclock TYPE LEVEL                            Set Clock Frequency Level(s) for specified clock (requires manual Perf level)

  --setsclk LEVEL \[LEVEL ...\]                      Set GPU Clock Frequency Level(s) (requires manual  Perf level)

  --setmclk LEVEL \[LEVEL ...\]                     Set GPU Memory Clock Frequency Level(s) (requires manual Perf level)

  --setpcie LEVEL \[LEVEL ...\]                     Set PCIE Clock Frequency Level(s) (requires manual Perf level)

  --setslevel SCLKLEVEL SCLK SVOLT          Change GPU Clock frequency (MHz) and Voltage (mV) for a specific Level

  --setmlevel MCLKLEVEL MCLK MVOLT       Change GPU Memory clock frequency (MHz) and Voltage for (mV) a specific Level

  --setvc POINT SCLK SVOLT                      Change SCLK Voltage Curve (MHz mV) for a specific point

  --setsrange SCLKMIN SCLKMAX                 Set min and max SCLK speed

  --setextremum min\|max sclk\|mclk CLK    Set min/max of SCLK/MCLK speed

  --setmrange MCLKMIN MCLKMAX              Set min and max MCLK speed

  --setfan LEVEL                                        Set GPU Fan Speed (Level or %)

  --setperflevel LEVEL                                Set Performance Level

  --setoverdrive %                                      Set GPU OverDrive level (requires manual\|high Perf level)

  --setmemoverdrive %                               Set GPU Memory Overclock OverDrive level (requires manual\|high Perf level)

  --setpoweroverdrive WATTS                     Set the maximum GPU power using Power OverDrive in Watts

  --setprofile SETPROFILE                          Specify Power Profile level (#) or a quoted string of CUSTOM Profile attributes "# # # #..." (requires manual Perf level)

  --setperfdeterminism SCLK                     Set clock frequency limit to get minimal performance variation

  --setcomputepartition {CPX,SPX,DPX,TPX,QPX,cpx,spx,dpx,tpx,qpx}  Set compute partition

  --setmemorypartition {NPS1,NPS2,NPS4,NPS8,nps1,nps2,nps4,nps8}   Set memory partition

  --rasenable BLOCK ERRTYPE                   Enable RAS for specified block and error type

  --rasdisable BLOCK ERRTYPE                  Disable RAS for specified block and error type

  --rasinject BLOCK                                   Inject RAS poison for specified block (ONLY WORKS ON UNSECURE BOARDS)

Reset options:

  -r, --resetclocks                                               Reset clocks and OverDrive to default

  --resetfans                                                      Reset fans to automatic (driver) control

  --resetprofile                                                   Reset Power Profile back to default

  --resetpoweroverdrive                                     Set the maximum GPU power back to the device deafult state

  --resetxgmierr                                                 Reset XGMI error count

  --resetperfdeterminism                                   Disable performance determinism

  --resetcomputepartition                                  Resets to boot compute partition state

  --resetmemorypartition                                   Resets to boot memory partition state

Auto-response options:

  --autorespond RESPONSE                            Response to automatically provide for all prompts (NOT RECOMMENDED)

Output options:

  --loglevel LEVEL                                            How much output will be printed for what program is doing, one of debug/info/warning/error/critical

  --json                                                            Print output in JSON format

  --csv                                                             Print output in CSV format

**3\. AMD-SMI Settings**

Based on the current development direction of ROCm, it’s possible that amd-smi might replace rocm-smi. Below are commonly used commands for amd-smi:

usage: amd-smi set \[-h\] -g GPU \[GPU ...\] \[-f %\] \[-l LEVEL\] \[-P SETPROFILE\] \[-d SCLKMAX\]

                   \[-C PARTITION\] \[-M PARTITION\] \[-o WATTS\] \[-p POLICY\_ID\] \[-x POLICY\_ID\]

                   \[--json \| --csv\] \[--file FILE\] \[--loglevel LEVEL\]

A GPU must be specified to set a configuration.

A set argument must be provided; Multiple set arguments are accepted

Set Arguments:

  -h, --help                                         show this help message and exit

  -g, --gpu GPU \[GPU ...\]                     Select a GPU ID, BDF, or UUID from the possible choices

  -f, --fan %                                        Set GPU fan speed (0-255 or 0-100%)

  -l, --perf-level LEVEL                       Set performance level

  -P, --profile SETPROFILE                   Set power profile level (#) or a quoted string of custom profile attributes

  -d, --perf-determinism SCLKMAX      Set GPU clock frequency limit and performance level to determinism to get minimal performance variation

  -C, --compute-partition PARTITION   Set one of the following the compute partition modes: CPX, SPX, DPX, TPX, QPX

  -M, --memory-partition PARTITION    Set one of the following the memory partition modes: NPS1, NPS2, NPS4, NPS8

  -o, --power-cap WATTS                     Set power capacity limit

  -p, --dpm-policy POLICY\_ID              Set the GPU DPM policy using policy id

  -x, --xgmi-plpd POLICY\_ID                Set the GPU XGMI per-link power down policy using policy id

Command Modifiers:

  --json                             Displays output in JSON format (human readable by default).

  --csv                              Displays output in CSV format (human readable by default).

  --file FILE                      Saves output into a file on the provided path (stdout by default).

  --loglevel LEVEL             Set the logging level from the possible choices: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Note:**

> It’s worth mentioning that some settings fail when using amd-smi, but succeed with rocm-smi. For example: sudo amd-smi set -g 0 -f 90% failed: "ValueError: Unable to set fan speed 229 on GPU ID: 0 BDF:0000:03:00.0".In such cases, switch to using rocm-smi: sudo rocm-smi --setfan 90%. It works!

**4\. Software Supporting Hardware Settings**

In Ubuntu, there are also UI-based management tools available. While we won’t delve into them in detail in this article, interested users can explore the following options:

LACT: [https://github.com/ilya-zlobintsev/LACT](https://www.mql5.com/go?link=https://github.com/ilya-zlobintsev/LACT "https://github.com/ilya-zlobintsev/LACT")

radeon-profile: [https://github.com/marazmista/radeon-profile](https://www.mql5.com/go?link=https://github.com/marazmista/radeon-profile "https://github.com/marazmista/radeon-profile")

### Development Environment

Once your hardware is set up, you’ll need to configure the following environment on your system for GPU acceleration. Let’s continue using AMD devices as an example:

- Driver Support for Accelerated Computing: We need install the amdgpu DKMS driver and the ROCm HIP SDK.
- For Python libraries supporting accelerated computing: PyTorch compiled with ROCm.If you need quantization for models, consider using bitsandbytes (compiled with ROCm). For accelerated computing, you might need flash-attention and triton (both compiled with ROCm). During inference testing, you may use vllm. In our example, installing PyTorch should suffice.

**1\. Driver and Dependency Installation**

Check if your hardware and system are supported by the latest ROCm version. If not, explore earlier ROCm versions.If no version supports your device, congratulations—you’re finally free from AMD’s torment! Consider exploring other brands.If your device is ROCm-supported, you have two options: Deploy the environment using Docker ro install directly on the local host. The steps are similar whether you’re installing locally or within a Docker container. However, I strongly recommend using Docker. If you make mistakes during operations, you can simply delete the container and run a new one from the already pulled Ubuntu image — no need to reinstall the system! Therefore, this article focuses on deploying the development environment using Docker. When installing Docker, use Docker Engine (not Docker Desktop). Refer to the installation guide: [https://docs.docker.com/engine/install](https://www.mql5.com/go?link=https://docs.docker.com/engine/install "https://docs.docker.com/engine/install").

**2\. Pull the Base Image**

Pull the Ubuntu 22.04 base image : docker pull ubuntu:22.04

**Note:**

> If you prefer a non-customized deployment, directly pull the ROCm/PyTorch image:docker pull rocm/pytorch:latest.

Then run the container:

```
docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
		--device=/dev/kfd --device=/dev/dri --group-add video \
		--ipc=host --network=host --shm-size 8G rocm/pytorch:latest
```

Inside the Docker container, most development environments are already configured. You can either copy llm.c into the container or clone it from the official repository using Git. Then proceed directly to the model training section. However, if you encounter uncommon issues, you won’t know how to troubleshoot. Therefore, my recommendation is to deploy step by step, so you’ll always know where to find solutions.

**3\. Create a Container**

Run the following command to create a container (you can choose a different name if desired):

```
docker run -it --name llmc --cap-add=SYS_PTRACE --ipc=host --network=host \
		--security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video ubuntu:22.04
```

**Note:**

> "--name llmc" is optional, it names the created container "llmc". If you omit this parameter, the container will have a random name.   I recommend using --network=host to simplify network access from the container to the host. Otherwise, the default network mode is bridged, which complicates container-to-host network access (e.g., accessing the host’s network proxy).   "--ipc=host" allows the container to share memory with the host, improving efficiency. However, it reduces security. Use it based on your needs.

**4\. Driver Installation and ROCm Setup**

Install the necessary software dependencies. If you’re installing on the host, prefix the commands with sudo:

```
apt install wget

apt install gpg
```

Configure the amdgpu APT repository:

```
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/6.1.1/ubuntu jammy main" \    | tee /etc/apt/sources.list.d/amdgpu.list

apt update

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.1.1 jammy main" \    | tee --append /etc/apt/sources.list.d/rocm.list

echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \    | tee /etc/apt/preferences.d/rocm-pin-600

apt update
```

Install the amdgpu driver:

```
apt install amdgpu-dkms
```

If you’re installing on the local host, reboot the system:  reboot.

Install ROCm (I’ve chosen rocm-hip-sdk):

```
apt install rocm-hip-sdk
```

Of course, you have other options as well:

- rocm：All ROCm core packages, tools, and libraries.
- rocm-language-runtime：The ROCm runtime.
- rocm-developer-tools：Debug and profile HIP applications.
- rocm-hip-runtime：Run HIP applications writen for the AMD platform.
- rocm-hip-runtime-devel：Develop applications on HIP or port from CUDA.
- rocm-opencl-runtime：Run OpenCL-based applications on the AMD platform.
- rocm-opencl-sdk：Develop OpenCL-based applications for the AMD platform.
- rocm-hip-libraries：HIP libraries optimized for the AMD platform.
- rocm-hip-sdk：Develop or port HIP applications and libraries for the AMD platform.
- rocm-ml-libraries：Key machine learning libraries. Includes MIOpen.
- rocm-ml-sdk：Develop and run machine learning applications for AMD.
- rocm-openmp-runtime：Run OpenMP-based applications on the AMD platform.
- rocm-openmp-sdk：Develop OpenMP-based applications for the AMD software.

After installation, add the following environment variables:

```
tee --append /etc/ld.so.conf.d/rocm.conf <<EOF

/opt/rocm/lib/opt/rocm/lib64EOF

ldconfig
```

Check DKMS status:

```
dkms status
```

Add binary paths to the PATH environment variable:

```
export PATH=$PATH:/opt/rocm-6.1.1/bin
```

Verify the installation: /opt/rocm-6.1.1/bin/rocminfo  /opt/rocm-6.1.1/bin/clinfo

**5\. Installing PyTorch**

If Python is not installed, use apt to install it, or set up a Conda (Miniconda) environment:   apt install libjpeg-dev python3-dev python3-pip

Next, check the official PyTorch website to see if your ROCm version is supported. As of now, the official release does not support ROCm 6.1.1, so we’ll install the preview version:

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
```

![torch](https://c.mql5.com/2/81/124049.png)

If your ROCm version is lower than the officially supported version, adjust the installation command accordingly. For example, if the official release command is:  pip3 install torch --index-url https://download.pytorch.org/whl/rocm6.0,   and your ROCm version is "5.7.\*", you can install as follows:

```
pip3 install torch --index-url https://download.pytorch.org/whl/rocm5.7
```

If you encounter a "NameError: name ‘amdsmi’ is not defined" issue, it’s because newer Torch versions require Python support for amdsmi. Unfortunately, directly using pip won’t work. You’ll need to build and install it locally.Navigate to /opt/rocm/share/amd\_smi and install amd-smi-lib:

```
apt install amd-smi-lib
```

If you’re installing to the host’s default Python version:

```
python3 -m pip install --user .
```

If you encounter a permission error like:

```
error: Cannot update time stamp of directory 'amdsmi.egg-info'

error: could not create 'amdsmi.egg-info': Permission denied
```

Run the following command with root privileges:

```
sudo python3 -m pip install --user .
```

If Python3 is not found, use:

```
sudo -H python3 -m pip install --user .
```

If you’re using a Conda environment (e.g., named "train\_llm"), install as follows:

```
sudo /home/deeper/miniconda/envs/train_llm/bin/python -m pip install .
```

Replace the path with your specific virtual environment’s Python path.

Finally, retry:

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.1/
```

### Training the Model

Similar to the previous article, we’ll continue using the open-source project "llm.c". Personally, I find this project excellent—the code logic is clear, and the comments are detailed, making it easy to read. Interested readers can explore the entire source code of this project to gain a clear understanding of the entire process of training large language models. I won’t reiterate the code for each training step here, the comments in the source code are already quite explicit.

To download the "llm.c" repository, it should be noted here that you should not directly git the official repository but use the " [https://github.com/anthonix/llm.c](https://www.mql5.com/go?link=https://github.com/anthonix/llm.c "https://github.com/anthonix/llm.c")" project that supports AMD devices. If you have any problems with training, you can switch to the source code version I use, the current version I am using is "37dfcf6":

```
cd llm.c
git checkout 37dfcf6
```

If you need to switch versions, use the following command line to switch after git clone down the project:

**1\. Install Project Dependencies**

First, install the project dependencies:

```
pip install -r requirements.txt
```

Note that you should install Torch first before installing the dependencies from requirements.txt to avoid version conflicts.

Next, compile the project:

```
make train_gpt2amd AMDGPU_TARGETS=gfx1100
```

To determine your specific ISA (Instruction Set Architecture), use rocminfo and filter the output with grep:

```
rocminfo | grep Name:
```

**2\. Prepare for training**

This step is similar to training on a CPU, so I won’t go into detail here. If you need a comprehensive explanation, refer to the previous article. Below is the training command.

Get the data:

```
python data_enc
```

Prepare for training:

```
python train_gpt2.py --input_bin data/val_data.bin
```

Replace "data/val\_data.bin" with the output path from running data\_enc.

Execute the following command:

```
make train_gpt2amd AMDGPU_TARGETS=gfx1100
```

Again, replace gfx1100 with your specific ISA obtained from rocminfo.

**Note:**

> If you encounter a "fatal error: ‘cmath’ file not found", it’s likely due to a mismatched g++ version. Some Ubuntu versions come with g++11, but not g++12. Install g++12 and set it as the default version:
>
> ```
> apt install g++-12
>
> update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 --slave /usr/bin/g++ g++ /usr/bin/g++-12
> ```

Ignore the "4 warnings generated when compiling for host." , execute the following command:

```
./train_gpt2amd
```

When I When I run the last command, I received a program crash with "\[CUDA ERROR\] at file build/hip/llmc/layernorm.cuh:400". If you regret choosing AMD hardware at this point, believe me, this is just the beginning—many more challenges await.The journey with AMD hardware has just begun!

If you manage to execute successfully, great! If not, don’t be discouraged, we can still use Python scripts for training the model.

Using python scripts to train the model, you can run the following command:

```
python train_gpt2.py \
    --input_bin "./data/train_data.bin" \
    --input_val_bin "./data/val_data.bin" \
    --val_loss_every 10 \
    --sample_every 0 \
    --output_dir pylog124M \
    --write_tensors 0 \
    --model d12 \
    --batch_size 16 \
    --sequence_length 128 \
    --total_batch_size 4096 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 100 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --warmup_iters 0 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0
```

Here are a few parameters that need to be mentioned:

- –input\_bin "./data/train\_data.bin" This is our training dataset;
- –input\_val\_bin "./data/val\_data.bin" This is our validation dataset;
- –val\_loss\_every 10 Print the loss every certain number of steps;
- –batch\_size 16 This value must be a power of 2 otherwise an error will occur, since we are just testing and the dataset is not large, there is no need for it to be too large ;
- –sequence\_length 128 This value must be a power of 2 otherwise an error will occur, since the sequence length of our dataset is 64, so I set it to 128 here;
- –total\_batch\_size 4096 We are just testing, so this value does not need to be too large, similarly, this value must be a power of 2 otherwise an error will occur;
- –num\_iterations 100 This is the number of training steps, similarly, because it is a test, I simply set it to 100;
- –flash 1 This value is whether to turn on flash attention, "1" is to turn on, "0" is to turn off.


The running results:

![re](https://c.mql5.com/2/81/resulte.png)

As you can see, we successfully trained a financial language model exclusive to us using GPU. Now let’s take out the time we used to train with CPU to compare:

![old](https://c.mql5.com/2/81/old.png)

Originally, our CPU training process took an average of about 8000ms per step, while GPU training took less than 80ms per step.

Although some parameters may have changed a bit, it is not difficult to see from the time of each training step that using GPU will greatly speed up the training process, almost accelerating by more than 100 times!

Since our model is just a simple example trained, and has not been optimized, there is no need to create an EA strategy for testing, as you can imagine the result must be very bad.

But don’t worry, the following articles will discuss how to fine-tune a large language model using our own data, and then create an EA and test it.

**Note:**

If this place reports an error RuntimeError: HIP error: the operation cannot be performed in the present state. Compile with TORCH\_USE\_HIP\_DSA to enable device-side assertions.

Try adding environment variables:

```
export HSA_OVERRIDE_GFX_VERSION=11.0.0

export HIP_VISIBLE_DEVICES=0

export ROCM_PATH=/opt/rocm

export AMDGPU_TARGETS=gfx1100
```

Or add the following code at the beginning of train\_gpt2.py:

```
from os import putenv

putenv(“HSA_OVERRIDE_GFX_VERSION”, “11.0.0”)
```

Where 11.0.0 is the HSA of 7900xtx, you need to check the HSA according to your own device and change it to the appropriate value. The gfx1100 is the ISA of 7900xtx, you need to change the appropriate value according to your own hardware.

### Summary

In this article, we discussed how to train a large language model exclusive to us on a graphics card using our own designed dataset. Of course, we also discussed the development environment configuration in AMD graphics card accelerated computing, supplemented the part that was not mentioned in the second part of this series, and also completed the environment configuration of the two mainstream graphics cards (amd and nvidia) in the current market for accelerated computing.

But because this demonstration example has not been optimized, so the same did not evaluate the model, nor did it formulate an EA strategy based on the model and test it in the MetaTrader client, because at present this demonstration model is just in the pre-training stage, so these operations are unnecessary and have not reached a practical level. In the following articles, we will discuss the fine-tuning of large language models and formulate corresponding EA strategies and test them in the MetaTrader client.

See you in our next article!

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13498.zip "Download all attachments in the single ZIP archive")

[data\_enc.py](https://www.mql5.com/en/articles/download/13498/data_enc.py "Download data_enc.py")(2.87 KB)

[llm\_data.csv](https://www.mql5.com/en/articles/download/13498/llm_data.csv "Download llm_data.csv")(1139.04 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(IV) — Test Trading Strategy](https://www.mql5.com/en/articles/13506)
- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (III) – Adapter-Tuning](https://www.mql5.com/en/articles/13500)
- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs (II)-LoRA-Tuning](https://www.mql5.com/en/articles/13499)
- [Integrate Your Own LLM into EA (Part 5): Develop and Test Trading Strategy with LLMs(I)-Fine-tuning](https://www.mql5.com/en/articles/13497)
- [Integrate Your Own LLM into EA (Part 3): Training Your Own LLM with CPU](https://www.mql5.com/en/articles/13920)
- [Data label for time series mining (Part 6)：Apply and Test in EA Using ONNX](https://www.mql5.com/en/articles/13919)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469148)**
(1)


![yangqibiao](https://c.mql5.com/avatar/avatar_na2.png)

**[yangqibiao](https://www.mql5.com/en/users/yangqibiao)**
\|
14 Apr 2025 at 11:36

Very good article and very valuable to read and learn from, I love your articles on AI and machine learning type, thanks for providing it, very informative!


![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)](https://c.mql5.com/2/82/Building_A_Candlestick_Trend_Constraint_Model_Part_5__NEXT_LOGO_2.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)](https://www.mql5.com/en/articles/14968)

Today, we are discussing a working Telegram integration for MetaTrader 5 Indicator notifications using the power of MQL5, in partnership with Python and the Telegram Bot API. We will explain everything in detail so that no one misses any point. By the end of this project, you will have gained valuable insights to apply in your projects.

![MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://c.mql5.com/2/82/MQL5_Wizard_Techniques_you_should_know_Part_24__LOGO.png)[MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://www.mql5.com/en/articles/15135)

Moving Averages are a very common indicator that are used and understood by most Traders. We explore possible use cases that may not be so common within MQL5 Wizard assembled Expert Advisors.

![Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://c.mql5.com/2/71/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://www.mql5.com/en/articles/14246)

Having started developing a multi-currency EA, we have already achieved some results and managed to carry out several code improvement iterations. However, our EA was unable to work with pending orders and resume operation after the terminal restart. Let's add these features.

![The base class of population algorithms as the backbone of efficient optimization](https://c.mql5.com/2/71/The_basic_class_of_population_algorithms____LOGO_2_.png)[The base class of population algorithms as the backbone of efficient optimization](https://www.mql5.com/en/articles/14331)

The article represents a unique research attempt to combine a variety of population algorithms into a single class to simplify the application of optimization methods. This approach not only opens up opportunities for the development of new algorithms, including hybrid variants, but also creates a universal basic test stand. This stand becomes a key tool for choosing the optimal algorithm depending on a specific task.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/13498&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082885081229430994)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).