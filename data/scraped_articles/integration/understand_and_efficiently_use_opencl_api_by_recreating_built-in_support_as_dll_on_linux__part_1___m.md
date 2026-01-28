---
title: Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation
url: https://www.mql5.com/en/articles/12108
categories: Integration
relevance_score: 7
scraped_at: 2026-01-22T17:51:34.158971
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/12108&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049433290357713780)

MetaTrader 5 / Examples


### Contents

- Introduction
- Problem (as Motivation)
- Workaround
- Game plan
- Understanding OpenCL (Just Enough)
- OpenCL Terminology (Recap)
- OpenCL Class Diagram
- Phase I - Validation Using GPU Device with OpenCL by Developing a Simple OpenCL Testing Program

### Contents (To be continued in next following up parts...)

- Phase II - Development of a Simple Proof-of-Concept OpenCL Support as DLL
- Phase III - Development of full OpenCL Support as DLL
- Port _Includes/OpenCL/OpenCL.mqh_
- Porting OpenCL Examples (MQL5) to Use Our Newly Developed OpenCL Solution

  - BitonicSort
  - FFT
  - MatrixMult
  - Wavelet

- Performance Benchmark of Both Built-in OpenCL, and Our OpenCL Solution (Just for Fun)
- Conclusion

My previous article on [Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux](https://www.mql5.com/en/articles/12042 "Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux") can build the foundation of understanding and knowing on how to develop tools / solutions for MetaTrader 5 on Linux. Such grounds will be used onwards for my future articles as well.

### Introduction

OpenCL (Open Computing Language) is framework that allows users to write programs to execute across CPU (Central Processing Unit), GPU (Graphics Processing Unit), or dedicated accelerator device with benefit that it can speed up heavy computation required as per problem domain. Especially using GPU with OpenCL as GPU is designed to handle large amounts of data in parallel with high memory bandwidth and dedicated instruction set which is optimized in mathematics calculations. It has lots of processing core called compute units each of which can independently carry on the computation. In contrast, CPU is designed to carry out more general task, not specialized in the same sense as GPU. It has less processing cores. In short, CPU is suitable for different task than GPU. GPU is much faster for heavy duty computation especially in regards to graphics domain.

MetaTrader 5 supports OpenCL version 1.2. It has [several built-in functions](https://www.mql5.com/en/docs/opencl "MetaTrader 5 Official Document on Working with OpenCL") that users can take benefit in using of out-of-box.

### Problem (as Motivation)

The problem is that MetaTrader 5's built-in OpenCL support although able to detect GPU device, but it is unable to select such GPU device to use with OpenCL whenever we call _CLContextCreate()_ with either _CL\_USE\_GPU\_ONLY_ or _CL\_USE\_GPU\_DOUBLE\_ONLY_. It always return error code of 5114.

![MT5 able to detect GPU device](https://c.mql5.com/2/0/opencl-mt5-detected-GPU-linux.png)

MetaTrader 5's OpenCL Built-in Support is able to Detect GPU Device

![MT5 OpenCL built-in support device selection error 5114](https://c.mql5.com/2/0/opencl-device-selection-error-5114.png)

MetaTrader 5's OpenCL Built-in Support Always has error=5114 (Device Selection Error) as Tested on My Linux System

If we consult [Runtime Errors](https://www.mql5.com/en/docs/constants/errorswarnings/errorcodes "Go to Runtime Errors - MQL5 Official Document"), such error code corresponds to the following explanation.

|     |     |     |
| --- | --- | --- |
| ERR\_OPENCL\_SELECTDEVICE | 5114 | OpenCL device selection error |

For reference, the following is my machine spec

- Ubuntu 20.04.3 LTS with kernel 5.16.0
- CPU: AMD Ryzen 5 3600 6-Core Processor (2 threads per core)
- GPU: AMD Radeon RX 570 Series (support OpenCL 1.2)
- Graphics driver support is through open source driver available on Ubuntu (both OpenGL, and Vulkan) not via proprietary one namely AMDGPU-Pro

As I have validated that MetaTrader 5 is able to list out all devices included GPU as seen in Journal tab, so there is no issue with graphics driver installed on my Linux machine. It's not a problem with either it is open source or proprietary graphics driver. Thus we could say at this point that it is highly likely a bug in device selection code whenever those two flags as mentioned have been used with _CLContextCreate()_. We shall take this problem as our motivation to conduct a validation for our assumption of the bug, then proceed next to develop a full-fledge solution later in the series. The major benefit of our effort is to allow us to understand more about OpenCL concepts, its terminology, and most importantly how we can use its API in efficient manner especially when we use it with MQL5 to develop related tools on MetaTrader 5 platform.

Please feel free to look at [Discussion of article "How to install and Use OpenCL for Calculations"](https://www.mql5.com/en/forum/12777) which has seen some users reportedly still face with inability to detect GPU when using OpenCL. The problem has dated back since 2013 up until now included myself. I believe this affected only a small group of users both on Windows and Linux.

### Workaround

For those affected users who want to immediately know the workaround to solve such problem, although not the cleanest solution, [I've found a workaround by using an ordinal number of GPU device](https://www.mql5.com/en/forum/12777/page3#comment_44942605)as listed in Journal tab directly with such function.

So for my case as GPU device listed out at the first record seen in the Journal tab (see the first figure at the top of the article), we shall call

_CLContextCreate(0)_

It will work like a charm.

In case ones use built-in header _OpenCL/OpenCL.mqh_, for **minimally** changes, we have to make one line modification as follows.

From the code below,

```
...
bool COpenCL::Initialize(const string program,const bool show_log)
  {
   if(!ContextCreate(CL_USE_ANY))
      return(false);
   return(ProgramCreate(program,show_log));
  }
...
```

change it to

```
...
bool COpenCL::Initialize(const string program,const bool show_log)
  {
   if(!ContextCreate(0))
      return(false);
   return(ProgramCreate(program,show_log));
  }
...
```

wheres the parameter value should be our desire ordinal number of GPU device as seen in Journal tab when your MetaTrader 5 launched. In my case, it is 0.

Although it is not a clean solution, and we can do better by rewriting as

```
bool COpenCL::Initialize(const string program, const int device, const bool show_log)
  {
   if(!ContextCreate(device))
      return(false);
   return(ProgramCreate(program,show_log));
  }
```

then we will have flexibility to initialize OpenCL context depending on use case without interfering with the existing logic. We can still supply existing flags e.g. _CL\_USE\_ANY_, or specific device ordinal number.

With this change, we have to also make changes to all those samples that use it.

Such _COpenCL::Initialize()_ is used all across the board by built-in OpenCL samples (found at _Scripts/Examples/OpenCL_) namely

- BitonicSort - both Float & Double
- FFT - both Float & Double
- MatrixMult - both Float & Double
- Wavelet - both Float & Double
- Seascape

For example of _Float/BitonicSort.mq5,_

```
void OnStart()
  {
//--- OpenCL
   COpenCL OpenCL;
   if(!OpenCL.Initialize(cl_program,true))
     {
      PrintFormat("Error in OpenCL initialization. Error code=%d",GetLastError());
      return;
     }
     ...
```

change it to

```
void OnStart()
  {
//--- OpenCL
   COpenCL OpenCL;
   if(!OpenCL.Initialize(cl_program, 0, true))
     {
      PrintFormat("Error in OpenCL initialization. Error code=%d",GetLastError());
      return;
     }
     ...
```

Nonetheless, I'm seeking a cleaner solution that would allow us to properly use those context creation flags as they intended to be used. There should be no hard-coded of ordinal number because device listing order probably can change, and we never know exactly what would be the desire value. It makes code safer to deploy to end-users with more flexibility. Our journey to achieve such objective shall begin.

### Game plan

What I thought is that we can try to create a very simple standalone OpenCL application that tries to use GPU as a device then execute some works. If the application is able to detect such GPU, then use it to complete the work, then we can proceed to a more complete solution.

Then we develop a proof-of-concept OpenCL support as a shared library (DLL) in C++ that attempts to connect to GPU as usual then consumed it with MQL5 on MetaTrader 5 platform. After such proof-of-concept is validated, then we can proceed further to implement an equivalent OpenCL support as seen on built-in APIs.

Not just that, we will port _Includes/OpenCL/OpenCL.mqh_ to be basing on our newly developed solution. Also port bunch of OpenCL samples to heavily test our solution, then finally conduct benchmark between built-in and our developed OpenCL support (just for fun).

So following is the summary of our game plan, finish one then go on the next

1. Develop a simple OpenCL testing program as a standalone executable (focus solely on validation in using GPU to execute kernel function)
2. Develop a simple OpenCL support as DLL to test with MQL5 as Script project on MetaTrader 5
3. Develop an OpenCL support as DLL which has equivalent features of built-in OpenCL on MetaTrader 5
4. Port _Includes/OpenCL/OpenCL.mqh_
5. Port OpenCL samples
6. Conduct benchmark between built-in and our OpenCL solution

The objective here is to DIY (do-it-yourself) to resolve the situation at hands for cleaner solution, and learn about OpenCL. It doesn't intend to replace MetaTrader 5's built-in OpenCL support.

### Pre-requisite

I base the development on Ubuntu 20.04, if you use other distros or variants, please kindly adapt to fit your needs.

- Install _mingw64_ package
- Install _winehq-devel_ package
- Install OpenCL driver support as need which depends on your graphics card model.

Please take a look at guideline [How to Install and Use OpenCL for Calculations](https://www.mql5.com/en/articles/690 "Go to article How to Install and Use OpenCL for Calculations") or if you prefer open source graphics driver, then I suggest to look for [ROCm](https://www.mql5.com/go?link=https://github.com/ROCm/ROCm "Go to github project of ROCm") just for OpenCL support but you can install open source graphics driver as usual as it is available normally on Ubuntu.

### Understanding OpenCL (Just Enough)

We are not going to drill down to every single detail and functions of OpenCL, but start from the top to bottom, just enough to get the concept of OpenCL for what it offers in which we use such knowledge to efficiently implement our code.

Please see the following overview of OpenCL platform model, then we will go through each sub-section to get to know more about OpenCL.

![OpenCL Platform Model](https://c.mql5.com/2/0/opencl-platform-model.png)

OpenCL Platform Model - Courtesy of [OpenCL: A Hands-on Introduction (Tim Mattson, Alice Koniges, and Simon McIntosh-Smith)](https://www.mql5.com/go?link=https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf "OpenCL: A Hands-on Introduction (PDF)")

As we can see from the figure, OpenCL device (think GPU) consists of bunch of Compute unit. Each of it consists of dozen of Processing Element (PE). At the big picture, memory divided into host memory, and device memory. Zoom in a little more, see OpenCL Device Architecture Diagram below.

![OpenCL Device Architecture Diagram](https://c.mql5.com/2/0/Screenshot_from_2023-02-10_06-16-20.png)

OpenCL Device Architecture Diagram - Courtesy of [OpenCL API 1.2 Reference Guide](https://www.mql5.com/go?link=https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf "OpenCL API 1.2 Reference Guide (PDF)") by Khronos

Figure above zooms in further and gives us another perspective look on the same model we've seen previously but with additional of memory model layout along with their corresponding capabilities. Each PE has a private memory for fast access before requiring to communicate with local memory, and global/constant cache. Constant memory is fixed (read-only) memory type that is expected not to change during the course of kernel execution. Higher up from cache memory is the main memory of the device itself which requires more latency to get access there. Also as seen, there are similar global/constant memory type in the main memory of the device.

See the following figure for another clearer overview of memory model with some interchangeably terms.

![OpenCL Memory Model](https://c.mql5.com/2/0/opencl-memory-model.png)

OpenCL Memory Model - Courtesy of [OpenCL: A Hands-on Introduction (Tim Mattson, Alice Koniges, and Simon McIntosh-Smith)](https://www.mql5.com/go?link=https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf "OpenCL: A Hands-on Introduction (PDF)")

Similarly, but with notable key terms on work-item, and work-group. We can see PE as work-item. There are lots of work-items as per single work-group. Memory model as previously mentioned dispersed all across the whole architecture working from work-item to work-group, and interconnecting between device and host (think PC). Notable note as seen at the bottom of the figure is that we as a user of OpenCL would be responsible for moving data back and forth between host and device. We will see why this is the case when we get involved with the code. But in short, because data on both ends need to be synchronized for consistency in consuming result from computation or feeding data for kernel execution.

Let's see yet another layout of how work-item, and work-group would actually mean in context of the work to be carried out.

![An N-dimensional domain of work-items (related to work-group)](https://c.mql5.com/2/0/opencl-dim.png)

Relationship between work-items and work-groups in context of kernel to execute - Courtesy of [OpenCL: A Hands-on Introduction (Tim Mattson, Alice Koniges, and Simon McIntosh-Smith)](https://www.mql5.com/go?link=https://www.nersc.gov/assets/pubs_presos/MattsonTutorialSC14.pdf "OpenCL: A Hands-on Introduction (PDF)")

From figure above, that depicts imaginary work to be done in terms of problem space. Understand a problem space, will allow us to know global dimensions and potentially leads us to know local dimensions as well. Such dimension is one of important configurations to be efficient in working with OpenCL API. API itself limits to use at max 3 dimensions (3-dim), so if our problem space needs more than that, then we have to translate such problem into 3-dim problem. Image processing is the prime example here whose global dimensions can be set to 2-dim for WxH which expresses width and height of the image. We can deduce local dimensions to best fit the capability of device we will be using to carry out such task, but mostly value of each global dimension is divisible by corresponding value in local dimension. But how to know such value for local dimension? No worry, we can query device's capability with OpenCL API. We don't have to do a guess work. We can implement a generic solution that works equally the same adapting for all range of GPU device's capability.

As mentioned previously, a single work-group consists of several work-items. Each work-item will be performed by a single thread inside a device itself. A device especially GPU is able to spawn tons of threads to carry out the computation in parallel; imagine each work-group is computed in parallel to others but by how many work-group to be performed in parallel at the time is up to the GPU's capability.

### OpenCL Terminology (Recap)

We've glanced on the key terms (underlined from previous section) related to OpenCL, here is the summary and short recap.

| Key Term | Explanation |
| --- | --- |
| Host | A system that runs OpenCL application and communicates with OpenCL device. The host is responsible for preparation, and management work prior to execution of kernel. It can be PC, workstation, or cluster. |
| Device | A component in OpenCL that executes OpenCL kernel. It can be CPU, GPU, Accelerator, or custom one. |
| Kernel | A function written in OpenCL C language in which it will perform mathematical calculation for us on devices that OpenCL supports |
| Work-item | A unit of execution within a kernel. Collection of work-items are individual instances of the kernel that executed in parallel. Multiple work-items are called work-group. |
| Work-group | A group of work-items that executed in parallel. It is a logical group in which inside can share data and synchronize their execution |
| Private memory | A memory region that is specific to a work-item. Each work-item has its own private memory, and it is not shared between work-items. It is intended to store local variables, and temporary data which are only needed by a single work-item. |
| Local memory | A memory region that shared among work-items within a same work-group. Thus allow work-items to communicate. It is faster to access than global/constant memory. |
| Global memory/cache | A shared memory region that can be accessed by work-group (thus work-items). Cache would be located near the processing unit, but memory is far away (analogy to CPU cache, and RAM respectively). |
| Constant memory/cache | A shared memory region which is intended for read-only (data is not supposed to be changed during execution) that can be accessed by work-group (thus work-items). Cache would be located near the processing unit, but memory is far away (analogy to CPU's instruction cache, and ROM). |
| Global dimension | A configuration as value and number of dimensions e.g. 1024x1024 (2 dimensions with 1024 in each dimension) to describe a problem space. Maximum at 3 dimensions. |
| Local dimensions | A configuration as value and number of dimensions e.g. 128x128 (2 dimensions with 128 in each dimension) to describe number of work-items and work-groups to execute a kernel. For example, in case of global dimensions as of 1024x1024, and local dimensions of 128x128, number of work-items for a single work-group are 128\*128=16,384, number of work-group needed is 1024/128 \* 1024/128 = 8\*8 = 64, total work-item is 16,384\*64 or 1024\*1024 =1,048,576. |
| Processing Element (PE) | A physical computing unit for support device e.g. CPU or GPU. |
| Compute Unit | A collection of PE within an OpenCL device that can execute a multiple threads in parallel. |

### OpenCL Class Diagram

![OpenCL 1.2 Class Diagram](https://c.mql5.com/2/0/opencl-class-diagram.png)

OpenCL Class Diagram - Courtesy of [OpenCL API 1.2 Reference Guide](https://www.mql5.com/go?link=https://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf "OpenCL API 1.2 Reference Guide (PDF)") by Khronos

Frequently used classes in OpenCL that you will be dealing with most of the time are as follows.

You can skim through the following table before we will be seeing usage of all of them in real code soon after.

| Class | Description |
| --- | --- |
| _cl::Platform_ | Provides information about an OpenCL platform e.g. name, vendor, profile, and OpenCL extensions. |
| _cl::Device_ | Represents an OpenCL device e.g. CPU, GPU, or other type of processor that implements OpenCL standard. |
| _cl::Context_ | Represents a logical container for other classes. Users can start from context to query other information. |
| _cl::CommandQueue_ | Represents a command queue which is a queue of commands that will be executed on an OpenCL device. |
| _cl::Program_ | Represents a program which is a set of kernel functions that can be executed on an OpenCL Device. It provides methods for creating a program from kernel code, build a program, and provide ability to query for program information e.g. number of kernels, name, binary size, etc. |
| _cl::Kernel_ | Represents an entry point of OpenCL function name to execute the entire kernel. Whenever users create a kernel, it needs a correct kernel function name as entry point to execute. Users can set arguments prior to execution. |
| _cl::Buffer_ | Represents an OpenCL memory buffer which is a linear region of memory storing data for input and output from kernel execution. |
| _cl::Event_ | Represents an OpenCL event in asynchronous manner for the status of OpenCL command. Users can use it to synchronize between operations between host and device. |

The starting point that can lead to all other classes is Platform ( _cl::Platform_). We will be seeing in the real code soon that this is the case. Usually the workflow of executing computation with OpenCL is as follows

1. Start from _cl::Platform_ to get _cl::Device_ to filter desire type of devices to work with
2. Create _cl::Context_ from _cl::Device_
3. Create _cl::CommandQueue_ from _cl::Context_
4. Create a _cl::Program_ from kernel code
5. Create a _cl::Kernel_ from _cl::Program_
6. Create a few of _cl::Buffer_ to hold input and output data prior to kernel execution (how many depends on problem domain)
7. (Optional) Create _cl::Event_ to synchronize operations; mostly for high performance aim and consistency in data between host and device, in case of proceed in asynchronize API call
8. Begin kernel execution through  _cl::CommandQueue_, and wait for resultant data to be transferred back from device to host.

### Phase I - Validation Using GPU Device with OpenCL by Developing a Simple OpenCL Testing Program

We start with a simple testing program. It will be a standalone executable to validate our assumption that there is something wrong in MetaTrader 5 for its device selection code. So we will create a program to use GPU with OpenCL.

As usual, if you are familiar with cross-compile workflow, you will get used to it by now as we will do similar thing.

You can download the whole project from the attached zip file, and follow along looking at specific section by section following the article. I won't exhaustively list out all the steps as it will make the article too lengthy to read.

The structure of the project files is as follows

**standalonetest**

You can download the project file (.zip) at the bottom of this article.

- opencltest.cpp
- Makefile
- (dependencies .dll as sym-link files to installed .dll files)

  - libgcc\_s\_seh-1.dll - link to _/usr/lib/gcc/x86\_64-w64-mingw32/9.3-posix/libgcc\_s\_seh-1.dll_
  - libstdc++-6.dll - link to _/usr/lib/gcc/x86\_64-w64-mingw32/9.3-posix/libstdc++-6.dll_
  - libwinpthread-1.dll - link to _/usr/x86\_64-w64-mingw32/lib/libwinpthread-1.dll_
  - opencl.dll - link to _~/.mt5/drive\_c/windows/system32/opencl.dll_

You might have to edit the sym-link files to point to the correct location on your systems, Default locations based on Ubuntu 20.04, and default wine's prefix installation of MetaTrader 5. You can edit it with the command _ln -sf <path-to-new-file> <link-name>._

**Makefile**

```
.PHONY: all clean main.exe

COMPILER := x86_64-w64-mingw32-g++-posix
FLAGS := -O2 -fno-rtti -std=c++17 -Wall -Wextra
MORE_FLAGS ?=

all: main.exe

main.exe: opencltest.cpp
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -o $@ $< -L. -lopencl

clean:
        rm -f main.exe
```

We can ignore building for native Linux in this case, and go straight to cross-compile for Windows. In doubt, please refer back to my previous article [Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux](https://www.mql5.com/en/articles/12042 "Go to article Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux") regarding to how to do cross-compile with Makefile build system.

**opencltest.cpp**

We list the full source code first, then explain chunk by chunk.

We will sum all elements of two input arrays together then write the result into 3rd array.

Kernel function will be simple code to reflect what we need to get done. There is a little bit of code management whenever we create buffers associated to all those arrays.

In this case two input arrays will be read-only as we have set their values before the kernel execution. The 3rd array (resultant array) will be write-only in which GPU will take care in writing result for us.

We will talk more about nuances of flags used in buffer creation later as there is something notable to be aware.

So here we go, let's drill down to the code.

```
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/cl2.hpp>

#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cassert>
#include <cmath>

int main() {
        // Get the platform
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms[0];

        // Get the device
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices[0];

        // Create the context
        cl::Context context(device);

        // Create the command queue
        cl::CommandQueue queue(context, device);

        // Create the kernel
        std::string kernelCode = "__kernel void add(__global int* a, __global int* b, __global int* c, int size) { "
                "    int i = get_global_id(0);"
                "         if (i < size)"
                "               c[i] = a[i] + b[i];"
                "}";
        cl::Program::Sources sources;
        sources.push_back({kernelCode.c_str(), kernelCode.length()});
        cl::Program program(context, sources);
        if (auto ret_code = program.build({device});
                ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Program::build, code=" << ret_code << std::endl;
                return -1;
        }
        cl::Kernel kernel(program, "add");

        // Create the input and output arrays
        const int SIZE = 10000000;
        std::vector<int> a(SIZE);
        std::vector<int> b(SIZE);
        std::vector<int> c(SIZE, 0);

        // prepare data
        std::iota(a.begin(), a.end(), 1);
        std::iota(b.rbegin(), b.rend(), 1);

        // Create the buffer
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * a.size(), a.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * b.size(), b.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * c.size());

        {
                cl::Event write_bufferA_event, write_bufferB_event;

                if (auto ret_code = queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, sizeof(int) * a.size(), a.data(), nullptr, &write_bufferA_event);
                        ret_code != CL_SUCCESS) {
                        std::cerr << "1 Error enqueueWriteBuffer() code=" << ret_code << std::endl;
                        return -1;
                }
                if (auto ret_code = queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, sizeof(int) * b.size(), b.data(), nullptr, &write_bufferB_event);
                        ret_code != CL_SUCCESS) {
                        std::cerr << "2 Error enqueueWriteBuffer() code=" << ret_code << std::endl;
                        return -1;
                }

                cl::Event::waitForEvents({write_bufferA_event, write_bufferB_event});
        }

        // Set the kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, SIZE);

        auto start = std::chrono::steady_clock::now();

        // Execute the kernel
        if (auto ret_code = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(SIZE), cl::NullRange);
                ret_code != CL_SUCCESS) {
                std::cerr << "Error enqueueNDRangeKernel() code=" << ret_code << std::endl;
                return -1;
        }

        // Read the result
        if (auto ret_code = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * c.size(), c.data());
                ret_code != CL_SUCCESS) {
                std::cerr << "Error enqueueReadBuffer() code=" << ret_code << std::endl;
                return -1;
        }

        std::chrono::duration<double, std::milli> exec_time = std::chrono::steady_clock::now() - start;
        std::cout << "elapsed time: " << exec_time.count() << "ms" << std::endl;

        // check the result
        for (int i = 0; i < SIZE; i++) {
                assert(c[i] == SIZE + 1);
        }

        return 0;
}
```

Firstly take a look at the top of the source file.

```
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/cl2.hpp>
```

Notice the first two lines. Those definitions via pre-processor of _#define_ _are_ necessary to let OpenCL knows that we aims for OpenCL version 1.2. These two lines need to be there before we include _CL/cl2.hpp_ header file.

Although we aims for OpenCL 1.2, but we chose to include _cl2.hpp_ header file because there is some support features from OpenCL e.g. SVM (Shared Virtual Memory) for more efficient memory access in certain type of applications, device-side enqueue, pipes and possibly a few of modern C++ syntax changed that allows for more convenient in use especially to be aligned with recent version of C++ standard, although we didn't use those OpenCL related features just yet, but whenever MetaTrader 5 upgrades itself later to support OpenCL 2.x, we will have more smooth effort in migration our code base later.

In short, we aim for version 1.2 as it's the official OpenCL version supported by MetaTrader 5.

Next, we will create _cl::Context_ which involves _cl::Platform,_ and _cl::Device._

```
        ...
        // Get the platform
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        cl::Platform platform = platforms[0];

        // Get the device
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Device device = devices[0];

        // validate that it is GPU
        assert(device.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
        ...
```

Platform consists of multiple devices. So in order to find a desire device to work with. In reality, we have to iterate all platforms, and all of devices for each platform. Check whether such device is the right type e.g. CPU, GPU, Accelerator (somewhat specialized device), or Custom. But in our case for quick testing, we hard-code to use the first platform and first ordinal number i.e. 0 indicating a GPU device which found from Journal tab as mentioned at the beginning of this article. We also validate that the device we get is GPU.

By using a flag _CL\_DEVICE\_TYPE\_GPU_, it will list all available GPU associated with such platform for us. We just grab the first one found.

Please be aware that this code is for simple and fast testing, in production as we will do in the future, it's better idea to iterate for all platforms and filter for device that matches our desire, add them into the list then return from the function. Then we can do whatever from such list.

Next, write a kernel function, create _cl::Program_ from the written kernel code, then create _cl::Kernel_ from it.

```
        ...
        // Create the command queue
        cl::CommandQueue queue(context, device);

        // Create the kernel
        std::string kernelCode = "__kernel void add(__global int* a, __global int* b, __global int* c, int size) { "
                "    int i = get_global_id(0);"
                "         if (i < size)"
                "               c[i] = a[i] + b[i];"
                "}";
        cl::Program::Sources sources;
        sources.push_back({kernelCode.c_str(), kernelCode.length()});
        cl::Program program(context, sources);
        if (auto ret_code = program.build({device});
                ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Program::build, code=" << ret_code << std::endl;
                return -1;
        }
        cl::Kernel kernel(program, "add");
        ...
```

Asides, we create _cl::CommandQueue_ for later use. Kernel code can be written into a separate file (generally with file extension _.cl_) or baked-in string along side C++ code as we did here.

There are some notable annotations, and function as seen in kernel code here as follows.

- _\_\_kernel_ \- an important annotation to be placed in front of the function name to signify that such function is a kernel function. Kernel function will be executed on the device. It is callable from the host along with setting of arguments. A kernel function can also call other normal functions such that those normal functions are not needed to be annotated with _\_\_kernel_. So we can think of a kernel function as an entry point of execution or an end-user facing function into computation work done by OpenCL.
- _\_\_global_\- used to declare that the data passed as a parameter stored on global memory which is shared all across work-items (recall diagram figures at the top).
- _get\_global\_id(dimindx)_\- return the unique global work-item ID value from the specified dimension index value ( _dimindx)_. For our case, we have only 1-dim whose value is the size of input array. So this function will uniquely identify which element of the array to be computed altogether in parallel (think of it as index of array).

Aided by above 3 annotations used in the kernel code, the following is its explanation.

- argument _a_, _b_, and _c_ are annotated with _\_\_global_ which means the data that these pointers point to is from global memory shared for all across work-items. So all work-items will be working together to compute the summation and set the result value to the array _c_.
- It has a safe check against _size_ to limit the computation to the size of input/output array; although it is not necessary as it is already limited by value in global dimension.
- _get\_global\_id(0)_ is used to return which index for the current work-item to work on

After that we create _cl::Program_:: _Sources_ to be used as an input into _cl::Kernel_. Note that the name of kernel needs to be correct and exactly the same as the kernel function as we wrote above.

Next we allocate and prepare the data.

```
        ...
        // Create the input and output arrays
        const int SIZE = 10000000;
        std::vector<int> a(SIZE);
        std::vector<int> b(SIZE);
        std::vector<int> c(SIZE, 0);

        // prepare data
        std::iota(a.begin(), a.end(), 1);
        std::iota(b.rbegin(), b.rend(), 1);
        ...
```

Use _std::vector_ here to allocate enough memory expected to use for both _a_, _b_, and _c._ Then use _std::iota_ to populate all element as follows

- array _a_ \- starts from 1 to SIZE filling from the first element of array til the end
- array _b_ \- starts from 1 to SIZE filling from the last element of array til the beginning

This means whenever each element of array _a_ and _b_ sums up together, it will equal to SIZE+1 as always. We will use this as an assert validation later.

Next we create buffers, and prepare data for device.

```
        ...
        // Create the buffer
        cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * a.size(), a.data());
        cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * b.size(), b.data());
        cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(int) * c.size());

        // (This block is OPTIONAL as CL_MEM_COPY_HOST_PTR already took care of copying data from host to device for us. It is for a demonstration of cl::Event usage)
        {
                cl::Event write_bufferA_event, write_bufferB_event;

                if (auto ret_code = queue.enqueueWriteBuffer(bufferA, CL_FALSE, 0, sizeof(int) * a.size(), a.data(), nullptr, &write_bufferA_event);
                        ret_code != CL_SUCCESS) {
                        std::cerr << "1 Error enqueueWriteBuffer() code=" << ret_code << std::endl;
                        return -1;
                }
                if (auto ret_code = queue.enqueueWriteBuffer(bufferB, CL_FALSE, 0, sizeof(int) * b.size(), b.data(), nullptr, &write_bufferB_event);
                        ret_code != CL_SUCCESS) {
                        std::cerr << "2 Error enqueueWriteBuffer() code=" << ret_code << std::endl;
                        return -1;
                }

                cl::Event::waitForEvents({write_bufferA_event, write_bufferB_event});
        }
        ...
```

_bufferA_, _bufferB_, and _bufferC_ are associated with array _a_, _b_, and _c_ respectively. Notice the flags used to create each buffer as it affects how we should prepare the data associated with each corresponding buffer.

See below table

| Flag | Meaning |
| --- | --- |
| _CL\_MEM\_READ\_WRITE_ | Memory object is permitted for reading and writing by a kernel |
| _CL\_MEM\_WRITE\_ONLY_ | Memory object is permitted for writing only by a kernel |
| _CL\_MEM\_READ\_ONLY_ | Memory object is permitted for reading only by a kernel |
| _CL\_MEM\_USE\_HOST\_PTR_ | Indicate that the application wants OpenCL implementation to use the memory that is already allocated by the application |
| _CL\_MEM\_ALLOC\_HOST\_PTR_ | Indicate that the application wants OpenCL implementation to allocate memory for the memory object and also allow accessibility to host |
| _CL\_MEM\_COPY\_HOST\_PTR_ | Same as _CL\_MEM\_ALLOC\_HOST\_PTR_ but it also automatically copies the data to device for us |

So in our code that uses _CL\_MEM\_COPY\_HOST\_PTR_, we allocate memory on both end which are host, and device then automatically copy data from host to device. The reason for this is that the data needs to be synchronized for consistency, and performance for device to operate on such data without a need to use longer latency channel to access memory located on the host, but its own memory region.

Choosing the right flags when create a buffer is somewhat important for high performance application. It depends on a problem space

_bufferA_, and _bufferB_ are created using _CL\_MEM\_READ\_ONLY_ which means the kernel can only read data from it. This makes sense because we already prepared the data prior to kernel execution. Kernel just has to read from it then compute. Contrast this to _bufferC_ which created with _CL\_MEM\_WRITE\_ONLY_. That means kernel is responsible to put the result into it. That also makes sense as the host has no reason at this point to modified the result again.

(optional) For the code inside the parenthesis block,

Such code inside the block is optional. Because we use _CL\_MEM\_COPY\_HOST\_PTR_ for both _bufferA_, and _bufferB_, so underlying system of OpenCL will take care of copying data from host to device for us without us needing to do it again. We added such code there for demonstration of using _cl::Event_ to synchronize the operations.

_cl::CommandQueue::enqueueWriteBuffer()_ will enqueue a command writing data associated with a specified buffer into device. There are two choices whether we want it to

1. _CL\_FLASE_ \- it won't wait until the operation is complete, return immediately (asynchronous, or non-blocking)
2. _CL\_TRUE_ \- it will wait until the enqueued operation is complete, then return (synchronous, or blocking)

As you can see, if supplied parameter with _CL\_FALSE_ for related API call such as _cl::CommandQueue::enqueueWriteBuffer()_ then we need to combine its usage with one of synchronizing primitive such as _cl::Event._

The benefit is that both calls of _cl::CommandQueue::enqueueWriteBuffer()_ will return immediately, and we will wait for them all to finish at once later. Instead of waiting for it to finish one by one, we save time by enqueuing another operation while waiting for all of them to finish.

Next, we set all arguments of kernel function.

```
        ...
        // Set the kernel arguments
        kernel.setArg(0, bufferA);
        kernel.setArg(1, bufferB);
        kernel.setArg(2, bufferC);
        kernel.setArg(3, SIZE);
        ...
```

Straight forward to say, each one corresponds to the type of function's argument. Fourth argument is not necessary as we already set the size to dimensions (1st dimension). Thus it will limit the number of total work-items for all work-groups for our problem. This implies the number of times the kernel execution will be performed. Anyway, for safety and for certain case, setting the size is totally fine.

Next, we finally begin executing the kernel, and wait for the result.

```
        ...
        auto start = std::chrono::steady_clock::now();

        // Execute the kernel
        if (auto ret_code = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(SIZE), cl::NullRange);
                ret_code != CL_SUCCESS) {
                std::cerr << "Error enqueueNDRangeKernel() code=" << ret_code << std::endl;
                return -1;
        }

        // Read the result
        if (auto ret_code = queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(int) * c.size(), c.data());
                ret_code != CL_SUCCESS) {
                std::cerr << "Error enqueueReadBuffer() code=" << ret_code << std::endl;
                return -1;
        }

        std::chrono::duration<double, std::milli> exec_time = std::chrono::steady_clock::now() - start;
        std::cout << "elapsed time: " << exec_time.count() << "ms" << std::endl;
        ...
```

Using _cl::CommandQueue::enqueueNDRangeKernel()_ to enqueue the command (or per say operation) to execute the specified kernel with the specified dimensions.

Right after, we wait for the result to be written back to our resultant array associated with _bufferC_ by using _cl::CommandQueue::enqueueReadBuffer()_. Notice also for parameter of _CL\_TRUE_ which will wait until the operation is done, just like what we've explained before. This means, the device writes result back to resultant array on device's memory, then copy such data back to host which is array _c_ associated with _bufferC._ That's the meaning of calling _enqueueReadBuffer()._

Wrapping the code by benchmarking the execution time using usual _std::chrono::steady\_clock_ for forward-only monotonic clock which is not affected by external factors e.g. adjustment of a system's clock.

Finally, we validate the correctness of the results we got.

```
        ...
        // check the result
        for (int i = 0; i < SIZE; i++) {
                assert(c[i] == SIZE + 1);
        }
        ...
```

As now the resultant data is ready, and it locates on host's memory which is array _c._ We can safely begin looping through all elements in the array then use _assert()_ to do the job.

You can build and execute the program by just executing following commands

```
$ make
x86_64-w64-mingw32-g++-posix -O2 -fno-rtti -std=c++17 -Wall -Wextra  -I. -o main.exe opencltest.cpp -L. -lopencl

$ WINEPREFIX=~./mt5 wine main.exe
elapsed time: 9.1426ms
```

Awesome! If you see no error message, or see no termination of the program (result from _assert())_, then it works totally fine from start til finish.

So we've just validated that finding and using GPU device on the PC with OpenCL has no problem at all! **We also confirm that there is an issue with device selection code in MetaTrader 5 itself.**

### Continued Next Part...

We have learned about OpenCL from top to bottom, understand its concept, architecture, its memory model, then went through practical code example from start to finish. Those knowledge will be well propagated to using it in real proper implementation of DLL, or even normal usage of built-in MQL5's OpenCL API later. Why? Because understanding OpenCL concept will help us generalizing our problem domain to fit OpenCL's global/local dimensions and its work-item/work-group concept, not to mention about efficient memory usage and utilize fullness of device's capability in executing the parallel task at hands.

Next part, we will come back to abstract our standalone project we've done successfully to transform it as a simple DLL then test on MetaTrader 5 as a Script before begin a full-fledge development on OpenCL support as DLL.

See you in next part.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12108.zip "Download all attachments in the single ZIP archive")

[OpenCLSimple.zip](https://www.mql5.com/en/articles/download/12108/openclsimple.zip "Download OpenCLSimple.zip")(2.93 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Understand and Efficiently use OpenCL API by Recreating built-in support as DLL on Linux (Part 2): OpenCL Simple DLL implementation](https://www.mql5.com/en/articles/12387)
- [Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux](https://www.mql5.com/en/articles/12042)

**[Go to discussion](https://www.mql5.com/en/forum/442372)**

![Population optimization algorithms: Invasive Weed Optimization (IWO)](https://c.mql5.com/2/51/invasive-weed-avatar.png)[Population optimization algorithms: Invasive Weed Optimization (IWO)](https://www.mql5.com/en/articles/11990)

The amazing ability of weeds to survive in a wide variety of conditions has become the idea for a powerful optimization algorithm. IWO is one of the best algorithms among the previously reviewed ones.

![Experiments with neural networks (Part 3): Practical application](https://c.mql5.com/2/51/neural_network_experiments_p3_avatar.png)[Experiments with neural networks (Part 3): Practical application](https://www.mql5.com/en/articles/11949)

In this article series, I use experimentation and non-standard approaches to develop a profitable trading system and check whether neural networks can be of any help for traders. MetaTrader 5 is approached as a self-sufficient tool for using neural networks in trading.

![Creating an EA that works automatically (Part 04): Manual triggers (I)](https://c.mql5.com/2/50/aprendendo_construindo_004_avatar.png)[Creating an EA that works automatically (Part 04): Manual triggers (I)](https://www.mql5.com/en/articles/11232)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode.

![Creating an EA that works automatically (Part 03): New functions](https://c.mql5.com/2/50/aprendendo_construindo_003_avatar.png)[Creating an EA that works automatically (Part 03): New functions](https://www.mql5.com/en/articles/11226)

Today we'll see how to create an Expert Advisor that simply and safely works in automatic mode. In the previous article, we started to develop an order system that we will use in our automated EA. However, we have created only one of the necessary functions.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/12108&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049433290357713780)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

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