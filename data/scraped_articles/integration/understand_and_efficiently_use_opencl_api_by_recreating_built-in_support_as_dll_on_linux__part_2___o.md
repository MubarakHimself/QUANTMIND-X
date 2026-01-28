---
title: Understand and Efficiently use OpenCL API by Recreating built-in support as DLL on Linux (Part 2): OpenCL Simple DLL implementation
url: https://www.mql5.com/en/articles/12387
categories: Integration
relevance_score: 10
scraped_at: 2026-01-22T17:23:13.269894
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/12387&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049094103905445068)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/12387#intro)
- [Key Points](https://www.mql5.com/en/articles/12387#key_points)
- [Simple DLL Implementation](https://www.mql5.com/en/articles/12387#simple_dll)

  - _openclsimple.h_ & _openclsimple.cpp_

    - _DLOG_
    - _clsimple\_listall()_
    - _clsimple\_compute()_

  - _util.h_ & _util.cpp_
  - _Makefile_
  - _Makefile-g++_

- [Testing on Linux and for Windows (via Wine)](https://www.mql5.com/en/articles/12387#testing_linux)
- [Testing with MetaTrader 5](https://www.mql5.com/en/articles/12387#testing_mt5)

  - _mql5/OpenCLSimple.mqh_
  - _mql5/TestCLSimple.mq5_

- [Download Source code](https://www.mql5.com/en/articles/12387#download)

### Introduction

This part will walk us through abstracting works we've done in Part 1 of the series for a successful standalone test for OpenCL into a DLL which is usable with MQL5 program on MetaTrader 5.

This altogether will prepare us for developing a full-fledge OpenCL as DLL support in the following part to come.

### Key Points

Sometimes my article is quite long that readers might get lost during the reading process, so from now I will include _Key Points_ section emphasizing notable points worth to pay attention to.

The following is the key points readers would get from reading this article

1. How to properly pass string from DLL to MQL5 program. Notice that we need to make sure the encoding is UTF-16 as MetaTrader 5 uses it for printing out via _Print()._
2. How to build DLL that is usable by MQL5 program on MetaTrader 5.
3. How to use key important APIs as offered by OpenCL C++ API mainly in getting platforms/devices information, and executing kernel function from initialization til getting result back.

### Simple DLL Implementation

What we need to do is to abstract the code we've done in the [previous article](https://www.mql5.com/en/articles/12108 "Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation") into a proper simple library as DLL that we can consume later with MQL5.

The project file structure is as follows

| File | Definition |
| --- | --- |
| _Makefile_ | Cross-compilation for both Windows, and Linux via Mingw64. It automatically copy resultant _openclsimple.dll_ into _Libraries/_ directory as used for DLL search path by MetaTrader 5. |
| _Makefile-g++_ | Native linux compilation for testing purpose. |
| _openclsimple.h_ & _openclsimple.cpp_ | Main header and implementation of openclsimple DLL library. |
| _util.h_ & _util.cpp_ | Part of openclsimple library, it provides a utility function especially a string conversion for the library. |
| _main.cpp_ | Cross-platform main testing program. |
| _mql5/OpenCLSimple.mqh &_ _mql5/TestCLSimple.mq5_ | MQL5 header and script testing program on MetaTrader 5 |

We will have the following 2 function signatures in which we will be implementing as exposed by our DLL

1. _**clsimple\_listall(char\* out, int len, bool utf16=true)**_

    List all platforms and devices for notable information

2. _**clsimple\_compute(const int arr\_1\[\], const int arr\_2\[\], int arr\_3\[\], int num\_elem)**_

    Compute a summation of two input arrays then write into output array as specified

Let's start by implementing a header file.

As usual, we will show the full source code first, then we go through chunk by chunk for its explanation.

**openclsimple.h**

```
#pragma once

#ifdef WINDOWS
        #ifdef CLSIMPLE_API_EXPORT
                #define CLSIMPLE_API __declspec(dllexport)
        #else
                #define CLSIMPLE_API __declspec(dllimport)
        #endif
#else
        #define CLSIMPLE_API
#endif

/**
 * We didn't define CL_HPP_ENABLE_EXCEPTIONS thus there would be no exceptions thrown
 * from any OpenCL related API.
 */
extern "C" {
        /**
         * List all platforms, and devices available.
         * If there any error occurs during the operation of this function, it will
         * print error onto standard error. The resultant text output is still maintained
         * separately.
         *
         * # Arguments
         * - out - output c-string to be filled
         * - len - length of output c-string to be filled
         * - utf16 - whether or not to convert string to UTF-16 encoding. Default is true.
         *                       If used on MetaTrader 5, this flag should be set to true.
         */
        CLSIMPLE_API void clsimple_listall(char* out, int len, bool utf16=true) noexcept;

        /**
         * Compute a summation of two input arrays then output into 3rd array limiting
         * by the number of elements specified.
         *
         * # Arguments
         * - arr_1 - first read-only array input holding integers
         * - arr_2 - second read-only array input holding integers
         * - arr_3 - output integer array to be filled with result of summation of both arr_1 and arr_2
         * - num_elem - number of element to be processed for both arr_1 and arr_2
         *
         * # Return
         * Returned code for result of operation. 0 means success, otherwise means failure.
         */
        CLSIMPLE_API [[nodiscard]] int clsimple_compute(const int arr_1[], const int arr_2[], int arr_3[], int num_elem) noexcept;
};
```

_#ifdef_ section would be common to readers by now as it's required for Windows to export functions from DLL. We can see _WINDOWS_, and _CLSIMPLE\_API\_EXPORT_ definitions that play key role in explicitly export each function. We will be clear whenever we see code of _Makefile_ later.

_extern "C"_ section wraps the public API for functions that can be called by program.

**openclsimple.cpp**

```
#include "openclsimple.h"
#include "util.h"

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/cl2.hpp>

#include <iostream>
#include <vector>
#include <sstream>

#ifdef ENABLE_DEBUG
#include <cstdarg>
#endif

#ifdef ENABLE_DEBUG
const int LOG_BUFFER_SIZE = 2048;
char log_buffer[LOG_BUFFER_SIZE];

inline void DLOG(const char* ctx, const char* format, ...) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(log_buffer, LOG_BUFFER_SIZE-1, format, args);
        va_end(args);

        std::cerr << "[DEBUG] [" << ctx << "] " << log_buffer << std::endl;
}
#else
        #define DLOG(...)
#endif

CLSIMPLE_API void clsimple_listall(char* out, int len, bool utf16) noexcept {
        // Get the platform
        std::vector<cl::Platform> platforms;
        int ret_code = cl::Platform::get(&platforms);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Platform::get(), code=" << ret_code << std::endl;
                return;
        }

        std::stringstream output_str;

        for (size_t i=0; i<platforms.size(); ++i) {
                auto& p = platforms[i];

                std::string tmp_str;

                ret_code = p.getInfo(CL_PLATFORM_NAME, &tmp_str);
                if (ret_code != CL_SUCCESS)
                        std::cerr << "Error cl::Platform::getInfo(), code=" << ret_code << std::endl;
                else
                        output_str << "[" << i << "] Platform: " << tmp_str << std::endl;

                ret_code = p.getInfo(CL_PLATFORM_VENDOR, &tmp_str);
                if (ret_code != CL_SUCCESS)
                        std::cerr << "Error cl::Platform::getInfo(), code=" << ret_code << std::endl;
                else
                        output_str << "Vendor: " << tmp_str << std::endl;

                std::vector<cl::Device> devices;
                ret_code = p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                if (ret_code != CL_SUCCESS) {
                        std::cerr << "Error cl::Platform::getDevices(), code=" << ret_code << std::endl;
                        continue;
                }

                for (size_t j=0; j<devices.size(); ++j) {
                        const auto& d = devices[j];
                        cl_device_type tmp_device_type;

                        ret_code = d.getInfo(CL_DEVICE_NAME, &tmp_str);
                        if (ret_code != CL_SUCCESS)
                                std::cerr << "Error cl::Device::getInfo(), code=" << ret_code << std::endl;
                        else
                                output_str << " -[" << j << "] Device name: " << tmp_str << std::endl;

                        ret_code = d.getInfo(CL_DEVICE_TYPE, &tmp_device_type);
                        if (ret_code != CL_SUCCESS)
                                std::cerr << "Error cl::Device::getInfo(), code=" << ret_code << std::endl;
                        else {
                                if (tmp_device_type & CL_DEVICE_TYPE_GPU)
                                        output_str << " -Type: GPU" << std::endl;
                                else if (tmp_device_type & CL_DEVICE_TYPE_CPU)
                                        output_str << " -Type: CPU" << std::endl;
                                else if (tmp_device_type & CL_DEVICE_TYPE_ACCELERATOR)
                                        output_str << " -Type: Accelerator" << std::endl;
                                else
                                        output_str << " -Type: Unknown" << std::endl;
                        }
                }
        }

        // keep a copy of the string from stringstream
        std::string copy_str = output_str.str();
        if (utf16)
                util::str_to_cstr_u16(copy_str, out, len);
        else
                util::str_to_cstr(copy_str, out, len);
}

CLSIMPLE_API int clsimple_compute(const int arr_1[], const int arr_2[], int arr_3[], int num_elem) noexcept {
        cl_int ret_code = CL_SUCCESS;

        // Get the platform
        std::vector<cl::Platform> platforms;
        ret_code = cl::Platform::get(&platforms);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Platform::get(), code=" << ret_code << std::endl;
                return ret_code;
        }
        DLOG(__FUNCTION__, "found %d platform(s)", platforms.size());

        if (platforms.empty()) {
                std::cerr << "Error found 0 platform." << std::endl;
                return CL_DEVICE_NOT_FOUND;             // reuse this error value
        }

        cl::Platform platform = platforms[0];
        DLOG(__FUNCTION__, "%s", "passed getting platforms");

        // Get the device
        std::vector<cl::Device> devices;
        ret_code = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Platform::getDevices(), code=" << ret_code << std::endl;
                return ret_code;
        }
        DLOG(__FUNCTION__, "found %d GPU device(s)", devices.size());

        if (devices.empty()) {
                std::cerr << "Error found 0 device." << std::endl;
                return CL_DEVICE_NOT_FOUND;
        }
        cl::Device device = devices[0];
        DLOG(__FUNCTION__, "%s", "passed getting a GPU device");

        // Create the context
        cl::Context context(device);

        DLOG(__FUNCTION__, "%s", "passed creating a context");

        // Create the command queue
        cl::CommandQueue queue(context, device);

        DLOG(__FUNCTION__, "%s", "passed creating command queue");

        // Create the kernel
        std::string kernelCode = "__kernel void add(__global int* a, __global int* b, __global int* c, int size) { "
                                                         "              int i = get_global_id(0);"
                                                         "              if (i < size)"
                                                         "                      c[i] = a[i] + b[i];"
                                                         "}";
        cl::Program::Sources sources;
        sources.push_back({kernelCode.c_str(), kernelCode.length()});
        cl::Program program(context, sources);
        ret_code = program.build({device});
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Program::build(), code=" << ret_code << std::endl;
                return ret_code;
        }

        DLOG(__FUNCTION__, "%s", "passed building a kernel program");

        cl::Kernel kernel(program, "add");

        DLOG(__FUNCTION__, "%s", "passed adding kernel function");

        // Create buffers
        cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num_elem, const_cast<int*>(arr_1));
        cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num_elem, const_cast<int*>(arr_2));
        cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num_elem, arr_3);

        kernel.setArg(0, buffer_a);
        kernel.setArg(1, buffer_b);
        kernel.setArg(2, buffer_c);
        kernel.setArg(3, num_elem);

        DLOG(__FUNCTION__, "%s", "passed setting all arguments");

        // execute the kernel function
        // NOTE: this is a blocking call although enqueuing is async call but the current thread
        // will be blocked until he work is done. Work is done doesn't mean that the result buffer
        // will be written back at the same time.
        //
        ret_code = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_elem), cl::NullRange);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::CommandQueue::enqueueNDRangeKernel(), code=" << ret_code << std::endl;
                return ret_code;
        }

        // CL_TRUE to make it blocking call
        // it requires for moving data from device back to host
        // NOTE: Important to call this function to make sure the result is sent back to host.
        ret_code = queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, sizeof(int) * num_elem, arr_3);
        if (ret_code != CL_SUCCESS)
                std::cerr << "Error cl::CommandQueue::enqueueReadBuffer(), code=" << ret_code << std::endl;

        return ret_code;
}
```

There are 3 main parts to be looking at from the code above.

1. _DLOG_ utility for debug logging purpose
2. _clsimple\_listall()_

   - _util.h & util.cpp -_ string conversion utility making it ready to pass string from DLL to MQL5

5. _clsimple\_compute()_

**DLOG**

```
#ifdef ENABLE_DEBUG
#include <cstdarg>
#endif

#ifdef ENABLE_DEBUG
const int LOG_BUFFER_SIZE = 2048;
char log_buffer[LOG_BUFFER_SIZE];

inline void DLOG(const char* ctx, const char* format, ...) {
        va_list args;
        va_start(args, format);
        std::vsnprintf(log_buffer, LOG_BUFFER_SIZE-1, format, args);
        va_end(args);

        std::cerr << "[DEBUG] [" << ctx << "] " << log_buffer << std::endl;
}
#else
        #define DLOG(...)
#endif
```

This is a logging utility that will print out onto standard error output. There is _#ifdef_ guard for conditional checking whenever we build the project with _ENABLE\_DEBUG_ supplied or not, if so then we include required header as well as _DLOG()_ would now mean something, and usable. Otherwise, it is just nothing.

Log buffer is set with fixed size of 2048 bytes per call. We don't expect to have that long debug message, thus it's quite enough for us.

**clsimple\_listall()**

A function to list all devices across all platforms. Those devices are available to be used with OpenCL.

It all starts with _cl::Platform_ to get other information.

```
        ...
        // Get the platform
        std::vector<cl::Platform> platforms;
        int ret_code = cl::Platform::get(&platforms);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Platform::get(), code=" << ret_code << std::endl;
                return;
        }
        ...
```

We will see this pattern of error handling a lot in this function. Firstly, we get a vector of platform. In case of error, we return with return code immediately as we cannot do anything further.

The success return code for working with OpenCL API is _CL\_SUCCESS_. Notice that in case of error, we will always print error message out onto standard error.

Iterate all platforms and devices for each to get information we need

```
        ...
        std::stringstream output_str;

        for (size_t i=0; i<platforms.size(); ++i) {
                ...

                for (size_t j=0; j<devices.size(); ++j) {
                        ...
                }
        }
        ...
```

This function bases on writing the string output to the specified c-string pointer. This means we will be using _std::stringstream_ to avoid having to create a temporary _std::string_ and copying operation every time we need to append a string to the current result we have.

_cl::Platform_ is the starting point to get other information, each platform contains one or more _cl::Device._ So we have a double for-loop to do our work.

Inside the loop

```
        ...
        for (size_t i=0; i<platforms.size(); ++i) {
                auto& p = platforms[i];

                // temporary variables to hold temporary platform/device informatin
                std::string tmp_str;

                ret_code = p.getInfo(CL_PLATFORM_NAME, &tmp_str);
                if (ret_code != CL_SUCCESS)
                        std::cerr << "Error cl::Platform::getInfo(), code=" << ret_code << std::endl;
                else
                        output_str << "[" << i << "] Platform: " << tmp_str << std::endl;

                ret_code = p.getInfo(CL_PLATFORM_VENDOR, &tmp_str);
                if (ret_code != CL_SUCCESS)
                        std::cerr << "Error cl::Platform::getInfo(), code=" << ret_code << std::endl;
                else
                        output_str << "Vendor: " << tmp_str << std::endl;

                std::vector<cl::Device> devices;
                ret_code = p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                if (ret_code != CL_SUCCESS) {
                        std::cerr << "Error cl::Platform::getDevices(), code=" << ret_code << std::endl;
                        continue;
                }

                for (size_t j=0; j<devices.size(); ++j) {
			cl_device_type tmp_device_type;
                        const auto& d = devices[j];

                        ret_code = d.getInfo(CL_DEVICE_NAME, &tmp_str);
                        if (ret_code != CL_SUCCESS)
                                std::cerr << "Error cl::Device::getInfo(), code=" << ret_code << std::endl;
                        else
                                output_str << " -[" << j << "] Device name: " << tmp_str << std::endl;

                        ret_code = d.getInfo(CL_DEVICE_TYPE, &tmp_device_type);
                        if (ret_code != CL_SUCCESS)
                                std::cerr << "Error cl::Device::getInfo(), code=" << ret_code << std::endl;
                        else {
                                if (tmp_device_type & CL_DEVICE_TYPE_GPU)
                                        output_str << " -Type: GPU" << std::endl;
                                else if (tmp_device_type & CL_DEVICE_TYPE_CPU)
                                        output_str << " -Type: CPU" << std::endl;
                                else if (tmp_device_type & CL_DEVICE_TYPE_ACCELERATOR)
                                        output_str << " -Type: Accelerator" << std::endl;
                                else
                                        output_str << " -Type: Unknown" << std::endl;
                        }
                }
        }
        ...
```

Inside the loop, we have temporary variables namely _tmp\_str_, and _tmp\_device\_type_ to hold temporary information as acquiring from platform or device.

Along the way if something goes wrong, we print out error onto standard error, otherwise append the result string onto our _output\_str_.

Notice that we also print out ordinal index number for both platform, and device. This can be useful if we want users to be specific in choosing which platform, and device to work with without us to find the right device every single time. This is an optional, and served as an idea for future expansion of the library.

Information that we are looking to get that is enough for making decision later to choose to use it with OpenCL later is as follows

- _CL\_PLATFORM\_NAME_\- name of the platform
- _CL\_PLATFORM\_VENDOR -_ name of the vendor e.g. AMD, Nvidia, the pocl project, etc
- _CL\_DEVICE\_NAME -_ device name e.g. code name of GPU, or name of CPU
- _CL\_DEVICE\_TYPE -_ device type e.g. GPU, CPU

There are whole bunch of information relating to platform, and device. Developers can take a peek at header file namely _CL/CL2.h_ from location of your system's include path i.e. _/usr/include/_. Excerpted example as follows

Platform information

```
...
/* cl_platform_info */
#define CL_PLATFORM_PROFILE                         0x0900
#define CL_PLATFORM_VERSION                         0x0901
#define CL_PLATFORM_NAME                            0x0902
#define CL_PLATFORM_VENDOR                          0x0903
#define CL_PLATFORM_EXTENSIONS                      0x0904
#ifdef CL_VERSION_2_1
#define CL_PLATFORM_HOST_TIMER_RESOLUTION           0x0905
#endif
...
```

Device information

```
...
/* cl_device_info */
#define CL_DEVICE_TYPE                                   0x1000
#define CL_DEVICE_VENDOR_ID                              0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS                      0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS               0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE                    0x1004
#define CL_DEVICE_MAX_WORK_ITEM_SIZES                    0x1005
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR            0x1006
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT           0x1007
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT             0x1008
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG            0x1009
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT           0x100A
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE          0x100B
#define CL_DEVICE_MAX_CLOCK_FREQUENCY                    0x100C
#define CL_DEVICE_ADDRESS_BITS                           0x100D
#define CL_DEVICE_MAX_READ_IMAGE_ARGS                    0x100E
#define CL_DEVICE_MAX_WRITE_IMAGE_ARGS                   0x100F
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE                     0x1010
#define CL_DEVICE_IMAGE2D_MAX_WIDTH                      0x1011
#define CL_DEVICE_IMAGE2D_MAX_HEIGHT                     0x1012
#define CL_DEVICE_IMAGE3D_MAX_WIDTH                      0x1013
#define CL_DEVICE_IMAGE3D_MAX_HEIGHT                     0x1014
#define CL_DEVICE_IMAGE3D_MAX_DEPTH                      0x1015
#define CL_DEVICE_IMAGE_SUPPORT                          0x1016
#define CL_DEVICE_MAX_PARAMETER_SIZE                     0x1017
#define CL_DEVICE_MAX_SAMPLERS                           0x1018
#define CL_DEVICE_MEM_BASE_ADDR_ALIGN                    0x1019
#define CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE               0x101A
#define CL_DEVICE_SINGLE_FP_CONFIG                       0x101B
#define CL_DEVICE_GLOBAL_MEM_CACHE_TYPE                  0x101C
#define CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE              0x101D
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE                  0x101E
#define CL_DEVICE_GLOBAL_MEM_SIZE                        0x101F
#define CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE               0x1020
#define CL_DEVICE_MAX_CONSTANT_ARGS                      0x1021
#define CL_DEVICE_LOCAL_MEM_TYPE                         0x1022
#define CL_DEVICE_LOCAL_MEM_SIZE                         0x1023
#define CL_DEVICE_ERROR_CORRECTION_SUPPORT               0x1024
#define CL_DEVICE_PROFILING_TIMER_RESOLUTION             0x1025
#define CL_DEVICE_ENDIAN_LITTLE                          0x1026
#define CL_DEVICE_AVAILABLE                              0x1027
#define CL_DEVICE_COMPILER_AVAILABLE                     0x1028
#define CL_DEVICE_EXECUTION_CAPABILITIES                 0x1029
#define CL_DEVICE_QUEUE_PROPERTIES                       0x102A    /* deprecated */
#ifdef CL_VERSION_2_0
#define CL_DEVICE_QUEUE_ON_HOST_PROPERTIES               0x102A
#endif
#define CL_DEVICE_NAME                                   0x102B
#define CL_DEVICE_VENDOR                                 0x102C
#define CL_DRIVER_VERSION                                0x102D
#define CL_DEVICE_PROFILE                                0x102E
#define CL_DEVICE_VERSION                                0x102F
#define CL_DEVICE_EXTENSIONS                             0x1030
#define CL_DEVICE_PLATFORM                               0x1031
#ifdef CL_VERSION_1_2
#define CL_DEVICE_DOUBLE_FP_CONFIG                       0x1032
#endif
...
```

Just that we would need to know which information would be useful to our use case.

Conversion of string into UTF-16 to be consumed by MetaTrader 5

```
        ...
        // keep a copy of the string from stringstream
        std::string copy_str = output_str.str();
        if (utf16)
                util::str_to_cstr_u16(copy_str, out, len);
        else
                util::str_to_cstr(copy_str, out, len);
```

Lastly, string output from _output\_str_ needs to be converted into UTF-16 encoding due to MetaTrader 5 uses it to display text onto its Experts tab.

Now, it's good time to see how _util::str\_to\_cstr and util::str\_to\_cstr\_u16_ implemented.

Note that _util.h_& _util.cpp_ are never meant to be exposed and used by users of DLL. It's internally used within the library only. Thus it has no need to be compliant to C i.e. _export "C"_ as it it the case for MetaTrader 5 when users consume exported functions from DLL.

**util.h**

```
#pragma once

#include <string>

namespace util {
        /**
         * Convert via copying from std::string to C-string.
         *
         * # Arguments
         * - str - input string
         * - out - destination c-string pointer to copy the content of string to
         * - len - length of string to copy from
         */
        void str_to_cstr(const std::string& str, char* out, unsigned len);

        /**
         * Convert via copying from std::string to UTF-16 string.
         *
         * # Arguments
         * - str - input string
         * - out - destination c-string pointer to copy the content of converted string
         *                 of UTF-16 to
         * - len - length of string to copy from
         */
        void str_to_cstr_u16(const std::string& str, char* out, unsigned len);
};
```

Those functions will do the conversion if needed then copy to the destination c-string pointer whose buffer length is the specified one.

**util.cpp**

```
#include "util.h"

#include <cuchar>
#include <locale>
#include <codecvt>
#include <cstring>

namespace util {
        /* converter of byte character to UTF-16 (2 bytes) */
        std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> ch_converter;

        void str_to_cstr(const std::string& str, char* out, unsigned len) {
                const char* str_cstr = str.c_str();
                const size_t capped_len = strlen(str_cstr) <= (len-1) ? strlen(str_cstr) : (len-1);
                std::memcpy(out, str_cstr, capped_len+1);
        }

        void str_to_cstr_u16(const std::string& str, char* out, unsigned len) {
                const char* str_cstr = str.c_str();

                std::u16string u16_str = ch_converter.from_bytes(str);
                const char16_t* u16_str_cstr = u16_str.c_str();

                const size_t capped_len = strlen(str_cstr) <= (len-1) ? strlen(str_cstr) : (len-1);
                std::memcpy(out, u16_str_cstr, capped_len*2+1);
        }
};
```

The lines caps the length of the string to be working with. If length of the string is less than the specified _len_, then just use the length of the string. Otherwise, use _len-1_.

We subtract by 1 to leave a room for a null-terminated character which will be added later in the next line.

The line multiplies by 2 as UTF-16 has double a size of normal character encoding as used by C++ program which is UTF-8.

The line constructs a converter for us which will convert

- from UTF-8 to UTF-16 via _std::wstring\_convert::from\_bytes()_ function
- from UTF-16 to UTF-8 via _std::wstring\_convert::to\_bytes()_ function

The two template arguments for _std::wstring\_convert<\_Codecvt, \_Elem>_ can be described as follows

- _\_Codecvt -_ the source character encoding to convert **from**
- _\_Elem_\- the target character encoding to convert **to**

Finally we use _std::memcpy()_ to copy stream of bytes from the converted source string to the destination c-string pointer.

The following is the example output as tested on my machine.

We will revisit this fully at testing section.

![Example of clsimple_listall() as seen from Journal tab](https://c.mql5.com/2/52/Screenshot_from_2023-03-27_14-36-56.png)

Example of calling clsimple\_listall() on MetaTrader 5

**clsimple\_compute()**

Firstly, let's see the function signature.

```
CLSIMPLE_API int clsimple_compute(const int arr_1[], const int arr_2[], int arr_3[], int num_elem) noexcept {
          ...
}
```

The aim is to abstract the code what we've done in the previous part of the series into a function. Fortunately for this case, most of the code can just be moved into a single function.

We defer fully or at most abstract the code for full-fledge development later. For now, we mostly test DLL to be working properly in full loop with MQL5 on MetaTrader 5.

Thus the code would be very similar.

_clsimple\_compute_ function accepts the following arguments

- _arr\_1_\- array of read-only integer numbers
- _arr\_2 -_ array of read-only integer numbers
- _arr\_3_\- output array for summation of _arr\_1_ and _arr\_2_
- _num\_elem_\- number of elements to process from both input array

The function returns the return code, not the result of the summation of both arrays.

In reality we could change it to _float_ or _double_ for the type of input/output array in order to accommodate the price of asset in floating-point format. But we go with simple concept in this implementation.

Get the platform

```
        ...
        cl_int ret_code = CL_SUCCESS;

        // Get the platform
        std::vector<cl::Platform> platforms;
        ret_code = cl::Platform::get(&platforms);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Platform::get(), code=" << ret_code << std::endl;
                return ret_code;
        }
        DLOG(__FUNCTION__, "found %d platform(s)", platforms.size());

        if (platforms.empty()) {
                std::cerr << "Error found 0 platform." << std::endl;
                return CL_DEVICE_NOT_FOUND;             // reuse this error value
        }

        cl::Platform platform = platforms[0];
        DLOG(__FUNCTION__, "%s", "passed getting platforms");
        ...
```

What has been improved from previous part of the series is extensive error handling. We print out error message along with return error code.

The lines with _DLOG()_ can be ignored but it is useful when we build with _ENABLE\_DEBUG_ for debugging purpose.

In this case, we hard-coded to use the first platform. But we can change the function to accept the ordinal index value of which platform to use based on string listing output if call with the first function _clsimple\_listall()._

Get the device

```
        ...
        // Get the device
        std::vector<cl::Device> devices;
        ret_code = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Platform::getDevices(), code=" << ret_code << std::endl;
                return ret_code;
        }
        DLOG(__FUNCTION__, "found %d GPU device(s)", devices.size());

        if (devices.empty()) {
                std::cerr << "Error found 0 device." << std::endl;
                return CL_DEVICE_NOT_FOUND;
        }
        cl::Device device = devices[0];
        DLOG(__FUNCTION__, "%s", "passed getting a GPU device");
        ...
```

In this case, we seek to find a GPU device from such platform only, and use the first one that found.

Create the context

```
        ...
        // Create the context
        cl::Context context(device);

        DLOG(__FUNCTION__, "%s", "passed creating a context");
        ...
```

Create the command queue

```
        ...
        // Create the command queue
        cl::CommandQueue queue(context, device);

        DLOG(__FUNCTION__, "%s", "passed creating command queue");
        ...
```

Create the kernel

```
        ...
        // Create the kernel
        std::string kernelCode = "__kernel void add(__global int* a, __global int* b, __global int* c, int size) { "
                                                         "              int i = get_global_id(0);"
                                                         "              if (i < size)"
                                                         "                      c[i] = a[i] + b[i];"
                                                         "}";
        cl::Program::Sources sources;
        sources.push_back({kernelCode.c_str(), kernelCode.length()});
        cl::Program program(context, sources);
        ret_code = program.build({device});
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::Program::build(), code=" << ret_code << std::endl;
                return ret_code;
        }

        DLOG(__FUNCTION__, "%s", "passed building a kernel program");

        cl::Kernel kernel(program, "add");

        DLOG(__FUNCTION__, "%s", "passed adding kernel function");
        ...
```

To creating a kernel function, we need to create _cl::Program_ from source string in which we need to construct _cl::Program::Sources_, then feed it as part of parameters of _cl::Kernel's_ constructor.

Create buffers

```
        ...
        // Create buffers
        cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num_elem, const_cast<int*>(arr_1));
        cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num_elem, const_cast<int*>(arr_2));
        cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int) * num_elem, arr_3);
        ...
```

There are 3 buffers.

1. _buffer\_a_\- first input array. It's read-only, and allocated on the host that allows access from device as well.
2. _buffer\_b_\- second input array. Same as 1.
3. _buffer\_c_\- resultant array. It's write-only, and allocated on the host that allows access from device as well.

You can refer to the meaning of flags used in creating OpenCL buffer in [the previous part of the series](https://www.mql5.com/en/articles/12108 "Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation").

Notice that for _arr\_1,_ and _arr\_2_ we do _const\_cast<int\*>_ to remove _const_ out from such variable. This is OK due to we receive _const_ variables into the function. That ensures users that we won't modify anything to them.

But due to constructor of _cl::Buffer_ that requires to pass in pointer of a certain type, we need to satisfy it. So we trust such constructor to not modify anything. It should behave.

Set argument to kernel function

```
        ...
        kernel.setArg(0, buffer_a);
        kernel.setArg(1, buffer_b);
        kernel.setArg(2, buffer_c);
        kernel.setArg(3, num_elem);

        DLOG(__FUNCTION__, "%s", "passed setting all arguments");
        ...
```

Set arguments properly according to the kernel function signature as seen in OpenCL kernel code above.

Execute the kernel function, and wait for result to be written back

```
        ...
        // execute the kernel function
        // NOTE: this is a blocking call although enqueuing is async call but the current thread
        // will be blocked until he work is done. Work is done doesn't mean that the result buffer
        // will be written back at the same time.
        //
        ret_code = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(num_elem), cl::NullRange);
        if (ret_code != CL_SUCCESS) {
                std::cerr << "Error cl::CommandQueue::enqueueNDRangeKernel(), code=" << ret_code << std::endl;
                return ret_code;
        }

        // CL_TRUE to make it blocking call
        // it requires for moving data from device back to host
        // NOTE: Important to call this function to make sure the result is sent back to host.
        ret_code = queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, sizeof(int) * num_elem, arr_3);
        if (ret_code != CL_SUCCESS)
                std::cerr << "Error cl::CommandQueue::enqueueReadBuffer(), code=" << ret_code << std::endl;

        return ret_code;
```

Text indicates the global dimension to be used for such kernel to be executed. In this case, it is the number of elements of array input. We specify _cl::NullRange_ for local dimension to let OpenCL automatically determines the value for us.

It's important to make a call to a function highlighted with red as we need to wait for output (as now stored on the device e.g. GPU) to be written back to the host (our machine). Ignore to do this, we may have a chance that result is not ready to be read yet after returning from this function.

Notice that such function call is blocking-call as we specified with _CL\_TRUE._

**Makefile**

```
.PHONY: all clean openclsimple.dll main.exe

COMPILER := x86_64-w64-mingw32-g++-posix
FLAGS := -O2 -fno-rtti -std=c++17 -Wall -Wextra
MORE_FLAGS ?=

all: openclsimple.dll main.exe
        cp -afv $< ~/.mt5/drive_c/Program\ Files/MetaTrader\ 5/MQL5/Libraries/

openclsimple.dll: util.o openclsimple.o
        @# check if symbolic link file to wine's opencl.dll exists, if not then create one
        test -h opencl.dll && echo "opencl.dll exists, no need to create symbolic link again" || ln -s ~/.mt5/drive_c/windows/system32/opencl.dll ./opencl.dll
        $(COMPILER) -shared $(FLAGS) $(MORE_FLAGS) -fPIC -o $@ $^ -L. -lopencl

openclsimple.o: openclsimple.cpp openclsimple.h
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -DCLSIMPLE_API_EXPORT -DWINDOWS -I. -fPIC -o $@ -c $<

util.o: util.cpp util.h
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -fPIC -o $@ -c $<

main.exe: main.cpp openclsimple.dll
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -DWINDOWS -o $@ $< -L. -lopenclsimple

clean:
        rm -f openclsimple.dll main.exe opencl.dll util.o openclsimple.o
```

Line indicates a technique to not print out a comment line when we build by prefixing the comment line with @.

Line indicates an improvement over Makefile we've done in previous article of the series. Now instead of creating a symlink file pointing to _opencl.dll_ located at wine's prefix i.e. the place of installation of MetaTrader 5 which is at _~/.mt5_ in which the problem lies in different user name as part of the home directory path, we dynamically and newly create a symlink file every time user builds. So the symlink file will point to the correct path according to their username and home directory without a need to overwriting a path as pointed by symlink file we packaged and delivered to user.

Line indicates that we copy the resultant DLL file namely _openclsimple.dll_ to the location of _Libraries/_ that would be used by MetaTrader 5 to find DLLs in run-time. This saves us ton of time during development.

**Makefile-g++**

```
.PHONY: all clean libopenclsimple.so main.exe

COMPILER := g++
FLAGS := -O2 -fno-rtti -std=c++17 -Wall -Wextra
MORE_FLAGS ?=

all: libopenclsimple.so main.out

libopenclsimple.so: util.o openclsimple.o
        @# NOTE: notice capital letters in -lOpenCL
        $(COMPILER) -shared $(FLAGS) $(MORE_FLAGS) -I. -fPIC -o $@ $^ -lOpenCL

openclsimple.o: openclsimple.cpp openclsimple.h util.h
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -fPIC -o $@ -c $<

util.o: util.cpp util.h
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -fPIC -o $@ -c $<

main.out: main.cpp libopenclsimple.so
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -o $@ $< -L. -lopenclsimple

clean:
        rm -f libopenclsimple.so main.out util.o openclsimple.o
```

Similarly for _Makefile-g++_ which intended to be used natively on Linux system for quick testing purpose. Content is similar but with notable difference is that we will be linking with OpenCL library as installed on the system. Its name is different compared to Windows.

### Testing on Linux and for Windows (via Wine)

We have build system ready. Everything is ready for us to at least test natively on Linux, and on Windows (via Wine).

**Linux**

Execute the following command

make -f Makefile-g++

We will have the following output files

- _libopenclsimple.so_
- _main.out_

We can execute the test program with the following command

./main.out

We shall see output similar to the following

![Output from testing a test program on Linux built with Makefile-g++](https://c.mql5.com/2/52/Screenshot_from_2023-03-27_15-52-35.png)

Output from testing main.out on Linux as built from Makefile-g++

The output is correct as I have no on-board GPU, but a graphics card, and of course I do have CPU.

**Windows (via Wine)**

Execute the following command

make

We will have the following output files

- _openclsimple.dll_
- _main.exe_

We can execute the test program with the following command

```
WINEPREFIX=~/.mt5 wine ./main.exe
```

We shall see output similar to the following

![Ouput from testing a test program on Windows (via Wine)](https://c.mql5.com/2/52/Screenshot_from_2023-03-27_15-57-32.png)

Output from testing main.exe for Windows (via Wine)

The reason we always use _WINEPREFIX=~/.mt5_ is because that is the wine's prefix where MetaTrader 5 is installed by default. So we test on the same environment as MetaTrader 5 would be running.

Same output as previously tested on Linux.

Readers can further take output files built from _Makefile_ to test natively on Windows. It would work and output similar result. This is left as exercise to readers.

### Testing with MetaTrader 5

We are ready to test with MQL5 on MetaTrader 5 now.

**mql5/OpenCLSimple.mqh**

```
//+------------------------------------------------------------------+
//|                                                      OpenCLX.mqh |
//|                                          Copyright 2022, haxpor. |
//|                                                 https://wasin.io |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, haxpor."
#property link      "https://wasin.io"

#import "openclsimple.dll"
void clsimple_listall(string& out, int len);
int clsimple_compute(const int& arr_1[], const int& arr_2[], int& arr_3[], int num_elem);
#import
```

Notice highlighted text below from the function signature of _clsimple\_listall()_ as exposed from DLL, the function itself has 3 arguments

```
CLSIMPLE_API void clsimple_listall(char* out, int len, bool utf16=true) noexcept;
```

We don't need to include _utf16_ argument in _.mqh_ file because as per usage with MQL5, we always set such argument to true as we need to convert string to UTF-16 to be printable onto Experts tab of MetaTrader 5.

Defining only first two parameters is enough.

**mql5/TestCLSimple.mq5**

```
//+------------------------------------------------------------------+
//|                                                  TestOpenCLX.mq5 |
//|                                          Copyright 2022, haxpor. |
//|                                                 https://wasin.io |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, haxpor."
#property link      "https://wasin.io"
#property version   "1.00"

#include "OpenCLSimple.mqh"

#define STR_BUFFER_LEN 2048

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
        // 1: test clsimple_listall()
        // construct a string to hold resultant of platforms/devices listing
        string listall_str;
        StringInit(listall_str, STR_BUFFER_LEN, '\0');

        // get platforms/devices and print the result
        clsimple_listall(listall_str, STR_BUFFER_LEN);
        Print(listall_str);

        // 2: test clsimple_compute()
        int arr_1[];
        int arr_2[];
        int arr_3[];

        ArrayResize(arr_1, 10000000);
        ArrayFill(arr_1, 0, ArraySize(arr_1), 1);

        ArrayResize(arr_2, 10000000);
        ArrayFill(arr_2, 0, ArraySize(arr_2), 1);

        ArrayResize(arr_3, 10000000);

        uint start_time = GetTickCount();

        int ret_code = clsimple_compute(arr_1, arr_2, arr_3, ArraySize(arr_1));
        if (ret_code != 0) {
                Print("Error occurs, code=", ret_code);
                return;
        }
        Print("Elapsed time: " + (string)(GetTickCount() - start_time) + " ms");

        bool is_valid = true;
        for (int i=0; i<ArraySize(arr_3); ++i) {
                if (arr_3[i] != 2) {
                        Print("Something is wrong at index=" + (string)i);
                        is_valid = false;
                }
        }

        if (is_valid) {
                Print("Passed test");
        }
}
```

Notice that in order to receive a string output from DLL (returned as c-string pointer by copying its buffer), we need to define a _string_ variable and initialize its capacity for maximum length we would be supporting.

The preparation to call _clsimple\_compute()_ would need a little more effort. We need to declare arrays of integers input, fill them with proper values, and declare an array of integers used for output. Anyway in reality, we would be reading such input data tick by tick from the asset's price and we just need to clean or prepare data on top of that slightly more before supplying them as part of arguments whenever we call _clsimple\_compute()._

Finally we validate the result by checking value of each element in output array. If all things went well, it will print out

Passed test

So place _.mqh_ into a proper location either at the same location as _.mq5_ or in _Includes/_ directory of MetaTrader 5 installation path, then compile such _.mq5_ source, and finally drag-and-drop the the built program onto a chart on MetaTrader 5.

We would see the following similar output as seen on Experts tab.

![Output from testing MQL5 program on MetaTrader 5 as seen on Experts tab](https://c.mql5.com/2/52/Screenshot_from_2023-03-27_16-16-58.png)

Output from testing MQL5 (a Script type) program on MetaTrader 5.

Notice that text is shown properly thanks to our working string conversion utility.

### Download Source code

Readers can download the source code from zip file at the bottom-most of this article, or via github repository at [github.com/haxpor/opencl-simple](https://www.mql5.com/go?link=https://github.com/haxpor/opencl-simple "Github repository for opencl-simple project") (look at _simple/_ directory, _standalonetest/_ is for the previous part of the series).

### Continue next part...

This part 2 of the whole series just walked us through abstracting away the previous work we've done into a proper library implementation, properly build a DLL that can be consumed by both normal C++ program on Linux, Windows (via Wine, or natively), and MQL5 on MetaTrader 5.

It also emphasizes on how to properly pass a string from DLL to MQL5 program as we need to convert it to UTF-16 encoding as used by MetaTrader 5 itself for displaying at least on Experts tab. If string displayed on MetaTrader 5 is correct, then we know we have done things correctly.

Next part in the series, we will dive deeply into OpenCL C++ API to develop a full-fledge feature of OpenCL support as DLL to be used with MQL5.

Along the process until that time, we will understand and know the requirements for us to be at most working efficiently with OpenCL API, and thus transfer such knowledge into developing a high performance MQL5 program powered by OpenCL.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12387.zip "Download all attachments in the single ZIP archive")

[OpenCLSimpleDLL.zip](https://www.mql5.com/en/articles/download/12387/openclsimpledll.zip "Download OpenCLSimpleDLL.zip")(8.3 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://www.mql5.com/en/articles/12108)
- [Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux](https://www.mql5.com/en/articles/12042)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/445042)**
(1)


![Shephard Mukachi](https://c.mql5.com/avatar/2017/5/5920C545-5DFD.jpg)

**[Shephard Mukachi](https://www.mql5.com/en/users/mukachi)**
\|
5 Jun 2024 at 00:00

Great article. Looking forward to the next articles in the series, thanks.


![Population optimization algorithms: Gravitational Search Algorithm (GSA)](https://c.mql5.com/2/0/avatar_GSA.png)[Population optimization algorithms: Gravitational Search Algorithm (GSA)](https://www.mql5.com/en/articles/12072)

GSA is a population optimization algorithm inspired by inanimate nature. Thanks to Newton's law of gravity implemented in the algorithm, the high reliability of modeling the interaction of physical bodies allows us to observe the enchanting dance of planetary systems and galactic clusters. In this article, I will consider one of the most interesting and original optimization algorithms. The simulator of the space objects movement is provided as well.

![Alan Andrews and his methods of time series analysis](https://c.mql5.com/2/0/avatar_Alan_Andrews.png)[Alan Andrews and his methods of time series analysis](https://www.mql5.com/en/articles/12140)

Alan Andrews is one of the most famous "educators" of the modern world in the field of trading. His "pitchfork" is included in almost all modern quote analysis programs. But most traders do not use even a fraction of the opportunities that this tool provides. Besides, Andrews' original training course includes a description not only of the pitchfork (although it remains the main tool), but also of some other useful constructions. The article provides an insight into the marvelous chart analysis methods that Andrews taught in his original course. Beware, there will be a lot of images.

![Backpropagation Neural Networks using MQL5 Matrices](https://c.mql5.com/2/51/Avatar_lggz1x9t4-860i-3kodu0uiq-f2ofqhb1q5z5e1m-rrhtix-35-bsg11hrh.png)[Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)

The article describes the theory and practice of applying the backpropagation algorithm in MQL5 using matrices. It provides ready-made classes along with script, indicator and Expert Advisor examples.

![How to use ONNX models in MQL5](https://c.mql5.com/2/52/onnx_models_avatar.png)[How to use ONNX models in MQL5](https://www.mql5.com/en/articles/12373)

ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. In this article, we will consider how to create a CNN-LSTM model to forecast financial timeseries. We will also show how to use the created ONNX model in an MQL5 Expert Advisor.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/12387&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049094103905445068)

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