---
title: Develop a Proof-of-Concept DLL with C++ multi-threading support for MetaTrader 5 on Linux
url: https://www.mql5.com/en/articles/12042
categories: Trading Systems
relevance_score: 4
scraped_at: 2026-01-23T17:46:37.094720
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=qkyrpjamlvcvlnlzhxzwittqudxycaia&ssn=1769179595381828514&ssn_dr=0&ssn_sr=0&fv_date=1769179595&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F12042&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Develop%20a%20Proof-of-Concept%20DLL%20with%20C%2B%2B%20multi-threading%20support%20for%20MetaTrader%205%20on%20Linux%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917959556998808&fz_uniq=5068635178605149131&sv=2552)

MetaTrader 5 / Examples


### Introduction

Linux has a vibrant development ecosystem and a good ergonomic for software development.

It is attractive for those who enjoy working with command line, ease of installation of application via package manager, OS that is not a blackbox but still pulls you to dig deep learning about its internals, configurable for almost all of the sub-systems, essential development tools that are ready to be used out-of-box, flexible and streamlined environment suitable for software development, etc.

It is available ranging from end-user PC desktop to cloud solution e.g. VPS, or cloud service providers like AWS, Google Cloud.

So, I strongly believe there are some developers here that want to stick to their OS of choice but still want to be able to develop & deliver products to Windows users. Of course, products must work the same seamlessly across platform.

Normally MetaTrader 5 developers just use MQL5 programming language to develop their indicators/experts or related products then publish on the market to end-users without any concern about which OS to base on. They can just rely on MT5's IDE to take care of compiling and building .EX5 executable before delivery (provided that they know how to launch MetaTrader 5 on Linux).

But when developers need to involve developing a custom solution as a shared library (DLL) to further expand and provide additional services that MQL5 programming language alone cannot offer, then they will have to spend more time and effort in seeking for solution of cross-compilation, discovery of gotchas and best practices, getting familiar with the tools, etc.

Those are the reasons that come into this article. By involve cross-compilation solution and ability to build DLL with C++ multi-threading capable, these twos combined are at least a foundation that developer can use as a base to expand further.

I hope it will help you able to continue developing MetaTrader 5 related products on your beloved OS of choice which is Linux going forward.

### Who is this article for

I assume readers who come to read this article already had some experience in interacting with Linux via command line and has general concept of compilation and building a C++ source code on Linux.

Anyhow, this article is for those who want to explore steps and workflow to develop DLL capable of multi-threading capability on Linux but works also on Windows. Expand threading programming options at hands not only just built-in OpenCL, but more flexible back-to-basic portable C++ code with its multi-threading capability to integrate with some other systems that are tightly based on it.

### System & Software Used

- Ubuntu 20.04.3 LTS with kernel version 5.16.0 on AMD Ryzen 5 3600 6-Core Processor (2 threads per core), 32 GB of RAM
- Wine ( _winehq-devel_ package) version 8.0-rc3 (as of this writing) - (also see [MT5 build 3550 immediately crashed if launched with _winehq-stable_ package](https://www.mql5.com/en/blogs/post/751281 "MT5 BUILD 3550 BROKEN LAUNCHING ON LINUX THROUGH WINE. HOW TO SOLVE?") as to why this article decided to use devel, but not stable package)
- Mingw ( _mingw-w64_ package) version 7.0.0-2
- Virtualbox version 6.1 for testing on Windows system


### Game plan

We will base on the following plan

01. Get to know Wine
02. Get to know Mingw
03. Mingw's threading implementations

    1. POSIX (pthread)
    2. Win32 (via [mingw-std-threads](https://www.mql5.com/go?link=https://github.com/meganz/mingw-std-threads "Github page of mingw-std-threads as header-only drop-in working with mingw"))

05. Prepare a Linux Development Machine

    1. Wine Installation
    2. MetaTrader 5 Installation
    3. Mingw Installation
    4. (Optional) mingw-std-threads Installation

07. Proof of concept, Development phase I - DLL (C++ multi-threading support)
08. Proof of concept, Development phase II - MQL5 Code to Consume DLL
09. Testing on Windows system
10. Simple benchmark of Mingw's Threading Implementations

### Wine

[Wine](https://www.mql5.com/go?link=https://www.winehq.org/ "Wine official website") is shorted for (technically saying as [recursive](https://en.wikipedia.org/wiki/Recursive_acronym "Wikipedia for recursive acronym") [backronym](https://en.wikipedia.org/wiki/Backronym "Wikipedia of Backronym")) "Wine is Not an Emulator". It is not an emulator to emulate any processor, or target hardware. Instead, it is a wrapper for win32 API that works on non-Windows OS.

Wine introduces another abstract layer that intercepts the call to win32 API from users on non-Windows system, then route it to Wine's internals then process and behave the request just like (or almost the same way) when it should behave on Windows.

This means that Wine operates on those win32 API by means of using [POSIX](https://en.wikipedia.org/wiki/POSIX "Wikipedia of POSIX") API. Readers might experience using Wine software without knowing when they launch such Windows software on Linux, or even play steam games on Linux as its runtime bases on variant of Wine called Proton.

This allows for flexibility in testing out, or using Windows software whereas its alternatives are not available on Linux.

Normally when you want to run Windows based application via Wine, you would execute the following command

```
wine windows_app.exe
```

or if you want to run the application while associating with a specific Wine environment prefix, you would do

```
WINEPREFIX=~/.my_mt5_env wine mt5_terminal64.exe
```

### Mingw

[Mingw](https://www.mql5.com/go?link=https://www.mingw-w64.org/ "https://www.mingw-w64.org/") stands for "Minimalist GNU for Windows". It is a port of GNU compiler collection (GCC) and its toolchains for use to compile C/C++ and some other programming languages on Linux but target Windows.

Features, compilation flags/options are available consistently and similarly from both GCC and Mingw, so users can easily transition their existing knowledge from GCC to Mingw. Also please note that GCC has very similarity in compilation flags/options to Clang as well. So you can see seamless of usage, and users are able to retain their knowledge but with additional benefit to be able to expand user base onto Windows system.

See the following comparison table for a glimpse of differences.

- **Compile C++ source code and build a shared library**

| Compiler | Command line |
| --- | --- |
| GCC | ```<br>g++ -shared -std=c++17 -fPIC -o libexample.so example.cpp -lpthread<br>``` |
| Mingw | ```<br>x86_64-w64-mingw32-g++-posix -shared -std=c++17 -fPIC -o example.dll example.cpp -lpthread<br>``` |

### - Compile C++ source code and build an executable binary         | Compiler | Command line | | --- | --- | | GCC | ```<br>g++ -std=c++17 -I. -o main.out main.cpp -L. -lexample<br>``` | | Mingw | ```<br>x86_64-w64-mingw32-g++-posix -std=c++17 -I. -o main.exe main.cpp -L. -lexample<br>``` |

Readers would notice that differences are minimal. Compilation flags are very similar, mostly the same. Only that we use difference compiler binary to compile and build all the things we need.

There are 3 variants to use in which it leads to topic of threading implementation in which we will explain it in the next section.

1. **x86\_64-w64-mingw32-g++**

    It is aliased to _x86\_64-w64-mingw32-g++-win32_.

2. **x86\_64-w64-mingw32-g++-posix**

    Binary executable that intended to work with pthread.

3. **x86\_64-w64-mingw32-g++-win32**

    Binary executable that intended to work with win32 API threading model. It is aliased to by _86\_64-w64-mingw32-g++_.

In additional, there are several other tools prefixed with

```
x86_64-w64-mingw32-...
```

Some examples are as follows

- _x86\_64-w64-mingw32-gcc-nm_ \- name mangling tool
- _x86\_64-w64-mingw32-gcc-ar_\- archive management tool

- _x86\_64-w64-mingw32-gcc-gprof -_ a performance analysis tool for Unix-like operating systems

as well, there is _x86\_64-w64-mingw32-gcc-nm- **posix**,_ and _x86\_64-w64-mingw32-gcc-nm- **win32**_ variants but for some.

### Mingw Threading Implementations

From previous section, we now know that there are 2 variant of threading implementation provided by Mingw.

1. POSIX (pthread)
2. Win32

Why do we need to be so concerned about this at all? There are 2 reasons I can think of

1. **For safety and compatibility**

Whenever your code potentially uses both C++ multi-threading capability (e.g._std::thread_,_std::promise_, etc) along with OS's native multi-threading support e.g._CreateThread()_for win32 API, and_pthread\_create()_for POSIX API, so it's better to stick to using one API over the other.



    Anyhow, it's very less likely we will mingle the code using muti-threading capability from C++, and OS support together unless a very specific situation arises that OS support API offers more features that C++ cannot. So better maintain consistency, and use the same threading model for both.

    If we use Mingw's pthread implementation, then try not to use win32 API's threading capability. Likewise, if we use Mingw's win32 threading implementation (from now will say shortly as "win32 thread"), then better avoid using OS's pthread API.

2. **Performance** (later see in Simple benchmark of Mingw Threading Implementations section)

    Of course, users want a low latency multi-threading solution. Faster solution to execute in certain situation is likely the one user would pick to use.

We will firstly develop our proof-of-concept DLL and test program first before we will conduct benchmark for both threading implementations.

For the project we will provide portable code to use either pthread or win32 thread whose our build system is able to switch to one another easily.

In case of using win32 thread, we need to install headers from [mingw-std-threads](https://www.mql5.com/go?link=https://github.com/meganz/mingw-std-threads "Github page of mingw-std-threads project") project in which we will guide readers on how to do it next.

### Prepare Linux Development Machine

Before jump right into coding part, we need to install required software first.

**Wine Installation**

Execute the following command to install Wine devel pacakge.

```
sudo apt install winehq-devel
```

then check if it works properly with the following command,

wine --version

Its output will be something like

wine-8.0-rc3

**MetaTrader 5 Installation**

Most users already installed MetaTrader 5 long before build 3550 which is the build that has the crash problem. In order to switch to use _winehq-devel_ package to solve the problem and able to launch MetaTrader 5, we cannot directly use an official installation script as seen on [How to Install the Platform on Linux](https://www.metatrader5.com/en/terminal/help/start_advanced/install_linux "MT5 Official guideline on How to Install the Platform on Linux").

It's better to execute commands ourselves because directly execute official installation script will overwrite our Wine back to stable package.

I've written the guideline on [MT5 Build 3550 Broken Launching On Linux Through Wine. How To Solve?](https://www.mql5.com/en/blogs/post/751281) . That article should cover all the cases either for users that already install Wine stable package, or if users that want to start fresh with devel package.

After all, try launching MetaTrader 5 through Wine again. See if there's any problem.

Note

Official installation script will create a Wine environment (called prefix) at _~/.mt5_. It could be convenient to have the following line in your ~/.bash\_aliases so you can launch MetaTrader 5 with ease.

alias mt5trader="WINEPREFIX=~/.mt5 wine '/home/haxpor/.mt5/drive\_c/Program Files/MetaTrader 5/terminal64.exe'"

then source it with

source ~/.bash\_aliases

finally execute the following command to launch MT5 in which its debug output will be shown on terminal.

mt5trader

Launching MetaTrader 5 in this way will allow us to see our debug log from our proof-of-concept application with ease later without complicate our code unnecessary.

**Mingw Installation**

Execute the following command to install Mingw.

sudo apt install mingw-w64

This will install bunch of tools as a collection into your system in which those tools prefixed with _x86\_64-w64-mingw32-._ Mostly we will be working with either _x86\_64-w64-mingw32-g++-posix_ (or _x86\_64-w64-mingw32-win32_ in case of using win32 thread).

**mingw-std-threads Installation**

mingw-std-threads is a project that glues Mingw's win32 thread to work on Linux. It is a header-only as a drop-in solution. So the installation is straight forward and only need to place its header file into system's include path.

Follow the steps below to install.

Firstly, clone the git repository into your system.

git clone git@github.com:Kitware/CMake.git

Then make a directory to hold its header at system's include path.

sudo mkdir /usr/x86\_64-w64-mingw32/include/mingw-std-threads

Finally copy all header files (.h) from the cloned project's directory to such newly created directory.

cp -av \*.h /usr/x86\_64-w64-mingw32/include/mingw-std-threads/

That's all. Then in code, if we decide to use win32 thread, for some header files related to multi-threading capability (e.g. threads, synchronization primitives, etc) we will need to include it from proper path with a name substitution. See a table below for the exhaustive list.

| C++11 Multi-threading Header File Inclusion | mingw-std-threads Header File Inclusion to Change to |
| --- | --- |
| #include <mutex> | #include <mingw-std-threads/mingw.mutex.h> |
| #include <thread> | #include <mingw-std-threads/mingw.thread.h> |
| #include <shared\_mutex> | #include <mingw-std-threads/mingw.shared\_mutex.h> |
| #include <future> | #include <mingw-std-threads/mingw.future.h> |
| #include <condition\_variable> | #include <mingw-std-threads/mingw.condition\_variable.h> |

### Proof-of-Concept, Development Phase I - DLL (C++ multi-threading support)

Now it's time to get to code.

Our goal here is to implement a proof-of-concept DLL solution capable of using multi-threading capability from C++11 standard library so readers can get the idea, and expand further.

The following is our library and application structure.

**Project Structure**

- **DLL**

  - example.cpp
  - example.h

- **Consumer**

  - main.cpp

- **Build system**

  - **Makefile**\- a cross-compilation build file using Mingw's pthread
  - **Makefile-th\_win32** \- a cross-compilation build file using Mingw's win32 thread
  - **Makefile-g++** \- build file to test on native Linux. This is for quick iteration and debugging while developing the project.

**C++ Standard Used**

We will use C++17 standard although we will be mostly using features from C++11, but a few e.g. attribute of code annotation like _\[\[nodiscard\]\]_ requires C++17.

**DLL**

**example.h**

```
#pragma once

#ifdef WINDOWS
        #ifdef EXAMPLE_EXPORT
                #define EXAMPLE_API __declspec(dllexport)
        #else
                #define EXAMPLE_API __declspec(dllimport)
        #endif
#else
        #define EXAMPLE_API
#endif

// we have to use 'extern "C"' in order to export functions from DLL to be used
// in MQL5 code.
// Using 'namespace' or without such extern won't make it work for MQL5 code, it
// won't be able to find such functions.
extern "C" {
	/**
	 * Add two specified number together.
	 */
        EXAMPLE_API [[nodiscard]] int add(int a, int b) noexcept;

	/**
	 * Subtract two specified number.
	 */
        EXAMPLE_API [[nodiscard]] int sub(int a, int b) noexcept;

	/**
	 * Get the total number of hardware's concurrency.
	 */
	EXAMPLE_API [[nodiscard]] int num_hardware_concurrency() noexcept;

	/**
	 * Sum all elements from specified array for number of specified elements.
	 * The computation will be done in a single thread linearly manner.
	 */
	EXAMPLE_API [[nodiscard]] int single_threaded_sum(const int arr[], int num_elem);

	/**
	 * Sum all elements from specified array for number of specified elements.
	 * The computation will be done in a multi-thread.
	 *
	 * This version is suitable for processor that bases on MESI cache coherence
	 * protocol. It won't make a copy of input array of data, but instead share
	 * it among all threads for reading purpose. It still attempt to write both
	 * temporary and final result with minimal number of times thus minimally
	 * affect the performance.
	 */
	EXAMPLE_API [[nodiscard]] int multi_threaded_sum_v2(const int arr[], int num_elem);
};
```

_#pragma once_ although is not part of C++ standard but it's supported by GCC thus also Mingw. It's a flexible and shorter way to prevent duplicated header inclusion.

If not use such directive, users would use both #ifdef and _#define_ and need to make sure each definition has unique name for each header file. It can be time consuming.

We have _#ifdef WINDOWS_ to guard definition declaration of _EXAMPLE\_API._ This allows us to be able to do compilation with Mingw and native Linux system. Thus whenever we want to do a cross-compilation for a shared library then we add both _-DWINDOWS_ and - _DEXAMPLE\_EXPORT_ to compilation flag, otherwise if we compile for just a testing main program, then we can drop _-DEXAMPLE\_EXPORT_.

_\_\_declspec(dllexport)_ is a directive to export a function from DLL

_\_\_declspec(dllimport)_ is a directive to import a function from DLL.

Above twos are needed for compilation in order to work with DLL on Windows. For non-Windows system, we don't need them but still need for cross-compilation. Thus it's empty for _EXAMPLE\_API_ in case of no definition of _WINDOWS_ for compiling for Linux.

Next, it's a juicy important part of function signatures. Function signatures need to be compatible with C (programming language) calling convention.

This _extern "C"_ will prevent function signatures to be mangled into C++ calling convention.

We cannot wrap function signatures inside _namespace_, or declare them as free functions because MQL5 code won't be able to find those signatures when we consume the DLL later.

For _num\_hardware\_concurrency()_, it will return number of concurrent threads supported by the implementation.

For example, I use 6-core processor with 2 threads per core, thus it has virtually 12 threads that can be work concurrently. In my case, it will return 12.

Both _single\_threaded\_sum()_ and _multi\_threaded\_sum\_v2()_ are prime example for our proof-of-concept application to show benefit of using multi-threading, and compare performance between the twos.

**example.cpp**

```
#include "example.h"

#ifdef USE_MINGW_STD_THREAD
        #include <mingw-std-threads/mingw.thread.h>
#else
        #include <thread>
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <atomic>

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

        std::cout << "[DEBUG] [" << ctx << "] " << log_buffer << std::endl;
}
#else
        #define DLOG(...)
#endif

EXAMPLE_API int add(int a, int b) noexcept {
        return a + b;
}

EXAMPLE_API int sub(int a, int b) noexcept {
        return a - b;
}

EXAMPLE_API int num_hardware_concurrency() noexcept {
        return std::thread::hardware_concurrency();
}

EXAMPLE_API int single_threaded_sum(const int arr[], int num_elem) {
        auto start = std::chrono::steady_clock::now();

        int local_sum = 0;
        for (int i=0; i<num_elem; ++i) {
                local_sum += arr[i];
        }

        std::chrono::duration<double, std::milli> exec_time = std::chrono::steady_clock::now() - start;
        std::cout << "elapsed time: " << exec_time.count() << "ms" << std::endl;
        return local_sum;
}

EXAMPLE_API int multi_threaded_sum_v2(const int arr[], int num_elem) {
        auto start = std::chrono::steady_clock::now();

        std::vector<std::pair<int, int>> arr_indexes;

        const int num_max_threads = std::thread::hardware_concurrency() == 0 ? 2 : std::thread::hardware_concurrency();
        const int chunk_work_size = num_elem / num_max_threads;

        std::atomic<int> shared_total_sum(0);

        // a lambda function that accepts input vector by reference
        auto worker_func = [&shared_total_sum](const int* arr, std::pair<int, int> indexes) {
                int local_sum = 0;
                for (int i=indexes.first; i<indexes.second; ++i) {
                        local_sum += arr[i];
                }
                shared_total_sum += local_sum;
        };

        DLOG("multi_threaded_sum_v2", "chunk_work_size=%d", chunk_work_size);
        DLOG("multi_threaded_sum_v2", "num_max_threads=%d", num_max_threads);

        std::vector<std::thread> threads;
        threads.reserve(num_max_threads);

        for (int i=0; i<num_max_threads; ++i) {
                int start = i * chunk_work_size;
                // also check if there's remaining to piggyback works into the last chunk
                int end = (i == num_max_threads-1) && (start + chunk_work_size < num_elem-1) ? num_elem : start+chunk_work_size;
                threads.emplace_back(worker_func, arr, std::make_pair(start, end));
        }

        DLOG("multi_threaded_sum_v2", "thread_size=%d", threads.size());

        for (auto& th : threads) {
                th.join();
        }

        std::chrono::duration<double, std::milli> exec_time = std::chrono::steady_clock::now() - start;
        std::cout << "elapsed time: " << exec_time.count() << "ms" << std::endl;

        return shared_total_sum;
}
```

Above is the full code, but let's dissect each part individually for ease of understanding. Each chunk of code will be explained more below.

Switchable between using pthread and win32 thread.

```
#include "example.h"

#ifdef USE_MINGW_STD_THREAD
        #include <mingw-std-threads/mingw.thread.h>
#else
        #include <thread>
#endif
```

With this setup, we can integrate nicely with our build system to switch between using and linking with either pthread or win32 thread. By adding _-DUSE\_MINGW\_STD\_THREAD_ in compilation flag in order to use win32 thread when we do cross-compilation.

Implement simple interfaces, straight forward.

```
EXAMPLE_API int add(int a, int b) noexcept {
        return a + b;
}

EXAMPLE_API int sub(int a, int b) noexcept {
        return a - b;
}

EXAMPLE_API int num_hardware_concurrency() noexcept {
        return std::thread::hardware_concurrency();
}
```

_add()_ and _sub()_ are straight forward and easy to understand. For _num\_hardware\_concurrency()_, we need to include header _<thread>_ in order to use _std::thread::hardware\_concurrency()_.

Debug log utility function.

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

        std::cout << "[DEBUG] [" << ctx << "] " << log_buffer << std::endl;
}
#else
        #define DLOG(...)
#endif
```

By adding _-DENABLE\_DEBUG_ in compilation flag, we enable debug log to be printed out onto console. That's why I suggest to launch MetaTrader 5 via command line, so we can debug our program accordingly.

Whenever we didn't define such definition, _DLOG()_ means nothing and it won't have any effect to our code either in terms of speed in execution, or in terms of binary size of shared library or executable binary. It's very nice.

_DLOG()_ is designed with inspiration from Android development as usually there will be a context string (from whichever component debug log comes from) which is _ctx_ in this case, then follows with the debug log string.

Implementation of a single threaded summation function.

```
EXAMPLE_API int single_threaded_sum(const int arr[], int num_elem) {
        auto start = std::chrono::steady_clock::now();

        int local_sum = 0;
        for (int i=0; i<num_elem; ++i) {
                local_sum += arr[i];
        }

        std::chrono::duration<double, std::milli> exec_time = std::chrono::steady_clock::now() - start;
        std::cout << "elapsed time: " << exec_time.count() << "ms" << std::endl;
        return local_sum;
}
```

It simulates the real usage working with MQL5 code. Imagine the situation that MQL5 sends some data as array into DLL function to compute something before DLL returns the result to MQL5 code.

This is it. But for this function, it linearly iterates all element from the specified input array one by one for total number of element as specified by _num\_elem._

The code also benchmarks the total time in execution by using _std::chrono_ library to compute elapsed time. Carefully note that we use _std::chrono::steady\_clock_ which is a monotonic clock, moving forward without any effect from system clock adjustment. It is totally suitable for measuring interval of time.

Implementation of a multi-threaded summation function.

```
EXAMPLE_API int multi_threaded_sum_v2(const int arr[], int num_elem) {
        auto start = std::chrono::steady_clock::now();

        std::vector<std::pair<int, int>> arr_indexes;

        const int num_max_threads = std::thread::hardware_concurrency() == 0 ? 2 : std::thread::hardware_concurrency();
        const int chunk_work_size = num_elem / num_max_threads;

        std::atomic<int> shared_total_sum(0);

        // a lambda function that accepts input vector by reference
        auto worker_func = [&shared_total_sum](const int arr[], std::pair<int, int> indexes) {
                int local_sum = 0;
                for (int i=indexes.first; i<indexes.second; ++i) {
                        local_sum += arr[i];
                }
                shared_total_sum += local_sum;
        };

        DLOG("multi_threaded_sum_v2", "chunk_work_size=%d", chunk_work_size);
        DLOG("multi_threaded_sum_v2", "num_max_threads=%d", num_max_threads);

        std::vector<std::thread> threads;
        threads.reserve(num_max_threads);

        for (int i=0; i<num_max_threads; ++i) {
                int start = i * chunk_work_size;
                // also check if there's remaining to piggyback works into the last chunk
                int end = (i == num_max_threads-1) && (start + chunk_work_size < num_elem-1) ? num_elem : start+chunk_work_size;
                threads.emplace_back(worker_func, arr, std::make_pair(start, end));
        }

        DLOG("multi_threaded_sum_v2", "thread_size=%d", threads.size());

        for (auto& th : threads) {
                th.join();
        }

        std::chrono::duration<double, std::milli> exec_time = std::chrono::steady_clock::now() - start;
        std::cout << "elapsed time: " << exec_time.count() << "ms" << std::endl;

        return shared_total_sum;
}
```

Notice that it's marked as v2, I leave this for historical reason. In short, for modern processor which employs [MESI](https://en.wikipedia.org/wiki/MESI_protocol#:~:text=The%20MESI%20protocol%20is%20an,Illinois%20at%20Urbana-Champaign). "Wikipedia of MESI Cache Coherence Protocol") cache coherence protocol, it's not necessary to make a copy of data set feeding into each thread because MESI will mark such cacheline to be shared among multiple threads and not waste any CPU cycle for signaling & waiting for respond back.

My previous v1 implementation went in full effort to make a copy of data set feeding into each thread. But as per mentioned reason of modern processor already use MESI, it is not necessary to include such attempt in the source code. v1 is much slower than v2 for ~2-5x times.

Notice _worker\_func_ which is a lambda function that works on original array of data, and range of data to be working with (pair of beginning and ending indexes). It sums all elements inside the loop into a local variable in order to avoid [false sharing](https://en.wikipedia.org/wiki/False_sharing "Wikipedia of false sharing") problem which can significantly slow performance down before finally adds up to a shared summation variable across all threads atomically. It uses _std::atomic_ to help making it thread-safe. The number of time such shared summation variable needs to be modified is minimal enough that it won't significant affect the performance. Balancing the practical of implementation and speed gain is a way to go.

We compute how many threads would be needed to split the work, thus will know later the range of work for each thread. Notice that _std::hardware\_concurrency()_ can return 0 which means it might not be able to determine the number of threads, thus we handle such case as well and fallback to 2.

Next we create a vector of threads. Reserve its capacity to _num\_max\_threads._ Then iteratively compute the range of data set for each thread to be working on. Notice that for the last thread, it will take all the remaining data as mostly the number of elements of work to be done might not be divisive by the number of computed threads to use.

Importantly, we join all threads. For more complicated circumstance, we might need asynchronous environment which doesn't block MQL5 code from waiting for the result. With that, we usually use _std::future_ which is a base for all those _std::async, std::promise,_ and _std::packaged\_task._ So we usually have at least 2 interfaces, one to make a request sending data from MQL5 code to compute by DLL without blocking, and another to receive the result from such request back on-demand at which point, it will block the call on MQL5 code. I might write about this in future article.

In additional, along the way, we can use _DLOG()_ to print some debugging states. It's helpful for debugging.

Next, let's implement the portable main testing program that will run on native Linux, and cross-compiled environment via Wine.

**main.cpp**

```
#include "example.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <memory>

int main() {
        int res = 0;

        std::cout << "--- misc ---\n";
        res = add(1,2);
        std::cout << "add(1,2): " << res << std::endl;
        assert(res == 3);

	res = 0;

        res = sub(2,1);
        std::cout << "sub(2,1): " << res << std::endl;
        assert(res == 1);

	res = 0;

        std::cout << "hardware concurrency: " << num_hardware_concurrency() << std::endl;
        std::cout << "--- end ---\n" << std::endl;

        std::vector<int> arr(1000000000, 1);

        std::cout << "--- single-threaded sum(1000M) ---\n";
        res = single_threaded_sum(arr.data(), arr.size());
        std::cout << "sum: " << res << std::endl;
        assert(res == 1000000000);
        std::cout << "--- end ---\n" << std::endl;

        res = 0;

        std::cout << "--- multi-threaded sum_v2(1000M) ---\n";
        res = multi_threaded_sum_v2(arr.data(), arr.size());
        std::cout << "sum: " << res << std::endl;
        assert(res == 1000000000);
        std::cout << "--- end ---" << std::endl;

        return 0;
}
```

We include _example.h_ header file normally to be able to make calls to those implemented interfaces. We also validate that the results are correct by using _assert()_.

Next, let's build both the shared library (as _libexample.so_ for native Linux), and main testing program namely _main.out_. Not with a build system first, but via command line execution. We will properly implement a build system via Makefilelater.

We test it locally on Linux first before doing cross-compilation.

Execute the following command to build a shared library to output as _libexample.so_.

```
$ g++ -shared -std=c++17 -Wall -Wextra -fno-rtti -O2 -I. -fPIC -o libexample.so example.cpp -lpthread
```

Explanation for each flag is as follows

| Flag | Description |
| --- | --- |
| -shared | Instruct compiler to build a shared library |
| -std=c++17 | Instruct compiler to base on C++ syntax on C++17 standard |
| -Wall | Instruct to output all warnings when compile |
| -Wextra | Instruct to output even extra warnings when compile |
| -fno-rtti | This is as part of optimization. It disables RTTI (runtime type information).<br> RTTI allows the type of object to be determined during program execution in which we don't need as it will incur performance cost. |
| -O2 | Enable optimization level 2 which includes more aggressive optimizations on top of level 1 |
| -I. | Set the include path to be at the current directory, thus compiler will be able to find our header file _example.h_ which locates at the same directory. |
| -fPIC | Usually required whenever build a shared library as it instructs a compiler to generate position-independent code (PIC) suitable for a shared library to be made<br> and work with the main program to be linking with. With no fixed memory address to load particular function from shared library, it increases the security as well. |
| -lpthread | Instruct to link with pthread library |

Execute the following command to build a main testing program linking with _libexample.so_ and output as _main.out._

```
$ g++ -std=c++17 -Wall -Wextra -fno-rtti -O2 -I. -o main.out main.cpp -L. -lexample
```

Explanation for each flag as different from what we've mentioned above is as follows

| Flag | Description |
| --- | --- |
| -L. | Set the include path for shared library to be at the same directory |
| -lexample | Link with shared library namely _libexample.so_. |

Finally, we execute the executable file.

```
$ ./main.out
--- misc ---
add(1,2): 3
sub(2,1): 1
hardware concurrency: 12
--- end ---

--- single-threaded sum(1000M) ---
elapsed time: 568.401ms
sum: 1000000000
--- end ---

--- multi-threaded sum_v2(1000M) ---
elapsed time: 131.697ms
sum: 1000000000
--- end ---
```

Result as shown shows that multi-threaded function works faster significantly compared to single-threaded function (~4.33x faster).

As we are familiar with how to compile and build both shared library and main program with command line, we further expand on that to make a proper build system via Makefile.

Although there is CMake for this, but as we mainly develop and build on Linux, CMake to me seems to be overkill. We don't need such compatibility to be able to build on Windows. So Makefile is the right choice.

We will have three variants of Makefile.

1. **Makefile**

    It's for cross-compilation for both Linux and Windows. It uses pthread. We use this to build DLL working with MetaTrader 5, in additional to main testing program that can launched via W _ine._
2. **Makefile-th\_win32**

    Same as Makefile but it uses win32 thread.

3. **Makefile-g++**

    It's for compile on native Linux system. It's the steps we just did above.

**Makefile**

```
# script to build project with mingw with posix thread
.PHONY: all clean example.dll main.exe

COMPILER := x86_64-w64-mingw32-g++-posix
FLAGS := -O2 -fno-rtti -std=c++17 -Wall -Wextra
MORE_FLAGS ?=

all: example.dll main.exe

example.dll: example.cpp example.h
        $(COMPILER) -shared $(FLAGS) $(MORE_FLAGS) -DEXAMPLE_EXPORT -DWINDOWS -I. -fPIC -o $@ $< -lpthread

main.exe: main.cpp example.dll
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -DWINDOWS -o $@ $< -L. -lexample

clean:
        rm -f example.dll main.exe
```

**Makefile-th\_win32**

```
# script to build project with mingw with win32 thread
.PHONY: all clean example.dll main.exe

COMPILER := x86_64-w64-mingw32-g++-win32
FLAGS := -O2 -fno-rtti -std=c++17 -Wall -Wextra
MORE_FLAGS ?=

all: example.dll main.exe

example.dll: example.cpp example.h
        $(COMPILER) -shared $(FLAGS) $(MORE_FLAGS) -DEXAMPLE_EXPORT -DWINDOWS -DUSE_MINGW_STD_THREAD -I. -fPIC -o $@ $<

main.exe: main.cpp example.dll
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -DWINDOWS -DUSE_MINGW_STD_THREAD -o $@ $< -L. -lexample

clean:
        rm -f example.dll main.exe
```

**Makefile-g++**

```
# script to build project with mingw with posix thread, for native linux
.PHONY: all clean example.dll main.exe

COMPILER := g++
FLAGS := -O2 -fno-rtti -std=c++17 -Wall -Wextra
MORE_FLAGS ?=

all: libexample.so main.out

libexample.so: example.cpp example.h
        $(COMPILER) -shared $(FLAGS) $(MORE_FLAGS) -I. -fPIC -o $@ $< -lpthread

main.out: main.cpp libexample.so
        $(COMPILER) $(FLAGS) $(MORE_FLAGS) -I. -o $@ $< -L. -lexample

clean:
        rm -f libexample.so main.out
```

From 3 variants of Makefiles above, it shares almost all of the code, with some minimal differences.

Please take some time to see what are differences among them especially

- the name of compiler binary
- _-DUSE\_MINGW\_STD\_THREAD_
- with or without _-lpthread_
- output binary name e.g. _libexample.so_ or _example.dll_, and _main.out_ or _main.exe_; depending on target system to build for

_MORE\_FLAGS_ is declared as

_MORE\_FLAGS ?=_

which means it allows users to pass in additional compilation flags from command line, thus on-demand user can add more flags as needed. If there's no flags passing in externally from users, then it uses what is already defined in Makefile code.

Next make all those Makefile executable via

```
$ chmod 755 Makefile*
```

So to build for a specific variant of Makefile above, see the following table.

| Target System | Build command | Clean Command |
| --- | --- | --- |
| Cross-compilation using pthread | make | make clean |
| Cross-compilation using win32 thread | make -f Makefile-th\_win32 | make -f Makefile-th\_win32 clean |
| Native Linux | make -f Makefile-g++ | make -f Makefile-g++ clean |

Let's build DLL for use with MetaTrader 5, and Wine. So we can test both.

So do

```
$ make
```

We will have the following files generated

1. example.dll
2. main.exe

**Test executing cross-compiled executable file.**

```
$ wine main.exe
...
0118:err:module:import_dll Library libgcc_s_seh-1.dll (which is needed by L"Z:\\mnt\\datadrive\\_extended\\home\\haxpor\\Data\\Projects\\ExampleLib\\t\\main.exe") not found
0118:err:module:import_dll Library libstdc++-6.dll (which is needed by L"Z:\\mnt\\datadrive\\_extended\\home\\haxpor\\Data\\Projects\\ExampleLib\\t\\main.exe") not found
0118:err:module:import_dll Library libgcc_s_seh-1.dll (which is needed by L"Z:\\mnt\\datadrive\\_extended\\home\\haxpor\\Data\\Projects\\ExampleLib\\t\\example.dll") not found
0118:err:module:import_dll Library libstdc++-6.dll (which is needed by L"Z:\\mnt\\datadrive\\_extended\\home\\haxpor\\Data\\Projects\\ExampleLib\\t\\example.dll") not found
0118:err:module:import_dll Library example.dll (which is needed by L"Z:\\mnt\\datadrive\\_extended\\home\\haxpor\\Data\\Projects\\ExampleLib\\t\\main.exe") not found
0118:err:module:LdrInitializeThunk Importing dlls for L"Z:\\mnt\\datadrive\\_extended\\home\\haxpor\\Data\\Projects\\ExampleLib\\t\\main.exe" failed, status c0000135
```

Ok, we have a problem. _main.exe_ cannot find those required DLLs.

The solution is to bring them all to be at the same directory as of our executable file.

The required DLLs are as follows

- **libgcc\_s\_seh-1.dll**

Used to provide support for C++ exception handling and other low-level features that are not natively supported by Windows system.

- **libstdc++6.dll**

Foundation to serve C++ program. It contains functions and classes used to perform various operations such as input and output, mathematical operations, and memory management.

- **libwinpthread-1.dll**

It is the implementation of pthread API for Windows.

This DLL might not be shown on terminal output, but it's a DLL dependency from both previous two mentioned DLLs.

As we've installed Mingw, those DLLs are already there on our Linux system. We just need to find it.

Use the following technique to find them.

```
sudo find / -type f -name libgcc_s_seh-1.dll 2>/dev/null
```

Such command will locate _libgcc\_s\_seh-1.dll_ and ignore directories (use _-type f)_ by beginning the search from root directory (use _/_). If there is any error, then dump it to _/dev/null_(via _2>/dev/null_) _._

We will see the relevant output as of

- _/usr/lib/gcc/x86\_64-w64-mingw32/9.3- **win32**/_
- _/usr/lib/gcc/x86\_64-w64-mingw32/9.3- **posix**/_

Carefully pay attention to win32 and posix as part of the directory name. If you build via Makefile, then you should copy such DLL from posix-based directory. But if you build via Makefile-th\_win32, then copy DLLs from win32-based directory.

As we chose to base on pthread mainly, so I suggest the following

- Copy DLLs from posix-based directory into the same directory as of our project, the same as executable binary file
- From time to time, we might want to test with win32 thread, so we might create win32, and posix directory, then copy corresponding DLLs into each directory.

Whenever a need to test one or another, then copy the output built DLL and executable file into such newly created either win32 or posix directory, then launch the program from there via Wine. Or vise-versa.

Finally, we can test the program as follows

```
$ wine main.exe
0098:fixme:hid:handle_IRP_MN_QUERY_ID Unhandled type 00000005

        ...
0098:fixme:xinput:pdo_pnp IRP_MN_QUERY_ID type 5, not implemented!

        ...
--- misc ---
add(1,2): 3
sub(2,1): 1
hardware concurrency: 12
--- end ---

--- single-threaded sum(1000M) ---
elapsed time: 416.829ms
sum: 1000000000
--- end ---

--- multi-threaded sum_v2(1000M) ---
elapsed time: 121.164ms
sum: 1000000000
--- end ---
```

Ignore non-relevant output lines which are warning & minor errors from Wine itself.

We see that multi-threaded function is faster than single-threaded function by ~3.4x times. Still slightly slower than native Linux build which is understandable.

We will come back to a simple benchmark later again after we complete MQL5 code to consume it.

Awesome! We are ready to implement MQL5 code next.

### Proof-of-Concept, Development Phase II - MQL5 Code to Consume DLL

Such a long journey until we reach here in Phase II for MQL5 code development.

Implementation of _TestConsumeDLL.mq5_ as a Script.

```
//+------------------------------------------------------------------+
//|                                               TestConsumeDLL.mq5 |
//|                                          Copyright 2022, haxpor. |
//|                                                 https://wasin.io |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, haxpor."
#property link      "https://wasin.io"
#property version   "1.00"

#import "example.dll"
const int add(int, int);
const int sub(int, int);
const int num_hardware_concurrency();
const int single_threaded_sum(const int& arr[], int num_elem);
const int multi_threaded_sum_v2(const int& arr[], int num_elem);
#import

void OnStart()
{
   Print("add(1,2): ", example::add(1,2));
   Print("sub(2,1): ", example::sub(2,1));
   Print("Hardware concurrency: ", example::num_hardware_concurrency());

   int arr[];
   ArrayResize(arr, 1000000000);                // 1000M elements
   ArrayFill(arr, 0, ArraySize(arr), 1);

   // benchmark of execution time will be printed on terminal
   int sum = 0;
   Print("--- single_threaded_sum(1000M) ---");
   sum = single_threaded_sum(arr, ArraySize(arr));
   Print("sum: ", sum);
   if (sum != 1000000000) Alert("single_threaded_sum result not correct");
   Print("--- end ---");

   sum = 0;

   Print("--- multi_threaded_sum_v2(1000M) ---");
   sum = multi_threaded_sum_v2(arr, ArraySize(arr));
   Print("sum: ", sum);
   if (sum != 1000000000) Alert("multi_threaded_sum_v2 result not correct");
   Print("--- end ---");
}
```

Definitely you can create a MQL5 code as Experts, or Indicators, but for this proof-of-concept work, we proceed with hit and run to test all the steps and workflow. So _Script_ is suitable for our need.

Anyhow, in real world situation, you usually need Experts or Indicators in order to have ability to get data from terminal e.g. _OnTick(), OnTrade(), OnCalculate()._ More information for which function supported by each type of program on MT platform, please take a look at [Program Running](https://www.mql5.com/en/docs/runtime/running "Official MQL5 document for Program Running").

Now let's dissect chunk by chunk of the full code above.

Import function signatures from DLL.

```
#import "example.dll"
const int add(int, int);
const int sub(int, int);
const int num_hardware_concurrency();
const int single_threaded_sum(const int& arr[], int num_elem);
const int multi_threaded_sum_v2(const int& arr[], int num_elem);
#import
```

To be able to call functions exposed from DLL, we need to declare those signatures again in MQL5 code.

Things to notice

- We can skip function parameter naming e.g. _add(int, int),_ and _sub(int, int)._
- Array will be passed by reference only in MQL5. Notice difference of signatures as declared in DLL code, and MQL5. In MQL5 code, there is & (ampersand character) but not in DLL code.

Please note that C++ syntax as used in MQL5 and actual C++ itself is not 100% the same. In short, whenever we pass an array in MQL5, we need to add &.

Create an array of large data set

```
   int arr[];
   ArrayResize(arr, 1000000000);                // 1000M elements
   ArrayFill(arr, 0, ArraySize(arr), 1);
```

This will create array of integer for 1000M elements and set each element with value 1. With this, array is dynamic and lives on heap. Stack won't have enough space to hold such huge amount of data size.

So to make an array to be dynamic, use declaration syntax of _int arr\[\]_.

After that, we just call each DLL function from the declared signatures as needed. Notice also that we validate the output by checking the result and if not correct then _Alert()_ to users. Although we don't exit right away.

Use _ArraySize()_ to get the number of elements of the array. To pass the array to the function, just pass its variable to the function directly.

Compile the script, and we're done for the implementation.

**Copy all required DLLs to MetaTrader 5**

Before we try to launch MQL5 script. We need to copy all required DLLs to _<terminal>/Libraries_ directory. Usually its full path is _~/.mt5/drive\_c/Program Files/MetaTrader 5/MQL5/Libraries._

That's where MetaTrader 5 will be seeking for any required DLLs as needed by the programs we built for MetaTrader 5. Head back to Test executing cross-compiled executable file section to see the list of DLL to copy to.

By default, MetaTrader 5 official installation script will install Wine with prefix of _~/.mt5_ automatically. This only applies for users who use official installation script.

**Testing**

![Drag and Drop compiled TestConsumeDLL onto the chart](https://c.mql5.com/2/51/TestConsumeDLL_DragAndDropScript.png)

Drag and drop compiled TestConsumeDLL onto the chart to begin executing.

Test on MetaTrader 5 launching via Wine on Linux first.

Drag and drop the compiled _TestConsumeDLL_ into the chart. Then you will see it shows a dialog asking for permission to allow importing from DLL, along with listing of DLL dependencies for such MQL5 program we've built.

![Dialog asking for DLL import permission, along with listing of DLL dependencies.](https://c.mql5.com/2/51/TestConsumeDLL_AllowDLLImport_DLLDependencies.png)

Dialog asking for DLL import permission, along with listing of DLL dependencies.

Even though we didn't see _libwinpthread-1.dll_ because it's not an immediate dependency of compiled MQL5 script, but it's a dependency for both _libgcc\_s\_seh-1.dll_, and _libstdc++6.dll._ We can check DLL dependency of target DLL file with _objdump_ as follows.

```
$ objdump -x libstdc++-6.dll  | grep DLL
        DLL
 vma:            Hint    Time      Forward  DLL       First
        DLL Name: libgcc_s_seh-1.dll
        DLL Name: KERNEL32.dll
        DLL Name: msvcrt.dll
        DLL Name: libwinpthread-1.dll

$ objdump -x libgcc_s_seh-1.dll  | grep DLL
        DLL
 vma:            Hint    Time      Forward  DLL       First
        DLL Name: KERNEL32.dll
        DLL Name: msvcrt.dll
        DLL Name: libwinpthread-1.dll
```

_objdump_ is able to read binary file (shared library, or executable file) created from Windows and Linux. It's versatile to dump out information available as needed. _-x_ flag means to display contents of all headers.

See the result output at Experts tab

![Output as seen in Experts tab from executing TestConsumeDLL](https://c.mql5.com/2/51/TestConsumeDLL_outputFromExpertsTab.png)

Output as seen in Experts tab from executing TestConsumeDLL.

as well as result of elapsed time in execution for each function at the original terminal window which used to launch MetaTrader 5 on Linux.

![Elapsed time in execution output in console for each function](https://c.mql5.com/2/51/TestConsumeDLL_elapsedTimeExecution.png)

On the same terminal window which used to launch MetaTrader 5, users will see elapsed time in execution output from DLL.

As long as you didn't see any alerts from _Alerts(),_ and elapsed time in execution is considerably proper. Then all is fine and we almost finish all the things in this proof-of-concept program.

### Testing on Windows system

We needs the following

- **Virtualbox with Guest Additions installed**

Please kindly do research on Internet on how to install it on your Linux system. Other multiple sources already provided comprehensive information on how to do it better than I exhaustively include such information here thus make the article too lengthy unnecessary.

Important note you need _Guest Additions_ in order to use features of sharing data between host and guest machine, so you can copy _example.dll_ along with bunch other DLLs to guest machine (Windows machine).

- **Windows 7+ 64 bit ISO image**

This is to be loaded and installed into harddrive via Virtualbox.

![Virtualbox Main Interface](https://c.mql5.com/2/51/VirtualBox_testing.png)

Virtualbox Main Interface. Depends on your availability of hardware resource to spare for it, more is better if you need to test speed of execution on DLL.

Also depends on how generous and availability of your machine resource that can be spared onto Windows system launching via Virtualbox for the case of testing speed of execution from DLL. In my case, I have the following configurations

- System -> Motherboard -> Base Memory set to 20480 MB or 20 GB (I have 32 GB on host machine)
- System -> Processor -> Processor(s) set to 6 along with Execution Cap set to 100% (cannot fully set to 12, as 6 is the maximum of valid value here)
- Display -> Screen -> Video Memory set to maximum (not that necessary, but it will in case you would like to utilize all monitors. More monitors, more video memory in need)
- Display -> Screen -> Monitor Count set to 1

Now it's time to test. We can either copy the compiled MQL5 code from Linux machine or just copy all the code, then use MetaEditor to compile it on Windows machine again.

I found that the latter option is totally fine, and it's just another copy and paste away. So I did just that.

Test result

![TestConsumeDLL Output Result on Experts Tab on Windows](https://c.mql5.com/2/51/TestConsumeDLL_windows_test_output.png)

Result output shown on Experts tab as tested on Windows.

Problem is that elapsed time in execution is coded to output via standard output (stdout) and I cannot find a way to capture such output from launching MetaTrader 5 on Windows. One way I tried is launching MetaTrader 5 with configuration file to execute a script from start then redirect the output to file, but attempt failed as MetaTrader 5 doesn't allow to load any DLL at startup from command line. So to amend for this without interfere to main DLL code, we will make a minor adjustment to MQL5 code to compute elapsed execution time from there by using _GetTickCount()_.

```
   ...
   int sum = 0;
   uint start_time = 0;
   uint elapsed_time = 0;

   Print("--- single_threaded_sum(1000M) ---");
   start_time = GetTickCount();                         // *
   sum = single_threaded_sum(arr, ArraySize(arr));
   elapsed_time = GetTickCount() - start_time;          // *
   Print("sum: ", sum);
   if (sum != 1000000000) Alert("single_threaded_sum result not correct");
   Print("elapsed time: ", elapsed_time, " ms");
   Print("--- end ---");

   sum = 0;
   start_time = 0;
   elapsed_time = 0;

   Print("--- multi_threaded_sum_v2(1000M) ---");
   start_time = GetTickCount();                         // *
   sum = multi_threaded_sum_v2(arr, ArraySize(arr));
   elapsed_time = GetTickCount() - start_time;          // *
   Print("sum: ", sum);
   if (sum != 1000000000) Alert("multi_threaded_sum_v2 result not correct");
   Print("elapsed time: ", elapsed_time, " ms");
   Print("--- end ---");
}
```

Notice the line with a comment saying "// \*". That's the main additional lines to pay attention to. It's simple enough to understand.

Now, let's re-test.

![TestConsumeDLL Windows Retest Result as shown on Experts Tab](https://c.mql5.com/2/51/TestConsumeDLL_updated_MQL5_code_benchmark.png)

Updated MQL5 code to measure elapsed execution time as tested on Windows.

We have completed the whole proof-of-concept application in making multi-threading capable DLL, then consume it in MQL5 code, tested on both Linux and Windows system, all started and developed on Linux. All works as expect with expected result.

### Simple benchmark of both Mingw threading implementations

We will conduct benchmark in a simple way specifically based on our proof-of-concept program. Because in order to conduct a full benchmark on C++ multi-threading capability across platform, there are multiple factors to considers especially multiple synchronization primitives, _thread\_local_, problem domain, etc.

How we conduct the benchmark is as follows

- **Linux**

  - Build via _Makefile_ then conduct the result 5 times before averaging, and do the same for _Makefile-th\_win32_
  - Execute the binary file with _WINEPREFIX=~/.mt5 wine main.exe_
  - Use full 12 threads, and all available RAM as of 32 GB

- **Windows**

  - Build via _Makefile_ then conduct the result 5 times before avergaing, and do the same for _Makefile-th\_win32_
  - Copy necessary DLLs and executable files into guest machine (Windows) through Virtualbox
  - Execute the binary file using command prompt of _main.exe_
  - Capped at 6 threads, and with 20 GB of RAM (both due to adhering to valid configurations on Virtualbox)

Result numbers will be rounded for 2 decimal places.

Result as shown in the following table.

| Function | Linux + pthread (ms) | Linux + win32 thread (ms) | Windows + pthread (ms) | Windows + win32 thread (ms) |
| --- | --- | --- | --- | --- |
| single\_threaded\_sum | 417.53 | 417.20 | 467.77 | 475.00 |
| multi\_threaded\_sum\_v2 | 120.91 | 122.51 | 121.98 | 125.00 |

### Conclusion

Mingw and Wine are the cross-platform tools that allows developers to use Linux to develop cross-platform application which works seamlessly on either Linux, and Windows. It also applies to the case of developing for MT platform as well. With our proof-of-concept application to develop DLL capable of C++ multi-threading, tested on both Linux and Windows, it offers alternative options to expand the reach from developers into the ecosystem.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12042.zip "Download all attachments in the single ZIP archive")

[ExampleLib.zip](https://www.mql5.com/en/articles/download/12042/examplelib.zip "Download ExampleLib.zip")(5.37 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Understand and Efficiently use OpenCL API by Recreating built-in support as DLL on Linux (Part 2): OpenCL Simple DLL implementation](https://www.mql5.com/en/articles/12387)
- [Understand and efficiently use OpenCL API by recreating built-in support as DLL on Linux (Part 1): Motivation and validation](https://www.mql5.com/en/articles/12108)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/440596)**
(5)


![BeeXXI Corporation](https://c.mql5.com/avatar/2024/9/66dbee89-a47e.png)

**[Nikolai Semko](https://www.mql5.com/en/users/nikolay7ko)**
\|
13 Mar 2023 at 09:11

**MetaQuotes:**

The article [Development of an experimental DLL with multithreading support in C++ for MetaTrader 5 on Linux has](https://www.mql5.com/en/articles/12042) been published:

Author: [Wasin Thonkaew](https://www.mql5.com/en/users/haxpor "haxpor")

Thanks to the author for the interesting material!

It would be interesting to learn and read more about experiments on MT5 with Docker.

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
13 Mar 2023 at 09:30

now I have only one question: "why my series of similar (about C/C++/mingw) articles were rejected with the wording **do not correspond to the ideology of the company**".

![Aleksey Nikolayev](https://c.mql5.com/avatar/2018/8/5B813025-B4F2.jpeg)

**[Aleksey Nikolayev](https://www.mql5.com/en/users/alexeynikolaev2)**
\|
13 Mar 2023 at 10:00

It would be interesting something like this in the context of creating fast custom C/C++ functions to process python arrays and tables (from numpy and pandas.) You can even do it without linux)


![Wasin Thonkaew](https://c.mql5.com/avatar/2023/1/63c80533-804f.jpg)

**[Wasin Thonkaew](https://www.mql5.com/en/users/haxpor)**
\|
13 Mar 2023 at 10:43

**Nikolai Semko [#](https://www.mql5.com/ru/forum/443470#comment_45561206):**

Thanks to the author for the interesting material!

It would be interesting to learn and read more about MT5 experiments with Docker.

Thank you for your kind words. Sorry I didn't speak Russian.


![Wasin Thonkaew](https://c.mql5.com/avatar/2023/1/63c80533-804f.jpg)

**[Wasin Thonkaew](https://www.mql5.com/en/users/haxpor)**
\|
13 Mar 2023 at 10:46

**Aleksey Nikolayev [#](https://www.mql5.com/ru/forum/443470#comment_45561399):**

It would be interesting something like this in the context of creating fast custom C/C++ functions to process python arrays and tables (from numpy and pandas.) You can even do it without linux)

Yes, exactly no need to be only Linux. It is just that I base on it solely, offer in perspective of cross platform development on platform of your choice.

Ideally, it would be best to use compiler native to each platform. I might write something about it using CMake build system.

Thanks for your comment!

![Category Theory in MQL5 (Part 2)](https://c.mql5.com/2/51/Category-Theory-avatar-002.png)[Category Theory in MQL5 (Part 2)](https://www.mql5.com/en/articles/11958)

Category Theory is a diverse and expanding branch of Mathematics which as of yet is relatively uncovered in the MQL5 community. These series of articles look to introduce and examine some of its concepts with the overall goal of establishing an open library that attracts comments and discussion while hopefully furthering the use of this remarkable field in Traders' strategy development.

![MQL5 Cookbook — Services](https://c.mql5.com/2/50/mql5-recipes-Services.png)[MQL5 Cookbook — Services](https://www.mql5.com/en/articles/11826)

The article describes the versatile capabilities of services — MQL5 programs that do not require binding graphs. I will also highlight the differences of services from other MQL5 programs and emphasize the nuances of the developer's work with services. As examples, the reader is offered various tasks covering a wide range of functionality that can be implemented as a service.

![DoEasy. Controls (Part 29): ScrollBar auxiliary control](https://c.mql5.com/2/50/MQL5-avatar-doeasy-library-2__6.png)[DoEasy. Controls (Part 29): ScrollBar auxiliary control](https://www.mql5.com/en/articles/11847)

In this article, I will start developing the ScrollBar auxiliary control element and its derivative objects — vertical and horizontal scrollbars. A scrollbar is used to scroll the content of the form if it goes beyond the container. Scrollbars are usually located at the bottom and to the right of the form. The horizontal one at the bottom scrolls content left and right, while the vertical one scrolls up and down.

![Population optimization algorithms: Cuckoo Optimization Algorithm (COA)](https://c.mql5.com/2/50/Cuckoo-Optimization-Algorithm-avatar.png)[Population optimization algorithms: Cuckoo Optimization Algorithm (COA)](https://www.mql5.com/en/articles/11786)

The next algorithm I will consider is cuckoo search optimization using Levy flights. This is one of the latest optimization algorithms and a new leader in the leaderboard.

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/12042&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068635178605149131)

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