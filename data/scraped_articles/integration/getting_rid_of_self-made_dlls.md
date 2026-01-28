---
title: Getting Rid of Self-Made DLLs
url: https://www.mql5.com/en/articles/364
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:07:48.317059
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/364&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083354439550507522)

MetaTrader 5 / Examples


_Do you still write your DLLs?_

_Then we go to you!_

### Introduction

There always comes a moment when MQL5 language functional is not enough for fulfilling tasks. In that case an MQL5 programmer has to use additional tools. For example, it is possible to work with a database, use communication sockets or utilize operation system functions. An MQL5 programmer also has to deal with various APIs to expand the possibilities of the MQL5 program he/she uses. But for several reasons, the programmer cannot access the required functions directly from MQL5, as he/she does not know the following things:

- How to transfer a complex data type (for example, structure) to API function;
- How to work with the pointer that is returned by the API function.

Therefore, the programmer is forced to use a different programming language and to create an intermediate DLL to work with the required functionality. Although MQL5 has the possibility to present various data types and transfer them to API,
unfortunately, MQL5 cannot solve the issue concerning data extraction from the accepted pointer.

In this article we will dot all the "i"s and show simple mechanisms of transferring and receiving complex data types and working with return indices.

**Contents**

[1\. Memory is Everything](https://www.mql5.com/en/articles/364#1)

- Getting the Indices
- Copying memory areas

[2\. Transferring the Structures to API Functions](https://www.mql5.com/en/articles/364#2)

- Transforming the structures using MQL5
- Example of transferring the structure for sockets

[3\. Working with API Functions Pointers](https://www.mql5.com/en/articles/364#3)

- Examples for Memory Mapping File,
- Example for MySQL

[4\. Reading NULL-Terminated Strings from API Functions](https://www.mql5.com/en/articles/364#4)

### 1\. Memory Is Everything

As you may know, any variable (including complex data types variables) has its specific address, from which that variable is stored in memory. This address is a four-byte integer value (of int type) equal to the address of the first byte of this variable.

And if all is well defined, it is possible to work with this memory area. C language library (msvcrt.dll)
contains memcpy function. Its purpose is the missing element, which binds MQL5 and various API libraries and provides great possibilities
for a programmer.

**Let's Turn to the Knowledge of Our Ancestors**

[Memcpy](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/dswaw1wk.aspx "http://msdn.microsoft.com/en-us/library/dswaw1wk.aspx") function copies the specified number of bytes from one buffer to another and returns the pointer to a receiver buffer.

```
void *memcpy(void *dst, const void *src, int cnt);
dst - pointer to the receiver buffer
src - pointer to the source buffer
cnt - number of bytes for copying
```

In other words, a memory area with a size of _cnt_ bytes beginning from _src_ address is copied to the memory area beginning from _dst_ address.

The data located at _src_ address can be of various types. This may be _char_ one byte variable, _double_ eight byte number, array, any structure and any memory volume. It means that you can freely transmit data from one area to another, if you know addresses and a size.

**How Does It Work**

Diagram 1 shows the comparative sizes of some data [types](https://www.mql5.com/en/docs/basis/types).

![Sizes of various data types in MQL5](https://c.mql5.com/2/4/11__1.png)

_Memcpy_ function is needed to copy the data from one memory area to another.

Figure 2 shows the copying of four bytes.

![Example of copying 4 bytes with the use of memcpy](https://c.mql5.com/2/4/22.png)

In MQL5 that will look as follows.

```
Example 1. Using memcpy
#import "msvcrt.dll"
  int memcpy(int &dst, int &src, int cnt);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  int dst, src=4, cnt=sizeof(int);
  int adr=memcpy(dst, src, cnt);
  Print("dst value="+string(dst)+"   Address dst="+string(adr));
}
```

It should be noted that various data types (of the same _cnt_ size) can be used as memory areas _dst_ and _src_ point at. For example, _src_ pointer can refer to _double_ variable ( _cnt_ =8 bytes) and _dst_ can refer to the array having the equivalent size _char\[8\]_ or _int\[2\]_.

It does not matter for the memory, what idea a programmer has about it at the moment. It does not matter, whether it is an array _char\[8\]_ or just one _long_ variable or structure _{ int a1; int a2; }_.

Memory data can be considered as data of various types. For example, it is possible to transfer five byte array to _{int i; char c;}_ structure or vice versa. This relationship provides an opportunity to work directly with API functions.

Let's examine _memcpy_ application versions in the definite order.

**Getting the Indices**

In example 1 we showed that _memcpy_ function returns _dst_ variable address.

This property can be used to get an address of any variable (including the arrays of other complex types).
To do this, we need only to specify the same variable as a source and receiver parameters. In _cnt_ it is possible to transfer 0, as the actual copying is not necessary.

For example, we may get the address of _double_ variable and _short_ array:

```
Example 2. Getting pointers to the variable
#import "msvcrt.dll"
  int memcpy(short &dst[], short &src[], int cnt);
  int memcpy(double &dst,  double &src, int cnt);
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  short src[5];
  //--- getting src array address (i.е., the address of the first element)
  int adr=memcpy(src, src, 0);
  double var;
  //--- getting var variable address
  adr=memcpy(var, var, 0);
}
```

Received address then can be transferred to the required API function or
as a structure parameter and also as a parameter of the same _memcpy_ function.

**Copying the Arrays**

As you know, an array is some dedicated memory chunk. The size of dedicated memory depends on the elements type and their amount. For example, if the _short_ array elements type and the number of the elements is 10, such an array occupies 20 bytes in memory (as _short_ size is 2 bytes).

But these 20 bytes are also shown as arrays consisting of 20 _char_ or 5 _int_. In any case, they occupy the same 20 bytes in memory.

To copy the arrays, it is necessary to do the following things:

- Allocate the required amount of the elements (not less than resulting _cnt_ bytes) for _dst_ memory;
- Specify the number of bytes in _cnt_ that must be copied from _src._

```
Example 3. Copying the arrays
#import "msvcrt.dll"
  int memcpy(double &dst[],  double &src[], int cnt);
#import

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  double src[5];
  //--- calculating the number of bytes!!!
  int cnt=sizeof(double)*ArraySize(src);
  double dst[];
  ArrayResize(dst, 5);
  //--- the array has been copied from src to dst
   memcpy(dst, src, cnt);
}
```

### 2\. Transferring the Structures to API Functions

Suppose, you need to transfer the filled structure pointer to API. MQL5 language sets limitations for
transmitting the structures. At the beginning of the article I declared that the memory can be presented differently. That means that the
required structure can be copied to the data type supported by MQL5. In general, an array is a type that is suitable for the structures.
Therefore, we will have to get an array from a structure and then transfer an array to the API function.

The option of copying the memory using the structures is described in the [documentation](https://www.mql5.com/en/docs/basis/types/casting#casting_structure) section. We cannot use memcpy function, as it is impossible to transfer
the structures as parameters and copying the structures is the only way here.

Figure 3 shows the representation of the structure consisting of 5 variables of different types and its equivalent presented as _char_ array.

![Presenting the structure consisting of 5 variables of different types and its equivalent presented as char array](https://c.mql5.com/2/4/11.jpg)

```
Example 4. Copying the structures by means of MQL5
struct str1
{
  double d; // 8 bytes
  long l;   // 8 bytes
  int i[3]; // 3*4=12 bytes
};
struct str2
{
  uchar c[8+8+12]; // str1 structure size
};
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  str1 src;
  src.d=-1;
  src.l=20;
  //--- filling the structure parameters
  ArrayInitialize(src.i, 0);
  str2 dst;
  //--- turning the structure into the byte array
  dst=src;
}
```

In such simple way we have copied the structure into the byte array.

Let's consider the [socket](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/ms737625(v=vs.85).aspx "http://msdn.microsoft.com/en-us/library/ms737625(v=vs.85).aspx") creation function to make this example more practical.

```
int connect(SOCKET s, const struct sockaddr *name, int namelen);
```

In this function the second parameter is problematic, as it accepts the pointer for the structure. But we already know what to do with that. So, let's begin.

1\. Let's write connect function for import by the method permissible in MQL5:

```
int connect(int s, uchar &name[], int namelen);
```

2\. Let's observe the required structure in the documentation:

```
struct sockaddr_in
{
  short   sin_family;
  u_short sin_port;
  in_addr sin_addr; // additional 8 byte structure
  char sin_zero[8];
};
```

3\. Creating a structure with an array of similar size:

```
struct ref_sockaddr_in
{
  uchar c[2+2+8+8];
};
```

4\. After filling out the required _sockaddr\_in_ structure, we transfer it to the byte array and submit as _connect_ parameter.

Below is the code section made according to these steps.

```
Example 5. Referring of the client socket to the server
#import "Ws2_32.dll"
  ushort htons(ushort hostshort);
  ulong inet_addr(char &cp[]);
  int connect(int s, char &name[], int namelen);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  //--- connecting the host after the socket initialization

  char ch[];
  StringToCharArray("127.0.0.1", ch);
  //--- preparing the structure
  sockaddr_in addrin;
  addrin.sin_family=AF_INET;
  addrin.sin_addr=inet_addr(ch);
  addrin.sin_port=htons(1000);
  //--- copying the structure to the array
  ref_sockaddr_in ref=addrin;
  //--- connecting the host
  res=connect(asock, ref.c, sizeof(addrin));

  //--- further work with the socket
}
```

As you can see, you do not need to make your DLL at all. The structures are transferred to API directly.

### 3\. Working with API Functions Pointers

In most cases API functions return a pointer to the data: structures and arrays. MQL5 is not suitable for extracting the data, memcpy function can be used here.

**Example of working with memory arrays from Memory Mapping File (MMF)**

When working with MMF the function is used, which returns a pointer to a dedicated memory array.

```
int MapViewOfFile(int hFile, int DesiredAccess, int OffsetHigh, int OffsetLow, int NumOfBytesToMap);
```

**Data** **reading** from this array is executed by simple copying of the required amount of bytes by _memcpy_ function.

**Writing** the data into the array is performed by the same use of _memcpy._

```
Example 6. Recording and reading data from MMF memory
#import "kernel32.dll"
  int OpenFileMappingW(int dwDesiredAccess, int bInheritHandle,  string lpName);
  int MapViewOfFile(int hFileMappingObject, int dwDesiredAccess,
                      int dwFileOffsetHigh, int dwFileOffsetLow, int dwNumberOfBytesToMap);
  int UnmapViewOfFile(int lpBaseAddress);
  int CloseHandle(int hObject);
#import "msvcrt.dll"
  int memcpy(uchar &Destination[], int Source, int Length);
  int memcpy(int Destination, int &Source, int Length);
  int memcpy(int Destination, uchar &Source[], int Length);
#import

#define FILE_MAP_ALL_ACCESS   0x000F001F

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  //--- opening the memory object
  int hmem=OpenFileMappingW(FILE_MAP_ALL_ACCESS, 0, "Local\\file");
  //--- getting pointer to the memory
  int view=MapViewOfFile(hmem, FILE_MAP_ALL_ACCESS, 0, 0, 0);
  //--- reading the first 10 bytes from the memory
  uchar src[10];
  memcpy(src, view, 10);
  int num=10;
  //--- recording the 4 byte int number to the memory beginning
  memcpy(view, num, 4);
  //--- unmapping the view
  UnmapViewOfFile(view);
  //--- closing the object
  CloseHandle(hmem);
}
```

As you can see, it is not so difficult to work with pointers for the memory array. And most importantly, you do not need to create your additional DLL for that.

**Example of working with returned structures for MySQL**

One of the urgent problems when working with MySQL has been getting data from it. _mysql\_fetch\_row_ function returns the strings array. Each string is a fields array. So, this function returns the pointer to the pointer. Our task is to extract all these data from the returned pointer.

The task is a bit complicated by the fact that the fields are various data types including binary ones. It
means that it will be impossible to present them as _string_ array. The functions _mysql\_num\_rows, mysql\_num\_fields, mysql\_fetch\_lengths_ are used for getting the information about the strings and field sizes.

Figure 4 shows the structure of presenting the result in memory.

The addresses of the beginning of three strings are gathered into the array. And the address of the array beginning (in the example = 94) is what _mysql\_fetch\_row_ function will return.

![The structure of presenting the request result in memory](https://c.mql5.com/2/4/22__2.jpg)

Below is the example of the code for getting data from a database request.

```
Example 7. Getting data from MySQL
#import "libmysql.dll"
  int mysql_real_query(int mysql, uchar &query[], int length);
  int mysql_store_result(int mysql);
  int mysql_field_count(int mysql);
  uint mysql_num_rows(int result);
  int mysql_num_fields(int result);
  int mysql_fetch_lengths(int result);
  int mysql_fetch_row(int result);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  //--- ... preliminarily initialized mysql data base
  //--- request for getting all the strings from the table
  string query="SELECT * FROM table";
  uchar aquery[];
  StringToCharArray(query, aquery);

  //--- sending the request
  err=mysql_real_query(mysql, aquery, StringLen(query));
  int result=mysql_store_result(mysql);

  //--- in case it contains the strings
  if (result>0)
  {
    ulong num_rows=mysql_num_rows(result);
    int num_fields=mysql_num_fields(result);

    //--- getting the first string pointer
    int r=0, row_ptr=mysql_fetch_row(result);
    while(row_ptr>0)
    {

       //--- getting the pointer to the current string columns lengths
      int len_ptr=mysql_fetch_lengths(result);
      int lens[];
       ArrayResize(lens, num_fields);
      //--- getting the sizes of the string fields
      memcpy(lens, len_ptr, num_fields*sizeof(int));
      //--- getting the data fields
      int field_ptr[];
      ArrayResize(field_ptr, num_fields);
      ArrayInitialize(field_ptr, 0);

      //--- getting the pointers to the fields
      memcpy(field_ptr, row_ptr, num_fields*sizeof(int));
      for (int f=0; f<num_fields; f++)
      {
        ArrayResize(byte, lens[f]);
        ArrayInitialize(byte, 0);
         //--- copy the field to the byte array
        if (field_ptr[f]>0 && lens[f]>0) memcpy(byte, field_ptr[f], lens[f]);
      }
      r++;
      //--- getting the pointer to the pointer to the next string
      row_ptr=mysql_fetch_row(result);
    }
  }
}
```

### 4\. Reading NULL-Terminated Strings from API Functions

Some API functions return the pointer to the string but do not show us the length of this string. In this situation we deal with strings that end in zero. This zero helps to determine the end of the string. This means that its size can be specified.

![Presenting the NULL-terminated string in memory](https://c.mql5.com/2/4/33__2.jpg)

C (msvcrt.dll) library already has the function that copies the contents of the NULL-terminated string from the appropriate pointer to another string. The size of the string is defined by the function. It is better to use a byte array as a receiver, as APIs often return multibyte strings instead of Unicode.

[strcpy](https://www.mql5.com/go?link=https://msdn.microsoft.com/en-us/library/kk6xf663.aspx "http://msdn.microsoft.com/en-us/library/kk6xf663.aspx") \- copies NULL-terminated strings

```
char *strcpy(char *dst, const char *src);
dst - the pointer to the destination string
src - the pointer to the Null-terminated source string
```

In fact, it is a special case of _memcpy_ function. The system stops the copying on the found zero in a string. This function will always be used when working with such pointers.

For example, there are several functions in API from MySQL that return the pointers to strings. And getting data from them using _strcpy_ is a trivial task.

```
Example 8. Getting the strings from the pointers
#import "libmysql.dll"
  int mysql_init(int mysql);
  int mysql_real_connect(int mysql, uchar &host[], uchar &user[], uchar &password[],
                            uchar &DB[], uint port, uchar &socket[], int clientflag);
  int mysql_get_client_info();
  int mysql_get_host_info(int mysql);
  int mysql_get_server_info(int mysql);
  int mysql_character_set_name(int mysql);
  int mysql_stat(int mysql);
#import "msvcrt.dll"
  int strcpy(uchar &dst[], int src);
#import
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
{
  uchar byte[];
  ArrayResize(byte, 300);

  int ptr;
  string st;
  //--- pointer to the string
  ptr=mysql_get_client_info();

  if (ptr>0) strcpy(byte, ptr);
  Print("client_info="+CharArrayToString(byte));
  //--- initializing the base
  int mysql=mysql_init(mysql);

  //--- transferring the strings to the byte arrays
  uchar ahost[];
  StringToCharArray("localhost", ahost);
  uchar auser[];
  StringToCharArray("root", auser);
  uchar apwd[];
  StringToCharArray("", apwd);
  uchar adb[];
  StringToCharArray("some_db", adb);
  uchar asocket[];
  StringToCharArray("", asocket);
  //--- connecting the base
  int rez=mysql_real_connect(mysql, ahost, auser, apwd, adb, port, asocket, 0);
  //--- determining the connection and the base status
  ptr=mysql_get_host_info(mysql);
  if (ptr>0) strcpy(byte, ptr);
  Print("mysql_host_info="+CharArrayToString(byte));
  ptr=mysql_get_server_info(mysql);
  if (ptr>0) strcpy(byte, ptr);
  Print("mysql_server_info="+CharArrayToString(byte));
  ptr=mysql_character_set_name(mysql);
  if (ptr>0) strcpy(byte, ptr);
  Print("mysql_character_set_name="+CharArrayToString(byte));
  ptr=mysql_stat(mysql);
  if (ptr>0) strcpy(byte, ptr);
  Print("mysql_stat="+CharArrayToString(byte));
}
```

### Conclusion

Thus, the use of three basic mechanisms of working with memory (copying the **structures**, getting pointers and their data on **memcpy** and getting **strcpy** strings) covers virtually all tasks when working with various API functions.

Warning. It may be unsafe to work with memcpy and strcpy, unless a sufficient amount of data has been allocated for the receiver buffer. Therefore, be careful about the size of the amounts allocated for receiving data.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/364](https://www.mql5.com/ru/articles/364)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Working with sockets in MQL, or How to become a signal provider](https://www.mql5.com/en/articles/2599)
- [SQL and MQL5: Working with SQLite Database](https://www.mql5.com/en/articles/862)
- [Promote Your Development Projects Using EX5 Libraries](https://www.mql5.com/en/articles/362)
- [Using WinInet in MQL5. Part 2: POST Requests and Files](https://www.mql5.com/en/articles/276)
- [Tracing, Debugging and Structural Analysis of Source Code](https://www.mql5.com/en/articles/272)
- [The Prototype of a Trading Robot](https://www.mql5.com/en/articles/132)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/6594)**
(36)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
11 Dec 2025 at 16:22

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/6049/page3#comment_58705380):**

of course it does...prototypes are from 4 (32 bit address a la unsigned int), and you compile/run in 5 (it has 64).

Retarded, I [wrote](https://www.mql5.com/en/code/viewcode/61283/337275/FileMap.mqh) via longs [earlier](https://www.mql5.com/en/code/viewcode/61283/337275/FileMap.mqh) myself.

```
#define  MEMCPY_MACROS(A)                                                     \
long memcpy( const long Destination, const A &Source[], const uint Length ); \
long memcpy( A &Destination[], const long Source, const uint Length );
```

![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
11 Dec 2025 at 16:55

**Edgar Akhmadeev [#](https://www.mql5.com/ru/forum/6049/page3#comment_58705972):**

I missed the point with the 64bit address. But I'm still crashing with the corrected address. Is it sure it should work? Can I see a full example of the corrected fxsaber code?

So far I am still of my own opinion - the address from WinAPI is incompatible with MQL.

I found a mistake, why the corrected version did not work for me. I made a typo, missed & in one place.

My opinion has changed, thank you all.

![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
11 Dec 2025 at 18:11

By the way, MS considers the [memcpy](https://www.mql5.com/en/articles/364 "Article: Getting rid of the ballast of homemade DLLs ") function unsafe and obsolete and suggests memcpy\_s instead. Besides, the first parameter is not const. So the result is:

```
#define  DEF_MEMCPY_S(T)                                                   \
        ulong memcpy_s(T &dst,   ulong size, const ulong src, ulong cnt); \
        ulong memcpy_s(T &dst[], ulong size, const ulong src, ulong cnt); \
        ulong memcpy_s(T &dst,   ulong size, const T &src[],  ulong cnt); \
        ulong memcpy_s(T &dst[], ulong size, const T &src[],  ulong cnt);

#import "msvcrt.dll"
        DEF_MEMCPY_S(char)
        DEF_MEMCPY_S(uchar)
        DEF_MEMCPY_S(int)
        DEF_MEMCPY_S(uint)
        DEF_MEMCPY_S(long)
        DEF_MEMCPY_S(ulong)
#import

void OnStart() {
        int Array[];
        ArrayResize(Array, 1);
        Array[0] = 123;
        int Value1 = 0;
        int Value2 = 0;

        ulong Address = memcpy(Array, 0, 0);
        memcpy_s(Value1, sizeof(int), Address, sizeof(int));

        memcpy_s(Value2, sizeof(int), Array,   sizeof(int));

        Print(Value1, " ", Value2);
}
```

![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
11 Dec 2025 at 23:39

Interesting examples. A question for connoisseurs. Is it possible to get the address of a [vector](https://www.mql5.com/en/docs/matrix/matrix_types " MQL5 Documentation: Matrix and vector types"), matrix, object of any class?

![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
12 Dec 2025 at 04:23

**Denis Kirichenko [#](https://www.mql5.com/ru/forum/6049/page4#comment_58709493):**

Interesting examples. Question for connoisseurs. Is it possible to get the address of a vector, matrix, object of any class?

no

![Who Is Who in MQL5.community?](https://c.mql5.com/2/0/whoiswho.png)[Who Is Who in MQL5.community?](https://www.mql5.com/en/articles/386)

The MQL5.com website remembers all of you quite well! How many of your threads are epic, how popular your articles are and how often your programs in the Code Base are downloaded – this is only a small part of what is remembered at MQL5.com. Your achievements are available in your profile, but what about the overall picture? In this article we will show the general picture of all MQL5.community members achievements.

![Mechanical Trading System "Chuvashov's Fork"](https://c.mql5.com/2/17/944_28.png)[Mechanical Trading System "Chuvashov's Fork"](https://www.mql5.com/en/articles/1352)

This article draws your attention to the brief review of the method and program code of the mechanical trading system based on the technique proposed by Stanislav Chuvashov. The market analysis considered in the article has something in common with Thomas DeMark's approach to drawing trend lines for the last closest time interval, fractals being the reference points in the construction of trend lines.

![AutoElliottWaveMaker - MetaTrader 5 Tool for Semi-Automatic Analysis of Elliott Waves](https://c.mql5.com/2/0/ElliottWaveMaker2_0.png)[AutoElliottWaveMaker - MetaTrader 5 Tool for Semi-Automatic Analysis of Elliott Waves](https://www.mql5.com/en/articles/378)

The article provides a review of AutoElliottWaveMaker - the first development for Elliott Wave analysis in MetaTrader 5 that represents a combination of manual and automatic wave labeling. The wave analysis tool is written exclusively in MQL5 and does not include external dll libraries. This is another proof that sophisticated and interesting programs can (and should) be developed in MQL5.

![Econometrics EURUSD One-Step-Ahead Forecast](https://c.mql5.com/2/12/1003_13.png)[Econometrics EURUSD One-Step-Ahead Forecast](https://www.mql5.com/en/articles/1345)

The article focuses on one-step-ahead forecasting for EURUSD using EViews software and a further evaluation of forecasting results using the programs in EViews. The forecast involves regression models and is evaluated by means of an Expert Advisor developed for MetaTrader 4.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=juyizokewckcvgnfgnanbqbrkhdksyqx&ssn=1769252867559205281&ssn_dr=0&ssn_sr=0&fv_date=1769252867&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F364&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Getting%20Rid%20of%20Self-Made%20DLLs%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925286728073531&fz_uniq=5083354439550507522&sv=2552)

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