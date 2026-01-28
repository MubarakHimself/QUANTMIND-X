---
title: How to Use Crashlogs to Debug Your Own DLLs
url: https://www.mql5.com/en/articles/1414
categories: Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:59:38.877370
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/1414&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083264820762908697)

MetaTrader 4 / Examples


MetaTrader 4 Client Terminal has an integrated means to detect error conditions
that occur during the terminal work and generate crashlogs where such errors are
reported. The generated report is stored in the logs\\crashlog.log file to be sent
to the trade server at the next startup of the client terminal. It should be noted
that the error condition report does not contain any user's personal details, but
only the system data that allow to localize the error in the client terminal. This
information is very important for manufacturers since it is used to correct critical
errors. Then the software developed will become even more crashproof.

25 to 30% of all crashlogs received from users appear due to errors occurring when
functions imported from custom dlls are executed. This information will not help
the developers of the client temrinal in any way, but it can help the developers
of the corresponding dlls in their trouble-shooting. We will show how the data
from the error report can be used. Examples named ExpertSample.dll and ExportFunctions.
mq4 that can be found in the experts\\samples directory were taken as a basis.

![](https://c.mql5.com/2/13/crashlog_1.png)

The full text of the error report is given below:

```
Time        : 2006.07.12 14:43
Program     : Client Terminal
Version     : 4.00 (build: 195, 30 Jun 2006)
Owner       : MetaQuotes Software Corp. (MetaTrader)
OS          : Windows XP Professional 5.1 Service Pack 2 (Build 2600)
Processors  : 2, type 586, level 15
Memory      : 2095848/1727500 kb
Exception   : C0000005
Address     : 77C36FA3
Access Type : read
Access Addr : 00000000

Registers   : EAX=000000FF CS=001b EIP=77C36FA3 EFLGS=00010202
            : EBX=FFFFFFFF SS=0023 ESP=024DFABC EBP=024DFAC4
            : ECX=0000003F DS=0023 ESI=00000000 FS=003b
            : EDX=00000003 ES=0023 EDI=10003250 GS=0000

Stack Trace : 10001079 0045342E 0045D627 004506EC
            : 7C80B50B 00000000 00000000 00000000
            : 00000000 00000000 00000000 00000000
            : 00000000 00000000 00000000 00000000
Modules     :
          1 : 00400000 00292000 C:\Program Files\MetaTrader 4\terminal.exe
          2 : 10000000 00005000 C:\Program Files\MetaTrader 4\experts\libraries\ExpertSample.dll
         ...   ..........................................................
         35 : 7C9C0000 00819000 C:\WINDOWS\system32\SHELL32.dll

Call stack  :
77C36F70:0033 [77C36FA3] memcpy                           [C:\WINDOWS\system32\msvcrt.dll]
10001051:0028 [10001079] GetStringValue                   [C:\Program Files\MetaTrader 4\experts\libraries\ExpertSample.dll]
00452DD0:065E [0045342E] ?CallDllFunction@CExpertInterior
00459AC0:3B67 [0045D627] ?ExecuteStaticAsm@CExpertInterior
004505E0:010C [004506EC] ?RunExpertInt@CExpertInterior
7C80B357:01B4 [7C80B50B] GetModuleFileNameA               [C:\WINDOWS\system32\kernel32.dll]
```

So, what has happened?

- Exception : C0000005 means an error condition that has occurred due to Access Violation.

- Access Type : read means that there has been an attempt to read.
- Acess Addr : 00000000 means that the out-of-process memory has a zero address.


Now, let us look at the call stack.

The address of 77C36FA3 is the same as that of the top of stack. This means that
the error occurred during execution of the memcpy function that copies the content
of one memory area to another one. At that, we can judge with much certainty about
whether there was an attempt to copy data from the memory area having a zero address.

The second line of the call stack informs us about what function called the memcpy
function with wrong parameters. This is the GetStringValue function from the library
named ExpertSample.dll.

Let us have a look at the source code of this function:

```
__declspec(dllexport) char* __stdcall GetStringValue(char *spar)
  {
   static char temp_string[256];
//----
   printf("GetStringValue takes \"%s\"\n",spar);
   memcpy(temp_string,spar,sizeof(temp_string)-1);
   temp_string[sizeof(temp_string)-1]=0;
//----
   return(temp_string);
  }
```

We can see that the memcpy function is only once called within the above function.
Since the first parameter indicates the existing memory area occupied by the temp\_string
variable, we can conclude that it is the second parameter which is worng. Indeed,
there is no checking of the variable for 0 in the given example. Line if(spar==NULL)
would protect us against the crash.

So, what should be done if there were more than one call of the memcpy function
in the function anayzed? In our project settings, let us set up the output of the
most detailed listing of the compilation.

![](https://c.mql5.com/2/13/settings.png)

After the project has been rebuilt, we will have a listing file with the extension
.cod for every source .cpp file. We are now interested in the ExpertSample. cod,
but only the part of code obtained for the GetStringValue function. Here it is:

```
?GetStringValue@@YGPADPAD@Z PROC NEAR           ; GetStringValue

; 70   :   {

  00051 55       push    ebp
  00052 8b ec        mov     ebp, esp

; 71   :    static char temp_string[256];
; 72   : //----
; 73   :    printf("GetStringValue takes \"%s\"\n",spar);

  00054 8b 45 08     mov     eax, DWORD PTR _spar$[ebp]
  00057 50       push    eax
  00058 68 00 00 00 00   push    OFFSET FLAT:$SG19680
  0005d ff 15 00 00 00
    00       call    DWORD PTR __imp__printf
  00063 83 c4 08     add     esp, 8

; 74   :    memcpy(temp_string,spar,sizeof(temp_string)-1);

  00066 68 ff 00 00 00   push    255            ; 000000ffH
  0006b 8b 4d 08     mov     ecx, DWORD PTR _spar$[ebp]
  0006e 51       push    ecx
  0006f 68 00 00 00 00   push    OFFSET FLAT:_?temp_string@?1??GetStringValue@@YGPADPAD@Z@4PADA
  00074 e8 00 00 00 00   call    _memcpy
  00079 83 c4 0c     add     esp,  12            ; 0000000cH

; 75   :    temp_string[sizeof(temp_string)-1]=0;

  0007c c6 05 ff 00 00
    00 00        mov     BYTE PTR _?temp_string@?1??GetStringValue@@YGPADPAD@Z@4PADA+255, 0

; 76   : //----
; 77   :    return(temp_string);

  00083 b8 00 00 00 00   mov     eax, OFFSET FLAT:_?temp_string@?1??GetStringValue@@YGPADPAD@Z@4PADA

; 78   :   }

  00088 5d       pop     ebp
  00089 c2 04 00     ret     4
?GetStringValue@@YGPADPAD@Z ENDP            ; GetStringValue

```

Digits 10001051:0028 in the second line of the call stack give the address within
the GetStringValue function. After the function placed a line above in the call
stack is executed, the control will be given to this address. In the object code,
the GetStringValue function starts with address 00051 (it should be noted that
addresses are presented in hexadecimal notation). Let us add 0028 to this value
and so we will get address 00079. At this address, the add esp,12 instruction is situated that follows immediately after the memcpy function calling
instruction. We have found this spot.

Let us investigate the case when the error occurs immediately inside of the imported
function. Let us modify the code:

```
__declspec(dllexport) char* __stdcall GetStringValue(char *spar)
  {
   static char temp_string[256];
//----
   printf("GetStringValue takes \"%s\"\n",spar);
   for(int i=0; i<sizeof(temp_string)-1; i++)
     {
      temp_string[i]=spar[i];
      if(spar[i]==0) break;
     }
   temp_string[sizeof(temp_string)-1]=0;
//----
   return(temp_string);
  }
```

We have replaced the memcpy function call with our own byte-wise data copying loop.
But we did not use the checking for 0 in order to create an error condition and
the error report. In the new report, the call stack looks a bit different:

```
Call stack  :
10001051:003A [1000108B] GetStringValue                   [C:\Program Files\MetaTrader 4\experts\libraries\ExpertSample.dll]
00452DD0:065E [0045342E] ?CallDllFunction@CExpertInterior
00459AC0:3B67 [0045D627] ?ExecuteStaticAsm@CExpertInterior
004505E0:010C [004506EC] ?RunExpertInt@CExpertInterior
7C80B357:01B4 [7C80B50B] GetModuleFileNameA               [C:\WINDOWS\system32\kernel32.dll]
```

The error occurred at address 003A in the GetStringValue function. Let as look
at the generated listing.

```
?GetStringValue@@YGPADPAD@Z PROC NEAR           ; GetStringValue

; 70   :   {

  00051 55       push    ebp
  00052 8b ec        mov     ebp, esp
  00054 51       push    ecx

; 71   :    static char temp_string[256];
; 72   : //----
; 73   :    printf("GetStringValue takes \"%s\"\n",spar);

  00055 8b 45 08     mov     eax, DWORD PTR _spar$[ebp]
  00058 50       push    eax
  00059 68 00 00 00 00   push    OFFSET FLAT:$SG19680
  0005e ff 15 00 00 00
    00       call    DWORD PTR __imp__printf
  00064 83 c4 08     add     esp, 8

; 74   :    for(int i=0; i<sizeof(temp_string)-1; i++)

  00067 c7 45 fc 00 00
    00 00        mov     DWORD PTR _i$[ebp], 0
  0006e eb 09        jmp     SHORT $L19682
$L19683:
  00070 8b 4d fc     mov     ecx, DWORD PTR _i$[ebp]
  00073 83 c1 01     add     ecx, 1
  00076 89 4d fc     mov     DWORD PTR _i$[ebp], ecx
$L19682:
  00079 81 7d fc ff 00
    00 00        cmp     DWORD PTR _i$[ebp], 255    ; 000000ffH
  00080 73 22        jae     SHORT $L19684

; 76   :       temp_string[i]=spar[i];

  00082 8b 55 08     mov     edx, DWORD PTR _spar$[ebp]
  00085 03 55 fc     add     edx, DWORD PTR _i$[ebp]
  00088 8b 45 fc     mov     eax, DWORD PTR _i$[ebp]
  0008b 8a 0a        mov     cl, BYTE PTR [edx]
  0008d 88 88 00 00 00
    00       mov     BYTE PTR _?temp_string@?1??GetStringValue@@YGPADPAD@Z@4PADA[eax], cl

; 77   :       if(spar[i]==0) break;

  00093 8b 55 08     mov     edx, DWORD PTR _spar$[ebp]
  00096 03 55 fc     add     edx, DWORD PTR _i$[ebp]
  00099 0f be 02     movsx   eax, BYTE PTR [edx]
  0009c 85 c0        test    eax, eax
  0009e 75 02        jne     SHORT $L19685
  000a0 eb 02        jmp     SHORT $L19684
$L19685:

; 78   :      }

  000a2 eb cc        jmp     SHORT $L19683
$L19684:

; 79   :    temp_string[sizeof(temp_string)-1]=0;

  000a4 c6 05 ff 00 00
    00 00        mov     BYTE PTR _?temp_string@?1??GetStringValue@@YGPADPAD@Z@4PADA+255, 0

; 80   : //----
; 81   :    return(temp_string);

  000ab b8 00 00 00 00   mov     eax, OFFSET FLAT:_?temp_string@?1??GetStringValue@@YGPADPAD@Z@4PADA

; 82   :   }

  000b0 8b e5        mov     esp, ebp
  000b2 5d       pop     ebp
  000b3 c2 04 00     ret     4
?GetStringValue@@YGPADPAD@Z ENDP            ; GetStringValue
```

The initial address is the same: 00051. Let us add 003A and obtain address 0008B.
At this address, the mov cl, BYTE PTR \[edx\] instruction is situated. Let us see the registers contents in the report:

```
Registers   : EAX=00000000 CS=001b EIP=1000108B EFLGS=00010246
            : EBX=FFFFFFFF SS=0023 ESP=0259FAD4 EBP=0259FAD8
            : ECX=77C318BF DS=0023 ESI=018ECD80 FS=003b
            : EDX=00000000 ES=0023 EDI=000000E8 GS=0000
```

Well, of course, EDX register contains zeros. We accessed to the out-of-process
memory and got the crash.

In the end, two lines about how we have passed zero indication to the imported
function.

```
   string null_string;
   string sret=GetStringValue(null_string);
```

We passed an uninitialized string as a parameter. Be careful with uninitialized
strings, always check the received indications for NULL, and let you have as few
crashes as possible.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/1414](https://www.mql5.com/ru/articles/1414)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Testing of Expert Advisors in the MetaTrader 4 Client Terminal: An Outward Glance](https://www.mql5.com/en/articles/1417)
- [How to Evaluate the Expert Testing Results](https://www.mql5.com/en/articles/1403)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39208)**
(2)


![okwh](https://c.mql5.com/avatar/2011/9/4E7F67FD-3C19.jpg)

**[okwh](https://www.mql5.com/en/users/dxdcn)**
\|
28 Jun 2009 at 10:32

char\\* \_\_stdcallGetStringValue(char \*spar)

Is it possible to return a changed string with spar ???


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
11 Nov 2009 at 00:02

Sometimes  I have an empty

CALL [STACK](https://www.mql5.com/en/articles/4228 "Article: Deep Neural Networks (Part VII). Ensemble of neural networks: stacking") crash log section. Why?

![MagicNumber: "Magic" Identifier of the Order](https://c.mql5.com/2/13/105_2.gif)[MagicNumber: "Magic" Identifier of the Order](https://www.mql5.com/en/articles/1359)

The article deals with the problem of conflict-free trading of several experts on the same МТ 4 Client Terminal. It "teaches" the expert to manage only "its own" orders without modifying or closing "someone else's" positions (opened manually or by other experts). The article was written for users who have basic skills of working with the terminal and programming in MQL 4.

![Working with Files. An Example of Important Market Events Visualization](https://c.mql5.com/2/13/112_2.gif)[Working with Files. An Example of Important Market Events Visualization](https://www.mql5.com/en/articles/1382)

The article deals with the outlook of using MQL4 for more productive work at FOREX markets.

![A Pause between Trades](https://c.mql5.com/2/12/103_1.gif)[A Pause between Trades](https://www.mql5.com/en/articles/1355)

The article deals with the problem of how to arrange pauses between trade operations when a number of experts work on one МТ 4 Client Terminal. It is intended for users who have basic skills in both working with the terminal and programming in MQL 4.

![Error 146 ("Trade context busy") and How to Deal with It](https://c.mql5.com/2/17/94_1.gif)[Error 146 ("Trade context busy") and How to Deal with It](https://www.mql5.com/en/articles/1412)

The article deals with conflict-free trading of several experts on one МТ 4 Client Terminal. It will be useful for those who have basic command of working with the terminal and programming in MQL 4.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=maexmcfnjwilcocsmjqivwosvvhxlguh&ssn=1769252377066047695&ssn_dr=0&ssn_sr=0&fv_date=1769252377&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1414&back_ref=https%3A%2F%2Fwww.google.com%2F&title=How%20to%20Use%20Crashlogs%20to%20Debug%20Your%20Own%20DLLs%20-%20MQL4%20Articles&scr_res=1920x1080&ac=176925237798284396&fz_uniq=5083264820762908697&sv=2552)

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