---
title: Using Pseudo-Templates as Alternative to C++ Templates
url: https://www.mql5.com/en/articles/253
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:08:27.239361
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/253&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083360675843021353)

MetaTrader 5 / Examples


### Introduction

The question of implementation of templates as a standard of the language was raised for many times at the [mql5.com](https://www.mql5.com/en/forum) forum. As I faced the wall of refusal from the developers of MQL5, my interest in implementation of templates using custom methods started to grow. The result of my studies is presented in this article.

### Some History of C and C++

From the very beginning, the C language was developed to offer the possibility of performing system tasks. Creators of the C language didn't implement an abstract model of execution environment of the language; they just implemented features for the needs of system programmers. First of all, those are the methods of direct work with memory, structure constructions of controlling and module management of applications.

Actually, nothing more was included in the language; all the other things were taken to the runtime library. That's why some ill-disposed people sometimes refer to the C language as to the structured assembler. But whatever they say, the approach appeared to be very successful. Owing to it, a new level of ratio between simplicity and power of the language was reached.

So, the C appeared as a universal language for system programming. But it didn't stay within those limits. At late 80-s, pushing Fortran aside from leadership, C earned a wide popularity among programmers all over the world and became widely used in different applications. A significant contribution to its popularity was made by distribution of Unix (and so the C language) in universities, where a new generation of programmers was educated.

But if everything is so unclouded, then why all the other languages are still used and what supports their existence? The Achilles heel of the C language is being too low-level for problems set in 90-s. This problem has two aspects.

On the first hand, the language contains too low-level means: first of all, it is the work with memory and address arithmetic. Thus, a change in processor bitness causes a lot of problem for many C-applications. On the other hand, there is a lack of high-level means in C - abstract types of data and objects, polymorphism and handling of exceptions. Thus, in C applications a technique of implementation of a task often dominates over its material side.

First attempts to fix these disadvantages were made at early 80-s. At that time Bjarne Stroustrup in AT&T Bell Labs started developing the extension of the C language called "C with classes". The style of the development corresponded with the spirit of creation of the C language itself - adding of different features to make work of certain groups of people more convenient.

The main innovation in C++ is the mechanism of classes that gives a possibility of using new types of data. A programmer describes the internal representation of class object and the set of functions-methods to access that representation. One of the main purposes of creation of C++ was to increase proportion of reusing of already written code.

Innovation of the C++ language does not consist in only introduction of classes. It has implemented mechanism of structural handling of exceptions (lack of it complicated the development of fail-safe applications), mechanism of templates and many other things. Thus, the main line of development of the language was directed at extending its possibilities by introducing new high-level constructions with keeping of the full compatibility with [ANSI С](https://en.wikipedia.org/wiki/ANSI_C "https://en.wikipedia.org/wiki/ANSI_C").

### Template as a Mechanism of Macro Substitution

To understand how to implement a template in MQL5, you need to understand how they work in C++.

Let's see the [definition](https://en.wikipedia.org/wiki/Template_(programming) "https://en.wikipedia.org/wiki/Template_(programming)").

Templates are a feature of the C++ programming language that allow functions and classes to operate with generic types. This allows a function or class to work on many different data types without being rewritten for each one.

MQL5 doesn't have templates, but it doesn't mean that it's impossible to use a style of programming with templates. The mechanism of templates in the C++ language is, actually, a sophisticated mechanism of macro generation deeply embedded in the language. In other words, when a programmer uses a template, the compiler determines the type of data where the corresponding function is called, not where it is declared.

Templates where introduced in C++ to decrease the amount of code written by programmers. But you shouldn't forget that a code typed on a keyboard by a programmer is not the same to the one created by the compiler. The mechanism of templates itself did not result in decreasing of size of programs; it just decreased the size of their source code. That's why the main problem solved by using templates is decreasing of code typed by programmers.

Since the machine code is generated during compilation, ordinary programmers don't see if the code of a function is generated once or for several times. During compilation of a template code, the code of function is generated as many times as there are types, where the template was used. Basically, a template is overriding on the stage of compilation.

The second aspect of introducing templates in C++ is allocation of memory. The matter is memory in the C language is allocated statically. To make this allocation more flexible, a template that sets the size of memory for arrays is used. But this aspect has been already implemented by developers of MQL4 in the form of dynamic arrays, and it has also been done in MQL5 in the form of dynamic objects.

Thus, only the problem of substitution of types remains unsolved. Developers of MQL5 refused from solving it, referring to using of the mechanism of template substitution would allow cracking of the compiler, what would lead to appearing of a decompiler.

Well, they know it better. And we have only one choice - to implement this paradigm in a custom way.

First of all, let me make a remark that we are not going to change the compiler or change the standards of the language. I suggest changing the approach to templates themselves. If we cannot create templates at the stage of compilation, it doesn't mean that we are not allowed to write the machine code. I suggest moving the use of templates from the part of generation of binary code to the part, where the text code is written. Let's call this approach "pseudo-templates".

### Pseudo-Templates

A pseudo-template has its advantages and disadvantages, comparing to a C++ template. The disadvantages include additional manipulations with moving of files. The advantages include more flexible possibilities than that determined by the standards of the language. Let's pass from words to deeds.

To use pseudo-templates, we need an analogue of preprocessor. We will use the 'Templates' script for this purpose. Here are the general requirements to the script: it must read a specified file (keeping the data structure), find a template and replace it with specified types.

Here I need to make a remark. Since we are going to use the mechanism of overriding instead of templates, the code will be rewritten as many times as there are types that should be overridden. In other words, the substitution will be performed in the whole code given for the analysis. Then, the code will be rewritten for several times by the script, each time creating a new substitution. Thus, we can realize the slogan "manual work performed by machines".

### Developing the Script Code

Let's determine required input variables:

1. Name of a file to be processed.
2. A variable to store the type of data to be overridden.
3. Name of a template that will be used instead of real types of data.


```
input string folder="Example templat";//name of file for processing

input string type="long;double;datetime;string"
                 ;//names of custom types, separator ";"
string TEMPLAT="_XXX_";// template name
```

To make script multiply only a part of the code, set names of markers. The opening marker is intended to indicate the beginning of the part to be processed, and the closing marker - to indicate its end.

While using the script, I faced the problem of reading the markers.

During analysis I discovered, that when formatting a document in [MetaEditor](https://www.metatrader5.com/en/automated-trading/metaeditor "https://www.metatrader5.com/en/automated-trading/metaeditor"), a space or tabulation (depending on situation) is often added to the lines of comments. The problem was solved by deleting spaces before and after a significant symbol when determining markers. This feature is realized as automatic in the script, but there is a remark.

A marker name must not start or end with a space.

A closing marker is not obligatory; if it is absent, the code will be processed down to the end of the file. But there must be an opening one. Since the names of markers are constant, I use the [#define](https://www.mql5.com/en/docs/basis/preprosessor/constant) preprocessor directive instead of variables.

```
#define startread "//start point"
#define endread "//end point"
```

To form an array of types, I created the function void ParserInputType(int i,string &type\_d\[\],string text), which fills the type\_dates\[\] array with values using the 'type' variable.

Once the script receives the name of a file and markers, it starts reading the file. To save formatting of the document, the script reads the information line by line, saving found lines in the array.

Of course, you can flush everything in one variable; but in this case, you'll lose hyphenation and the text will turn into an endless line. That's why the function of reading the file uses the array of strings that changes its size at each iteration of getting a new string.

```
//+------------------------------------------------------------------+
//| downloading file                                                 |
//+------------------------------------------------------------------+
void ReadFile()
  {
   string subfolder="Templates";
   int han=FileOpen(subfolder+"\\"+folder+".mqh",FILE_READ|FILE_SHARE_READ|FILE_TXT|FILE_ANSI,"\r");
   if(han!=INVALID_HANDLE)
     {
      string temp="";
      //--- scrolling file to the starting point
      do {temp=FileReadString(han);StringTrimLeft(temp);StringTrimRight(temp);}
      while(startread!=temp);

      string text=""; int size;
      //--- reading the file to the array until a break point or the end of the file
      while(!FileIsEnding(han))
        {
         temp=text=FileReadString(han);
         // deleting symbols of tabulation to check the end
         StringTrimLeft(temp);StringTrimRight(temp);
         if(endread==temp)break;
         // flushing data to the array
         if(text!="")
           {
            size=ArraySize(fdates);
            ArrayResize(fdates,size+1);
            fdates[size]=text;
           }
        }
      FileClose(han);
     }
   else
     {
      Print("File open failed"+subfolder+"\\"+folder+".mqh, error",GetLastError());
      flagnew=true;
     }
  }
```

For convenience of use, the file is opened in the [FILE\_SHARE\_READ](https://www.mql5.com/en/docs/constants/io_constants/fileflags) mode. It gives a possibility of starting the script without closing the edited file. The file extension is specified as 'mqh'. Thus, the script directly reads the text of the code that is stored in the include file. The matter is a file with the 'mqh' extension is, actually, a text file; you can make it sure by simply renaming the file into 'txt' and opening the 'mqh' file using any text editor.

At the end of reading, the length of the array is equal to the number of lines between the start and end markers.

**Name of the opened file must have the "templat" extension, otherwise the initial file will be overwritten and all information will be lost.**

Now let's turn to the analysis of information. The function that analyses and replaces information is called from the function of writing to a file void WriteFile(int count). Comments are given inside the function.

```
void WriteFile(int count)
  {
   ...
   if(han!=INVALID_HANDLE)
     {
      if(flagnew)// if the file cannot be read
        {
         ...
        }
      else
        {// if the file exists
         ArrayResize(tempfdates,count);
         int count_type=ArraySize(type_dates);
         //--- the cycle rewrites the contents of the file for each type of the type_dates template
         for(int j=0;j<count_type;j++)
           {
            for(int i=0;i<count;i++) // copy data into the temporary array
               tempfdates[i]=fdates[i];
            for(int i=0;i<count;i++) // replace templates with types
               Replace(tempfdates,i,j);

            for(int i=0;i<count;i++)
               FileWrite(han,tempfdates[i]); // flushing array in the file
           }
        }
     ...
  }
```

Since the data is replaced at its place and the array is changed after transformation, we will work with a copy of it. Here we set the size of the tempfdates\[\] array used for temporary storage of data and fill it according to the fdates\[\] example.

Then the substitution of templates using the Replace() function is performed. Parameters of the function are: array to be processed (where the substitution of the template is performed), the counter of lines **i** (to move inside the array), and the counter of types **j**(to navigate through the array of types).

Since we have two nested cycles, the source code is printed as many times as there are types specified.

```
//+------------------------------------------------------------------+
//| replacing templates with types                                   |
//+------------------------------------------------------------------+
void Replace(string &temp_m[],int i,int j)
  {
   if(i>=ArraySize(temp_m))return;
   if(j<ArraySize(type_dates))
      StringReplac(temp_m[i],TEMPLAT,type_dates[j]);// replacing  templat with types
  }
```

The Replace() function contains checks (to avoid calling of a nonexistent index of an array) and it calls the nested function StringReplac(). There is a reason why the name of the function is similar to the standard function [StringReplace](https://www.mql5.com/en/docs/strings/stringreplace), they also have the same number of parameters.

Thus, by adding a single letter " **e**", we can change the entire logic of replacing. The [standard function](https://www.mql5.com/en/docs/strings/stringreplace) takes the value of the 'find' example and replaces it with the specified string 'replacement'. And my function not only replaces, but analyses if there are symbols before 'find' (i.e. checks if 'find' is a part of a word); and if there are, it replaces 'find' with 'replacement' but in the upper case, otherwise the replacement is performed as is. Therefore, in addition to setting types, you can use them in the names of overridden data.

### Innovations

Now let me tell about innovations that were added while using. I already mentioned that there were problems of reading markers while using the script.

The problem is solved by the following code inside the void ReadFile() function:

```
      string temp="";
      //--- scrolling the file to the start point
      do {temp=FileReadString(han);StringTrimLeft(temp);StringTrimRight(temp);}
      while(startread!=temp);
```

The cycle itself was implemented in previous version, but cutting off the symbols of tabulation using the [StringTrimLeft()](https://www.mql5.com/en/docs/strings/stringtrimleft) and [StringTrimRight()](https://www.mql5.com/en/docs/strings/stringtrimright) functions appeared only in the enhanced version.

In addition, the innovations include cutting off the "templat" extension from the name of output file, so the output file is ready for being used. It is implemented using the function of deleting of a specified example from a specified string.

Code of the function of deletion:

```
//+------------------------------------------------------------------+
//| Deleting the 'find' template from the 'text' string              |
//+------------------------------------------------------------------+
string StringDel(string text,const string find)
  {
   string str=text;
   StringReplace(str,find,"");
   return(str);
  }
```

Code that performs cutting of a file name is located in the function void WriteFile(int count):

```
   string newfolder;
   if(flagnew)newfolder=folder;// if it is the first start, create an empty file of pre-template
   else newfolder=StringDel(folder," templat");// or create the output file according to the template
```

In addition to it, the mode of preparing of a pre-template is introduced. If the required file does not exist in the Files/Templates directory, it will be formed as a pre-template file.

Example:

```
//#define _XXX_ long

//this is the start point
 _XXX_
//this is the end point
```

Code that creates that lines is located in the void WriteFile(int count) function:

```
      if(flagnew)// if the file couldn't be read
        {// fill the template file with the pre-template
         FileWrite(han,"#define "+TEMPLAT+" "+type_dates[0]);
         FileWrite(han," ");
         FileWrite(han,startread);
         FileWrite(han," "+TEMPLAT);
         FileWrite(han,endread);
         Print("Creating pre-template "+subfolder+"\\"+folder+".mqh");
        }
```

Execution of the code is protected by the global variable flagnew, which takes the 'true' value if there was an error of reading the file.

While using the script, I added an additional template. The process of connection of the second template is the same. Functions that require changes in them are placed closer to the [OnStart()](https://www.mql5.com/en/docs/basis/function/events) function for connection of an additional template. A path for connecting new templates is beaten. Thus, we have a possibility to connect as many templates as we need. Now let's check the operation.

### Checking Operation

First of all, let's start the script specifying all the require parameters. In the window that appears, specify the "Example templat" file name.

Fill the fields of custom types of data using the ';' separator.

![Start window of the script](https://c.mql5.com/2/2/1__2.png)

As soon as the "OK" button is pressed, the Templates directory is created; it contains the pre-template file "Example templat.mqh".

This event is displayed in the journal with the message:

![Journal messages](https://c.mql5.com/2/2/2__1__1.png)

Let's change the pre-template and start the script once again. This time the file already exists in the Templates directory (as well as the directory itself), that's why the message about the error of opening the file will not be displayed. The replacement will be performed according to the specified template:

```
//this_is_the_start_point
 _XXX_ Value_XXX_;
//this_is_the_end_point
```

Open the created file "Example.mqh" once again.

```
 long ValueLONG;
 double ValueDOUBLE;
 datetime ValueDATETIME;
 string ValueSTRING;
```

As you can see, 4 lines are made from one line according to the number of types that we passed as the parameter. Now write two following lines in the template file:

```
//this_is_the_start_point
 _XXX_ Value_XXX_;
 _XXX_ Type_XXX_;
//this_is_the_end_point
```

The result demonstrates the logic of the script operation in a clear manner.

First of all, the entire code is rewritten with one type of data, then the processing of another type is performed. This is done until all the types are processed.

```
 long ValueLONG;
 long TypeLONG;
 double ValueDOUBLE;
 double TypeDOUBLE;
 datetime ValueDATETIME;
 datetime TypeDATETIME;
 string ValueSTRING;
 string TypeSTRING;
```

Now include the second template in the text of example.

```
//this_is_the_start_point
 _XXX_ Value_XXX_(_xxx_ ind){return((_XXX_)ind);};
 _XXX_ Type_XXX_(_xxx_ ind){return((_XXX_)ind);};

 //this_is_the_end_button
```

Result:

```
 long ValueLONG(int ind){return((long)ind);};
 long TypeLONG(int ind){return((long)ind);};

 double ValueDOUBLE(float ind){return((double)ind);};
 double TypeDOUBLE(float ind){return((double)ind);};

 datetime ValueDATETIME(int ind){return((datetime)ind);};
 datetime TypeDATETIME(int ind){return((datetime)ind);};

 string ValueSTRING(string ind){return((string)ind);};
 string TypeSTRING(string ind){return((string)ind);};
```

In the last example, I intentionally put a space after the last line. That space demonstrates where the script ends processing of one type and starts processing another. Regarding the second template, we can note that processing of types is performed similarly to the first template. If a corresponding type is not found for a type of the first template, nothing is printed.

Now I want to clarify the question of debugging the code. The given examples are pretty simple for debugging. During programming, you may need to debug a pretty big part of the code, and multiply it as soon as it is done. To do it, there is a reserved commented line in the pre-template: "//#define \_XXX\_ long".

If you remove the comments, our template will become a real type. In other words, we will tell the compiler how the template must be interpreted.

Unfortunately, we cannot debug all the types in this way. But we can debug one type, and then change the type of the template in 'define'; so we can debug all the types one by one. Of course, for debugging, we need to move the file to the directory of the called file or to the Include directory. This is the inconvenience of debugging I mentioned before when talking about disadvantages of the pseudo-templates.

### Conclusion

In conclusion, I want to say, though the idea of using pseudo-templates is interesting and pretty productive, it is only the idea with a small start of implementation. Though the code described above is working and it saved a lot of hours of writing the code for me, many questions are still open. First of all, it is the question of developing the standards.

My script implements the block replacement of templates. But this approach is not obligatory. You can create a more complex analyzer that interprets certain rules. But here is the beginning. Hope for a big discussion. Thought thrives on conflict. Good luck!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/253](https://www.mql5.com/ru/articles/253)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/253.zip "Download all attachments in the single ZIP archive")

[templates.mq5](https://www.mql5.com/en/articles/download/253/templates.mq5 "Download templates.mq5")(9.91 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Self-organizing feature maps (Kohonen maps) - revisiting the subject](https://www.mql5.com/en/articles/2043)
- [Debugging MQL5 Programs](https://www.mql5.com/en/articles/654)
- [The Player of Trading Based on Deal History](https://www.mql5.com/en/articles/242)
- [Electronic Tables in MQL5](https://www.mql5.com/en/articles/228)
- [How to Copy Trading from MetaTrader 5 to MetaTrader 4](https://www.mql5.com/en/articles/189)
- [Evaluation of Trade Systems - the Effectiveness of Entering, Exiting and Trades in General](https://www.mql5.com/en/articles/137)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/3443)**
(11)


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
7 Mar 2011 at 13:07

[Nikolai](https://www.mql5.com/en/users/Urain "https://www.mql5.com/en/users/Urain"), here is a question. Do you have a function that would evaluate a string? I.e. for example:

```
str="double a=1.5;"

void eval(str);
Print(a); // a=1.5
```

I need a **void eval(str)** function, with the result a=1.5.

The article is good, thanks!

![Mykola Demko](https://c.mql5.com/avatar/2014/7/53C7D9B0-F88C.jpg)

**[Mykola Demko](https://www.mql5.com/en/users/urain)**
\|
7 Mar 2011 at 15:27

**denkir:**

[Nikolai](https://www.mql5.com/en/users/Urain "https://www.mql5.com/en/users/Urain"), here is a question. Do you have a function that would evaluate a string? I.e. for example:

I need a **void eval(str)** function, with the result a=1.5.

The article is good, thank you!

I understand that the question is not about evaluation, but about parsing. Parsing is very versatile.

You can write different rules. It all depends on what you want to get.

For example: how should the parser behave in such examples?

```
"double a=1.5;"
"double a =1.5;"
"double a = 1.5;"
"double a=1.5,b=2.5;"
"double a =1.5,b =2.5;"
"double a = 1.5,b = 2.5;"
"double a = 1.5 , b = 2.5 ;"
```

you know. I can only recommend you to study the functionality of the string functions section on simple examples. If you will be able to operate functions from memory, you will be able to plan your development based on the possibilities of the functionality. Without thinking about what you can and cannot do. Knowing the possibilities of the functionality, you will have a clear idea which algorithm will be the most suitable for solving your task.


![Denis Kirichenko](https://c.mql5.com/avatar/2019/5/5CEDB8D2-7CB7.jpg)

**[Denis Kirichenko](https://www.mql5.com/en/users/denkir)**
\|
26 Mar 2011 at 17:36

**Urain:**

So I understand that the question is not about evaluation, but about parsing. Parsing is very versatile.

You can have different rules. It all depends on what you want to get.

For example: how the parser should behave in such examples.

you know. I can only recommend you to study the functionality of the string functions section on simple examples. If you will be able to operate functions from memory, you will be able to plan your development based on the possibilities of the functionality. Without thinking about what you can and cannot do. Knowing the capabilities of the functionality, you will have a clear idea what algorithm will be the most suitable for solving your task.

thank you very much for the answer... I will study the question


![Mykola Demko](https://c.mql5.com/avatar/2014/7/53C7D9B0-F88C.jpg)

**[Mykola Demko](https://www.mql5.com/en/users/urain)**
\|
26 Mar 2011 at 18:08

**denkir:**

thanks so much for the reply... I will study the question

By the way, the [logical operations](https://www.mql5.com/en/docs/basis/operations/bool "MQL5 Documentation: Logical Operations") < > \> == are defined for **strings**.


![Alain Verleyen](https://c.mql5.com/avatar/2024/5/663a6cdf-e866.jpg)

**[Alain Verleyen](https://www.mql5.com/en/users/angevoyageur)**
\|
15 Dec 2013 at 14:32

Interesting article but [it is now obsolete](https://www.mql5.com/en/forum/53/page14#comment_365002) as [mql5 have now native function template](https://www.mql5.com/en/docs/basis/oop/templates).


![Electronic Tables in MQL5](https://c.mql5.com/2/0/MQL5_table__1.png)[Electronic Tables in MQL5](https://www.mql5.com/en/articles/228)

The article describes a class of dynamic two-dimensional array that contains data of different types in its first dimension. Storing data in the form of a table is convenient for solving a wide range of problems of arrangement, storing and operation with bound information of different types. The source code of the class that implements the functionality of working with tables is attached to the article.

![Random Walk and the Trend Indicator](https://c.mql5.com/2/0/coin_course.png)[Random Walk and the Trend Indicator](https://www.mql5.com/en/articles/248)

Random Walk looks very similar to the real market data, but it has some significant features. In this article we will consider the properties of Random Walk, simulated using the coin-tossing game. To study the properties of the data, the trendiness indicator is developed.

![The Implementation of Automatic Analysis of the Elliott Waves in MQL5](https://c.mql5.com/2/0/MQL5_Elliott_Waves_Automated.png)[The Implementation of Automatic Analysis of the Elliott Waves in MQL5](https://www.mql5.com/en/articles/260)

One of the most popular methods of market analysis is the Elliott Wave Principle. However, this process is quite complicated, which leads us to the use of additional tools. One of such instruments is the automatic marker. This article describes the creation of an automatic analyzer of Elliott Waves in MQL5 language.

![Use of Resources in MQL5](https://c.mql5.com/2/0/Resources_MQL5.png)[Use of Resources in MQL5](https://www.mql5.com/en/articles/261)

MQL5 programs not only automate routine calculations, but also can create a full-featured graphical environment. The functions for creating truly interactive controls are now virtually the same rich, as those in classical programming languages. If you want to write a full-fledged stand-alone program in MQL5, use resources in them. Programs with resources are easier to maintain and distribute.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/253&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5083360675843021353)

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