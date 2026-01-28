---
title: Tips from a professional programmer (Part I): Code storing, debugging and compiling. Working with projects and logs
url: https://www.mql5.com/en/articles/9266
categories: Integration
relevance_score: 0
scraped_at: 2026-01-24T14:04:57.301081
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/9266&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083321849338665304)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/9266#topic00)
- [Store your code in separate subdirectories](https://www.mql5.com/en/articles/9266#topic01)
- [One code for multiple terminals](https://www.mql5.com/en/articles/9266#topic02)
- [Use a version control system](https://www.mql5.com/en/articles/9266#topic03)
- [Use a separate terminal with a demo account to debug the code](https://www.mql5.com/en/articles/9266#topic04)
- [Compile all code files at once](https://www.mql5.com/en/articles/9266#topic05)
- [Use project and task management systems](https://www.mql5.com/en/articles/9266#topic06)
- [Log selection](https://www.mql5.com/en/articles/9266#topic07)
- [Log highlighting](https://www.mql5.com/en/articles/9266#topic08)
- [Contextual search](https://www.mql5.com/en/articles/9266#topic09)
- [Conclusion](https://www.mql5.com/en/articles/9266#topicNN)

### Introduction

Everyone has their own programming habits, styles and preferences. I will share some methods which make my job (my favorite activity) easier. I hope this information will be useful for beginners. More experienced programmers might also find something useful.

### Store your code in separate subdirectories

Terminal program files are located under the MQL5 directory. This catalog is a so-called "sandbox". Data access from the outside is closed. This is a good decision. However, the use of a DLL probably enables access to anywhere.

For example, here is the structure of the Cayman project:

- /Experts/Cayman/ - Expert Advisor
- /Files/Cayman/ - data files (settings, parameters)
- /Include/Cayman/ - library of classes (functions)
- /Scripts/Cayman/ - main operational scripts
- /Scripts/CaymanDev/ - developer scripts (used for debugging)

The main advantages of this placement are:

- Contextual search only in project files via TotalCommander
- Version control via Git (control is enabled only for the project files, while all other files are ignored)
- Easy copying to another terminal (demo -> real – release)

### One code for multiple terminals

One of the best practices in programming is avoiding duplicate code. If the same code lines appear in several different places, then you should better "wrap" these lines in a function. The same applies to MQL5 files: there should be only one text of a program file. This can be implemented by using a symbolic link to the MQL5 directory.

Suppose the project category is located at D:\\Project, while the terminal data directory is located at C:\\Users\\Pro\\AppData\\Roaming\\MetaQuotes\\Terminal\\9EB2973C469D24060397BB5158EA73A5

1. Close the terminal
2. Go to the data directory
3. Move the MQL5 directory to the projects directory
4. While in the data directory, run cmd and enter the following command



**mklink /D MQL5 D:\\Project\\MQL5**
5. Launch the terminal

The terminal will not even notice that the "sandbox" (program files) has moved to D:\\Project\\MQL5.

The main advantage of this placement is that all personal projects are gathered under the same directory (D:\\Project).

### Use a version control system

Professional programmers do not doubt the need to use such system. It is an indispensable tool, especially when a team of programmers is working on the same project. The question is which system you are going to use. This is de facto Git.

The main advantages of Git are:

- Local repository. Ability to experiment with branches. Switching to any branch (version) with one click.
- Convenient graphical interface (TortoiseGit). Control with the mouse.
- Free cloud repository for personal projects (Bitbucket). No fear that your PC hard disk will fail
- File change history with the ability to restore or view older versions (conveniently viewed in Bitbucket)

I will not provide here detailed installation and configuration instructions, but I will mention some specific features. Install Git (for command line operation) and TortoiseGit (to use the mouse). Under the D:\\Project\\MQL5 directory, create the .gitignore file with the following contents

```
# exclude files
*.ex4
*.ex5
*.dat
log.txt

# exclude directories completely
Images
Indicators
Libraries
Logs
Presets
Profiles
Services
"Shared Projects"
Levels
Params

# exclude directory contents
Experts/*
Files/*
Include/*
Scripts/*

# except for directories
!Experts/Cayman
!Files/Cayman
!Include/Cayman
!Scripts/Cayman
!Scripts/CaymanDev
```

This file will enable the version tracking of (\*.mq?) program files of the project. Create a local repository in the MQL5 directory. Add files and make the first commit (commit the version). The local repository is ready. Work with the code and do not forget to make frequent commits with a brief description of changes. The comments make it easier to browse and search the history later on.

To connect to a cloud repository, you first need to create a [Bitbucket](https://www.mql5.com/go?link=https://bitbucket.org/ "A repository hosting service with collaboration options") account and repository. Logically, the repository name should match the project name. In my case it is CaymanMQL5, while there is also CaymanMQL4. Import the local repository to the cloud one. Now, the cloud repository is ready. Basic actions via TortoiseGit(TG):

- Work with the code (one task - one commit)
- Check modifications (TG/Check for modifications…)
- Add new files (Add)
- Delete unnecessary (missing) files (Delete)
- Commit to the local repository (Commit)
- Push to the cloud repository (Push)

### Use a separate terminal with a demo account to debug the code

You should have the latest working version of code on a real account (only \*.ex? files, without \*.mq?). The code in the debugging process should be on a demo account. To copy from demo to real, you can use the following batch file:

```
@echo off
setlocal
set PROJECT=Cayman
set SOURCE=d:\Project\MQL5
set TARGET=c:\Users\Pro\AppData\Roaming\MetaQuotes\Terminal\2E8DC23981084565FA3E19C061F586B2\MQL5
set PARAMS=/MIR /NJH /NJS
rem MIR - MIRror a directory tree and delete dest files/folders that no longer exist in source
rem NJH - No Job Header
rem NJS - No Job Summary
echo Copy *.ex? // Source to Production
echo Source = %SOURCE%
echo Production = %TARGET%
robocopy %SOURCE%\Experts\%PROJECT% %TARGET%\Experts\%PROJECT% *.ex? %PARAMS%
robocopy %SOURCE%\Scripts\%PROJECT% %TARGET%\Scripts\%PROJECT% *.ex? %PARAMS%
rem Copy all files except AppSettings.txt, [Levels], [Params]
robocopy %SOURCE%\Files\%PROJECT% %TARGET%\Files\%PROJECT% *.* %PARAMS% /XF AppSettings.txt /XD Levels Params
robocopy %SOURCE%\Scripts\Cayman %TARGET%\Scripts\Cayman *.ex? /NJH /NJS
robocopy %SOURCE%\Scripts\CaymanDev %TARGET%\Scripts\CaymanDev *.ex? /NJH /NJS
echo.
endlocal
pause
```

### Compile all code files at once

The goal is to maintain code consistency. So, you should avoid, for example, changes in function parameters. The Expert Advisor can be compiled well, but there can also be a script which uses the old version of the function. The script (which was compiled earlier) can run well, but it will not perform the required functionality. For batch compilation, you can use the following file:

```
@echo off
setlocal
set METAEDITOR="C:\Program Files\RoboForex - MetaTrader 5\metaeditor64.exe"
set CAYMAN=d:\Project\MQL5\Scripts\Cayman
set CAYMAN_DEV=d:\Project\MQL5\Scripts\CaymanDev

echo METAEDITOR=%METAEDITOR%
echo CAYMAN=%CAYMAN%
echo CAYMAN_DEV=%CAYMAN_DEV%
echo.
echo Wait compile...

D:

cd %CAYMAN%
echo %CAYMAN%
for %%F in (*.mq?) do (
        %METAEDITOR% /compile:%%F /log
        type %%~dpnF.log
)
del *.log

cd %CAYMAN_DEV%
echo %CAYMAN_DEV%
for %%F in (*.mq?) do (
        %METAEDITOR% /compile:%%F /log
        type %%~dpnF.log
)
del *.log

endlocal
echo.
pause
```

### Use project and task management systems

Such systems are a must for development teams. They are also useful for personal projects. There are a plethora of different project management systems and methodologies. I personally use [ZenKit](https://www.mql5.com/go?link=https://zenkit.com/en/base/ "A project management web service for small groups and collaborations"). It is free for small teams. It offers a very convenient Kanban board.

I used to have multiple Kanban boards (one for each project). The disadvantage of using multiple boards is that you do not see the overall picture. Then I decided to **add a project as a development stage**. This way it is much easier to manage tasks. For example, my Kanban board has 5 development stages:

- Legend - project description, links and instructions. Entries are not moved to other stages. You can always view the general purpose of the board and can easily access links to useful resources.
- Cayman – MQL5 and MQL4 project
- Website – my personal website project
- Accepted - task in progress
- Done - task completed

The board using process is very simple and intuitive. I find an interesting idea, a solution or a program error. Then I briefly formulate it and add it to the board, to the corresponding project. You can also add here images and files. Next, I select a task and move it to the "Accepted" stage. I work with the task, test it and move to the "Done" stage.

By analyzing the number of tasks in projects you can easily determine problems and bottlenecks. I try to take tasks from "overloaded" projects. My goal is to align projects so that they have similar number of tasks. The team development has another stage entitled "Testing" stage, which is not relevant to me as I work alone.

### Log selection

As you know, all outputs of Print functions are written to the Terminal log MQL5\\Logs\\yyyyMMdd.log. All symbols (trading instruments are mixed in this file. I use the following batch file to select logs for a required symbol:

```
echo off
if "%2"=="" goto Help
find /I "%2" < %1 > "%~n1-%2.log"
goto Exit
:Help
echo.
echo Filter log-file
echo findLog.cmd logFile text
echo Example
echo findLog.cmd 20200515.log gbpjpy
echo Result
echo 20200515-gbpjpy.log
echo.
pause
:Exit
```

### Log highlighting

A lot of information is written to log when I perform debugging. Analyzing it in a running Terminal can be quite tricky. Here is what you can do to simplify this process:

- Clear the log in the Toolbox/Experts tab
- Stimulate printing to log
- Copy all lines to the log.txt file
- Analyze this file in Notepad++ (NPP) using highlighting

NPP supports different text highlighting variants

- Double click on a word highlights this word throughout the entire text
- Search/Mark/Using style - this applies the selected style to the required text throughout the document
- Custom syntax with the highlighting of specific tokens

### Contextual search

To find a specific text, I use the [TotalCommander](https://www.mql5.com/go?link=http://www.ghisler.com/ "http://www.ghisler.com/") file manager. The main advantages of this method are as follows:

- Search is only performed in project files (all other files do not appear in search results)
- Ability to use regular expressions in search

![Contextual file search](https://c.mql5.com/2/42/context-search__4.png)

### Conclusion

This is my first article. I am planning to continue the series. Next time I will share some successful solutions, providing the relevant description and explanation. I invite everyone to share their programming experience and to discuss other useful tips and solutions.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/9266](https://www.mql5.com/ru/articles/9266)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Tips from a professional programmer (Part III): Logging. Connecting to the Seq log collection and analysis system](https://www.mql5.com/en/articles/10475)
- [Tips from a professional programmer (Part II): Storing and exchanging parameters between an Expert Advisor, scripts and external programs](https://www.mql5.com/en/articles/9327)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/370090)**
(19)


![mktr8591](https://c.mql5.com/avatar/avatar_na2.png)

**[mktr8591](https://www.mql5.com/en/users/mktr8591)**
\|
20 May 2021 at 15:40

**Andrey Khatimlianskii:**

I have AkelPad. I open the file - Save as - UTF-8.

After that MetaEditor does not change the encoding itself.

Thanks, it works OK.


![TeeCee69](https://c.mql5.com/avatar/avatar_na2.png)

**[TeeCee69](https://www.mql5.com/en/users/teecee69)**
\|
26 May 2021 at 16:11

**MetaQuotes:**

New article [Tips from a professional programmer (Part I): Code storing, debugging and compiling. Working with projects and logs](https://www.mql5.com/en/articles/9266) has been published:

Author: [Malik Arykov](https://www.mql5.com/en/users/armas "armas")

Good morning I've been searching all over the Internet for mt5 assistants customer service but I have yet to find anything helpful I made a real account and I have $6,000 in there that I have been trying to withdraw but it seems impossible if you could point me in the right direction I would really appreciate it. My email is TcChrltn@aol.com  and my name is Terry. Attached are some files that might be helpful.

![Simon Anderson Hamonangan](https://c.mql5.com/avatar/2020/5/5EB3F7D5-F068.jpg)

**[Simon Anderson Hamonangan](https://www.mql5.com/en/users/f4xtrd)**
\|
26 May 2021 at 16:44

Thank you for sharing your insight. It always refreshing to see other programmers perspective.

I already do some of this advice. This is one of my routine :

On weekend, I usually clear unnecessary log file and file created in MQL/files directory (example: spread monitor file created during the week) using .bat file with [delete command](https://www.mql5.com/en/articles/7463 "Article: SQLite: Native handling of SQL databases in MQL5 ").

After that, I compress the MT4/MT5 folder and proceed to copy the compressed file to 2 of my external drive/usb drive (this option is quicker rather than upload the backup into some online backup service).

![tyup](https://c.mql5.com/avatar/avatar_na2.png)

**[tyup](https://www.mql5.com/en/users/tyup)**
\|
12 Jul 2021 at 05:07

**Andrey Khatimlianskii:**

I have AkelPad. I open the file - Save as - UTF-8.

After that MetaEditor does not change the encoding itself.

You can do it in metaeditore itself. [Open file](https://www.mql5.com/en/docs/files/fileopen "MQL5 documentation: FileOpen function") \- Save as UTF-8

![mktr8591](https://c.mql5.com/avatar/avatar_na2.png)

**[mktr8591](https://www.mql5.com/en/users/mktr8591)**
\|
13 Jul 2021 at 11:20

**tyup:**

You can do it in metaeditore itself. [Open file](https://www.mql5.com/en/docs/files/fileopen "MQL5 documentation: FileOpen function") \- Save as - UTF-8

Thank you! Maybe this option has recently appeared in ME?

![Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://c.mql5.com/2/42/MQL5-avatar-doeasy-library__8.png)[Other classes in DoEasy library (Part 70): Expanding functionality and auto updating the chart object collection](https://www.mql5.com/en/articles/9293)

In this article, I will expand the functionality of chart objects and arrange navigation through charts, creation of screenshots, as well as saving and applying templates to charts. Also, I will implement auto update of the collection of chart objects, their windows and indicators within them.

![Combination scalping: analyzing trades from the past to increase the performance of future trades](https://c.mql5.com/2/42/logo_01.png)[Combination scalping: analyzing trades from the past to increase the performance of future trades](https://www.mql5.com/en/articles/9231)

The article provides the description of the technology aimed at increasing the effectiveness of any automated trading system. It provides a brief explanation of the idea, as well as its underlying basics, possibilities and disadvantages.

![Swaps (Part I): Locking and Synthetic Positions](https://c.mql5.com/2/42/33201.png)[Swaps (Part I): Locking and Synthetic Positions](https://www.mql5.com/en/articles/9198)

In this article I will try to expand the classic concept of swap trading methods. I will explain why I have come to the conclusion that this concept deserves special attention and is absolutely recommended for study.

![MVC design pattern and its possible application](https://c.mql5.com/2/42/MVC.png)[MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)

The article discusses a popular MVC pattern, as well as the possibilities, pros and cons of its usage in MQL programs. The idea is to split an existing code into three separate components: Model, View and Controller.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=letsieyxeynsssrxeacwjlagozrnakws&ssn=1769252696519486296&ssn_dr=0&ssn_sr=0&fv_date=1769252696&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F9266&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Tips%20from%20a%20professional%20programmer%20(Part%20I)%3A%20Code%20storing%2C%20debugging%20and%20compiling.%20Working%20with%20projects%20and%20logs%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925269625238103&fz_uniq=5083321849338665304&sv=2552)

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