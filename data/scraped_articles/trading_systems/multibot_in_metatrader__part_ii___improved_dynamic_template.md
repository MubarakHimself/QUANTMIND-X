---
title: Multibot in MetaTrader (Part II): Improved dynamic template
url: https://www.mql5.com/en/articles/14251
categories: Trading Systems, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:14:07.543672
---

[![](https://www.mql5.com/ff/si/fx5m8s6u6uxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ysducdhemkrdsdtzzbfkclolrllnhezk&s=33f180a31db6c3b846d77732b0bc78169421a47b8cf9f076ca717f4e4846d1c7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=orsgfnsbfibunvqljxdbjqrtoxifbvsv&ssn=1769184846819038887&ssn_dr=0&ssn_sr=0&fv_date=1769184846&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14251&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Multibot%20in%20MetaTrader%20(Part%20II)%3A%20Improved%20dynamic%20template%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918484613237788&fz_uniq=5070121589476888653&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/14251#para1)
- [Dynamic template concept](https://www.mql5.com/en/articles/14251#para2)
- [Basic dynamic template settings](https://www.mql5.com/en/articles/14251#para3)
- [Rules for naming files with settings and applied directives](https://www.mql5.com/en/articles/14251#para4)
- [Methods for reading files and creating them](https://www.mql5.com/en/articles/14251#para5)
- [Creating virtual charts and EAs](https://www.mql5.com/en/articles/14251#para6)
- [Dynamic reading and reconfiguration of virtual EAs](https://www.mql5.com/en/articles/14251#para7)
- [Auto generation of magic numbers](https://www.mql5.com/en/articles/14251#para8)
- [Volume normalization system](https://www.mql5.com/en/articles/14251#para9)
- [Auto lot](https://www.mql5.com/en/articles/14251#para10)
- [Synchronization with API](https://www.mql5.com/en/articles/14251#para11)
- [General method with trading logic](https://www.mql5.com/en/articles/14251#para12)
- [Graphical interface](https://www.mql5.com/en/articles/14251#para13)
- [Conclusion](https://www.mql5.com/en/articles/14251#para14)
- [Links](https://www.mql5.com/en/articles/14251#para15)

### Introduction

In the last article, I was inspired by some of the most popular solutions from the Market, and I was able to create my own version of such a template. But, given some of the projects I am working on, it turns out that this solution is not optimal. In addition, it still has a number of limitations and inconveniences associated with the overall architecture of such a template. Such a template may be sufficient for the vast majority of average solutions, but not for mine. The second very important point is the fact that from the point of view of a potential EA buyer and end user, I personally would like to see maximum simplicity and a minimum number of settings. Ideally, such an EA should require no need for re-optimization and other manipulations on user's part. After all, I pay money not only for a working solution, but, more importantly, for an interface that is as friendly as possible and understandable to everyone.

### Dynamic template concept

Based on the previous EA, let's recall why such a template was created. The main goal was the ability to launch one EA with different settings for each "instrument - period" pair. This was necessary in order not to launch an EA on each chart separately. This simplified the management of such a system and its configuration, because the more copies of the same EA in one terminal, the higher the likelihood of a potential user’s error. In addition, conflicts could arise between such EAs due to, say, magic number order or for other reasons based on the human factor. Let's first describe the pros and cons of the first template. Then I will do the same for the new template, so that the differences and advantages of the new approach are clearly visible.

**Static (base) template pros:**

1. Individual setting of each instrument - period (virtual EA on a virtual chart).
2. Simpler and faster code (will execute faster).
3. The approach prevalence (widely represented in the market and has proven its viability).

**Static (base) template cons:**

1. Setup complexity and high probability of error (long string chains in settings).
2. Limited possibilities for expanding functionality.
3. Inability to add or remove virtual robots without manual reconfiguration.

Understanding these data, we can see that the solution is very limited, although widespread. It is possible to come up with a much more interesting solution. Additionally, integrating my EA into my solution, which performs independent dynamic optimization outside the MetaTrader terminal, pushed me towards an improved solution. Let's start with the diagram that reveals the essence of the new template and helps us better understand its advantages as compared to the previous version.

![new template usage](https://c.mql5.com/2/69/1.png)

Here, please pay attention to the lower left element containing CLOSING. The template provides for saving the settings, at which the last position was opened on the corresponding virtual chart. This is a special protective mechanism against frequent changes of settings and strategy breakdown. If we have opened positions according to some logic, we are interested in closing it according to the same logic, and only after that can we change the setting to a more recent one. Otherwise, we could end up with an absolute mess. Yes, this is not always the case, but it is better to immediately block such moments at the general logic level.

The new template uses the capabilities and structure of creating working directories of MetaTrader 4 and MetaTrader 5 together with the file system. In other words, we manage our template using text files that can contain the necessary settings for each virtual trading pair. This ensures maximum ease of setup and dynamic restructuring of the EA during operation without the need to manually reconfigure it. But that is not all. This also provides for a number of benefits I will list below.

**Dynamic (new) template pros:**

1. Each instrument-period is configured using text files regardless of the trading terminal.
2. Dynamic reading of settings from folders and automatic opening or closing of virtual charts occur together with their EAs.
3. It is possible to automatically synchronize with the Web API (required via port 443).
4. Configuration takes place using the terminal common folder (\*\\Common\\Files).
5. The leading EA is needed to synchronize all terminals on one machine with the API. It can also potentially work on several machines via a shared folder within the local network.
6. It is possible to integrate with external programs that can also manage these files, for example, create or delete them, as well as change settings.
7. It is possible to create a pair of paid and free templates. If we cut out the connection to the API, we can design this as a demo version that requires a paid version to work effectively.

**Dynamic (new) template cons:**

1. Working through the file system

The disadvantage here is also very conditional, because the only alternative is web integration or communication using some other methods (for example, through RAM), but in this case our trump card is that we do not use any additional libraries, and our code becomes cross-platform (suitable for both MQL4 and MQL5 EAs). Thanks to this, such EAs comply with the Market requirements and can easily be put up for sale if desired. Of course, you will have to add some things and adjust them according to your needs, but this will be extremely easy to do using my insights provided here.

Let's have a look at the inner workings of such an EA, as well as consider how we can visually and schematically represent the things I mentioned above. I think, this can be done using the following diagram:

![simplified template handling structure](https://c.mql5.com/2/69/2.png)

This whole scheme works quite simply, with the help of two independent timers, each responsible for its own tasks. The first timer uploads the settings from the API, while the second one reads the same settings into the EA's memory. Additionally, two arrows are shown, which symbolize the possibility of manually creating and changing these settings, or automating this process using a third-party application or other code. Theoretically, this template can be controlled from any other code, such as from another MetaTrader 4 and MetaTrader 5 EA or script.

After reading the settings, it is determined whether these settings are new, or whether there are new settings in the folder, or perhaps we have deleted some of them. If the comparison shows that there are no changes in the files, the template continues to work, but if something has changed, then it would be much more correct to restart the entire code with the new configuration and continue working. For example, if we had an EA that works only with the chart it operates on, we would have to do all this manually inside the terminal. However, in this case, all control occurs completely automatically inside the template, without the need for manual intervention.

### Basic dynamic template settings

Now, we can move on to the code and analyze its main points. Let me first show you the minimum set of inputs, which, in my opinion, are sufficient to manage trading using the bar-by-bar logic. But before doing this, it is necessary to mention the important paradigms this pattern uses in order to better understand its code in the future.

- Trading operations and calculations of input signals occur when a new bar opens (this point is calculated automatically based on ticks for each virtual chart separately).
- Only one position can be opened on one chart at a time (EURUSD M1 and EURUSD M5 are considered different charts).
- A new position on a separate chart cannot be opened until the previous one is closed (I consider this structure to be the simplest and most correct, because all kinds of additional purchases and averaging are already additional weights to a certain extent).

Now, taking into account these rules and restrictions, we need to understand that we can modify this code to work with averaging or martingale, or, for example, to handle pending orders. You can modify its structure yourself if necessary. Now let's look at our minimum set of parameters that will allow us to manage trading.

```
//+------------------------------------------------------------------+
//|                         main variables                           |
//+------------------------------------------------------------------+
input bool bToLowerE=false;//To Lower Symbol
input string SymbolPrefixE="";//Symbol Prefix
input string SymbolPostfixE="";//Symbol Postfix

input string SubfolderE="folder1";//Subfolder In Files Folder
input bool bCommonReadE = true;//Read From Common Directory

input bool bWebSyncE = false;//Sync with API
input string SignalDirectoryE = "folder1";//Signal Name(Folder)
input string ApiDomen = "https://yourdomen.us";//API DOMEN (add in terminal settings!)

input bool bInitLotControl=true;//Auto Lot
input double DeltaBarPercent=1.5;//Middle % of Delta Equity Per M1 Bar (For ONE! Bot)
input double DepositDeltaEquityE=100.0;//Deposit For ONE! Bot

input bool bParallelTradingE=true;//Parallel Trading

input int SLE=0;//Stop Loss Points
input int TPE=0;//Take Profit Points
```

In this example, I divided the variables into different blocks based on similar characteristics. As we can see, there are not many variables. The first block is designed to handle different styles of naming instruments for different brokers. For example, if your broker has "EURUSD", then all the variables from this block should be as they are now, but other options are also possible, for example:

- eurusd
- EURUSDt
- tEURUSD
- \_EURUSD
- eurusd\_

And so on. I will not continue further, I think you can figure it out on your own. Of course, a good template should handle most of these variations.

The second subblock tells us from which folder we are reading files. Besides, we are loading data from the API into the same folder. If we specify the correct folder name, MetaTrader will create the corresponding subdirectory and work with it. If we do not specify, all files will be located in the root. An interesting feature is provided by the shared terminal folder: if we enable this option, we can combine not only several EAs running inside MetaTrader 5, but also all terminals, regardless of whether they run on MetaTrader 4 or MetaTrader 5. This way, both templates will be able to work identically using the same settings, ensuring maximum integration and synchronization.

As you might have already guessed, the third subblock enables/disables the synchronization timer with your API, if there is one. It is important to note that communication with the API is carried out using only the WebRequest function, which works in both MQL4 and MQL5. The only limitation here is that your API should run on port 443. However, in MQL5 this method has been expanded, and it has the ability to connect through a different port. However, I abandoned this idea in my API in order to ensure that my templates and solutions built on top of it are cross-platform.

I built the API so that the signal name was also the file directory. This way, I am able to connect to different signals by knowing their names. We can disconnect from the old signal and connect to the new one at any time. Of course, you cannot download the files themselves this way, but I did it a little differently. I get JSON with the file contents, and then create it myself from the code of the template itself. This requires additional methods to extract the data from your JSON string, but personally this did not create any problems for me, because MQL5 allows us to easily do this. Of course, before using our API, we will need any domain and add it to the list of allowed connections in the MetaTrader settings.

The fourth block is very important, because it controls the risks and entry volumes. Of course, we are free to set up volumes for each instrument-period separately using the same text files, but I decided to make automatic settings using the volatility data of the trading pair used. This approach also involves an automatic proportional increase in volumes along with the growth of our balance - this is called an auto lot.

The fifth block consists of only one variable. By default, all virtual EAs that work inside the template trade independently and do not pay attention to positions opened by EAs that are on other virtual charts. This variable makes sure that only one position can be opened in one EA, others wait until it closes, and only after that they can open their own. In some cases, this trading style can be extremely useful.

The last (fifth) block contains only stop settings. If we set them to zero, then we trade without them. In the sixth block, which does not exist yet, you can add your parameters if necessary.

### Rules for naming files with settings and applied directives

I would like to start with the main methods that upload settings from our files, and how they are read, as well as how these methods find exactly the files we need. To ensure that such files are read correctly, we first need to introduce rules for naming them that are simple and understandable. First we need to find this directive in our template:

```
//+------------------------------------------------------------------+
//|                     your bot unique name                         |
//+------------------------------------------------------------------+
#define BotNick "dynamictemplate" //bot
```

This directive sets a bot nickname. Absolutely everything happens under this nickname:

1. Naming newly created files,
2. Reading files,
3. Creating terminal global variables,
4. Reading terminal global variables,
5. Other.

Our template will only accept "\*\*\*\* dynamictemplate.txt" files as its settings. In other words, we have defined the first rule for naming files with settings - the file name before its extension should always end in "dynamictemplate". You are free to change this name to any one you like. Thus, if you create two EAs with different aliases, they will safely ignore the settings of their "sibling" and will only work with their own files.

Nearby is another similar directive that achieves the same thing:

```
//+------------------------------------------------------------------+
//|               unique shift for difference of EA                  |
//+------------------------------------------------------------------+
#define MagicHelp 0 //bot magic shift
```

The only difference is that this directive ensures the distinction between different EAs at the order magic level, just like with files, so that two or more EAs do not close the orders of other EAs, if you suddenly decide to use several such EAs within one trading account. In the same way, when creating the next EA, we should also change this number along with changing the alias. Just do not make it too large, it is better to just add one to this number every time we create a new EA based on this template.

Next we move on to the part of the file name that will tell our EA which chart it is intended for. But first, let's have a look at this array:

```
//+------------------------------------------------------------------+
//|                        applied symbols                           |
//+------------------------------------------------------------------+
string III[] = {
   "EURUSD",
   "GBPUSD",
   "USDJPY",
   "USDCHF",
   "USDCAD",
   "AUDUSD",
   "NZDUSD",
   "EURGBP",
   "EURJPY",
   "EURCHF",
   "EURCAD",
   "EURAUD",
   "EURNZD",
   "GBPJPY",
   "GBPCHF",
   "GBPCAD",
   "GBPAUD",
   "GBPNZD",
   "CHFJPY",
   "CADJPY",
   "AUDJPY",
   "NZDJPY",
   "CADCHF",
   "AUDCHF",
   "NZDCHF",
   "AUDCAD",
   "NZDCAD",
   "AUDNZD",
   "USDPLN",
   "EURPLN",
   "USDMXN",
   "USDZAR",
   "USDCNH",
   "XAUUSD",
   "XAGUSD",
   "XAUEUR"
};
```

This array has a very important file filtering function. In other words, the template will load only those charts and the corresponding settings that are present in this list of symbols. Here we are obliged to fix a couple more rules that work equally both for naming files with settings and for adjusting the specified list.

1. All instrument names are converted to uppercase.
2. All instrument postfixes and prefixes are removed, leaving only the true instrument names.

Let's look at the example now. Suppose that the name of the "EURUSD" symbol has the following look at your broker: "eurusd\_". This means that you still name the settings file using "EURUSD", but additionally do the following in the settings:

```
//+------------------------------------------------------------------+
//|                        symbol correction                         |
//+------------------------------------------------------------------+
input bool bToLowerE=true;//To Lower Symbol
input string SymbolPrefixE="";//Symbol Prefix
input string SymbolPostfixE="_";//Symbol Postfix
```

In other words, you signal to the template that inside the EA our names will be converted to their original by converting the names to lowercase and adding the appropriate postfix. This will not happen often, as brokers generally stick to the classic uppercase naming scheme without using any prefixes or postfixes, simply because it does not make any sense at all. But still this option is not excluded.

For now, we have only figured out how to name the file so that the template understands that this is the necessary "BotNick" setting, and how to match it with the correct trading instrument. We still need to match the setting to a specific chart period. To do this, I have introduced the following rule:

- After the file name, put a space and write the equivalent of this period in minutes.
- The allowed chart period range is M1... H4.

I think it is important to specifically list the periods that are in this range. It was very important for me that all these periods were represented in both MetaTrader 4 and MetaTrader 5, this was the main reason for choosing the periods. Another very important reason was that very high periods above "H4" are not generally used in automated bar trading. In any case, I have not seen such examples, so the choice fell on the following periods:

- M1 - 1 minute
- M5 - 5 minutes
- M15 - 15 minutes
- M30 - 30 minutes
- H1 - 60 minutes
- H4 - 240 minutes

These periods are quite sufficient. Besides, for each of these periods it is very easy to calculate their equivalent in minutes, and these numbers are very easy to remember. Now we can show the general structure of the file name, according to which you will create settings manually or automatically using third-party code or your API. First let's take a look at the general outline:

- "INSTRUMENT" + " " + "PERIOD IN MINUTES" + " " + BotNick + ".txt"

Additionally, let's have a look at the structure the template will use to duplicate the files intended for closing positions:

- "CLOSING" + " " + "INSTRUMENT" + " " + "PERIOD IN MINUTES" + " " + BotNick + ".txt"

As you can see, these two files will differ only in the addition of "CLOSING" and the subsequent separating space. All signal lines in the name are, of course, separated by a single space so that the template interpreter can recognize and extract these markers from the file name. Thus, whether a setting belongs to a particular chart is determined only by its name. Let's now look at a few examples of settings that follow this rule:

- EURUSD 15 dynamictemplate.txt
- GBPUSD 240 dynamictemplate.txt
- EURCHF 60 dynamictemplate.txt
- CLOSING GBPUSD 240 dynamictemplate.txt

Obviously, this is sufficient for an example. Please pay attention to the last name. This file will be copied based on "GBPUSD 240 dynamictemplate.txt" and placed in the folder of the exact terminal, in which the EA made the copy. This is done in order to prevent multiple writing to the same files by different but identical EAs within several terminals. If we disable the option to read from the terminal shared folder, then regular files will be written there too. This may be necessary if we need to configure each specific EA in the corresponding terminal with its own independent number of settings. I will leave several files next to the templates as an example, so that it is clearer with a specific example, and so that you can experiment with them, moving them to different folders. This concludes the consideration of general aspects of using the settings.

### Methods for reading files and creating them

In order to fully master the template functionality, it is advisable to understand how reading and writing files work. This, ultimately, will allow us to start using them not only as markers of instrument-periods we want to attach virtual robots to, but also to individually customize each of them if desired. To do this, we can start to consider the following method.

```
//+------------------------------------------------------------------+
//|                 used for configuration settings                  |
//+------------------------------------------------------------------+
bool QuantityConfiguration()
{
    FilesGrab(); // Determine the names of valid files

    // Check if there are changes in the configuration settings (either add or delete)
    if (bNewConfiguration())
    {
        return true;
    }
    return false;
}
```

This method determines whether the set of files in our working directory has been updated. If yes, then we use this method as a signal to restart all charts and EAs in order to add new or remove unnecessary instrument-periods. Now let's have a look at the FilesGrab method.

```
//+------------------------------------------------------------------+
//|   reads all files and forms a list of instruments and periods    |
//+------------------------------------------------------------------+
void FilesGrab()
   {
   string file;
   string tempsubfolder= SubfolderE == "" ? ""  : SubfolderE + "\\"; // SubfolderE is the path to the specific subfolder
   // Returns the handle of the first found file with the specified characteristics, based on whether CommonReadE is True or False
   long total_files = !bCommonReadE? FileFindFirst(tempsubfolder+"*"+BotNick+".txt", file) :FileFindFirst(tempsubfolder+"*"+BotNick+".txt", file,FILE_COMMON);
   if(total_files > 0)
      {
         ArrayResize(SettingsFileNames,0); // Clear the array from previous values if there are files to be read
         do
         {
            int second_space = StringFind(file, " ", StringFind(file, " ") + 1); // Searches for the index of the second space in the file's name
            if(second_space > 0)
            {
                string filename = StringSubstr(file, 0, second_space); // Extracts the string/characters from the filename up to the second space
                ArrayResize(SettingsFileNames, ArraySize(SettingsFileNames) + 1); // Increases the size of the array by one
                SettingsFileNames[ArraySize(SettingsFileNames) - 1] = filename; // Adds the new filename into the existing array
            }
         }
         while(FileFindNext(total_files, file)); // Repeat for all the files
         FileFindClose(total_files); // Close the file handle to free resources
      }
   }
```

This method performs a preliminary collection of the names of the files that relate to our EA into something like "EURUSD 60". In other words, it leaves only that part of the name that will later be parsed into an instrument-period pair. The reading of these files, however, does not take place here, but within each virtual EA separately. But to do this, we first need to parse the string itself into a symbol and a period. This is preceded by several points. One of them is as follows.

```
//+------------------------------------------------------------------+
//|                        symbol validator                          |
//+------------------------------------------------------------------+
bool AdaptDynamicArrays()
{
    bool RR=QuantityConfiguration();
    // If a new configuration of files is detected (new files, changed order, etc.)
    if (RR)
    {
        // Read the settings (returns the count)
        int Readed = ArraySize(SettingsFileNames);
        int Valid =0;

        // Only valid symbol name needs to be populated (filenames are taken from already prepared array)
        ArrayResize(S, Readed);

        for ( int j = 0; j < Readed; j++ )
        {
            for ( int i = 0; i < ArraySize(III); i++ )
            {
                // check the symbol to valid
                if ( III[i] == BasicNameToSymbol(SettingsFileNames[j]) )
                {
                    S[Valid++]=SettingsFileNames[j];
                    break; // stop the loop
                }
            }
        }
        //resize S with the actual valid quantity
        ArrayResize(S, Valid);
        return true;
    }
    return false;
}
```

This method is very important in order to discard the settings (charts) that are not present in our list of allowed instruments. The point is to ultimately add all the charts that have been filtered by the list into the "S" array, preparing it for further use in the code to create virtual chart objects.

An important point is also the reservation of settings, which occurs constantly as the basic settings are read. If a base setting position has opened, then we stop periodical settings reservation. The backup file with the "CLOSING" prefix is always saved in the current terminal directory.

```
//+------------------------------------------------------------------+
//|         сopy settings from the main file to a CLOSING file       |
//+------------------------------------------------------------------+
void SaveCloseSettings()
   {
   string FileNameString=Charts[chartindex].BasicName;
   bool bCopied;
   string filenametemp;
   string filename="";
   long handlestart;

   //Checking if SubfolderE doesn't exist, if yes, assign tempsubfolder to be an empty string
   string tempsubfolder= SubfolderE == "" ? ""  : SubfolderE + "\\";

   //Find the first file in the subfolder according to bCommonReadE and assign the result to handlestart
   if (bCommonReadE) handlestart=FileFindFirst(tempsubfolder+"*",filenametemp,FILE_COMMON);
   else handlestart=FileFindFirst(tempsubfolder+"*",filenametemp);

   //Check if the start of our found file name matches FileNameString
   if ( StringSubstr(filenametemp,0,StringLen(FileNameString)) == FileNameString )
      {
      //if yes, complete the file's path
      filename=tempsubfolder+filenametemp;
      }
     //keep finding the next file while conditions are aligned
   while ( FileFindNext(handlestart,filenametemp) )
      {
      //if found file's name matches FileNameString then add found file's name to the path
      if ( StringSubstr(filenametemp,0,StringLen(FileNameString)) == FileNameString )
         {
         filename=tempsubfolder+filenametemp;
         break;
         }
      }
   //if handlestart is not INVALID_HANDLE then close the handle to release the resources after the search
   if (handlestart != INVALID_HANDLE) FileFindClose(handlestart);

   //Perform file copy operation and notice if it was successful
   if ( bCommonReadE ) bCopied=FileCopy(filename,FILE_COMMON,tempsubfolder+"CLOSING "+FileNameString+".txt",FILE_REWRITE|FILE_TXT|FILE_ANSI);
   else bCopied=FileCopy(filename,0,tempsubfolder+"CLOSING "+FileNameString+".txt",FILE_REWRITE|FILE_TXT|FILE_ANSI);
   }
```

Here it is worth clarifying that the backup setting works in such a way that, for example, when the template is restarted or read again, the data will be read from it. This is only possible with an open position that corresponds to this setting. If a specific instance of a virtual EA does not have open positions, then the template is synchronized with the general setting.

### Creating virtual charts and EAs

Then we retrieve all the data from there in the next method, while simultaneously creating the necessary virtual charts.

```
//+------------------------------------------------------------------+
//|                      creates chart objects                       |
//+------------------------------------------------------------------+
void CreateCharts()
   {
   bool bAlready;
   int num=0;
   string TempSymbols[];
   string Symbols[];
   ArrayResize(TempSymbols,ArraySize(S)); // Resize TempSymbols array to the size of S array
   for (int i = 0; i < ArraySize(S); i++) // Populate TempSymbols array with empty strings
      {
      TempSymbols[i]="";
      }
   for (int i = 0; i < ArraySize(S); i++) // Count the required number of unique trading instruments
      {
      bAlready=false;
      for (int j = 0; j < ArraySize(TempSymbols); j++)
         {
         if ( S[i] == TempSymbols[j] ) // If any symbol is already present in TempSymbols from S, then it's not unique
            {
            bAlready=true;
            break;
            }
         }
      if ( !bAlready ) // If the symbol is not found in TempSymbols i.e., it is unique, add it to TempSymbols
         {
         for (int j = 0; j < ArraySize(TempSymbols); j++)
            {
            if ( TempSymbols[j] == "" )
               {
               TempSymbols[j] = S[i];
               break;
               }
            }
         num++; // Increments num if a unique element is added
         }
      }
   ArrayResize(Symbols,num); // Resize the Symbols array to the size of the num

   for (int j = 0; j < ArraySize(Symbols); j++) // Now that the Symbols array has the appropriate size, populate it
      {
      Symbols[j]=TempSymbols[j];
      }
   ArrayResize(Charts,num); // Resize Charts array to the size of num

   int tempcnum=0;
   tempcnum=1000; // Sets all charts to a default of 1000 bars
   Chart::TCN=tempcnum;
   for (int j = 0; j < ArraySize(Charts); j++)
      {
      Charts[j] = new Chart();
      Charts[j].lastcopied=0; // Initializes the array position where the last copy of the chart was stored
      Charts[j].BasicName=Symbols[j];
      ArrayResize(Charts[j].CloseI,tempcnum+2); // Resizes the CloseI array to store closing price of each bar
      ArrayResize(Charts[j].OpenI,tempcnum+2); // Resizes the OpenI array for opening prices
      ArrayResize(Charts[j].HighI,tempcnum+2); // HighI array for high price points in each bar
      ArrayResize(Charts[j].LowI,tempcnum+2); // LowI array for low price points of each bar
      ArrayResize(Charts[j].TimeI,tempcnum+2); // TimeI array is resized to store time of each bar
      string vv = BasicNameToSymbol(Charts[j].BasicName);
      StringToLower(vv);
      // Append prefix and postfix to the basic symbol name to get the specific symbol of the financial instrument
      Charts[j].CurrentSymbol = SymbolPrefixE +  (!bToLowerE ? BasicNameToSymbol(Charts[j].BasicName) : vv) + SymbolPostfixE;
      Charts[j].Timeframe = BasicNameToTimeframe(Charts[j].BasicName); // Extracts the timeframe from the basic name string
      }
   ArrayResize(Bots,ArraySize(S)); // Resize Bots array to the size of S array
   }
```

This method focuses on creating a collection of non-repeating instrument-periods, based on which objects of the corresponding charts are created. In addition, please pay attention to the fact that the size of the bars array for each chart is set to just over 1000 bars. I believe that this is more than enough to implement most strategies. If something happens, we can change this quantity to the required one. Now let's consolidate the material with a method that creates virtual EA objects.

```
//+------------------------------------------------------------------+
//|              attaching all virtual robots to charts              |
//+------------------------------------------------------------------+
void CreateInstances()
   {
   // iterating over the S array
   for (int i = 0; i < ArraySize(S); i++)
      {
      // iterating over the Charts array
      for (int j = 0; j < ArraySize(Charts); j++)
         {
         // checking if the BasicName of current Chart matches with the current item in S array
         if ( Charts[j].BasicName == S[i] )
            {
            // creating a new Bot instance with indices i, j and assigning it to respective position in Bots array
            Bots[i] = new BotInstance(i,j);
            break;
            }
         }
      }
   }
```

Here virtual EAs are created and attached to the corresponding charts, storing the ID of this "j" chart inside the EA, so that in the future we know which chart to take data from within the virtual EA. As for the internal structure of these two classes, I mentioned this in the previous article. In many ways, the new code is similar to it, except for a few insignificant changes.

### Dynamic reading and reconfiguration of virtual EAs

Obviously, this section is extremely important for us, since this is a good half of the entire concept of the new template. After all, creating virtual charts and EAs is only half the work. It seems like we have figured out the re-creation of virtual charts at a minimal level, but this is not enough. It is advisable to figure out how to pick up new settings on the fly and instantly reconfigure the EA without interfering with the trading terminal. To solve this problem, a simple timer is used as illustrated in the diagrams at the beginning of the article.

```
//+------------------------------------------------------------------+
//|             we will read the settings every 5 minutes +          |
//+------------------------------------------------------------------+
bool bReadTimer()
   {
   if (  TimeCurrent() - LastTime > 5*60 + int((double(MathRand())/32767.0) * 60) )
      {
      LastTime=TimeCurrent();
      int orders=OrdersG();
      bool bReaded=false;
      if (orders == 0)  bReaded = ReadSettings(false,Charts[chartindex].BasicName);//reading a regular file
      else bReaded = ReadSettings(true,Charts[chartindex].BasicName);//reading file to close position
      if (orders == 0 && bReaded) SaveCloseSettings();//save settings for closing position
      return bReaded;
      }
   return false;
   }
```

As you can see, the timer is triggered once every "5" minutes, which means that new files will not be picked up instantly. But, in my opinion, this timing is quite enough to ensure dynamics. If this is not enough for you, you can reduce it to as low as one second. The only thing you should understand is that frequent use of file operations is not encouraged and this approach should be avoided if possible. In this code, pay attention to the ReadSettings method. It reads the required file (for each virtual EA separately) and then reconfigures the EA after reading. The method is designed in such a way that it can either read general settings (if there are no open positions in the selected virtual EA), or pause updating the settings and wait until the position is closed according to the settings by which this position was created.

```
//+------------------------------------------------------------------+
//|                        reading settings                          |
//+------------------------------------------------------------------+
bool BotInstance::ReadSettings(bool bClosingFile,string Path)
   {
   string FileNameString=Path;
   int Handle0x;
   string filenametemp;
   string filename="";
   long handlestart;

   string tempsubfolder= SubfolderE == "" ? ""  : SubfolderE + "\\";

   if (!bClosingFile)//reading a regular file
      {
      if (!bCommonReadE)
         {
         handlestart=FileFindFirst(tempsubfolder+"*",filenametemp);
         int SearchStart=0;

         if ( StringSubstr(filenametemp,SearchStart,StringLen(FileNameString)) == FileNameString )
            {
            filename=tempsubfolder+filenametemp;
            }
         if (filename != filenametemp || filename == "")
            {
            while ( FileFindNext(handlestart,filenametemp) )
               {
               if ( StringSubstr(filenametemp,SearchStart,StringLen(FileNameString)) == FileNameString )
                  {
                  filename=tempsubfolder+filenametemp;
                  break;
                  }
               }
            }
         if (handlestart != INVALID_HANDLE) FileFindClose(handlestart);// Release resources after search

         if (filename != "")
            {
            Handle0x=FileOpen(filename,FILE_READ|FILE_SHARE_READ|FILE_TXT|FILE_ANSI);

            if ( Handle0x != INVALID_HANDLE )//if the file exists
               {
               FileSeek(Handle0x,0,SEEK_SET);
               ulong size = FileSize(Handle0x);
               string str = "";
               for(ulong i = 0; i < size; i++)
                  {
                     str += FileReadString(Handle0x);
                  }
               if (str != "" && str != PrevReaded)
                  {
                  FileSeek(Handle0x,0,SEEK_SET);
                  //read the required parameters
                  ReadFileStrings(Handle0x);
                  //
                  FileClose(Handle0x);
                  LastRead = TimeCurrent();
                  RestartParams();
                  }
               else
                  {
                  FileClose(Handle0x);
                  }
               return true;
               }
            else
               {
               return false;
               }
            }
         }
      else
         {
         handlestart=FileFindFirst(tempsubfolder+"*",filenametemp,FILE_COMMON);
         int SearchStart=0;

         if ( StringSubstr(filenametemp,SearchStart,StringLen(FileNameString)) == FileNameString )
            {
            filename=tempsubfolder+filenametemp;
            }
         if (filename != filenametemp || filename == "")
            {
            while ( FileFindNext(handlestart,filenametemp) )
               {
               if ( StringSubstr(filenametemp,SearchStart,StringLen(FileNameString)) == FileNameString )
                  {
                  filename=tempsubfolder+filenametemp;
                  break;
                  }
               }
            }
         if (handlestart != INVALID_HANDLE) FileFindClose(handlestart);// Release resources after search

         if (filename != "")
            {
            Handle0x=FileOpen(filename,FILE_READ|FILE_SHARE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);

            if ( Handle0x != INVALID_HANDLE )//if the file exists
               {
               FileSeek(Handle0x,0,SEEK_SET);
               ulong size = FileSize(Handle0x);
               string str = "";
               for(ulong i = 0; i < size; i++)
                  {
                     str += FileReadString(Handle0x);
                  }
               if (str != "" && str != PrevReaded)
                  {
                  FileSeek(Handle0x,0,SEEK_SET);
                  //read the required parameters
                  ReadFileStrings(Handle0x);
                  //
                  FileClose(Handle0x);
                  LastRead = TimeCurrent();
                  RestartParams();
                  }
               else
                  {
                  FileClose(Handle0x);
                  }
               return true;
               }
            else
               {
               return false;
               }
            }
         }
      }
   else//reading a file to close a position
      {
      handlestart=FileFindFirst(tempsubfolder+"*",filenametemp);
      int SearchStart=8;//when the line starts with "CLOSING "

      if ( StringLen(filenametemp) >= (8 + StringLen(FileNameString)) && StringSubstr(filenametemp,0,8) == "CLOSING "
      && StringSubstr(filenametemp,SearchStart,StringLen(FileNameString)) == FileNameString )
         {
         filename=tempsubfolder+filenametemp;
         }
      if (filename != filenametemp || filename == "")
         {
         while ( FileFindNext(handlestart,filenametemp) )
            {
            if ( StringLen(filenametemp) >= (8 + StringLen(FileNameString)) && StringSubstr(filenametemp,0,8) == "CLOSING "
            && StringSubstr(filenametemp,SearchStart,StringLen(FileNameString)) == FileNameString )
               {
               filename=tempsubfolder+filenametemp;
               break;
               }
            }
         }
      if (handlestart != INVALID_HANDLE) FileFindClose(handlestart);// Release resources after search

      if (filename != "")
         {
         Handle0x=FileOpen(filename,FILE_READ|FILE_SHARE_READ|FILE_TXT|FILE_ANSI);

         if ( Handle0x != INVALID_HANDLE )//if the file exists
            {
            FileSeek(Handle0x,0,SEEK_SET);
            ulong size = FileSize(Handle0x);
            string str = "";
            for(ulong i = 0; i < size; i++)
               {
                  str += FileReadString(Handle0x);
               }
            if (str != "" && str != PrevReaded)
               {
               PrevReaded=str;
               FileSeek(Handle0x,0,SEEK_SET);
               //read the required parameters
               ReadFileStrings(Handle0x);
               //
               FileClose(Handle0x);
               LastRead = TimeCurrent();
               RestartParams();
               }
            else
               {
               FileClose(Handle0x);
               }
            return true;
            }
         else
            {
            return false;
            }
         }
      }
   return false;
   }
```

First, I want to emphasize that this method is specifically designed for reading two types of files. Depending on the bClosingFile marker passed, either the general setting or "for closing" is read. Each file reading consists of several stages:

1. Comparison of the contents of the previous read file and the current one;
2. If the content is different, we read our updated settings;
3. We restart our virtual EA with new settings, if required.

The method is built in such a way that resource cleaning and other actions are already thought out. We just have to implement the next method, which is called in the previous one. I tried to do everything in such a way as to save you from the hassle of these file operations so that you can concentrate as much as possible on writing exactly the reading code that you need. Reading is performed here.

```
//+------------------------------------------------------------------+
//|               read settings from file line by line               |
//+------------------------------------------------------------------+
void BotInstance::ReadFileStrings(int handle)
   {
   //FileReadString(Handle,0);

   }
```

There is no need to open or close the file here. All you have to do is read the file string by string and correctly add what you read into the appropriate variables. To do this, we can use both temporary variables and immediately write all the data into the variables that you will use as settings for your strategy. But I would recommend filling out the settings already in this method, which was intended for these purposes.

```
//+------------------------------------------------------------------+
//|                function to prepare new parameters                |
//+------------------------------------------------------------------+
void BotInstance::RestartParams()
{
   //additional code

   //
   MagicF=SmartMagic(BasicNameToSymbol(Charts[chartindex].BasicName), Charts[chartindex].Timeframe);
   CurrentSymbol=Charts[chartindex].CurrentSymbol;
   m_trade.SetExpertMagicNumber(MagicF);
}
```

There is no need to touch the last three strings, as they are mandatory. The most interesting thing there is the SmartMagic method, which is designed to automatically assign magic numbers to each of the virtual EAs. All we need to know at this stage is that we need to write this logic for reassigning the EA settings a little higher - into the empty block. If necessary, we can also work on recreating the indicators and everything else that may still be there.

### Auto generation of magic numbers

Without departing from the previous method, I want to immediately reveal to you a method for generating unique order IDs that ensure independent trading for all virtual EAs inside the template. For this I used the following method.

I assign a step of "10000", for example, between the nearest magic numbers. For each instrument, I first record its preliminary magic number, for example, "10000" or "70000". However, this is not enough, because the instrument also has a period. Therefore, I add one more number to this intermediate magic.

The simplest thing is to add the minute equivalent of these periods, just like in the file reading structure. This is how it is done.

```
//+------------------------------------------------------------------+
//|              Smart generation of magical numbers                 |
//|    (each instrument-period has its own fixed magic number)       |
//+------------------------------------------------------------------+
int BotInstance::SmartMagic(string InstrumentSymbol,ENUM_TIMEFRAMES InstrumentTimeframe)
{
   // initialization
   int magicbuild=0;

   // loop through the array
   for ( int i=0; i<ArraySize(III); i++ )
   {
      // check the symbol to assign a magic number
      if ( III[i] == InstrumentSymbol )
      {
          magicbuild=MagicHelp+(i+1)*10000;
          break; // stop the loop
      }
   }

   // add identifier for time frame
   magicbuild+=InstrumentTimeframe;
   return magicbuild;
}
```

This is where our additional shift of magic numbers appears to make magic number sets unique, albeit between different EAs inside the terminal.

```
//+------------------------------------------------------------------+
//|               unique shift for difference of EA                  |
//+------------------------------------------------------------------+
#define MagicHelp 0 //bot magic shift [0...9999]
```

In general, everything is quite simple. The large step between magic numbers provides quite a lot of shift options, which is more than enough to create the required number of EAs. The only condition for using this structure is not to exceed the number "9999". In addition, we need to make sure that the shift does not match our timeframe equivalents in minutes, since in this case there may be coincidences in the magic numbers of two different templates. In order not to think about such options, we can simply make a shift of little more than “240”, for example, “241”, “241\*2”, “241\*3”, “241\*N”.

To summarize this approach, we can see that this structure completely frees us from setting up magic numbers, which was one of the unspoken intermediate goals of this solution. The only drawback is the impossibility of connecting two or more independent virtual EAs, since their magic numbers will coincide ultimately leading to the interaction of these strategies and, as a result, to the breakdown of their logic. In fact, I do not know who might need such exotic move. Besides, this does not fall into the originally intended functionality. Perhaps I will add it in the next article if anyone is interested.

### Volume normalization system

If I have a template that is simple and easy to customize, then it is extremely important to choose the right method for setting trading volumes. It is very interesting that the final conclusions from my articles on probability theory, specifically from this article, helped me to carry out a simple and effective equalization of volumes. I decided to equalize the volumes, assuming that the average duration and the size of the absolute value of the final financial result of the position should be similar, which means that the only correct solution to such an equalization should be based on the following considerations.

The average rate of increase or decrease in the equity of the final trading chart should be provided equally by each of the EAs (independent instrument-period) included in the final assembly. All necessary values should be calculated without using data from those instrument-periods that do not appear in the list of our virtual charts. Volumes should be distributed not only in proportion to the number of EAs, but also in proportion to the deposit (auto lot).

To do this, it is important to immediately introduce the following definitions and equations. To begin with, I implemented the following parameters to adjust risks:

- DeltaBarPercent - percentage of DepositDeltaEquity,
- DepositDeltaEquity - deposit of one bot to calculate its acceptable equity delta for one M1 bar with an open position.

The terms might seem unclear first. So let me clarify. For convenience, we specify a deposit with which a separate virtual EA works, and then indicate in percentage what part of this deposit should increase or decrease (in the form of our equity) if we opened a position and if there was a movement from the top point " M1" bar to the bottom or vice versa.

The goal of our code is to automatically select the entry volumes taking into account our requirements. To do this, we will need additional mathematical quantities and equations based on them. I will not draw any conclusions, I will just provide them to you to explain the code:

- "Mb" - average size of bars in points on the selected history range of "bars" size on the EA's working chart,
- "Mb1" - average size of bars in points on the selected history range of "bars" size on the EA's working chart recalculated to M1,
- "Kb" - connection ratio between the average size of bars of the current chart and its "M1" equivalent,
- "T" - period of the selected chart reduced to a minute equivalent (just like we had in our files),
- "BasisI" - required average increase or decrease in the equity line in the deposit currency for the average size of the M1 candle on the selected instrument chart,
- "Basisb" - actual average increase or decrease in the equity line in the deposit currency for the average size of the M1 candle on the selected instrument chart for a trade of "1" lot size,
- "Lot" - selected lot (volume).

Now that I have listed all the quantities used in the calculation, let’s begin to analyze and comprehend it. To calculate the required lot, first of all, we should understand how the relationship between bar sizes on higher timeframes relative to “M1” is carried out. The following equation will help here.

![Scaling factor to go to M1](https://c.mql5.com/2/70/5.png)

This is exactly the expression that allows, without loading data from "M1", to calculate the same characteristic, albeit on the presented period of the virtual chart the selected instance of the virtual EA works with. After multiplying by this factor, we obtain almost the same data as if we calculated them on "M1" period. This is the first thing to do. The method for calculating this value will look like this:

```
//+------------------------------------------------------------------+
//|       timeframe to average movement adjustment coefficient       |
//+------------------------------------------------------------------+
double PeriodK(ENUM_TIMEFRAMES tf)
   {
   double ktemp;
   switch(tf)
      {
      case  PERIOD_H1:
          ktemp = MathSqrt(1.0/60.0);
          break;
      case  PERIOD_H4:
          ktemp = MathSqrt(1.0/240.0);
          break;
      case PERIOD_M1:
          ktemp = 1.0;
          break;
      case PERIOD_M5:
          ktemp = MathSqrt(1.0/5.0);
          break;
      case PERIOD_M15:
          ktemp = MathSqrt(1.0/15.0);
          break;
      case PERIOD_M30:
          ktemp = MathSqrt(1.0/30.0);
          break;
      default: ktemp = 0;
      }
   return ktemp;
   }
```

Now, of course, we need to understand what value we are adapting to "M1". Here it is.

![](https://c.mql5.com/2/70/3.png)

In other words, we calculate the average size of candles in points on the chart the virtual EA works with. After calculating the value, we should convert it using the previous value like this.

![](https://c.mql5.com/2/70/4.png)

Both of these actions occur in the following method.

```
//+------------------------------------------------------------------+
//|     average candle size in points for M1 for the current chart   |
//+------------------------------------------------------------------+
double CalculateAverageBarPoints(Chart &Ch)
   {
   double SummPointsSize=0.0;
   double MaxPointSize=0.0;
   for (int j = 0; j < ArraySize(Ch.HighI); j++)
      {
      if (Ch.HighI[j]-Ch.LowI[j] > MaxPointSize) MaxPointSize= Ch.HighI[j]-Ch.LowI[j];
      }

   for (int j = 0; j < ArraySize(Ch.HighI); j++)
      {
      if (Ch.HighI[j]-Ch.LowI[j] > 0) SummPointsSize+=(Ch.HighI[j]-Ch.LowI[j]);
      else SummPointsSize+=MaxPointSize;
      }
   SummPointsSize=(SummPointsSize/ArraySize(Ch.HighI))/Ch.ChartPoint;
   return PeriodK(Ch.Timeframe)*SummPointsSize;//return the average size of candles reduced to a minute using the PeriodK() adjustment function
   }
```

Now we can use the resulting value, reduced to M1, in order to calculate the "Basisb" variable. It should be calculated like this.

![](https://c.mql5.com/2/70/7.png)

The tick size is the amount of change in the equity of an open position with a volume of "1" lot when the price moves by "1" point. If we multiply it by the average size of a minute candle, we will get the amount of equity change for a position with a single lot taking into account that the size of the movement has become equal to the average size of the minute bar. Next, we should calculate the remaining value of "BasisI", and this is done like this.

![](https://c.mql5.com/2/70/8.png)

The percentage and deposit in it are exactly the control parameters, with which we require the basis that we need. All that remains to do is to select a proportionality ratio so that the bases are equal, and this ratio will be our final lot.

![](https://c.mql5.com/2/70/6.png)

All described operations are performed in the following method.

```
//+------------------------------------------------------------------+
//|    calculate the optimal balanced lot for the selected chart     |
//+------------------------------------------------------------------+
double OptimalLot(Chart &Ch)
   {
   double BasisDX0 =  (DeltaBarPercent/100.0) * DepositDeltaEquityE;
   double DY0=CalculateAverageBarPoints(Ch)*SymbolInfoDouble(Ch.CurrentSymbol,SYMBOL_TRADE_TICK_VALUE);
   return BasisDX0/DY0;
   }
```

Thus, we effectively equalize the volumes of all instruments for an equal contribution to the equity of all virtual EAs. It turns out to be something like a fixed lot mode, as if we had set it separately for each EA, but in this case we got rid of this need. This is a completely different risk control system, but its difference is that it is adapted to diversified trading, and using it, you will have to do it anyway if you want to achieve optimal stability of the profit curve and get rid of this routine. In fact, balancing several EAs is the most sensitive moment. I think, those who have done similar things will appreciate this decision. In any case, if this balancing method does not suit you, you can always write your settings in the appropriate files and modify this system.

### Auto lot

The normalized lot can be increased in proportion to the deposit. To understand how, I will introduce the following notations:

- \- Lot - normalized (balanced) lot for a specific virtual EA;
- \- AutoLot - recalculated to the "Lot" deposit for the auto lot mode (we need to receive it when the auto lot is enabled);
- \- DepositPerOneBot - part of the current deposit (part of Deposit), which can only be controlled by one of the virtual EAs;
- \- DepositDeltaEquity - deposit, against which we performed normalization (balanced lots);
- \- Deposit - our current bank;
- \- BotQuantity - current number of virtual EAs trading inside the multibot.

Then you can write what our "AutoLot" is equal to:

- AutoLot = Lot \* (DepositPerOneBot / DepositDeltaEquity).

It turns out that in the case of the usual normalized volume, we ignore our deposit and accept that the following deposit is allocated for one virtual EA - DepositDeltaEquity. But in case of the auto lot, this deposit is not real, and we should proportionally change the normalized volumes so that our risks adapt to the real deposit. However, it needs to be adapted taking into account that one virtual EA accounts for only a portion of the deposit.

- DepositPerOneBot = Deposit / BotQuantity.

This is how the auto lot works in my template. I think this approach is quite convenient and provides the necessary adjustment to the steepness of the curve exponential growth. You can find the source code in the attachment. Let's see the result of proper adjustment of these values.

![auto lot with normalized volumes](https://c.mql5.com/2/70/f6f8_ci0.png)

Please note that the profit curve in this mode will look approximately as shown here, provided that the settings are correct and the initial signal is profitable. This curve will be smoother and more exponential-like the higher the profit factor in your strategy and the more trades it has in your testing area. This is precisely what diversification achieves very well. Our template contains all the basics for maximizing this effect. Additionally, pay attention to the loading of the deposit: it is its smoothness and uniformity that indirectly indicates the correct normalization and subsequent scaling of the current deposit load. I took this example from my product based on the template in question.

### Synchronization with API

This feature is optional and can be very easily disabled or completely cut out of the template, but I personally found it very useful for my product. As mentioned earlier, synchronization is also triggered through the timer.

```
//+------------------------------------------------------------------+
//|            used to read the settings every 5 minutes +           |
//+------------------------------------------------------------------+
void DownloadTimer()
{
    // Check if the passed time from the last download time is more than 5 minutes
    if (  TimeCurrent() - LastDownloadTime > 5*60 + int((double(MathRand())/32767.0) * 60) )
    {
        // Set the last download time to the current time
        LastDownloadTime=TimeCurrent();
        // Download files again
        DownloadFiles();
     }
}
```

Now let's look at the main DownloadFiles method.

```
//+------------------------------------------------------------------+
//|       used to download control files if they isn't present       |
//+------------------------------------------------------------------+
void DownloadFiles()
{
    string Files[];
    // Initialize the response code by getting files from the signal directory
    int res_code=GetFiles(SignalDirectoryE,Files);

    // Check if the list of files is successfully got
    if (res_code == 200)
    {
        // Proceed if there is at least one file in the server
        if (ArraySize(Files) > 0)
        {
            // Download each file individually
            for (int i = 0; i < ArraySize(Files); i++)
            {
                string FileContent[];
                // Get the content of the file
                int resfile =  GetFileContent(SignalDirectoryE,Files[i],FileContent);

                // Check if the file content is successfully got
                if (resfile == 200)
                {
                    // Write the file content in our local file
                    WriteData(FileContent,Files[i]);
                }
            }
        }
    }
}
```

I have arranged the entire structure in such a way that the first step is to access the API in order to find out the entire list of files that lies in the specified folder on our server. It will distribute the settings. The folder name is SignalDirectoryE. This is also the name of the signal according to my idea. After receiving the list of files, each of them is downloaded separately. In my opinion, this construction logic is very convenient. This way we can create many signals (folders) we can switch between at any time. How to do and arrange this is up to you to decide. My task is to provide ready-made functionality for easy connection. Now let's look at the method template that gets a list of file names from our server.

```
//+------------------------------------------------------------------+
//|              getting the list of files into an array             |
//+------------------------------------------------------------------+
int GetFiles(string directory,string &fileArray[])
   {
   //string for getting a list of files in the form of JSON via GET to API
   string urlList = ApiDomen+"/filelist/"+directory;//URL
   char message[];//Body of the request
   string headers = "Password_key: " + key;// We form the headers of the request
   string resultheaders = "";//returning headers
   string cookie = "";//cookies
   int timeout = 1500;//waiting for a response when requesting a file or json
   char result[];

   // We send a GET request to the server to receive JSON with a list of files
   int res_code =  WebRequest("GET", urlList, headers, timeout, message, result, resultheaders);
   bool rez = extractFiles(CharArrayToString(result),fileArray);
   if (rez) return res_code;
   else return 400;
   }
```

All you have to do here is to form your "URL" string in the same way. The most important parts here are the following strings:

- filelist
- Password\_key

The first string is one of the functions of your API. You can change its name if you want. You will have several such functions. For example, to provide the following operations:

1. Uploading settings files to your API (from your individual application or program).
2. Clearing settings files on your API (from your individual application or program).
3. Uploading lists of files from an existing directory (from your EA).
4. Uploading file contents (from your EA).

There may be other functions that you need, but specifically, only the last two will be needed in the EA. The second string is one of the "headers". You will also need to form it based on what you pass to the API. As a rule, all you need is an access key to prevent anyone from breaking in. But you can add more if you need to convey some additional data. The received string from the server should be parsed here.

```
//+------------------------------------------------------------------+
//|                  get file list from JSON string                  |
//+------------------------------------------------------------------+
bool extractFiles(string json, string &Files[])
   {

   return false;
   }
```

We receive JSON and parse it into names. Unfortunately, there is no universal parsing code. Each case is individual. Personally, I would not say that writing the parsing code is too difficult. Of course, it is good to have the appropriate libraries, but personally I prefer to write as much code as possible myself. Now let's look at a similar method that gets the file contents.

```
//+------------------------------------------------------------------+
//|                    getting the file content                      |
//+------------------------------------------------------------------+
int GetFileContent(string directory,string filename,string &OutContent[])
   {
   //string for getting a file content in the form of JSON via GET to API
   string urlList = ApiDomen+"/file_content/"+directory+"/"+filename;//
   char message[];// Body of the request
   string headers = "Password_key: " + key;// We form the headers of the request
   string resultheaders = "";//returning headers
   string cookie = "";//cookies
   int timeout = 1500;//waiting for a response when requesting a file or json
   char result[];

   // We send a GET request to the server to receive JSON with a file content
   int res_code =  WebRequest("GET", urlList, headers, timeout, message, result, resultheaders);
   bool rez = extractContent(CharArrayToString(result),OutContent);
   if (rez) return res_code;
   else return 400;
   }
```

Everything is exactly the same as in the previous one, except that we get not a list of files, but a list of strings inside the file. Of course, parsing JSON with the file contents string by string is done by a separate logic in the following method, which has the identical purpose as its brother - “extractFiles”.

```
//+------------------------------------------------------------------+
//|   read the contents of the file from JSON each line separately   |
//+------------------------------------------------------------------+
bool extractContent(string json, string &FileLines[])
   {

   return false;
   }
```

Of course, you do not have to do everything exactly as I say, I just already have a product that is built exactly like this. Using a specific working example seems to me much easier for understanding all this. After receiving the file contents, you can safely write it, string by string, using the following method, which is actually already integrated into the template logic.

```
//+-----------------------------------------------------------------------+
//|    fill the file with its lines, which are all contained in data      |
//|  with a new line separator, and save in the corresponding directory   |
//+-----------------------------------------------------------------------+
void WriteData(string &data[],string FileName)
   {
   int fileHandle;
   string tempsubfolder= SubfolderE == "" ? ""  : SubfolderE + "\\";

   if (!bCommonReadE)
      {
      fileHandle = FileOpen(tempsubfolder+FileName,FILE_REWRITE|FILE_WRITE|FILE_TXT|FILE_ANSI);
      }
   else
      {
      fileHandle = FileOpen(tempsubfolder+FileName,FILE_REWRITE|FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
      }

   if(fileHandle != INVALID_HANDLE)
       {
       FileSeek(fileHandle,0,SEEK_SET);
       for (int i=0; i < ArraySize(data) ; i++)
          {
          FileWriteString(fileHandle,data[i]+"\r\n");
          }
       FileClose(fileHandle);
       }
   }
```

This is how a file is created by its name. And of course, its contents are also written into it in the form of separate strings. This is a completely adequate solution for small files. I did not notice any slowdowns during its work.

### General method with trading logic

I touched on this issue in the previous article, but I want to highlight it again and remind you that this method works when a new bar is opened in each virtual EA. We can consider it a handler of the OnTick type, but in our case it will, of course, be OnBar. By the way, there is no such handler in MQL5. It works a little differently than we would like, but it does not really have a significant impact on bar trading, so that is the least of our problems.

```
//+------------------------------------------------------------------+
//|      the main trading function of individual robot instance      |
//+------------------------------------------------------------------+
void BotInstance::Trade()
{
   //data access

   //Charts[chartindex].CloseI[0]//current bar (zero bar is current like in mql4)
   //Charts[chartindex].OpenI[0]
   //Charts[chartindex].HighI[0]
   //Charts[chartindex].LowI[0]
   //Charts[chartindex]. ???

   //close & open

   //CloseBuyF();
   //CloseSellF();
   //BuyF();
   //SellF();

   // Here we can include operations such as closing the buying position, closing selling position and opening new positions.
   // Other information from the chart can be used for making our buying/selling decisions.

   // Here is a simple trading logic example
   if ( Charts[chartindex].CloseI[1] > Charts[chartindex].OpenI[1] )
   {
      CloseBuyF();
      SellF();
   }
   if ( Charts[chartindex].CloseI[1] < Charts[chartindex].OpenI[1] )
   {
      CloseSellF();
      BuyF();
   }
}
```

I have implemented some basic logic inside the template so you can build your own using it as an example. I advise adding your own logic and variables next to this method in the BotInstance class, so as not to get confused. I recommend building your logic using methods and variables that you will use in the main Trade method.

### Graphical interface

The template, just like the previous version, contains an example of a simple user interface, whose color scheme and content can be changed. This interface is identical for both templates: both MetaTrader 4 and MetaTrader 5, and it looks better.

![GUI](https://c.mql5.com/2/70/vcu0ykgbz.png)

Question marks indicate places where you can add some additional data or remove unnecessary blocks. This is very easy to do. There are two methods for working with the interface: CreateSimpleInterface and UpdateStatus. They are very simple. I will not show them in action. You can find them by their respective names.

I have added three very useful fields to this interface. If you look at the last three strings, you can see your "reserved magic number corridor", which is relevant for the current configuration you use. If we delete or add settings files, then, accordingly, this corridor will either narrow or expand. In addition, we need to somehow protect different EAs from conflicts, and this field will help us with this. The remaining two fields signal the last time any of the settings were read, and the time of the last synchronization with our API, provided that synchronization occurs at all.

### Conclusion

In this article, we have arrived at a more convenient and functional template model, which, among other things, is suitable for further expansion and modification. The code is still far from perfect, and there is still a lot to be optimized and fixed, but even taking all this into account, I have several clearly formed ideas on what to add there and for what purpose. For the next article, I want to virtualize positions and, based on this data, get a unique cross-currency optimizer that will optimize each of our EAs individually and generate ready-made files with settings for our strategies.

The cross-currency optimizer will allow us to simultaneously optimize all virtual instrument-periods. By merging all these settings, we will get a better and safer profit curve with reduced risks. I consider automatic diversification and profit enhancement an absolute priority for further improvements. As a result, I would like to get some basic add-on over any strategy that would allow extracting maximum profit from it, while maintaining maximum functionality and convenience for the end user. This should be something like an exoskeleton for our trading signal.

**Links**

- [Previous article](https://www.mql5.com/en/articles/12434)
- [Example of a free product based on the template](https://www.mql5.com/en/market/product/112698)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14251](https://www.mql5.com/ru/articles/14251)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14251.zip "Download all attachments in the single ZIP archive")

[DynamicTemplate.zip](https://www.mql5.com/en/articles/download/14251/dynamictemplate.zip "Download DynamicTemplate.zip")(45.56 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Brute force approach to patterns search (Part VI): Cyclic optimization](https://www.mql5.com/en/articles/9305)
- [Brute force approach to patterns search (Part V): Fresh angle](https://www.mql5.com/en/articles/12446)
- [OpenAI's ChatGPT features within the framework of MQL4 and MQL5 development](https://www.mql5.com/en/articles/12475)
- [Rebuy algorithm: Multicurrency trading simulation](https://www.mql5.com/en/articles/12579)
- [Rebuy algorithm: Math model for increasing efficiency](https://www.mql5.com/en/articles/12445)
- [Multibot in MetaTrader: Launching multiple robots from a single chart](https://www.mql5.com/en/articles/12434)

**[Go to discussion](https://www.mql5.com/en/forum/468883)**

![The base class of population algorithms as the backbone of efficient optimization](https://c.mql5.com/2/71/The_basic_class_of_population_algorithms____LOGO_2_.png)[The base class of population algorithms as the backbone of efficient optimization](https://www.mql5.com/en/articles/14331)

The article represents a unique research attempt to combine a variety of population algorithms into a single class to simplify the application of optimization methods. This approach not only opens up opportunities for the development of new algorithms, including hybrid variants, but also creates a universal basic test stand. This stand becomes a key tool for choosing the optimal algorithm depending on a specific task.

![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part I)](https://c.mql5.com/2/81/Building_A_Candlestick_Trend_Constraint_Model_Part_5___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part I)](https://www.mql5.com/en/articles/14963)

We will breakdown the main MQL5 code into specified code snippets to illustrate the integration of Telegram and WhatsApp for receiving signal notifications from the Trend Constraint indicator we are creating in this article series. This will help traders, both novices and experienced developers, grasp the concept easily. First, we will cover the setup of MetaTrader 5 for notifications and its significance to the user. This will help developers in advance to take notes to further apply in their systems.

![MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://c.mql5.com/2/82/MQL5_Wizard_Techniques_you_should_know_Part_24__LOGO.png)[MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://www.mql5.com/en/articles/15135)

Moving Averages are a very common indicator that are used and understood by most Traders. We explore possible use cases that may not be so common within MQL5 Wizard assembled Expert Advisors.

![Data Science and Machine Learning (Part 24): Forex Time series Forecasting Using Regular AI Models](https://c.mql5.com/2/81/Data_Science_and_Machine_Learning_Part_24__LOGO.png)[Data Science and Machine Learning (Part 24): Forex Time series Forecasting Using Regular AI Models](https://www.mql5.com/en/articles/15013)

In the forex markets It is very challenging to predict the future trend without having an idea of the past. Very few machine learning models are capable of making the future predictions by considering past values. In this article, we are going to discuss how we can use classical(Non-time series) Artificial Intelligence models to beat the market

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/14251&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070121589476888653)

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