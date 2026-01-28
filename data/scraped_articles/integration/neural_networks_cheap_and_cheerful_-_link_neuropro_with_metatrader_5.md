---
title: Neural Networks Cheap and Cheerful - Link NeuroPro with MetaTrader 5
url: https://www.mql5.com/en/articles/830
categories: Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T17:26:39.750352
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/830&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068238504015623965)

MetaTrader 5 / Integration


### About NeuroPro

The NeuroPro program was written in one of Russian institutes back in 1998 and is still relevant today.

It efficiently runs on Windows XP, Vista and Windows 7. I cannot tell how it works on later versions of Windows as I have not tested it.

![About NeuroPro](https://c.mql5.com/2/11/neuropro-0.png)

Fig. 1. About NeuroPro

Version 0.25 is free and can be found on many websites on the Internet. NeuroPro can create [multilayer neural networks with the sigmoid activation function](https://www.mql5.com/en/articles/497). If you have just started learning neural networks, you do not need more features at this stage. It should be kept in mind that the interface of NeuroPro is in Russian and has not been translated into any other languages.

Neural network can be trained on one data array and then tested on another one. It is an essential feature for traders as it allows to understand quickly if the selected network structure is prone to overfitting and if it can consistently trade outside historical data, i.e. on a real account.

Those who like to dig deeper have an opportunity to see neural network weights as well as which network inputs influence the result of network operation most of all. Beginners do not need that and they may skip this part of the program. This information is useful for experienced traders looking for the Grail because it lets them assume what pattern was identified by the neural network and see where they can continue their search.

Other than that, there are no significant features in NeuroPro except various settings and useful utilities like minimizer of the network structure. These menu sections are not compulsory to use so novices do not have to complicate things and use only the default settings.

From a trader's point of view, NeuroPro has only one disadvantage - absence of integration with MetaTrader 5. Actually, this article is mostly dedicated to loading market and indicator data from MetaTrader 5 to NeuroPro and then turn the received neural network into an Expert in MQL5.

Advancing the topic, I can say that the neural network that we are going to create with NeuroPro will be converted with all the neuron weights into an MQL5 script (unlike the system of include DLL like in any [other neural network program](https://www.mql5.com/en/articles/236)). It will ensure fast work and minimum use of computer resources. That is a clear advantage of using NeuroPro. It can be used for creating any trading strategies, even scalping ones with their requirement for the Expert to make decisions nearly immediately.

### Trading Strategy

In this article we are not going to consider scalping because the process of creating, training and testing scalping Experts has a lot of peculiarities and goes beyond this article.

For educational purposes we shall create a simple Expert for the H1 timeframe and popular currency pair EURUSD. So, let our Expert analyze last 24 bars i.e. market behavior in the last day, forecast the direction of the price movement in the following hour and then trade based on that information.

### How to Load Data to NeuroPro from MetaTrader

**Supported Data Format**

NeuroPro reads data only in the DBMS formats (tables DBMS Paradox) and DBF (tables DBMS FoxPro and dBase). DBF is the most common format in the world and if you are an experienced programmer, you must have come across it many times. We are going to use this format too.

Algorithm of the data transfer in NeuroPro looks as follows:

1. write a script for MetaTrader, load required data to a text file in the CSV format dividing data with commas;
2. using special programs, convert CSV into DBF;
3. open DBF in NeuroPro.

**Loading Data from MetaTrader**

When writing a loading data script, some nuances should be taken into account:

- a data field name in DBF cannot be longer than 11 symbols, besides some converters cut them down to 10. That is why field names should not be longer than 10 symbols;
- in neural networks with a great number of bars input, field names are usually of the "BarN" type, where N is a sequential number of the bar. In our case there will be 24 fields and the names consequently will vary from "Bar1" to "Bar24". I recommend, though it is not compulsory, to write names of such fields like "Bar\_\_\_N\_\_" (at first three underscores and then two). Further down the line, when we are writing the Expert, you will see why.

Below is a ready script for our testing strategy (it is also attached to this article):

```
#property script_show_inputs
//+------------------------------------------------------------------+
input string    Export_FileName  = "NeuroPro\\data.csv"; // File for export (in the "MQL5/Files" folder)
input int       Export_Bars_Skip = 0;                    // Number of historical bars to skip before export
input int       Export_Bars      = 5000;                 // Number of lines for export
//+------------------------------------------------------------------+
const int inputlen=24;    // Number of past bars analyzed by the trading strategy
//+------------------------------------------------------------------+
void OnStart()
  {
   //--- create a file
   int file=FileOpen(Export_FileName,FILE_WRITE|FILE_CSV|FILE_ANSI,',');

   if(file!=INVALID_HANDLE)
     {
      //--- write the data header
      string row="date";
      for(int i=0; i<=inputlen; i++)
        {
         if(StringLen(row)) row+=",";
         //========================================================
         // Note!
         // In the Expert substitute underscores for [].
         // A field name in the DBase format is no longer than 11 symbols. Calc reduces it down to 10.
         // Maximum number of fields in the DBase format is 128-512, depending on the version.
         //========================================================
         StringConcatenate(row,row,"Bar___",i,"__");
        }
      FileWrite(file,row);

      //--- copy required data from history
      MqlRates rates[],rate;
      int count=Export_Bars+inputlen;
      if(CopyRates(Symbol(),Period(),1+Export_Bars_Skip,count,rates)<count)
        {
         Print("Error! Insufficient historical data for exporting required data.");
         return;
        }
      ArraySetAsSeries(rates,true);

      //--- write down the data
      for(int bar=0; bar<Export_Bars; bar++)
        {
         row="";
         //--- closing price of the 1st bar will be the zero level for normalization of others
         double zlevel=rates[bar+1].close;
         for(int i=0; i<=inputlen; i++)
           {
            if(StringLen(row)) row+=",";
            rate=rates[bar+i];
            if(i==0) row+=TimeToString(rate.time,TIME_DATE || TIME_MINUTES)+",";
            row+=DoubleToString(rate.close-zlevel,Digits());
           }
         FileWrite(file,row);
        }
      FileClose(file);
      Print("Data export successfully completed.");
     }
   else Print("Error! Failed to create a file for data export. ",GetLastError());
  }
```

Now let us launch it in the terminal. Upon successful completion, it will enter a correspondent message to the Expert journal.

Data file created by the script contains approximately the following.

In the first string there are names of the table fields; then follow the strings with the values of those fields divided by commas:

date,Bar\_\_\_0\_\_,Bar\_\_\_1\_\_,Bar\_\_\_2\_\_,Bar\_\_\_3\_\_,Bar\_\_\_4\_\_,Bar\_\_\_5\_\_,Bar\_\_\_6\_\_,Bar\_\_\_7\_\_,Bar\_\_\_8\_\_,Bar\_\_\_9\_\_,Bar\_\_\_10\_\_,Bar\_\_\_11\_\_,Bar\_\_\_12\_\_,Bar\_\_\_13\_\_,Bar\_\_\_14\_\_,Bar\_\_\_15\_\_,Bar\_\_\_16\_\_,Bar\_\_\_17\_\_,Bar\_\_\_18\_\_,Bar\_\_\_19\_\_,Bar\_\_\_20\_\_,Bar\_\_\_21\_\_,Bar\_\_\_22\_\_,Bar\_\_\_23\_\_,Bar\_\_\_24\_\_

2014.09.25,-0.0008,0.0000,-0.0005,-0.0014,0.0007,0.0035,0.0035,0.0036,0.0047,0.0052,0.0050,0.0046,0.0046,0.0047,0.0049,0.0052,0.0049,0.0053,0.0055,0.0056,0.0067,0.0056,0.0097,0.0105,0.0113

2014.09.25,0.0005,0.0000,-0.0009,0.0012,0.0040,0.0040,0.0041,0.0052,0.0057,0.0055,0.0051,0.0051,0.0052,0.0054,0.0057,0.0054,0.0058,0.0060,0.0061,0.0072,0.0061,0.0102,0.0110,0.0118,0.0123

2014.09.25,0.0009,0.0000,0.0021,0.0049,0.0049,0.0050,0.0061,0.0066,0.0064,0.0060,0.0060,0.0061,0.0063,0.0066,0.0063,0.0067,0.0069,0.0070,0.0081,0.0070,0.0111,0.0119,0.0127,0.0132,0.0130

2014.09.25,-0.0021,0.0000,0.0028,0.0028,0.0029,0.0040,0.0045,0.0043,0.0039,0.0039,0.0040,0.0042,0.0045,0.0042,0.0046,0.0048,0.0049,0.0060,0.0049,0.0090,0.0098,0.0106,0.0111,0.0109,0.0122

**Conversion from CSV to DBF**

There are many ways of doing it.

- Microsoft Excel, versions earlier than 2007. It can open CSV files and also save data in the DBF format. Bear in mind that very old versions of Excel accommodate only 65535 strings. Saying that, this capacity is usually sufficient for trading as this volume can fit a 10 years' history of the H1 timeframe;
- Microsoft Excel, versions of 2007 and later. Saving in DBF is not available in them. An add-on enriching Excel with this functionality can be found on the Internet;
- Microsoft Access (Microsoft Office package program for working with data bases). A table can be created in a data base by importing from a text file (CSV) and then exporting it to a DBF one;
- designated utilities-converters CSV-DBF. There are a lot of them and different makes can be found on the Internet, though majority of them have to be paid for;
- Calc from the free OpenOffice package. Calc is nearly a full analogue of Excel. It can open CSV files and save them in DBF.

All the above methods are intuitive and should be easy enough to use.

I will perform conversion in one of the listed ways. I will use the last one because NeuroPro is a free program and the idea behind free Calc is the closest to it. You can download OpenOffice from the official web site - [http://www.openoffice.org/](https://www.mql5.com/go?link=http://www.openoffice.org/ "http://www.openoffice.org/").

Process of conversion.

1. Start Calc. Open our data file with the CSV extension.
2. Calc will launch data recognition wizard.
3. Specify in the wizard parameters that the fields are to be separated by commas.

Another important point here is a separator of the integer and fractional parts of the number. As per configurations on my computer ("START" > "Control Panel" > "Language and Regional Standards"), dot is used as a separator. In our CSV file we also use dot. For Calc to read numbers correctly, a separator has to be specified too. It can be done by selecting the required language in the settings of the conversion wizard. Select one of the variations of the English language as dot is a standard separator there.

Correct settings of the wizard are shown on the screenshot:

![](https://c.mql5.com/2/12/import-eng-1.png)

Fig. 2. Setup of the import wizard from the CSV file

Tip: to skip selecting a language at every CSV file conversion, it can be set up as a default language in the Calc settings: menu "Tools" > "Options" and then as highlighted in green on the screenshot:

![](https://c.mql5.com/2/12/import-eng-2.png)

Fig. 3. Locale setup in Calc

4) So, the CSV file has uploaded and the data is automatically put into columns:

![](https://c.mql5.com/2/12/import-eng-3.png)

Fig. 4. Successfully opened CSV file

5) For the data to be correctly written in the DBF format their type and precision have to be specified.

For that all the columns with numbers are to be highlighted and correspondent properties assigned as highlighted in green on the screenshot:

![](https://c.mql5.com/2/12/import-eng-4.png)

Fig. 5. Setting up columns with numbers

6) Save in DBF: menu "File" > "Save As". In the dialog window select the file type "dBase (\*.dbf)":

![](https://c.mql5.com/2/12/import-eng-5.png)

Fig. 6. Dialog of saving a file in the DBF format

Then press "Save".

7) Calc will ask to confirm selected format:

![](https://c.mql5.com/2/12/import-eng-6.png)

Fig. 7. Calc suggests saving a file in the standard ODF format instead of DBF

Confirm our choice by pressing the "Keep Current Format" button.

8) Calc will inquire what coding to use for text data in the DBF file. Since there is no such data in our example and text data is impossible to use in neural networks anyway, you can specify any:

![](https://c.mql5.com/2/12/import-eng-7.png)

Fig. 8. Selecting text coding of the file

Now we have a file with the DBF extension that contains data received from MetaTrader required for the neural network.

### How to Create and Train a Neural Network in NeuroPro

1) Start NeuroPro.

2) Create a working project: menu "File" > "Create":

![](https://c.mql5.com/2/11/neuropro-1.png)

Fig. 9. Empty project created

3) In the project window press the "Open data file" button and open the DBF file in the appeared dialog:

![](https://c.mql5.com/2/11/neuropro-2.png)

Fig. 10. The DBF file opened for using with the future neural network

4) In the project window click on "New network". In the opened window there are two tabs. We are going to fill the "Inputs and outputs" tab first.

Specify in the "Bar\_\_\_0\_\_" field that it will be the output neuron of the network. The rest of the fields "Bar\_\_\_N\_\_" will be appointed to be inputs:

![](https://c.mql5.com/2/11/neuropro-3.png)

Fig. 11. Configuration of the inputs and outputs of the neural network

We can also specify the required accuracy for the output neuron. In Forex it is 1 point i.e. in our case it is 0.0001.

5) Now we are moving on to the "Network Structure" tab. Here the number of intermediate layers (excluding input and output layers) and the number of neurons in each of them can be specified. In this particular learning example we are going to create 3 layers each containing 20 neurons:

![](https://c.mql5.com/2/11/neuropro-4.png)

Fig. 12. Configuration of the neural network layers

6) Press the "Create" button and the network is ready to use:

![](https://c.mql5.com/2/11/neuropro-5.png)

Fig. 13. A newly made and setup neural network appeared in the project

7) Go to the "Neural Network" menu > "Testing", to see how a new and untrained network is coping with producing price forecasts.

As test results showed, this untrained neural network predicted a price with a specified accuracy (with error not exceeding 1 point) in less than 5% cases. On average, a price forecast error was around 10 points:

![](https://c.mql5.com/2/11/neuropro-6.png)

Fig. 14. Statistics of forecast accuracy of untrained neural network

8) Now we have to train the network based on our data.

Again, go to the project window and press: "Neural network" > "Training" in the menu. An indicator of learning progress will appear. Wait for the end of the process:

![](https://c.mql5.com/2/11/neuropro-7.png)

Fig. 15. Indicator of the learning process of the neural network

9) Get back to the project window and go to "Neural network" > "Testing".

The network has significantly improved: its price forecast was accurate in 16% cases, and the average forecast error was 4 points:

![](https://c.mql5.com/2/11/neuropro-8.png)

Fig. 16. Statistics of forecast accuracy of a trained network

The network has learned a few things. We shall transfer it to MetaTrader.

### How to Transfer a Neural Network from NeuroPro to MetaTrader 5

NeuroPro knows nothing about MetaTrader 5 and cannot pass a neural network directly to it. I have worked out a semi-automatic way of converting a neural network into a fragment of MQL5 code.

Unlike many other neural network programs, NeuroPro can show the structure of the neural network in use as a text. It is a set of formulas sequentially describing all the data transformations from the input moment till it leaves the network. Formulas include every layer, every neuron, every connection with already substituted (trained) connection weight values.

To see it, go to the "Neural network" > "Verbalization" menu. In our case the formulas are as follows:

![](https://c.mql5.com/2/11/advisor-1.png)

Fig. 17. Formulas determining the work of the trained neural network

Actually, the set of these formulas can be considered as a source code of a program written in an abstract programming language. All we need to do is to modify this code so its syntax matches MQL5. These changes can be made in any text editor. To partially automate this process, I recommend using an editor that can perform a mass replacement of phrases. It can be done in: Word, its free version Writer (from the OpenOffice package), Excel, Calc and even Notepad in Windows.

I am sure you could change those formulas into an MQL5 code yourself, nevertheless I am going to share my experience in optimizing this process so you can do it quicker.

In my example I am going to use Notepad in Windows 7.

1) So, we have an open project in NeuroPro with the trained neural network. We go to "Neural network" > "Verbalization", where a window with formulas opened (see the screenshot above).

2) Save the content of this window in the: menu "File" > "Save as".

3) Now open this file in Notepad.

4) Call the phrase replacement function: menu "Edit" > "Replace".

List of replacements to be done:

| What to replace | Replace with | Comment |
| --- | --- | --- |
| \\_\\_\\_ | \[ | triple underscore |\
| \_\_ | \] | double underscore |
| -- | \- - | two minuses (deduction of a negative number in formulas) are divided by a space because a double minus in MQL (as well as other languages similar to C) can have a double meaning, which leads to a compilation error |
| Sigmoid | Sigmoid | translate function names into Latin (mind you, the program is in Russian. There is no necessity to do it though because MetaEditor supports Cyrillic too) |
| Syndrom | Syndrome | translate variable names into Latin (again, there is no point to do it as MetaEditor supports Cyrillic too) |

![](https://c.mql5.com/2/11/advisor-3.png)

Fig. 18. Replace triple underscore with square brackets

Use the "Replace All" button to perform replacement.

Now you understand that I called the price fields "BAR\_\_\_N\_\_" to be able to quickly substitute underscores for square brackets, i.e. present all network inputs as an array.

It is easier to declare an array and fill it with serial price data than a number of individual variables.

5) As I mentioned before, enumeration of all inputs and outputs has to be changed for declaring an array:

| Before | After |
| --- | --- |
| Data base fields (initial symptoms):<br>       BAR\[1\]<br>       BAR\[2\]<br>       BAR\[3\]<br>       BAR\[4\]<br>       BAR\[5\]<br>       BAR\[6\]<br>       BAR\[7\]<br>       BAR\[8\]<br>       BAR\[9\]<br>       BAR\[10\]<br>       BAR\[11\]<br>       BAR\[12\]<br>       BAR\[13\]<br>       BAR\[14\]<br>       BAR\[15\]<br>       BAR\[16\]<br>       BAR\[17\]<br>       BAR\[18\]<br>       BAR\[19\]<br>       BAR\[20\]<br>       BAR\[21\]<br>       BAR\[22\]<br>       BAR\[23\]<br>       BAR\[24\]<br>Data base fields (final syndromes):<br>       BAR\[0\] | ```<br>double BAR [25];<br>``` |

6) Neuron activation functions must look like MQL5 program functions:

| Before | After |
| --- | --- |
| Sigmoid1(A)=A/(0.1+\|A\|)<br>Sigmoid2(A)=A/(0.1+\|A\|)<br>Sigmoid3(A)=A/(0.1+\|A\|) | ```<br>double Sigmoid1 (double A)<br>{<br>  return A/(0.1 + MathAbs(A));<br>}<br>double Sigmoid2 (double A)<br>{<br>  return A/(0.1 + MathAbs(A));<br>}<br>double Sigmoid3 (double A)<br>{<br>  return A/(0.1 + MathAbs(A));<br>}<br>``` |

7) According to the MQL5 rules, a semicolon must be put in the end of all formulas, comments have to be written correctly (or deleted) and type declaration added to all initialized variables.

In our case, the type has not been declared only for the neuron names in the intermediary layers. We are going to use mass text replacement again instead of inputting the word "double" manually for 60 times. We need to highlight the beginning of a string with a neuron name (the margin in the beginning of the string has to be highlighted too because neuron names will be also used in the right part of the formulas and the word "double" will not need to be inserted there) :

![](https://c.mql5.com/2/12/advisor-4-eng.png)

Fig. 19. Highlighting the text for replacement

Having copied the highlighted part of the text, insert it in the text replacement dialog to replace with the same text and the word "double" added to it:

![](https://c.mql5.com/2/12/advisor-5-eng.png)

Fig. 20. Add thetype name  to the variables

Don't forget to press the "Replace all" button.

8) NeuroPro has a little bug. If you input a constant value to the neural network, then in the text format the normalization formula of this input will contain dividing by zero. In our case "BAR\_\_\_1\_\_" is such an input. It always has zeros because it is the reference point for our bars normalizing.

Ideally, "BAR\_\_\_1\_\_" should not be input in the neural network as inputs with constant values do not influence the forecast anyway. However, if that value was input, the formula produced by NeuroPro will have to be adjusted. To avoid error messages from the compiler, the "BAR\_\_\_1\_\_" will have to be replaced with the value that is permanently input here. In our case it is zero:

| Before | After |
| --- | --- |
| ```<br>BAR[1]=(BAR[1]-0)/0;<br>``` | ```<br>BAR[1]=0;<br>``` |

9) There is another very insignificant bug (the developer of NeuroPro did not think that a text description of a neural network would be used as a program code and therefore did not check it carefully).

In the very last formula in the end there is an extra closing parenthesis. This bug is very small but it confuses the MetaEditor compiler. It won't point at the extra parenthesis in that string but will not take a curly bracket in another part of the program. Please bear it in mind so you can rectify it when come across.

| Before | After |
| --- | --- |
| ```<br>BAR[0]=((BAR[0]*0.0180000001564622)+0.000599999912083149)/2);<br>``` | ```<br>BAR[0]=((BAR[0]*0.0180000001564622)+0.000599999912083149)/2;<br>``` |

Operations described in this section take only a few minutes if practiced regularly. There is no need to remember the list to the letter. At the following compilation MetaEditor will point the uncorrected parts of code as errors.

Finally, after bringing all formulas into the MQL5 format, all we need to do is to transfer the resulting code from the Notepad to MetaEditor and add the rest of the code required for the Expert. Surely, if you regularly use neural networks created in NeuroPro, this stage will be easy. You will simply replace the previous neural network in the existing Expert with a new MQL5 code of a neural network from the Notepad. It will literally take a minute.

The final code of the Expert, completely ready for work in MetaTrader 5 (you can also download this code from the application to this article):

```
input double    Lots = 0.1;        // Deal volume
input double    MinPrognosis = 0;  // Open deals with a forecast more promising than the current one
//+------------------------------------------------------------------+
const int inputlen=24; // Number of past bars analyzed by the trading strategy
//+------------------------------------------------------------------+
double Sigmoid1(double A)
  {
   return A/(0.1 + MathAbs(A));
  }
//+------------------------------------------------------------------+
double Sigmoid2(double A)
  {
   return A/(0.1 + MathAbs(A));
  }
//+------------------------------------------------------------------+
double Sigmoid3(double A)
  {
   return A/(0.1 + MathAbs(A));
  }
//+------------------------------------------------------------------+
double CalcNeuroNet()
  {
//--- get current quotes for neural network
   MqlRates rates[],rate;
   CopyRates(Symbol(),Period(),0,inputlen+1,rates);
   ArraySetAsSeries(rates,true);

//--- neural network inputs
   double BAR[512]; // 512 - maximum permissible number of fields in the DBF format

//--- fill the array of the neural network input data
//--- closing price of the 1st bar will be the zero level for normalization of others
   double zlevel=rates[1].close;

   for(int bar=0; bar<=inputlen; bar++)
     {
      rate=rates[bar];
      BAR[bar]=rate.close-zlevel;
     }

//==============================================
// Calculate the neural network with NeuroPro formulas
//==============================================

//--- preprocessing of the data base input fields for training the network:
   BAR[1]=0;//(BAR[1]-0)/0;
   BAR[2]=(BAR[2]- -0.0003)/0.009;
   BAR[3]=(BAR[3]-4.999992E-5)/0.01045;
   BAR[4]=(BAR[4]-0.0011)/0.011;
   BAR[5]=(BAR[5]-0.00285)/0.01335;
   BAR[6]=(BAR[6]-0.004050001)/0.01625;
   BAR[7]=(BAR[7]-0.00495)/0.01695;
   BAR[8]=(BAR[8]-0.0049)/0.0172;
   BAR[9]=(BAR[9]-0.0046)/0.0171;
   BAR[10]=(BAR[10]-0.00395)/0.01755;
   BAR[11]=(BAR[11]-0.0037)/0.0184;
   BAR[12]=(BAR[12]-0.0034)/0.0188;
   BAR[13]=(BAR[13]-0.0029)/0.0194;
   BAR[14]=(BAR[14]-0.002499999)/0.0196;
   BAR[15]=(BAR[15]-0.00245)/0.01935;
   BAR[16]=(BAR[16]-0.00275)/0.01925;
   BAR[17]=(BAR[17]-0.0028)/0.0194;
   BAR[18]=(BAR[18]-0.002950001)/0.01965;
   BAR[19]=(BAR[19]-0.002649999)/0.01965;
   BAR[20]=(BAR[20]-0.002699999)/0.0197;
   BAR[21]=(BAR[21]-0.00275)/0.01945;
   BAR[22]=(BAR[22]-0.00225)/0.01955;
   BAR[23]=(BAR[23]-0.0019)/0.0195;
   BAR[24]=(BAR[24]-0.00225)/0.01935;

//--- syndromes of the 1st level:
   double Syndrome1_1=Sigmoid1( 0.07165167*BAR[1]-0.08914512*BAR[2]+0.160242*BAR[3]-0.1136391*BAR[4]+0.01358515*BAR[5]+0.3755009*BAR[6]-0.1433693*BAR[7]+0.224411*BAR[8]+0.03298632*BAR[9]-0.2551045*BAR[10]-0.1418581*BAR[11]+0.007130164*BAR[12]-0.08727393*BAR[13]-0.2567087*BAR[14]+0.1118081*BAR[15]+0.73848*BAR[16]+0.05880548*BAR[17]-0.1544689*BAR[18]+0.192913*BAR[19]-0.1743894*BAR[20]-0.2184512*BAR[21]-0.2290305*BAR[22]+0.3946579*BAR[23]-0.02947071*BAR[24]-0.08091708 );
   double Syndrome1_2=Sigmoid1( -0.08248464*BAR[1]+0.3076621*BAR[2]-0.0500868*BAR[3]-0.6526818*BAR[4]+0.04266862*BAR[5]+0.581119*BAR[6]-0.0356447*BAR[7]+0.0292943*BAR[8]-0.3660156*BAR[9]-0.3244759*BAR[10]+0.05519342*BAR[11]+0.2419113*BAR[12]-0.2178954*BAR[13]+0.4037299*BAR[14]-0.1593139*BAR[15]+0.3567515*BAR[16]+0.08094382*BAR[17]-0.01788837*BAR[18]-0.379636*BAR[19]+0.6658992*BAR[20]-0.1899142*BAR[21]+0.02259956*BAR[22]+0.767949*BAR[23]-0.5380562*BAR[24]-0.06307755 );
   double Syndrome1_3=Sigmoid1( -0.08426282*BAR[1]-0.172721*BAR[2]+0.1749717*BAR[3]-0.07916483*BAR[4]-0.0523758*BAR[5]+0.1935233*BAR[6]+0.01627235*BAR[7]+0.1254414*BAR[8]-0.1101555*BAR[9]-0.02285305*BAR[10]-0.14389*BAR[11]+0.1788775*BAR[12]-0.007144043*BAR[13]+0.1925385*BAR[14]-0.08001231*BAR[15]-0.2021703*BAR[16]+0.08694438*BAR[17]+0.3090158*BAR[18]-0.3330302*BAR[19]+0.2519112*BAR[20]-0.2170611*BAR[21]-0.2216277*BAR[22]+0.09618518*BAR[23]+0.049888*BAR[24]-0.06465426 );
   double Syndrome1_4=Sigmoid1( 0.02806905*BAR[1]+0.07787746*BAR[2]+0.1972721*BAR[3]-0.247464*BAR[4]-0.008635854*BAR[5]-0.1975036*BAR[6]-0.0652089*BAR[7]-0.1276176*BAR[8]-0.3386112*BAR[9]-0.103951*BAR[10]+0.08352495*BAR[11]-0.1821419*BAR[12]-0.05604611*BAR[13]-0.05922695*BAR[14]-0.1670811*BAR[15]+0.002476109*BAR[16]-0.03657883*BAR[17]-0.09295338*BAR[18]+0.2500353*BAR[19]-0.03980102*BAR[20]+0.1059941*BAR[21]-0.4037244*BAR[22]-0.08735184*BAR[23]+0.1546644*BAR[24]+0.1966186 );
   double Syndrome1_5=Sigmoid1( 0.03832016*BAR[1]-0.09065858*BAR[2]+0.2356484*BAR[3]-0.2436682*BAR[4]+0.09812659*BAR[5]+0.09220826*BAR[6]+0.434221*BAR[7]-0.005478878*BAR[8]-0.1657191*BAR[9]-0.2605299*BAR[10]+0.3523667*BAR[11]+0.3595579*BAR[12]+0.3402678*BAR[13]-0.3346431*BAR[14]+0.1215327*BAR[15]-0.1869196*BAR[16]+0.07256371*BAR[17]-0.09229603*BAR[18]-0.09961994*BAR[19]+0.2491707*BAR[20]+0.3703756*BAR[21]+0.1369175*BAR[22]+0.0560869*BAR[23]-0.007567503*BAR[24]-0.01722363 );
   double Syndrome1_6=Sigmoid1( -0.06897662*BAR[1]-0.4182717*BAR[2]+0.200378*BAR[3]-0.4152234*BAR[4]-0.2081593*BAR[5]+0.3120443*BAR[6]-0.1582431*BAR[7]+0.1900958*BAR[8]+0.002503331*BAR[9]+0.02297609*BAR[10]+0.03145982*BAR[11]+0.1816629*BAR[12]+0.1854629*BAR[13]-0.1660063*BAR[14]+0.3112128*BAR[15]-0.4799304*BAR[16]-0.100519*BAR[17]-0.1523588*BAR[18]+0.07141552*BAR[19]+0.2336634*BAR[20]+0.01279082*BAR[21]-0.2179644*BAR[22]+0.4898897*BAR[23]-0.1818153*BAR[24]-0.1783737 );
   double Syndrome1_7=Sigmoid1( -0.003986856*BAR[1]-0.3409385*BAR[2]-0.3122248*BAR[3]+0.5656545*BAR[4]+0.07564658*BAR[5]+0.07956024*BAR[6]+0.1820322*BAR[7]-0.05595554*BAR[8]+0.1027963*BAR[9]+0.2596273*BAR[10]+0.1156801*BAR[11]+0.04490443*BAR[12]+0.1426405*BAR[13]+0.06763341*BAR[14]-0.03249188*BAR[15]-0.1912978*BAR[16]-0.2003477*BAR[17]-0.2413947*BAR[18]+0.3188735*BAR[19]-0.2899658*BAR[20]+0.06846272*BAR[21]+0.08726751*BAR[22]-0.2134383*BAR[23]-0.436768*BAR[24]+0.08075105 );
   double Syndrome1_8=Sigmoid1( 0.05597013*BAR[1]+0.3358757*BAR[2]+0.1041476*BAR[3]-0.334706*BAR[4]-0.07069201*BAR[5]+0.06152828*BAR[6]+0.1577689*BAR[7]+0.1737777*BAR[8]-0.7711719*BAR[9]-0.2970988*BAR[10]+0.06691784*BAR[11]+0.0528774*BAR[12]+0.06260363*BAR[13]+0.2449201*BAR[14]-0.3098814*BAR[15]+0.06859511*BAR[16]+0.1355444*BAR[17]-0.15844*BAR[18]+0.2791151*BAR[19]-0.412524*BAR[20]+0.228981*BAR[21]-0.4042732*BAR[22]+0.197847*BAR[23]+0.477078*BAR[24]-0.2478239 );
   double Syndrome1_9=Sigmoid1( 0.02181781*BAR[1]-0.1042198*BAR[2]-0.02412975*BAR[3]+0.1485616*BAR[4]+0.07645424*BAR[5]-0.02779776*BAR[6]-0.1519209*BAR[7]-0.1878287*BAR[8]+0.1637603*BAR[9]+0.248636*BAR[10]+0.2032469*BAR[11]-0.03869069*BAR[12]+0.02014448*BAR[13]-0.2079489*BAR[14]+0.08846121*BAR[15]+0.1025348*BAR[16]+0.01593455*BAR[17]-0.4964754*BAR[18]+0.1635097*BAR[19]-0.04561989*BAR[20]-0.0662128*BAR[21]-0.2423395*BAR[22]+0.2898602*BAR[23]+0.03824728*BAR[24]-0.07471437 );
   double Syndrome1_10=Sigmoid1( -0.02918137*BAR[1]+0.06085975*BAR[2]-0.3056079*BAR[3]-0.5144019*BAR[4]-0.1966296*BAR[5]+0.04413594*BAR[6]+0.03249943*BAR[7]+0.08405613*BAR[8]-0.08797813*BAR[9]+0.06621616*BAR[10]-0.2226632*BAR[11]-0.1000158*BAR[12]+0.0106046*BAR[13]-0.1383344*BAR[14]+0.05141285*BAR[15]-0.1009147*BAR[16]-0.1503479*BAR[17]+0.2877283*BAR[18]-0.2209365*BAR[19]+0.1310906*BAR[20]-0.1188305*BAR[21]-0.002668453*BAR[22]+0.1106755*BAR[23]+0.3884961*BAR[24]+0.0006983803 );
   double Syndrome1_11=Sigmoid1( -0.04872056*BAR[1]-0.5066758*BAR[2]+0.08158222*BAR[3]+0.2647052*BAR[4]+0.3632542*BAR[5]+0.4538754*BAR[6]-0.1346472*BAR[7]+0.16742*BAR[8]+0.2974689*BAR[9]+0.3446769*BAR[10]-0.2784187*BAR[11]+0.2461497*BAR[12]-0.166853*BAR[13]-0.4296628*BAR[14]+0.7343794*BAR[15]+0.2154892*BAR[16]-0.4086125*BAR[17]-0.6446049*BAR[18]-0.5614476*BAR[19]-0.593914*BAR[20]+0.5039462*BAR[21]+0.113933*BAR[22]+0.3599374*BAR[23]-0.5517*BAR[24]+0.1249064 );
   double Syndrome1_12=Sigmoid1( -0.09035824*BAR[1]-0.2619464*BAR[2]+0.5151641*BAR[3]+0.08415102*BAR[4]+0.007849894*BAR[5]-0.3585253*BAR[6]-0.3458216*BAR[7]-0.006490127*BAR[8]+0.1933572*BAR[9]+0.1655464*BAR[10]-0.2591909*BAR[11]+0.2810482*BAR[12]-0.3552095*BAR[13]+0.1032239*BAR[14]-0.2380441*BAR[15]-0.6082169*BAR[16]-0.3652177*BAR[17]+0.4065064*BAR[18]-0.1538232*BAR[19]-0.03332642*BAR[20]+0.06235149*BAR[21]-0.08935639*BAR[22]-0.2274701*BAR[23]+0.2350571*BAR[24]-0.1009272 );
   double Syndrome1_13=Sigmoid1( -0.05370994*BAR[1]+0.2999545*BAR[2]-0.2855853*BAR[3]+0.1123754*BAR[4]+0.2561198*BAR[5]-0.2846766*BAR[6]+0.008345681*BAR[7]+0.1896221*BAR[8]-0.1973753*BAR[9]+0.3510076*BAR[10]+0.4492245*BAR[11]-0.09004608*BAR[12]+0.002758034*BAR[13]+0.03157447*BAR[14]+0.02175433*BAR[15]-0.399723*BAR[16]-0.2736914*BAR[17]+0.1198452*BAR[18]+0.2808644*BAR[19]-0.06968442*BAR[20]-0.5771574*BAR[21]+0.3748633*BAR[22]-0.2721373*BAR[23]-0.2329663*BAR[24]+0.07683773 );
   double Syndrome1_14=Sigmoid1( 0.094418*BAR[1]+0.2155959*BAR[2]-0.4787674*BAR[3]+0.3605456*BAR[4]+0.06799955*BAR[5]+0.607367*BAR[6]-0.3518007*BAR[7]+0.1633829*BAR[8]+0.3040094*BAR[9]+0.3707297*BAR[10]+0.02556368*BAR[11]-0.0885786*BAR[12]-0.3713907*BAR[13]-0.2014098*BAR[14]-0.289242*BAR[15]-0.09950806*BAR[16]-0.5361071*BAR[17]+0.4154459*BAR[18]+0.02827369*BAR[19]-0.04972957*BAR[20]-0.1700879*BAR[21]+0.2973098*BAR[22]-0.2097459*BAR[23]-0.0422597*BAR[24]+0.2318914 );
   double Syndrome1_15=Sigmoid1( 0.02161242*BAR[1]+0.5484816*BAR[2]+0.002152426*BAR[3]-0.3017516*BAR[4]+0.02010602*BAR[5]-0.8008425*BAR[6]-0.2985114*BAR[7]+0.5151479*BAR[8]+0.1572166*BAR[9]-0.04494689*BAR[10]+0.2529401*BAR[11]-0.02046412*BAR[12]-0.05892481*BAR[13]-0.1359019*BAR[14]-0.2005993*BAR[15]+0.03077302*BAR[16]+0.745619*BAR[17]-0.4197147*BAR[18]-0.1354882*BAR[19]-0.6034228*BAR[20]-0.04950687*BAR[21]-0.1093793*BAR[22]-0.46851*BAR[23]+0.2340346*BAR[24]-0.1910115 );
   double Syndrome1_16=Sigmoid1( 0.06201033*BAR[1]+0.2311719*BAR[2]-0.6587076*BAR[3]-0.1937433*BAR[4]-0.3063492*BAR[5]+0.0458253*BAR[6]+0.2621455*BAR[7]-0.3292437*BAR[8]-0.07124191*BAR[9]+0.03962434*BAR[10]-0.03539502*BAR[11]+0.1602975*BAR[12]+0.1252141*BAR[13]-0.1939677*BAR[14]-0.3524359*BAR[15]-0.02675135*BAR[16]-0.1550312*BAR[17]+0.2015329*BAR[18]-0.1383009*BAR[19]+0.3079963*BAR[20]+0.06971535*BAR[21]-0.2415089*BAR[22]-0.03791533*BAR[23]+0.01494107*BAR[24]+0.01395546 );
   double Syndrome1_17=Sigmoid1( -0.03211073*BAR[1]-0.2057187*BAR[2]-0.2208917*BAR[3]+0.1034868*BAR[4]+0.003785761*BAR[5]-0.1510143*BAR[6]-0.04637882*BAR[7]-0.01963908*BAR[8]-0.3622932*BAR[9]+0.03135398*BAR[10]-0.1296021*BAR[11]-0.2571803*BAR[12]+0.02485986*BAR[13]-0.05831699*BAR[14]+0.2441404*BAR[15]+0.4313999*BAR[16]-0.05117986*BAR[17]-0.06832605*BAR[18]-0.01433043*BAR[19]-0.3331767*BAR[20]-0.09270683*BAR[21]+0.1077102*BAR[22]+0.0517161*BAR[23]+0.1463209*BAR[24]+0.08033083 );
   double Syndrome1_18=Sigmoid1( -0.01044874*BAR[1]+0.8255618*BAR[2]-0.3581862*BAR[3]+0.2379437*BAR[4]-0.05247816*BAR[5]+0.3858318*BAR[6]-0.04216846*BAR[7]+0.2305764*BAR[8]-0.2754549*BAR[9]+0.1255125*BAR[10]-0.1954638*BAR[11]+0.04934186*BAR[12]-0.08713531*BAR[13]+0.08193728*BAR[14]-0.01578137*BAR[15]+0.04301662*BAR[16]-0.01941852*BAR[17]+0.0321704*BAR[18]-0.4490997*BAR[19]-0.2165072*BAR[20]+0.5094138*BAR[21]-0.08077756*BAR[22]-0.1167052*BAR[23]+0.008337143*BAR[24]-0.1847742 );
   double Syndrome1_19=Sigmoid1( 0.07863438*BAR[1]+0.6541001*BAR[2]-0.0287532*BAR[3]-0.07992863*BAR[4]-0.1936443*BAR[5]+0.2021953*BAR[6]+0.5814793*BAR[7]+0.1076662*BAR[8]-0.2505759*BAR[9]-0.1958519*BAR[10]+0.2982949*BAR[11]-0.130183*BAR[12]-0.2418064*BAR[13]-0.03213368*BAR[14]-0.1050228*BAR[15]-0.04116086*BAR[16]+0.1059578*BAR[17]-0.09407587*BAR[18]+0.2511382*BAR[19]+0.03090675*BAR[20]-0.2050715*BAR[21]+0.07968493*BAR[22]-0.1085312*BAR[23]-0.3073632*BAR[24]+0.1479857 );
   double Syndrome1_20=Sigmoid1( 0.01779699*BAR[1]+0.1517631*BAR[2]+0.1832252*BAR[3]+0.4329565*BAR[4]-0.1528609*BAR[5]-0.2424133*BAR[6]+0.1942621*BAR[7]+0.1390828*BAR[8]-0.3387062*BAR[9]+0.3891163*BAR[10]+0.3485644*BAR[11]+0.06489421*BAR[12]-0.01458877*BAR[13]-0.1127466*BAR[14]+0.1122861*BAR[15]-0.1973242*BAR[16]+0.4340822*BAR[17]-0.633949*BAR[18]+0.1276167*BAR[19]+0.2476585*BAR[20]-0.4445719*BAR[21]+0.6248969*BAR[22]-0.2169943*BAR[23]-0.501359*BAR[24]-0.1358235 );

//--- syndromes of the 2nd level:
   double Syndrome2_1=Sigmoid2( 0.2332734*Syndrome1_1-0.2002641*Syndrome1_2-0.03174414*Syndrome1_3-0.3868614*Syndrome1_4-0.1933812*Syndrome1_5-0.2366997*Syndrome1_6+0.3920829*Syndrome1_7+0.1015497*Syndrome1_8-0.1333193*Syndrome1_9+0.05584235*Syndrome1_10-0.2983295*Syndrome1_11+0.1034668*Syndrome1_12-0.4040487*Syndrome1_13-0.2103508*Syndrome1_14-0.2480657*Syndrome1_15-0.1906435*Syndrome1_16+0.2692898*Syndrome1_17+0.2760854*Syndrome1_18-0.1738693*Syndrome1_19-0.1861307*Syndrome1_20-0.07152162 );
   double Syndrome2_2=Sigmoid2( -0.1242675*Syndrome1_1+0.05587832*Syndrome1_2+0.1567961*Syndrome1_3+0.1077346*Syndrome1_4-0.2112047*Syndrome1_5+0.04008683*Syndrome1_6-0.1716478*Syndrome1_7+0.3083204*Syndrome1_8-0.1864694*Syndrome1_9+0.08867304*Syndrome1_10-0.06801239*Syndrome1_11-0.1810985*Syndrome1_12-0.05133555*Syndrome1_13+0.2981661*Syndrome1_14-0.01543425*Syndrome1_15-0.1859617*Syndrome1_16+0.027973*Syndrome1_17-0.1715439*Syndrome1_18-0.1249511*Syndrome1_19+0.5925598*Syndrome1_20-0.279602 );
   double Syndrome2_3=Sigmoid2( -0.4745722*Syndrome1_1-0.1248492*Syndrome1_2-0.1128288*Syndrome1_3+0.1485692*Syndrome1_4-0.3948999*Syndrome1_5+0.2633227*Syndrome1_6-0.2046695*Syndrome1_7-0.03632757*Syndrome1_8+0.259578*Syndrome1_9-0.07442582*Syndrome1_10+0.06552354*Syndrome1_11-0.2452848*Syndrome1_12-0.1599011*Syndrome1_13+0.1749917*Syndrome1_14-0.07113215*Syndrome1_15-0.1524421*Syndrome1_16+0.3606906*Syndrome1_17+0.3524929*Syndrome1_18+0.1315838*Syndrome1_19+0.1981817*Syndrome1_20+0.0126604 );
   double Syndrome2_4=Sigmoid2( -0.3605324*Syndrome1_1+0.2803221*Syndrome1_2+0.07412126*Syndrome1_3+0.2101911*Syndrome1_4-0.1933928*Syndrome1_5-0.2068641*Syndrome1_6+0.1302721*Syndrome1_7+0.04962961*Syndrome1_8+0.2879501*Syndrome1_9-0.04214102*Syndrome1_10-0.02194729*Syndrome1_11-0.0501424*Syndrome1_12+0.007969459*Syndrome1_13+0.1151657*Syndrome1_14+0.04063402*Syndrome1_15+0.1461606*Syndrome1_16-0.07482237*Syndrome1_17-0.3319329*Syndrome1_18+0.2494595*Syndrome1_19-0.09345333*Syndrome1_20-0.1831799 );
   double Syndrome2_5=Sigmoid2( -0.03081687*Syndrome1_1-0.419345*Syndrome1_2-0.01301429*Syndrome1_3+0.008855551*Syndrome1_4+0.2869771*Syndrome1_5+0.06881366*Syndrome1_6-0.1612982*Syndrome1_7-0.491662*Syndrome1_8+0.04266098*Syndrome1_9-0.7546657*Syndrome1_10+0.0472151*Syndrome1_11-0.5099863*Syndrome1_12+0.1196823*Syndrome1_13+0.2611973*Syndrome1_14-0.0241531*Syndrome1_15-0.5843646*Syndrome1_16+0.08374172*Syndrome1_17+0.041931*Syndrome1_18-0.181801*Syndrome1_19+0.6314354*Syndrome1_20+0.2967799 );
   double Syndrome2_6=Sigmoid2( 0.2783457*Syndrome1_1+0.05858535*Syndrome1_2+0.03348543*Syndrome1_3-0.09202126*Syndrome1_4+0.09466362*Syndrome1_5-0.01946918*Syndrome1_6-0.008507644*Syndrome1_7+0.1967683*Syndrome1_8-0.1593684*Syndrome1_9+0.2202749*Syndrome1_10-0.2754305*Syndrome1_11-0.08108314*Syndrome1_12+0.1606592*Syndrome1_13+0.03723634*Syndrome1_14+0.3494412*Syndrome1_15-0.139782*Syndrome1_16+0.03641316*Syndrome1_17-0.1216527*Syndrome1_18-0.2194063*Syndrome1_19+0.3015033*Syndrome1_20-0.1307777 );
   double Syndrome2_7=Sigmoid2( -0.1451617*Syndrome1_1-0.1851998*Syndrome1_2-0.2149245*Syndrome1_3-0.05804037*Syndrome1_4-0.03970402*Syndrome1_5+2.506166E-6*Syndrome1_6+0.223578*Syndrome1_7-0.1718342*Syndrome1_8+0.001228896*Syndrome1_9-0.03911417*Syndrome1_10+0.3167912*Syndrome1_11+0.2213001*Syndrome1_12-0.3518667*Syndrome1_13-0.6146168*Syndrome1_14-0.1061097*Syndrome1_15-0.3044312*Syndrome1_16-0.04269538*Syndrome1_17-0.1753355*Syndrome1_18+0.1989161*Syndrome1_19-0.3667244*Syndrome1_20+0.2514035 );
   double Syndrome2_8=Sigmoid2( -0.1430153*Syndrome1_1-Syndrome1_2+0.02704678*Syndrome1_3+0.09941091*Syndrome1_4+0.07057924*Syndrome1_5-0.3370984*Syndrome1_6+0.1565579*Syndrome1_7-0.6226992*Syndrome1_8-0.4750121*Syndrome1_9+0.0914355*Syndrome1_10+0.7518402*Syndrome1_11-0.3350138*Syndrome1_12-0.3099903*Syndrome1_13+0.01266479*Syndrome1_14-0.7965527*Syndrome1_15-0.1753905*Syndrome1_16-0.1435609*Syndrome1_17+0.1683903*Syndrome1_18+0.1800467*Syndrome1_19+0.02699256*Syndrome1_20+0.3138063 );
   double Syndrome2_9=Sigmoid2( -0.2611458*Syndrome1_1-0.03994129*Syndrome1_2-0.2299157*Syndrome1_3+0.3549923*Syndrome1_4-0.001759748*Syndrome1_5-0.1117837*Syndrome1_6+0.03037107*Syndrome1_7+0.2023677*Syndrome1_8+0.2628252*Syndrome1_9+0.09683131*Syndrome1_10+0.2576693*Syndrome1_11-0.06357097*Syndrome1_12-0.2162403*Syndrome1_13-0.2190126*Syndrome1_14-0.1675369*Syndrome1_15-0.2458067*Syndrome1_16-0.06660707*Syndrome1_17-0.2096998*Syndrome1_18+0.2432118*Syndrome1_19+0.06210691*Syndrome1_20+0.1555794 );
   double Syndrome2_10=Sigmoid2( 0.1120118*Syndrome1_1-0.09789048*Syndrome1_2-0.1146162*Syndrome1_3-0.02268722*Syndrome1_4-0.4754501*Syndrome1_5+0.1567527*Syndrome1_6+0.4281512*Syndrome1_7+0.1428995*Syndrome1_8+0.4317052*Syndrome1_9-0.1987304*Syndrome1_10-0.3471439*Syndrome1_11-0.2485701*Syndrome1_12+0.2200699*Syndrome1_13-0.1804247*Syndrome1_14+0.5553524*Syndrome1_15+0.004284344*Syndrome1_16-0.5408193*Syndrome1_17-0.2304406*Syndrome1_18+0.2462995*Syndrome1_19+0.1687378*Syndrome1_20+0.480715 );
   double Syndrome2_11=Sigmoid2( 0.2892572*Syndrome1_1+0.2819389*Syndrome1_2-0.2116477*Syndrome1_3-0.1031269*Syndrome1_4-0.2198152*Syndrome1_5-0.2882532*Syndrome1_6-0.7462316*Syndrome1_7+0.7820893*Syndrome1_8-0.05574411*Syndrome1_9-0.1144354*Syndrome1_10-0.1073154*Syndrome1_11+0.5092962*Syndrome1_12-0.07017706*Syndrome1_13-0.5550667*Syndrome1_14-0.5170746*Syndrome1_15-0.1299864*Syndrome1_16+0.03325708*Syndrome1_17-0.5107772*Syndrome1_18+0.04024922*Syndrome1_19+0.1836878*Syndrome1_20+0.0346345 );
   double Syndrome2_12=Sigmoid2( -0.10614*Syndrome1_1+0.06027444*Syndrome1_2+0.08108542*Syndrome1_3-0.1568731*Syndrome1_4+0.1509192*Syndrome1_5-0.1630516*Syndrome1_6+0.01426157*Syndrome1_7+0.02186926*Syndrome1_8+0.1099893*Syndrome1_9-0.02269597*Syndrome1_10-0.04576464*Syndrome1_11-0.161096*Syndrome1_12-0.1901706*Syndrome1_13-0.02513908*Syndrome1_14+0.1317106*Syndrome1_15-0.06866668*Syndrome1_16+0.1083753*Syndrome1_17+0.1449683*Syndrome1_18+0.006118122*Syndrome1_19+0.1255394*Syndrome1_20-0.3822223 );
   double Syndrome2_13=Sigmoid2( -0.01638931*Syndrome1_1+0.1172011*Syndrome1_2-0.1022018*Syndrome1_3+0.1098846*Syndrome1_4+0.3456185*Syndrome1_5-0.276273*Syndrome1_6-0.1697723*Syndrome1_7-0.1394644*Syndrome1_8+0.0530486*Syndrome1_9+0.04139024*Syndrome1_10-0.02131393*Syndrome1_11+0.1144992*Syndrome1_12-0.1791101*Syndrome1_13+0.124498*Syndrome1_14+0.2169005*Syndrome1_15+0.06764794*Syndrome1_16+0.3542189*Syndrome1_17+0.0647957*Syndrome1_18+0.01778502*Syndrome1_19-0.0183728*Syndrome1_20-0.09863564 );
   double Syndrome2_14=Sigmoid2( 0.1046498*Syndrome1_1+0.1199886*Syndrome1_2-0.3787079*Syndrome1_3+0.568437*Syndrome1_4-0.09216721*Syndrome1_5-0.07998162*Syndrome1_6-0.1422648*Syndrome1_7-0.220407*Syndrome1_8+0.00417607*Syndrome1_9+0.2042087*Syndrome1_10+0.2614584*Syndrome1_11+0.04491196*Syndrome1_12+0.1860093*Syndrome1_13-0.1642074*Syndrome1_14+0.3918036*Syndrome1_15+0.05427575*Syndrome1_16-0.0002294437*Syndrome1_17+0.008295977*Syndrome1_18-0.2818146*Syndrome1_19-0.3877438*Syndrome1_20+0.03536745 );
   double Syndrome2_15=Sigmoid2( -0.1754033*Syndrome1_1-0.0528489*Syndrome1_2-0.1744897*Syndrome1_3+0.1113354*Syndrome1_4+0.1185713*Syndrome1_5-0.0231303*Syndrome1_6+0.006316248*Syndrome1_7-0.08525342*Syndrome1_8+0.1568578*Syndrome1_9+0.2965699*Syndrome1_10+0.2781587*Syndrome1_11+0.2391527*Syndrome1_12-0.08555941*Syndrome1_13-0.2362186*Syndrome1_14+0.1128907*Syndrome1_15-0.04770778*Syndrome1_16-0.0139725*Syndrome1_17+0.1079882*Syndrome1_18-0.09141354*Syndrome1_19+0.3320866*Syndrome1_20-0.3015116 );
   double Syndrome2_16=Sigmoid2( 0.1962015*Syndrome1_1+0.0192374*Syndrome1_2-0.1578716*Syndrome1_3+0.03360523*Syndrome1_4+0.04818176*Syndrome1_5+0.2462966*Syndrome1_6-0.2103649*Syndrome1_7+0.01318523*Syndrome1_8-0.09349868*Syndrome1_9+0.08476428*Syndrome1_10-0.06272572*Syndrome1_11+0.2246324*Syndrome1_12+0.2539908*Syndrome1_13-0.2059217*Syndrome1_14-0.08641216*Syndrome1_15-0.09780023*Syndrome1_16+0.0005770256*Syndrome1_17-0.2842666*Syndrome1_18-0.05383059*Syndrome1_19-0.2822465*Syndrome1_20+0.2277268 );
   double Syndrome2_17=Sigmoid2( 0.5981864*Syndrome1_1+0.5172131*Syndrome1_2-0.2310352*Syndrome1_3-0.1814138*Syndrome1_4-0.2148922*Syndrome1_5+0.562911*Syndrome1_6+0.5865576*Syndrome1_7-0.2790301*Syndrome1_8-0.3841165*Syndrome1_9+0.3223535*Syndrome1_10+0.2096305*Syndrome1_11+0.08284206*Syndrome1_12+0.7050048*Syndrome1_13+0.4129859*Syndrome1_14+0.2116682*Syndrome1_15+0.2213966*Syndrome1_16-0.1637594*Syndrome1_17+0.1191863*Syndrome1_18-0.6626714*Syndrome1_19-0.9127383*Syndrome1_20-0.1505798 );
   double Syndrome2_18=Sigmoid2( -0.008298698*Syndrome1_1-0.1847953*Syndrome1_2-0.1930849*Syndrome1_3-0.1005524*Syndrome1_4+0.0737519*Syndrome1_5+0.04218475*Syndrome1_6-0.422835*Syndrome1_7+0.06019862*Syndrome1_8-0.2056148*Syndrome1_9+0.3398327*Syndrome1_10-0.2526269*Syndrome1_11-0.06098709*Syndrome1_12-0.1447722*Syndrome1_13-0.05216306*Syndrome1_14-0.09496115*Syndrome1_15+0.2071376*Syndrome1_16+0.03088453*Syndrome1_17-0.521363*Syndrome1_18-0.06449924*Syndrome1_19-0.4105364*Syndrome1_20+0.3204305 );
   double Syndrome2_19=Sigmoid2( -0.1376712*Syndrome1_1-0.0153131*Syndrome1_2+0.04377801*Syndrome1_3+0.08896239*Syndrome1_4+0.03197494*Syndrome1_5-0.02259021*Syndrome1_6+0.008662836*Syndrome1_7-0.1961185*Syndrome1_8-0.0720102*Syndrome1_9+0.05738823*Syndrome1_10-0.004060962*Syndrome1_11-0.3752605*Syndrome1_12+0.02065136*Syndrome1_13+0.1263955*Syndrome1_14-0.05906902*Syndrome1_15+0.4029721*Syndrome1_16-0.159444*Syndrome1_17-0.1619136*Syndrome1_18+0.3338208*Syndrome1_19-0.0656369*Syndrome1_20+0.1602566 );
   double Syndrome2_20=Sigmoid2( -0.003900121*Syndrome1_1+0.3159288*Syndrome1_2+0.2550703*Syndrome1_3+0.05409481*Syndrome1_4+0.06660215*Syndrome1_5-0.1948439*Syndrome1_6-0.370153*Syndrome1_7+0.5337713*Syndrome1_8-0.06716464*Syndrome1_9+0.550526*Syndrome1_10+0.4723933*Syndrome1_11+0.09457724*Syndrome1_12+0.5613732*Syndrome1_13+0.3709611*Syndrome1_14-0.07680532*Syndrome1_15-0.5097623*Syndrome1_16+0.4023384*Syndrome1_17+0.2330064*Syndrome1_18-0.09448317*Syndrome1_19+0.2668969*Syndrome1_20-0.2110061 );

//--- syndromes of the 3rd level:
   double Syndrome3_1=Sigmoid3( -0.05101856*Syndrome2_1-0.04933448*Syndrome2_2+0.03248681*Syndrome2_3-0.05835526*Syndrome2_4-0.01888579*Syndrome2_5-0.07940733*Syndrome2_6-0.04341835*Syndrome2_7-0.07906266*Syndrome2_8+0.2054683*Syndrome2_9+0.1553352*Syndrome2_10-0.07296721*Syndrome2_11-0.01849408*Syndrome2_12-0.07505544*Syndrome2_13+0.08666297*Syndrome2_14-0.2001411*Syndrome2_15+0.07931387*Syndrome2_16+0.1598745*Syndrome2_17+0.01308129*Syndrome2_18+0.159161*Syndrome2_19+0.1903208*Syndrome2_20+0.0190388 );
   double Syndrome3_2=Sigmoid3( 0.0643296*Syndrome2_1+0.3451192*Syndrome2_2-0.1247545*Syndrome2_3+0.03276825*Syndrome2_4+0.303136*Syndrome2_5+0.03152885*Syndrome2_6+0.1118743*Syndrome2_7-0.3860323*Syndrome2_8-0.08593427*Syndrome2_9-0.2664599*Syndrome2_10+0.213205*Syndrome2_11-0.0977626*Syndrome2_12-0.2923501*Syndrome2_13-0.3133417*Syndrome2_14-0.1915279*Syndrome2_15+0.4333939*Syndrome2_16+0.02110274*Syndrome2_17+0.5802879*Syndrome2_18+0.03386912*Syndrome2_19+0.08908307*Syndrome2_20+0.06071822 );
   double Syndrome3_3=Sigmoid3( -0.08613513*Syndrome2_1+0.1200513*Syndrome2_2+0.3818525*Syndrome2_3-0.09603316*Syndrome2_4-0.2353039*Syndrome2_5-0.1816488*Syndrome2_6+0.002517342*Syndrome2_7-0.2414117*Syndrome2_8+0.2011739*Syndrome2_9-0.3057347*Syndrome2_10-0.4593749*Syndrome2_11-0.2228307*Syndrome2_12+0.03512295*Syndrome2_13+0.4402955*Syndrome2_14-0.1967632*Syndrome2_15+0.07873345*Syndrome2_16+0.1981131*Syndrome2_17-0.2677957*Syndrome2_18+0.1719814*Syndrome2_19-0.474854*Syndrome2_20+0.01101439 );
   double Syndrome3_4=Sigmoid3( 0.02534361*Syndrome2_1+0.1845266*Syndrome2_2+0.149674*Syndrome2_3-0.1454014*Syndrome2_4+0.00701888*Syndrome2_5+0.08219463*Syndrome2_6+0.05163066*Syndrome2_7-0.1836077*Syndrome2_8+0.1429968*Syndrome2_9+0.518382*Syndrome2_10-0.00966637*Syndrome2_11-0.1674386*Syndrome2_12+0.1387497*Syndrome2_13+0.1385897*Syndrome2_14-0.01148864*Syndrome2_15+0.3751494*Syndrome2_16-0.08906862*Syndrome2_17-0.06286599*Syndrome2_18+0.2061662*Syndrome2_19-0.07524439*Syndrome2_20-0.08077133 );
   double Syndrome3_5=Sigmoid3( 0.3856083*Syndrome2_1-0.01700347*Syndrome2_2-0.1044575*Syndrome2_3+0.111998*Syndrome2_4-0.5157402*Syndrome2_5-0.05508286*Syndrome2_6-0.3101066*Syndrome2_7-0.5261913*Syndrome2_8-0.05983765*Syndrome2_9+0.1723307*Syndrome2_10-0.2564277*Syndrome2_11+0.06385356*Syndrome2_12-0.07245655*Syndrome2_13+0.1154206*Syndrome2_14-0.3492871*Syndrome2_15+0.136372*Syndrome2_16+0.3627071*Syndrome2_17-0.3074959*Syndrome2_18+0.4425845*Syndrome2_19-0.9329191*Syndrome2_20+0.01476912 );
   double Syndrome3_6=Sigmoid3( 0.5246867*Syndrome2_1-0.2347829*Syndrome2_2+0.01062111*Syndrome2_3+0.2374777*Syndrome2_4-0.02361662*Syndrome2_5+0.1804156*Syndrome2_6+0.07669501*Syndrome2_7-0.142881*Syndrome2_8+0.2566245*Syndrome2_9+0.1024709*Syndrome2_10-0.04695484*Syndrome2_11-0.004103919*Syndrome2_12+0.3340242*Syndrome2_13-0.3702791*Syndrome2_14+0.1852374*Syndrome2_15+0.02175477*Syndrome2_16+0.09901489*Syndrome2_17-0.1502062*Syndrome2_18+0.3814779*Syndrome2_19-0.06319473*Syndrome2_20+0.2657273 );
   double Syndrome3_7=Sigmoid3( 0.1613003*Syndrome2_1-0.2738772*Syndrome2_2-0.03304096*Syndrome2_3+0.3934855*Syndrome2_4+0.3955218*Syndrome2_5-0.3004892*Syndrome2_6+0.1339742*Syndrome2_7+0.09475601*Syndrome2_8+0.03064043*Syndrome2_9-0.7264652*Syndrome2_10-0.4579849*Syndrome2_11-0.1183059*Syndrome2_12+0.2197721*Syndrome2_13-0.08493897*Syndrome2_14+0.2115426*Syndrome2_15-0.07834542*Syndrome2_16-0.3884689*Syndrome2_17-0.101394*Syndrome2_18+0.1002519*Syndrome2_19-0.07787764*Syndrome2_20+0.3529212 );
   double Syndrome3_8=Sigmoid3( -0.3544801*Syndrome2_1+0.03471621*Syndrome2_2-0.2373467*Syndrome2_3-0.2836286*Syndrome2_4+0.01646966*Syndrome2_5+0.06978795*Syndrome2_6-0.03310004*Syndrome2_7+0.01844743*Syndrome2_8+0.05259214*Syndrome2_9-0.05343668*Syndrome2_10+0.3971725*Syndrome2_11-0.08770485*Syndrome2_12-0.2040168*Syndrome2_13+0.1109144*Syndrome2_14-0.06249888*Syndrome2_15-0.5860764*Syndrome2_16+0.1217078*Syndrome2_17+0.2471277*Syndrome2_18-0.03716509*Syndrome2_19-0.1908655*Syndrome2_20+0.03838157 );
   double Syndrome3_9=Sigmoid3( 0.1542789*Syndrome2_1+0.3505224*Syndrome2_2+0.06042741*Syndrome2_3+0.08956298*Syndrome2_4-0.03655836*Syndrome2_5-0.3083843*Syndrome2_6+0.2483124*Syndrome2_7-0.1132483*Syndrome2_8-0.3571556*Syndrome2_9-0.04335312*Syndrome2_10+0.005499069*Syndrome2_11+0.371572*Syndrome2_12-0.1199554*Syndrome2_13+0.1160574*Syndrome2_14-0.01656827*Syndrome2_15+0.09481092*Syndrome2_16-0.07926448*Syndrome2_17+0.3847227*Syndrome2_18+0.1039986*Syndrome2_19-0.02874756*Syndrome2_20-0.2311832 );
   double Syndrome3_10=Sigmoid3( -0.5099882*Syndrome2_1-0.2619184*Syndrome2_2+0.2441412*Syndrome2_3-0.02311796*Syndrome2_4+0.004243354*Syndrome2_5-0.04681544*Syndrome2_6+0.1402575*Syndrome2_7-0.03166823*Syndrome2_8-0.2629028*Syndrome2_9-0.03275445*Syndrome2_10-0.311464*Syndrome2_11+0.3158014*Syndrome2_12-0.04689252*Syndrome2_13+0.1556217*Syndrome2_14-0.02266529*Syndrome2_15-0.15192*Syndrome2_16+0.02253294*Syndrome2_17+0.04638374*Syndrome2_18-0.4847055*Syndrome2_19-0.0543578*Syndrome2_20-0.4383866 );
   double Syndrome3_11=Sigmoid3( 0.09181526*Syndrome2_1-0.009475656*Syndrome2_2+0.08283823*Syndrome2_3+0.06638021*Syndrome2_4-0.04110251*Syndrome2_5+0.03041244*Syndrome2_6-0.2266526*Syndrome2_7+0.3537511*Syndrome2_8+0.2091044*Syndrome2_9-0.2312607*Syndrome2_10-0.01409533*Syndrome2_11-0.06294888*Syndrome2_12+0.1980267*Syndrome2_13+0.07864135*Syndrome2_14-0.01312789*Syndrome2_15+0.02964603*Syndrome2_16-0.1720168*Syndrome2_17-0.01523064*Syndrome2_18+0.07354444*Syndrome2_19+0.1534344*Syndrome2_20+0.04784121 );
   double Syndrome3_12=Sigmoid3( -0.01962976*Syndrome2_1-0.1254692*Syndrome2_2+0.01237085*Syndrome2_3-0.006583595*Syndrome2_4-0.06446695*Syndrome2_5-0.1581757*Syndrome2_6-0.01416831*Syndrome2_7+0.08909909*Syndrome2_8+0.02427519*Syndrome2_9+0.06101634*Syndrome2_10-0.07296847*Syndrome2_11-0.02960677*Syndrome2_12+0.1195403*Syndrome2_13+0.007260199*Syndrome2_14-0.005008513*Syndrome2_15+0.07686368*Syndrome2_16-0.1097991*Syndrome2_17+0.02348211*Syndrome2_18-0.01508969*Syndrome2_19+0.06078456*Syndrome2_20+0.1424098 );
   double Syndrome3_13=Sigmoid3( -0.1845686*Syndrome2_1-0.1120369*Syndrome2_2+0.1346949*Syndrome2_3+0.2425685*Syndrome2_4+0.1310953*Syndrome2_5-0.1957272*Syndrome2_6+0.2163845*Syndrome2_7+0.04189415*Syndrome2_8+0.05685329*Syndrome2_9-0.1108158*Syndrome2_10-0.04702755*Syndrome2_11-0.2698838*Syndrome2_12+0.05045844*Syndrome2_13+0.1487544*Syndrome2_14+7.648221E-5*Syndrome2_15-0.04902162*Syndrome2_16+0.3119571*Syndrome2_17-0.2076546*Syndrome2_18+0.1465537*Syndrome2_19+0.2386554*Syndrome2_20+0.09121808 );
   double Syndrome3_14=Sigmoid3( 0.015057*Syndrome2_1-0.07630379*Syndrome2_2+0.10373*Syndrome2_3-0.01276504*Syndrome2_4+0.01637872*Syndrome2_5+0.1570177*Syndrome2_6+0.02290879*Syndrome2_7+0.1426407*Syndrome2_8-0.3037595*Syndrome2_9-0.1183627*Syndrome2_10-0.05010238*Syndrome2_11-0.06874149*Syndrome2_12+0.0325584*Syndrome2_13-0.1127614*Syndrome2_14+0.1010367*Syndrome2_15+0.2743505*Syndrome2_16+0.02752565*Syndrome2_17-0.01011515*Syndrome2_18-0.1072115*Syndrome2_19-0.1723324*Syndrome2_20-0.1862434 );
   double Syndrome3_15=Sigmoid3( -0.0602835*Syndrome2_1+0.1044827*Syndrome2_2-0.03398157*Syndrome2_3+0.1103081*Syndrome2_4-0.2517793*Syndrome2_5-0.1388755*Syndrome2_6+0.1680355*Syndrome2_7+0.08541053*Syndrome2_8+0.2264198*Syndrome2_9+0.1319854*Syndrome2_10+0.2397746*Syndrome2_11+0.04893836*Syndrome2_12+0.07067535*Syndrome2_13+0.03666123*Syndrome2_14-0.2249698*Syndrome2_15+0.1039975*Syndrome2_16+0.03130547*Syndrome2_17+0.1295152*Syndrome2_18-0.1380298*Syndrome2_19-0.2716908*Syndrome2_20+0.3049682 );
   double Syndrome3_16=Sigmoid3( 0.006898584*Syndrome2_1+0.172121*Syndrome2_2+0.08287619*Syndrome2_3-0.2843233*Syndrome2_4+0.3360839*Syndrome2_5-0.06360124*Syndrome2_6+0.08605669*Syndrome2_7+0.1303328*Syndrome2_8+0.176666*Syndrome2_9+0.3064248*Syndrome2_10+0.03492442*Syndrome2_11-0.1337793*Syndrome2_12+0.2166045*Syndrome2_13+0.1651906*Syndrome2_14-0.2159452*Syndrome2_15-0.02087162*Syndrome2_16-0.1321865*Syndrome2_17+0.02330898*Syndrome2_18-0.1607926*Syndrome2_19+0.100959*Syndrome2_20+0.3113509 );
   double Syndrome3_17=Sigmoid3( 0.2484581*Syndrome2_1+0.07501616*Syndrome2_2-0.2955785*Syndrome2_3-0.06893355*Syndrome2_4-0.110545*Syndrome2_5+0.009258383*Syndrome2_6-0.04150206*Syndrome2_7-0.1581711*Syndrome2_8-0.1503464*Syndrome2_9-0.1641756*Syndrome2_10+0.2800875*Syndrome2_11+0.1470316*Syndrome2_12+0.08529772*Syndrome2_13-0.07939056*Syndrome2_14+0.1105667*Syndrome2_15-0.003909521*Syndrome2_16-0.1663841*Syndrome2_17+0.1384012*Syndrome2_18-0.2260507*Syndrome2_19-0.1310463*Syndrome2_20+0.03011392 );
   double Syndrome3_18=Sigmoid3( 0.2167049*Syndrome2_1+0.1083723*Syndrome2_2+0.03713056*Syndrome2_3-0.07394339*Syndrome2_4-0.08689396*Syndrome2_5+0.1893489*Syndrome2_6-0.004869457*Syndrome2_7+0.06987588*Syndrome2_8-0.1505099*Syndrome2_9+0.1717843*Syndrome2_10+0.07792218*Syndrome2_11+0.02835098*Syndrome2_12+0.03617713*Syndrome2_13+0.1599271*Syndrome2_14-0.1617647*Syndrome2_15-0.04720658*Syndrome2_16+0.004165665*Syndrome2_17-0.1073883*Syndrome2_18+0.06164433*Syndrome2_19+0.01017194*Syndrome2_20-0.1073146 );
   double Syndrome3_19=Sigmoid3( 0.1966043*Syndrome2_1-0.06785608*Syndrome2_2-0.02568222*Syndrome2_3+0.2323583*Syndrome2_4-0.1949882*Syndrome2_5-0.0180097*Syndrome2_6-0.1995831*Syndrome2_7-0.3007537*Syndrome2_8+0.03133066*Syndrome2_9-0.3836962*Syndrome2_10+0.8646971*Syndrome2_11-0.04459784*Syndrome2_12+0.1127359*Syndrome2_13+0.3645059*Syndrome2_14+0.3924035*Syndrome2_15+0.2070317*Syndrome2_16-0.1975317*Syndrome2_17+0.249992*Syndrome2_18-0.1090982*Syndrome2_19+0.9234442*Syndrome2_20+0.0260936 );
   double Syndrome3_20=Sigmoid3( -0.1054238*Syndrome2_1+0.01094678*Syndrome2_2+0.1854347*Syndrome2_3-0.03105933*Syndrome2_4-0.1428708*Syndrome2_5+0.1660853*Syndrome2_6-0.0540761*Syndrome2_7+0.08364562*Syndrome2_8+0.01462638*Syndrome2_9+0.05958234*Syndrome2_10+0.05540805*Syndrome2_11+0.1415959*Syndrome2_12-0.2088391*Syndrome2_13-0.02437577*Syndrome2_14+0.03789431*Syndrome2_15+0.1342704*Syndrome2_16+0.02136465*Syndrome2_17+0.1529594*Syndrome2_18-0.2515772*Syndrome2_19-0.009984408*Syndrome2_20-0.02554057 );

//--- final syndromes:
   BAR[0]=0.377357*Syndrome3_1-0.1995524*Syndrome3_2+0.44664*Syndrome3_3-0.2634062*Syndrome3_4-0.1150927*Syndrome3_5-0.3349093*Syndrome3_6-0.3639574*Syndrome3_7+0.2705039*Syndrome3_8+0.5313437*Syndrome3_9+0.2664694*Syndrome3_10+0.1713557*Syndrome3_11+0.1208919*Syndrome3_12-0.4120659*Syndrome3_13+0.3021899*Syndrome3_14+0.4149051*Syndrome3_15+0.7103375*Syndrome3_16+0.1180793*Syndrome3_17-0.2354599*Syndrome3_18-0.1013937*Syndrome3_19+0.3054902*Syndrome3_20+0.03919306;

//--- postprocessing of the final syndromes:
   BAR[0]=((BAR[0]*0.0180000001564622)+0.000599999912083149)/2;

   return (BAR[0]);
  }
//+------------------------------------------------------------------+

double Prognosis;

//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
void OnTick()
  {
//--- receive a price forecast from a neural network
   Prognosis=CalcNeuroNet();
//--- perform necessary trade actions
   Trade();
  }
//+------------------------------------------------------------------+
void Trade()
  {
   //--- close an opened position if it does not fit the forecast
   if(PositionSelect(_Symbol))
     {
      long type=PositionGetInteger(POSITION_TYPE);
      bool close=false;
      if((type == POSITION_TYPE_BUY)  && (Prognosis <= 0)) close = true;
      if((type == POSITION_TYPE_SELL) && (Prognosis >= 0)) close = true;
      if(close)
        {
         CTrade trade;
         trade.PositionClose(_Symbol);
        }
     }

   //--- if there are no positions, then open by the forecast
   if((Prognosis!=0) && (!PositionSelect(_Symbol)))
     {
      CTrade trade;
      if(Prognosis >  MinPrognosis) trade.Buy (Lots);
      if(Prognosis < -MinPrognosis) trade.Sell(Lots);
     }
  }
```

**Testing**

Launch the Expert on the same period that provided the data for training the neural network. I remind that this Expert was written for EURUSD, H1 (the learning time is nearly 10 months).

There is no point to enter a deal when a forecast profit is a value comparable with the spread. The Expert has a built-in filter for such a case. Set the input parameter MinPrognosis at 0.0005.

The constant trade volume is 0.1 lot.

We received the following results:

![Fig. 21. Statistics of testing the Expert Advisor in MetaTester](https://c.mql5.com/2/17/Figure21_Statistics_of_testing_Expert_Advisor_MetaTester__1.png)

Fig. 21. Statistics of testing the Expert Advisor in MetaTester

![Fig. 22. Equity chart after the Expert Advisor has been tested in MetaTester](https://c.mql5.com/2/17/Figure22_Equity_chart_Testing_results_MetaTrader5.png)

Fig. 22. Equity chart after the Expert Advisor has been tested in MetaTester

Constantly growing equity shows that all the stages of developing a neural network Expert were implemented correctly.

It should be kept in mind that profit on a period of time when the Expert was learning does not guarantee profit outside of that. Creating real profitable neural network Experts requires in-depth knowledge of the neural network operational principle and significant experience in trading. In this article I showed how to use a neural network tool and now it is up to you to make it efficient.

### Conclusion

NeuroPro is a unique program. We had an opportunity to see that a neural network can be carried over from NeuroPro to the MetaTrader 5 Expert in minutes using only tools at hand.

A lot of other neural network programs do not have this advantage. That is why NeuroPro is highly recommended.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/830](https://www.mql5.com/ru/articles/830)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/830.zip "Download all attachments in the single ZIP archive")

[NeuroPro\_allOS.zip](https://www.mql5.com/en/articles/download/830/neuropro_allos.zip "Download NeuroPro_allOS.zip")(7824.54 KB)

[neuropro-export.mq5](https://www.mql5.com/en/articles/download/830/neuropro-export.mq5 "Download neuropro-export.mq5")(5.78 KB)

[neuropro.mq5](https://www.mql5.com/en/articles/download/830/neuropro.mq5 "Download neuropro.mq5")(68.46 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Trading strategy based on the improved Doji candlestick pattern recognition indicator](https://www.mql5.com/en/articles/12355)
- [Improved candlestick pattern recognition illustrated by the example of Doji](https://www.mql5.com/en/articles/9801)
- [3 Methods of Indicators Acceleration by the Example of the Linear Regression](https://www.mql5.com/en/articles/270)
- [Decreasing Memory Consumption by Auxiliary Indicators](https://www.mql5.com/en/articles/259)
- [Connecting NeuroSolutions Neuronets](https://www.mql5.com/en/articles/236)
- [Parallel Calculations in MetaTrader 5](https://www.mql5.com/en/articles/197)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/40349)**
(80)


![Эдуард](https://c.mql5.com/avatar/avatar_na2.png)

**[Эдуард](https://www.mql5.com/en/users/47rxkfn)**
\|
8 Mar 2023 at 07:46

The network has shown itself well on training, but price charts have no regularities (completely random) and it makes no sense to use indicators or networks for forecasting.


![Ivan Butko](https://c.mql5.com/avatar/2017/1/58797422-0BFA.png)

**[Ivan Butko](https://www.mql5.com/en/users/capitalplus)**
\|
8 Mar 2023 at 11:52

**Эдуард [#](https://www.mql5.com/ru/forum/37798/page7#comment_45455660):**

The network has shown itself well on training, but price charts have no regularities (completely random) and it makes no sense to use indicators or networks for forecasting.

What is surprising, having trained the network to grail even for 20 years (every candle the neural network knows where to open), on the forward literally from the very first candle - the network breaks down, 50/50.

It would seem that 140,000 candles are behind us, there will probably be patterns on the forward.... but no. What a miracle of non-stationarity.

Moreover, I exported all currencies for the last year to one dataset (40 years of "fresh" history in total, i.e. each currency pair has only the last year), I thought, well, now it will play for sure, a universal neuron that can trade on all currency pairs.

The result: the grail on all currencies for a year of backtesting, and forward.... from the first candle 50/50.

Well, at least for a month, at least for a week, at least for a day, but trade steadily!..... Yeah, no way.

![Renat Akhtyamov](https://c.mql5.com/avatar/2017/4/58E95577-1CA0.jpg)

**[Renat Akhtyamov](https://www.mql5.com/en/users/ya_programmer)**
\|
8 Mar 2023 at 12:46

It's simple

You open a trade, the price goes the other way, against the crowd or against the maximum risk.

What neuro can counteract that?

It's useless.

![Victor Castillejo](https://c.mql5.com/avatar/avatar_na2.png)

**[Victor Castillejo](https://www.mql5.com/en/users/castillejovictor)**
\|
3 Aug 2023 at 03:54

Good evening andrew, I can't get the NeuroPro software, can you tell me where I can get it, please....


![Dmitrii Shershov](https://c.mql5.com/avatar/avatar_na2.png)

**[Dmitrii Shershov](https://www.mql5.com/en/users/expert_systems)**
\|
4 Apr 2024 at 13:59

**Советник на этом примере скомпилировался, но при тестировании выходит ошибка "array out of range in" и советник закрывается.**

On debugging it shows the line - BAR\[bar\]=rate.close-zlevel;

Who can tell me what is the reason?

Who can tell me what is the reason?

![Third Generation Neural Networks: Deep Networks](https://c.mql5.com/2/12/Deep_neural_network_MetaTrader5__2.png)[Third Generation Neural Networks: Deep Networks](https://www.mql5.com/en/articles/1103)

This article is dedicated to a new and perspective direction in machine learning - deep learning or, to be precise, deep neural networks. This is a brief review of second generation neural networks, the architecture of their connections and main types, methods and rules of learning and their main disadvantages followed by the history of the third generation neural network development, their main types, peculiarities and training methods. Conducted are practical experiments on building and training a deep neural network initiated by the weights of a stacked autoencoder with real data. All the stages from selecting input data to metric derivation are discussed in detail. The last part of the article contains a software implementation of a deep neural network in an Expert Advisor with a built-in indicator based on MQL4/R.

![Programming EA's Modes Using Object-Oriented Approach](https://c.mql5.com/2/12/Expert_Advisor_modes_programming_img.png)[Programming EA's Modes Using Object-Oriented Approach](https://www.mql5.com/en/articles/1246)

This article explains the idea of multi-mode trading robot programming in MQL5. Every mode is implemented with the object-oriented approach. Instances of both mode classes hierarchy and classes for testing are provided. Multi-mode programming of trading robots is supposed to take into account all peculiarities of every operational mode of an EA written in MQL5. Functions and enumeration are created for identifying the mode.

![Trader's Statistical Cookbook: Hypotheses](https://c.mql5.com/2/12/Trader_Statistics_Recipes_MetaTrader5_Alglib_MQL5__1.png)[Trader's Statistical Cookbook: Hypotheses](https://www.mql5.com/en/articles/1240)

This article considers hypothesis - one of the basic ideas of mathematical statistics. Various hypotheses are examined and verified through examples using methods of mathematical statistics. The actual data is generalized using nonparametric methods. The Statistica package and the ported ALGLIB MQL5 numerical analysis library are used for processing data.

![Random Forests Predict Trends](https://c.mql5.com/2/11/Random_Forest_MetaTrader5.png)[Random Forests Predict Trends](https://www.mql5.com/en/articles/1165)

This article considers using the Rattle package for automatic search of patterns for predicting long and short positions of currency pairs on Forex. This article can be useful both for novice and experienced traders.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=pzgrbfoyflxswxkeyyskubzinkphlbtx&ssn=1769178398340386416&ssn_dr=0&ssn_sr=0&fv_date=1769178398&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F830&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20Cheap%20and%20Cheerful%20-%20Link%20NeuroPro%20with%20MetaTrader%205%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917839812981268&fz_uniq=5068238504015623965&sv=2552)

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