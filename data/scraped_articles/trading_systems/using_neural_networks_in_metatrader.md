---
title: Using Neural Networks In MetaTrader
url: https://www.mql5.com/en/articles/1565
categories: Trading Systems
relevance_score: 6
scraped_at: 2026-01-23T11:53:23.931976
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/1565&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=6467253706173622530)

MetaTrader 4 / Examples


### Introduction

Many of you probably have considered the possibility of using neural networks in your EA. This subject was very hot specially after 2007 Automated Trading Championship and the spectacular winning by [Better](https://championship.mql5.com/ "https://championship.mql5.com/") with his system based on neural networks. Many internet forums were flooded with topics related to neural networks and Forex trading. Unfortunately writing native MQL4 implementation of NN is not easy. It requires some programming skills and the result would not be very efficient specially if you'd like to test your final result in tester on large number of data.

In this article I'll show you how you can use the freely available (under LGPL), renowned [Fast Artificial Neural Network Library](https://www.mql5.com/go?link=http://leenissen.dk/fann/ "http://leenissen.dk/fann/?") (FANN) in your MQL4 code while avoiding certain obstacles and limitations. Further I assume that reader is familiar with [Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network "https://en.wikipedia.org/wiki/Artificial_neural_network") (ann) and terminology related to this subject so I'll concentrate on practical aspects of using particular implementation of ann in MQL4 language.

### FANN features

To fully understand the possibilities of FANN implementation one need to familiarize with its [documentation](https://www.mql5.com/go?link=http://leenissen.dk/fann/html_latest/files/fann-h.html "http://leenissen.dk/fann/html_latest/files/fann-h.html") and most commonly used functions. The typical use of FANN is to create a simple feedforward network, train it with some data and run. The created and trained network might then be saved to file and restored later for further use. To create an ann one have to use fann\_create\_standard() function. Let's see its syntax:

```
FANN_EXTERNAL struct fann *FANN_API fann_create_standard(unsigned int num_layers, int lNnum, ... )
```

Where **num\_layers** represents the total number of layers including the input and the output layer. The lNnum and following arguments represents the number of neurons in each layer starting with the input layer and ending with the output layer. To create a network with one hidden layer with 5 neurons, 10 inputs and 1 output one would have to call it as follows:

```
fann_create_standard(3,10,5,1);
```

Once the ann is created the next operation would be to train it with some input and output data. The simplest training method is incremental training which can be achieved by the following function:

```
FANN_EXTERNAL void FANN_API fann_train(
        struct  fann    *       ann,


        fann_type       *       input,


        fann_type       *       desired_output  )
```

This function takes the pointer to **struct fann** returned previously by fann\_create\_standard() and both input data vector and output data vector. The input and output vectors are of array of **fann\_type** type. That type is in the matter of fact a **double** or **float** type, depending on the way the FANN is compiled. In this implementation the input and output vectors are going to be arrays of **double**.

Once the ann is trained the next desired feature would be to run that network. The function implementing that is defined as follows:

```
FANN_EXTERNAL fann_type * FANN_API fann_run(    struct  fann    *       ann,


        fann_type       *       input   )
```

This function takes the pointer to **struct fann** representing the previously created network and an input vector of the defined type ( **double** array). The returned value is an output vector array. This fact is important as for one utput network we allways get one element array with the output value rather than the output value itself.

Unfortunately most of FANN functions use a pointer to a **struct fann** representing the ann which cannot be directly handled by MQL4 which does not support structures as datatypes. To avoid that limitation we have to wrap that in some way and hide from MQL4. The easiest method is to create an array of **struct fann** pointers holding the proper values and refer to them with an index represented by an **_int_** _variable._ This way we can replace the unsupported type of variable with supported one and create a wrapper library that can be [easily integrated](https://docs.mql4.com/basis/preprosessor/import "https://docs.mql4.com/basis/preprosessor/import") with MQL4 code.

### Wrapping the FANN around

As to my best knowledge MQL4 does not support functions with variable arguments list so we have to deal with that too. On the other hand if the C function (of variable arguments length) is called with too many arguments nothing wrong happens so we can assume a fixed maximum number of arguments in MQL4 function passed to C library. The resulting wrapper function would look like follows:

```
/* Creates a standard fully connected backpropagation neural network.
* num_layers - The total number of layers including the input and the output layer.
* l1num - number of neurons in 1st layer (inputs)
* l2num, l3num, l4num - number of neurons in hidden and output layers (depending on num_layers).
* Returns:
* handler to ann, -1 on error
*/

int __stdcall f2M_create_standard(unsigned int num_layers, int l1num, int l2num, int l3num, int l4num);
```

We changed the leading **fann\_** with **f2M\_** (which stands for FANN TO MQL), used static number of arguments (4 layers) and the returning value is now an index to internal array of anns holding the **struct fann** data required by FANN to operate. This way we can easily call such function from within MQL code.

The same goes for:

```
/* Train one iteration with a set of inputs, and a set of desired outputs.
* This training is always incremental training, since only one pattern is presented.
* ann - network handler returned by f2M_create_*
* *input_vector - array of inputs
* *output_vector - array of outputs
* Returns:
* 0 on success and -1 on error
*/

int __stdcall f2M_train(int ann, double *input_vector, double *output_vector);
```

and

```
/* Run fann network
* ann - network handler returned by f2M_create_*
* *input_vector - array of inputs
* Returns:
* 0 on success, negative value on error
* Note:
* To obtain network output use f2M_get_output().
* Any existing output is overwritten
*/

int __stdcall f2M_run(int ann, double *input_vector);
```

Last, but not least is the fact that you should destroy your once created ann by the call to:

```
/* Destroy fann network
* ann - network handler returned by f2M_*
* Returns:
* 0 on success -1 on error
* WARNING: the ann handlers cannot be reused if ann!=(_ann-1)
* Other handlers are reusable only after the last ann is destroyed.
*/

int __stdcall f2M_destroy(int ann);
```

To release ann handles you should destroy networks in reverse order than they were created created. Alternatively you could use :

```
/* Destroy all fann networks
* Returns:
* 0 on success -1 on error
*/
int __stdcall f2M_destroy_all_anns();
```

However I'm pretty sure some of you might prefer to save their trained network for later use with:

```
/* Save the entire network to a configuration file.
* ann - network handler returned by f2M_create*
* Returns:
* 0 on success and -1 on failure
*/
int __stdcall f2M_save(int ann,char *path);
```

Of course the saved network can later be loaded (or rather recreated) with:

```
/* Load fann ann from file
* path - path to .net file
* Returns:
* handler to ann, -1 on error
*/
int __stdcall f2M_create_from_file(char *path);
```

Once we know the basic functions we might try to use that in our EA, but first we need to install the Fann2MQL package.

### Installing Fann2MQL

To facilitate the usage of this package I have create the msi [installer](https://www.mql5.com/go?link=https://fann2mql.wordpress.com/download/ "installer") that contains all the source code plus precompiled libraries and **Fann2MQL.mqh** header file that declares all Fann2MQL functions.

The procedure of installation is quite straightforward. First you are informed that Fann2MQL is under [GPL](https://www.mql5.com/go?link=http://www.gnu.org/licenses/gpl-2.0.html "http://www.gnu.org/licenses/gpl-2.0.html") license:

![](https://c.mql5.com/2/16/install1.jpg)

Installation of Fann2MQL, step 1

Then pick the folder to install the package. You can use the default **Program Files\\Fann2MQL\** or install directly into your **Meta Trader\\experts\** directory. The later will place all files directly into their places otherwise you'll have to copy them manually.

![](https://c.mql5.com/2/16/install2.jpg)

Installation of Fann2MQL, step 2

The installer puts files into following folders:

![](https://c.mql5.com/2/16/fann2mql_include_1.jpg)

include\ folder

![](https://c.mql5.com/2/16/fann2mql_libraries_1.jpg)

libraries\ folder

![](https://c.mql5.com/2/16/fann2mql_src_1.jpg)

src\ folder

If you chose to install into dedicated Fann2MQL folder, please copy the content of its **include** and **libraries** subfolders into your Meta Trader appropriate directory.

The installer installs also the FANN library into your system libraries folder (Windows\\system32 in most cases). The **src** folder contains all the source code of Fann2MQL. You can read the source code which is an ultimate documentation if you need any more information about the internals. You can also improve the code and add additional features if you like. I encourage you to send me your patches if you implement anything interesting.

### Using neural networks in your EA

Once the Fann2MQL is installed you can start to write your own EA or indicator. There's plenty of possible usage of NN. You can use them to forecast future price movements but the quality of such predictions and possibility of taking real advantage of it is doubtful. You can try to write your own strategy using [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning "Reinforcement Leanrning") techniques, say a [Q-Learning](https://en.wikipedia.org/wiki/Q-Learning "Q-Learning") or something similar. You may try to use NN as a signal filter for your heuristic EA or combine all of this techniques plus whatever you really wish to. You're limited by your imagination only.

Here I will show you an example of using NN as a simple filter for signals generated by MACD. Please do not consider it as valuable EA but as an example application of Fann2MQL. During the explanation of the way the example EA: NeuroMACD.mq4 works I'll show you how the Fann2MQL can be effectively used in MQL.

The very first thing for every EA is the declaration of global variables, defines and include section. Here is the begin of NeuroMACD containing those things:

```
// Include Neural Network package
#include <Fann2MQL.mqh>

// Global defines
#define ANN_PATH "C:\\ANN\\"
// EA Name
#define NAME "NeuroMACD"

//---- input parameters
extern double Lots=0.1;
extern double StopLoss=180.0;
extern double TakeProfit=270.0;
extern int FastMA=18;
extern int SlowMA=36;
extern int SignalMA=21;
extern double Delta=-0.6;
extern int AnnsNumber=16;
extern int AnnInputs=30;
extern bool NeuroFilter=true;
extern bool SaveAnn=false;
extern int DebugLevel=2;
extern double MinimalBalance=100;
extern bool Parallel=true;

// Global variables

// Path to anns folder
string AnnPath;

// Trade magic number
int MagicNumber=65536;

// AnnsArray[ann#] - Array of anns
int AnnsArray[];

// All anns loded properly status
bool AnnsLoaded=true;

// AnnOutputs[ann#] - Array of ann returned returned
double AnnOutputs[];

// InputVector[] - Array of ann input data
double InputVector[];

// Long position ticket
int LongTicket=-1;

// Short position ticket
int ShortTicket=-1;

// Remembered long and short network inputs
double LongInput[];
double ShortInput[];
```

The include command says to load the Fann2MQL.mqh header file containing the declaration of all Fann2MQL functions. After that all the Fann2MQL package functions are available for use in the script. The **ANN\_PATH** constant defines the path to store and load files with trained FANN networks. **You need to create that folder i.e. C:\\ANN**. The **NAME** constant contains the name of this EA, which is used later for loading and saving network files. Input parameters are rather obvious and those that aren't will be explained later, as well as global variables.

The entry point of every EA is its init() function:

```
int init()
  {
   int i,ann;

   if(!is_ok_period(PERIOD_M5))
     {
      debug(0,"Wrong period!");
      return(-1);
     }

   AnnInputs=(AnnInputs/3)*3; // Make it integer divisible by 3

   if(AnnInputs<3)
     {
      debug(0,"AnnInputs too low!");
     }
// Compute MagicNumber and AnnPath
   MagicNumber+=(SlowMA+256*FastMA+65536*SignalMA);
   AnnPath=StringConcatenate(ANN_PATH,NAME,"-",MagicNumber);

// Initialize anns
   ArrayResize(AnnsArray,AnnsNumber);
   for(i=0;i<AnnsNumber;i++)
     {
      if(i%2==0)
        {
         ann=ann_load(AnnPath+"."+i+"-long.net");
           } else {
         ann=ann_load(AnnPath+"."+i+"-short.net");
        }
      if(ann<0)
         AnnsLoaded=false;
      AnnsArray[i]=ann;
     }
   ArrayResize(AnnOutputs,AnnsNumber);
   ArrayResize(InputVector,AnnInputs);
   ArrayResize(LongInput,AnnInputs);
   ArrayResize(ShortInput,AnnInputs);

// Initialize Intel TBB threads
   f2M_parallel_init();

   return(0);
  }
```

First it checks whether the EA is applied to correct time frame period. **AnnInputs** variable contains the number of neural network inputs. As we'll use 3 sets of different arguments we want it to be divisible by 3. **AnnPath** is computed to reflect the EA **NAME** and **MagicNumber**, which is computed from the **SlowMA**, **FastMA** and **SignalMA** input arguments which are later used for MACD indicator signaling. Once it knows the AnnPath the EA tries to load neural networks using **ann\_load()** function which I'll describe below. Half of the loaded networks is meant for the long position filtering and the other half is meant for shorts. **AnnsLoaded** variable is used to indicate the fact that all networks were initialized correctly. As you probably noticed this example EA is trying to load multiple networks. I doubt it's really necessary in this application yet I wanted to show you the full potential of Fann2MQL, which is handling multiple networks at the same time and can process them in parallel taking advantage of multiple cores or CPUs. To make it possible Fann2MQL is taking advantage of [Intel® Threading Building Blocks](https://www.mql5.com/go?link=https://www.threadingbuildingblocks.org/ "http://www.threadingbuildingblocks.org/") technology. The function **f2M\_parallel\_init()** is used to initialize that interface.

Here is the way I used to initialize networks:

```
int ann_load(string path)
  {
   int ann=-1;

   /* Load the ANN */
   ann=f2M_create_from_file(path);
   if(ann!=-1)
     {
      debug(1,"ANN: '"+path+"' loaded successfully with handler "+ann);
     }
   if(ann==-1)
     {

      /* Create ANN */
      ann=
          f2M_create_standard(4,AnnInputs,AnnInputs,AnnInputs/2+1,1);
      f2M_set_act_function_hidden(ann,FANN_SIGMOID_SYMMETRIC_STEPWISE);
      f2M_set_act_function_output(ann,FANN_SIGMOID_SYMMETRIC_STEPWISE);
      f2M_randomize_weights(ann,-0.4,0.4);
      debug(1,"ANN: '"+path+"' created successfully with handler "+ann);
     }
   if(ann==-1)
     {
      debug(0,"INITIALIZING NETWORK!");
     }
   return(ann);
  }
```

As you can see if the **f2M\_create\_from\_file()** fails, which is indicated by the negative return value, the network is created with **f2M\_create\_standard()** function with arguments indicating that the created network should have 4 layers (including input and output), AnnInput inputs, AnnInput neurons in first hidden layer, AnnInput/2+1 neurons in 2nd hidden layer and 1 neuron in output layer. **f2M\_set\_act\_function\_hidden()** is used to set the activation function of hidden layers to SIGMOID\_SYMMETRIC\_STEPWISE (please refer to FANN documentation of fann\_activationfunc\_enum) and the same goes for the output layer. Then there is the call to **f2m\_randomize\_weights()** which is used to initialize neuron connection weights inside the network. Here I used the range of <-0.4; 0.4> but you can use any other depending on your application.

At this point you probably have noticed the **debug()** function I used a couple of times. It's one of the simplest methods to alter the verbose level of your EA. Together with it and the input parameter **DebugLevel** you can tune the way that your code is producing the debug output.

```
void debug(int level,string text)
  {
   if(DebugLevel>=level)
     {
      if(level==0)
         text="ERROR: "+text;
      Print(text);
     }
  }
```

If the first argument of **debug()** function, the debug **level** is higher than **DebugLevel** the function does not produce any output. If it's lower of equal the **text** string is printed out. If the debug level is 0 the string "ERROR: " is appended to the begin. This way you can split debug produced by your code to multiple levels. The most important are probably errors so they are assigned to the level 0. They will be printed unless you lower your **DebugLevel** to below 0 (which is not advised). At level 1 some important information will be printed, like confirmation of successful network loading or creation. At level 2 or higher the importance of printed information is gradually decreasing.

Before the detailed explanation of **start()** function, which is quite lengthy, I need to show you some more functions meant to prepare the network input and running the actual networks:

```
void ann_prepare_input()
  {
   int i;

   for(i=0;i<=AnnInputs-1;i=i+3)
     {
      InputVector[i]=
         10*iMACD(NULL,0,FastMA,SlowMA,SignalMA,PRICE_CLOSE,
                  MODE_MAIN,i*3);
      InputVector[i+1]=
         10*iMACD(NULL,0,FastMA,SlowMA,SignalMA,PRICE_CLOSE,
                  MODE_SIGNAL,i*3);
      InputVector[i+2]=InputVector[i-2]-InputVector[i-1];
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double ann_run(int ann,double &vector[])
  {
   int ret;
   double out;
   ret=f2M_run(ann,vector);
   if(ret<0)
     {
      debug(0,"Network RUN ERROR! ann="+ann);
      return(FANN_DOUBLE_ERROR);
     }
   out=f2M_get_output(ann,0);
   debug(3,"f2M_get_output("+ann+") returned: "+out);
   return(out);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int anns_run_parallel(int anns_count,int &anns[],double &input_vector[])
  {
   int ret;

   ret=f2M_run_parallel(anns_count,anns,input_vector);

   if(ret<0)
     {
      debug(0,"f2M_run_parallel("+anns_count+") returned: "+ret);
     }
   return(ret);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void run_anns()
  {
   int i;

   if(Parallel)
     {
      anns_run_parallel(AnnsNumber,AnnsArray,InputVector);
     }

   for(i=0;i<AnnsNumber;i++)
     {
      if(Parallel)
        {
         AnnOutputs[i]=f2M_get_output(AnnsArray[i],0);
           } else {
         AnnOutputs[i]=ann_run(AnnsArray[i],InputVector);
        }
     }
  }
//+------------------------------------------------------------------+
```

The function **ann\_prepare\_input()** is used to prepare the input name for the networks (thus the name). The purpose of it is quite straightforward, yet this is the point I must remind you that the input data has to be properly normalized. There is no sophisticated normalization in this case, I simply used the MACD main and signal values which are never exceeding the desired range on the accounted data. In the real example you probably should pay more attention to this issue. As you probably might suspect choosing the proper input arguments for network input, coding it, decomposing and normalizing is one of the most important factors in neural network processing.

As I mentioned before the Fann2MQL has the ability extending the normal functionality of MetaTrader, that is parallel multithreaded processing of neural networks. The global argument **Parallel** controls this behavior. The **run\_anns()** function runs all the initialized networks and gets the outputs of them and stores in **AnnOutput\[\]** array. **anns\_run\_parallel** function is responsible for handling the job in the multithreaded way. It calls the **f2m\_run\_parallel()** which takes as a first argument the number of networks to process, the second argument is an array containing handles to all networks you wish to run providing the input vector as a third argument. All the networks has to be run on the very same input data. Obtaining the output from network is done by multiple calls to **f2m\_get\_output**().

Now let's see the **start()** function:

```
int
start()
  {
   int i;
   bool BuySignal=false;
   bool SellSignal=false;

   double train_output[1];

   /* Is trade allowed? */
   if(!trade_allowed())
     {
      return(-1);
     }

   /* Prepare and run neural networks */
   ann_prepare_input();
   run_anns();

   /* Calulate last and previous MACD values.
* Lag one bar as current bar is building up
*/
   double MacdLast=iMACD(NULL,0,FastMA,SlowMA,SignalMA,PRICE_CLOSE,
                         MODE_MAIN,1);
   double MacdPrev=iMACD(NULL,0,FastMA,SlowMA,SignalMA,PRICE_CLOSE,
                         MODE_MAIN,2);

   double SignalLast=iMACD(NULL,0,FastMA,SlowMA,SignalMA,PRICE_CLOSE,
                           MODE_SIGNAL,
                           1);
   double SignalPrev=iMACD(NULL,0,FastMA,SlowMA,SignalMA,PRICE_CLOSE,
                           MODE_SIGNAL,
                           2);

   /* BUY signal */
   if(MacdLast>SignalLast && MacdPrev<SignalPrev)
     {
      BuySignal=true;
     }
   /* SELL signal */
   if(MacdLast<SignalLast && MacdPrev>SignalPrev)
     {
      SellSignal=true;
     }

   /* No Long position */
   if(LongTicket==-1)
     {
      /* BUY signal */
      if(BuySignal)
        {
         /* If NeuroFilter is set use ann wise to decide :) */
         if(!NeuroFilter || ann_wise_long()>Delta)
           {
            LongTicket=
          OrderSend(Symbol(),OP_BUY,Lots,Ask,3,
                    Bid-StopLoss*Point,
                    Ask+TakeProfit*Point,
                    NAME+"-"+"L ",MagicNumber,0,Blue);
           }
         /* Remember network input */
         for(i=0;i<AnnInputs;i++)
           {
            LongInput[i]=InputVector[i];
           }
        }
        } else {
      /* Maintain long position */
      OrderSelect(LongTicket,SELECT_BY_TICKET);
      if(OrderCloseTime()==0)
        {
         // Order is opened
         if(SellSignal && OrderProfit()>0)
           {
            OrderClose(LongTicket,Lots,Bid,3);
           }
        }
      if(OrderCloseTime()!=0)
        {
         // Order is closed
         LongTicket=-1;
         if(OrderProfit()>=0)
           {
            train_output[0]=1;
              } else {
            train_output[0]=-1;
           }
         for(i=0;i<AnnsNumber;i+=2)
           {
            ann_train(AnnsArray[i],LongInput,train_output);
           }
        }
     }

   /* No short position */
   if(ShortTicket==-1)
     {
      if(SellSignal)
        {
         /* If NeuroFilter is set use ann wise to decide ;) */
         if(!NeuroFilter || ann_wise_short()>Delta)
           {
            ShortTicket=
          OrderSend(Symbol(),OP_SELL,Lots,Bid,3,
                    Ask+StopLoss*Point,
                    Bid-TakeProfit*Point,NAME+"-"+"S ",
                    MagicNumber,0,Red);
           }
         /* Remember network input */
         for(i=0;i<AnnInputs;i++)
           {
            ShortInput[i]=InputVector[i];
           }
        }
        } else {
      /* Maintain short position */
      OrderSelect(ShortTicket,SELECT_BY_TICKET);
      if(OrderCloseTime()==0)
        {
         // Order is opened
         if(BuySignal && OrderProfit()>0)
           {
            OrderClose(LongTicket,Lots,Bid,3);
           }
        }
      if(OrderCloseTime()!=0)
        {
         // Order is closed
         ShortTicket=-1;
         if(OrderProfit()>=0)
           {
            train_output[0]=1;
              } else {
            train_output[0]=-1;
           }
         for(i=1;i<AnnsNumber;i+=2)
           {
            ann_train(AnnsArray[i],ShortInput,train_output);
           }
        }
     }

   return(0);
  }
//+------------------------------------------------------------------+
```

I'll describe it briefly as it is quite well commented. The **trade\_allowed()** checks whether it is allowed to trade. Basically it checks the **AnnsLoaded** variable indicating that all anns were initialize properly, then checks for the proper time frame period minimum account balance and at the very end allows to trade only on the first tick of a new bar. Next two function which are used to prepare network input and run the network processing were described just few lines above. Next we calculate and put into variables for later processing the MACD values of signal and main line for the last buildup bar and the previous one. The current bar is omitted as it is not build up yet and probably will be redrawed. The **SellSignal** and **BuySignal** are calculated accordingly to MACD signal and main line crossover. Both signals are used for long and short position processing which are symmetrical so I'll describe only the case for longs.

The **LongTicket** variable holds the ticket number of currently opened position. If it's equal to -1 no position is opened so if the **BuySignal** is set that might indicate good opportunity to open long position. If the variable **NeuroFilter** is not set the long position is opened and that is the case without the neural network filtering of signals -- the order is sent to buy. At this point the **LongInput** variable is meant to remember the **InputVector** prepared by **ann\_prepare\_input()** for later use.

If **LongTicekt** variable holds the valid ticket number the EA checks whether is is still opened or was closed by the StopLoss or TakeProfit. If the order is not closed nothing happens, however if the order is closed the **train\_output\[\]** vector, which has only one otput, is calculated to hold the value of -1 if the order was closed with loss or 1 if the order was closed with profit. That value is then passed to **ann\_train()** function and all the networks responsible for handling the long position are trained with it. As the input vector the variable **LongInput** is used, which is holding the **InputVector** at the moment of opening the position. This way the network is taught which signal is bringing profits and which one is not.

Once you have a trained network switching the **NeuroFilter** to **true** turns the network filtering. The **ann\_wise\_long()** is using the neural network wise calculated as a mean of values returned by all networks meant to handle the long position. The **Delta** parameter is used as a threshold value indicating that the filtered signal is valid or no. As many other values it was obtained through the process of optimization.

Now once we know how it works I'll show you how it can be used. The test pair is of course EURUSD. I used the data from [Alpari](https://www.mql5.com/go?link=http://www.alpari-idc.com/en/dc/databank.php "http://www.alpari-idc.com/en/dc/databank.php"), converted to M5 time frame. I used the period from 2007.12.31 to 2009.01.01 for training/optimizing and 2009.01.01-2009.03.22 for testing purposes. In the very first run I tried to obtain the most profitable values for StopLoss, TakeProfit, SlowMA, FastMA and SignalMA argument, which I then coded into NeuroMACD.mq4 file. The **NeuroFIlter** was turned off as well as **SaveAnn**, the **AnnsNumber** was set to 0 to avoid neural processing. I used the genetic algorithm for optimization process. Once the values were obtained the resulting report looked as follows:

[![](https://c.mql5.com/2/16/testdataeraport_small.gif)](https://c.mql5.com/2/16/testdataeraport.gif "https://c.mql5.com/2/16/testdataeraport.gif")

Report on training data after basic parameter optimization.

As you can see I have run this EA on the mini account with the Lot size of 0.01 and the initial balance of 200. However you can tune these parameters accordingly to your account settings or preferences.

At this point we have enough profitable and losing trades so we could turn on the **SaveAnn** and set the **AnnsNumber** to 30. Once done so I run the tester once again. The result was exactly the same with the except of the fact that the process was much slower (as a result of neural processing) and the folder C:\\ANN was populated with the trained networks as shown on the image below. Make sure the C:\\ANN folder existed prior to this run!

![](https://c.mql5.com/2/16/annhfolder.gif)

The C:\\\ANN\\\ folder.

Once we have trained networks it's time to test how it behaves. First we'll try it on the training data. Change the **NeuroFilter** to **true** and **SaveAnn** to **false** and start the tester. The result I have obtained is shown below. Note that it might vary slightly for you case as there is some randomness inside networks in neuron connection weights provided at the network initialization process (in this example I used explicit call to **f2M\_randomize\_weights()** inside **ann\_load()**).

[![](https://c.mql5.com/2/16/neurofilterxtestdatawreport_small.gif)](https://c.mql5.com/2/16/neurofilterxtestdatawreport.gif "https://c.mql5.com/2/16/neurofilterxtestdatawreport.gif")

Result obtained on training data with signal neural filtering turned on.

The net profit is little greater (20.03 versus 16.92), yet the profit factor is a lot higher (1.25 versus 1.1). The number of trades is much less (83 vs 1188) and the average consecutive losses number is lowered from 7 to 2. However it only shows that neural signal filtering is working but it says nothing about how it operates on data that were not used for during the training. The result I have obtain from the testing period (2009.01.01 - 2009.30.28) is shown below:

[![](https://c.mql5.com/2/16/testingdataoneurofilterareport_small.gif)](https://c.mql5.com/2/16/testingdataoneurofilterareport.gif "https://c.mql5.com/2/16/testingdataoneurofilterareport.gif")

Result obtained from testing data with neural filtering turned on.

The number of trades performed is quite low and it's hard to tell the quality of this strategy, yet I was not going to show you how to write the best profitable EA but to explain how you could use neural networks in your MQL4 code. The real effect of using neural networks in this case can be seen only when compared the results of the EA on test data with **NeuroFilter** turned on and off. Below is the result obtained from testing data period without neural signal filtering:

[![](https://c.mql5.com/2/16/testingdatavnofilterhreport_small.gif)](https://c.mql5.com/2/16/testingdatavnofilterhreport.gif "https://c.mql5.com/2/16/testingdatavnofilterhreport.gif")

Results from testing data without neural filtering.

The difference is quite obvious. As you can see the neural signal filtering turned the losing EA into a profitable one!

### Conclusion

I hope that you have learned from this article how to use neural networks in MetaTrader. With the help of simple, free and opensource package [Fann2MQL](https://www.mql5.com/go?link=https://fann2mql.wordpress.com/ "Fann2MQL") you can easily add the neural network layer into virtually any Expert Advisor or start writing your own one which is fully or partially based on neural networks. The unique multithreading capability can speed up your processing many times, depending on number of your CPU cores, specially when optimizing certain parameters. In one case it shortened optimizing of my Reinforcement Learning based EA processing from about 4 days to 'only' 28 hours on a 4 core Intel CPU.

During the writing of this article I have decided to put Fann2MQL on its own website: [http://fann2mql.wordpress.com/](https://www.mql5.com/go?link=https://fann2mql.wordpress.com/ "http://fann2mql.wordpress.com/"). You can find there the latest version of Fann2MQL and possibly all future versions as well as the documentation of all functions. I promise to keep this software under GPL license for all releases so if you send me any comments, feature requests or patches that I will find interesting be sure to find it next releases.

Please note that this article shows only the very basic usage of Fann2MQL. As this package is not much more than FANN you can use all the tools designed for managing FANN networks, like:

- Fanntool: [http://code.google.com/p/fanntool/](https://www.mql5.com/go?link=https://code.google.com/archive/p/fanntool "http://code.google.com/p/fanntool/")
- FannExplorer: [http://leenissen.dk/fann/index.php?p=gui.php](https://www.mql5.com/go?link=http://leenissen.dk/fann/index.php?p=gui.php "http://leenissen.dk/fann/index.php?p=gui.php")
- Fann Matlab bindings: [http://www.sumowiki.intec.ugent.be/index.php/FANN\_Bindings](https://www.mql5.com/go?link=http://www.sumowiki.intec.ugent.be/index.php/FANN_Bindings "http://www.sumowiki.intec.ugent.be/index.php/FANN_Bindings")

And there's much more about FANN on the Fast Artificial Neural Network Library homepage: [http://leenissen.dk/fann/](https://www.mql5.com/go?link=http://leenissen.dk/fann/ "http://leenissen.dk/fann/")!

### Post Scriptum

After writing this article I have found a insignificant error in NeuroMACD.mq4. The OrderClose() function for short position was fed with long position ticket number. It resulted in a skewed strategy which was more likely to hold shorts and close longs:

```
/* Maintain short position */
OrderSelect(ShortTicket,SELECT_BY_TICKET);
if(OrderCloseTime()==0)
  {
// Order is opened
   if(BuySignal && OrderProfit()>0)
     {
      OrderClose(LongTicket,Lots,Bid,3);
     }
  }
```

In the correct version of the script I have fixed this error and removed the OrderClose() strategy at all. This did not changed the overall picture of the influence of neural filtering on the EA yet the balance curve shape was quite different. You can find both versions of this EA attached to this article.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/1565.zip "Download all attachments in the single ZIP archive")

[NeuroMACD-fixed.mq4](https://www.mql5.com/en/articles/download/1565/NeuroMACD-fixed.mq4 "Download NeuroMACD-fixed.mq4")(10.13 KB)

[NeuroMACD.mq4](https://www.mql5.com/en/articles/download/1565/NeuroMACD.mq4 "Download NeuroMACD.mq4")(10.12 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/39530)**
(106)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
24 May 2021 at 07:20

In ann\_load

```
ann = f2M_create_from_file_string (path);
```

and in ann\_save

```
ret = f2M_save_string (ann, path);
```

changed functions to string functions. Now works for me. Attached is my updated example code for this great work

![MataNui](https://c.mql5.com/avatar/avatar_na2.png)

**[MataNui](https://www.mql5.com/en/users/matanui)**
\|
22 Mar 2022 at 13:13

Simply isnt working.

Install went fine. No errors logged. Training files are saved correctly. It just doesnt do anything. The results are identical withthe filter on and off.

![Aldo Marco Ronchese](https://c.mql5.com/avatar/2025/1/6779d872-abf0.jpg)

**[Aldo Marco Ronchese](https://www.mql5.com/en/users/aldo1003)**
\|
12 Jan 2023 at 14:10

Hi Guys i wonder if you could help . Ive been hacking this a bit but a few issues perhaps already covered . I have to edit the file to set the learning rate as i couldnt set it through the included dll converter . Also limited to 2 layers . Cant think of a way without making file in different system like python with 3 or more layers and then using file as it ignores settings and uses files specs? So what i need help with is setting learning rate and number of layers above 2 .. perhaps can be hard coded a new command to create ann for 3 .. the extra layer makes it deep learning and this could make a huge difference. Thanks for your time


![Aldo Marco Ronchese](https://c.mql5.com/avatar/2025/1/6779d872-abf0.jpg)

**[Aldo Marco Ronchese](https://www.mql5.com/en/users/aldo1003)**
\|
12 Jan 2023 at 14:24

**Stanislav Korotky [#](https://www.mql5.com/en/forum/39530/page9#comment_5332054):**

According to some researchers (for example, read [here](https://www.mql5.com/go?link=https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw "https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw")) 2 hidden layers are sufficient for classical MLP in almost all tasks. Increasing number of layers will only slow down performance without result improvement.

This is opinion . The way it works a lot of neurons are waisted . The randomization is actually where the intelligence ls . Some initial randomizations are closer to being optimized than others . Others are so far out they will never train . Having more layer gives more focus and and context to the data . This helps it to learn faster . To make an analogy each layer is like an attribute. Each layer is a filter . 2 dimensions vs 3 dimensions.. you can represent 3 in 2 but its easier to just have 3  . All this to justify request for 3 layers pls . It is a big learning curve for me to change the source and recompile. After all the fabulous work you have done this plus the ability to change the learning rate and possibly the momentum would complete the package . So far you ea converted to my ideas trains to 99.9% accurate on any data . The new data however it falls short . I feel the extra layer will provide context  . I will also give you the EA if you could help me . It is your work plus fann plus 2 years of my own research


![Aldo Marco Ronchese](https://c.mql5.com/avatar/2025/1/6779d872-abf0.jpg)

**[Aldo Marco Ronchese](https://www.mql5.com/en/users/aldo1003)**
\|
12 Jan 2023 at 14:27

This is what i was able to do a year ago .. since then no progress and i think this is ley , extra layer . Thank you so much for your excellent work !


![On the Long Way to Be a Successful Trader - The Two Very First Steps](https://c.mql5.com/2/17/813_17.gif)[On the Long Way to Be a Successful Trader - The Two Very First Steps](https://www.mql5.com/en/articles/1571)

The main point of this article is to show a practical way to implement an effective MM. This can be achieved only by using a certain kind of strategies that we need to identify and describe first. In the following we’ll cover the basic concepts of how to build such a strategy and we’ll point out the common mistakes which always end up in draining a trader’s account.

![Alert and Comment for External Indicators](https://c.mql5.com/2/17/789_15.gif)[Alert and Comment for External Indicators](https://www.mql5.com/en/articles/1568)

In practical work a trader can face the following situation: it is necessary to get "alert" or a text message on a display (in a chart window) indicating about an appeared signal of an indicator. The article contains an example of displaying information about graphical objects created by an external indicator.

![Superposition and Interference of Financial Securities](https://c.mql5.com/2/17/800_14.gif)[Superposition and Interference of Financial Securities](https://www.mql5.com/en/articles/1570)

The more factors influence the behavior of a currency pair, the more difficult it is to evaluate its behavior and make up future forecasts. Therefore, if we managed to extract components of a currency pair, values of a national currency that change with the time, we could considerably delimit the freedom of national currency movement as compared to the currency pair with this currency, as well as the number of factors influencing its behavior. As a result we would increase the accuracy of its behavior estimation and future forecasting. How can we do that?

![Channels. Advanced Models. Wolfe Waves](https://c.mql5.com/2/16/774_6.gif)[Channels. Advanced Models. Wolfe Waves](https://www.mql5.com/en/articles/1564)

The article describes rules of marking patterns of Wolfe Waves. You will find here details of constructing and rules of accurate marking, which help to find correct formations of waves quickly and correctly.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/1565&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=6467253706173622530)

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