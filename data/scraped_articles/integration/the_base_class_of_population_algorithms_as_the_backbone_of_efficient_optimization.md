---
title: The base class of population algorithms as the backbone of efficient optimization
url: https://www.mql5.com/en/articles/14331
categories: Integration, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T17:20:48.794265
---

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/14331&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068125572145542589)

MetaTrader 5 / Tester


### Contents

1\. [Introduction. Prospects and opportunities of inheriting from the base class of population algorithms](https://www.mql5.com/en/articles/14331#tag1)

2\. [Implementing the base class of population algorithms](https://www.mql5.com/en/articles/14331#tag2)

3\. [Code of descendant algorithms](https://www.mql5.com/en/articles/14331#tag3)

4\. [Code of the unified test stand for all algorithms](https://www.mql5.com/en/articles/14331#tag4)

5\. [Add popular well-known test functions](https://www.mql5.com/en/articles/14331#tag5)

6\. [Building 3D test functions](https://www.mql5.com/en/articles/14331#tag6)

7\. [Conclusions](https://www.mql5.com/en/articles/14331#tag7)

### 1\. Introduction. Prospects and opportunities of inheriting from the base class of population algorithms

In the world of modern computing and artificial intelligence, the base class, designed to integrate optimization algorithms into final software solutions, is a key element that opens up endless horizons of technical possibilities for developers. Inheritance from this base class in the context of population algorithms not only provides convenience and efficiency in the development of new optimization methods, but also expands the prospects for creating hybrid algorithms that can adapt to a wide variety of problems.

Combining optimization algorithms within a base class opens the door to creating innovative solutions that combine the best features of different methods. The hybrid algorithms that emerged from this approach are able to effectively overcome the limitations of individual methods and reach new heights in solving complex optimization problems.

In addition, the base class for population algorithms ensures ease of use and testing of the developed algorithms on standard sets of test functions. This allows researchers and developers to quickly evaluate the efficiency of new optimization methods by comparing their performance with existing solutions.

Let's imagine that the world of optimization and search for solutions is like the amazing culinary world, where each optimization method is a unique ingredient that gives its own unique taste to a dish. Hybridization in this context is like skillfully combining different ingredients to create new, tastier and more interesting dishes.

You have a wide range of different optimization methods - genetic algorithms, evolutionary strategies, ant algorithms, particle swarm optimization and many others. Each of them has its own strengths and abilities, but also has its limitations.

This is where hybridization comes in! You can take the best from each method, combining them into unique combinations like a seasoned chef. In this way, hybrid optimization methods can combine the strengths of different approaches, compensating for their weaknesses and creating more efficient and powerful tools for finding optimal solutions.

Think of the combination of a genetic algorithm with local search as a perfect combination of spicy peppers and sweet honey in a dish, giving it a deep and rich flavor. Likewise, the hybridization of population algorithms allows the creation of innovative methods that can quickly and accurately find optimal solutions in various fields, be it engineering problems, financial analytics or artificial intelligence.

Thus, hybridization in optimization is not just mixing methods, it is the art of creating new approaches that can maximize the potential of each method and achieve outstanding results. Ultimately, through hybridization, we can create more efficient, innovative and powerful optimization methods that can solve the most complex problems and lead to new discoveries and advances in various fields.

Moreover, the unified base class allows individual elements of each algorithm to be integrated into custom solutions for use in the design of new, unique and powerful optimization methods.

### 2\. Implementing the base class of population algorithms

In the context of the article about inheriting from a base class for population algorithms and creating hybrid optimization methods, we can consider several interesting combinations as examples:

- **Genetic algorithm with enhanced local search.** In this combination, a genetic algorithm is used to globally search for an optimal solution, and a local search is used to refine the found solution in the neighborhood. This allows combining the advantages of global and local search, increasing the accuracy and speed of the algorithm convergence.

- **Evolutionary strategy with the ant algorithm.** Here, the evolutionary strategy can be used to change the model parameters, while the ant algorithm can be used to find the optimal path in the parameter space. This combination can be effective for optimizing complex problems where finding the optimal combination of parameters is required.

- **Swarm particles with genetic programming.** In this combination, swarm particles can be used to explore the solution space, and genetic programming can be used to evolve program structures that solve the optimization problem. This allows for efficient exploration of both parameter space and solution structures.

- **Simulated annealing search with genetic algorithm.** Here, simulated annealing can be used to explore the solution space taking into account the temperature regime, and a genetic algorithm can be used to find optimal solutions in a given space. This combination can provide deeper exploration of the solution space and improve the convergence of the algorithm.

We can also consider the following directions in using the capabilities of population algorithms combined into a single base class:

- **Combined metaheuristic method.** Several different metaheuristic algorithms such as genetic algorithms, ant algorithms, particle swarm optimization algorithms, simulated annealing, etc. can be combined in this method. These algorithms can work in parallel or sequentially, exchanging information and combining their strengths to more efficiently find the optimal solution.

- **Hybrid method with adaptive control of search strategies.** In this approach, adaptive control mechanisms can be used to dynamically combine different optimization strategies depending on the problem characteristics. For example, we can change the weights or parameters of each method depending on their performance at the current optimization stage.

- **Hybrid method with artificial neural networks.** In this approach, artificial neural networks can be used to adaptively control the parameters and optimization strategies of population algorithms. Neural networks can learn on the fly, adapting to changes in the search space, and suggest optimal parameters for each optimization method.

- **Joint optimization and reinforcement learning method.** This approach can combine population algorithms with reinforcement learning techniques to create a hybrid system that can efficiently explore and optimize complex solution spaces. Reinforcement learning agents can learn from the results of population algorithms and vice versa, creating interactions between different optimization methods.

The different number of external parameters in each optimization algorithm can create problems in inheritance and uniform application. To solve this problem, it was decided to specify the external parameters of the default algorithms in the constructor, based on the results of extensive tests. At the same time, it remains possible to change these parameters before initializing the algorithms. Thus, the object of each algorithm will represent the final solution ready for use. Previously, algorithms and their parameters were separate entities.

So, let's start with the algorithm parameters. It is convenient to describe each parameter using the S\_AlgoParam structure containing the parameter name and value. Accordingly, the array of objects of this structure will represent a set of external parameters of the algorithms.

```
struct S_AlgoParam
{
    double val;
    string name;
};
```

Each optimization algorithm has a search agent - this is an elementary unit and an indispensable participant in the search strategy - a firefly in the firefly algorithm, a diligent bee in the bee algorithm, a hardworking ant in the ant algorithm, etc. They are unique artists in the optimization labyrinth, revealing the brilliance of the art of searching and discovering optimal paths to success. Their efforts and aspirations, like magic, transform the chaos of data into the harmony of solutions, illuminating the path to new horizons of ideal optimization.

Thus, each agent represents a specific solution to the optimization problem and has two mandatory properties: coordinates in the search space (optimized parameters) and the solution quality (fitness function). To be able to expand functionality and capabilities and implement specific properties of algorithms, we will formalize the agent in the form of the C\_AO\_Agent class, which we can subsequently inherit.

```
class C_AO_Agent
{
  public:
  ~C_AO_Agent () { }

  double c []; //coordinates
  double f;    //fitness
};
```

Most logical operations in optimization algorithms are repeated and can be designed separately as a set of functions of the C\_AO\_Utilities class, whose object, in turn, can be used both in algorithm classes and in agents.

The C\_AO\_Utilities class contains the following methods:

- Scale: Overloaded method that scales the "In" input from the \[InMIN, InMAX\] range to the \[OutMIN, OutMAX\] range. It is also possible to perform reverse scaling if the "revers" parameter is set to "true".
- RNDfromCI: Generates a random real number within the specified \[min, max\] range.
- RNDintInRange: Generates a random integer number within the specified \[min, max\] range.
- RNDbool: Generates a random boolean value (true/false).
- RNDprobab: Generates a random probability (real number between 0 and 1).
- SeInDiSp: Calculates the value considering the specified "Step" within the \[InMin, InMax\] range.
- The "DecimalToGray", "IntegerToBinary", "GrayToDecimal", "BinaryToInteger" and "GetMaxDecimalFromGray" methods: Perform conversions between decimal numbers, binary numbers and Gray codes.
- The "GaussDistribution" and "PowerDistribution" methods: Perform calculations for the normal distribution and power distribution, respectively.
- "Sorting" method (template method): Sort the "p" array of "T" type in descending order.
- "S\_Roulette" structure: Contains the "start" and "end" fields to represent the range.
- The "PreCalcRoulette" and "SpinRoulette" methods: "PreCalcRoulette" calculates ranges for "T" type objects and saves them in the "roulette" array. "SpinRoulette" performs the roulette "spin" based on the "aPopSize" population size.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_Utilities
{
  public: //--------------------------------------------------------------------
  double Scale                 (double In, double InMIN, double InMAX, double OutMIN, double OutMAX);
  double Scale                 (double In, double InMIN, double InMAX, double OutMIN, double OutMAX,  bool revers);
  double RNDfromCI             (double min, double max);
  int    RNDintInRange         (int min, int max);
  bool   RNDbool               ();
  double RNDprobab             ();
  double SeInDiSp              (double In, double InMin, double InMax, double Step);
  void   DecimalToGray         (ulong decimalNumber, char &array []);
  void   IntegerToBinary       (ulong number, char &array []);
  ulong  GrayToDecimal         (const char &grayCode [], int startInd, int endInd);
  ulong  BinaryToInteger       (const char &binaryStr [], const int startInd, const int endInd);
  ulong  GetMaxDecimalFromGray (int digitsInGrayCode);
  double GaussDistribution     (const double In, const double outMin, const double outMax, const double sigma);
  double PowerDistribution     (const double In, const double outMin, const double outMax, const double p);

  //----------------------------------------------------------------------------
  template<typename T>
  void Sorting (T &p [], T &pTemp [], int size)
  {
    int    cnt = 1;
    int    t0  = 0;
    double t1  = 0.0;
    int    ind [];
    double val [];

    ArrayResize (ind, size);
    ArrayResize (val, size);

    for (int i = 0; i < size; i++)
    {
      ind [i] = i;
      val [i] = p [i].f;
    }

    while (cnt > 0)
    {
      cnt = 0;
      for (int i = 0; i < size - 1; i++)
      {
        if (val [i] < val [i + 1])
        {
          t0 = ind [i + 1];
          t1 = val [i + 1];
          ind [i + 1] = ind [i];
          val [i + 1] = val [i];
          ind [i] = t0;
          val [i] = t1;
          cnt++;
        }
      }
    }

    for (int u = 0; u < size; u++) pTemp [u] = p [ind [u]];
    for (int u = 0; u < size; u++) p [u] = pTemp [u];
  }

  //----------------------------------------------------------------------------
  struct S_Roulette
  {
      double start;
      double end;
  };
  S_Roulette roulette [];

  template<typename T>
  void PreCalcRoulette (T &agents [])
  {
    int aPopSize = ArraySize (agents);
    roulette [0].start = agents [0].f;
    roulette [0].end   = roulette [0].start + (agents [0].f - agents [aPopSize - 1].f);

    for (int s = 1; s < aPopSize; s++)
    {
      if (s != aPopSize - 1)
      {
        roulette [s].start = roulette [s - 1].end;
        roulette [s].end   = roulette [s].start + (agents [s].f - agents [aPopSize - 1].f);
      }
      else
      {
        roulette [s].start = roulette [s - 1].end;
        roulette [s].end   = roulette [s].start + (agents [s - 1].f - agents [s].f) * 0.1;
      }
    }
  }
  int  SpinRoulette (int aPopSize);
};
//——————————————————————————————————————————————————————————————————————————————
```

Since stochastic optimization algorithms are based on generating random numbers, this operation can be performed hundreds or even thousands of times to obtain each solution. Therefore, it is advisable to optimize generation of random numbers by separating specific tasks. Given that the standard generator creates integers, there is potential to speed up this process.

We have already met the "RNDfromCI" method, which generates a random real number within the given \["min", "max"\] range:

```
double C_AO_Utilities ::RNDfromCI (double min, double max)
{
  if (min == max) return min;
  if (min > max)
  {
    double temp = min;
    min = max;
    max = temp;
  }
  return min + ((max - min) * rand () / 32767.0);
}
```

There is often a need to generate a random integer, for example, to randomly select an agent in a population. The "RNDintInRange" method will help us with this.

```
int C_AO_Utilities :: RNDintInRange (int min, int max)
{
  if (min == max) return min;
  if (min > max)
  {
    int temp = min;
    min = max;
    max = temp;
  }
  return min + rand () % (max - min + 1);
}
```

It is possible to obtain a random Boolean variable using the "RNDbool" method very quickly, compared to the two methods above, which is why it makes sense to separate the random variables into separate methods depending on the task.

```
bool C_AO_Utilities :: RNDbool ()
{
  return rand () % 2 == 0;
}
```

There is yet another method "RNDprobab", which allows us to get a random real number in the \[0.0, 1.0\] range. It is great for performing the probability of certain operations, such as the probability of a crossover in the genetic algorithm. Such operations are also performed quite often.

```
double C_AO_Utilities :: RNDprobab ()
{
  return (double)rand () / 32767;
}
```

Now let's look at the "C\_AO" base class of population optimization algorithms. This class describes the required attributes of all population algorithms, such as:

- Methods and properties of the "C\_AO" class:

> \- SetParams: virtual method for setting algorithm parameters.
>
> \- Init: virtual method to initialize the algorithm, passing the minimum and maximum search range, step and number of epochs.
>
> \- Moving: virtual method for executing an algorithm step.
>
> \- Revision: virtual method for carrying out a revision of the algorithm.
>
> \- GetName: method for getting the algorithm name.
>
> \- GetDesc: method for obtaining the algorithm description.
>
> \- GetParams: method for obtaining algorithm parameters as a string.

- Protected properties of the "C\_AO" class:

> \- ao\_name: algorithm name.
>
> \- "ao\_desc: algorithm description.
>
> \- rangeMin, rangeMax, rangeStep: arrays for storing the minimum and maximum search range, as well as the step.
>
> \- coords: number of coordinates.
>
> \- popSize: population size.
>
> \- revision: revision flag.
>
> \- u: "C\_AO\_Utilities" class object for performing auxiliary functions.

```
#include "#C_AO_Utilities.mqh"

//——————————————————————————————————————————————————————————————————————————————
class C_AO
{
  public: //--------------------------------------------------------------------
  C_AO () { }
  ~C_AO () { for (int i = 0; i < ArraySize (a); i++) delete a [i];}

  double      cB     []; //best coordinates
  double      fB;        //FF of the best coordinates
  C_AO_Agent *a      []; //agents
  S_AlgoParam params []; //algorithm parameters

  virtual void SetParams () { }
  virtual bool Init (const double &rangeMinP  [], //minimum search range
                     const double &rangeMaxP  [], //maximum search range
                     const double &rangeStepP [], //step search
                     const int     epochsP = 0)   //number of epochs
  { return false;}

  virtual void Moving   () { }
  virtual void Revision () { }

  string GetName   () { return ao_name;}
  string GetDesc   () { return ao_desc;}
  string GetParams ()
  {
    string str = "";
    for (int i = 0; i < ArraySize (params); i++)
    {
      str += (string)params [i].val + "|";
    }
    return str;
  }

  protected: //-----------------------------------------------------------------
  string ao_name;      //ao name;
  string ao_desc;      //ao description

  double rangeMin  []; //minimum search range
  double rangeMax  []; //maximum search range
  double rangeStep []; //step search

  int    coords;       //coordinates number
  int    popSize;      //population size
  bool   revision;

  C_AO_Utilities u;     //auxiliary functions

  bool StandardInit (const double &rangeMinP  [], //minimum search range
                     const double &rangeMaxP  [], //maximum search range
                     const double &rangeStepP []) //step search
  {
    MathSrand ((int)GetMicrosecondCount ()); //reset of the generator
    fB       = -DBL_MAX;
    revision = false;

    coords  = ArraySize (rangeMinP);
    if (coords == 0 || coords != ArraySize (rangeMaxP) || coords != ArraySize (rangeStepP)) return false;

    ArrayResize (rangeMin,  coords);
    ArrayResize (rangeMax,  coords);
    ArrayResize (rangeStep, coords);
    ArrayResize (cB,        coords);

    ArrayCopy (rangeMin,  rangeMinP,  0, 0, WHOLE_ARRAY);
    ArrayCopy (rangeMax,  rangeMaxP,  0, 0, WHOLE_ARRAY);
    ArrayCopy (rangeStep, rangeStepP, 0, 0, WHOLE_ARRAY);

    return true;
  }
};
//——————————————————————————————————————————————————————————————————————————————
```

Also in the same file, along with the base class, there is the "E\_AO" enumeration, which contains IDs of optimization algorithms and the SelectAO function, which allows us to create an instance of the corresponding algorithm and obtain its pointer.

```
#include "AO_BGA_Binary_Genetic_Algorithm.mqh"
#include "AO_(P_O)ES_Evolution_Strategies.mqh"
#include "AO_DE_Differential_Evolution.mqh"
#include "AO_SDSm_Stochastic_Diffusion_Search.mqh"
#include "AO_ESG_Evolution_of_Social_Groups.mqh";

//——————————————————————————————————————————————————————————————————————————————
enum E_AO
{
  AO_BGA,
  AO_P_O_ES,
  AO_SDSm,
  AO_ESG,
  AO_DE,
  AO_NONE
};
C_AO *SelectAO (E_AO a)
{
  C_AO *ao;
  switch (a)
  {
    case  AO_BGA:
      ao = new C_AO_BGA (); return (GetPointer (ao));
    case  AO_P_O_ES:
      ao = new C_AO_P_O_ES (); return (GetPointer (ao));
    case  AO_SDSm:
      ao = new C_AO_SDSm (); return (GetPointer (ao));
    case  AO_ESG:
      ao = new C_AO_ESG (); return (GetPointer (ao));
    case  AO_DE:
      ao = new C_AO_DE (); return (GetPointer (ao));

    default:
      ao = NULL; return NULL;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

### 3\. Code of descendant algorithms

Let's consider the stochastic diffusion search algorithm (SDSm) as an example of inheritance from the base class. We will inherit the "C\_SDS\_Agent" of this algorithm from the base "C\_AO\_Agent". Note that the agent initialization method contains "c" coordinates and "f" fitness, but they are not declared in the "C\_SDS\_Agent" class. This makes sense, since these attributes are required for all optimization algorithm agents and are inherited from the base algorithm, so there is no need to declare them again.

```
//——————————————————————————————————————————————————————————————————————————————
class C_SDS_Agent : public C_AO_Agent
{
  public: //--------------------------------------------------------------------
  ~C_SDS_Agent () { }

  int    raddr     []; //restaurant address
  int    raddrPrev []; //previous restaurant address
  double cPrev     []; //previous coordinates (dishes)
  double fPrev;        //previous fitness

  void Init (int coords)
  {
    ArrayResize (c,         coords);
    ArrayResize (cPrev,     coords);
    ArrayResize (raddr,     coords);
    ArrayResize (raddrPrev, coords);
    f        = -DBL_MAX;
    fPrev    = -DBL_MAX;
  }
};
//——————————————————————————————————————————————————————————————————————————————
```

The "C\_AO\_SDSm" class of the SDSm algorithm is derived from the "C\_AO" class. When declaring a class object in the constructor, we will initialize the external parameters of the algorithm, which can subsequently be changed by the user if desired. The parameters will be available in the form of an array and we will not have to worry about compatibility with the test bench.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_SDSm : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_SDSm () { }
  C_AO_SDSm ()
  {
    ao_name = "SDSm";
    ao_desc = "Stochastic Diffusion Search";

    popSize    = 100; //population size

    restNumb   = 100;  //restaurants number
    probabRest = 0.05; //probability restaurant choosing

    ArrayResize (params, 3);

    params [0].name = "popSize";    params [0].val  = popSize;

    params [1].name = "restNumb";   params [1].val  = restNumb;
    params [2].name = "probabRest"; params [2].val  = probabRest;
  }

  void SetParams ()
  {
    popSize    = (int)params [0].val;

    restNumb   = (int)params [1].val;
    probabRest = params      [2].val;
  }

  bool Init (const double &rangeMinP  [], //minimum search range
             const double &rangeMaxP  [], //maximum search range
             const double &rangeStepP [], //step search
             const int     epochsP = 0);  //number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  int    restNumb;          //restaurants number
  double probabRest;        //probability restaurant choosing

  C_SDS_Agent *agent []; //candidates

  private: //-------------------------------------------------------------------
  struct S_Riverbed //river bed
  {
      double coordOnSector []; //coordinate on the sector (number of cells: number of sectors on the coordinate, cell value: specific coordinate on the sector)
  };

  double restSpace [];      //restaurants space
  S_Riverbed    rb [];      //riverbed

  void Research  (const double  ko,
                  const int     raddr,
                  const double  restSpace,
                  const double  rangeMin,
                  const double  rangeStep,
                  const double  pitOld,
                  double       &pitNew);
};
//——————————————————————————————————————————————————————————————————————————————
```

Next, we should especially consider the "Init" initialization method of the "C\_AO\_SDSm" class. The method does the following:

1\. From the very beginning, we need to call the base class method "StandardInit" and pass "rangeMinP", "rangeMaxP", "rangeStepP" to it. If the method returns "false", the "Init" function also returns "false", indicating that the algorithm initialization failed.

2\. Removing agents using "delete". This is necessary when reusing an algorithm object.

3\. Then we change the size of the "agent" arrays of the SDSm algorithm and "a" of the base class to "popSize" and perform a type cast.

4\. Next, we perform actions similar to the algorithm described [in the article about SDSm](https://www.mql5.com/en/articles/13540).

```
//——————————————————————————————————————————————————————————————————————————————
bool C_AO_SDSm::Init (const double &rangeMinP  [], //minimum search range
                      const double &rangeMaxP  [], //maximum search range
                      const double &rangeStepP [], //step search
                      const int     epochsP = 0)   //number of epochs
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  for (int i = 0; i < ArraySize (agent); i++) delete agent [i];

  ArrayResize (agent, popSize);
  ArrayResize (a,     popSize);

  for (int i = 0; i < popSize; i++)
  {
    a     [i] = new C_SDS_Agent ();
    agent [i] = (C_SDS_Agent *)a [i];

    agent [i].Init (coords);
  }

  ArrayResize (restSpace, coords);
  ArrayResize (rb,        coords);
  for (int i = 0; i < coords; i++)
  {
    ArrayResize     (rb [i].coordOnSector, restNumb);
    ArrayInitialize (rb [i].coordOnSector, -DBL_MAX);
  }

  for (int i = 0; i < coords; i++)
  {
    restSpace [i] = (rangeMax [i] - rangeMin [i]) / restNumb;
  }

  return true;
}
//——————————————————————————————————————————————————————————————————————————————
```

### 4\. Code of the unified test stand for all algorithms

Although at the moment there is no need to "multiply" test stands, we will still transfer all the functions of the test stand to the "C\_TestStand" class. This will allow us to conveniently encapsulate the stand functionality. Since no significant changes have occurred in the stand functions, I will not describe them in detail. Let's simply have a look at its current state:

```
#include <Canvas\Canvas.mqh>
#include <\Math\Functions.mqh>

//——————————————————————————————————————————————————————————————————————————————
class C_TestStand
{
  public: void Init (int width, int height)
  {
    W = width;  //750;
    H = height; //375;

    WscrFunc = H - 2;
    HscrFunc = H - 2;

    //creating a table ---------------------------------------------------------
    string canvasName = "AO_Test_Func_Canvas";

    if (!Canvas.CreateBitmapLabel (canvasName, 5, 30, W, H, COLOR_FORMAT_ARGB_RAW))
    {
      Print ("Error creating Canvas: ", GetLastError ());
      return;
    }

    ObjectSetInteger (0, canvasName, OBJPROP_HIDDEN, false);
    ObjectSetInteger (0, canvasName, OBJPROP_SELECTABLE, true);

    ArrayResize (FunctScrin, HscrFunc);

    for (int i = 0; i < HscrFunc; i++) ArrayResize (FunctScrin [i].clr, HscrFunc);

  }

  struct S_CLR
  {
    color clr [];
  };

  //----------------------------------------------------------------------------
  public: void CanvasErase ()
  {
    Canvas.Erase (XRGB (0, 0, 0));
    Canvas.FillRectangle (1,     1, H - 2, H - 2, COLOR2RGB (clrWhite));
    Canvas.FillRectangle (H + 1, 1, W - 2, H - 2, COLOR2RGB (clrWhite));
  }

  //----------------------------------------------------------------------------
  public: void MaxMinDr (C_Function & f)
  {
    //draw Max global-------------------------------------------------------------
    int x = (int)Scale(f.GetMaxFuncX(), f.GetMinRangeX(), f.GetMaxRangeX(), 1, W/2 - 1, false);
    int y = (int)Scale(f.GetMaxFuncY(), f.GetMinRangeY(), f.GetMaxRangeY(), 1, H   - 1, true);

    Canvas.Circle(x, y, 12, COLOR2RGB(clrBlack));
    Canvas.Circle(x, y, 13, COLOR2RGB(clrBlack));
    Canvas.Circle(x, y, 14, COLOR2RGB(clrBlack));
    Canvas.Circle(x, y, 15, COLOR2RGB(clrBlack));

    //draw Min global-------------------------------------------------------------
    x = (int)Scale(f.GetMinFuncX(), f.GetMinRangeX(), f.GetMaxRangeX(), 0, W/2 - 1, false);
    y = (int)Scale(f.GetMinFuncY(), f.GetMinRangeY(), f.GetMaxRangeY(), 0, H - 1, true);

    Canvas.Circle(x, y, 12, COLOR2RGB(clrBlack));
    Canvas.Circle(x, y, 13, COLOR2RGB(clrBlack));
  }

  //----------------------------------------------------------------------------
  public: void PointDr (double &args [], C_Function & f, int shiftX, int shiftY, int count, bool main)
  {
    double x = 0.0;
    double y = 0.0;

    double xAve = 0.0;
    double yAve = 0.0;

    int width  = 0;
    int height = 0;

    color clrF = clrNONE;

    for(int i = 0; i < count; i++)
    {
      xAve += args [i * 2];
      yAve += args [i * 2 + 1];

      x = args [i * 2];
      y = args [i * 2 + 1];

      width  = (int)Scale(x, f.GetMinRangeX(), f.GetMaxRangeX(), 0, WscrFunc - 1, false);
      height = (int)Scale(y, f.GetMinRangeY(), f.GetMaxRangeY(), 0, HscrFunc - 1, true);

      clrF = DoubleToColor(i, 0, count - 1, 0, 270);
      Canvas.FillCircle(width + shiftX, height + shiftY, 1, COLOR2RGB(clrF));
    }

    xAve /=(double)count;
    yAve /=(double)count;

    width  = (int)Scale(xAve, f.GetMinRangeX(), f.GetMaxRangeX(), 0, WscrFunc - 1, false);
    height = (int)Scale(yAve, f.GetMinRangeY(), f.GetMaxRangeY(), 0, HscrFunc - 1, true);

    if(!main)
    {
      Canvas.FillCircle(width + shiftX, height + shiftY, 3, COLOR2RGB(clrBlack));
      Canvas.FillCircle(width + shiftX, height + shiftY, 2, COLOR2RGB(clrWhite));
    }
    else
    {
      Canvas.Circle (width + shiftX, height + shiftY, 5, COLOR2RGB (clrBlack));
      Canvas.Circle (width + shiftX, height + shiftY, 6, COLOR2RGB (clrBlack));
    }
  }

  //----------------------------------------------------------------------------
  public: void SendGraphToCanvas ()
  {
    for (int w = 0; w < HscrFunc; w++)
    {
      for (int h = 0; h < HscrFunc; h++)
      {
        Canvas.PixelSet (w + 1, h + 1, COLOR2RGB (FunctScrin [w].clr [h]));
      }
    }
  }

  //----------------------------------------------------------------------------
  public: void DrawFunctionGraph (C_Function & f)
  {
    double ar [2];
    double fV;

    for (int w = 0; w < HscrFunc; w++)
    {
      ar [0] = Scale (w, 0, H, f.GetMinRangeX (), f.GetMaxRangeX (), false);
      for (int h = 0; h < HscrFunc; h++)
      {
        ar [1] = Scale (h, 0, H, f.GetMinRangeY (), f.GetMaxRangeY (), true);
        fV = f.CalcFunc (ar, 1);
        FunctScrin [w].clr [h] = DoubleToColor (fV, f.GetMinFunValue (), f.GetMaxFunValue (), 0, 270);
      }
    }
  }

  //----------------------------------------------------------------------------
  public: void Update ()
  {
    Canvas.Update ();
  }

  //----------------------------------------------------------------------------
  //Scaling a number from a range to a specified range
  public: double Scale (double In, double InMIN, double InMAX, double OutMIN, double OutMAX, bool Revers = false)
  {
    if (OutMIN == OutMAX) return (OutMIN);
    if (InMIN == InMAX) return ((OutMIN + OutMAX) / 2.0);
    else
    {
      if (Revers)
      {
        if (In < InMIN) return (OutMAX);
        if (In > InMAX) return (OutMIN);
        return (((InMAX - In) * (OutMAX - OutMIN) / (InMAX - InMIN)) + OutMIN);
      }
      else
      {
        if (In < InMIN) return (OutMIN);
        if (In > InMAX) return (OutMAX);
        return (((In - InMIN) * (OutMAX - OutMIN) / (InMAX - InMIN)) + OutMIN);
      }
    }
  }

  //----------------------------------------------------------------------------
  private: color DoubleToColor (const double In,    //input value
                                const double inMin, //minimum of input values
                                const double inMax, //maximum of input values
                                const int    loH,   //lower bound of HSL range values
                                const int    upH)   //upper bound of HSL range values
  {
    int h = (int) Scale (In, inMin, inMax, loH, upH, true);
    return HSLtoRGB (h, 1.0, 0.5);
  }

  //----------------------------------------------------------------------------
  private: color HSLtoRGB (const int    h, //0   ... 360
                           const double s, //0.0 ... 1.0
                           const double l) //0.0 ... 1.0
  {
    int r;
    int g;
    int b;
    if (s == 0.0)
    {
      r = g = b = (unsigned char)(l * 255);
      return StringToColor ((string) r + "," + (string) g + "," + (string) b);
    }
    else
    {
      double v1, v2;
      double hue = (double) h / 360.0;
      v2 = (l < 0.5) ? (l * (1.0 + s)) : ((l + s) - (l * s));
      v1 = 2.0 * l - v2;
      r = (unsigned char)(255 * HueToRGB (v1, v2, hue + (1.0 / 3.0)));
      g = (unsigned char)(255 * HueToRGB (v1, v2, hue));
      b = (unsigned char)(255 * HueToRGB (v1, v2, hue - (1.0 / 3.0)));
      return StringToColor ((string) r + "," + (string) g + "," + (string) b);
    }
  }

  //----------------------------------------------------------------------------
  private: double HueToRGB (double v1, double v2, double vH)
  {
    if (vH < 0) vH += 1;
    if (vH > 1) vH -= 1;
    if ((6 * vH) < 1) return (v1 + (v2 - v1) * 6 * vH);
    if ((2 * vH) < 1) return v2;
    if ((3 * vH) < 2) return (v1 + (v2 - v1) * ((2.0f / 3) - vH) * 6);
    return v1;
  }

  //----------------------------------------------------------------------------
  public: int W; //monitor screen width
  public: int H; //monitor screen height

  private: int WscrFunc; //test function screen width
  private: int HscrFunc; //test function screen height

  public:  CCanvas Canvas;      //drawing table
  private: S_CLR FunctScrin []; //two-dimensional matrix of colors
};
//——————————————————————————————————————————————————————————————————————————————
```

Now let's take a look at the test stand code itself. When studying the stand inputs, it becomes clear that we are now able to select the optimization algorithm and test functions in the settings. This allows us to create unique sets of test functions to test the algorithm. Now we can perform tests only on smooth or only on discrete functions. Moreover, it is now possible to select any combination of test functions that best suits the user's needs.

```
#include "PAO\#C_TestStandFunctions.mqh"
#include "PAO\#C_AO.mqh"

//——————————————————————————————————————————————————————————————————————————————
input string AOparam            = "----------------"; //AO parameters-----------
input E_AO   AOexactly_P        = AO_NONE;

input string TestStand_1        = "----------------"; //Test stand--------------
input double ArgumentStep_P     = 0.0;   //Argument Step

input string TestStand_2        = "----------------"; //------------------------
input int    Test1FuncRuns_P    = 5;     //Test #1: Number of functions in the test
input int    Test2FuncRuns_P    = 25;    //Test #2: Number of functions in the test
input int    Test3FuncRuns_P    = 500;   //Test #3: Number of functions in the test

input string TestStand_3        = "----------------"; //------------------------
input EFunc  Function1          = Hilly;
input EFunc  Function2          = Forest;
input EFunc  Function3          = Megacity;

input string TestStand_4        = "----------------"; //------------------------
input int    NumbTestFuncRuns_P = 10000; //Number of test function runs
input int    NumberRepetTest_P  = 10;    //Test repets number

input string TestStand_5        = "----------------"; //------------------------
input bool   Video_P            = true;  //Show video
//——————————————————————————————————————————————————————————————————————————————

//——————————————————————————————————————————————————————————————————————————————
void OnStart ()
{
  C_AO *AO = SelectAO (AOexactly_P);
  if (AO == NULL)
  {
    Print ("AO is not selected...");
    return;
  }

  Print (AO.GetName (), "|", AO.GetDesc (), "|", AO.GetParams ());

  //============================================================================
  C_TestStand ST; //stand
  ST.Init (750, 375);

  double allScore = 0.0;
  double allTests = 0.0;

  C_Function *F1 = SelectFunction (Function1);
  C_Function *F2 = SelectFunction (Function2);
  C_Function *F3 = SelectFunction (Function3);

  if (F1 != NULL)
  {
    Print ("=============================");
    ST.CanvasErase ();

    FuncTests (AO, ST, F1, Test1FuncRuns_P, clrLime,      allScore, allTests);
    FuncTests (AO, ST, F1, Test2FuncRuns_P, clrAqua,      allScore, allTests);
    FuncTests (AO, ST, F1, Test3FuncRuns_P, clrOrangeRed, allScore, allTests);
    delete F1;
  }

  if (F2 != NULL)
  {
    Print ("=============================");
    ST.CanvasErase ();
    FuncTests (AO, ST, F2, Test1FuncRuns_P, clrLime,      allScore, allTests);
    FuncTests (AO, ST, F2, Test2FuncRuns_P, clrAqua,      allScore, allTests);
    FuncTests (AO, ST, F2, Test3FuncRuns_P, clrOrangeRed, allScore, allTests);
    delete F2;
  }

  if (F3 != NULL)
  {
    Print ("=============================");
    ST.CanvasErase ();
    FuncTests (AO, ST, F3, Test1FuncRuns_P, clrLime,      allScore, allTests);
    FuncTests (AO, ST, F3, Test2FuncRuns_P, clrAqua,      allScore, allTests);
    FuncTests (AO, ST, F3, Test3FuncRuns_P, clrOrangeRed, allScore, allTests);
    delete F3;
  }

  Print ("=============================");
  if (allTests > 0.0) Print ("All score: ", DoubleToString (allScore, 5), " (", DoubleToString (allScore * 100 / allTests, 2), "%)");
  delete AO;
}
//——————————————————————————————————————————————————————————————————————————————

//——————————————————————————————————————————————————————————————————————————————
void FuncTests (C_AO          &ao,
                C_TestStand   &st,
                C_Function    &f,
                const  int     funcCount,
                const  color   clrConv,
                double        &allScore,
                double        &allTests)
{
  if (funcCount <= 0) return;

  allTests++;

  if (Video_P)
  {
    st.DrawFunctionGraph (f);
    st.SendGraphToCanvas ();
    st.MaxMinDr          (f);
    st.Update            ();
  }

  int    xConv      = 0.0;
  int    yConv      = 0.0;
  double aveResult  = 0.0;
  int    params     = funcCount * 2;
  int    epochCount = NumbTestFuncRuns_P / (int)ao.params [0].val;

  //----------------------------------------------------------------------------
  double rangeMin  [], rangeMax  [], rangeStep [];
  ArrayResize (rangeMin,  params);
  ArrayResize (rangeMax,  params);
  ArrayResize (rangeStep, params);

  for (int i = 0; i < funcCount; i++)
  {
    rangeMax  [i * 2] = f.GetMaxRangeX ();
    rangeMin  [i * 2] = f.GetMinRangeX ();
    rangeStep [i * 2] = ArgumentStep_P;

    rangeMax  [i * 2 + 1] = f.GetMaxRangeY ();
    rangeMin  [i * 2 + 1] = f.GetMinRangeY ();
    rangeStep [i * 2 + 1] = ArgumentStep_P;
  }

  for (int test = 0; test < NumberRepetTest_P; test++)
  {
    //--------------------------------------------------------------------------
    if (!ao.Init (rangeMin, rangeMax, rangeStep)) break;

    // Optimization-------------------------------------------------------------
    for (int epochCNT = 1; epochCNT <= epochCount && !IsStopped (); epochCNT++)
    {
      ao.Moving ();

      for (int set = 0; set < ArraySize (ao.a); set++)
      {
        ao.a [set].f = f.CalcFunc (ao.a [set].c, funcCount);
      }

      ao.Revision  ();

      if (Video_P)
      {
        //drawing a population--------------------------------------------------
        st.SendGraphToCanvas  ();

        for (int i = 0; i < ArraySize (ao.a); i++)
        {
          st.PointDr (ao.a [i].c, f, 1, 1, funcCount, false);
        }
        st.PointDr (ao.cB, f, 1, 1, funcCount, true);

        st.MaxMinDr (f);

        //drawing a convergence graph---------------------------------------------
        xConv = (int)st.Scale (epochCNT, 1, epochCount, st.H + 2, st.W - 3, false);
        yConv = (int)st.Scale (ao.fB, f.GetMinFunValue (), f.GetMaxFunValue (), 2, st.H - 2, true);
        st.Canvas.FillCircle (xConv, yConv, 1, COLOR2RGB (clrConv));

        st.Update ();
      }
    }

    aveResult += ao.fB;
  }

  aveResult /= (double)NumberRepetTest_P;

  double score = aveResult;

  Print (funcCount, " ", f.GetFuncName (), "'s; Func runs: ", NumbTestFuncRuns_P, "; result: ", aveResult);
  allScore += score;
}
//——————————————————————————————————————————————————————————————————————————————
```

### 5\. Let's add popular test functions

Sometimes I am asked why I did not include such widely known and actively used optimization algorithms in the research and development set of test functions. The reason lies in the fact that they all fall into the category of "simple" test functions according to my classification. More than half of their surfaces are concentrated near the global extreme, making them too "predictable" for proper tests. But despite this, people are accustomed to trusting them and tend to test their algorithms on such functions. So I decided to include Ackley, Goldstein-Price and Shaffer #2 in the set. This will help balance the choice of test functions for users and provide more comprehensive and reliable tests of optimization algorithms, opening new horizons for researchers and contributing to a deeper understanding of their efficiency.

Ackley equation:

> **f(x, y) = -(-20 \* exp(-0.2 \* sqrt(0.5 \* (x^2 + y^2))) - exp(0.5 \* (cos(2 \* pi \* x) + cos(2 \* pi \* y))) + e + 20)**

where:

\- x, y - function inputs,

\- e - Euler number (approximately 2.71828),

\- π \- Pi number (approximately 3.14159).

Function code:

```
double Core (double x, double y)
{
  double res1 = -20.0 * MathExp (-0.2 * MathSqrt (0.5 * (x * x + y * y)));
  double res2 = -MathExp (0.5 * (MathCos (2.0 * M_PI * x) + MathCos (2.0 * M_PI * y)));
  double res3 = -(res1 + res2 + M_E + 20.0);

  return Scale (res3, -14.302667500265278, 0.0, 0.0, 1.0);
}
```

Goldstein-Price equation:

> **f(x, y) = -(\[1 + (x + y + 1)^2 \* (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\] \* \[30 + (2x - 3y)^2 \* (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\])**

Function code:

```
double Core (double x, double y)
{
  double part1 = 1 + MathPow ((x + y + 1), 2) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y);
  double part2 = 30 + MathPow ((2 * x - 3 * y), 2) * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y);

  return Scale (-part1 * part2, -1015690.2717980597, -3.0, 0.0, 1.0);
}
```

Shaffer #2 equation:

> **f(x, y) = -(0.5 + ((sin(x^2 - y^2)^2 - 0.5) / (1 + 0.001 \* (x^2 + y^2))^2))**

Function code:

```
double Core (double x, double y)
{
  double numerator   = MathPow (MathSin (x * x - y * y), 2) - 0.5;
  double denominator = MathPow (1 + 0.001 * (x * x + y * y), 2);

  return Scale (-(0.5 + numerator / denominator), -0.9984331449753265, 0, 0, 1.0);
}
```

Note: test function values are normalized to \[0.0, 1.0\].

### 6\. Building 3D test functions

Sometimes, it is necessary not just to abstract from numbers and equations, but to see them visually for a deeper understanding of test functions and their relief. So I decided to use the **ability to build 3D scenes using DirectX in the MetaTrader 5 platform**. I was inspired by the "...\\MQL5\\Experts\\Examples\\Math 3D Morpher\\Math 3D Morpher.mq5" file from the developers included in the standard distribution and visualizes the functions that I developed earlier (and which I plan to add to the list of functions for the test stand). So I decided to create a visualizer for test functions.

To do this, I had to expand the Functions.mqh file, which stores the class of test functions used in testing optimization algorithms. The additional functions that I have added will allow you not only to study changes in functions visually when modifying them if such a need arises, but also to more deeply explore their properties and features.

This not only makes my research more fun and visual, but also helps me more fully understand the behavior of test functions. Ultimately, 3D visualization helps me not just see numbers on screen, but directly interact with the shape and structure of functions, which is important when modifying them and analyzing properties.

Code of additional functions for building a model for a 3D scene:

```
//——————————————————————————————————————————————————————————————————————————————
//GenerateFunctionDataFixedSize
bool GenerateFunctionDataFixedSize (int x_size, int y_size, double &data [], double x_min, double x_max, double y_min, double y_max, C_Function &function)
{
  if (x_size < 2 || y_size < 2)
  {
    PrintFormat ("Error in data sizes: x_size=%d,y_size=%d", x_size, y_size);
    return (false);
  }

  double dx = (x_max - x_min) / (x_size - 1);
  double dy = (y_max - y_min) / (y_size - 1);

  ArrayResize (data, x_size * y_size);

  //---
  for (int j = 0; j < y_size; j++)
  {
    for (int i = 0; i < x_size; i++)
    {
      double x = x_min + i * dx;
      double y = y_min + j * dy;

      data [j * x_size + i] = function.Core (x, y);
    }
  }
  return (true);
}
//——————————————————————————————————————————————————————————————————————————————

//——————————————————————————————————————————————————————————————————————————————
//GenerateDataFixedSize
bool GenerateDataFixedSize (int x_size, int y_size, C_Function &function, double &data [])
{
  if (x_size < 2 || y_size < 2)
  {
    PrintFormat ("Error in data sizes: x_size=%d,y_size=%d", x_size, y_size);
    return (false);
  }

  return GenerateFunctionDataFixedSize (x_size, y_size, data,
                                        function.GetMinRangeX (),
                                        function.GetMaxRangeX (),
                                        function.GetMinRangeY (),
                                        function.GetMaxRangeY (),
                                        function);
}
//——————————————————————————————————————————————————————————————————————————————
```

Additionally, I added discrete variants of the already familiar functions - SkinDiscrete and HillyDiscrete.

![Hilly Discrete](https://c.mql5.com/2/70/Hilly_Discrete.gif)

**HillyDiscrete function**

![Skin Discrete](https://c.mql5.com/2/70/Skin_Discrete.gif)

**SkinDiscrete function**

![Shaffer](https://c.mql5.com/2/70/Shaffer.gif)

**Shaffer #2 function**

### 7\. Summary

We will continue to use the Hilly, Forest and Megacity test functions, which have already gained a reputation as true tests for optimization algorithms, helping to create a reliable ranking table. However, we should not limit ourselves to just these functions - after all, the list of test functions has been successfully expanded! So you are given freedom of choice: experiment, explore, discover new horizons, because this is the beauty of scientific research.

At the conclusion of our culinary exploration of hybrid optimization methods through base class inheritance, we see that establishing such a starting point opens the door to endless research possibilities. Combining various population algorithms into one class allows us not only to improve the accuracy and speed of finding optimal solutions, but also to create a universal tool for research.

Creating a single test bench that combines a variety of test functions — discrete, smooth, sharp, non-differentiable — opens up new opportunities for testing and comparing different optimization methods. This approach allows researchers not only to use standard sets of functions, but also to create their own test benches adapted to specific tasks and requirements.

Thus, combining population algorithms into one class and creating a universal test bench opens up an exciting path for us to create new "gastronomic" combinations of optimization methods allowing us to discover new flavor notes in the world of searching for optimal solutions. Let's continue this culinary experiment and create new masterpieces on the path to excellence and innovation in the field of optimization!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14331](https://www.mql5.com/ru/articles/14331)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14331.zip "Download all attachments in the single ZIP archive")

[AOs.zip](https://www.mql5.com/en/articles/download/14331/aos.zip "Download AOs.zip")(36.92 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Central Force Optimization (CFO) algorithm](https://www.mql5.com/en/articles/17167)
- [Neuroboids Optimization Algorithm (NOA)](https://www.mql5.com/en/articles/16992)
- [Successful Restaurateur Algorithm (SRA)](https://www.mql5.com/en/articles/17380)
- [Billiards Optimization Algorithm (BOA)](https://www.mql5.com/en/articles/17325)
- [Chaos Game Optimization (CGO)](https://www.mql5.com/en/articles/17047)
- [Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)
- [Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/469075)**
(2)


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
29 Feb 2024 at 14:02

A typo?

```
    W = width;  //750;
    H = height; //375;

    WscrFunc = H - 2; // W - 2
    HscrFunc = H - 2;
```

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
29 Feb 2024 at 15:13

**Stanislav Korotky [#](https://www.mql5.com/ru/forum/463313#comment_52564468):**

A typo?

No.

![MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://c.mql5.com/2/82/MQL5_Wizard_Techniques_you_should_know_Part_24__LOGO.png)[MQL5 Wizard Techniques you should know (Part 24): Moving Averages](https://www.mql5.com/en/articles/15135)

Moving Averages are a very common indicator that are used and understood by most Traders. We explore possible use cases that may not be so common within MQL5 Wizard assembled Expert Advisors.

![Multibot in MetaTrader (Part II): Improved dynamic template](https://c.mql5.com/2/71/Multibot_in_MetaTrader_Part_II_____LOGO__1.png)[Multibot in MetaTrader (Part II): Improved dynamic template](https://www.mql5.com/en/articles/14251)

Developing the theme of the previous article, I decided to create a more flexible and functional template that has greater capabilities and can be effectively used both in freelancing and as a base for developing multi-currency and multi-period EAs with the ability to integrate with external solutions.

![Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://c.mql5.com/2/82/Integrate_Your_Own_LLM_into_EA_Part_4____LOGO.png)[Integrate Your Own LLM into EA (Part 4): Training Your Own LLM with GPU](https://www.mql5.com/en/articles/13498)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part I)](https://c.mql5.com/2/81/Building_A_Candlestick_Trend_Constraint_Model_Part_5___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part I)](https://www.mql5.com/en/articles/14963)

We will breakdown the main MQL5 code into specified code snippets to illustrate the integration of Telegram and WhatsApp for receiving signal notifications from the Trend Constraint indicator we are creating in this article series. This will help traders, both novices and experienced developers, grasp the concept easily. First, we will cover the setup of MetaTrader 5 for notifications and its significance to the user. This will help developers in advance to take notes to further apply in their systems.

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/14331&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068125572145542589)

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

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).