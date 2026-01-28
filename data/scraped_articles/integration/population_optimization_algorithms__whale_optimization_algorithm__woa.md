---
title: Population optimization algorithms: Whale Optimization Algorithm (WOA)
url: https://www.mql5.com/en/articles/14414
categories: Integration, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T17:20:37.621537
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/14414&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068122428229481908)

MetaTrader 5 / Examples


### Contents

1\. [Introduction](https://www.mql5.com/en/articles/14414#tag1)

2\. [Algorithm](https://www.mql5.com/en/articles/14414#tag2)

3\. [Test results](https://www.mql5.com/en/articles/14414#tag3)

### 1\. Introduction

Humpback whales are majestic marine giants and masters of the ocean. When humpback whales begin their dance of the hunt, time seems to stand still, and every movement is filled with grace and precision. Bubble-net masters create a watery curtain of bubbles to confine and collect their prey in the center of the ring. This unique feeding method emphasizes their intelligence and strategic thinking in the hunting process.

The synchronization of their actions with other individuals, even strangers, speaks of deep mutual understanding and unity, demonstrating the ability for collective work and coordination, regardless of the circumstances.

Their ability to consume up to 1.4 metric tons of food per day underscores their role in the marine ecosystem as one of the ocean's most powerful consumers. Their life rhythm, alternating from periods of abundant hunting to times of peace and fasting, tells us about the greatness and unpredictability of the ocean, in which they reign. Winter time, when humpback whales hardly feed at all, provides evidence of their amazing ability to survive. They rely on fat reserves accumulated during rich catches to survive times when food becomes scarce. This reminds us of how nature teaches animals to be saving and wise in their use of resources.

Humpback whales are living chronicles of survival and adaptability, symbols of wisdom, the embodiment of the power and beauty of the ocean and an inexhaustible source of inspiration for us.

Whale Optimization Algorithm is a metaheuristic optimization algorithm proposed by Mirjalili and Lewis in 2016. They were inspired by the hunting behavior of whales.

Whales use a variety of hunting strategies, including "bubble net" and "spiral penetration". In a "bubble net", whales surround their prey by creating a "net" of bubbles to confuse and frighten the prey. In "spiral penetration", whales rise from the depths of the ocean in a spiral motion to capture prey.

These hunting strategies were abstractly modeled in the WOA algorithm. In the WOA algorithm, "whales" represent solutions to an optimization problem, while the "hunt" represents the search for the optimal solution.

### 2\. Algorithm

The WOA algorithm starts by initializing a population of whales with random positions. Then a leader is selected - the whale with the best value of the objective function (in fact, the best global solution). The positions of the remaining whales are updated taking into account the position of the leader. This occurs in two modes: exploration mode and exploitation mode. In exploration mode, whales update their positions using a random search around the current global best solution. In exploitation mode, whales update their positions, moving closer to their current best solution.

The WOA algorithm has been successfully applied to solve many optimization problems and has shown good results. However, like any optimization algorithm, it has its advantages and disadvantages. For example, it may be prone to hitting local optima and have a relatively slow convergence rate.

Here's how to relate interesting facts about humpback whales to the principles of the WOA algorithm:

- **Cooperation and coordination.** Humpback whales often hunt in groups, cooperating with each other for mutual success. This is reminiscent of the principles of the WOA algorithm, where individual agents (like whales) work together, exchanging information and coordinating their actions to achieve an optimal solution.

- **Intelligent strategies.** Humpback whales use a variety of intelligent hunting strategies, such as bubble-net formation and group coordination. The WOA algorithm is also based on intelligent optimization strategies, including finding optimal solutions and adapting to changing conditions.

- **Adaptability and efficiency.** Humpback whales demonstrate adaptability and efficiency in the hunting process, changing their methods depending on the situation. The WOA algorithm also strives for adaptability and efficiency by applying various optimization strategies and modifying its behavior to achieve better results.

Thus, the WOA algorithm takes inspiration from the unique hunting strategies and behaviors of humpback whales, which helps it solve optimization problems in various domains efficiently and intelligently.

The modified WOAm (Whale Optimization Algorithm) optimization algorithm includes several main stages:

1\. **Movement towards a global solution:**

\- At the initial stage of the algorithm, each "whale" (solution) moves towards the global optimal solution. This helps to explore the solution space and find the general direction towards the optimum.

2\. **Improving one's position:**

    \- Each whale tries to improve its current position, approaching a better solution. Individuals can change their parameters or strategies to achieve better performance.

3\. **Spiral movement:**

\- This stage is an important mechanism of the WOA algorithm. Whales can move in a spiral around the best solution, which helps them more efficiently explore the search space and get closer to the optimal solution.

4. **Migration:**

    \- This step involves randomly changing the positions of individuals to provide diversity and prevent getting stuck in local optima. Migration helps the algorithm avoid prematurely converging on a solution that is not good enough.

These steps together provide an efficient search for the optimal solution in the space of the optimization problem. I made changes by adding the fourth stage to improve the algorithm's resistance to getting stuck. Modeling the migration of whales in the wild to new territories in search of food is necessary when previous sources are depleted. This addition gives the algorithm additional flexibility and helps it explore the solution space more efficiently, reflecting survival and adaptation strategies in nature.

![probab](https://c.mql5.com/2/72/probab.png)

Figure 1. The range of A ratio values in the equation **A = 2.0 \* aKo \* r - aKo** depending on the current epoch

Pseudocode for the whale search optimization algorithm (WOAm):

1\. Initialize the population with a random position.

2\. Calculate fitness.

3\. Calculate aKo ratio: **aKo = 2.0 - epochNow \* (2.0 / epochs)**.

4\. Generate the "r" random number from a uniform distribution in the range from -1 to 1.

5\. Calculate A and C variables:

\- **A = 2.0 \* aKo \* r - aKo**(figure 1)

\- **C = 2.0 \* r**

6\. Set the values for "spiralCoeff", the random number "l" from the uniform distribution and the "p" random probability.

7\. If "p" is less than "refProb":

\- If the absolute value of A is greater than 1.0: **X = Xbest - A \* \|Xbest - Xprev\|**

\- Otherwise, select the "leaderInd" random index from 0 to "popSize" and: **X = Xlead - A \* \|Xlead - Xprev\|**

8\. Otherwise, if the random probability is less than spiralProb: **X = Xprev + \| Xprev** **\- X\| \* exp (b \* l) \* cos (2 \* M\_PI \* l)**

9\. Otherwise: **X = PowerDistribution (cB \[c\], rangeMin \[c\], rangeMin \[c\], 30)**

10\. Calculate fitness.

11\. Update global solution.

12\. Repeat from step 3 until the stop criterion is met.

![WOA](https://c.mql5.com/2/72/WOA.jpg)

Figure 2. The "bubble" whale hunt that inspired the WOA optimization algorithm

Let's move on to writing the code for the WOAm algorithm.

Define the structure "S\_WOA\_Agent" to describe each whale. Let's look at what is going on here:

1\. The structure contains the following fields:

    \- cPrev\[\] - array for storing the agent’s previous coordinates.

    \- fPrev - variable for storing the previous assessment (fitness) of the agent.

2\. Init - "S\_WOA\_Agent" structure method, which initializes the structure fields. It takes the "coords" integer argument used to resize the "cPrev" array using the ArrayResize function.

3\. fPrev = -DBL\_MAX - sets the initial value of the "fPrev" variable equal to the minimum possible value of a double number.

This code represents the basic data structure for agents in the WOAm optimization algorithm and initializes their fields when a new agent is created.

```
//——————————————————————————————————————————————————————————————————————————————
struct S_WOA_Agent
{
    double cPrev []; //previous coordinates
    double fPrev;    //previous fitness

    void Init (int coords)
    {
      ArrayResize (cPrev, coords);
      fPrev = -DBL_MAX;
    }
};
//——————————————————————————————————————————————————————————————————————————————
```

Let's define the "C\_AO\_WOAm" class of the WOAm algorithm, which is an inheritor of the base class of "C\_AO" population algorithms and contains the following fields and methods:

1\. Public fields:

- ao\_name - optimization algorithm name.
- ao\_desc - optimization algorithm description.
- popSize - population size.
- params - array of algorithm parameters.
- refProb - refinement probability.
- spiralCoeff - spiral ratio.
- spiralProb - spiral probability.
- agent - vector of agents.

2\. The options available are:

- C\_AO\_WOAm - class constructor that initializes the class fields.
- SetParams - method for setting algorithm parameters.
- Init - method for initializing the algorithm. The method accepts minimum and maximum search ranges, search step and number of epochs.
- Moving - method for moving agents.
- Revision - method for revising agents.

3\. Private fields:

- epochs - number of epochs.
- epochNow - current epoch.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_WOAm : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_WOAm () { }
  C_AO_WOAm ()
  {
    ao_name = "WOAm";
    ao_desc = "Whale Optimization Algorithm M";

    popSize = 100;   //population size

    ArrayResize (params, 4);

    params [0].name = "popSize";     params [0].val = popSize;
    params [1].name = "refProb";     params [1].val = refProb;
    params [2].name = "spiralCoeff"; params [2].val = spiralCoeff;
    params [3].name = "spiralProb";  params [3].val = spiralProb;
  }

  void SetParams ()
  {
    popSize     = (int)params [0].val;
    refProb     = params      [1].val;
    spiralCoeff = params      [2].val;
    spiralProb  = params      [3].val;
  }

  bool Init (const double &rangeMinP  [], //minimum search range
             const double &rangeMaxP  [], //maximum search range
             const double &rangeStepP [], //step search
             const int     epochsP = 0);  //number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  double refProb;     //refinement probability
  double spiralCoeff; //spiral coefficient
  double spiralProb;  //spiral probability

  S_WOA_Agent agent []; //vector

  private: //-------------------------------------------------------------------
  int  epochs;
  int  epochNow;
};
//——————————————————————————————————————————————————————————————————————————————
```

The Init method of the "C\_AO\_WOAm" class is used to initialize class variables based on the passed parameters. This method performs standard initialization using the StandardInit method, which takes the minimum and maximum search ranges as well as the search step.

If standard initialization is successful, the method continues initializing the "epochs" and "epochNow" variables. The value of "epochs" is set to the "epochsP" passed parameter, and "epochNow" is initialized to 0.

The method then resizes the "agent" array to the size of "popSize". The Init method is called with the "coords" parameter for each element in "agent".

The method returns "true" if initialization was successful, and "false" otherwise.

This method performs the initial setup of the WOAm optimization algorithm with given parameters and prepares it to perform optimization for a given number of epochs.

```
//——————————————————————————————————————————————————————————————————————————————
bool C_AO_WOAm::Init (const double &rangeMinP  [], //minimum search range
                     const double &rangeMaxP  [], //maximum search range
                     const double &rangeStepP [], //step search
                     const int     epochsP = 0)   //number of epochs
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  epochs   = epochsP;
  epochNow = 0;

  ArrayResize (agent, popSize);
  for (int i = 0; i < popSize; i++) agent [i].Init (coords);

  return true;
}
//——————————————————————————————————————————————————————————————————————————————
```

The Moving method of the "C\_AO\_WOAm" class is used to move agents during optimization. The method does the following:

"epochNow++;" - current epoch value increases.

"if (!revision)" - checking if "revision" is "false".

If "revision" is "false" then:

- The agents' coordinates "a\[i\].c\[c\]" are initialized using random values in the specified ranges.
- The "revision" flag is set to "true".
- The method completes its work.

If "revision" is not equal to "false", then:

- For each agent, new coordinates are calculated using certain equations and probabilities.
- Various mathematical calculations, random numbers and probabilities are used to determine the new coordinates of the agents.
- New "x" coordinates are calculated according to the conditions and probabilities.
- The new coordinates "a\[i\].c\[c\]" are set using the SeInDiSp method to adjust the values according to the search ranges and steps.

The method is responsible for updating the coordinates of agents in the WOAm optimization algorithm in accordance with the current epoch, random values and probabilities. Let's highlight in color the corresponding sections of code that describe different whale behavior patterns:

**Global solution improvement:** With each new epoch, whales more carefully explore the vicinity of the found global solution according to the linear law and the calculated A ratio trying to get closer to the global optimum.

**Exploring the vicinity of the population "leader":** Taking into account the A ratio, the whales explore the vicinity of the population "leader", which is selected randomly according to the uniform distribution law. Thus, research continues into all areas where whales are present in the population.

**Spiral movement and "bubble" network:** : Whales move in a spiral pattern, taking into account their best individual position and current location, which ensures that the fish are collected in a dense cloud and the food is concentrated in one place.

**Whale migration:** Whale migration is simulated by abruptly moving away from the global best power law solution. This process determines a high probability of being in the vicinity of the global solution and a small but non-zero probability of being very far from it.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_WOAm::Moving ()
{
  epochNow++;

  //----------------------------------------------------------------------------
  if (!revision)
  {
    for (int i = 0; i < popSize; i++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [i].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [i].c [c] = u.SeInDiSp  (a [i].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      }
    }

    revision = true;
    return;
  }

  for (int i = 0; i < popSize; i++)
  {
    for (int c = 0; c < coords; c++)
    {
      double aKo = 2.0 - epochNow * (2.0 / epochs);
      double r = u.RNDfromCI (-1, 1);
      double A = 2.0 * aKo * r - aKo;
      double C = 2.0 * r;
      double b = spiralCoeff;
      double l = u.RNDfromCI (-1, 1);
      double p = u.RNDprobab ();
      double x;

      if (p < refProb)
      {
        if (MathAbs (A) > 1.0)
        {
          x = cB [c] - A * MathAbs (cB [c] - agent [i].cPrev [c]);                                                      //Xbest - A * |Xbest - X|
        }
        else
        {
          int leaderInd = u.RNDminusOne (popSize);
          x = agent [leaderInd].cPrev [c] - A * MathAbs (agent [leaderInd].cPrev [c] - agent [i].cPrev [c]);            //Xlid - A * |Xlid - X|;
        }
      }
      else
      {
        if (u.RNDprobab () < spiralProb)
        {
          x = agent [i].cPrev [c] + MathAbs (agent [i].cPrev [c] - a [i].c [c]) * MathExp (b * l) * cos (2 * M_PI * l); //XbestPrev + |XbestPrev - X| * MathExp (b * l) * cos (2 * M_PI * l)
        }
        else
        {
          x = u.PowerDistribution (cB [c], rangeMin [c], rangeMin [c], 30);
        }
      }

      a [i].c [c] = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The Revision method of the "C\_AO\_WOAm" class is used to update the best global solution and update the best positions of the whales themselves. The method does the following:

1\. Updating the global solution. In the "for" loop, the method iterates through all individuals. If the value of the fitness function of the current individual exceeds the current best value of the fitness function, the best value is updated and the array of coordinates of the current individual is copied to the array of coordinates of the best solution.

2\. Update previous fitness function values and agent coordinates. In the "for" loop, the method iterates through all individuals. If the fitness function of the current individual exceeds the previous value of the agent fitness function, then the previous value of the fitness function is updated and the array of coordinates of the current individual is copied to the array of the agent previous coordinates.

The Revision method is responsible for updating the best agents based on their feature values, as well as updating previous feature values and agent coordinates as part of the WOAm optimization algorithm.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_WOAm::Revision ()
{
  int ind = -1;

  for (int i = 0; i < popSize; i++)
  {
    if (a [i].f > fB) ind = i;
  }

  if (ind != -1)
  {
    fB = a [ind].f;
    ArrayCopy (cB, a [ind].c, 0, 0, WHOLE_ARRAY);
  }

  for (int i = 0; i < popSize; i++)
  {
    if (a [i].f > agent [i].fPrev)
    {
      agent [i].fPrev = a [i].f;
      ArrayCopy (agent [i].cPrev, a [i].c, 0, 0, WHOLE_ARRAY);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

### 3\. Test results

The basic version of the algorithm proposed by the authors, unfortunately, leaves much to be desired, demonstrating relatively weak results shown below.

WOA\|Whale Optimization Algorithm\|100.0\|0.1\|0.5\|0.8\|

=============================

5 Hilly's; Func runs: 10000; result: 0.45323929163422483

25 Hilly's; Func runs: 10000; result: 0.3158990997230676

500 Hilly's; Func runs: 10000; result: 0.25544320870775555

=============================

5 Forest's; Func runs: 10000; result: 0.43485195446891795

25 Forest's; Func runs: 10000; result: 0.2454326019188397

500 Forest's; Func runs: 10000; result: 0.1557433572339264

=============================

5 Megacity's; Func runs: 10000; result: 0.3400000000000001

25 Megacity's; Func runs: 10000; result: 0.18800000000000003

500 Megacity's; Func runs: 10000; result: 0.10146153846153938

=============================

All score: 2.49007 (27.67%)

However, through effort and creativity, significant improvements have been made, including elements such as migration and power-law distribution. Additionally, instead of using the whales' current position as a starting point to calculate the next position, the best previous whale positions are now used. These modifications transformed the algorithm, giving it a new quality and significantly improving the results of the modified version. Thus, the algorithm has become more efficient and capable of achieving greater success in solving the assigned problems.

Below are the results of a modified version of WOAm where there is an improvement of more than 22% (where 0% is the worst possible result and 100% is the best theoretical result achievable).

WOAm\|Whale Optimization Algorithm\|100.0\|0.1\|0.5\|0.8\|

=============================

5 Hilly's; Func runs: 10000; result: 0.8452089588169466

25 Hilly's; Func runs: 10000; result: 0.562977678943021

500 Hilly's; Func runs: 10000; result: 0.262626056156147

=============================

5 Forest's; Func runs: 10000; result: 0.9310009723200832

25 Forest's; Func runs: 10000; result: 0.5227806126625986

500 Forest's; Func runs: 10000; result: 0.1636489301696601

=============================

5 Megacity's; Func runs: 10000; result: 0.6630769230769229

25 Megacity's; Func runs: 10000; result: 0.41138461538461535

500 Megacity's; Func runs: 10000; result: 0.11356923076923182

=============================

All score: 4.47627 (49.74%)

In the visualization of the modified version operation, we can see a significant scatter of results for the Hilly function and a small scatter for the discrete Megacity. It is interesting to note that usually for most algorithms the Megacity function is more complex and has a larger scatter of results, and it is even more surprising that WOAm shows very stable and good results on this function.

The algorithm successfully explores local surface areas in various areas of the search space, highlighting promising directions. This is facilitated by the division of the general gregarious behavior of the population into stages of behavior of individual animals, aimed at improving the situation of each whale individually.

This approach allows the algorithm to effectively explore the search space for the optimal solution, focusing on improving the positions of individual agents in the pack. This facilitates more accurate and in-depth exploration of potentially promising areas, which in turn improves the overall performance of the algorithm.

The base version is not shown in this visualization. Additionally, a visualization of the algorithm’s operation on the Ackley test function is provided. This function is not involved in the rating table calculation.

![Hilly](https://c.mql5.com/2/72/Hilly.gif)

**WOAm on the [Hilly](https://www.mql5.com/en/articles/13923#tagHilly) test function**

![Forest](https://c.mql5.com/2/72/Forest.gif)

**WOAm on the [Forest](https://www.mql5.com/en/articles/11785#tag3) test function**

![Megacity](https://c.mql5.com/2/72/Megacity.gif)

**WOAm on the [Megacity](https://www.mql5.com/en/articles/11785#tag3) test function**

![Ackley](https://c.mql5.com/2/72/Ackley.gif)

**WOAm on the [Ackley](https://www.mql5.com/en/articles/14331#tag5) test function**

The modified WOAm algorithm took an honorable tenth place at the top of the table, demonstrating good and stable overall results.

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \# | AO | Description | Hilly | Hilly final | Forest | Forest final | Megacity (discrete) | Megacity final | Final result | % of MAX |
| 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) |
| 1 | BGA | [binary genetic algorithm](https://www.mql5.com/en/articles/14040) | 0.99992 | 0.99484 | 0.50483 | 2.49959 | 1.00000 | 0.99975 | 0.32054 | 2.32029 | 0.90667 | 0.96400 | 0.23035 | 2.10102 | 6.921 | 76.90 |
| 2 | (P+O)ES | [(P+O) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.99934 | 0.91895 | 0.56297 | 2.48127 | 1.00000 | 0.93522 | 0.39179 | 2.32701 | 0.83167 | 0.64433 | 0.21155 | 1.68755 | 6.496 | 72.18 |
| 3 | SDSm | [stochastic diffusion search M](https://www.mql5.com/en/articles/13540) | 0.93066 | 0.85445 | 0.39476 | 2.17988 | 0.99983 | 0.89244 | 0.19619 | 2.08846 | 0.72333 | 0.61100 | 0.10670 | 1.44103 | 5.709 | 63.44 |
| 4 | ESG | [evolution of social groups](https://www.mql5.com/en/articles/14136) | 0.99906 | 0.79654 | 0.35056 | 2.14616 | 1.00000 | 0.82863 | 0.13102 | 1.95965 | 0.82333 | 0.55300 | 0.04725 | 1.42358 | 5.529 | 61.44 |
| 5 | SIA | [simulated isotropic annealing](https://www.mql5.com/en/articles/13870) | 0.95784 | 0.84264 | 0.41465 | 2.21513 | 0.98239 | 0.79586 | 0.20507 | 1.98332 | 0.68667 | 0.49300 | 0.09053 | 1.27020 | 5.469 | 60.76 |
| 6 | DE | [differential evolution](https://www.mql5.com/en/articles/13781) | 0.95044 | 0.61674 | 0.30308 | 1.87026 | 0.95317 | 0.78896 | 0.16652 | 1.90865 | 0.78667 | 0.36033 | 0.02953 | 1.17653 | 4.955 | 55.06 |
| 7 | HS | [harmony search](https://www.mql5.com/en/articles/12163) | 0.86509 | 0.68782 | 0.32527 | 1.87818 | 0.99999 | 0.68002 | 0.09590 | 1.77592 | 0.62000 | 0.42267 | 0.05458 | 1.09725 | 4.751 | 52.79 |
| 8 | SSG | [saplings sowing and growing](https://www.mql5.com/en/articles/12268) | 0.77839 | 0.64925 | 0.39543 | 1.82308 | 0.85973 | 0.62467 | 0.17429 | 1.65869 | 0.64667 | 0.44133 | 0.10598 | 1.19398 | 4.676 | 51.95 |
| 9 | (PO)ES | [(PO) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.79025 | 0.62647 | 0.42935 | 1.84606 | 0.87616 | 0.60943 | 0.19591 | 1.68151 | 0.59000 | 0.37933 | 0.11322 | 1.08255 | 4.610 | 51.22 |
| 10 | WOAm | [wale optimization algorithm M](https://www.mql5.com/en/articles/14414) | 0.84521 | 0.56298 | 0.26263 | 1.67081 | 0.93100 | 0.52278 | 0.16365 | 1.61743 | 0.66308 | 0.41138 | 0.11357 | 1.18803 | 4.476 | 49.74 |
| 11 | ACOm | [ant colony optimization M](https://www.mql5.com/en/articles/11602) | 0.88190 | 0.66127 | 0.30377 | 1.84693 | 0.85873 | 0.58680 | 0.15051 | 1.59604 | 0.59667 | 0.37333 | 0.02472 | 0.99472 | 4.438 | 49.31 |
| 12 | BFO-GA | [bacterial foraging optimization - ga](https://www.mql5.com/en/articles/14011) | 0.89150 | 0.55111 | 0.31529 | 1.75790 | 0.96982 | 0.39612 | 0.06305 | 1.42899 | 0.72667 | 0.27500 | 0.03525 | 1.03692 | 4.224 | 46.93 |
| 13 | MEC | [mind evolutionary computation](https://www.mql5.com/en/articles/13432) | 0.69533 | 0.53376 | 0.32661 | 1.55569 | 0.72464 | 0.33036 | 0.07198 | 1.12698 | 0.52500 | 0.22000 | 0.04198 | 0.78698 | 3.470 | 38.55 |
| 14 | IWO | [invasive weed optimization](https://www.mql5.com/en/articles/11990) | 0.72679 | 0.52256 | 0.33123 | 1.58058 | 0.70756 | 0.33955 | 0.07484 | 1.12196 | 0.42333 | 0.23067 | 0.04617 | 0.70017 | 3.403 | 37.81 |
| 15 | Micro-AIS | [micro artificial immune system](https://www.mql5.com/en/articles/13951) | 0.79547 | 0.51922 | 0.30861 | 1.62330 | 0.72956 | 0.36879 | 0.09398 | 1.19233 | 0.37667 | 0.15867 | 0.02802 | 0.56335 | 3.379 | 37.54 |
| 16 | COAm | [cuckoo optimization algorithm M](https://www.mql5.com/en/articles/11786) | 0.75820 | 0.48652 | 0.31369 | 1.55841 | 0.74054 | 0.28051 | 0.05599 | 1.07704 | 0.50500 | 0.17467 | 0.03380 | 0.71347 | 3.349 | 37.21 |
| 17 | SDOm | [spiral dynamics optimization M](https://www.mql5.com/en/articles/12252) | 0.74601 | 0.44623 | 0.29687 | 1.48912 | 0.70204 | 0.34678 | 0.10944 | 1.15826 | 0.42833 | 0.16767 | 0.03663 | 0.63263 | 3.280 | 36.44 |
| 18 | NMm | [Nelder-Mead method M](https://www.mql5.com/en/articles/13805) | 0.73807 | 0.50598 | 0.31342 | 1.55747 | 0.63674 | 0.28302 | 0.08221 | 1.00197 | 0.44667 | 0.18667 | 0.04028 | 0.67362 | 3.233 | 35.92 |
| 19 | FAm | [firefly algorithm M](https://www.mql5.com/en/articles/11873) | 0.58634 | 0.47228 | 0.32276 | 1.38138 | 0.68467 | 0.37439 | 0.10908 | 1.16814 | 0.28667 | 0.16467 | 0.04722 | 0.49855 | 3.048 | 33.87 |
| 20 | GSA | [gravitational search algorithm](https://www.mql5.com/en/articles/12072) | 0.64757 | 0.49197 | 0.30062 | 1.44016 | 0.53962 | 0.36353 | 0.09945 | 1.00260 | 0.32667 | 0.12200 | 0.01917 | 0.46783 | 2.911 | 32.34 |
| 21 | BFO | [bacterial foraging optimization](https://www.mql5.com/en/articles/12031) | 0.61171 | 0.43270 | 0.31318 | 1.35759 | 0.54410 | 0.21511 | 0.05676 | 0.81597 | 0.42167 | 0.13800 | 0.03195 | 0.59162 | 2.765 | 30.72 |
| 22 | ABC | [artificial bee colony](https://www.mql5.com/en/articles/11736) | 0.63377 | 0.42402 | 0.30892 | 1.36671 | 0.55103 | 0.21874 | 0.05623 | 0.82600 | 0.34000 | 0.14200 | 0.03102 | 0.51302 | 2.706 | 30.06 |
| 23 | BA | [bat algorithm](https://www.mql5.com/en/articles/11915) | 0.59761 | 0.45911 | 0.35242 | 1.40915 | 0.40321 | 0.19313 | 0.07175 | 0.66810 | 0.21000 | 0.10100 | 0.03517 | 0.34617 | 2.423 | 26.93 |
| 24 | SA | [simulated annealing](https://www.mql5.com/en/articles/13851) | 0.55787 | 0.42177 | 0.31549 | 1.29513 | 0.34998 | 0.15259 | 0.05023 | 0.55280 | 0.31167 | 0.10033 | 0.02883 | 0.44083 | 2.289 | 25.43 |
| 25 | IWDm | [intelligent water drops M](https://www.mql5.com/en/articles/13730) | 0.54501 | 0.37897 | 0.30124 | 1.22522 | 0.46104 | 0.14704 | 0.04369 | 0.65177 | 0.25833 | 0.09700 | 0.02308 | 0.37842 | 2.255 | 25.06 |
| 26 | PSO | [particle swarm optimisation](https://www.mql5.com/en/articles/11386) | 0.59726 | 0.36923 | 0.29928 | 1.26577 | 0.37237 | 0.16324 | 0.07010 | 0.60572 | 0.25667 | 0.08000 | 0.02157 | 0.35823 | 2.230 | 24.77 |
| 27 | MA | [monkey algorithm](https://www.mql5.com/en/articles/12212) | 0.59107 | 0.42681 | 0.31816 | 1.33604 | 0.31138 | 0.14069 | 0.06612 | 0.51819 | 0.22833 | 0.08567 | 0.02790 | 0.34190 | 2.196 | 24.40 |
| 28 | SFL | [shuffled frog-leaping](https://www.mql5.com/en/articles/13366) | 0.53925 | 0.35816 | 0.29809 | 1.19551 | 0.37141 | 0.11427 | 0.04051 | 0.52618 | 0.27167 | 0.08667 | 0.02402 | 0.38235 | 2.104 | 23.38 |
| 29 | FSS | [fish school search](https://www.mql5.com/en/articles/11841) | 0.55669 | 0.39992 | 0.31172 | 1.26833 | 0.31009 | 0.11889 | 0.04569 | 0.47467 | 0.21167 | 0.07633 | 0.02488 | 0.31288 | 2.056 | 22.84 |
| 30 | RND | [random](https://www.mql5.com/en/articles/8122) | 0.52033 | 0.36068 | 0.30133 | 1.18234 | 0.31335 | 0.11787 | 0.04354 | 0.47476 | 0.25333 | 0.07933 | 0.02382 | 0.35648 | 2.014 | 22.37 |
| 31 | GWO | [grey wolf optimizer](https://www.mql5.com/en/articles/11785) | 0.59169 | 0.36561 | 0.29595 | 1.25326 | 0.24499 | 0.09047 | 0.03612 | 0.37158 | 0.27667 | 0.08567 | 0.02170 | 0.38403 | 2.009 | 22.32 |
| 32 | CSS | [charged system search](https://www.mql5.com/en/articles/13662) | 0.44252 | 0.35454 | 0.35201 | 1.14907 | 0.24140 | 0.11345 | 0.06814 | 0.42299 | 0.18333 | 0.06300 | 0.02322 | 0.26955 | 1.842 | 20.46 |
| 33 | EM | [electroMagnetism-like algorithm](https://www.mql5.com/en/articles/12352) | 0.46250 | 0.34594 | 0.32285 | 1.13129 | 0.21245 | 0.09783 | 0.10057 | 0.41085 | 0.15667 | 0.06033 | 0.02712 | 0.24412 | 1.786 | 19.85 |

### Summary

Overall, I am pleased with the improvements made to this algorithm as described in this article. However, there is more in-depth research into modifying this version, which is worth considering for those who are passionate about this topic and eager to further experiment. Diving into these techniques can open up new horizons and inspire even better solutions to a given algorithm.

Here are a few strategies that can help improve the whale search optimization algorithm (WOA) and reduce the likelihood of getting stuck at local extremes:

1\. **Population diversification.** Population diversity can help avoid getting stuck in local optima. Consider methods for maintaining the population diversity, such as mechanisms of mutation, crossover or solution neighborhood study.

2. **Elite strategy.** This strategy involves maintaining a number of several different best solutions (elites) from generation to generation, thereby avoiding a decrease in population diversity.

3\. **Multi-strategy mechanism**. This approach involves using multiple search strategies simultaneously, which can help the algorithm better explore the solution space and avoid local pitfalls.

4\. **Hybridization with other algorithms.** Hybridizing WOA with other optimization algorithms can also improve its performance. For example, we can use differential evolution or particle swarms to improve the exploration phase of the algorithm.

![tab](https://c.mql5.com/2/72/tab__1.png)

Figure 3. Color gradation of algorithms according to relevant tests Results greater than or equal to 0.99 are highlighted in white

The color of the value cell on the smooth Hilly function with 1000 variables is noteworthy, indicating that the result is the worst among all the algorithms presented in the table. I would also like to note the high performance indicator for the Forest function with five variables, and generally good performance for the Forest and Megacity functions.

![chart](https://c.mql5.com/2/72/chart__1.png)

Figure 4. The histogram of algorithm test results (on a scale from 0 to 100, the more the better,

where 100 is the maximum possible theoretical result, the archive features a script for calculating the rating table)

**WOAm pros and cons:**

Advantages:

1. Simple architecture and implementation.

2. Stable and good results on the sharp Forest function and discrete Megacity.
3. Not demanding on computing resources.

Disadvantages:

1. Low convergence (no results close to 100%).
2. Low scalability on smooth functions such as Hilly (problems with high dimensionality tasks).

The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14414](https://www.mql5.com/ru/articles/14414)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14414.zip "Download all attachments in the single ZIP archive")

[WOAm.zip](https://www.mql5.com/en/articles/download/14414/woam.zip "Download WOAm.zip")(24.34 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/470522)**
(2)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
21 Mar 2024 at 16:50

Variant of local multicore optimisation:

1. An [Expert Advisor](https://www.mql5.com/en/market/mt5/ "A Market of Applications for the MetaTrader 5 and MetaTrader 4")-Tester is launched on a chart.
2. It opens several charts with advisor-readers (optimisation algorithms from this article series): Agents.
3. The Expert Advisor from step 1 receives real-time data from the Expert Advisors from step 2.

Probably, if you try hard enough, you can make such a scheme.

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
21 Mar 2024 at 17:33

**fxsaber [#](https://www.mql5.com/ru/forum/464310#comment_52799301):**

A variant of local multicore optimisation:

1. An Expert Advisor-Tester is launched on a chart.
2. It opens several charts with advisor-readers (optimisation algorithms from this article series): Agents.
3. The Expert Advisor from step 1 receives real-time data from the Expert Advisors from step 2.

Probably, if you try hard enough, you can make such a scheme.

Yes, you can, assuming that each chart works in a separate thread. I tried that, but the charts hang, probably because I did it in scripts and not in Expert Advisors. I have not studied the question completely.

I know that a fully working scheme is to parallel on the kernels-agents of the staff optimiser, which searches only one single counter, and the advisor on the chart feeds the agents sets and takes back the result of FF.

![Developing a Replay System (Part 42): Chart Trade Project (I)](https://c.mql5.com/2/69/Desenvolvendo_um_sistema_de_Replay_3Parte_42x_Projeto_do_Chart_Trade_tIw___LOGO_.png)[Developing a Replay System (Part 42): Chart Trade Project (I)](https://www.mql5.com/en/articles/11652)

Let's create something more interesting. I don't want to spoil the surprise, so follow the article for a better understanding. From the very beginning of this series on developing the replay/simulator system, I was saying that the idea is to use the MetaTrader 5 platform in the same way both in the system we are developing and in the real market. It is important that this is done properly. No one wants to train and learn to fight using one tool while having to use another one during the fight.

![Build Self Optimizing Expert Advisors With MQL5 And Python](https://c.mql5.com/2/85/Build_Self_Optimizing_Expert_Advisors_With_MQL5_And_Python__LOGO.png)[Build Self Optimizing Expert Advisors With MQL5 And Python](https://www.mql5.com/en/articles/15040)

In this article, we will discuss how we can build Expert Advisors capable of autonomously selecting and changing trading strategies based on prevailing market conditions. We will learn about Markov Chains and how they can be helpful to us as algorithmic traders.

![Building A Candlestick Trend Constraint Model (Part 7): Refining our model for EA development](https://c.mql5.com/2/86/Building_A_Candlestick_Trend_Constraint_Model_Part_7___LOGO.png)[Building A Candlestick Trend Constraint Model (Part 7): Refining our model for EA development](https://www.mql5.com/en/articles/15154)

In this article, we will delve into the detailed preparation of our indicator for Expert Advisor (EA) development. Our discussion will encompass further refinements to the current version of the indicator to enhance its accuracy and functionality. Additionally, we will introduce new features that mark exit points, addressing a limitation of the previous version, which only identified entry points.

![MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://c.mql5.com/2/85/MQL5_Trading_Toolkit_Part_2___LOGO.png)[MQL5 Trading Toolkit (Part 2): Expanding and Implementing the Positions Management EX5 Library](https://www.mql5.com/en/articles/15224)

Learn how to import and use EX5 libraries in your MQL5 code or projects. In this continuation article, we will expand the EX5 library by adding more position management functions to the existing library and creating two Expert Advisors. The first example will use the Variable Index Dynamic Average Technical Indicator to develop a trailing stop trading strategy expert advisor, while the second example will utilize a trade panel to monitor, open, close, and modify positions. These two examples will demonstrate how to use and implement the upgraded EX5 position management library.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/14414&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068122428229481908)

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