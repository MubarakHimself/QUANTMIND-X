---
title: Adaptive Social Behavior Optimization (ASBO): Schwefel, Box-Muller Method
url: https://www.mql5.com/en/articles/15283
categories: Integration, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:08:10.311879
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/15283&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071601502063110920)

MetaTrader 5 / Examples


**Contents**

1. [Introduction](https://www.mql5.com/en/articles/15283#tag1)
2. [Implementing algorithm methods](https://www.mql5.com/en/articles/15283#tag2)

### 1\. Introduction

The collective behavior of living organisms shows an incredible wisdom. From schools of fish to human societies, cooperation and collaboration are key to survival and prosperity. But what if we could use these principles of social structure to create new optimization algorithms that can solve complex problems efficiently and accurately?

There are many examples of group behavior in nature, where living organisms join together in societies to increase their chances of survival and innovation. This phenomenon, observed in the animal kingdom, in human society, and in other forms of life, has become a fascinating subject of study for evolutionary biologists and social philosophers. By studying such societies, a computational model has been developed that simulates their successful functioning with respect to certain goals. These models, such as particle swarm optimization and ant colony optimization, demonstrate the efficiency of group work in solving optimization problems.

This article examines the concept of social structure and its influence on decision-making processes in groups. We also present a mathematical model based on principles of social behavior and interaction in societies that can be applied to achieve global optimization. This model, called ASBO (Adaptive Social Behavior Optimization), takes into account the influence of the environment on group decision making, including leadership, neighborhood, and self-organization. The algorithm was proposed by Manojo Kumar Singh and published in 2013 in "Proceedings of ICAdC, AISC 174" edited by Aswatha Kumar M. et al.

The structure of society and the model of influence:

- A society is a group of interconnected living beings united by common behavior or characteristics.
- The benefits of living in society: increased opportunities for hunting prey, reproduction, protection from predators, innovation.
- Key factors influencing the development of an individual in society: leadership, neighborhood, self-esteem.
- The proposed ASBO model is based on dynamic leadership, dynamic logical neighborhood and self-assessment.

The main principles of the ASBO algorithm:

- A population of solutions, each of which is a vector in a D-dimensional space.
- For each solution, three vectors are calculated: global best, personal best, and neighbor center.
- The new solution position is calculated using adaptive influence ratios.

- The influence ratios are adapted using a self-adaptive mutation strategy.

### 2\. Implementing algorithm methods

Before we dive into the algorithm itself, it is important to understand the basic concept behind it. This concept is related to the use of the Schwefel method, which is one of the approaches to self-adaptive mutation and is used in some optimization algorithms, such as evolutionary ones. Its main features:

1\. Self-adaptation of mutation parameters:

- Each individual (solution) has its own mutation parameters (for example, mutation step size).
- These mutation parameters also evolve along with the solutions themselves.
- Thus, the mutation step size is adapted to the function landscape for each individual.

2\. Gaussian mutation:

- Gaussian (normal) distribution is used to generate new solutions.
- The mean mutation value is equal to the previous solution, and the standard deviation is determined by the mutation parameters.

3\. Relationship between solution and mutation parameters:

- The mutation parameters (step size) depend on the value of the objective function (fitness) of the solution.
- The better the solution, the smaller the mutation step size, and vice versa.

The idea behind Schwefel's concept is that adapting the mutation parameters allows the algorithm to explore the solution space more efficiently, especially in the later stages of the search when more precise tuning of solutions is required.

The example below demonstrates the implementation of Schwefel's concept for optimizing the parameters of a trading strategy. The method operation itself is considered below.

As an example, let's take the simplest hypothetical EA, in which an initial population of random solutions is created during **OnInit** initialization. One step of the evolutionary process is completed in **OnTick**:

> a. The fitness of each individual in the population is assessed.
>
> b. The population is sorted by fitness.
>
> c. The mutation is applied to all individuals except the best one.
>
> d. The generation counter is increased.

This process is repeated until a specified number of generations is reached. After the optimization is completed, the best solution found is displayed.

```
// Inputs
input int    PopulationSize = 50;   // Population size
input int    Generations = 100;     // Number of generations
input double InitialStepSize = 1.0; // Initial step size

// Structure for storing an individual
struct Individual
{
    double genes [3];  // Strategy parameters
    double stepSizes [3];  // Step sizes for each parameter
    double fitness;   // Fitness
};

// Global variables
Individual population [];
int generation = 0;

// Initialization function
int OnInit ()
{
  ArrayResize (population, PopulationSize);
  InitializePopulation ();
  return (INIT_SUCCEEDED);
}

// Main loop function
datetime lastOptimizationTime = 0;

void OnTick ()
{
  datetime currentTime = TimeCurrent ();

  // Check if a day has passed since the last optimization
  if (currentTime - lastOptimizationTime >= 86400) // 86400 seconds in a day
  {
    Optimize ();
    lastOptimizationTime = currentTime;
  }

  // Here is the code for trading with the current optimal parameters
  TradingLogic ();
}

void Optimize ()
{
  // Optimization code (current OnTick content)
}

void TradingLogic ()
{
  // Implementing trading logic using optimized parameters
}

// Initialize the population
void InitializePopulation ()
{
  for (int i = 0; i < PopulationSize; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      population [i].genes [j] = MathRand () / 32767.0 * 100;  // Random values from 0 to 100
      population [i].stepSizes [j] = InitialStepSize;
    }
  }
}

// Population fitness assessment
void EvaluatePopulation ()
{
  for (int i = 0; i < PopulationSize; i++)
  {
    population [i].fitness = CalculateFitness (population [i]);
  }
}

// Calculate the fitness of an individual (this is where you need to implement your objective function)
double CalculateFitness (Individual &ind)
{
  // Example of a simple objective function
  return -(MathPow (ind.genes [0] - 50, 2) + MathPow (ind.genes [1] - 50, 2) + MathPow (ind.genes [2] - 50, 2));
}

// Sort the population in descending order of fitness
void SortPopulation ()
{
  ArraySort (population, WHOLE_ARRAY, 0, MODE_DESCEND);
}

// Population mutation according to Schwefel's concept
void Mutate ()
{
  for (int i = 1; i < PopulationSize; i++)  // Start from 1 to keep the best solution
  {
    for (int j = 0; j < 3; j++)
    {
      // Step size mutation
      population [i].stepSizes [j] *= MathExp (0.2 * MathRandom () - 0.1);

      // Gene mutation
      population [i].genes [j] += population [i].stepSizes [j] * NormalRandom ();

      // Limiting gene values
      population [i].genes [j] = MathMax (0, MathMin (100, population [i].genes [j]));
    }
  }
}

// Auxiliary function for displaying information about an individual
void PrintIndividual (Individual &ind)
{
  Print ("Genes: ", ind.genes [0], ", ", ind.genes [1], ", ", ind.genes [2]);
  Print ("Step sizes: ", ind.stepSizes [0], ", ", ind.stepSizes [1], ", ", ind.stepSizes [2]);
  Print ("Fitness: ", ind.fitness);
}
```

Let's look at the method in parts:

1\. Structure and inputs.

First we define the inputs for the algorithm and the **Individual** structure, which represents a single solution in the population. Each individual has genes (strategy parameters), step sizes for mutation, and a fitness value.

```
input int PopulationSize = 50;   // Population size
input int Generations = 100;     // Number of generations
input double InitialStepSize = 1.0; // Initial step size

struct Individual
{
  double genes[3];  // Strategy parameters
  double stepSizes[3];  // Step sizes for each parameter
  double fitness;   // Fitness
};
```

2\. Initialization.

In the **OnInit()** function, create the population and initialize it. The **InitializePopulation()** function fills the population with random gene values and sets the initial step sizes.

```
int OnInit ()
{
  ArrayResize (population, PopulationSize);
  InitializePopulation ();
  return (INIT_SUCCEEDED);
}

void InitializePopulation ()
{
  for (int i = 0; i < PopulationSize; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      population [i].genes [j] = MathRand () / 32767.0 * 100;
      population [i].stepSizes [j] = InitialStepSize;
    }
  }
}
```

3\. Main loop.

The optimization process is managed in the **OnTick ()** function. It evaluates the population, sorts it, performs the mutation, and moves on to the next generation.

```
datetime lastOptimizationTime = 0;

void OnTick ()
{
  datetime currentTime = TimeCurrent ();

  // Check if a day has passed since the last optimization
  if (currentTime - lastOptimizationTime >= 86400) // 86400 seconds in a day
  {
    Optimize ();
    lastOptimizationTime = currentTime;
  }

  // Here is the code for trading with the current optimal parameters
  TradingLogic ();
}

void Optimize ()
{
  // Optimization code (current OnTick content)
}

void TradingLogic ()
{
  // Implementing trading logic using optimized parameters
}
```

4\. Population assessment and sorting.

These functions evaluate the fitness of each individual and sort the population in descending order of fitness. In this example, the **CalculateFitness()** function is simple, but in real use it should have an objective function to evaluate the trading strategy.

```
void EvaluatePopulation ()
{
  for (int i = 0; i < PopulationSize; i++)
  {
    population [i].fitness = CalculateFitness (population [i]);
  }
}

double CalculateFitness (Individual &ind)
{
  return -(MathPow (ind.genes [0] - 50, 2) + MathPow (ind.genes [1] - 50, 2) + MathPow (ind.genes [2] - 50, 2));
}

void SortPopulation ()
{
  ArraySort (population, WHOLE_ARRAY, 0, MODE_DESCEND);
}
```

5\. Mutation.

This is the key part that implements Schwefel's concept. For every individual (except the best) we:

- Mutate the step size by multiplying it by the exponent of a random number.
- Mutate a gene by adding to it a normally distributed random number multiplied by the step size.
- Limit the values of genes in the range **\[0, 100\]**.

Basic implementation of Schwefel's concept for parameter optimization. In real-world applications, it is necessary to adapt the target function to a specific trading strategy.

```
void Mutate ()
{
  for (int i = 1; i < PopulationSize; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      population [i].stepSizes [j] *= MathExp (0.2 * MathRandom () - 0.1);
      population [i].genes [j] += population [i].stepSizes [j] * NormalRandom ();
      population [i].genes [j] = MathMax (0, MathMin (100, population [i].genes [j]));
    }
  }
}
```

We should also note the implementation of the **NormalRandom()** function, which is part of Schwefel's concept for adaptive mutation and implements the Box-Muller method for generating random numbers with a normal (Gaussian) distribution. Let's break this function down:

1\. Generating uniformly distributed numbers. Generate two independent random numbers **u1** and **u2**, uniformly distributed in the interval **\[0, 1\]**.

```
double u1 = u.RNDfromCI(0, 1);
double u2 = u.RNDfromCI(0, 1);
```

2\. Transformation to normal distribution. The Box-Muller transformation equation transforms uniformly distributed numbers into normally distributed ones.

```
return MathSqrt(-2 * MathLog(u1)) * MathCos(2 * M_PI * u2);
```

It is important to note that this is a half of the implementation of the Box-Muller transform, which generates a single normally distributed number. The full transformation generates two normally distributed numbers:

```
z0 = MathSqrt(-2 * MathLog(u1)) * MathCos(2 * M_PI * u2);
z1 = MathSqrt(-2 * MathLog(u1)) * MathSin(2 * M_PI * u2);
```

Our implementation uses only cosine, which gives a single normally distributed number. This is perfectly acceptable if you only need one number per call. If both numbers are needed, a sine calculation can be added.

This implementation is efficient and widely used for generating normally distributed random numbers in various applications including evolutionary algorithms and stochastic optimization.

Characteristics of generated numbers:

1\. Distribution: Normal (Gaussian) distribution

2\. Average value: **0**

3\. Standard deviation: **1**

Range of generated numbers: Theoretically, the normal distribution can generate numbers ranging from minus infinity to plus infinity. In practice:

- About **68%** of the generated numbers will be in the range **\[-1, 1\]**.
- About **95%** of numbers will be in the range **\[-2, 2\]**.
- About **99.7%** of numbers will be in the range **\[-3, 3\]**.

On some very rare occasions, there may be numbers outside **\[-4, 4\]**.

The Box-Muller method is used to generate random numbers with normal distribution, which is important for the implementation of self-adaptive mutation in the algorithm based on the Schwefel concept. This distribution allows for smaller changes to be generated more frequently, but sometimes allows for larger mutations, which facilitates efficient exploration of the solution space. Let's test and evaluate the Box-Muller method in practice.

Let's implement a script to test the **NormalRandom()** function:

```
#property version   "1.00"
#property script_show_inputs

input int NumSamples = 10000; // Amount of generated numbers

double NormalRandom ()
{
  double u1 = (double)MathRand () / 32767.0;
  double u2 = (double)MathRand () / 32767.0;
  return MathSqrt (-2 * MathLog (u1)) * MathCos (2 * M_PI * u2);
}

void OnStart ()
{
  double sum = 0;
  double sumSquared = 0;
  double min = DBL_MAX;
  double max = DBL_MIN;

  int histogram [];
  ArrayResize (histogram, 20);
  ArrayInitialize (histogram, 0);

  // Random number generation and analysis
  for (int i = 0; i < NumSamples; i++)
  {
    double value = NormalRandom ();
    sum += value;
    sumSquared += value * value;

    if (value < min) min = value;
    if (value > max) max = value;

    // Filling the histogram
    int index = (int)((value + 4) / 0.4); // Split the range [-4, 4] into 20 intervals
    if (index >= 0 && index < 20) histogram [index]++;
  }

  // Calculate statistics
  double mean = sum / NumSamples;
  double variance = (sumSquared - sum * sum / NumSamples) / (NumSamples - 1);
  double stdDev = MathSqrt (variance);

  // Display results
  Print ("Statistics for ", NumSamples, " generated numbers:");
  Print ("Average value: ", mean);
  Print ("Standard deviation: ", stdDev);
  Print ("Minimum value: ", min);
  Print ("Maximum value: ", max);

  // Display the histogram
  Print ("Distribution histogram:");
  for (int i = 0; i < 20; i++)
  {
    string bar = "";
    for (int j = 0; j < histogram [i] * 50 / NumSamples; j++) bar += "*";
    PrintFormat ("[%.1f, %.1f): %s", -4 + i * 0.4, -3.6 + i * 0.4, bar);\
  }\
}\
```\
\
The test script does the following:\
\
1\. Define the **NormalRandom()** function.\
\
2\. Generate a specified number of random numbers (default is 10,000).\
\
3\. Calculate basic statistical characteristics: mean, standard deviation, minimum and maximum values.\
\
4\. Create a distribution histogram by splitting the range **\[-4, 4\]** for 20 intervals.\
\
5\. Display results to the MetaTrader terminal log.\
\
Now let's test the script. We will take 20,000 values. Printout of the script running to test the Box-Muller transform method:\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    Statistics for 20,000 generated numbers:\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    Average value: -0.003037802901958332\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    Standard deviation: 0.9977477093538349\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    Minimum value: -3.865371560675546\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    Maximum value: 3.4797509297243994\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    Distribution histogram:\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-4.0, -3.6):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-3.6, -3.2):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-3.2, -2.8):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-2.8, -2.4):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-2.4, -2.0):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-2.0, -1.6): \*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-1.6, -1.2): \*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-1.2, -0.8): \*\*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-0.8, -0.4): \*\*\*\*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[-0.4, 0.0): \*\*\*\*\*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[0.0, 0.4): \*\*\*\*\*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[0.4, 0.8): \*\*\*\*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[0.8, 1.2): \*\*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[1.2, 1.6): \*\*\*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[1.6, 2.0): \*\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[2.0, 2.4):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[2.4, 2.8):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[2.8, 3.2):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[3.2, 3.6):\
\
2024.07.12 13:11:05.437    checkRND (.US500Cash,M5)    \[3.6, 4.0):\
\
From the printout it is clear that the method works correctly, the standard deviation is almost equal to **1**, the average value is **0**, while the spread corresponds to the interval **\[-4;4\]**.\
\
Next we move on to calculating the adaptive parameters of the mutation and writing the function:\
\
```\
//——————————————————————————————————————————————————————————————————————————————\
void AdaptiveMutation (double &Cg, double &Cs, double &Cn)\
{\
  Cg *= MathExp (tau_prime * NormalRandom() + tau * NormalRandom());\
  Cs *= MathExp (tau_prime * NormalRandom() + tau * NormalRandom());\
  Cn *= MathExp (tau_prime * NormalRandom() + tau * NormalRandom());\
}\
//——————————————————————————————————————————————————————————————————————————————\
```\
\
The **Cg**, **Cs** and **Cn** adaptive parameters are calculated using a self-adaptive mutation strategy based on the concept proposed by Schwefel. Equations for calculating these parameters:\
\
1\. Initialization:\
\
- The population of **N** solutions where each solution is represented as a pair of vectors **(pi, σi)**, where **i ∈ {0, 1, 2}** corresponds to three parameters **Cg**, **Cs** and **Cn**.\
- The initial values of the **pi** components are selected randomly according to a uniform distribution in the assumed solution space.\
- The **σi** initial values are set to a fixed value.\
\
2\. Generation of descendants:\
\
- For every parent **(pi, σi)**, a descendant is generated **(pi', σi')** according to the following equations:\
\
> **σ'i (j) = σi (j) \* exp (τ' \* N (0,1) + τ \* Nj (0,1))**\
>\
> **p'i (j) = pi (j) + σi (j) \* N (0,1)**\
\
where **pi (j)**, **p'i (j)**, **σi (j)**, **σ'i (j)** \- **j** th components of **pi**, **p'i**, **σi** and **σ'i** vectors respectively.\
\
- **N (0,1)** \- a random number taken from a normal distribution with mean **0** and standard deviation **1**.\
- **Nj (0,1)** \- a random number taken from a normal distribution with mean **0** and standard deviation **1**, where **j** is a counter.\
- **τ** and **τ'** are scaling factors set in **(√(2√n))^-1** and **(√(2n))^-1** accordingly, where **n** is the dimension of the problem.\
\
Thus, the **Cg**, **Cs** and **Cn** adaptive parameters mutate according to this self-adaptive strategy, allowing them to dynamically adjust during the optimization process.\
\
As we can see below from the printout of the obtained values of the **Cg**, **Cs** and **Cn** ratios, in some individual cases these ratios become too big. This happens because in the equation for updating strategic parameters **σ** the new value is obtained by multiplying it by the previous one. This allows the parameters to be adapted to the complex surface of the target function, but at the same time can lead to instability and sharp jumps in values.\
\
Let's see what values **Cg**, **Cs** and **Cn** take:\
\
1.3300705071425474 0.0019262948596068497 0.00015329292900983978\
\
1.9508542383574192 0.00014860860614036015 7007.656113084095\
\
52.13323602205895 1167.5425054524449 0.0008421503214593158\
\
1.0183156975753507 0.13486291437434697 0.18290674582014257\
\
0.00003239513683361894 61.366694225534175 45.11710564761292\
\
0.0004785111900713668 0.4798661298436033 0.007932035893070274\
\
2712198854.6325 0.00003936758705232012 325.9282730206205\
\
0.0016658142882911 22123.863582276706 1.6844067196638965\
\
2.0422888701555126 0.007999762224016285 0.02460292446531715\
\
7192.66545492709 0.000007671729921045711 0.3705939923291289\
\
0.0073279981653727785 3237957.2544339723 1.6750241266497745e-07\
\
550.7433921368721 13.512466529311943 223762.44571145615\
\
34.571961515974785 0.000008292503593679501 0.008122937723317175\
\
0.000002128739177639208 63.17654973794633 128927.83801094144\
\
866.72934816608881260.0820389718326 1.8496629497788273\
\
0.000008459817609364248 25.623751292511788 0.0013478840638847347\
\
27.956286711833616 0.0006967869388129299 0.0006885039945210606\
\
66.6928872126239 47449.76869262452 8934.079392419668\
\
0.15058617433681198 0.003114981958516487 7.703748428996011e-07\
\
0.22147618633450808 265.4903003920267315.20318731505455\
\
0.0000015873778483580056 1134.6304274682934 0.7883024873065534\
\
When **Cg**, **Cs** and **Cn** take on very large values as a result of self-adaptive mutation according to the Schwefel method, it may be necessary to take measures to control and limit these values. This is important to maintain the algorithm stability and efficiency. There are several possible approaches that can be used to limit the numerical values of the ratios:\
\
1\. Limiting values.\
\
Set upper and lower limits for **Cg**, **Cs** and **Cn**. For example:\
\
```\
void LimitParameters (double &param, double minValue, double maxValue)\
{\
  param = MathMax (minValue, MathMin (param, maxValue));\
}\
\
// Usage:\
LimitParameters (Cg, 0.0, 2.0);\
LimitParameters (Cs, 0.0, 2.0);\
LimitParameters (Cn, 0.0, 2.0);\
```\
\
2\. Normalization.\
\
Normalize the parameter values after mutation so that their sum is always equal to **1**:\
\
```\
void NormalizeParameters (double &Cg, double &Cs, double &Cn)\
{\
  double sum = Cg + Cs + Cn;\
  if (sum > 0)\
  {\
    Cg /= sum;\
    Cs /= sum;\
    Cn /= sum;\
  }\
  else\
  {\
    // If sum is 0, set equal values\
    Cg = Cs = Cn = 1.0 / 3.0;\
  }\
}\
```\
\
3\. Logarithmic scaling.\
\
Apply logarithmic scaling to smooth out large values:\
\
```\
double ScaleParameter (double param)\
{\
  if (param == 0) return 0;\
\
  double sign = (param > 0) ? 1 : -1;\
  return sign * MathLog (1 + MathAbs (param));\
}\
```\
\
4\. Adaptive scaling of mutation step.\
\
Reduce the mutation step size if the parameters become too large:\
\
```\
 void AdaptMutationStep(double &stepSize, double paramValue)\
{\
  if(MathAbs(paramValue) > threshold)\
  {\
    stepSize *= 0.9; // Reduce the step size\
  }\
}\
```\
\
5\. Periodic reset.\
\
Periodically reset the parameters to their initial values or to the population mean:\
\
```\
void ResetParameters(int generationCount)\
{\
  if(generationCount % resetInterval == 0)\
  {\
    Cg = initialCg;\
    Cs = initialCs;\
    Cn = initialCn;\
  }\
}\
```\
\
6\. Using the exponential function.\
\
Apply an exponential function to limit the growth of parameters:\
\
```\
double LimitGrowth(double param)\
{\
  return 2.0 / (1.0 + MathExp(-param)) - 1.0; // Limit values in [-1, 1]\
}\
```\
\
7\. Monitoring and adaptation.\
\
Monitor parameter values and adapt the mutation strategy if they frequently exceed acceptable limits:\
\
```\
void MonitorAndAdapt(double &param, int &outOfBoundsCount)\
{\
  if(MathAbs(param) > maxAllowedValue)\
  {\
    outOfBoundsCount++;\
    if(outOfBoundsCount > threshold)\
    {\
      // Adaptation of the mutation strategy\
      AdjustMutationStrategy();\
    }\
  }\
}\
```\
\
It is also important to remember that overly constraining the parameters can reduce the algorithm ability to explore the solution space, so it is necessary to find the right balance between control and flexibility. These methods can be applied for further research by optimization enthusiasts, but I used my own **GaussDistribution** function described in the [article](https://www.mql5.com/en/articles/13893).\
\
```\
//——————————————————————————————————————————————————————————————————————————————\
void C_AO_ASBO::AdaptiveMutation (S_ASBO_Agent &ag)\
{\
  ag.Cg *= MathExp (tau_prime * u.GaussDistribution (0, -1, 1, 1) + tau * u.GaussDistribution (0, -1, 1, 8));\
  ag.Cs *= MathExp (tau_prime * u.GaussDistribution (0, -1, 1, 1) + tau * u.GaussDistribution (0, -1, 1, 8));\
  ag.Cn *= MathExp (tau_prime * u.GaussDistribution (0, -1, 1, 1) + tau * u.GaussDistribution (0, -1, 1, 8));\
}\
//——————————————————————————————————————————————————————————————————————————————\
```\
\
As we can see below from the printout of the obtained values of the **Cg**, **Cs** and **Cn** ratios, after applying my function, large values occur much less frequently than when applying Box-Muller method.\
\
0.025582051880112085 0.6207719272290446 0.005335225840354781\
\
0.9810075068811726 0.16583946164135704 0.01016969794039794\
\
0.006133031813953609 17.700790930206647 0.3745475117676483\
\
1.4547663270710334 0.3537259667123157 0.08834618264409702\
\
0.11125695415944291 0.28183794982955684 0.09051405673590024\
\
0.06340035225180855 0.16270375413207716 0.36885953030567936\
\
0.008575136469231127 2.5881627332149053 0.11237602809318312\
\
0.00001436227841400286 0.02323530434501054 10.360403964016376\
\
0.936476760121053 0.017321731852758693 0.40372788912091845\
\
0.009288586536835293 0.0000072639468670123115 15.463139841665908\
\
0.15092489031689466 0.02160456749606 0.011008504295160867\
\
0.0100807047494077 0.4592091893288436 0.0343285901385665\
\
0.010014652012224212 0.0014577046664934706 0.006484475820059919\
\
0.0002654495048564554 0.0005018788250576451 1.8639207859646574\
\
5.972802450172414 0.10070170017416721 0.9226557559293936\
\
0.011441827382547332 14.599641192191408 0.00007257778906744059\
\
0.7249805357484554 0.000004301248511125035 0.2718776654314797\
\
5.019113547774523 0.11351424171113386 0.02129848352762841\
\
0.023978285994614518 0.041738711812672386 1.0247944259605422\
\
0.0036842456260203237 12.869472963773408 1.5167205157941646\
\
0.4529181577133935 0.0000625576761842319 30.751931508050227\
\
0.5555092369559338 0.9606330180338433 0.39426099685543164\
\
0.036106836251057275 2.57811344513881 0.042016638784243526\
\
3.502119772985753 128.0263928713568 0.9925745499516576\
\
279.2236061102195 0.6837013166327449 0.01615639677602729\
\
0.09687457825904996 0.3812813151133578 0.5272720937749686\
\
Now that we have understood the Schwefel concept as well as the adaptive values of the ratios, let's move on to the method of determining the nearest neighbors in the algorithm. The following method is used to determine the coordinates of the nearest neighbors **Nc**:\
\
1\. For each individual in the population, its nearest neighbors are determined.\
\
2\. The nearest neighbors are considered to be three individuals with the closest values of the objective function (fitness) to a given individual.\
\
3\. The coordinates of the center of the group formed by a given individual and its nearest neighbors are calculated as the arithmetic mean of the coordinates of these three neighbors.\
\
Thus, the **Nc** coordinates are not simply taken as three arbitrary coordinates, but are calculated as the center of the group of nearest neighbors by the value of the objective function. This allows information about an individual's immediate environment to be used to determine its next position.\
\
The key point is that the nearest neighbors are determined not by geographic proximity, but by the proximity of the values of the objective function. This corresponds to a logical, rather than geographical, proximity in the social structure.\
\
```\
//——————————————————————————————————————————————————————————————————————————————\
void C_AO_ASBO::FindNeighborCenter (const S_ASBO_Agent &ag, double &center [])\
{\
// Create arrays for indices and fitness differences\
  int indices [];\
  double differences [];\
  ArrayResize (indices, popSize - 1);\
  ArrayResize (differences, popSize - 1);\
\
// Fill the arrays\
  int count = 0;\
  for (int i = 0; i < popSize; i++)\
  {\
    if (&a [i] != &ag)  // Exclude the current agent\
    {\
      indices [count] = i;\
      differences [count] = MathAbs (a [i].fitness - ag.fitness);\
      count++;\
    }\
  }\
\
// Sort arrays by fitness difference (bubble sort)\
  for (int i = 0; i < count - 1; i++)\
  {\
    for (int j = 0; j < count - i - 1; j++)\
    {\
      if (differences [j] > differences [j + 1])\
      {\
        // Exchange of differences\
        double tempDiff = differences [j];\
        differences [j] = differences [j + 1];\
        differences [j + 1] = tempDiff;\
\
        // Exchange of indices\
        int tempIndex = indices [j];\
        indices [j] = indices [j + 1];\
        indices [j + 1] = tempIndex;\
      }\
    }\
  }\
\
// Initialize the center\
  ArrayInitialize (center, 0.0);\
\
// Calculate the center based on the three nearest neighbors\
  for (int j = 0; j < coords; j++)\
  {\
    for (int k = 0; k < 3; k++)\
    {\
      int nearestIndex = indices [k];\
      center [j] += a [nearestIndex].c [j];\
    }\
    center [j] /= 3;\
  }\
}\
//——————————————————————————————————————————————————————————————————————————————\
```\
\
Explanation of the method:\
\
- Determines the nearest neighbors based on the proximity of the objective function (fitness) values.\
- Uses three nearest neighbors.\
- Calculates the group center as the arithmetic mean of the coordinates of these three neighbors.\
\
Let's analyze this function:\
\
1\. Sorting direction:\
\
Sorting is done in ascending order of fitness difference.\
\
If the current element is greater than the next one, they swap places. Thus, after sorting, the **differences** array will be sorted from smallest to largest values, and the corresponding indices in the **indices** array will reflect this sorting.\
\
2\. The function performs the following steps:\
\
- Excludes the current agent from consideration.\
- Calculates the fitness difference between the current agent and all others.\
- Sorts agents by this difference.\
- Selects the three nearest neighbors (with the smallest fitness difference)\
- Calculates the center of a group based on the coordinates of these three neighbors\
\
```\
void C_AO_ASBO::FindNeighborCenter(int ind, S_ASBO_Agent &ag[], double &center[])\
{\
  int indices[];\
  double differences[];\
  ArrayResize(indices, popSize - 1);\
  ArrayResize(differences, popSize - 1);\
\
  int count = 0;\
  for (int i = 0; i < popSize; i++)\
  {\
    if (i != ind)\
    {\
      indices[count] = i;\
      differences[count] = MathAbs(ag[ind].f - ag[i].f);\
      count++;\
    }\
  }\
\
// Sort by fitness difference ascending\
  for (int i = 0; i < count - 1; i++)\
  {\
    for (int j = 0; j < count - i - 1; j++)\
    {\
      if (differences[j] > differences[j + 1])\
      {\
        double tempDiff = differences[j];\
        differences[j] = differences[j + 1];\
        differences[j + 1] = tempDiff;\
\
        int tempIndex = indices[j];\
        indices[j] = indices[j + 1];\
        indices[j + 1] = tempIndex;\
      }\
    }\
  }\
\
  ArrayInitialize(center, 0.0);\
\
  int neighborsCount = MathMin(3, count);  // Protection against the case when there are less than 3 agents\
  for (int j = 0; j < coords; j++)\
  {\
    for (int k = 0; k < neighborsCount; k++)\
    {\
      int nearestIndex = indices[k];\
      center[j] += ag[nearestIndex].c[j];\
    }\
    center[j] /= neighborsCount;\
  }\
}\
```\
\
This version of the function is more robust to errors and correctly handles cases with a small number (less than 3) of agents in the population. We have become familiar with the basic logical methods in the algorithm, now we can move on to examining the structure of the algorithm itself. ASBO algorithm consists of the following main steps:\
\
1\. Initialization:\
\
- An initial population of solutions is defined, where each solution is represented explicitly (not in encoded format).\
- For each solution, the value of the objective function is calculated, which determines its suitability.\
- The solution with the highest fitness value is declared the global leader at that point in time.\
- For each solution, a group of nearest neighbors with the next highest fitness values is determined.\
\
2\. Adaptive mutation of parameters:\
\
- For each solution, a vector of three adaptive parameters is determined: **Cg**, **Cs** and **Cn**, responsible for the influence of the global leader, the personal best solution and the center of the group of neighbors, respectively.\
- A self-adaptive Schwefel mutation strategy is used to update the values of these parameters.\
\
3\. Updating the decision status:\
\
- For each solution, the change in its position is calculated using the current values of the **Cg**, **Cs** and **Cn** parameters.\
- The new solution position is calculated by adding the change to the current position.\
\
4\. Two-phase process:\
\
- Phase 1: M independent populations are created, each of which is handled by the ASBO algorithm for a fixed number of iterations. The fitness and parameter values for each solution from the final populations are stored.\
- Phase 2: The best fitness solutions are selected from all final populations and their parameters are used to form a new population. The basic logic of the ASBO algorithm is applied to the resulting new population to obtain the final solution.\
\
So, let's highlight the key features of the ASBO algorithm:\
\
- Applying self-adaptive mutation strategy to dynamically update parameters.\
- Using the concepts of leadership and neighbor influence to model social behavior.\
- A two-phase process to maintain solution diversity and speed up convergence.\
\
In this article, we have considered an example of Schwefel's concept, the Box-Muller method, which includes a normal distribution, the use of self-adaptive mutation rates, and a function for determining the nearest neighbors by their fitness value. We have touched upon the structure of ASBO. In the next article, we will examine in detail the two-phase process of artificial evolution, complete the formation of the algorithm, conduct testing on test functions and draw conclusions about its efficiency.\
\
Translated from Russian by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/ru/articles/15283](https://www.mql5.com/ru/articles/15283)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/15283.zip "Download all attachments in the single ZIP archive")\
\
[Box\_Muller\_transform.mq5](https://www.mql5.com/en/articles/download/15283/box_muller_transform.mq5 "Download Box_Muller_transform.mq5")(3.27 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Central Force Optimization (CFO) algorithm](https://www.mql5.com/en/articles/17167)\
- [Neuroboids Optimization Algorithm (NOA)](https://www.mql5.com/en/articles/16992)\
- [Successful Restaurateur Algorithm (SRA)](https://www.mql5.com/en/articles/17380)\
- [Billiards Optimization Algorithm (BOA)](https://www.mql5.com/en/articles/17325)\
- [Chaos Game Optimization (CGO)](https://www.mql5.com/en/articles/17047)\
- [Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)\
- [Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)\
\
**[Go to discussion](https://www.mql5.com/en/forum/479600)**\
\
![Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://c.mql5.com/2/82/Neural_networks_are_simple_Piecewise_linear_representation_of_time_series__LOGO.png)[Neural Networks in Trading: Piecewise Linear Representation of Time Series](https://www.mql5.com/en/articles/15217)\
\
This article is somewhat different from my earlier publications. In this article, we will talk about an alternative representation of time series. Piecewise linear representation of time series is a method of approximating a time series using linear functions over small intervals.\
\
![Developing a Replay System (Part 55): Control Module](https://c.mql5.com/2/83/Desenvolvendo_um_sistema_de_Replay_Parte_55__LOGO.png)[Developing a Replay System (Part 55): Control Module](https://www.mql5.com/en/articles/11988)\
\
In this article, we will implement a control indicator so that it can be integrated into the message system we are developing. Although it is not very difficult, there are some details that need to be understood about the initialization of this module. The material presented here is for educational purposes only. In no way should it be considered as an application for any purpose other than learning and mastering the concepts shown.\
\
![MetaTrader 5 on macOS](https://c.mql5.com/2/0/1045_13.png)[MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)\
\
We provide a special installer for the MetaTrader 5 trading platform on macOS. It is a full-fledged wizard that allows you to install the application natively. The installer performs all the required steps: it identifies your system, downloads and installs the latest Wine version, configures it, and then installs MetaTrader within it. All steps are completed in the automated mode, and you can start using the platform immediately after installation.\
\
![Developing A Swing Entries Monitoring (EA)](https://c.mql5.com/2/109/Developing_A_Swing_Entries_Monitoring___LOGO.png)[Developing A Swing Entries Monitoring (EA)](https://www.mql5.com/en/articles/16563)\
\
As the year approaches its end, long-term traders often reflect on market history to analyze its behavior and trends, aiming to project potential future movements. In this article, we will explore the development of a long-term entry monitoring Expert Advisor (EA) using MQL5. The objective is to address the challenge of missed long-term trading opportunities caused by manual trading and the absence of automated monitoring systems. We'll use one of the most prominently traded pairs as an example to strategize and develop our solution effectively.\
\
[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ecdilgaxswjvknrzouvisawcytonvrlp&ssn=1769191689982076686&ssn_dr=0&ssn_sr=0&fv_date=1769191689&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15283&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Adaptive%20Social%20Behavior%20Optimization%20(ASBO)%3A%20Schwefel%2C%20Box-Muller%20Method%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919168908112916&fz_uniq=5071601502063110920&sv=2552)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).