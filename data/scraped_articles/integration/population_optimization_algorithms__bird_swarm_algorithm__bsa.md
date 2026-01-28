---
title: Population optimization algorithms: Bird Swarm Algorithm (BSA)
url: https://www.mql5.com/en/articles/14491
categories: Integration, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T17:20:18.806906
---

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/14491&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068116660088403361)

MetaTrader 5 / Examples


### Contents

1. [Introduction](https://www.mql5.com/en/articles/14491#tag1)

2. [Algorithm description](https://www.mql5.com/en/articles/14491#tag2)

3. [Test results](https://www.mql5.com/en/articles/14491#tag3)

### 1\. Introduction

Birds are amazing creatures that occupy an important place in nature and the ecosystem. Birds are believed to have evolved from dinosaurs, their closest relatives. One of the most famous examples is Archaeopteryx, the oldest known bird, which lived about 150 million years ago. They often act as indicators of environmental health because changes in numbers and behavior can indicate problems in the ecosystem, such as pollution, habitat loss, and climate change. There are more than 10,000 known species of birds on Earth, and each of them has a unique adaptation to its way of life.

Some birds are capable of flying great distances, some can dive to depths, and others have amazing vocal abilities. Birds play an important role in the ecosystem, they disperse plant seeds, control the population of insects and other animals, and are a food source for predators. Many bird species undertake long migrations, living together and interacting with other members of their species in a flock, traveling thousands of kilometers together in search of food or a place to breed. This phenomenon highlights outstanding navigation skills, endurance, interaction and cooperation in a group. Birds are an incredibly diverse and important part of our planet.

Bird Swarm Algorithm (BSA) is an exciting bioinspired evolutionary algorithm using swarm intelligence based on social interactions and behavior of bird flocks. Developed by Meng and colleagues in 2015, BSA is a unique optimization approach that combines three key aspects of bird behavior: **flight**, **foraging** and **vigilance**. Among the electronic flocks, where each "bird" has individual tactics and strategies, a unique system of collective interaction is born, filled with algorithmic intelligence and creativity. What is important here is not only personal effort, but also the ability to cooperate, exchange and support each other in pursuit of the common goal of optimization.

Different individuals in the BSA may have different search strategies. Birds can randomly switch between flight, vigilance and foraging behavior. The bionic design algorithm includes foraging based on global and individual fitness. Birds also try to move to the center of the population (which can lead to competition with other birds) or to move away from the flock. Bird behavior includes regular flight and migration, as well as switching between the roles of producer and beggar. In the BSA world, each individual at a given iteration has its own search strategy, making the algorithm multifaceted and capable of exerting its power.

However, like many swarm intelligence algorithms, BSA can suffer from premature convergence and get stuck in local optima. To achieve faster convergence with high accuracy from swarm-based optimization algorithms, various methods have been used to balance exploitation and exploration.

The BSA algorithm, based on bird behavior, is inspired by the collective flock interactions of birds in nature, whose behavior forms the basis of this algorithm:

- **Pack behavior.** Many species of birds, such as starlings, swallows and geese, exhibit flocking behavior when they fly together. This behavior helps them reduce air resistance and save energy during migrations or foraging.

- **Communication.** Birds use different types of communication such as sounds, gestures and postures to communicate with each other. This allows them to coordinate their actions, warn their relatives of danger, and coordinate the search for food.

- **Adaptability.** Birds have a high degree of adaptability to changing environmental conditions. They can respond quickly to danger, changes in weather and food availability, and adapt their behavior and migration routes depending on the circumstances.

- **Leading and following**. In a flock of birds there is usually a leader who determines the direction of flight, and the other birds follow him. This demonstrates the principle of leading and following, which is also taken into account in the BSA algorithm to effectively find optimal solutions.

The BSA algorithm uses these principles of bird behavior to develop an efficient optimization technique that simulates the collective behavior of bird flocks to solve various optimization problems. BSA is more than just an algorithm, it is a fascinating journey into the world of optimization, where social interactions of birds become a source of inspiration for efficient solution of complex problems.

### 2\. Algorithm

Let's move on to a more detailed look at the logic of the BSA algorithm, which may seem complex and confusing at first glance. Before we start implementing the code, let's develop a pseudocode of the algorithm, which will serve as the basis for its implementation. This will make it much easier to understand how BSA works.

Bird Swarm Algorithm (BSA) pseudocode, which is a high-level description of the algorithm that models the bird flock behavior:

1\. Initialization of N solutions and associated parameters

2\. Generating new solutions:

3\.  If the bird is flying:

    4\. If the bird is a producer:

        5\. Searching for a new food source

    6\. Otherwise:

        7\. The beggar bird follows the producer

8\. Otherwise:

    9\. If the bird is getting food:

        10\. The bird is feeding

    11\. Otherwise:

        12\.  The bird remains vigilant

13\. Evaluating new solutions

14\. Updating solutions

15\. If the stop criterion is reached:

    16\. Algorithm completion

17\. Otherwise:

    18\. Return to step 2

The equation for point 5, for the bird looking for a new food site:

> **xn = RNDg (min, max, producerPower)**

where:

- xn - new coordinate value
- RNDg - random number with normal distribution with the center of distribution at the current coordinate
- min and max - distribution boundaries
- producerPower - standard deviation for the producer

According to the equation, the breeding bird can migrate in any direction throughout the search space, with an increased probability in the vicinity of its current position. This allows birds to explore new areas in search of food.

The equation for point 7 for the beggar bird following the producer:

> **xn = x + (xK - x) \* FL \* RNDg (-1.0, 1.0, scroungerPower)**

where:

- xn -  new coordinate value
- x - the best beggar coordinate in history
- xK - the best coordinate of the producer in history, where a random bird with position K in the population is chosen as the producer
- RNDg - random number with normal distribution with the center of distribution at 0 and the borders "-1.0" and "1.0"
- scroungerPower - standard deviation for the beggar

The equation shows that the beggar bird is guided by its best coordinates and the best coordinates of the best individual in the flock (the producer is guided not by its best coordinates, but by the current ones). This models the behavior of following the leader in a pack.

The equation of the point 10 applicable to birds during the feeding period outside of flight:

> **xn = x + (p - x) \* C \* r1 + (g - x) \* S \* r2**

where:

- xn - new coordinate value
- x - current coordinate
- p - the best coordinate of the bird taking food in history
- g - the best population coordinates in history (the best global solution)
- r1 - random uniform number in the range \[0.0 ... 1.0\]
- r2 - random uniform number in the range \[0.0 ... 1.0\]
- C - cognitive ratio, external parameter
- S - social ratio, external parameter

The equation describes the moment of food intake, in which the bird's behavior is based on its own experience (current position and best position in the past) and the experience of the flock.

The equation for point 12 for the vigilant bird:

> **xn = x + A1 \* (mean \[c\] - x) \* r1 + A2 \* (xK - x) \* r2**

where:

- xn - new coordinate value
- x - best vigilant bird coordinate in history
- r1 - random uniform number in the range \[0.0 ... 1.0\]
- r2 - random uniform number in the range \[-1.0 ... 1.0\]
- mean \[c\] - average value of the c th coordinate based on the best coordinates of all birds in the flock


A1 - correction ratio of the influence of the average coordinates of the flock center:

> **A1 = a1 \* exp (-pFit \* N / (sumFit + e))**

where:

- a1 - ratio, external parameter
- e = DBL\_MIN, to avoid division by 0
- pFit - best fitness of the vigilant bird
- sumFit - sum of the best fitnesses of birds in a flock
- N - number of birds in a flock

A2 - correction ratio that takes into account the influence of the position of the bird selected for observation (which has fallen into the field of view of the alert bird) on the behavior of the latter. Equation for A2:

> **A2 = a2 \* exp (((pFit - pFitK) / (\|pFitK - pFit\| + e)) \* (N \* pFitK / (sumFit + e)))**

where:

- a2 - ratio, external parameter
- e = DBL\_MIN, to avoid division by 0
- pFit - best fitness of the vigilant bird
- pFitK - the best fitness of a randomly selected K th bird in the population (the bird that has come into the field of view of the vigilant bird)
- sumFit - sum of the best fitnesses of birds in a flock
- N - number of birds in a flock

Thus, the vigilant bird monitors its surroundings, which allows it to warn its relatives of danger in a timely manner. This is the most complex behavior of all described in the algorithm, taking into account the fitness of all birds in the population, as well as the fitness of the vigilant bird itself and the one selected for observation. In essence, a vigilant bird will move in the direction of overall population fitness, given the position of its counterpart that comes into its field of vision.

The highlighted text in the pseudocode corresponds to the BSA logic elements shown in Figure 1.

![BSA](https://c.mql5.com/2/73/BSA__2.png)

Figure 1. BSA algorithm logical diagram

The diagram in Figure 1 is a visualization of the BSA algorithm and models the behavior of a bird flock. Algorithm key features:

1. Initializing solutions. The algorithm starts by initializing a set of solutions and their associated parameters. This involves the initial distribution of birds (or solutions) in the search space.
2. Flight behavior. During the algorithm's operation, each bird can "fly" or "not fly". This condition affects the bird's ability to discover new solutions.
3. Foraging behavior. If a bird "flies", it can become a "producer" and start searching for a new area with food, or it can become a "beggar", following the producer.
4. Foraging behavior. If a bird is "not flying," it is either feeding or remaining vigilant. This may represent a state of anticipation or observation of the environment.
5. Evaluation and updating solutions. After generating new solutions, their fitness or quality is assessed.
6. Stop criterion. The algorithm continues the cycle of generating and updating solutions until a certain stop criterion is reached. This could be a certain number of iterations, achieving a given level of solution quality, or another criterion.

I would like to emphasize that BSA is a dynamic algorithm that adapts and evolves in the process of searching for the optimal solution.

Let's implement the code for the BSA algorithm. For each agent, we define the S\_BSA\_Agent structure, which will be a separate solution to the optimization problem and a description of the bird in the flock.

The structure contains the following fields:

- cBest\[\] - array for storing the best agent coordinates.
- fBest - variable for storing the best agent fitness score.
- Init - S\_BSA\_Agent structure method that initializes the structure fields. It takes the "coords" integer argument used to resize the "cBest" array using the ArrayResize function.

We set the initial value of the "fBest" variable to the minimum possible double value, which means the worst possible fitness.

```
//——————————————————————————————————————————————————————————————————————————————
struct S_BSA_Agent
{
    double cBest []; //best coordinates
    double fBest;    //best fitness

    void Init (int coords)
    {
      ArrayResize (cBest, coords);
      fBest = -DBL_MAX;
    }
};
//——————————————————————————————————————————————————————————————————————————————
```

Let's define the C\_AO\_BSA class of the BSA algorithm, which is an inheritor of the base class of C\_AO population algorithms and contains the following fields and methods:

1\. Public fields:

- ao\_name - optimization algorithm name.
- ao\_desc - optimization algorithm description.
- popSize - population size.
- params - array of algorithm parameters.
- flyingProb - flight probability.
- producerProb - production probability.
- foragingProb - foraging probability.
- a1 - a1 constant \[0...2\].
- a2 - a2 constant \[0...2\].
- C - cognitive ratio.
- S - social ratio.
- FL - FL constant \[0...2\].
- producerPower - standard deviation in producer behavior.
- scroungerPower - standard deviation in beggar behavior.

2\. The options available are:

- C\_AO\_BSA - class constructor that initializes the class fields.
- SetParams - method for setting algorithm parameters.
- Init - method for initializing the algorithm. The method accepts minimum and maximum search ranges, search step and number of epochs.
- Moving - method for moving agents.
- Revision - method for revising agents.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_BSA : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_BSA () { }
  C_AO_BSA ()
  {
    ao_name = "BSA";
    ao_desc = "Bird Swarm Algorithm";

    popSize        = 20;  //population size

    flyingProb     = 0.8;  //Flight probability
    producerProb   = 0.25; //Producer probability
    foragingProb   = 0.55; //Foraging probability
    a1             = 0.6;  //a1 constant [0...2]
    a2             = 0.05; //a2 constant [0...2]
    C              = 0.05; //Cognitive coefficient
    S              = 1.1;  //Social coefficient
    FL             = 1.75; //FL constant [0...2]
    producerPower  = 7.05; //Producer power
    scroungerPower = 2.60; //Scrounger power

    ArrayResize (params, 11);

    params [0].name = "popSize";         params [0].val  = popSize;

    params [1].name  = "flyingProb";     params [1].val  = flyingProb;
    params [2].name  = "producerProb";   params [2].val  = producerProb;
    params [3].name  = "foragingProb";   params [3].val  = foragingProb;
    params [4].name  = "a1";             params [4].val  = a1;
    params [5].name  = "a2";             params [5].val  = a2;
    params [6].name  = "C";              params [6].val  = C;
    params [7].name  = "S";              params [7].val  = S;
    params [8].name  = "FL";             params [8].val  = FL;
    params [9].name  = "producerPower";  params [9].val  = producerPower;
    params [10].name = "scroungerPower"; params [10].val = scroungerPower;
  }

  void SetParams ()
  {
    popSize        = (int)params [0].val;

    flyingProb     = params [1].val;
    producerProb   = params [2].val;
    foragingProb   = params [3].val;
    a1             = params [4].val;
    a2             = params [5].val;
    C              = params [6].val;
    S              = params [7].val;
    FL             = params [8].val;
    producerPower  = params [9].val;
    scroungerPower = params [10].val;
  }

  bool Init (const double &rangeMinP  [], //minimum search range
             const double &rangeMaxP  [], //maximum search range
             const double &rangeStepP [], //step search
             const int     epochsP = 0);  //number of epochs

  void Moving   ();
  void Revision ();
  void Injection (const int popPos, const int coordPos, const double value);

  //----------------------------------------------------------------------------
  double flyingProb;      //Flight probability
  double producerProb;    //Producer probability
  double foragingProb;    //Foraging probability
  double a1;              //a1 constant [0...2]
  double a2;              //a2 constant [0...2]
  double C;               //Cognitive coefficient
  double S;               //Social coefficient
  double FL;              //FL constant [0...2]
  double producerPower;   //Producer power
  double scroungerPower;  //Scrounger power

  S_BSA_Agent agent [];

  private: //-------------------------------------------------------------------
  double mean [];  //represents the element of the average position of the whole bird’s swarm
  double N;
  double e;        //epsilon

  void BirdProducer  (int pos);
  void BirdScrounger (int pos);
  void BirdForaging  (int pos);
  void BirdVigilance (int pos);
};
//——————————————————————————————————————————————————————————————————————————————
```

The Init method of the C\_AO\_BSA class is used to initialize class variables based on the passed parameters. This method performs standard initialization using the StandardInit method, which takes the minimum and maximum search ranges as well as the search step.

If standard initialization is successful, the method continues initializing the "N" and "e" variables. The value of "N" is set to the "popSize" population size, while "e" is an epsilon initialized by the minimum double value.

The method then resizes the "agent" array to the size of "popSize". The Init method is called with the "coords" parameter for each element in "agent". The size of the "mean" array is also changed to the size of "coords". The array is used to store the average population coordinates of birds.

The method returns "true" if initialization was successful, and "false" otherwise.

This method performs the initial setup of the BSA optimization algorithm with given parameters and prepares it to perform optimization.

```
//——————————————————————————————————————————————————————————————————————————————
bool C_AO_BSA::Init (const double &rangeMinP  [], //minimum search range
                     const double &rangeMaxP  [], //maximum search range
                     const double &rangeStepP [], //step search
                     const int     epochsP = 0)  //number of epochs
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  ArrayResize (agent, popSize);
  for (int i = 0; i < popSize; i++) agent [i].Init (coords);

  ArrayResize (mean, coords);

  N = popSize;
  e = DBL_MIN;

  return true;
}
//——————————————————————————————————————————————————————————————————————————————
```

The Moving method of the C\_AO\_BSA class is used to move agents during optimization. The method does the following:

- If "revision" is 'false', the agents' coordinates "a\[i\].c\[c\]" are initialized using random values in the specified ranges. The "revision" flag is then set to "true" and the method exits.
- If "revision" is not 'false', then new coordinates are calculated for each agent using equations and probabilities.

In the second and subsequent epochs, the method calls functions that determine the behavior of each bird in the flock depending on the probabilities fulfilled:

- If the probability of flying is met - "flyingProb", then the agent "flies". In this case, two possible behavior options are possible:

> 1. If the probability is less than "producerProb", then the agent is a "producer" and is looking for a new place to eat.
>
> 2. Otherwise, the agent is a "beggar" and follows the producer.

- If it "does not fly", then the following two behavior options are possible:


> 1. If the probability is less than "foragingProb", then the agent "forages" food.
> 2. Otherwise, the agent is in a "vigilant" state.

The method is responsible for updating the coordinates of agents in the BSA optimization algorithm in accordance with the current epoch, random values and probabilities.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_BSA::Moving ()
{
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

  //----------------------------------------------------------------------------
  for (int i = 0; i < popSize; i++)
  {
    //bird is flying------------------------------------------------------------
    if (u.RNDprobab () < flyingProb)
    {
      //bird producer
      if (u.RNDprobab () < producerProb) BirdProducer  (i); //bird is looking for a new place to eat
      //bird is not a producer
      else                               BirdScrounger (i); //scrounger follows the  producer
    }
    //bird is not flying--------------------------------------------------------
    else
    {
      //bird foraging
      if (u.RNDprobab () < foragingProb) BirdForaging  (i); //bird feeds
      //bird is not foraging
      else                               BirdVigilance (i); //bird vigilance
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The BirdProducer method of the C\_AO\_BSA class is used to simulate "producer" behavior in the BSA algorithm. The method does the following:

- Initializes the "x" variable, which will be used to store the bird's current position.
- Then, the following actions are performed for each agent coordinate:

  - The "x" value is set to the current agent coordinate.
  - The "x" value is updated using a Gaussian distribution, where the mean is the current coordinate and the range and standard deviation are determined by the "rangeMin", "rangeMax" and "producerPower" values.
  - The new value of the agent coordinate is set using the SeInDiSp method, which adjusts the "x" value according to the search range and step.

This method models the behavior of a "producer" in the BSA algorithm, which searches for new food source locations (i.e. new potential solutions) using a Gaussian distribution to explore the search space.

```
//——————————————————————————————————————————————————————————————————————————————
void  C_AO_BSA::BirdProducer  (int pos)
{
  double x = 0.0; //bird position

  for (int c = 0; c < coords; c++)
  {
    x = a [pos].c [c];
    x = u.GaussDistribution (x, rangeMin [c], rangeMax [c], producerPower);

    a [pos].c [c] = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The method that models the "beggar" behavior is the BirdScrounger function in the C\_AO\_BSA class. It performs the following actions:

- 1\. Initializes the "K", "x" and "xK" variables. "K" is the position of a randomly selected bird in the flock, "x" is the best position of the bird, and "xK" is the current best position of the randomly selected bird in the flock.
- 2\. Runs a loop through all coordinates.

  - Selects a random bird that is not the current one.
  - Updates "x" and "xK" based on the best positions of the current bird and a randomly selected bird.
  - Updates "x" using the Gaussian distribution.
  - Finally, updates the current position of the bird using the SeInDiSp method, adjusting the "x" value according to the search range and step.

This method models the "beggar" behavior in the BSA algorithm using the Gaussian distribution that follows the "producer" (i.e. adjusts its own location relative to the producer's position).

```
//——————————————————————————————————————————————————————————————————————————————
void  C_AO_BSA::BirdScrounger (int pos)
{
  int    K  = 0;   //position of a randomly selected bird in a swarm
  double x  = 0.0; //best bird position
  double xK = 0.0; //current best position of a randomly selected bird in a swarm

  for (int c = 0; c < coords; c++)
  {
    do K = u.RNDminusOne (popSize);
    while (K == pos);

    x  = agent [pos].cBest [c];
    xK = agent   [K].cBest [c];

    x = x + (xK - x) * FL * u.GaussDistribution (0, -1.0, 1.0, scroungerPower);

    a [pos].c [c] = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The BirdForaging method in the C\_AO\_BSA class is meant for a bird that is not currently flying and is busy eating. The method does the following inside the loop for all coordinates:

- Updates "x", "p" and "g" based on the bird's current and best positions, as well as the best global position.
- Generates two random numbers "r1" and "r2".
- Updates "x" using these random numbers, as well as the "C" and "S" constants.
- Adjusts the obtained bird position using the SeInDiSp function.

```
//——————————————————————————————————————————————————————————————————————————————
void  C_AO_BSA::BirdForaging  (int pos)
{
  double x  = 0.0; //current bird position
  double p  = 0.0; //best bird position
  double g  = 0.0; //best global position
  double r1 = 0.0; //uniform random number [0.0 ... 1.0]
  double r2 = 0.0; //uniform random number [0.0 ... 1.0]

  for (int c = 0; c < coords; c++)
  {
    x = a     [pos].c     [c];
    p = agent [pos].cBest [c];
    g = cB                [c];

    r1 = u.RNDprobab ();
    r2 = u.RNDprobab ();

    x = x + (p - x) * C * r1 + (g - x) * S * r2;

    a [pos].c [c] = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The latest and most complex method of simulating the behavior of a vigilant bird is BirdVigilance. It performs the following actions:

- Calculates the sum of the best fitness values of all birds in the flock.
- Calculates the average value of each coordinate for all birds in the flock.
- Selects a random bird that is not the current one.
- Updates "pFit" and "pFitK" based on the best fitness values of the current bird and a randomly selected bird.
- Calculates "A1" and "A2" using an exponential function that depends on "pFit", "pFitK", "N", "sumFit" and "e".
- Runs a loop through all coordinates:

  - Generates two random numbers "r1" and "r2".
  - Updates "x" and "xK" based on the best positions of the current bird and a randomly selected bird.
  - Updates "x" using "A1", "A2", "r1" and "r2".
  - Adjusts the current bird position using the SeInDiSp function.

```
//——————————————————————————————————————————————————————————————————————————————
void  C_AO_BSA::BirdVigilance (int pos)
{
  int    K      = 0;   //position of a randomly selected bird in a swarm
  double sumFit = 0.0; //best birds fitness sum
  double pFitK  = 0.0; //best fitness of a randomly selected bird
  double pFit   = 0.0; //best bird fitness
  double A1     = 0.0;
  double A2     = 0.0;
  double r1     = 0.0; //uniform random number [ 0.0 ... 1.0]
  double r2     = 0.0; //uniform random number [-1.0 ... 1.0]
  double x      = 0.0; //best bird position
  double xK     = 0.0; //best position of a randomly selected bird in a swarm

  ArrayInitialize (mean, 0.0);

  for (int i = 0; i < popSize; i++) sumFit += agent [i].fBest;

  for (int c = 0; c < coords; c++)
  {
    for (int i = 0; i < popSize; i++) mean [c] += a [i].c [c];

    mean [c] /= popSize;
  }

  do K = u.RNDminusOne (popSize);
  while (K == pos);

  pFit  = agent [pos].fBest;
  pFitK = agent   [K].fBest;

  A1 = a1 * exp (-pFit * N / (sumFit + e));
  A2 = a2 * exp (((pFit - pFitK) / (fabs (pFitK - pFit) + e)) * (N * pFitK / (sumFit + e)));

  for (int c = 0; c < coords; c++)
  {
    r1 = u.RNDprobab ();
    r2 = u.RNDfromCI (-1, 1);

    x  = agent [pos].cBest [c];
    xK = agent   [K].cBest [c];

    x = x + A1 * (mean [c] - x) * r1 + A2 * (xK - x) * r2;

    a [pos].c [c] = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The Revision method of the C\_AO\_BSA class is used to update the best global solution and update the best positions of the agents. The method does the following:

- Updating the global solution.
- Update previous best fitness function values and agent coordinates.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_BSA::Revision ()
{
  //----------------------------------------------------------------------------
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

  //----------------------------------------------------------------------------
  for (int i = 0; i < popSize; i++)
  {
    if (a [i].f > agent [i].fBest)
    {
      agent [i].fBest = a [i].f;
      ArrayCopy (agent [i].cBest, a [i].c, 0, 0, WHOLE_ARRAY);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

### 3\. Test results

I would like to dwell in more detail on the results of the BSA algorithm on various sets of functions. The overall BSA score across all test functions was 4.80947, which corresponds to 53.44% of the maximum possible score. This result indicates the overall efficiency of the algorithm. The Bird Swarm Algorithm has the potential to successfully solve a variety of optimization problems on different functions.

BSA\|Bird Swarm Algorithm\|20.0\|0.8\|0.25\|0.55\|0.6\|0.05\|0.05\|1.1\|1.75\|7.05\|2.6\|

=============================

5 Hilly's; Func runs: 10000; result: 0.8930600046782612

25 Hilly's; Func runs: 10000; result: 0.6489975525320968

500 Hilly's; Func runs: 10000; result: 0.262496551797822

=============================

5 Forest's; Func runs: 10000; result: 0.9241962617798402

25 Forest's; Func runs: 10000; result: 0.7112057472851052

500 Forest's; Func runs: 10000; result: 0.24938963509983267

=============================

5 Megacity's; Func runs: 10000; result: 0.6938461538461538

25 Megacity's; Func runs: 10000; result: 0.3261538461538461

500 Megacity's; Func runs: 10000; result: 0.1001230769230778

=============================

All score: 4.80947 (53.44%)

Visualization of the algorithm's operation shows a significant spread of results across different test functions. Despite successful exploration of local surface areas, the algorithm may encounter the problem of getting stuck in local traps. This limits its ability to achieve a global optimum and may lead to a lack of stability in the search for an optimal solution.

The visualization of the work on the Skin test function is only an example of the algorithm operation and does not participate in the compilation of the rating table.

![Hilly](https://c.mql5.com/2/72/Hilly__2.gif)

**BSA** **on the [Hilly](https://www.mql5.com/en/articles/13923#tagHilly) test function**

![Forest](https://c.mql5.com/2/72/Forest__2.gif)

**BSA** **on the [Forest](https://www.mql5.com/en/articles/11785#tag3) test function**

![Megacity](https://c.mql5.com/2/72/Megacity__2.gif)

**BSA on the [Megacity](https://www.mql5.com/en/articles/11785#tag3) test function**

![Skin](https://c.mql5.com/2/72/Skin__1.gif)

**BSA on the** **Skin** **test function**

It is important to note that on the smooth Hilly test function with the high number of variables, the algorithm turned out to be extremely ineffective demonstrating the lowest result in the rating table among all the algorithms considered. Although on Forest and discrete Megacity functions of high dimension, BSA shows decent results when compared with other algorithms, including those located higher in the table.

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
| 7 | BSA | [bird swarm algorithm](https://www.mql5.com/en/articles/14491) | 0.90857 | 0.73661 | 0.25767 | 1.90285 | 0.90437 | 0.81619 | 0.16401 | 1.88457 | 0.61692 | 0.54154 | 0.10951 | 1.26797 | 5.055 | 56.17 |
| 8 | HS | [harmony search](https://www.mql5.com/en/articles/12163) | 0.86509 | 0.68782 | 0.32527 | 1.87818 | 0.99999 | 0.68002 | 0.09590 | 1.77592 | 0.62000 | 0.42267 | 0.05458 | 1.09725 | 4.751 | 52.79 |
| 9 | SSG | [saplings sowing and growing](https://www.mql5.com/en/articles/12268) | 0.77839 | 0.64925 | 0.39543 | 1.82308 | 0.85973 | 0.62467 | 0.17429 | 1.65869 | 0.64667 | 0.44133 | 0.10598 | 1.19398 | 4.676 | 51.95 |
| 10 | (PO)ES | [(PO) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.79025 | 0.62647 | 0.42935 | 1.84606 | 0.87616 | 0.60943 | 0.19591 | 1.68151 | 0.59000 | 0.37933 | 0.11322 | 1.08255 | 4.610 | 51.22 |
| 11 | WOAm | [wale optimization algorithm M](https://www.mql5.com/en/articles/14414) | 0.84521 | 0.56298 | 0.26263 | 1.67081 | 0.93100 | 0.52278 | 0.16365 | 1.61743 | 0.66308 | 0.41138 | 0.11357 | 1.18803 | 4.476 | 49.74 |
| 12 | ACOm | [ant colony optimization M](https://www.mql5.com/en/articles/11602) | 0.88190 | 0.66127 | 0.30377 | 1.84693 | 0.85873 | 0.58680 | 0.15051 | 1.59604 | 0.59667 | 0.37333 | 0.02472 | 0.99472 | 4.438 | 49.31 |
| 13 | BFO-GA | [bacterial foraging optimization - ga](https://www.mql5.com/en/articles/14011) | 0.89150 | 0.55111 | 0.31529 | 1.75790 | 0.96982 | 0.39612 | 0.06305 | 1.42899 | 0.72667 | 0.27500 | 0.03525 | 1.03692 | 4.224 | 46.93 |
| 14 | MEC | [mind evolutionary computation](https://www.mql5.com/en/articles/13432) | 0.69533 | 0.53376 | 0.32661 | 1.55569 | 0.72464 | 0.33036 | 0.07198 | 1.12698 | 0.52500 | 0.22000 | 0.04198 | 0.78698 | 3.470 | 38.55 |
| 15 | IWO | [invasive weed optimization](https://www.mql5.com/en/articles/11990) | 0.72679 | 0.52256 | 0.33123 | 1.58058 | 0.70756 | 0.33955 | 0.07484 | 1.12196 | 0.42333 | 0.23067 | 0.04617 | 0.70017 | 3.403 | 37.81 |
| 16 | Micro-AIS | [micro artificial immune system](https://www.mql5.com/en/articles/13951) | 0.79547 | 0.51922 | 0.30861 | 1.62330 | 0.72956 | 0.36879 | 0.09398 | 1.19233 | 0.37667 | 0.15867 | 0.02802 | 0.56335 | 3.379 | 37.54 |
| 17 | COAm | [cuckoo optimization algorithm M](https://www.mql5.com/en/articles/11786) | 0.75820 | 0.48652 | 0.31369 | 1.55841 | 0.74054 | 0.28051 | 0.05599 | 1.07704 | 0.50500 | 0.17467 | 0.03380 | 0.71347 | 3.349 | 37.21 |
| 18 | SDOm | [spiral dynamics optimization M](https://www.mql5.com/en/articles/12252) | 0.74601 | 0.44623 | 0.29687 | 1.48912 | 0.70204 | 0.34678 | 0.10944 | 1.15826 | 0.42833 | 0.16767 | 0.03663 | 0.63263 | 3.280 | 36.44 |
| 19 | NMm | [Nelder-Mead method M](https://www.mql5.com/en/articles/13805) | 0.73807 | 0.50598 | 0.31342 | 1.55747 | 0.63674 | 0.28302 | 0.08221 | 1.00197 | 0.44667 | 0.18667 | 0.04028 | 0.67362 | 3.233 | 35.92 |
| 20 | FAm | [firefly algorithm M](https://www.mql5.com/en/articles/11873) | 0.58634 | 0.47228 | 0.32276 | 1.38138 | 0.68467 | 0.37439 | 0.10908 | 1.16814 | 0.28667 | 0.16467 | 0.04722 | 0.49855 | 3.048 | 33.87 |
| 21 | GSA | [gravitational search algorithm](https://www.mql5.com/en/articles/12072) | 0.64757 | 0.49197 | 0.30062 | 1.44016 | 0.53962 | 0.36353 | 0.09945 | 1.00260 | 0.32667 | 0.12200 | 0.01917 | 0.46783 | 2.911 | 32.34 |
| 22 | BFO | [bacterial foraging optimization](https://www.mql5.com/en/articles/12031) | 0.61171 | 0.43270 | 0.31318 | 1.35759 | 0.54410 | 0.21511 | 0.05676 | 0.81597 | 0.42167 | 0.13800 | 0.03195 | 0.59162 | 2.765 | 30.72 |
| 23 | ABC | [artificial bee colony](https://www.mql5.com/en/articles/11736) | 0.63377 | 0.42402 | 0.30892 | 1.36671 | 0.55103 | 0.21874 | 0.05623 | 0.82600 | 0.34000 | 0.14200 | 0.03102 | 0.51302 | 2.706 | 30.06 |
| 24 | BA | [bat algorithm](https://www.mql5.com/en/articles/11915) | 0.59761 | 0.45911 | 0.35242 | 1.40915 | 0.40321 | 0.19313 | 0.07175 | 0.66810 | 0.21000 | 0.10100 | 0.03517 | 0.34617 | 2.423 | 26.93 |
| 25 | SA | [simulated annealing](https://www.mql5.com/en/articles/13851) | 0.55787 | 0.42177 | 0.31549 | 1.29513 | 0.34998 | 0.15259 | 0.05023 | 0.55280 | 0.31167 | 0.10033 | 0.02883 | 0.44083 | 2.289 | 25.43 |
| 26 | IWDm | [intelligent water drops M](https://www.mql5.com/en/articles/13730) | 0.54501 | 0.37897 | 0.30124 | 1.22522 | 0.46104 | 0.14704 | 0.04369 | 0.65177 | 0.25833 | 0.09700 | 0.02308 | 0.37842 | 2.255 | 25.06 |
| 27 | PSO | [particle swarm optimisation](https://www.mql5.com/en/articles/11386) | 0.59726 | 0.36923 | 0.29928 | 1.26577 | 0.37237 | 0.16324 | 0.07010 | 0.60572 | 0.25667 | 0.08000 | 0.02157 | 0.35823 | 2.230 | 24.77 |
| 28 | MA | [monkey algorithm](https://www.mql5.com/en/articles/12212) | 0.59107 | 0.42681 | 0.31816 | 1.33604 | 0.31138 | 0.14069 | 0.06612 | 0.51819 | 0.22833 | 0.08567 | 0.02790 | 0.34190 | 2.196 | 24.40 |
| 29 | SFL | [shuffled frog-leaping](https://www.mql5.com/en/articles/13366) | 0.53925 | 0.35816 | 0.29809 | 1.19551 | 0.37141 | 0.11427 | 0.04051 | 0.52618 | 0.27167 | 0.08667 | 0.02402 | 0.38235 | 2.104 | 23.38 |
| 30 | FSS | [fish school search](https://www.mql5.com/en/articles/11841) | 0.55669 | 0.39992 | 0.31172 | 1.26833 | 0.31009 | 0.11889 | 0.04569 | 0.47467 | 0.21167 | 0.07633 | 0.02488 | 0.31288 | 2.056 | 22.84 |
| 31 | RND | [random](https://www.mql5.com/en/articles/8122) | 0.52033 | 0.36068 | 0.30133 | 1.18234 | 0.31335 | 0.11787 | 0.04354 | 0.47476 | 0.25333 | 0.07933 | 0.02382 | 0.35648 | 2.014 | 22.37 |
| 32 | GWO | [grey wolf optimizer](https://www.mql5.com/en/articles/11785) | 0.59169 | 0.36561 | 0.29595 | 1.25326 | 0.24499 | 0.09047 | 0.03612 | 0.37158 | 0.27667 | 0.08567 | 0.02170 | 0.38403 | 2.009 | 22.32 |
| 33 | CSS | [charged system search](https://www.mql5.com/en/articles/13662) | 0.44252 | 0.35454 | 0.35201 | 1.14907 | 0.24140 | 0.11345 | 0.06814 | 0.42299 | 0.18333 | 0.06300 | 0.02322 | 0.26955 | 1.842 | 20.46 |
| 34 | EM | [electroMagnetism-like algorithm](https://www.mql5.com/en/articles/12352) | 0.46250 | 0.34594 | 0.32285 | 1.13129 | 0.21245 | 0.09783 | 0.10057 | 0.41085 | 0.15667 | 0.06033 | 0.02712 | 0.24412 | 1.786 | 19.85 |

### Summary

The Bird Swarm Algorithm (BSA) is a fascinating research tool that captures the imagination with its rich logic that embodies the diverse states and strategies of bird flocks. Working on this algorithm was interesting to me because of the complex dynamics within it, where individual and group actions of birds are subject to different conditions and combinations.

The complexity of BSA is also reflected in the large number of parameters that require careful tuning to achieve optimal results. To optimize the algorithm parameters, I had to write an EA to work in the mathematical calculation mode of the standard MetaTrader 5 tester allowing me to find good external parameters that ensured the 7 th place in the rating for the algorithm. There may be potential for further improvements, and I encourage those interested to experiment with this algorithm. I believe that it has yet unexplored possibilities for achieving better results, as there are many variations in the order of execution and combination of sequences of bird behaviors (the article explores 4 types of behavior).

![tab](https://c.mql5.com/2/72/tab__5.png)

Figure 2. Color gradation of algorithms according to relevant tests Results greater than or equal to 0.99 are highlighted in white

![chart](https://c.mql5.com/2/72/chart__3.png)

Figure 3. The histogram of algorithm test results (on a scale from 0 to 100, the more the better,

where 100 is the maximum possible theoretical result, the archive features a script for calculating the rating table)

**BSA pros and cons:**

Advantages:

1. Good results on the sharp Forest function and discrete Megacity of large dimension.

Disadvantages:

1. Complex implementation.
2. Low convergence.
3. Low scalability on smooth functions such as Hilly (problems with high dimensionality tasks).

The article is accompanied by an archive with the current code versions. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14491](https://www.mql5.com/ru/articles/14491)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14491.zip "Download all attachments in the single ZIP archive")

[BSA.zip](https://www.mql5.com/en/articles/download/14491/bsa.zip "Download BSA.zip")(28.34 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/471655)**
(6)


![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
3 Apr 2024 at 11:45

**fxsaber [#](https://www.mql5.com/ru/forum/465068#comment_52924706):**

Which AO converges the fastest (number of FF calculations)? It doesn't matter where it converges to. As long as there are a minimum of steps.

Any of the top 5, they converge very quickly.


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
3 Apr 2024 at 11:50

**Andrey Dik [#](https://www.mql5.com/ru/forum/465068#comment_52924885):**

Any of the top 5, very quick to converge.

I wish there was a numerical value for fast.

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
3 Apr 2024 at 11:58

**fxsaber [#](https://www.mql5.com/ru/forum/465068#comment_52924906):**

Too bad there's no numerical value for quickness.

You could do it, make several runs of tests, save the FF values at each epoch, calculate the average improvement at each corresponding epoch. Of course, there will be different values for each number of variables. This is if you get very fussy with numerical indicators of "convergence speed".

In each first test for all three test functions (10 parameters), the Top 5 of the list will be very close to the theoretical maximum already around the 100th epoch (with a population of 50).

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
3 Apr 2024 at 12:02

**Andrey Dik [#](https://www.mql5.com/ru/forum/465068#comment_52925304):**

Of course, you can do it, do several runs of tests, save the FF values at each epoch, calculate the average improvement at each corresponding epoch. Of course, for each number of variables there will be different indicators. This is if you are very fussy with numerical indicators of "convergence speed".

In each first test for all three test functions (10 parameters), the Top 5 of the list will be very close to the theoretical maximum already around the 100th epoch (with a population of 50).

~5000 FF?

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
3 Apr 2024 at 12:08

**fxsaber [#](https://www.mql5.com/ru/forum/465068#comment_52925325):**

~5,000 FF?

Yes. Even at 50th epoch will be already around 70-80% of the theoretical max.

Well, this is of course with parameter step 0 (as it is done by me when testing). If the step is different from 0, the convergence is even higher.

![MQL5 Wizard Techniques you should know (Part 33): Gaussian Process Kernels](https://c.mql5.com/2/89/logo-midjourney_image_15615_403_3890__4.png)[MQL5 Wizard Techniques you should know (Part 33): Gaussian Process Kernels](https://www.mql5.com/en/articles/15615)

Gaussian Process Kernels are the covariance function of the Normal Distribution that could play a role in forecasting. We explore this unique algorithm in a custom signal class of MQL5 to see if it could be put to use as a prime entry and exit signal.

![Reimagining Classic Strategies (Part V): Multiple Symbol Analysis on USDZAR](https://c.mql5.com/2/89/logo-midjourney_image_15570_399_3853.png)[Reimagining Classic Strategies (Part V): Multiple Symbol Analysis on USDZAR](https://www.mql5.com/en/articles/15570)

In this series of articles, we revisit classical strategies to see if we can improve the strategy using AI. In today's article, we will examine a popular strategy of multiple symbol analysis using a basket of correlated securities, we will focus on the exotic USDZAR currency pair.

![Non-stationary processes and spurious regression](https://c.mql5.com/2/74/Non-stationary_processes_and_spurious_regression___LOGO.png)[Non-stationary processes and spurious regression](https://www.mql5.com/en/articles/14412)

The article demonstrates spurious regression occurring when attempting to apply regression analysis to non-stationary processes using Monte Carlo simulation.

![MQL5 Wizard Techniques you should know (Part 32): Regularization](https://c.mql5.com/2/90/logo-15576.png)[MQL5 Wizard Techniques you should know (Part 32): Regularization](https://www.mql5.com/en/articles/15576)

Regularization is a form of penalizing the loss function in proportion to the discrete weighting applied throughout the various layers of a neural network. We look at the significance, for some of the various regularization forms, this can have in test runs with a wizard assembled Expert Advisor.

[![](https://www.mql5.com/ff/sh/592yc11u3j4rs5z9z2/01.png)How AI helps create robots for MetaTrader 5Learn from our book "Neural Networks in Algo Trading with MQL5"Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=ghrobswocqgvhztzjldphupateyllpro&s=9929cb0b8629585b5a42fabc06c525e41f6c0ebdf3045d044a5413b93ea88b47&uid=&ref=https://www.mql5.com/en/articles/14491&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068116660088403361)

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