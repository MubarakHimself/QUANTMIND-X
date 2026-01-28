---
title: Artificial Bee Hive Algorithm (ABHA): Theory and methods
url: https://www.mql5.com/en/articles/15347
categories: Integration, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:07:51.151027
---

[![](https://www.mql5.com/ff/sh/0uquj7zv5pmx2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Market analytics in MQL5 Channels\\
\\
Tens of thousands of traders have chosen this messaging app to receive trading tips.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=epadtzgppsywkaeumqycnulasoijfbgz&s=9615c3e5c371aa0d7b34529539d05c10df73b35a1e2213e4ceee008933c7ede0&uid=&ref=https://www.mql5.com/en/articles/15347&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071596867793398517)

MetaTrader 5 / Examples


**Contents**

1. [Introduction](https://www.mql5.com/en/articles/15347#tag1)
2. [Implementation of the algorithm](https://www.mql5.com/en/articles/15347#tag2)
3. [Conclusion](https://www.mql5.com/en/articles/15347#tag3)

### 1\. Introduction

In one of the previous articles, we already considered the [Artificial Bee Colony (ABC) algorithm](https://www.mql5.com/en/articles/11736), which is a notable example of nature inspiring the creation of efficient computational methods. A bee colony is not just a group of insects, but a highly organized system in which each bee performs its own unique role. Some bees become scouts, exploring the surrounding area in search of food, while others take on the role of foragers, collecting nectar and pollen. This form of cooperation allows the colony to be more efficient in finding resources.

The new artificial beehive algorithm considered here provides a more comprehensive and in-depth look at bees' foraging behavior, demonstrating how collective interaction and role assignments facilitate the search for new food sources. It demonstrates how interactions between agents can lead to more efficient outcomes. The algorithm takes a closer look at the individual roles in a bee colony.

The main goal of ABHA is to find optimal solutions in high-dimensional spaces where functions may have many local minima and maxima. This makes the optimization problem particularly challenging, as traditional methods can get stuck at local extremes without reaching the global optimum. The ABHA algorithm draws inspiration from the efficient foraging strategies used by bees. In nature, bees use collective methods to efficiently find nectar sources, and this principle has been adapted to create an algorithm that can improve the process of finding optimal solutions.

The ABHA structure includes various states that reflect the dynamics of bee behavior. One such state is the "experimental state," during which bees exchange information about food sources they have found. This state promotes the accumulation of knowledge about the most productive areas of multidimensional space. Another important state is the "search state", when bees actively explore the space in search of the best sources, using information received from their brethren.

The Artificial Beehive Algorithm (ABHA) was developed in 2009 by a group of researchers led by Andrés Muñoz. It is aimed at solving continuous optimization problems.

### 2\. Implementation of the algorithm

ABHA is an interesting method that uses an analogy with the behavior of bees to solve difficult optimization problems in continuous search spaces. Let's look at the basic concepts of the algorithm:

1\. **Modeling the behavior of bees**:

- ABHA is based on the model of individual bee behaviour. Each agent (bee) in the algorithm follows a set of behavioral rules to determine what actions it should take.

2\. **Interaction within the hive**:

- ABHA's main focus is on the interactions between bees.
- Bees exchange information about the solutions they have found.

The algorithm uses four states (roles) of bees, which can be viewed as different types of bee behavior. For clarity, it is convenient to present these states in the form of a table:

| Bee's state | Types of behavior |
| --- | --- |
| Novice | Located in a "hive" (an abstract position where information is exchanged) and has no information about food sources. |
| Experienced | Has information about the food source and can share it. |
| Search | Looks for a better food source than the current one. |
| Food source | Assesses the profitability of its food source and decides whether it is worth declaring. |

Each of these states reflects different aspects of bees' behavior in searching for and exploiting food sources. Each state of a bee represents a set of rules that the corresponding bee follows:

| Bee's state | Rules for changing the position of a bee in space |
| --- | --- |
| Novice | The bee may initiate a random search or follow a dance if one is available. If information about the food source is not available, it may begin a random search. |
| Experienced | If the information about a food source is valid, a bee can pass it to other bees through dance. If the information is invalid, it may start a random search. |
| Search | A bee changes its position using information about direction and step size until it finds the best food source. |
| Food source | A bee evaluates the profitability of a source and, if it does not meet the criteria, may change the direction of the search. |

Not all bees can share information about a food source, and not every bee can receive information from other bees that have this ability. There are two types of bees that have the ability to receive or pass information about a food source:

1\. Having the ability to pass information:

- **Experienced**– passes information to other bees through dance if the information about the food source is valid.

2\. Having the ability to receive information through the dance of other bees:

- **Novice**– may follow a dance if available to gain information about food sources.
- **Experienced**– able to receive information through dance from other bees and has access to information about food sources.

Bees can change their states depending on conditions related to their experience and information about food sources. Below we list the types of bees and the conditions under which they can transition from one state to another:

1\. **Novice**

- _Into_ _experienced_. If a novice receives information about a highly profitable food source (for example, through dances of other bees), it may transition to the experienced state.
- _Into_ _search_. If a novice does not receive information about food sources, it may begin searching on its own and go into a search state.

2\. **Experienced**

- _Into search._ If the information about the current food source is not good enough (e.g. the current fitness value is below the threshold), a bee may enter a search state to search for new sources.
- _Into food source._ If information about a food source is confirmed (e.g. the current fitness value is high and stable), a bee may switch to a food source state to analyze the source in more depth.

3\. **Search**

- _Into food source._ If a searching bee finds a food source with good characteristics (e.g., the current fitness value is better than the threshold), it can switch to the food source state to periodically evaluate the profitability of the source.
- _Into novice._ If a searching bee does not find any food sources or the information is not valuable enough, it may revert to novice.

4\. **Food source**

- _Into search._ If the current food source turns out to be impractical (e.g. the current fitness value is worse than the threshold), a bee may switch to a search state to search for new sources.
- _Into experienced._ If a food source bee finds a food source that proves beneficial, it may enter the experienced state to pass on the information to other bees.

These state transitions allow the algorithm to dynamically change its search strategy depending on the position of individuals in the solution space and to efficiently exchange information about food sources.

" _Food source_" and bee " _dance_" concepts play an important role in the algorithm. In the context of the algorithm, a " _food source_" represents a potential solution to the optimization problem:

1\. Each " _food source_" corresponds to a certain position, which represents a possible solution of the optimization problem in the search space.

2\. Quality or " _profitability_" of a food source is determined by the value of the objective function at that point. The higher the value of the objective function, the " _more profitable_" the food source.

3\. Bees in the algorithm search for and exploit the most _"profitable_" food sources - positions that correspond to the best values of the target function.

4\. Depletion of a food source means that no better solution can be found at a given point in the search space, and the bees should switch to searching for new, more promising areas.

In the algorithm, a " _dance_" is a way for bees to exchange information about the found " _food sources_". This is how it happens:

- When a bee finds a promising " _food source_", it returns to a " _hive_" and performs a " _dance_".
- The duration of a " _dance_" depends on the " _profitability_" of a found food source - the better the solution, the longer it " _dances_".

- Other bees in the hive may listen to this " **dance**" and get information about the food source. The probability that a bee will follow the " _dance_", depends on the source " _profitability_".

Thus, a " _food source_" in the ABHA algorithm is an analogue of a potential solution to an optimization problem, while its " _profitability_" is determined by the value of the objective function at the corresponding point in the search space, and a " _dance_" is a way of passing information between bees about the most promising solutions found during the search.

In the ABHA algorithm, each type of bee uses the fitness (or cost) values from the previous step, the current step, and the best fitness value as follows:

| Bee's state | **Novice** | **Experienced** | **Search** | **Food source** |
| --- | --- | --- | --- | --- |
| Current cost | Used to enter either the Experienced or Search state. | If the current value is high and the information is valid, it can pass this information on to other bees through a dance. | A bee uses its current fitness value to assess its current position and to decide whether to continue searching or change direction. | If the current value is below the threshold, it may decide that the source is not good enough and start looking for a new one. |
| Previous value | Not used | A bee compares the current value with the previous one to determine if the situation has improved. If the current value is better, it may increase the probability of passing the information. | A bee compares the current value with the previous one to determine whether the situation has improved. If the current value is better, it may continue in the same direction. | Not used |
| Best value | Not used | A bee can use the best value to assess whether to continue exploring a given food source or to look for a new one. | The bee uses the best value to determine whether the current food source is more profitable. This helps it make decisions about whether to stay put or continue searching. | A bee uses the best value to decide whether to continue exploiting the current food source or to move to the experienced state. |

Each type of bee uses fitness information at different iterations to make decisions and pass to different states.

In the algorithm, action selection probabilities, such as the probability of random search, the probability of following a dance, and the probability of abandoning the source (or staying near the source), are calculated dynamically throughout the optimization. These probabilities help agents (bees) make decisions about how to act based on current information and the state of the environment.

**Calculating probabilities:**

**1\.** Probability of random search (Psrs). The probability of a bee starting a random search instead of following a dance or staying on the current food source.

**2\.** Probability of following a dance (Prul). The probability of a bee following another bee's dance.

**3\.** Probability of abandoning a source (Pab). The probability of a bee staying near the current food source or abandoning it.

Bees in different states use probabilities differently, and the probabilities for each state are also different:

1.  **Novice**:

- Psrs: high, since a bee has no information about other sources and can start a random search.
- Prul: can be used if other bees are dancing, but the probability will depend on the information available.
- Pab: not applicable because the bee has not yet found a food source.

2\. **Experienced**:

- Psrs: low, since the bee already has information about the source.
- Prul: used to pass information to other bees if the source information is considered valid.
- Pab: can be used to decide whether to continue exploring a current source or abandon it if its profit is low.

3\. **Search**:

- Psrs: can be used if the bee does not find satisfactory sources.
- Prul: can be used if a bee decides to follow another bee's dance to find a new source.
- Pab: used to evaluate whether to continue searching or return to the "hive".

4\. **Food source**:

- Psrs: low because the bee has already found the source.
- Prul: used to pass information about a valid source to other bees if the source is considered profitable.
- Pab: high if the source does not produce satisfactory results, which may lead to a decision to abandon it.

Hopefully, I have covered all the nuances of bee behavior and can now start writing pseudocode.

Initialization:

    Set algorithm parameters (popSize, maxSearchAttempts, abandonmentRate, etc.)

    Create a population of popSize agents with random positions

    Set the initial state of each agent to Novice

Main loop:

    Until the stopping condition is reached:

        For each agent:

            Perform an action depending on the current state:

                Novice: Random search or follow the dance

                Experienced: Random search, following a dance or local search

                Search: Movement in a given direction

                Food search: Local search around the best position

            Evaluate agent fitness

            Update the best global solution if found

        Calculate probabilities and average cost of decisions

        For each agent:

            Update status:

                Novice: transition to Experienced or Search

                Experienced: possible transition to Food source or Search

                Search: possible transition to Novice or Food source

                Food source: possible transition to Search or Experienced

            Update the best personal position and value

    Calculate probabilities for agents

    Calculate average cost

Now let's start writing the algorithm code. The ABHA logic is quite complex and the code is voluminous, so we will describe the structure, class and methods involved in the work in as much detail as possible.

The **S\_ABHA\_Agent** structure represents the " _bee_" agent in the algorithm based on the bee behavior. Structure:

1\. **BeeState** enumeration defines the various states of the bee:

- stateNovice\- novice state when a bee is just beginning its activity.
- stateExperienced - experienced state when some experience has already ben accumulated.
- stateSearch - the search state when the bee is actively searching for food sources.
- stateSource - the state of being near the source and its periodic evaluation.


2\. Structure fields:

- position \[\] - array of the bee's current position in space.
- bestPosition \[\] - array of the best position found for the bee.
- direction \[\] - array of the bee movement direction.
- cost\- current food source quality.
- prevCost - previous food source quality.
- bestCost - best food source found quality.
- stepSize - bee step size ratio when moving along the coordinates.
- state - current state of a bee, represented as an integer.
- searchCounter - counter of the bee actions in the search state.

3\. Fields of the structure that set probabilities:

- pab\- probability of remaining near the food source.
- p\_si - dynamic probability that other bees will choose this bee's dance.
- p\_srs\- random search probability.
- p\_rul\- probability of following the dance.
- p\_ab\- probability of staying near the food source.

4\. **Init** method:

- Initialize the agent by accepting the number of **coords** coordinates and **initStepSize** initial step size.
- The method allocates memory for the arrays, sets initial values for all members of the structure, including state, counters, and probabilities.

The **S\_ABHA\_Agent** structure is used to model the behavior of a bee in an algorithm, where the bee can be in different states, explore the space in search of food and interact with other bees. Initializing a structure allows us to set the initial parameters required to start the algorithm.

```
//——————————————————————————————————————————————————————————————————————————————
struct S_ABHA_Agent
{
    enum BeeState
    {
      stateNovice      = 0,    // Novice state
      stateExperienced = 1,    // Experienced state
      stateSearch      = 2,    // Search state
      stateSource      = 3     // Food source state
    };

    double position        []; // Current position of the bee
    double bestPosition    []; // Best bee position found
    double direction       []; // Bee movement direction vector
    double cost;               // Current food source quality
    double prevCost;           // Previous food source quality
    double bestCost;           // Best food source found quality
    double stepSize;           // Step ratio in all coordinates during a bee movement
    int    state;              // Bee's current state
    int    searchCounter;      // Counter of the bee actions in the search state

    double pab;                // Probability of remaining near the source
    double p_si;               // Dynamic probability of other bees choosing this bee's dance

    double p_srs;              // Random search probability
    double p_rul;              // Probability of following the dance
    double p_ab;               // Probability of source rejection

    void Init (int coords, double initStepSize)
    {
      ArrayResize (position,        coords);
      ArrayResize (bestPosition,    coords);
      ArrayResize (direction,       coords);
      cost              = -DBL_MAX;
      prevCost          = -DBL_MAX;
      bestCost          = -DBL_MAX;
      state             = stateNovice;
      searchCounter     = 0;
      pab               = 0;
      p_si              = 0;
      p_srs             = 0;
      p_rul             = 0;
      p_ab              = 0;

      stepSize        = initStepSize;
    }
};
//——————————————————————————————————————————————————————————————————————————————
```

The **C\_AO\_ABHA** is derived from the **C\_AO** base class, which implies that it uses the functionality defined in the parent class. Description of the class:

1\. **C\_AO\_ABHA ()** constructor:

- Sets parameters such as the population size ( **popSize**), maximum number of search attempts ( **maxSearchAttempts**), ratios for different probabilities and initial step size ( **initialStepSize**).
- Initializes the **params** array, which contains the algorithm parameters.

2\. The **SetParams ()** method sets the values of the algorithm parameters based on the values stored in the **params** array.

3\. The **Init ()** method initializes the algorithm by taking the minimum and maximum values of the search range, the search step, and the number of epochs as inputs.

4\. The **Moving ()** and **Revision ()** methods are the methods designed to implement the logic of the movement of agents (bees) and revise the solutions found.

5\. Class members:

- maxSearchAttempts - maximum number of search attempts.
- abandonmentRate\- step change in the probability of remaining near the food source.
- randomSearchProbability\- random search probability.
- stepSizeReductionFactor\- step size reduction ratio.
- initialStepSize\- initial step size.
- S\_ABHA\_Agent agents \[\] - array of agents (bees) participating in the algorithm.
- avgCost\- average cost of the solution found.

6\. Methods for bees' actions:

- ActionRandomSearch ()\- random search in a given range.
- ActionFollowingDance () - following another bee's dance.
- ActionMovingDirection () - moving in a given direction taking into account the step size.
- ActionHiveVicinity () - movement in the vicinity of a food source.

7\. Methods for bee's activity in different states: **StageActivityNovice ()**, **StageActivityExperienced ()**, **StageActivitySearch ()**, **StageActivitySource ()** determine the actions of bees depending on their state.

8\. The methods for changing the bees' states: **ChangingStateForNovice ()**, **ChangingStateForExperienced ()**, **ChangingStateForSearch ()**, **ChangingStateForSource ()** change the bees' state depending on their activity.

9\. Calculation methods:

- **CalculateProbabilities ()** \- calculation of probabilities for the bees' actions.
- **CalculateAverageCost ()** \- calculation of the average cost of the solutions found.

The **C\_AO\_ABHA** class implements an optimization algorithm based on bees' behavior and covers various aspects of bees' actions, such as movement, decision making, and state changes.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_ABHA : public C_AO
{
  public:
  C_AO_ABHA ()
  {
    ao_name = "ABHA";
    ao_desc = "Artificial Bee Hive Algorithm";
    ao_link = "https://www.mql5.com/en/articles/15347";

    popSize                 = 10;

    maxSearchAttempts       = 10;
    abandonmentRate         = 0.1;
    randomSearchProbability = 0.1;
    stepSizeReductionFactor = 0.99;
    initialStepSize         = 0.5;

    ArrayResize (params, 6);
    params [0].name = "popSize";                 params [0].val = popSize;

    params [1].name = "maxSearchAttempts";       params [1].val = maxSearchAttempts;
    params [2].name = "abandonmentRate";         params [2].val = abandonmentRate;
    params [3].name = "randomSearchProbability"; params [3].val = randomSearchProbability;
    params [4].name = "stepSizeReductionFactor"; params [4].val = stepSizeReductionFactor;
    params [5].name = "initialStepSize";         params [5].val = initialStepSize;
  }

  void SetParams ()
  {
    popSize                 = (int)params [0].val;

    maxSearchAttempts       = (int)params [1].val;
    abandonmentRate         = params      [2].val;
    randomSearchProbability = params      [3].val;
    stepSizeReductionFactor = params      [4].val;
    initialStepSize         = params      [5].val;
  }

  bool Init (const double &rangeMinP  [], //minimum search range
             const double &rangeMaxP  [], //maximum search range
             const double &rangeStepP [], //step search
             const int     epochsP = 0);  //number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  int    maxSearchAttempts;
  double abandonmentRate;
  double randomSearchProbability;
  double stepSizeReductionFactor;
  double initialStepSize;

  S_ABHA_Agent agents [];

  private: //-------------------------------------------------------------------
  double avgCost;

  //Types of bees' actions----------------------------------------------------------
  double ActionRandomSearch       (int coordInd);                      //1. Random search (random placement in a range of coordinates)
  double ActionFollowingDance     (int coordInd, double val);          //2. Follow the dance (move in the direction of the dancer)
  double ActionMovingDirection    (S_ABHA_Agent &agent, int coordInd); //3. Move in a given direction with a step
  double ActionHiveVicinity       (int coordInd, double val);          //4. Move in the vicinity of a food source

  //Actions of bees in different states----------------------------------------
  void   StageActivityNovice      (S_ABHA_Agent &agent); //actions 1 or 2
  void   StageActivityExperienced (S_ABHA_Agent &agent); //actions 1 or 2 or 4
  void   StageActivitySearch      (S_ABHA_Agent &agent); //actions 3
  void   StageActivitySource      (S_ABHA_Agent &agent); //actions 4

  //Change bees' state----------------------------------------------------
  void ChangingStateForNovice      (S_ABHA_Agent &agent);
  void ChangingStateForExperienced (S_ABHA_Agent &agent);
  void ChangingStateForSearch      (S_ABHA_Agent &agent);
  void ChangingStateForSource      (S_ABHA_Agent &agent);

  void CalculateProbabilities ();
  void CalculateAverageCost   ();
};
//——————————————————————————————————————————————————————————————————————————————
```

The **Init** method of the **C\_AO\_ABHA** class is responsible for initializing the ABHA algorithm. Let's break it down piece by piece:

1\. Method parameters:

- rangeMinP \[\] - array of minimum search range values. These are the lower bounds for each variable that will be optimized.
- rangeMaxP \[\] - array of maximum search range values. These are the upper bounds for each variable.
- rangeStepP \[\] - search step array. These are values that determine how much the variables change during the search.
- epochsP - number of epochs (iterations) covered by the algorithm.

2\. The method returns 'true' if initialization was successful, and 'false' otherwise.

Method logic:

- Calling the **StandardInit** method with search range parameters. This method performs standard initialization operations such as setting search bounds and steps. If initialization fails, the **Init** method terminates execution and returns 'false'.
- The **ArrayResize** method changes the size of the "agents" array, which represents the bees (agents) in the algorithm. The array size is set to popSize, which determines the number of agents participating in the optimization.
- The loop initializes each agent in the "agents" array. The **Init** method is called for each agent, which sets its initial coordinates (from the " **coords**" array) and the **initialStepSize** initial step size. This step determines how far the agent can move during the search.

The **Init** method of the **C\_AO\_ABHA** class performs the following tasks:

- Checks whether the standard initialization of search parameters was successful.
- Resizes the agent array according to the given number of bees.
- Initializes each agent with its own **Init** method setting the initial parameters required for the algorithm to work.

If all steps are completed successfully, the method returns "true", which indicates successful initialization of the algorithm.

```
//——————————————————————————————————————————————————————————————————————————————
bool C_AO_ABHA::Init (const double &rangeMinP  [], //minimum search range
                      const double &rangeMaxP  [], //maximum search range
                      const double &rangeStepP [], //step search
                      const int     epochsP = 0)   //number of epochs
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  ArrayResize (agents, popSize);
  for (int i = 0; i < popSize; i++)
  {
    agents [i].Init (coords, initialStepSize);
  }
  return true;
}
//————————————————————
```

Next comes the **Moving** method of the **C\_AO\_ABHA** class, which implements the agent movement stage in the algorithm. The **Moving** method takes no parameters and returns no value. It controls the movement of agents (bees) during the optimization. Method logic:

1\. Initial initialization.

- Checking the **revision** variable. If 'false', this is the first method call and agent positions should be initialized.
- Generating initial positions - nested loops iterate over all agents and coordinates. For each coordinate:

  - Generates a **val** random value within the specified range ( **rangeMin** and **rangeMax**).
  - This value is then adjusted using the **SeInDiSp**, which sets the value to the allowed range with the given step.
  - The current position and the best position of the agent are set (at the beginning they are the same).
  - A random direction of agent movement is generated.

- After the initialization is complete, **revision** is set to 'true' and the method terminates.

2\. The basic logic of agent movement. Moving agents by states:

- The current state of each agent (novice, experienced, search or food source) is determined using **switch**. Depending on the state, the appropriate method is called, which controls the behavior of the agent (for example, **StageActivityNovice**, **StageActivityExperienced** etc).
- After performing actions depending on the state, the positions of the agents are updated using the **SeInDiSp** to stay within acceptable limits.

3\. The loop updates the **a** array copying the current positions of the agents into the corresponding elements of the **a** array.

The final meaning of the **Moving** method, which controls the movement of agents in the ABHA algorithm. It first initializes the agents' positions if this is the first method call, then updates their positions depending on their state and includes:

- Generating random starting positions.
- Determining the behavior of agents depending on their state.
- Updating current positions of agents in the **a** array.

This method is the key and control method for calling other bee state methods for the dynamic behavior of agents in the optimization process.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::Moving ()
{
  //----------------------------------------------------------------------------
  if (!revision)
  {
    double val = 0.0;

    for (int i = 0; i < popSize; i++)
    {
      for (int c = 0; c < coords; c++)
      {
        val = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        val = u.SeInDiSp (val, rangeMin [c], rangeMax [c], rangeStep [c]);

        agents [i].position     [c] = val;
        agents [i].bestPosition [c] = val;
        agents [i].direction    [c] = u.RNDfromCI (-(rangeMax [c] - rangeMin [c]), (rangeMax [c] - rangeMin [c]));

        a [i].c [c] = val;
      }
    }
    revision = true;
    return;
  }
  //----------------------------------------------------------------------------
  for (int i = 0; i < popSize; i++)
  {
    switch (agents [i].state)
    {
      //------------------------------------------------------------------------
      //Novice
      case S_ABHA_Agent::stateNovice:
      {
        StageActivityNovice (agents [i]);
        break;
      }
        //------------------------------------------------------------------------
        //Experienced
      case S_ABHA_Agent::stateExperienced:
      {
        StageActivityExperienced (agents [i]);
        break;
      }
        //------------------------------------------------------------------------
        //Search
      case S_ABHA_Agent::stateSearch:
      {
        StageActivitySearch (agents [i]);
        break;
      }
        //------------------------------------------------------------------------
        //Food source
      case S_ABHA_Agent::stateSource:
      {
        StageActivitySource (agents [i]);
        break;
      }
    }
    //--------------------------------------------------------------------------
    for (int c = 0; c < coords; c++)
    {
      agents [i].position [c] = u.SeInDiSp (agents [i].position [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      a      [i].c        [c] = agents [i].position [c];
    }
  }
  for (int i = 0; i < popSize; i++) for (int c = 0; c < coords; c++) a [i].c [c] = agents [i].position [c];
}
//——————————————————————————————————————————————————————————————————————————————
```

The **Revision** method of the **C\_AO\_ABHA** class is responsible for updating the state of agents in the algorithm and performs several key actions related to evaluating and updating the positions and states of agents. The **Revision** method takes no parameters and returns no values. It serves for updating information about the status of agents and their positions based on performance. Method logic:

1\. Finding the best agent:

- The **ind** variable is initialized using the **-1** value to track the agent index with the best cost.
- When searching for the best agent, the cycle goes through all **popSize** agents:
- If the **a \[i\].f** agent cost exceeds the current **fB** maximum, **fB** is updated and the **ind** index is saved.
- If the agent with the best cost ( **ind** is not equal to **-1**) is found, the **ArrayCopy** function, which copies the coordinates of the best agent into the **cB** array, is called.

2\. The cycle goes through all agents and updates their **agents\[i\].cost** cost based on the **a** array values.

3\. The **CalculateProbabilities** method, calculating the probabilities for each agent based on their current costs, is called. This is used to determine how agents will act at the next step.

4\. The **CalculateAverageCost** method is called as well. It calculates the average cost of all agents, which is necessary for the bees to analyze their own state and move to new states.

5\. The cycle goes through all agents and calls the appropriate method for changing the agent state depending on their current one **ChangingStateForNovice**, **ChangingStateForExperienced** etc.

6\. The cycle goes through all agents and checks if the agent's current value is greater than its best **bestCost**:

- If yes, **bestCost** is updated and the current position of the agent is copied to **bestPosition**.
- The **prevCost** previous cost is updated to the **cost** current cost.


The final meaning of the **Revision** method operation, which is responsible for updating the state information of the agents in the algorithm and performs the following key actions:

1\. Finds the agent with the best cost and updates the corresponding variables.

2\. Updates the prices of all agents.

3\. Calculates probabilities based on current values.

4\. Calculates the average cost of agents.

5\. Updates agent states based on their performance.

6\. Updates best agent prices and positions.

This method is an important part of the algorithm, as it allows agents to adapt based on their success and allows them to exchange information.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::Revision ()
{
  //----------------------------------------------------------------------------
  int ind = -1;

  for (int i = 0; i < popSize; i++)
  {
    if (a [i].f > fB)
    {
      fB = a [i].f;
      ind = i;
    }
  }

  if (ind != -1) ArrayCopy (cB, a [ind].c, 0, 0, WHOLE_ARRAY);

  //----------------------------------------------------------------------------
  for (int i = 0; i < popSize; i++) agents [i].cost = a [i].f;

  //----------------------------------------------------------------------------
  //Calculate the probabilities for bees at the current cost
  CalculateProbabilities ();

  //----------------------------------------------------------------------------
  //Calculate the average cost
  CalculateAverageCost ();

  //----------------------------------------------------------------------------
  //update bees' states (novice, experienced, search, source)
  for (int i = 0; i < popSize; i++)
  {
    switch (agents [i].state)
    {
      case S_ABHA_Agent::stateNovice:
      {
        ChangingStateForNovice (agents [i]);
        break;
      }
      case S_ABHA_Agent::stateExperienced:
      {
        ChangingStateForExperienced (agents [i]);
        break;
      }
      case S_ABHA_Agent::stateSearch:
      {
        ChangingStateForSearch (agents [i]);
        break;
      }
      case S_ABHA_Agent::stateSource:
      {
        ChangingStateForSource (agents [i]);
        break;
      }
    }
  }
  //----------------------------------------------------------------------------
  //Update the cost for bees
  for (int i = 0; i < popSize; i++)
  {
    if (agents [i].cost > agents [i].bestCost)
    {
      agents [i].bestCost = agents [i].cost;

      ArrayCopy (agents [i].bestPosition, agents [i].position);
    }
    agents [i].prevCost = agents [i].cost;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

### Conclusion

We considered the ABHA beehive algorithm, analyzed in detail the principles of its operation, wrote the pseudocode of the algorithm, and also described the structure, class and initialization, as well as the **Moving** and **Revision** methods. In the next article, we will continue writing the algorithm code and cover all the other methods. As usual, we will conduct tests on test functions and summarize the results of the algorithm in the rating table.

- github: [https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5](https://www.mql5.com/go?link=https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5 "https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5")
- CodeBase: [https://www.mql5.com/ru/code/49355](https://www.mql5.com/ru/code/49355)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15347](https://www.mql5.com/ru/articles/15347)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15347.zip "Download all attachments in the single ZIP archive")

[ABHA.zip](https://www.mql5.com/en/articles/download/15347/abha.zip "Download ABHA.zip")(29.82 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/480956)**

![Custom Indicator: Plotting Partial Entry, Exit and Reversal Deals for Netting Accounts](https://c.mql5.com/2/85/Tra7ar_os_Pontos_de_Entradas_Parciais_em_contas_Netting___LOGO.png)[Custom Indicator: Plotting Partial Entry, Exit and Reversal Deals for Netting Accounts](https://www.mql5.com/en/articles/12576)

In this article, we will look at a non-standard way of creating an indicator in MQL5. Instead of focusing on a trend or chart pattern, our goal will be to manage our own positions, including partial entries and exits. We will make extensive use of dynamic matrices and some trading functions related to trade history and open positions to indicate on the chart where these trades were made.

![Trend Prediction with LSTM for Trend-Following Strategies](https://c.mql5.com/2/111/LSTM_logo.png)[Trend Prediction with LSTM for Trend-Following Strategies](https://www.mql5.com/en/articles/16940)

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) designed to model sequential data by effectively capturing long-term dependencies and addressing the vanishing gradient problem. In this article, we will explore how to utilize LSTM to predict future trends, enhancing the performance of trend-following strategies. The article will cover the introduction of key concepts and the motivation behind development, fetching data from MetaTrader 5, using that data to train the model in Python, integrating the machine learning model into MQL5, and reflecting on the results and future aspirations based on statistical backtesting.

![Price Action Analysis Toolkit Development (Part 11): Heikin Ashi Signal EA](https://c.mql5.com/2/117/Price_Action_Analysis_Toolkit_Development_Part_11___LOGO__2.png)[Price Action Analysis Toolkit Development (Part 11): Heikin Ashi Signal EA](https://www.mql5.com/en/articles/17021)

MQL5 offers endless opportunities to develop automated trading systems tailored to your preferences. Did you know it can even perform complex mathematical calculations? In this article, we introduce the Japanese Heikin-Ashi technique as an automated trading strategy.

![Neural Networks in Trading: Reducing Memory Consumption with Adam-mini Optimization](https://c.mql5.com/2/85/Reducing_memory_consumption_using_the_Adam_optimization_method___LOGO.png)[Neural Networks in Trading: Reducing Memory Consumption with Adam-mini Optimization](https://www.mql5.com/en/articles/15352)

One of the directions for increasing the efficiency of the model training and convergence process is the improvement of optimization methods. Adam-mini is an adaptive optimization method designed to improve on the basic Adam algorithm.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=aqvveietnavcnkpsakegfhajekfkkugy&ssn=1769191669789882104&ssn_dr=0&ssn_sr=0&fv_date=1769191669&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F15347&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Artificial%20Bee%20Hive%20Algorithm%20(ABHA)%3A%20Theory%20and%20methods%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691916699806769&fz_uniq=5071596867793398517&sv=2552)

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