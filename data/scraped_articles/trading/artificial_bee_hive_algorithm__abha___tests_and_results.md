---
title: Artificial Bee Hive Algorithm (ABHA): Tests and results
url: https://www.mql5.com/en/articles/15486
categories: Trading, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:57:50.937146
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/15486&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068849596257468011)

MetaTrader 5 / Tester


**Contents**

1. [Introduction](https://www.mql5.com/en/articles/15486#tag1)
2. [Resuming algorithm implementation](https://www.mql5.com/en/articles/15486#tag2)
3. [Test results](https://www.mql5.com/en/articles/15486#tag3)

### 1\. Introduction

In the previous article we delved into the fascinating world of the [Artificial Bee Hive Algorithm (ABHA)](https://www.mql5.com/en/articles/15347) by thoroughly examining its operating principles. We described the structure and class, as well as presented the algorithm pseudocode together with the Moving and Revision methods. This introduction will form the basis for further study and understanding of the algorithm.

In this article, we will continue delving into coding and covering all the remaining methods. As always, we will conduct testing on various test functions to evaluate the efficiency and performance of the algorithm. In conclusion, we will summarize the results of the algorithm work in the rating table.

Let us recall the main points of the ABHA algorithm, based on the model of the bees' individual state and behavior:

- Each bee is represented as an individual agent whose behavior is regulated by states (novice, experienced, search and food source).
- At any given moment, the behavior of a single bee is determined by the internal and external information available to it, as well as its motivational state, in accordance with a set of specific rules.
- The set of rules is the same for every bee, but because the perceived environment differs for bees in different spatial locations, their behavior also differs.
- Bees may exhibit different behaviors depending on their foraging experience and/or their motivational state.

Thus, the ABHA model represents the behavior of bees as governed by a set of rules that adapt to individual characteristics and the environment.

### 2\. Resuming algorithm implementation

Let's proceed to continue writing the algorithm methods using our pseudocode described in the previous article.

The **StageActivityNovice** method controls how novice bees change their positions depending on random searches or following the "dance" of other agents. Method description:

- Parameters - the method accepts a reference to the **agent** object of **S\_ABHA\_Agent** type, which represents a "novice" in the algorithm.
- Return value - the method returns nothing ( **void**).

Method logic:

1\. The **val** variable is declared for storing the agent's current coordinate.

2\. **for** loop iterates through all the agent's coordinates, where **coords** is a total number of coordinates (or dimensions) in the search space.

3\. Inside the loop, the value of the agent's current coordinate is stored in the **val** variable.

4\. Depending on the probability, it is determined which action to perform:

- **Random search**\- if the generated random number (via **u.RNDprobab ()**) is less than **randomSearchProbability**, the random search is performed and the **c** coordinate is updated using the **ActionRandomSearch (c)** method, which generates a new position in this dimension.
- **Following the dance**\- otherwise, the agent will follow the "dance" of other agents. In this case, the **c** coordinate is updated using the **ActionFollowingDance (c, val)** method, which uses the **val** value to determine a new position based on information about other agents.

The **StageActivityNovice** method controls the behavior of novices in the ABHA algorithm and ultimately performs the following key actions:

1\. Iterates through each agent coordinate.

2\. Depending on a random probability, decides whether the agent will perform a random search or follow the "dance" of other agents.

3\. Updates the agent's position at each coordinate according to the selected action.

This method allows novices to adapt to the environment using both random strategies and strategies based on interactions with other agents.

```
//——————————————————————————————————————————————————————————————————————————————
//Actions 1 or 2
void C_AO_ABHA::StageActivityNovice (S_ABHA_Agent &agent)
{
  double val;

  for (int c = 0; c < coords; c++)
  {
    val = agent.position [c];

    if (u.RNDprobab () < randomSearchProbability) agent.position [c] = ActionRandomSearch   (c);
    else                                          agent.position [c] = ActionFollowingDance (c, val);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **StageActivityExperienced** method of the **C\_AO\_ABHA** class is responsible for the actions of experienced agents in the ABHA algorithm and controls how experienced agents choose their actions depending on random probabilities and their current strategies.

1\. The **rnd** variable is declared. It is to be used to store a random number generated to decide on the agent's actions.

2\. **for** loop iterates through all the agent's coordinates, where **coords** is a total number of coordinates (or dimensions) in the search space.

3\. Inside the loop, the **rnd** random number is generated for each coordinate using the **RNDprobab ()** method, which returns the value from **0** to **1**.

4\. If **rnd** is less or equal to **agent.p\_srs** (random search probability), the agent performs a random search, updating its position at **c** coordinate using the **ActionRandomSearch (c)** method.

Probability of following the dance:

- If **rnd** exceeds **agent.p\_srs** and less or equal to **agent.p\_rul** (probability of following the dance), the agent will follow the "dance" of other agents, updating its position using the **ActionFollowingDance (c,** **agent.position \[c\])** method.

Probability of staying near a source:

- If none of the previous conditions are met, the agent remains near the source, updating its position using the **ActionHiveVicinity (c, agent.bestPosition \[c\])**, where **agent.bestPosition \[c\]** represents the best known position of the agent.

The **StageActivityExperienced** method controls the behavior of experienced agents in the ABHA algorithm and ultimately does the following:

1\. Iterates through each agent coordinate.

2\. Generates a random number to select an action.

3\. Depending on the generated number and probabilities, determines whether the agent will perform a random search, follow the "dance" of other agents, or stay near the source.

This method allows experienced agents to adapt to the environment using more complex strategies than that of novices, and thus improve their resource search performance.

```
//——————————————————————————————————————————————————————————————————————————————
//actions 1 or 2 or 4
void C_AO_ABHA::StageActivityExperienced (S_ABHA_Agent &agent)
{
  double rnd = 0;

  for (int c = 0; c < coords; c++)
  {
    rnd = u.RNDprobab ();

    // random search probability
    if (rnd <= agent.p_srs)
    {
      agent.position [c] = ActionRandomSearch (c);
    }
    else
    {
      // Probability of following the dance
      if (agent.p_srs < rnd && rnd <= agent.p_rul)
      {
        agent.position [c] = ActionFollowingDance (c, agent.position [c]);
      }
      // Probability of remaining near the source
      else
      {
        agent.position [c] = ActionHiveVicinity (c, agent.bestPosition [c]);
      }
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **StageActivitySearch** method of the **C\_AO\_ABHA** class is responsible for the actions of agents during the search phase and controls how agents move in the search space, updating their positions depending on the chosen direction. The method does the following:

1\. Iterates through each agent coordinate.

2\. For each coordinate, the **ActionMovingDirection** method is called, which determines the new direction of the agent's movement.

3\. Updates the agent's position at the corresponding coordinate.

This method allows agents to actively move in the search space, adapting their positions depending on the chosen directions.

```
//——————————————————————————————————————————————————————————————————————————————
//Actions 3
void C_AO_ABHA::StageActivitySearch (S_ABHA_Agent &agent)
{
  for (int c = 0; c < coords; c++)
  {
    agent.position [c] = ActionMovingDirection (agent, c);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **StageActivitySource** method in the **C\_AO\_ABHA** class is designed to perform actions related to determining food sources for agents in the model. It updates the positions of agents based on their best positions in the context of being near the hive. The method performs the following steps:

1\. Initializes the **val** variable.

2\. Iterates through each agent coordinate.

3\. For each coordinate, updates the agent's position by calling the **ActionHiveVicinity** method, which determines the agent's new position based on its best known position.

This method helps "food source" agents to concentrate all their attention on a detailed exploration of the surroundings of a known food source.

```
//——————————————————————————————————————————————————————————————————————————————
//Actions 4
void C_AO_ABHA::StageActivitySource (S_ABHA_Agent &agent)
{
  double val = 0;

  for (int c = 0; c < coords; c++)
  {
    agent.position [c] = ActionHiveVicinity (c, agent.bestPosition [c]);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **ActionRandomSearch** method in the **C\_AO\_ABHA** class is designed to perform a random search within a given range of coordinates and allows the agent to randomly select a value within a given range. The method performs extended exploration of the search space and performs the following actions:

1\. Accepts the index of the coordinate a random value should be generated for.

2\. Uses a random number generation method to obtain a value in the range defined by the minimum and maximum values for a given coordinate.

3\. Returns a randomly generated value of **double** type.

```
//——————————————————————————————————————————————————————————————————————————————
//1. Random search (random placement in a range of coordinates)
double C_AO_ABHA::ActionRandomSearch (int coordInd)
{
  return u.RNDfromCI (rangeMin [coordInd], rangeMax [coordInd]);
}
//——————————————————————————————————————————————————————————————————————————————
```

The **ActionFollowingDance** method of the **C\_AO\_ABHA** class implements the logic of following a dancer, which implies movement in the direction of an agent that has already reached a certain level of experience. This method uses a probabilistic approach to select the agent the current agent will follow, and introduces random noise into the computed direction.

1\. Calculating the overall probability:

- The **totalProbability** initialized variable, which will store the sum of the **p\_si** probabilities of all "experienced" agents.
- The state of all agents is checked in a loop. If an agent is experienced, its probability is added to **totalProbability**.

2\. Generating a random value and selecting an agent:

- The **randomValue** random value is generated. The value is normalized relative to **totalProbability**.
- In the next loop, the **p\_si** probabilities of experienced agents are accumulated. As soon as the accumulated probability exceeds **randomValue**, the **ind** index of the selected agent is saved and the loop is broken.


3\. Checking the selected agent and calculating the new value:

- If no agent has been selected ( **ind** index is equal to **-1**), the **ActionRandomSearch** method is called for a random search.
- If an agent has been selected, the **direction** movement is calculated as the difference between the best position of the selected agent and the **val** current value.
- The random **noise** is generated in the range from **-1** to **1**.
- A new value is returned, which is the current **val** value adjusted for direction and noise.

The **ActionFollowingDance** method implements the experienced agent-following strategy (the agent is selected using the roulette rule, where a more experienced dancing bee has a higher chance of being selected), using a probabilistic approach to select the agent and adding random noise to the direction of movement, which makes the behavior of agents more diverse and adaptive.

```
———————————————————————————————————————————————————————————————————————
//2. Follow the dance (move in the direction of the dancer)
double C_AO_ABHA::ActionFollowingDance (int coordInd, double val)
{
  //----------------------------------------------------------------------------
  double totalProbability = 0;

  for (int i = 0; i < popSize; i++)
  {
    if (agents [i].state == S_ABHA_Agent::stateExperienced)
    {
      totalProbability += agents [i].p_si;
    }
  }

  //----------------------------------------------------------------------------
  double randomValue = u.RNDprobab () * totalProbability;
  double cumulativeProbability = 0;
  int    ind = -1;

  for (int i = 0; i < popSize; i++)
  {
    if (agents [i].state == S_ABHA_Agent::stateExperienced)
    {
      cumulativeProbability += agents [i].p_si;

      if (cumulativeProbability >= randomValue)
      {
        ind = i;
        break;
      }
    }
  }

  //----------------------------------------------------------------------------
  if (ind == -1)
  {
    return ActionRandomSearch (coordInd);
  }

  double direction = agents [ind].bestPosition [coordInd] - val;
  double noise     = u.RNDfromCI (-1.0, 1.0);

  return val + direction * noise;
}
//——————————————————————————————————————————————————————————————————————————————
```

The **ActionMovingDirection** method of the **C\_AO\_ABHA** class is responsible for moving the agent in a given direction with a certain step. Let's break down the code piece by piece.

1\. Agent movement:

- Updating the agent's position at the specified **coordInd** coordinate.
- **agent.position \[coordInd\]**\- current position of the agent at the given coordinate.
- **agent.stepSize** \- step size, by which the agent moves in this direction.
- **agent.direction \[coordInd\]**\- direction of the agent's movement along the specified coordinate.

- Multiplying **stepSize** by **direction**, we get the amount of movement added to the agent's current position.

2\. Step decrease size:

- After the movement, the **stepSize** step size is decreased by the **stepSizeReductionFactor** ratio.
- This is necessary to simulate the effect of decreasing motion, when the agent begins to move more slowly in order to refine the solution found.


The **ActionMovingDirection** method implements a simple logic of the agent moving in a given direction considering the step size and decreasing this step after the movement.

```
//——————————————————————————————————————————————————————————————————————————————
//3. Move in a given direction with a step
double C_AO_ABHA::ActionMovingDirection (S_ABHA_Agent &agent, int coordInd)
{
  agent.position [coordInd] += agent.stepSize * agent.direction [coordInd];
  agent.stepSize *= stepSizeReductionFactor;

  return agent.position [coordInd];
}
//——————————————————————————————————————————————————————————————————————————————
```

The **ActionHiveVicinity** method of the **C\_AO\_ABHA** class is meant for defining the agent behavior in the vicinity of a food source. The **ActionHiveVicinity** method is responsible for generating a new position at a given coordinate. The probability of generating a new position in the vicinity of the current one is higher.

```
//——————————————————————————————————————————————————————————————————————————————
//4. Move in the vicinity of a food source
double C_AO_ABHA::ActionHiveVicinity (int coordInd, double val)
{
  return u.PowerDistribution (val, rangeMin [coordInd], rangeMax [coordInd], 12);
}
//——————————————————————————————————————————————————————————————————————————————
```

Let's look at the **ChangingStateForNovice** method in the **C\_AO\_ABHA** class responsible for changing the state of the agent depending on its current **cost** fitness and information about food sources.

1\. Previous and best cost are not used in this method.

2\. Checking the status:

- If the current cost of a food source **agent.cost** exceeds the average one **avgCost**, the agent's state changes to experienced **stateExperienced**. This means that the agent has received enough information about highly profitable food sources and is ready to act more efficiently.

3\. Moving to **Search** state:

- If the current value does not exceed the average one, the agent goes into the search state **stateSearch**. This state assumes that the agent has no information about food sources and should start a random search.

4\. Random search direction:

- The cycle goes through all **coords** coordinates and a random direction is assigned for each coordinate. The **RNDfromCI** method generates a random number within the specified range and is used to determine the amount of movement and direction within the specified range ( **rangeMin** and **rangeMax**).

5\. Initializing search parameters:

- The **stepSize** is set for the agent movement together with the **searchCounter**, which keeps track of the number of iterations the agent has made in searching for food.

The **ChangingStateForNovice** method is responsible for changing the state of the "novice" bee depending on the value of its food source. If the value is high, the agent becomes experienced. If the value is low, the agent goes into the search state, starts a random search, and initializes its parameters.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::ChangingStateForNovice (S_ABHA_Agent &agent)
{
  //Current cost   : Used to enter either the Experienced or Search state.
  //Previous cost: Not used.
  //Best cost    : Not used.

  //Into Experienced. If a novice receives information about a highly profitable food source (for example, through dances of other bees), it may transition to the experienced state.
  //Into Search. If a novice does not receive information about food sources, it may begin a random search and enter a search state.

  if (agent.cost > avgCost) agent.state = S_ABHA_Agent::stateExperienced;
  else
  {
    agent.state = S_ABHA_Agent::stateSearch;

    for (int c = 0; c < coords; c++)
    {
      agent.direction [c] = u.RNDfromCI (-(rangeMax [c] - rangeMin [c]), (rangeMax [c] - rangeMin [c]));
    }

    agent.stepSize        = initialStepSize;
    agent.searchCounter   = 0;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **ChangingStateForExperienced** method in the **C\_AO\_ABHA** class is designed to control the state of the experienced bee agent depending on its current and previous food source cost values. Let's look at the method in detail.

1\. Changing the **pab** parameter:

- If the current value is less than the previous value: the bee decreases its probability of staying near the food source ( **pab**). If **pab** becomes less than **0**, it is set to **0**.

- If the current value is greater than the previous value: the bee increases the probability of staying near the food source. If **pab** exceeds **1**, it is set in **1**.

- If the current value has exceeded the best value, the probability of remaining near the source is set to the maximum value of **1**.


2\. Moving to **Source** or **Search**:

-  If the current value is 20% higher than the population average: the bee enters the **food source** ( **stateSource**) state, which means it has found a good food source. The probability of staying at the source is set to **1**.

- If the current value is less than the average: the bee goes into the **Search** ( **stateSearch**) state. This signals the need to search for new food sources.


3\. Random search direction:

-  In case of transition to the **Search** state: The bee is given random directions to search for new food sources.


4\. Initializing search parameters:

- A step size is set for the agent's movement together with the search counter that tracks the number of steps taken in search of food.

The **ChangingStateForExperienced** method controls the state of an experienced bee depending on its current food source value and previous values. It uses logic based on comparing current, previous, and best values to determine whether the bee should continue exploring a food source or look for a new one. The **pab** parameter (the probability of remaining at the source) is adjusted depending on changes in cost.

```
//——————————————————————————————————————————————————————————————————————————————

void C_AO_ABHA::ChangingStateForExperienced (S_ABHA_Agent &agent)
{

  //Current cost   : If the current value is high and the information is valid, it can pass this information on to other bees through a dance.
  //Previous cost: A bee compares the current value with the previous one to determine if the situation has improved. If the current value is better, it may increase the probability of passing the information.
  //Best cost    : A bee can use the best value to assess whether to continue exploring a given food source or to look for a new one.

  //Into Search. If the information about the current food source is not good enough (e.g. the current fitness value is below the threshold), a bee may enter a search state to search for new sources.
  //Into Food Source. If information about a food source is confirmed (e.g. the current fitness value is high and stable), a bee may switch to a food source state to analyze the source in more depth.

  if (agent.cost < agent.prevCost)
  {
    agent.pab -= abandonmentRate;

    if (agent.pab < 0.0) agent.pab = 0.0;
  }

  if (agent.cost > agent.prevCost)
  {

    agent.pab += abandonmentRate;

    if (agent.pab > 1.0) agent.pab = 1.0;
  }

  if (agent.cost > agent.bestCost) agent.pab = 1.0;

  if (agent.cost > avgCost * 1.2)
  {

    agent.state = S_ABHA_Agent::stateSource;

    agent.pab = 1;
  }

  else

    if (agent.cost < avgCost)
    {

      agent.state = S_ABHA_Agent::stateSearch;

      for (int c = 0; c < coords; c++)
      {

        agent.direction [c] = u.RNDfromCI (-(rangeMax [c] - rangeMin [c]), (rangeMax [c] - rangeMin [c]));
      }

      agent.stepSize        = initialStepSize;

      agent.searchCounter   = 0;
    }
}
//——————————————————————————————————————————————————————————————————————————————
```

Now let's take a closer look at the **ChangingStateForSearch** method code from the **C\_AO\_ABHA** class, which controls the behavior of the bee agent while searching for a food source. The method performs the following steps:

1\. Comparison of the current value with the previous one:

- If the current value of **agent.cost** is less than the previous **agent.prevCost**, this means that the bee is moving away from the good source. In that case,

  - The bee changes its direction by generating random values for each coordinate direction using the **u.RNDfromCI** function.
  - The initial step size **initialStepSize** is set, the bee is ready to continue exploring.
  - Increasing the **searchCounter** search counter allows tracking the number of search attempts.

2\. Comparing the current value with the best one:

- If the current value exceeds **agent.bestCost**, this means that the bee has found a more profitable food source. In that case,

  - The step size is reduced by the specified ratio **stepSizeReductionFactor**, which indicates the need for a smaller step to refine the solution found.
  - The search counter is reset to **0**, because the bee found a more profitable source.

3\. Checking the maximum number of search attempts:

- If the search counter reaches the maximum number of attempts **maxSearchAttempts**, this means that the bee did not find anything profitable after a certain number of attempts. In that case,

  - The search counter is reset to **0**.
  - The bee goes into **Novice** **state**( **stateNovice**). This means it should start the search again.

4\. Check for a good food source:

- If the current value is 20% higher than the **avgCost** average one, this indicates that the bee has discovered a good food source. In that case,

  - The bee enters the **Food Source** state ( **stateSource**), which means that it will further evaluate the profitability of the source.
  - The probability of staying near the source ( **pab**) is set to **1**. The bee will be more inclined to stay near that source.

The **ChangingStateForSearch** method manages the behavior of the bee in the search state, making decisions based on comparisons of current, previous, and best fitness values. It allows the bee to adapt to the environment and pass into **Food Source** or **Novice** state, as well as adjust the step size depending on the food sources found.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::ChangingStateForSearch (S_ABHA_Agent &agent)
{
  //Current cost  : A bee uses its current fitness value to assess its current position and to decide whether to continue searching or change direction
  //Previous value: A bee compares the current value with the previous one  to determine if things have improved. If the current value is better, it can continue in the same direction.
  //Best value    : The bee uses the best value to determine whether the current food source is more profitable than the previous ones. This helps it make decisions about whether to stay put or continue searching.

  //Into Food Source. If a searching bee finds a food source with good characteristics (e.g., the current fitness value is better than the threshold), it can switch to the food source state to evaluate the profitability of the source.
  //Into Novice. If a searching bee does not find any food sources or the information is impractical, it may revert to novice.

  if (agent.cost < agent.prevCost)
  {
    for (int c = 0; c < coords; c++)
    {
      agent.direction [c] = u.RNDfromCI (-(rangeMax [c] - rangeMin [c]), (rangeMax [c] - rangeMin [c]));
    }

    agent.stepSize = initialStepSize;
    agent.searchCounter++;
  }

  if (agent.cost > agent.bestCost)
  {
    agent.stepSize *= stepSizeReductionFactor;
    agent.searchCounter = 0;
  }

  if (agent.searchCounter >= maxSearchAttempts)
  {
    agent.searchCounter = 0;
    agent.state = S_ABHA_Agent::stateNovice;
    return;
  }

  if (agent.cost > avgCost * 1.2)
  {
    agent.state = S_ABHA_Agent::stateSource;
    agent.pab = 1;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **ChangingStateForSource** method of the **C\_AO\_ABHA** class manages the behavior of the bee agent while exploiting the food source. Method general structure:

1\. Moving to **Search** state:

- If the current **agent.cost** value is below **avgCost**, this indicates that the current food source is not profitable.
- The bee reduces the probability of staying near the source ( **pab**) by **abandonmentRate**, meaning it becomes less likely to stay near the current source.
- If the random value obtained from **u.RNDprobab** exceeds **agent.pab**, the bee decides to go into the **Search** state ( **stateSearch**):

  - The probability is reset **pab = 0**.
  - The bee changes its direction by generating random values for each coordinate using **u.RNDfromCI**.
  -  The **initialStepSize** initial step size is set and the **searchCounter** search counter is reset to **0**.

3\. Moving to **Experienced** state:

- If the current value of **agent.cost** exceeds the best value **agent.bestCost**, this means that the current food source is proving to be profitable.

  - In this case, the bee goes into the **Experienced** ( **stateExperienced**) state and it will further pass information about a good source to other bees.
  - The probability of staying near the source ( **pab**) is set to **1**. The bee will remain near the source with the highest possible probability.

The **ChangingStateForSource** method managed the food source bee. The method allows the bee to decide whether to continue to stay near the current source or to look for a new one. The method takes into account both current value and best achievements.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::ChangingStateForSource      (S_ABHA_Agent &agent)
{
  //Current cost  : If the current value is below the threshold, it may decide that the source is not good enough and start looking for a new one.
  //Previous value: The bee can use the previous value to compare and determine if the situation has improved. If the current value is worse, it may signal the need to change the strategy.
  //Best value    : A bee uses the best value to decide whether to continue exploiting the current food source or to look for a new, more profitable one.

  //Into Search. If the current food source turns out to be impractical (e.g. the current fitness value is worse than the threshold), a bee may switch to a search state to search for new sources.
  //Into Experienced. If a food source bee finds a food source that proves beneficial, it may enter the experienced state to pass on the information to other bees.

  if (agent.cost < avgCost)
  {
    agent.pab -= abandonmentRate;

    if (u.RNDprobab () > agent.pab)
    {
      agent.state = S_ABHA_Agent::stateSearch;
      agent.pab = 0;

      for (int c = 0; c < coords; c++)
      {
        agent.direction [c] = u.RNDfromCI (-(rangeMax [c] - rangeMin [c]), (rangeMax [c] - rangeMin [c]));
      }

      agent.stepSize      = initialStepSize;
      agent.searchCounter = 0;
    }
  }

  if (agent.cost > agent.bestCost)
  {
    agent.state = S_ABHA_Agent::stateExperienced;
    agent.pab = 1;
  }
}
//—————————————————————————————————————————————————————————————————————————————
```

The **CalculateProbabilities** method in the **C\_AO\_ABHA** class is responsible for calculating the probabilities of different actions for each agent (bee) based on their current values. The method performs the following steps:

1\. Initialization of variables:

- **maxCost** is initialized to the minimum possible value to ensure that any agent cost will be greater than this value.
- **minCost** is initialized to a maximum value so that any agent cost can be less than this value.

2\. Search for the maximum and minimum costs:

- The loop iterates through all agents (bees) in the **popSize** population.
- Inside the loop, the current value of the agent is compared with **maxCost** and **minCost** updating their values if necessary.

3\. Calculating the value range:

- **costRange** represents the difference between the maximum and minimum values, which allows normalizing the probabilities.

4\. Calculating probabilities for each agent:

- **p\_si**\- probability for an agent based on its cost. The higher the cost, the higher the probability (normalized by range).
- **p\_srs**\- random search probability specified in advance.
- **p\_rul**\- probability of following the dance. This means that the higher the probability of staying at the source, the lower the probability of following the dance.
- **p\_ab**\- probability of remaining near the food source equal to the agent's **pab**.

After this, the sum of all three probabilities is calculated, and each of them is normalized so that the sum of the probabilities is equal to **1**. This is done by dividing each probability by the total sum.

The **CalculateProbabilities** method allows each bee to estimate its chances of performing various actions (random search, following a dance, locating at a source) based on their current cost.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::CalculateProbabilities ()
{
  double maxCost = -DBL_MAX;
  double minCost =  DBL_MAX;

  for (int i = 0; i < popSize; i++)
  {
    if (agents [i].cost > maxCost) maxCost = agents [i].cost;
    if (agents [i].cost < minCost) minCost = agents [i].cost;
  }

  double costRange = maxCost - minCost;

  for (int i = 0; i < popSize; i++)
  {
    agents [i].p_si = (maxCost - agents [i].cost) / costRange;

    agents [i].p_srs = randomSearchProbability; // random search probability
    agents [i].p_rul = 1.0 - agents [i].pab;    // probability of following the dance
    agents [i].p_ab  = agents [i].pab;          // probability of staying near the source

    double sum = agents [i].p_srs + agents [i].p_rul + agents [i].p_ab;

    agents [i].p_srs /= sum;
    agents [i].p_rul /= sum;
    agents [i].p_ab  /= sum;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **CalculateAverageCost** method of the **C\_AO\_ABHA** class is intended to calculate **average cost** of all bee agents in the population. This information is necessary for analyzing the state of the population and making decisions within the algorithm. The average cost serves as an indicator of the agents' success and is also used in further calculations.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ABHA::CalculateAverageCost ()
{
  double totalCost = 0;

  for (int i = 0; i < popSize; i++)
  {
    totalCost += agents [i].cost;
  }

  avgCost = totalCost / popSize;
}
//———————
```

### 3\. Test results

ABHA results:

ABHA\|Artificial Bee Hive Algorithm\|10.0\|10.0\|0.1\|0.1\|0.99\|0.5\|

=============================

5 Hilly's; Func runs: 10000; result: 0.8413125195861497

25 Hilly's; Func runs: 10000; result: 0.5422730855489947

500 Hilly's; Func runs: 10000; result: 0.2630407626746883

=============================

5 Forest's; Func runs: 10000; result: 0.8785786358650522

25 Forest's; Func runs: 10000; result: 0.47779307049664316

500 Forest's; Func runs: 10000; result: 0.17181208858518054

=============================

5 Megacity's; Func runs: 10000; result: 0.5092307692307693

25 Megacity's; Func runs: 10000; result: 0.3387692307692307

500 Megacity's; Func runs: 10000; result: 0.1039692307692317

=============================

All score: 4.12678 (45.85%)

We have finished writing the code and conducted a detailed analysis of all ABHA components. Now let's proceed directly to testing the algorithm on test functions and evaluate its efficiency. In the visualization of the algorithm, you can see a fairly strong spread in the test results.

![Hilly](https://c.mql5.com/2/117/Hilly__3.gif)

**ABHA on the [Hilly](https://www.mql5.com/en/articles/13923#tagHilly) test function**

![Forest](https://c.mql5.com/2/117/Forest__3.gif)

**ABHA on the [Forest](https://www.mql5.com/en/articles/11785#tag3) test function**

![Megacity](https://c.mql5.com/2/117/Megacity__3.gif)

**ABHA on the [Megacity](https://www.mql5.com/en/articles/11785#tag3) test function**

Based on the results of the conducted research, the algorithm confidently takes a place in the very middle of the rating table.

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | AO | Description | Hilly | Hilly final | Forest | Forest final | Megacity (discrete) | Megacity final | Final result | % of MAX |
| 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) |
| 1 | ANS | [across neighbourhood search](https://www.mql5.com/en/articles/15049) | 0.94948 | 0.84776 | 0.43857 | 2.23581 | 1.00000 | 0.92334 | 0.39988 | 2.32323 | 0.70923 | 0.63477 | 0.23091 | 1.57491 | 6.134 | 68.15 |
| 2 | CLA | [code lock algorithm](https://www.mql5.com/en/articles/14878) | 0.95345 | 0.87107 | 0.37590 | 2.20042 | 0.98942 | 0.91709 | 0.31642 | 2.22294 | 0.79692 | 0.69385 | 0.19303 | 1.68380 | 6.107 | 67.86 |
| 3 | (P+O)ES | [(P+O) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.92256 | 0.88101 | 0.40021 | 2.20379 | 0.97750 | 0.87490 | 0.31945 | 2.17185 | 0.67385 | 0.62985 | 0.18634 | 1.49003 | 5.866 | 65.17 |
| 4 | CTA | [comet tail algorithm](https://www.mql5.com/en/articles/14841) | 0.95346 | 0.86319 | 0.27770 | 2.09435 | 0.99794 | 0.85740 | 0.33949 | 2.19484 | 0.88769 | 0.56431 | 0.10512 | 1.55712 | 5.846 | 64.96 |
| 5 | SDSm | [stochastic diffusion search M](https://www.mql5.com/en/articles/13540) | 0.93066 | 0.85445 | 0.39476 | 2.17988 | 0.99983 | 0.89244 | 0.19619 | 2.08846 | 0.72333 | 0.61100 | 0.10670 | 1.44103 | 5.709 | 63.44 |
| 6 | ESG | [evolution of social groups](https://www.mql5.com/en/articles/14136) | 0.99906 | 0.79654 | 0.35056 | 2.14616 | 1.00000 | 0.82863 | 0.13102 | 1.95965 | 0.82333 | 0.55300 | 0.04725 | 1.42358 | 5.529 | 61.44 |
| 7 | SIA | [simulated isotropic annealing](https://www.mql5.com/en/articles/13870) | 0.95784 | 0.84264 | 0.41465 | 2.21513 | 0.98239 | 0.79586 | 0.20507 | 1.98332 | 0.68667 | 0.49300 | 0.09053 | 1.27020 | 5.469 | 60.76 |
| 8 | ACS | [artificial cooperative search](https://www.mql5.com/en/articles/15004) | 0.75547 | 0.74744 | 0.30407 | 1.80698 | 1.00000 | 0.88861 | 0.22413 | 2.11274 | 0.69077 | 0.48185 | 0.13322 | 1.30583 | 5.226 | 58.06 |
| 9 | TSEA | [turtle shell evolution algorithm](https://www.mql5.com/en/articles/14789) | 0.96798 | 0.64480 | 0.29672 | 1.90949 | 0.99449 | 0.61981 | 0.22708 | 1.84139 | 0.69077 | 0.42646 | 0.13598 | 1.25322 | 5.004 | 55.60 |
| 10 | DE | [differential evolution](https://www.mql5.com/en/articles/13781) | 0.95044 | 0.61674 | 0.30308 | 1.87026 | 0.95317 | 0.78896 | 0.16652 | 1.90865 | 0.78667 | 0.36033 | 0.02953 | 1.17653 | 4.955 | 55.06 |
| 11 | CRO | [chemical reaction optimization](https://www.mql5.com/en/articles/15080) | 0.94629 | 0.66112 | 0.29853 | 1.90593 | 0.87906 | 0.58422 | 0.21146 | 1.67473 | 0.75846 | 0.42646 | 0.12686 | 1.31178 | 4.892 | 54.36 |
| 12 | BSA | [bird swarm algorithm](https://www.mql5.com/en/articles/14491) | 0.89306 | 0.64900 | 0.26250 | 1.80455 | 0.92420 | 0.71121 | 0.24939 | 1.88479 | 0.69385 | 0.32615 | 0.10012 | 1.12012 | 4.809 | 53.44 |
| 13 | HS | [harmony search](https://www.mql5.com/en/articles/12163) | 0.86509 | 0.68782 | 0.32527 | 1.87818 | 0.99999 | 0.68002 | 0.09590 | 1.77592 | 0.62000 | 0.42267 | 0.05458 | 1.09725 | 4.751 | 52.79 |
| 14 | SSG | [saplings sowing and growing](https://www.mql5.com/en/articles/12268) | 0.77839 | 0.64925 | 0.39543 | 1.82308 | 0.85973 | 0.62467 | 0.17429 | 1.65869 | 0.64667 | 0.44133 | 0.10598 | 1.19398 | 4.676 | 51.95 |
| 15 | (PO)ES | [(PO) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.79025 | 0.62647 | 0.42935 | 1.84606 | 0.87616 | 0.60943 | 0.19591 | 1.68151 | 0.59000 | 0.37933 | 0.11322 | 1.08255 | 4.610 | 51.22 |
| 16 | BSO | [brain storm optimization](https://www.mql5.com/en/articles/14707) | 0.93736 | 0.57616 | 0.29688 | 1.81041 | 0.93131 | 0.55866 | 0.23537 | 1.72534 | 0.55231 | 0.29077 | 0.11914 | 0.96222 | 4.498 | 49.98 |
| 17 | WOAm | [wale optimization algorithm M](https://www.mql5.com/en/articles/14414) | 0.84521 | 0.56298 | 0.26263 | 1.67081 | 0.93100 | 0.52278 | 0.16365 | 1.61743 | 0.66308 | 0.41138 | 0.11357 | 1.18803 | 4.476 | 49.74 |
| 18 | AEFA | [artificial electric field algorithm](https://www.mql5.com/en/articles/15162) | 0.87700 | 0.61753 | 0.25235 | 1.74688 | 0.92729 | 0.72698 | 0.18064 | 1.83490 | 0.66615 | 0.11631 | 0.09508 | 0.87754 | 4.459 | 49.55 |
| 19 | ACOm | [ant colony optimization M](https://www.mql5.com/en/articles/11602) | 0.88190 | 0.66127 | 0.30377 | 1.84693 | 0.85873 | 0.58680 | 0.15051 | 1.59604 | 0.59667 | 0.37333 | 0.02472 | 0.99472 | 4.438 | 49.31 |
| 20 | BFO-GA | [bacterial foraging optimization - ga](https://www.mql5.com/en/articles/14011) | 0.89150 | 0.55111 | 0.31529 | 1.75790 | 0.96982 | 0.39612 | 0.06305 | 1.42899 | 0.72667 | 0.27500 | 0.03525 | 1.03692 | 4.224 | 46.93 |
| 21 | ABHA | [artificial bee hive algorithm](https://www.mql5.com/en/articles/15347) | 0.84131 | 0.54227 | 0.26304 | 1.64663 | 0.87858 | 0.47779 | 0.17181 | 1.52818 | 0.50923 | 0.33877 | 0.10397 | 0.95197 | 4.127 | 45.85 |
| 22 | ASBO | [adaptive social behavior optimization](https://www.mql5.com/en/articles/15347) | 0.76331 | 0.49253 | 0.32619 | 1.58202 | 0.79546 | 0.40035 | 0.26097 | 1.45677 | 0.26462 | 0.17169 | 0.18200 | 0.61831 | 3.657 | 40.63 |
| 23 | MEC | [mind evolutionary computation](https://www.mql5.com/en/articles/13432) | 0.69533 | 0.53376 | 0.32661 | 1.55569 | 0.72464 | 0.33036 | 0.07198 | 1.12698 | 0.52500 | 0.22000 | 0.04198 | 0.78698 | 3.470 | 38.55 |
| 24 | IWO | [invasive weed optimization](https://www.mql5.com/en/articles/11990) | 0.72679 | 0.52256 | 0.33123 | 1.58058 | 0.70756 | 0.33955 | 0.07484 | 1.12196 | 0.42333 | 0.23067 | 0.04617 | 0.70017 | 3.403 | 37.81 |
| 25 | Micro-AIS | [micro artificial immune system](https://www.mql5.com/en/articles/13951) | 0.79547 | 0.51922 | 0.30861 | 1.62330 | 0.72956 | 0.36879 | 0.09398 | 1.19233 | 0.37667 | 0.15867 | 0.02802 | 0.56335 | 3.379 | 37.54 |
| 26 | COAm | [cuckoo optimization algorithm M](https://www.mql5.com/en/articles/11786) | 0.75820 | 0.48652 | 0.31369 | 1.55841 | 0.74054 | 0.28051 | 0.05599 | 1.07704 | 0.50500 | 0.17467 | 0.03380 | 0.71347 | 3.349 | 37.21 |
| 27 | SDOm | [spiral dynamics optimization M](https://www.mql5.com/en/articles/12252) | 0.74601 | 0.44623 | 0.29687 | 1.48912 | 0.70204 | 0.34678 | 0.10944 | 1.15826 | 0.42833 | 0.16767 | 0.03663 | 0.63263 | 3.280 | 36.44 |
| 28 | NMm | [Nelder-Mead method M](https://www.mql5.com/en/articles/13805) | 0.73807 | 0.50598 | 0.31342 | 1.55747 | 0.63674 | 0.28302 | 0.08221 | 1.00197 | 0.44667 | 0.18667 | 0.04028 | 0.67362 | 3.233 | 35.92 |
| 29 | FAm | [firefly algorithm M](https://www.mql5.com/en/articles/11873) | 0.58634 | 0.47228 | 0.32276 | 1.38138 | 0.68467 | 0.37439 | 0.10908 | 1.16814 | 0.28667 | 0.16467 | 0.04722 | 0.49855 | 3.048 | 33.87 |
| 30 | GSA | [gravitational search algorithm](https://www.mql5.com/en/articles/12072) | 0.64757 | 0.49197 | 0.30062 | 1.44016 | 0.53962 | 0.36353 | 0.09945 | 1.00260 | 0.32667 | 0.12200 | 0.01917 | 0.46783 | 2.911 | 32.34 |
| 31 | BFO | [bacterial foraging optimization](https://www.mql5.com/en/articles/12031) | 0.61171 | 0.43270 | 0.31318 | 1.35759 | 0.54410 | 0.21511 | 0.05676 | 0.81597 | 0.42167 | 0.13800 | 0.03195 | 0.59162 | 2.765 | 30.72 |
| 32 | ABC | [artificial bee colony](https://www.mql5.com/en/articles/11736) | 0.63377 | 0.42402 | 0.30892 | 1.36671 | 0.55103 | 0.21874 | 0.05623 | 0.82600 | 0.34000 | 0.14200 | 0.03102 | 0.51302 | 2.706 | 30.06 |
| 33 | BA | [bat algorithm](https://www.mql5.com/en/articles/11915) | 0.59761 | 0.45911 | 0.35242 | 1.40915 | 0.40321 | 0.19313 | 0.07175 | 0.66810 | 0.21000 | 0.10100 | 0.03517 | 0.34617 | 2.423 | 26.93 |
| 34 | SA | [simulated annealing](https://www.mql5.com/en/articles/13851) | 0.55787 | 0.42177 | 0.31549 | 1.29513 | 0.34998 | 0.15259 | 0.05023 | 0.55280 | 0.31167 | 0.10033 | 0.02883 | 0.44083 | 2.289 | 25.43 |
| 35 | IWDm | [intelligent water drops M](https://www.mql5.com/en/articles/13730) | 0.54501 | 0.37897 | 0.30124 | 1.22522 | 0.46104 | 0.14704 | 0.04369 | 0.65177 | 0.25833 | 0.09700 | 0.02308 | 0.37842 | 2.255 | 25.06 |
| 36 | PSO | [particle swarm optimisation](https://www.mql5.com/en/articles/11386) | 0.59726 | 0.36923 | 0.29928 | 1.26577 | 0.37237 | 0.16324 | 0.07010 | 0.60572 | 0.25667 | 0.08000 | 0.02157 | 0.35823 | 2.230 | 24.77 |
| 37 | Boids | [boids algorithm](https://www.mql5.com/en/articles/14576) | 0.43340 | 0.30581 | 0.25425 | 0.99346 | 0.35718 | 0.20160 | 0.15708 | 0.71586 | 0.27846 | 0.14277 | 0.09834 | 0.51957 | 2.229 | 24.77 |
| 38 | MA | [monkey algorithm](https://www.mql5.com/en/articles/12212) | 0.59107 | 0.42681 | 0.31816 | 1.33604 | 0.31138 | 0.14069 | 0.06612 | 0.51819 | 0.22833 | 0.08567 | 0.02790 | 0.34190 | 2.196 | 24.40 |
| 39 | SFL | [shuffled frog-leaping](https://www.mql5.com/en/articles/13366) | 0.53925 | 0.35816 | 0.29809 | 1.19551 | 0.37141 | 0.11427 | 0.04051 | 0.52618 | 0.27167 | 0.08667 | 0.02402 | 0.38235 | 2.104 | 23.38 |
| 40 | FSS | [fish school search](https://www.mql5.com/en/articles/11841) | 0.55669 | 0.39992 | 0.31172 | 1.26833 | 0.31009 | 0.11889 | 0.04569 | 0.47467 | 0.21167 | 0.07633 | 0.02488 | 0.31288 | 2.056 | 22.84 |
| 41 | RND | [random](https://www.mql5.com/en/articles/8122) | 0.52033 | 0.36068 | 0.30133 | 1.18234 | 0.31335 | 0.11787 | 0.04354 | 0.47476 | 0.25333 | 0.07933 | 0.02382 | 0.35648 | 2.014 | 22.37 |
| 42 | GWO | [grey wolf optimizer](https://www.mql5.com/en/articles/11785) | 0.59169 | 0.36561 | 0.29595 | 1.25326 | 0.24499 | 0.09047 | 0.03612 | 0.37158 | 0.27667 | 0.08567 | 0.02170 | 0.38403 | 2.009 | 22.32 |
| 43 | CSS | [charged system search](https://www.mql5.com/en/articles/13662) | 0.44252 | 0.35454 | 0.35201 | 1.14907 | 0.24140 | 0.11345 | 0.06814 | 0.42299 | 0.18333 | 0.06300 | 0.02322 | 0.26955 | 1.842 | 20.46 |
| 44 | EM | [electroMagnetism-like algorithm](https://www.mql5.com/en/articles/12352) | 0.46250 | 0.34594 | 0.32285 | 1.13129 | 0.21245 | 0.09783 | 0.10057 | 0.41085 | 0.15667 | 0.06033 | 0.02712 | 0.24412 | 1.786 | 19.85 |

### Summary

The ABHA algorithm has been extensively tested on various test functions and the results have shown it to be competitive with other swarm intelligence-based algorithms. During these tests, ABHA demonstrated its efficiency and reliability.

The studies highlight the potential of the ABHA algorithm to solve not only traditional optimization problems but also more complex problems, such as multi-objective optimization and constraint problems. However, I expected more impressive results. Regardless, ABHA results are truly unrivaled among swarm algorithms.

In general, the algorithm can be assessed as a set of specific methods and techniques that are applicable to the vast majority of other optimization algorithms. The ability to reproduce results and adapt to different conditions makes this algorithm a promising tool in the field of computational optimization. Thus, ABHA not only expands the horizons of application of optimization algorithms, but also opens up new opportunities for research in the field of artificial intelligence and its practical application.

![tab](https://c.mql5.com/2/117/tab__1.jpg)

_Figure 1. Color gradation of algorithms according to relevant tests Results greater than or equal to_ _0.99_ are highlighted in white

![chart](https://c.mql5.com/2/117/chart__3.png)

_Figure 2. The histogram of algorithm test results (on a scale from 0 to 100, the more the better,_

_where 100 is the maximum possible theoretical result, the archive features a script for calculating the rating table)_

**ABHA pros and cons:**

Advantages:

1. Good results on low-dimensional problems.
2. Good results on discrete functions.


Disadvantages:

1. Complex logic and algorithm implementation.
2. Low convergence on high-dimensional problems on smooth functions.

3. A large number of external parameters.


The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

- github: [https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5](https://www.mql5.com/go?link=https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5 "https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5")
- CodeBase: [https://www.mql5.com/ru/code/49355](https://www.mql5.com/ru/code/49355)

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15486](https://www.mql5.com/ru/articles/15486)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15486.zip "Download all attachments in the single ZIP archive")

[ABHA.zip](https://www.mql5.com/en/articles/download/15486/abha.zip "Download ABHA.zip")(30.1 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/481078)**
(4)


![quargil34](https://c.mql5.com/avatar/avatar_na2.png)

**[quargil34](https://www.mql5.com/en/users/quargil34)**
\|
9 Feb 2025 at 14:35

**MetaQuotes:**

Check out the new article: [Artificial Bee Hive Algorithm (ABHA): Tests and results](https://www.mql5.com/en/articles/15486).

Author: [Andrey Dik](https://www.mql5.com/en/users/joo "joo")

Hi Andrew. I've known about this algorithm for over 10 years, and it was considered one of the best in the ant algorithm family.

The results are below expectations.  regards

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
9 Feb 2025 at 18:42

**quargil34 [#](https://www.mql5.com/ru/forum/471121#comment_55862681):**

Hi Andrei. I have known about this algorithm for more than 10 years, and it was considered one of the best in the ant algorithm family.

The results were lower than expected. with respect.

Hi. Yes, it does happen. For example, the same PSO turns out to be much weaker than one thinks it is. There are also situations when developers claim overestimated search capabilities of their algorithms, but in practice it turns out to be otherwise. This is one of the goals of my articles - to provide true and reproducible results of the most well-known optimisation algorithms.

There are several variations of algorithms on the theme of "bees", this one is one of the strongest among them.

![quargil34](https://c.mql5.com/avatar/avatar_na2.png)

**[quargil34](https://www.mql5.com/en/users/quargil34)**
\|
9 Feb 2025 at 20:21

**Andrey Dik [#](https://www.mql5.com/en/forum/481078#comment_55864062):**

Hi. Yes, it does happen. For example, the same PSO turns out to be much weaker than one thinks it is. There are also situations when developers claim overestimated search capabilities of their algorithms, but in practice it turns out to be otherwise. This is one of the goals of my articles - to provide true and reproducible results of the most well-known optimisation algorithms.

There are several variations of algorithms on the theme of "bees", this one is one of the strongest among them.

the Ans (Across Neighbourhood search) is like the cream of the crop

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
9 Feb 2025 at 21:43

**quargil34 [#](https://www.mql5.com/ru/forum/471121#comment_55864503):**

Ans (Across Neighbourhood Search) is like the cream of the crop.

Yes, it's good. But it is not without its flaws. The results are good in situations of limited computational resources, but if you look in the "long term", it tends to get stuck.


![Introduction to MQL5 (Part 12): A Beginner's Guide to Building Custom Indicators](https://c.mql5.com/2/117/Introduction_to_MQL5_Part_12___LOGO.png)[Introduction to MQL5 (Part 12): A Beginner's Guide to Building Custom Indicators](https://www.mql5.com/en/articles/17096)

Learn how to build a custom indicator in MQL5. With a project-based approach. This beginner-friendly guide covers indicator buffers, properties, and trend visualization, allowing you to learn step-by-step.

![Feature Engineering With Python And MQL5 (Part III): Angle Of Price (2) Polar Coordinates](https://c.mql5.com/2/117/Feature_Engineering_With_Python_And_MQL5_Part_III_Angle_Of_Price_2__LOGO.png)[Feature Engineering With Python And MQL5 (Part III): Angle Of Price (2) Polar Coordinates](https://www.mql5.com/en/articles/17085)

In this article, we take our second attempt to convert the changes in price levels on any market, into a corresponding change in angle. This time around, we selected a more mathematically sophisticated approach than we selected in our first attempt, and the results we obtained suggest that our change in approach may have been the right decision. Join us today, as we discuss how we can use Polar coordinates to calculate the angle formed by changes in price levels, in a meaningful way, regardless of which market you are analyzing.

![Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (I)](https://c.mql5.com/2/117/Creating_a_Trading_Administrator_Panel_in_MQL5_Part_IX___LOGO.png)[Creating a Trading Administrator Panel in MQL5 (Part IX): Code Organization (I)](https://www.mql5.com/en/articles/16539)

This discussion delves into the challenges encountered when working with large codebases. We will explore the best practices for code organization in MQL5 and implement a practical approach to enhance the readability and scalability of our Trading Administrator Panel source code. Additionally, we aim to develop reusable code components that can potentially benefit other developers in their algorithm development. Read on and join the conversation.

![Developing a Replay System (Part 58): Returning to Work on the Service](https://c.mql5.com/2/85/Desenvolvendo_um_sistema_de_Replay_Parte_58__LOGO.png)[Developing a Replay System (Part 58): Returning to Work on the Service](https://www.mql5.com/en/articles/12039)

After a break in development and improvement of the service used for replay/simulator, we are resuming work on it. Now that we've abandoned the use of resources like terminal globals, we'll have to completely restructure some parts of it. Don't worry, this process will be explained in detail so that everyone can follow the development of our service.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/15486&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068849596257468011)

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