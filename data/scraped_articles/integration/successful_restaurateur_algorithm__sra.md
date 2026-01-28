---
title: Successful Restaurateur Algorithm (SRA)
url: https://www.mql5.com/en/articles/17380
categories: Integration, Machine Learning, Strategy Tester
relevance_score: 3
scraped_at: 2026-01-23T21:03:51.823524
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/17380&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071544301688662533)

MetaTrader 5 / Tester


### **Contents**

1. [Introduction](https://www.mql5.com/en/articles/17380#tag1)
2. [Implementation of the algorithm](https://www.mql5.com/en/articles/17380#tag2)
3. [Test results](https://www.mql5.com/en/articles/17380#tag3)

### Introduction

I have always been fascinated by the parallels between optimization problems and real-life scenarios. While exploring new approaches to metaheuristic algorithms, I discovered similarities between population optimization and the evolution of the restaurant business, and this idea became the inspiration for what I call the Successful Restaurateur Algorithm (SRA).

Imagine a restaurant owner who is constantly striving to improve the menu to increase the restaurant's popularity and attract new customers. Instead of completely eliminating unpopular dishes, our restaurant owner takes a more subtle approach - identifying the least popular item and then carefully mixing it with elements from the most successful dishes. Sometimes conservative changes are made, and sometimes bold new ingredients are added. The goal is always the same: to turn the weakest offering into something that can become a new menu favorite for restaurant customers.

This culinary metaphor forms the basis of SRA. Unlike traditional evolutionary algorithms, which often completely discard poor performers, SRA focuses on rehabilitating poor performers by pairing them with successful elements. This approach preserves diversity in the solution space while steadily improving the overall quality of the population.

In this article, I will cover the basic mechanics of SRA, analyze its implementation, and how parameters like "temperature" and "intensity of cooking experiments" control the balance between exploitation and exploration. I will also share benchmark results comparing SRA to other well-known algorithms in the league table on various test functions.

What began as a creative thought experiment has evolved into a promising approach with unique characteristics. I invite you to explore how this restaurant-inspired algorithm offers optimization solutions with its own unique flavor.

### Implementation of the algorithm

Let's understand how the algorithm works through simple analogies. Imagine that I am the owner of a restaurant. I have a menu with different dishes and I notice that some of them are very popular, while some dishes are almost never ordered by guests. What am I going to do? I do not immediately remove unpopular dishes from the menu (thereby reducing the list of available dishes), but instead I take the least popular dish and try to improve it. How? I look at the restaurant's hit items and borrow some ideas or ingredients from there. For example, my fish dish does not sell well, but my salad is very popular. I take elements from a successful salad (maybe a special dressing or serving method) and apply them to the fish dish. It turns out to be something new.

Sometimes I make small changes, and sometimes I decide to try bold experiments. In the beginning, when I first opened the restaurant, I experimented more, but when I found a few really successful dishes, I began to fine-tune the composition of ingredients and their quantities. Over time, even the weakest dishes on my menu get better and better. And sometimes it happens that a former outsider, after some tweaking, becomes a new successful dish! Thus, the overall popularity of the restaurant grows due to the fact that all menu items are successful.

This is how my algorithm works: I do not throw out bad solutions, but constantly improve them, using ideas from the best solutions. And the further we go, the less we experiment, the more we refine what we have already found. Figure 1 shows the algorithm operation diagram.

![sra-algorithm-flow](https://c.mql5.com/2/124/sra-algorithm-flow1.png)

Figure 1. SRA algorithm operation diagram

The diagram starts with the Initialization block, where the initial menu is created. This is followed by the main loop of the algorithm, which centers on the restaurant current menu, sorted by dish quality. This menu is depicted as a color gradient from green (best dishes) to red (worst dishes). Below are four sequential steps: first, selecting dishes for improvement, taking the worst dish and the best donor with increased probability according to a quadratic law; second, creating new variants by combining recipes and mutating ingredients (the higher the temperature, the more daring the experiments); third, evaluating new dishes by calculating a fitness function; fourth, lowering the temperature to reduce the radicality of the experiments. On the left, the dotted arrow shows that the process is repeated until convergence is achieved or the stopping criterion is met. On the right are the designations: A (green circle) - the best dishes, B (red circle) - the worst dishes. The entire diagram visualizes a process reminiscent of a restaurateur who systematically improves weaker items on the menu by using elements of successful dishes.

Let's move on to writing the pseudocode for the SRA algorithm.

// **Initialization**

Create a population of popSize agents (menu items)

For each agent:

    Randomly initialize all coordinates within the allowed range

Set initial temperature = 1.0

Set cooling ratio = 0.98

Set the intensity of culinary experiments = 0.3

// **The main loop of the algorithm**

UNTIL the stopping condition is met:

    // **Step 1**: Rating of all agents

    For each agent:

        Calculate the value of the fitness function

    // Merge the current and previous populations

    Create a common menu from current agents and previous agents

    Sort the general menu by fitness function value from best to worst

    // **Step 2**: Creating new options

    For each agent in the new population:

        // Take the worst element from the first half of the sorted population

        Copy coordinates from the agent with index (popSize-1)

        // Choose between improvement and experimentation

        If (random number < (1.0 - menuInnovationRate \* temperature)):

            // We select a "donor" using the quadratic roulette method

            r = random number from 0 to 1

            r = r²

            donorIndex = scale r from 0 to (popSize-1)

            // For each coordinate

            For each c coordinate:

                // We take the coordinate from the donor with the probability of 0.8

                If (random number < 0.8):

                    current\_agent.c = donor.c

                // Adaptive mutation

                mutationRate = 0.1 + 0.4 \* temperature \* (agent\_index / popSize)

                If (random number < mutationRate):

                    // Select the mutation type

                    If (random number < 0.5):

                        current\_agent.c = gaussian distribution(current\_agent.c)

                    Otherwise:

                        current\_agent.c = random value in range

                    // Make sure the value is within acceptable limits

                    current\_agent.c = round to nearest valid value

        Otherwise:

            // Creating a new "dish"

            For each c coordinate:

                If (random number < 0.7):

                    current\_agent.c = random value in range

                Otherwise:

                    current\_agent.c = gaussian distribution(best\_solution.c)

                // Make sure the value is within acceptable limits

                current\_agent.c = round to the nearest valid value

        // Elitism - an occasional simple addition of elements from a better solution

        If (random number < 0.1):

            numEliteCoords = random number from 1 to (coords \* 0.3)

            For i from 1 to numEliteCoords:

                c = random coordinate index

                current\_agent.c = best\_solution.c

    // **Step 3**: Update the best solution

    For each agent:

        If agent.fitness > best\_solution.fitness:

            best\_solution = agent

    // **Step 4**: Temperature drop

    temperature = temperature \* cooling\_ratio

    If temperature < 0.1:

        temperature = 0.1

Return best\_solution

Now we can start writing the algorithm code. Write the C\_AO\_SRA class that inherits from the C\_AO main class and implements the SRA algorithm. Let's take a closer look at it.

**Constructor and destructor:** popSize, temperature, coolingRate, menuInnovationRate — these parameters determine the main characteristics of the algorithm, such as the number of agents and search control parameters.

**SetParams method:** updates the class parameters based on the values stored in the "params" array, the parameters previously initialized in the constructor.

**Init method:** intended to initialize the algorithm. It takes the minimum and maximum values, the step size of the parameters and the number of epochs and prepares the algorithm to perform the search task.

**Moving and Revision methods:** designed to perform the main stages of the algorithm operation related to the movement (or updating) of the state of agents and "Revision", which is responsible for revising and adapting the parameters.

**Class members:**

- temperature — current temperature associated with the study control and the temperature chart of the algorithm.
- coolingRate — cooling rate used to control how quickly the temperature will drop.
- menuInnovationRate — intensity of culinary experiments, the extent to which agents are encouraged to search for new solutions.

**Private class members:**

- S\_AO\_Agent menu \[\] — array of agents that represents a "menu" in the context of the SRA algorithm.
- S\_AO\_Agent menuT \[\] — array of agents used to temporarily store dish options.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_SRA : public C_AO
{
  public: //--------------------------------------------------------------------
  ~C_AO_SRA () { }
  C_AO_SRA ()
  {
    ao_name = "SRA";
    ao_desc = "Successful Restaurateur Algorithm (joo)";
    ao_link = "https://www.mql5.com/en/articles/17380";

    popSize            = 50;   // number of agents (size of the "menu")
    temperature        = 1.0;  // initial "temperature" for research control
    coolingRate        = 0.98; // cooling rate
    menuInnovationRate = 0.3;  // intensity of culinary experiments

    ArrayResize (params, 4);

    params [0].name = "popSize";            params [0].val = popSize;
    params [1].name = "temperature";        params [1].val = temperature;
    params [2].name = "coolingRate";        params [2].val = coolingRate;
    params [3].name = "menuInnovationRate"; params [3].val = menuInnovationRate;
  }

  void SetParams ()
  {
    popSize            = (int)params [0].val;
    temperature        = params      [1].val;
    coolingRate        = params      [2].val;
    menuInnovationRate = params      [3].val;
  }

  bool Init (const double &rangeMinP  [],  // minimum values
             const double &rangeMaxP  [],  // maximum values
             const double &rangeStepP [],  // step change
             const int     epochsP = 0);   // number of epochs

  void Moving   ();
  void Revision ();

  //----------------------------------------------------------------------------
  double temperature;        // current "temperature"
  double coolingRate;        // cooling rate
  double menuInnovationRate; // intensity of culinary experiments

  private: //-------------------------------------------------------------------
  S_AO_Agent menu  [];
  S_AO_Agent menuT [];
};
//——————————————————————————————————————————————————————————————————————————————
```

The Init method of the C\_AO\_SRA class initializes the algorithm:

**Check initialization:** The method calls StandardInit with the minimum and maximum values of the ranges and steps, if unsuccessful, "false" is returned.

**Initializing arrays:**

- Distributes the sizes of the "menu" and "menuT" arrays according to the number of agents.
- Initializes each agent in the "menu" array.

**Temperature reset:** sets the initial value of "temperature" to "1.0".

**Successful completion:** returns "true" if initialization was successful.

```
//——————————————————————————————————————————————————————————————————————————————
//--- Initialization
bool C_AO_SRA::Init (const double &rangeMinP  [],
                     const double &rangeMaxP  [],
                     const double &rangeStepP [],
                     const int epochsP = 0)
{
  if (!StandardInit (rangeMinP, rangeMaxP, rangeStepP)) return false;

  //----------------------------------------------------------------------------
  ArrayResize (menu,  popSize * 2);
  ArrayResize (menuT, popSize * 2);

  for (int p = 0; p < popSize * 2; p++) menu [p].Init (coords);

  temperature = 1.0; // reset temperature during initialization

  return true;
}
//——————————————————————————————————————————————————————————————————————————————
```

The Moving method of the C\_AO\_SRA class implements the main step of the algorithm. It has two main parts: initialization of agents and their adaptation through mutation and creation of new solutions.

**Initial initialization** (if "revision" is "false"):

- Each agent is initialized with random values within given ranges (rangeMin, rangeMax) using steps (rangeStep). The values are stored in "c" and "cB" for each agent.
- "revision" is set to "true" and the method terminates.

**Temperature drop**: the temperature is multiplied by the cooling ratio (coolingRate), which affects the likelihood of further changes.

**The main loop by agents**: for each agent, the worst element from the first half of the sorted population (from the menu array) is selected.

**Classification of actions**: with a certain probability, which depends on the temperature, the agent either:

- **modifies the current solution** (using the "donor recipe" from the best dish), applying a mutation with varying probability.
- **creates a new solution** (random values).

**Elitism**: with some probability, elements from the best solution found can be added to the new solution.

```
//——————————————————————————————————————————————————————————————————————————————
//--- The main step of the algorithm
void C_AO_SRA::Moving ()
{
  //----------------------------------------------------------------------------
  // Initial initialization
  if (!revision)
  {
    for (int p = 0; p < popSize; p++)
    {
      for (int c = 0; c < coords; c++)
      {
        a [p].c  [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        a [p].c  [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        a [p].cB [c] = a [p].c [c];
      }
    }

    revision = true;
    return;
  }

  //----------------------------------------------------------------------------
  // Lower the temperature
  temperature *= coolingRate;

  // Main loop on population agents
  for (int p = 0; p < popSize; p++)
  {
    // Take the worst element from the first half of the sorted population (with index popSize-1)
    // Remember that items are sorted from best to worst in the menu
    ArrayCopy (a [p].c, menu [popSize - 1].c, 0, 0, WHOLE_ARRAY);

    // Decide whether to create a hybrid or experiment with a new "dish"
    // The probability of an experiment depends on the temperature - there are more experiments at the beginning
    if (u.RNDprobab () < (1.0 - menuInnovationRate * temperature))
    {
      // Select a "donor-recipe" with a probability proportional to the success of the dish
      double r = u.RNDprobab ();
      r = pow (r, 2);                                         // Increased preference for better dishes
      int menuIND = (int)u.Scale (r, 0, 1.0, 0, popSize - 1); // The best ones are at the beginning of the array

      // For each coordinate
      for (int c = 0; c < coords; c++)
      {
        // Take the parameter from a successful dish with the probability depending on the temperature
        if (u.RNDprobab () < 0.8)
        {
          a [p].c [c] = menu [menuIND].c [c];
        }

        // Mutation with adaptive probability - the further from the best solution and the higher the temperature, the more mutations
        double mutationRate = 0.1 + 0.4 * temperature * (double)(p) / popSize;
        if (u.RNDprobab () < mutationRate)
        {
          // Combination of different types of mutations
          if (u.RNDprobab () < 0.5) a [p].c [c] = u.GaussDistribution (a [p].c [c], rangeMin [c], rangeMax [c], 2);
          else                      a [p].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]); // Sometimes a completely new value

          // Make sure the value is within acceptable limits
          a [p].c [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
        }
      }
    }
    else // Create a completely new "dish"
    {
      for (int c = 0; c < coords; c++)
      {
        // Variation 1: Completely random value
        if (u.RNDprobab () < 0.7)
        {
          a [p].c [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        }
        // Variation 2: based on the best solution found with a large deviation
        else
        {
          a [p].c [c] = u.GaussDistribution (cB [c], rangeMin [c], rangeMax [c], 1);
        }

        a [p].c [c] = u.SeInDiSp (a [p].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      }
    }

    // Sometimes we add elements from the best solution directly (elitism)
    if (u.RNDprobab () < 0.1)
    {
      int numEliteCoords = u.RNDintInRange (1, coords / 3); // Take from 1 to 30% of the coordinates
      for (int i = 0; i < numEliteCoords; i++)
      {
        int c = u.RNDminusOne (coords);
        a [p].c [c] = cB [c]; // Take the value from the best solution
      }
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The Revision method of the C\_AO\_SRA class is responsible for updating the best solution found and for managing the overall "menu" of solutions during the algorithm operation:

**Finding the best agent**: iterates through all agents in the current population and finds the agent with the best fitness function (f). If a new best agent is found, update the fB value and the bestIND index.

**Updating the best solution**: if the best agent is found (i.e. bestIND is not equal to -1), its decision parameters (c) are copied into the cB variable representing the current best decision.

**Updating the general "menu"**: adds the current parameters of all agents to the general "menu", which allows you to save completed experiments.

**Sorting the menu**: sorts the "menu" array by fitness function from best to worst solutions, ensuring that the best solutions are in the first half. This will be used in the next iteration of the algorithm.

**Temperature control**: sets a lower threshold for the temperature so that it does not fall below "0.1", which prevents it from converging too quickly.

```
//——————————————————————————————————————————————————————————————————————————————
//--- Update the best solution taking into account greedy selection and the probability of making worse decisions
void C_AO_SRA::Revision ()
{
  int bestIND = -1;

  // Find the best agent in the current population
  for (int p = 0; p < popSize; p++)
  {
    if (a [p].f > fB)
    {
      fB = a [p].f;
      bestIND = p;
    }
  }

  // If we find a better solution, update cB
  if (bestIND != -1) ArrayCopy (cB, a [bestIND].c, 0, 0, WHOLE_ARRAY);

  // Add the current set of dishes to the general "menu"
  for (int p = 0; p < popSize; p++)
  {
    menu [popSize + p] = a [p];
  }

  // Sort the entire "menu" from best to worst solutions
  // After sorting, the first half of the menu will contain the best solutions,
  // which will be used in the next iteration
  u.Sorting (menu, menuT, popSize * 2);

  // Prevent the temperature from falling below a certain threshold
  if (temperature < 0.1) temperature = 0.1;
}
//——————————————————————————————————————————————————————————————————————————————
```

### Test results

Now we can see how the SRA algorithm works. Below are the test results:

SRA\|Successful Restaurateur Algorithm\|50.0\|1.0\|0.98\|0.3\|

=============================

5 Hilly's; Func runs: 10000; result: 0.9688326305968623

25 Hilly's; Func runs: 10000; result: 0.6345483084017249

500 Hilly's; Func runs: 10000; result: 0.292167027537253

=============================

5 Forest's; Func runs: 10000; result: 0.946368863880973

25 Forest's; Func runs: 10000; result: 0.5550607959254661

500 Forest's; Func runs: 10000; result: 0.19124225531141872

=============================

5 Megacity's; Func runs: 10000; result: 0.7492307692307693

25 Megacity's; Func runs: 10000; result: 0.4403076923076923

500 Megacity's; Func runs: 10000; result: 0.12526153846153956

=============================

All score: 4.90302 (54.48%)

Visualization of the SRA algorithm operation on the test stand allows us to draw conclusions about the characteristic features of the search strategy. In this case, we observe a broad exploration of the search space: agents are evenly distributed throughout the space, exploring even its most remote corners. At the same time, there are no noticeable signs of grouping at local extremes; the movements of agents appear chaotic.

However, despite its good exploration ability, the algorithm exhibits some shortcomings in refining solutions, which is reflected in the relatively low convergence accuracy. It should also be noted that there is a small spread in the test results.

![Hilly](https://c.mql5.com/2/123/Hilly__1.gif)

_SRA on the [Hilly](https://www.mql5.com/en/articles/13923#tagHilly) test function_

![Forest](https://c.mql5.com/2/123/Forest__1.gif)

__SRA_ on the [Forest](https://www.mql5.com/en/articles/11785#tag3) test function_

![Megacity](https://c.mql5.com/2/123/Megacity__1.gif)

__SRA on the [Megacity](https://www.mql5.com/en/articles/11785#tag3) test function__

Based on the test results, the algorithm ranks 20th in the ranking of the strongest population optimization algorithms. Currently, the table presents nine proprietary optimization algorithms (joo), including the new SRA algorithm.

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \# | AO | Description | Hilly | Hilly final | Forest | Forest final | Megacity (discrete) | Megacity final | Final result | % of MAX |
| 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) |
| 1 | ANS | [across neighbourhood search](https://www.mql5.com/en/articles/15049) | 0.94948 | 0.84776 | 0.43857 | 2.23581 | 1.00000 | 0.92334 | 0.39988 | 2.32323 | 0.70923 | 0.63477 | 0.23091 | 1.57491 | 6.134 | 68.15 |
| 2 | CLA | [code lock algorithm (joo)](https://www.mql5.com/en/articles/14878) | 0.95345 | 0.87107 | 0.37590 | 2.20042 | 0.98942 | 0.91709 | 0.31642 | 2.22294 | 0.79692 | 0.69385 | 0.19303 | 1.68380 | 6.107 | 67.86 |
| 3 | AMOm | [animal migration ptimization M](https://www.mql5.com/en/articles/15543) | 0.90358 | 0.84317 | 0.46284 | 2.20959 | 0.99001 | 0.92436 | 0.46598 | 2.38034 | 0.56769 | 0.59132 | 0.23773 | 1.39675 | 5.987 | 66.52 |
| 4 | (P+O)ES | [(P+O) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.92256 | 0.88101 | 0.40021 | 2.20379 | 0.97750 | 0.87490 | 0.31945 | 2.17185 | 0.67385 | 0.62985 | 0.18634 | 1.49003 | 5.866 | 65.17 |
| 5 | CTA | [comet tail algorithm (joo)](https://www.mql5.com/en/articles/14841) | 0.95346 | 0.86319 | 0.27770 | 2.09435 | 0.99794 | 0.85740 | 0.33949 | 2.19484 | 0.88769 | 0.56431 | 0.10512 | 1.55712 | 5.846 | 64.96 |
| 6 | TETA | [time evolution travel algorithm (joo)](https://www.mql5.com/en/articles/16963) | 0.91362 | 0.82349 | 0.31990 | 2.05701 | 0.97096 | 0.89532 | 0.29324 | 2.15952 | 0.73462 | 0.68569 | 0.16021 | 1.58052 | 5.797 | 64.41 |
| 7 | SDSm | [stochastic diffusion search M](https://www.mql5.com/en/articles/13540) | 0.93066 | 0.85445 | 0.39476 | 2.17988 | 0.99983 | 0.89244 | 0.19619 | 2.08846 | 0.72333 | 0.61100 | 0.10670 | 1.44103 | 5.709 | 63.44 |
| 8 | BOAm | [billiards optimization algorithm M](https://www.mql5.com/en/articles/17325) | 0.95757 | 0.82599 | 0.25235 | 2.03590 | 1.00000 | 0.90036 | 0.30502 | 2.20538 | 0.73538 | 0.52523 | 0.09563 | 1.35625 | 5.598 | 62.19 |
| 9 | AAm | [archery algorithm M](https://www.mql5.com/en/articles/15782) | 0.91744 | 0.70876 | 0.42160 | 2.04780 | 0.92527 | 0.75802 | 0.35328 | 2.03657 | 0.67385 | 0.55200 | 0.23738 | 1.46323 | 5.548 | 61.64 |
| 10 | ESG | [evolution of social groups (joo)](https://www.mql5.com/en/articles/14136) | 0.99906 | 0.79654 | 0.35056 | 2.14616 | 1.00000 | 0.82863 | 0.13102 | 1.95965 | 0.82333 | 0.55300 | 0.04725 | 1.42358 | 5.529 | 61.44 |
| 11 | SIA | [simulated isotropic annealing (joo)](https://www.mql5.com/en/articles/13870) | 0.95784 | 0.84264 | 0.41465 | 2.21513 | 0.98239 | 0.79586 | 0.20507 | 1.98332 | 0.68667 | 0.49300 | 0.09053 | 1.27020 | 5.469 | 60.76 |
| 12 | ACS | [artificial cooperative search](https://www.mql5.com/en/articles/15004) | 0.75547 | 0.74744 | 0.30407 | 1.80698 | 1.00000 | 0.88861 | 0.22413 | 2.11274 | 0.69077 | 0.48185 | 0.13322 | 1.30583 | 5.226 | 58.06 |
| 13 | DA | [dialectical algorithm](https://www.mql5.com/en/articles/16999) | 0.86183 | 0.70033 | 0.33724 | 1.89940 | 0.98163 | 0.72772 | 0.28718 | 1.99653 | 0.70308 | 0.45292 | 0.16367 | 1.31967 | 5.216 | 57.95 |
| 14 | BHAm | [black hole algorithm M](https://www.mql5.com/en/articles/16655) | 0.75236 | 0.76675 | 0.34583 | 1.86493 | 0.93593 | 0.80152 | 0.27177 | 2.00923 | 0.65077 | 0.51646 | 0.15472 | 1.32195 | 5.196 | 57.73 |
| 15 | ASO | [anarchy society optimization](https://www.mql5.com/en/articles/15511) | 0.84872 | 0.74646 | 0.31465 | 1.90983 | 0.96148 | 0.79150 | 0.23803 | 1.99101 | 0.57077 | 0.54062 | 0.16614 | 1.27752 | 5.178 | 57.54 |
| 16 | RFO | [royal flush optimization (joo)](https://www.mql5.com/en/articles/17063) | 0.83361 | 0.73742 | 0.34629 | 1.91733 | 0.89424 | 0.73824 | 0.24098 | 1.87346 | 0.63154 | 0.50292 | 0.16421 | 1.29867 | 5.089 | 56.55 |
| 17 | AOSm | [atomic orbital search M](https://www.mql5.com/en/articles/16315) | 0.80232 | 0.70449 | 0.31021 | 1.81702 | 0.85660 | 0.69451 | 0.21996 | 1.77107 | 0.74615 | 0.52862 | 0.14358 | 1.41835 | 5.006 | 55.63 |
| 18 | TSEA | [turtle shell evolution algorithm (joo)](https://www.mql5.com/en/articles/14789) | 0.96798 | 0.64480 | 0.29672 | 1.90949 | 0.99449 | 0.61981 | 0.22708 | 1.84139 | 0.69077 | 0.42646 | 0.13598 | 1.25322 | 5.004 | 55.60 |
| 19 | DE | [differential evolution](https://www.mql5.com/en/articles/13781) | 0.95044 | 0.61674 | 0.30308 | 1.87026 | 0.95317 | 0.78896 | 0.16652 | 1.90865 | 0.78667 | 0.36033 | 0.02953 | 1.17653 | 4.955 | 55.06 |
| 20 | SRA | [successful restaurateur algorithm (joo)](https://www.mql5.com/en/articles/17380) | 0.96883 | 0.63455 | 0.29217 | 1.89555 | 0.94637 | 0.55506 | 0.19124 | 1.69267 | 0.74923 | 0.44031 | 0.12526 | 1.31480 | 4.903 | 54.48 |
| 21 | CRO | [chemical reaction optimization](https://www.mql5.com/en/articles/15080) | 0.94629 | 0.66112 | 0.29853 | 1.90593 | 0.87906 | 0.58422 | 0.21146 | 1.67473 | 0.75846 | 0.42646 | 0.12686 | 1.31178 | 4.892 | 54.36 |
| 22 | BIO | [blood inheritance optimization (joo)](https://www.mql5.com/en/articles/17246) | 0.81568 | 0.65336 | 0.30877 | 1.77781 | 0.89937 | 0.65319 | 0.21760 | 1.77016 | 0.67846 | 0.47631 | 0.13902 | 1.29378 | 4.842 | 53.80 |
| 23 | BSA | [bird swarm algorithm](https://www.mql5.com/en/articles/14491) | 0.89306 | 0.64900 | 0.26250 | 1.80455 | 0.92420 | 0.71121 | 0.24939 | 1.88479 | 0.69385 | 0.32615 | 0.10012 | 1.12012 | 4.809 | 53.44 |
| 24 | HS | [harmony search](https://www.mql5.com/en/articles/12163) | 0.86509 | 0.68782 | 0.32527 | 1.87818 | 0.99999 | 0.68002 | 0.09590 | 1.77592 | 0.62000 | 0.42267 | 0.05458 | 1.09725 | 4.751 | 52.79 |
| 25 | SSG | [saplings sowing and growing](https://www.mql5.com/en/articles/12268) | 0.77839 | 0.64925 | 0.39543 | 1.82308 | 0.85973 | 0.62467 | 0.17429 | 1.65869 | 0.64667 | 0.44133 | 0.10598 | 1.19398 | 4.676 | 51.95 |
| 26 | BCOm | [bacterial chemotaxis optimization M](https://www.mql5.com/en/articles/15711) | 0.75953 | 0.62268 | 0.31483 | 1.69704 | 0.89378 | 0.61339 | 0.22542 | 1.73259 | 0.65385 | 0.42092 | 0.14435 | 1.21912 | 4.649 | 51.65 |
| 27 | ABO | [african buffalo optimization](https://www.mql5.com/en/articles/16024) | 0.83337 | 0.62247 | 0.29964 | 1.75548 | 0.92170 | 0.58618 | 0.19723 | 1.70511 | 0.61000 | 0.43154 | 0.13225 | 1.17378 | 4.634 | 51.49 |
| 28 | (PO)ES | [(PO) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.79025 | 0.62647 | 0.42935 | 1.84606 | 0.87616 | 0.60943 | 0.19591 | 1.68151 | 0.59000 | 0.37933 | 0.11322 | 1.08255 | 4.610 | 51.22 |
| 29 | TSm | [tabu search M](https://www.mql5.com/en/articles/15654) | 0.87795 | 0.61431 | 0.29104 | 1.78330 | 0.92885 | 0.51844 | 0.19054 | 1.63783 | 0.61077 | 0.38215 | 0.12157 | 1.11449 | 4.536 | 50.40 |
| 30 | BSO | [brain storm optimization](https://www.mql5.com/en/articles/14707) | 0.93736 | 0.57616 | 0.29688 | 1.81041 | 0.93131 | 0.55866 | 0.23537 | 1.72534 | 0.55231 | 0.29077 | 0.11914 | 0.96222 | 4.498 | 49.98 |
| 31 | WOAm | [wale optimization algorithm M](https://www.mql5.com/en/articles/14414) | 0.84521 | 0.56298 | 0.26263 | 1.67081 | 0.93100 | 0.52278 | 0.16365 | 1.61743 | 0.66308 | 0.41138 | 0.11357 | 1.18803 | 4.476 | 49.74 |
| 32 | AEFA | [artificial electric field algorithm](https://www.mql5.com/en/articles/15162) | 0.87700 | 0.61753 | 0.25235 | 1.74688 | 0.92729 | 0.72698 | 0.18064 | 1.83490 | 0.66615 | 0.11631 | 0.09508 | 0.87754 | 4.459 | 49.55 |
| 33 | AEO | [artificial ecosystem-based optimization algorithm](https://www.mql5.com/en/articles/16058) | 0.91380 | 0.46713 | 0.26470 | 1.64563 | 0.90223 | 0.43705 | 0.21400 | 1.55327 | 0.66154 | 0.30800 | 0.28563 | 1.25517 | 4.454 | 49.49 |
| 34 | ACOm | [ant colony optimization M](https://www.mql5.com/en/articles/11602) | 0.88190 | 0.66127 | 0.30377 | 1.84693 | 0.85873 | 0.58680 | 0.15051 | 1.59604 | 0.59667 | 0.37333 | 0.02472 | 0.99472 | 4.438 | 49.31 |
| 35 | BFO-GA | [bacterial foraging optimization - ga](https://www.mql5.com/en/articles/14011) | 0.89150 | 0.55111 | 0.31529 | 1.75790 | 0.96982 | 0.39612 | 0.06305 | 1.42899 | 0.72667 | 0.27500 | 0.03525 | 1.03692 | 4.224 | 46.93 |
| 36 | SOA | [simple optimization algorithm](https://www.mql5.com/en/articles/16364) | 0.91520 | 0.46976 | 0.27089 | 1.65585 | 0.89675 | 0.37401 | 0.16984 | 1.44060 | 0.69538 | 0.28031 | 0.10852 | 1.08422 | 4.181 | 46.45 |
| 37 | ABHA | [artificial bee hive algorithm](https://www.mql5.com/en/articles/15347) | 0.84131 | 0.54227 | 0.26304 | 1.64663 | 0.87858 | 0.47779 | 0.17181 | 1.52818 | 0.50923 | 0.33877 | 0.10397 | 0.95197 | 4.127 | 45.85 |
| 38 | ACMO | [atmospheric cloud model optimization](https://www.mql5.com/en/articles/15921) | 0.90321 | 0.48546 | 0.30403 | 1.69270 | 0.80268 | 0.37857 | 0.19178 | 1.37303 | 0.62308 | 0.24400 | 0.10795 | 0.97503 | 4.041 | 44.90 |
| 39 | ADAMm | [adaptive moment estimation M](https://www.mql5.com/en/articles/16443) | 0.88635 | 0.44766 | 0.26613 | 1.60014 | 0.84497 | 0.38493 | 0.16889 | 1.39880 | 0.66154 | 0.27046 | 0.10594 | 1.03794 | 4.037 | 44.85 |
| 40 | CGO | [chaos game optimization](https://www.mql5.com/en/articles/17047) | 0.57256 | 0.37158 | 0.32018 | 1.26432 | 0.61176 | 0.61931 | 0.62161 | 1.85267 | 0.37538 | 0.21923 | 0.19028 | 0.78490 | 3.902 | 43.35 |
| 41 | ATAm | [artificial tribe algorithm M](https://www.mql5.com/en/articles/16588) | 0.71771 | 0.55304 | 0.25235 | 1.52310 | 0.82491 | 0.55904 | 0.20473 | 1.58867 | 0.44000 | 0.18615 | 0.09411 | 0.72026 | 3.832 | 42.58 |
| 42 | ASHA | [artificial showering algorithm](https://www.mql5.com/en/articles/15980) | 0.89686 | 0.40433 | 0.25617 | 1.55737 | 0.80360 | 0.35526 | 0.19160 | 1.35046 | 0.47692 | 0.18123 | 0.09774 | 0.75589 | 3.664 | 40.71 |
| 43 | ASBO | [adaptive social behavior optimization](https://www.mql5.com/en/articles/15347) | 0.76331 | 0.49253 | 0.32619 | 1.58202 | 0.79546 | 0.40035 | 0.26097 | 1.45677 | 0.26462 | 0.17169 | 0.18200 | 0.61831 | 3.657 | 40.63 |
| 44 | MEC | [mind evolutionary computation](https://www.mql5.com/en/articles/13432) | 0.69533 | 0.53376 | 0.32661 | 1.55569 | 0.72464 | 0.33036 | 0.07198 | 1.12698 | 0.52500 | 0.22000 | 0.04198 | 0.78698 | 3.470 | 38.55 |
| 45 | CSA | [circle search algorithm](https://www.mql5.com/en/articles/17143) | 0.66560 | 0.45317 | 0.29126 | 1.41003 | 0.68797 | 0.41397 | 0.20525 | 1.30719 | 0.37538 | 0.23631 | 0.10646 | 0.71815 | 3.435 | 38.17 |
|  | RW | [random walk](https://www.mql5.com/en/articles/16701#tagRW) | 0.48754 | 0.32159 | 0.25781 | 1.06694 | 0.37554 | 0.21944 | 0.15877 | 0.75375 | 0.27969 | 0.14917 | 0.09847 | 0.52734 | 2.348 | 26.09 |

### Summary

Having developed and tested the Successful Restaurateur Algorithm (SRA), I can confidently say that it has proven itself to be a success. Currently, the algorithm ranks 20th in the ranking table, which is quite good for a new concept. While analyzing the results, I noticed some peculiarities in its behavior. For small-dimensional problems, there is a scatter of results. This is especially noticeable in the discrete Megacity function, where the algorithm has a particularly large spread of values. This function is very difficult for algorithms and getting stuck in local extremes is more the rule than the exception.

On high-dimensional problems, SRA shows results that are slightly weaker than expected. This is probably due to the fact that in high-dimensional spaces, the strategy of improving the worst solutions requires more fine-tuning of the temperature and cooling rate parameters.

However, I consider SRA to be a decent algorithm with potential for further improvement. Its culinary metaphor not only makes the algorithm understandable, but also opens the way for intuitive modifications, refinement of the adaptive mutation mechanism, and the possibility of experimenting with various schemes for selecting "donor dishes."

While creating the successful restaurateur algorithm, I sought not so much to achieve superiority over existing optimization methods, but to reveal new conceptual horizons through an original real life metaphor. The results of the study convincingly demonstrate that this approach deserves its place in the ecosystem of metaheuristic algorithms.

The idea of focusing on the worst solution in a population and using it as a basis for experiments — the concept that might seem crazy at first glance — has proven unexpectedly productive. It is precisely this principle of "outsider rehabilitation" that reveals amazing potential for optimization. Like a skilled restaurateur transforming an unpopular dish into a future hit, the algorithm transforms weak solutions into superior ones, using the ingredients of the leaders.

This experience confirms a valuable rule in scientific research: even the most unorthodox ideas, if properly implemented, can bring practical benefits. Unconventional approaches often reveal aspects of a problem that remain unnoticed when using traditional methodologies.

![Tab](https://c.mql5.com/2/123/Tab.png)

__Figure 2. Color gradation of algorithms according to the corresponding tests__

![Chart](https://c.mql5.com/2/123/Chart.png)

_Figure 3. Histogram of algorithm testing results (scale from 0 to 100, the higher the better,_ _where 100 is the maximum possible theoretical result, in the archive there is a script for calculating the rating table)_

**SRA pros and cons:**

Pros:

1. Simple implementation.

2. The results are good.

Cons:

1. No serious downsides.


The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

- github: [https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5](https://www.mql5.com/go?link=https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5 "https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5")

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | #C\_AO.mqh | Include | Parent class of population optimization<br>algorithms |
| 2 | #C\_AO\_enum.mqh | Include | Enumeration of population optimization algorithms |
| 3 | TestFunctions.mqh | Include | Library of test functions |
| 4 | TestStandFunctions.mqh | Include | Test stand function library |
| 5 | Utilities.mqh | Include | Library of auxiliary functions |
| 6 | CalculationTestResults.mqh | Include | Script for calculating results in the comparison table |
| 7 | Testing AOs.mq5 | Script | The unified test stand for all population optimization algorithms |
| 8 | Simple use of population optimization algorithms.mq5 | Script | A simple example of using population optimization algorithms without visualization |
| 9 | Test\_AO\_SRA.mq5 | Script | SRA test stand |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17380](https://www.mql5.com/ru/articles/17380)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17380.zip "Download all attachments in the single ZIP archive")

[SRA.zip](https://www.mql5.com/en/articles/download/17380/SRA.zip "Download SRA.zip")(173.8 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Central Force Optimization (CFO) algorithm](https://www.mql5.com/en/articles/17167)
- [Neuroboids Optimization Algorithm (NOA)](https://www.mql5.com/en/articles/16992)
- [Billiards Optimization Algorithm (BOA)](https://www.mql5.com/en/articles/17325)
- [Chaos Game Optimization (CGO)](https://www.mql5.com/en/articles/17047)
- [Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)
- [Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)

**[Go to discussion](https://www.mql5.com/en/forum/503175)**

![Market Simulation (Part 08): Sockets (II)](https://c.mql5.com/2/120/Simula92o_de_mercado_Parte_08__LOGO.png)[Market Simulation (Part 08): Sockets (II)](https://www.mql5.com/en/articles/12672)

How about creating something practical using sockets? In today's article, we'll start creating a mini-chat. Let's look together at how this is done - it will be very interesting. Please note that the code provided here is for educational purposes only. It should not be used for commercial purposes or in ready-made applications, as it does not provide data transfer security and the content transmitted over the socket can be accessed.

![Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://c.mql5.com/2/118/Neural_Networks_in_Trading_Multi-Task_Learning_Based_on_the_ResNeXt_Model__LOGO.png)[Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)

We continue exploring a multi-task learning framework based on ResNeXt, which is characterized by modularity, high computational efficiency, and the ability to identify stable patterns in data. Using a single encoder and specialized "heads" reduces the risk of model overfitting and improves the quality of forecasts.

![Larry Williams Market Secrets (Part 3): Proving Non-Random Market Behavior with MQL5](https://c.mql5.com/2/189/20510-larry-williams-market-secrets-logo.png)[Larry Williams Market Secrets (Part 3): Proving Non-Random Market Behavior with MQL5](https://www.mql5.com/en/articles/20510)

Explore whether financial markets are truly random by recreating Larry Williams’ market behavior experiments using MQL5. This article demonstrates how simple price-action tests can reveal statistical market biases using a custom Expert Advisor.

![Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://c.mql5.com/2/188/20695-implementing-practical-modules-logo.png)[Implementing Practical Modules from Other Languages in MQL5 (Part 06): Python-Like File IO operations in MQL5](https://www.mql5.com/en/articles/20695)

This article shows how to simplify complex MQL5 file operations by building a Python-style interface for effortless reading and writing. It explains how to recreate Python’s intuitive file-handling patterns through custom functions and classes. The result is a cleaner, more reliable approach to MQL5 file I/O.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=isogwkixsjkduzbnacrziyolwonekees&ssn=1769191430223636314&ssn_dr=0&ssn_sr=0&fv_date=1769191430&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17380&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Successful%20Restaurateur%20Algorithm%20(SRA)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919143048468973&fz_uniq=5071544301688662533&sv=2552)

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