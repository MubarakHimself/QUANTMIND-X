---
title: Atmosphere Clouds Model Optimization (ACMO): Practice
url: https://www.mql5.com/en/articles/15921
categories: Trading, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:56:09.048222
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/15921&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068813509942246911)

MetaTrader 5 / Tester


### **Contents**

1. [Introduction](https://www.mql5.com/en/articles/15921#tag1)
2. [Implementation of the algorithm](https://www.mql5.com/en/articles/15921#tag2)
3. [Test results](https://www.mql5.com/en/articles/15921#tag3)

### Introduction

In the world of science, where technology and nature intersect, a unique idea of a metaheuristic ACMO (Atmospheric Cloud Model Optimization) algorithm for optimizing complex problems has emerged. In the [previous article](https://www.mql5.com/en/articles/15849), we have technically analyzed the implementation of an algorithm that models the process of formation and movement of clouds in the atmosphere based on various meteorological parameters. In the first part, we created a class to manage cloud simulation, containing methods for initialization, cloud movement, updating region properties, and other processes.

We have divided the search space into regions. The initial humidity and pressure values in these regions were determined. We set the parameters of the model, such as: initial entropy, hyperentropy, threshold humidity value for cloud formation, and others. The next step was to generate clouds by selecting an area with high humidity. The cloud center, entropy and hyperentropy were calculated. We updated meteorological parameters of humidity and pressure in regions after cloud generation. Besides, we implemented the movement of clouds into low-pressure areas and subsequent updating of cloud characteristics depending on their movement between regions, as well as cloud dispersal.

What else remains to be done? We need to implement functions for random placement of droplets and their distribution among clouds, complete the rain process and update the global solution, as well as test the model on our test functions with different parameters to evaluate its performance and accuracy. We will make changes to the process of rain and droplet formation to implement a more complete exchange of information about promising regions in the population.

### Implementation of the algorithm

Let's describe the entire meteorological process in the form of a pseudocode, which will allow us to assemble the final version of the algorithm based on it:

1\. In the first epoch, the clouds are placed randomly:

EnCk = EnM0;

HeCk = HeM0;

//------------------------------------------------------------------------------

1.1 Movement of clouds towards regions with lower pressure:

β = deltaP / normP

d = Tck.x - Cck.c

VC = β \* d;

Ck = Ck + VC

change in the number of droplets after movement:

nk = nk × (1 - γ)

change in entropy and hyperentropy:

α = ΔP / ΔPmax;

EnCk = EnCk \* (1 + α)

HeCk = HeCk \* (1 - α)

//------------------------------------------------------------------------------

2\. The process of rain, the falling of drops:

the distribution of droplets between clouds is proportional to the humidity of the region

increase in the number of droplets to those existing in the clouds

//------------------------------------------------------------------------------

3\. Calculating the fitness function for droplets

//------------------------------------------------------------------------------

4\. Update of global solution and minimum pressure in regions where drops fell

//------------------------------------------------------------------------------

5\. Check for cloud decay and creation of new ones to replace those that have decayed in regions above the threshold:

rule of disintegration due to expansion greater than the permissible value (cloud break):

En > 5 \* EnM0\_t

rule of decay at moisture content below critical value (cloud drying):

dCk < dMin

threshold value above which regions may form clouds:

HT = H\_min + λ \* (H\_max - H\_min);

//------------------------------------------------------------------------------

6\. Calculate entropy and hyperentropy for new clouds:

En = EnM0 / (1 + 2.72 ^ (-(8 - 16 \* (t / maxT))))

He = HeM0 / (1 + 2.72 ^ ((8 - 16 \* (t / maxT))))

Let's continue. Let's look at the **Moving** method of the **C\_AO\_ACMO** class. The method performs two operations: **MoveClouds (revision)** is responsible for clouds movement, **RainProcess (revision)** handles rain, which depends on the state of the clouds and the **revision** parameter. The **Moving** method performs two main actions related to cloud dynamics and rain. It encapsulates the logic of how clouds move and interact with the rain process. Thus, the **Moving** method is used to update the state of clouds and rain as part of the weather simulation.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::Moving ()
{
  MoveClouds       (revision);
  RainProcess      (revision);
}
//——————————————————————————————————————————————————————————————————————————————
```

Let's take a closer look at the **MoveClouds** method of the **C\_AO\_ACMO** class:

1\. The first block (if **rev** is **false**): the method creates clouds with random centers. For each cloud and each coordinate:

- A random value of the cloud center is generated in the given ranges (the **RNDfromCI** function).
- The center is adjusted using **SeInDiSp** to normalize the values.
- The index of the region the cloud is located in is determined using **GetRegionIndex**.
- The entropy and initial entropy values for the cloud are set.
- The initial value of hyperentropy is set to **hyperEntropy**.

The method terminates execution.

2\. The second block (if **rev** is equal to **true**):

- If **rev** is equal to **true**, the method begins to search for regions with the lowest pressure.
- Arrays are created to store indices of regions with the lowest humidity **lHind** and to normalize pressure **normP**.

3\. Loop to find the region with the lowest pressure:

- For each **c** cooridnate, the minimum and maximum pressure among all regions is determined.
- The index of the region with the lowest pressure is stored in the **lHind** array.
- The normalized pressure for each coordinate is stored in **normP**.

4\. Cloud movement for each of them and each coordinate:

- If the cloud is already in the region with the lowest pressure, the iteration is skipped.
- A random target region with less pressure is selected.
- The pressure difference between the current and target region is calculated.
- The pressure value is normalized and the **VC** cloud movement speed is calculated.
- The cloud center is updated based on the speed of movement.
- The region index is being updated.
- The cloud entropy is updated taking into account the change in pressure.
- The amount of moisture in the cloud decreases and the hyperentropy is updated, limiting its value to a maximum of 8.

The **MoveClouds** method is responsible for moving clouds to regions of lower pressure, updating their parameters such as entropy and hyperentropy. The method implements a dynamic model, reflecting changes in the atmosphere.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::MoveClouds (bool &rev)
{
  //----------------------------------------------------------------------------
  if (!rev)
  {
    //creating clouds with random centers---------------------------------------
    for (int i = 0; i < cloudsNumber; i++)
    {
      for (int c = 0; c < coords; c++)
      {
        clouds [i].center [c] = u.RNDfromCI (rangeMin [c], rangeMax [c]);
        clouds [i].center [c] = u.SeInDiSp  (clouds [i].center [c], rangeMin [c], rangeMax [c], rangeStep [c]);

        clouds [i].regionIndex [c] = GetRegionIndex (clouds [i].center [c], c);

        clouds [i].entropy      [c] = entropy [c] * EnM0;
        clouds [i].entropyStart [c] = clouds [i].entropy [c];
      }

      clouds [i].hyperEntropy = HeM0;
    }

    return;
  }

  //search for the region with the lowest pressure------------------------------
  int targetRegion = 0;

  int lHind []; //lowest humidity index
  ArrayResize     (lHind, coords);
  ArrayInitialize (lHind, 0);

  double normP [];
  ArrayResize (normP, coords);
  double minP;
  double maxP;

  for (int c = 0; c < coords; c++)
  {
    minP =  DBL_MAX;
    maxP = -DBL_MAX;

    for (int r = 0; r < regionsNumber; r++)
    {
      if (areas [c].regions [r].pressure < areas [c].regions [lHind [c]].pressure)
      {
        lHind [c] = r;
      }

      if (areas [c].regions [r].pressure < minP) minP = areas [c].regions [r].pressure;
      if (areas [c].regions [r].pressure > maxP) maxP = areas [c].regions [r].pressure;
    }

    normP [c] = maxP - minP;
  }

  //moving the cloud to a region with less pressure-----------------------------
  int    clRegIND = 0;
  double deltaP   = 0.0;
  double α        = 0.0; // Entropy factor
  double β        = 0.0; // Atmospheric pressure factor
  double VC       = 0.0; // Cloud velocity
  double d        = 0.0; // Cloud direction

  for (int i = 0; i < cloudsNumber; i++)
  {
    for (int c = 0; c < coords; c++)
    {
      //find a region with lower pressure---------------------------------------
      if (clouds [i].regionIndex [c] == lHind [c]) continue;

      clRegIND = clouds [i].regionIndex [c];

      do targetRegion = u.RNDminusOne (regionsNumber);
      while (areas [c].regions [clRegIND].pressure < areas [c].regions [targetRegion].pressure);

      //------------------------------------------------------------------------
      deltaP = areas [c].regions [clRegIND].pressure - areas [c].regions [targetRegion].pressure;

      β = deltaP / normP [c];
      d = areas [c].regions [targetRegion].x - areas [c].regions [clRegIND].centre;

      VC = β * d;

      clouds [i].center      [c] += VC;
      clouds [i].center      [c] = u.SeInDiSp (clouds [i].center [c], rangeMin [c], rangeMax [c], rangeStep [c]);
      clouds [i].regionIndex [c] = GetRegionIndex (clouds [i].center [c], c);

      α = β;
      clouds [i].entropy [c] *=(1 + α);
    }

    clouds [i].droplets     *=(1 - γ);
    clouds [i].hyperEntropy *=(1 + α);
    if (clouds [i].hyperEntropy > 8) clouds [i].hyperEntropy = 8;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

Next, let's analyze the **GetRegionIndex** method of the **C\_AO\_ACMO** class. Method description:

1\. Calculating the position of a region. The **regPos** region index with the specified **point** is calculated and the **floor** function is used to round down to the nearest integer number.

2\. Checking boundaries. This block checks if the **regPos** calculated index exceeds permissible values (it cannot exceed the permissible number of regions).

3\. The method returns the index of the region the point is located in.

The **GetRegionIndex** method is intended to determine the index of the region, in which a given point is located within a certain range. It takes into account the number of regions and correctly handles cases where the point is on the boundary of the range.

```
//——————————————————————————————————————————————————————————————————————————————
int C_AO_ACMO::GetRegionIndex (double point, int ind)
{
  int regPos = (int)floor ((point - rangeMin [ind]) / ((rangeMax [ind] - rangeMin [ind]) / regionsNumber));

  if (regPos >= regionsNumber) regPos = regionsNumber - 1;

  return regPos;
}
//——————————————————————————————————————————————————————————————————————————————
```

Describe the next method **RainProcess** of the **C\_AO\_ACMO** class:

2\. Initializing arrays:

- Create two arrays: **cloud** for storing cloud values and **drops** for storing the number of raindrops for each cloud.
- Both arrays vary in size depending on the amount of clouds ( **cloudsNumber**).

3\. Initializing the **cloud** array:

- If **rev** is equal to **false**, all **cloud** array elements are initialized using the value of **1.0**.
- Otherwise, the **cloud** array is initialized using the value of **0.0** and then the humidity is calculated for each cloud.

4\. Humidity calculation:

- For each cloud and each coordinate, humidity is calculated depending on the regions.
- If the humidity is not equal to **-DBL\_MAX**, it is added to the corresponding element of the **cloud** array. Otherwise, the **minGp** minimum drop value is added.

5\. Droplet distribution:

- Calling the **DropletsDistribution** method to distribute drops based on values in the **cloud** array.

6\. The main droplet handling loop, for each cloud and each droplet:

- The values of **dist**, **centre**, **xMin** and **xMax** are calculated.
- The **x** value is generated using normal (Gaussian) distribution.
- If **x** is out of range, it is corrected using the **RNDfromCI** method.
- The **x** value is normalized using the **SeInDiSp** method and saved in the **a** array.

After all droplets for a cloud have been handled, the total number of droplets in the cloud is updated. Thus, the **RainProcess** method simulates a rain falling from clouds, taking into account the humidity and distribution of drops. It initializes the arrays, calculates the humidity for each cloud, distributes the raindrops, and generates values for each drop assuming a normal distribution.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::RainProcess (bool &rev)
{
  //to shed drops from every cloud----------------------------------------------
  double cloud [];
  int    drops [];
  ArrayResize (cloud, cloudsNumber);
  ArrayResize (drops, cloudsNumber);

  if (!rev)
  {
    ArrayInitialize (cloud, 1.0);
  }
  else
  {
    ArrayInitialize (cloud, 0.0);

    double humidity;

    for (int i = 0; i < cloudsNumber; i++)
    {
      for (int c = 0; c < coords; c++)
      {
        for (int r = 0; r < regionsNumber; r++)
        {
          humidity = areas [c].regions [clouds [i].regionIndex [r]].humidity;
          if (humidity != -DBL_MAX) cloud [i] += humidity;
          else                      cloud [i] += minGp;
        }
      }
    }
  }

  DropletsDistribution (cloud, drops);

  double dist   = 0.0;
  double centre = 0.0;
  double xMin   = 0.0;
  double xMax   = 0.0;
  double x      = 0.0;
  int    dCNT   = 0;

  for (int i = 0; i < cloudsNumber; i++)
  {
    for (int dr = 0; dr < drops [i]; dr++)
    {
      for (int c = 0; c < coords; c++)
      {
        dist   = clouds [i].entropy [c];
        centre = clouds [i].center  [c];
        xMin   = centre - dist;
        xMax   = centre + dist;

        x = u.GaussDistribution (centre, xMin, xMax, clouds [i].hyperEntropy);

        if (x < rangeMin [c]) x = u.RNDfromCI (rangeMin [c], centre);
        if (x > rangeMax [c]) x = u.RNDfromCI (centre, rangeMax [c]);

        x = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);

        a [dCNT].c [c] = x;
      }

      dCNT++;
    }

    clouds [i].droplets += drops [i];
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The next method **DropletsDistribution** of the **C\_AO\_ACMO** class is designed to distribute raindrops between clouds based on their humidity. Let's have a thorough look at it.

2\. Initialization of variables:

- **minHumidity** is initialized to the maximum value so that the minimum humidity can be found.
- **indMinHumidity** stores the cloud index with minimum humidity.
- **totalHumidity** is used to store the sum of the humidity of all clouds.

3\. Humidity summation sums the humidity of all clouds and determines the cloud with the lowest humidity.

4\. Proportional distribution of drops - for each cloud, the number of drops is calculated proportionally to its humidity in relation to the total humidity. This value is stored in the **droplets** array.

5\. Distribution of remaining drops:

- First, the total number of distributed droplets **totalDrops** is calculated.
- The number of remaining drops ( **remainingDrops**) is then calculated.
- If there are any remaining droplets, they are added to the cloud with minimal humidity.

The **DropletsDistribution** method effectively distributes raindrops between clouds based on their moisture content. It first distributes the droplets proportionally and then adjusts the distribution by adding the remaining droplets to the cloud with the lowest humidity. This allows for a more realistic simulation of the rainfall, while maintaining a constant number of drops, which corresponds to the population size in the external parameters of the algorithm.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::DropletsDistribution (double &cloud [], int &droplets [])
{
  double minHumidity    = DBL_MAX;
  int    indMinHumidity = -1;
  double totalHumidity  = 0; //total amount of humidity in all clouds

  for (int i = 0; i < ArraySize (cloud); i++)
  {
    totalHumidity += cloud [i];

    if (cloud [i] < minHumidity)
    {
      minHumidity = cloud [i];
      indMinHumidity = i;
    }
  }

  // Filling the droplets array in proportion to the value in clouds
  for (int i = 0; i < ArraySize (clouds); i++)
  {
    droplets [i] = int((cloud [i] / totalHumidity)*popSize); //proportional distribution of droplets
  }

  // Distribute the remaining drops, if any
  int totalDrops = 0;

  for (int i = 0; i < ArraySize (droplets); i++)
  {
    totalDrops += droplets [i];
  }

  // If not all drops are distributed, add the remaining drops to the element with the lowest humidity
  int remainingDrops = popSize - totalDrops;

  if (remainingDrops > 0)
  {
    droplets [indMinHumidity] += remainingDrops; //add the remaining drops to the lightest cloud
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **Revision** method of the **C\_AO\_ACMO** class performs a system state update. Let's look at it in more detail:

1\. Loop through **a** array elements (population of optimization agents). This loop iterates through all the elements of the **a** array of **popSize** size:

- If the **f** fitness value of the current element is greater than the current **fB** maximum fitness value, **fB** is updated, while the **ind** index is set to the current index.
- The search for the minimum **f** fitness value is also carried out among all elements of the array, and if the current value is less than **minGp**, then **minGp** is updated.

2\. Copying data: if an element with the maximum **f** fitness value was found ( **ind** is not -1), the data (namely, a \[ind\]) is copied from the **c** array to the **cB** array.

3\. Updating region properties: **UpdateRegionProperties** method is called. It updates humidity and pressure parameters in different regions.

4\. Cloud generation: the **GenerateClouds** method, which is responsible for the disappearance of old clouds and the creation of new ones, is called.

5\. Status update:

- The **revision** flag is set to **true**, which indicates that the initial state of the system has been passed.
- The **epochNow** counter is increased to track the number of epochs.

The **Revision** method is responsible for updating the state of the system related to clouds. It finds the maximum **f** fitness value, updates the relevant parameters, initializes new clouds and updates the region properties. The method is key to keeping the data in the model up-to-date, allowing the system to adapt to changes.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::Revision ()
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

    if (a [i].f < minGp) minGp = a [i].f;
  }

  if (ind != -1) ArrayCopy (cB, a [ind].c, 0, 0, WHOLE_ARRAY);

  //----------------------------------------------------------------------------
  UpdateRegionProperties (); //updating humidity and pressure in the regions
  GenerateClouds         (); //disappearance of clouds and the creation of new ones

  revision = true;
  epochNow++;
}
//——————————————————————————————————————————————————————————————————————————————
```

The **GenerateClouds** method of the **C\_AO\_ACMO** class is responsible for creating clouds and managing their state depending on various factors, such as humidity and entropy. Method description:

1\. Humidity threshold calculation: the **CalculateHumidityThreshold** function is called, which returns the humidity threshold value required for clouds to form.

2\. Structure for storing region indices:

- The **S\_Areas** structure is defined. The structure contains an array of indices of regions capable of forming clouds.
- The **ar** method is initialized with a size equal to the **coords** number of coordinates.


3\. Region information collection: This double loop tests each region to see if it meets its humidity threshold. If the humidity of a region is greater than the threshold, the index of that region is added to the **regsIND** array of the appropriate **S\_Areas** structure.

4\. Checking cloud decay conditions:

- For each cloud, it is checked whether its entropy exceeds a certain limit (5 times the initial entropy). If this is the case, the cloud is considered to have disintegrated.
- Then it is checked if the amount of moisture in the cloud is less than the minimum value of **dMin**, which may also lead to the cloud disintegration.

5\. Creating a new cloud in the wettest regions:

- If the cloud disintegrates, a new cloud is created in one of the wettest regions. For each coordinate, a region index is randomly selected, and the cloud receives new center coordinates and a region index.
- The **CalculateNewEntropy** function is then called. The function recalculates the entropy for a new cloud depending on the current epoch.

The **GenerateClouds** method manages cloud creation and disintegration based on humidity and entropy. It collects information about regions capable of forming clouds, checks existing clouds for decay, and creates new clouds in suitable regions. This method is key to dynamically controlling the state of clouds in the model.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::GenerateClouds ()
{
  //Collecting statistics of regions capable of creating clouds-----------------
  double Ht = CalculateHumidityThreshold ();

  struct S_Areas
  {
      int regsIND []; //index of the potential region
  };

  S_Areas ar [];
  ArrayResize (ar, coords);

  int sizePr = 0;

  for (int i = 0; i < coords; i++)
  {
    for (int r = 0; r < regionsNumber; r++)
    {
      if (areas [i].regions [r].humidity > Ht)
      {
        sizePr = ArraySize (ar [i].regsIND);
        sizePr++;
        ArrayResize (ar [i].regsIND, sizePr, coords);
        ar [i].regsIND [sizePr - 1] = r;
      }
    }
  }

  //Check the conditions for cloud decay----------------------------------------
  bool   cloudDecay = false;

  for (int i = 0; i < cloudsNumber; i++)
  {
    cloudDecay = false;

    //checking the cloud for too much entropy-----------------------------------
    for (int c = 0; c < coords; c++)
    {
      if (clouds [i].entropy [c] > 5.0 * clouds [i].entropyStart [c])
      {
        //Print ("Disintegration of cloud #", i, " - tore at epoch ", epochNow);
        cloudDecay = true;
        break;
      }
    }

    //checking the cloud for decay----------------------------------------------
    if (!cloudDecay)
    {
      if (clouds [i].droplets < dMin)
      {
        //Print ("Disintegration of cloud #", i, " - dried up at epoch ", epochNow);
        cloudDecay = true;
      }
    }

    //if the cloud has decayed--------------------------------------------------
    int regIND = 0;

    if (cloudDecay)
    {
      //creating a cloud in a very humid region---------------------------------
      for (int c = 0; c < coords; c++)
      {
        regIND = u.RNDminusOne (ArraySize (ar [c].regsIND));
        regIND = ar [c].regsIND [regIND];

        clouds [i].center      [c] = areas [c].regions [regIND].x;
        clouds [i].regionIndex [c] = regIND;
      }

      CalculateNewEntropy (clouds [i], epochNow);
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The **CalculateHumidityThreshold** method of the **C\_AO\_ACMO** class is responsible for calculating the humidity threshold required for cloud formation. Here are the steps in detail:

1\. Double cycle to find minimum humidity. The outer loop goes through all **coords** coordinates, while the inner loop iterates through all regions ( **regionsNumber**) in each coordinate. If the humidity of the region is not equal to **-DBL\_MAX**, a check is performed: if the current humidity is less than the current **H\_min**, **H\_min** is updated.

2\. The method returns **H\_min** increased by the product of **λ** and the difference of **H\_max** and **H\_min**, which represents the moisture threshold required for clouds to form.

The **CalculateHumidityThreshold** method calculates the humidity threshold based on the minimum humidity among all regions and adjusts it based on the maximum humidity and the **λ** ratio. This makes it possible to determine under what conditions clouds can form, based on the state of the environment.

```
//——————————————————————————————————————————————————————————————————————————————
double C_AO_ACMO::CalculateHumidityThreshold ()
{
  double H_max = fB;
  double H_min = DBL_MAX;

  for (int c = 0; c < coords; c++)
  {
    for (int r = 0; r < regionsNumber; r++)
    {
      if (areas [c].regions [r].humidity != -DBL_MAX)
      {
        if (areas [c].regions [r].humidity < H_min)
        {
          H_min = areas [c].regions [r].humidity;
        }
      }
    }
  }

  return H_min + λ * (H_max - H_min);
}
//——————————————————————————————————————————————————————————————————————————————
```

The **CalculateNewEntropy** method of the **C\_AO\_ACMO** class is responsible for calculating the new entropy and hyperentropy for clouds represented by the **S\_ACMO\_Cloud** structure. Let's take a detailed look at it:

1\. Calculating entropy:

- The cycle goes through all **coords** coordinates.
- For each coordinate, a new entropy value is calculated " **cl.entropy \[c**\]", using the equation: **En = (entropy \[c\] \* EnM0) / (1 + e ^ (-(8 - 16 \* (t / epochs)))).**
- **cl.entropyStart \[c\]** and **cl.entropy \[c\]** are initialized with the value of **entropy \[c\]**, which serves to preserve the initial value of entropy.

2\. Calculating hyperentropy: **He = 1 / (1 + e ^ (8 - 16 \* (t / epochs))).**

3\. Hyperentropy is scaled using the **Scale** method of the **u** object, which allows us to scale the hyperentropy value to a given range (from 0 to 8) using the **HeM0** parameters and 8.0.

The **CalculateNewEntropy** method updates entropy and hyperentropy values for clouds based on the current time **t** and the specified parameters.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::CalculateNewEntropy (S_ACMO_Cloud &cl, int t)
{
  //----------------------------------------------------------------------------
  //En: 1/(1+2.72^(-(8-16*(t/maxT))))
  for (int c = 0; c < coords; c++)
  {
    cl.entropy      [c] = entropy [c] * EnM0 / (1.0 + pow (M_E, (-(8.0 - 16.0 * (t / epochs)))));
    cl.entropyStart [c] = cl.entropy [c] = entropy [c];
  }

  //----------------------------------------------------------------------------
  //He: 1/(1+2.72^((8-16*(t/maxT))))
  cl.hyperEntropy = 1.0 / (1.0 + pow (M_E, ((8.0 - 16.0 * (t / epochs)))));

  cl.hyperEntropy = u.Scale (cl.hyperEntropy, 0.0, 8.0, HeM0, 8.0);
}
//——————————————————————————————————————————————————————————————————————————————
```

![En and He 2](https://c.mql5.com/2/137/En_and_He_2__1.png)

_Figure 1. Variants of equations for calculating the **ζ** ratio depending on the current epoch **.**_ _We can select an equation and try out the algorithm with each of them (the code strings are commented out)_

### Test results

Let's move on to testing the algorithm. The meteorological model of cloud formation, as conceived by the authors, works as follows:

//original version

ACMO\|Atmospheric Cloud Model Optimization\|50.0\|5.0\|10.0\|0.2\|5.0\|5.0\|0.9\|0.2\|

=============================

5 Hilly's; Func runs: 10000; result: 0.6017884495404766

25 Hilly's; Func runs: 10000; result: 0.3426222382089618

500 Hilly's; Func runs: 10000; result: 0.2526410178225118

=============================

5 Forest's; Func runs: 10000; result: 0.4780554376190664

25 Forest's; Func runs: 10000; result: 0.261057831391174

500 Forest's; Func runs: 10000; result: 0.17318135866144563

=============================

5 Megacity's; Func runs: 10000; result: 0.3507692307692307

25 Megacity's; Func runs: 10000; result: 0.16153846153846158

500 Megacity's; Func runs: 10000; result: 0.09632307692307775

=============================

All score: 2.71798 (30.20%)

Unfortunately, the results are much lower than expected. I think that despite the beautiful model of the cloud formation principle with a large number of different equations and logical actions aimed at avoiding getting stuck in extremes, the convergence of the algorithm is low. The algorithm lacks direct interaction and exchange of information between agents about the best solutions, the presence of which usually leads to an improvement in the search qualities of any algorithm. As a result, I decided to add information exchange through probabilistic transmission of information from the best drops to the worst ones. Now let's see what came out of this:

ACMOm\|Atmospheric Cloud Model Optimization\|50.0\|4.0\|10.0\|0.2\|0.2\|2.0\|0.9\|0.9\|

=============================

5 Hilly's; Func runs: 10000; result: 0.9032099148349984

25 Hilly's; Func runs: 10000; result: 0.48545807643133143

500 Hilly's; Func runs: 10000; result: 0.30403284557071203

=============================

5 Forest's; Func runs: 10000; result: 0.8026793420899985

25 Forest's; Func runs: 10000; result: 0.3785708322859447

500 Forest's; Func runs: 10000; result: 0.1917777390119122

=============================

5 Megacity's; Func runs: 10000; result: 0.6230769230769231

25 Megacity's; Func runs: 10000; result: 0.244

500 Megacity's; Func runs: 10000; result: 0.10795384615384714

=============================

All score: 4.04076 (44.90%)

The results have improved significantly. The idea was successful. The idea is to transmit information about the best solution with some probability to cloud drops from other drops if they have higher humidity (in the context of the algorithm, this is fitness).

At the end, for each cloud, the total number of drops (for a cloud, this is an indicator of humidity) dropped from it is updated by adding the corresponding value from the **drops** array. The **RainProcess** method implements a mechanism that models the rain, taking into account humidity, droplet distribution, and interaction with the population. The location where changes were made to the code is highlighted in green.

For each generated **x** value, a random **p** index from the population is selected. Depending on the probability (95%), the values in the **a** array are updated. The value represents a population or a set of solutions. At the end, the total number of drops dropped from the cloud is updated for each cloud by adding the corresponding value from the **drops** array.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_ACMO::RainProcess (bool &rev)
{
  //to shed drops from every cloud----------------------------------------------
  double cloud [];
  int    drops [];
  ArrayResize (cloud, cloudsNumber);
  ArrayResize (drops, cloudsNumber);

  if (!rev)
  {
    ArrayInitialize (cloud, 1.0);
  }
  else
  {
    ArrayInitialize (cloud, 0.0);

    double humidity;

    for (int i = 0; i < cloudsNumber; i++)
    {
      for (int c = 0; c < coords; c++)
      {
        humidity = areas [c].regions [clouds [i].regionIndex [c]].humidity;

        if (humidity != -DBL_MAX) cloud [i] += humidity;

        else                      cloud [i] += minGp;
      }
    }
  }

  DropletsDistribution (cloud, drops);
  //ArrayPrint (drops);

  double dist   = 0.0;
  double centre = 0.0;
  double xMin   = 0.0;
  double xMax   = 0.0;
  double x      = 0.0;

  int    dCNT   = 0;

  for (int i = 0; i < cloudsNumber; i++)
  {
    for (int dr = 0; dr < drops [i]; dr++)
    {
      for (int c = 0; c < coords; c++)
      {
        dist   = clouds [i].entropy [c];
        centre = clouds [i].center  [c];
        xMin   = centre - dist;
        xMax   = centre + dist;

        x = u.GaussDistribution (centre, xMin, xMax, clouds [i].hyperEntropy);

        if (x < rangeMin [c]) x = u.RNDfromCI (rangeMin [c], centre);

        if (x > rangeMax [c]) x = u.RNDfromCI (centre, rangeMax [c]);

        x = u.SeInDiSp (x, rangeMin [c], rangeMax [c], rangeStep [c]);

        int p = u.RNDminusOne (popSize);

        if (a [p].f > a [dCNT].f)
        {
          if (u.RNDprobab () < 0.95) a [dCNT].c [c] = a [p].c [c];
        }

        else
        {
          a [dCNT].c [c] = x;
        }
      }
      dCNT++;
    }

    clouds [i].droplets += drops [i];
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The visualization of the algorithm's operation shows the following: good convergence of the algorithm, long flat sections on the convergence graph with a small number of optimized parameters indicate some tendency for the algorithm to get stuck in local extremes, and with an increase in the number of parameters, this disadvantage disappears.

The clouds in the visualization appear as dense clusters, but by choosing different settings for external parameters (number of regions, clouds, initial entropy, and drying threshold), they can simulate the appearance of floating clouds in the sky, as in nature.

![Hilly](https://c.mql5.com/2/137/Hilly__3.gif)

AСMO on the [Hilly](https://www.mql5.com/en/articles/13923#tagHilly) test function

![Forest](https://c.mql5.com/2/137/Forest__3.gif)

ACMO on the [Forest](https://www.mql5.com/en/articles/11785#tag3) test function

![Megacity](https://c.mql5.com/2/137/Megacity__3.gif)

ACMO on the [Megacity](https://www.mql5.com/en/articles/11785#tag3) test function

According to the results of testing the modified version, the algorithm ranks 27 th, which is a fairly stable indicator. I would like to emphasize that the table now always contains 45 algorithms and, in fact, with each new algorithm the difference between the previous and subsequent ones will gradually decrease, thus, we can say that the table represents the top known algorithms.

|     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| # | AO | Description | Hilly | Hilly final | Forest | Forest final | Megacity (discrete) | Megacity final | Final result | % of MAX |
| 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) |
| 1 | ANS | [across neighbourhood search](https://www.mql5.com/en/articles/15049) | 0.94948 | 0.84776 | 0.43857 | 2.23581 | 1.00000 | 0.92334 | 0.39988 | 2.32323 | 0.70923 | 0.63477 | 0.23091 | 1.57491 | 6.134 | 68.15 |
| 2 | CLA | [code lock algorithm](https://www.mql5.com/en/articles/14878) | 0.95345 | 0.87107 | 0.37590 | 2.20042 | 0.98942 | 0.91709 | 0.31642 | 2.22294 | 0.79692 | 0.69385 | 0.19303 | 1.68380 | 6.107 | 67.86 |
| 3 | AMOm | [animal migration ptimization M](https://www.mql5.com/en/articles/15543) | 0.90358 | 0.84317 | 0.46284 | 2.20959 | 0.99001 | 0.92436 | 0.46598 | 2.38034 | 0.56769 | 0.59132 | 0.23773 | 1.39675 | 5.987 | 66.52 |
| 4 | (P+O)ES | [(P+O) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.92256 | 0.88101 | 0.40021 | 2.20379 | 0.97750 | 0.87490 | 0.31945 | 2.17185 | 0.67385 | 0.62985 | 0.18634 | 1.49003 | 5.866 | 65.17 |
| 5 | CTA | [comet tail algorithm](https://www.mql5.com/en/articles/14841) | 0.95346 | 0.86319 | 0.27770 | 2.09435 | 0.99794 | 0.85740 | 0.33949 | 2.19484 | 0.88769 | 0.56431 | 0.10512 | 1.55712 | 5.846 | 64.96 |
| 6 | SDSm | [stochastic diffusion search M](https://www.mql5.com/en/articles/13540) | 0.93066 | 0.85445 | 0.39476 | 2.17988 | 0.99983 | 0.89244 | 0.19619 | 2.08846 | 0.72333 | 0.61100 | 0.10670 | 1.44103 | 5.709 | 63.44 |
| 7 | AAm | [archery algorithm M](https://www.mql5.com/en/articles/15782) | 0.91744 | 0.70876 | 0.42160 | 2.04780 | 0.92527 | 0.75802 | 0.35328 | 2.03657 | 0.67385 | 0.55200 | 0.23738 | 1.46323 | 5.548 | 61.64 |
| 8 | ESG | [evolution of social groups](https://www.mql5.com/en/articles/14136) | 0.99906 | 0.79654 | 0.35056 | 2.14616 | 1.00000 | 0.82863 | 0.13102 | 1.95965 | 0.82333 | 0.55300 | 0.04725 | 1.42358 | 5.529 | 61.44 |
| 9 | SIA | [simulated isotropic annealing](https://www.mql5.com/en/articles/13870) | 0.95784 | 0.84264 | 0.41465 | 2.21513 | 0.98239 | 0.79586 | 0.20507 | 1.98332 | 0.68667 | 0.49300 | 0.09053 | 1.27020 | 5.469 | 60.76 |
| 10 | ACS | [artificial cooperative search](https://www.mql5.com/en/articles/15004) | 0.75547 | 0.74744 | 0.30407 | 1.80698 | 1.00000 | 0.88861 | 0.22413 | 2.11274 | 0.69077 | 0.48185 | 0.13322 | 1.30583 | 5.226 | 58.06 |
| 11 | ASO | [anarchy society optimization](https://www.mql5.com/en/articles/15511) | 0.84872 | 0.74646 | 0.31465 | 1.90983 | 0.96148 | 0.79150 | 0.23803 | 1.99101 | 0.57077 | 0.54062 | 0.16614 | 1.27752 | 5.178 | 57.54 |
| 12 | TSEA | [turtle shell evolution algorithm](https://www.mql5.com/en/articles/14789) | 0.96798 | 0.64480 | 0.29672 | 1.90949 | 0.99449 | 0.61981 | 0.22708 | 1.84139 | 0.69077 | 0.42646 | 0.13598 | 1.25322 | 5.004 | 55.60 |
| 13 | DE | [differential evolution](https://www.mql5.com/en/articles/13781) | 0.95044 | 0.61674 | 0.30308 | 1.87026 | 0.95317 | 0.78896 | 0.16652 | 1.90865 | 0.78667 | 0.36033 | 0.02953 | 1.17653 | 4.955 | 55.06 |
| 14 | CRO | [chemical reaction optimization](https://www.mql5.com/en/articles/15080) | 0.94629 | 0.66112 | 0.29853 | 1.90593 | 0.87906 | 0.58422 | 0.21146 | 1.67473 | 0.75846 | 0.42646 | 0.12686 | 1.31178 | 4.892 | 54.36 |
| 15 | BSA | [bird swarm algorithm](https://www.mql5.com/en/articles/14491) | 0.89306 | 0.64900 | 0.26250 | 1.80455 | 0.92420 | 0.71121 | 0.24939 | 1.88479 | 0.69385 | 0.32615 | 0.10012 | 1.12012 | 4.809 | 53.44 |
| 16 | HS | [harmony search](https://www.mql5.com/en/articles/12163) | 0.86509 | 0.68782 | 0.32527 | 1.87818 | 0.99999 | 0.68002 | 0.09590 | 1.77592 | 0.62000 | 0.42267 | 0.05458 | 1.09725 | 4.751 | 52.79 |
| 17 | SSG | [saplings sowing and growing](https://www.mql5.com/en/articles/12268) | 0.77839 | 0.64925 | 0.39543 | 1.82308 | 0.85973 | 0.62467 | 0.17429 | 1.65869 | 0.64667 | 0.44133 | 0.10598 | 1.19398 | 4.676 | 51.95 |
| 18 | BCOm | [bacterial chemotaxis optimization M](https://www.mql5.com/en/articles/15711) | 0.75953 | 0.62268 | 0.31483 | 1.69704 | 0.89378 | 0.61339 | 0.22542 | 1.73259 | 0.65385 | 0.42092 | 0.14435 | 1.21912 | 4.649 | 51.65 |
| 19 | (PO)ES | [(PO) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.79025 | 0.62647 | 0.42935 | 1.84606 | 0.87616 | 0.60943 | 0.19591 | 1.68151 | 0.59000 | 0.37933 | 0.11322 | 1.08255 | 4.610 | 51.22 |
| 20 | TSm | [tabu search M](https://www.mql5.com/en/articles/15654) | 0.87795 | 0.61431 | 0.29104 | 1.78330 | 0.92885 | 0.51844 | 0.19054 | 1.63783 | 0.61077 | 0.38215 | 0.12157 | 1.11449 | 4.536 | 50.40 |
| 21 | BSO | [brain storm optimization](https://www.mql5.com/en/articles/14707) | 0.93736 | 0.57616 | 0.29688 | 1.81041 | 0.93131 | 0.55866 | 0.23537 | 1.72534 | 0.55231 | 0.29077 | 0.11914 | 0.96222 | 4.498 | 49.98 |
| 22 | WOAm | [wale optimization algorithm M](https://www.mql5.com/en/articles/14414) | 0.84521 | 0.56298 | 0.26263 | 1.67081 | 0.93100 | 0.52278 | 0.16365 | 1.61743 | 0.66308 | 0.41138 | 0.11357 | 1.18803 | 4.476 | 49.74 |
| 23 | AEFA | [artificial electric field algorithm](https://www.mql5.com/en/articles/15162) | 0.87700 | 0.61753 | 0.25235 | 1.74688 | 0.92729 | 0.72698 | 0.18064 | 1.83490 | 0.66615 | 0.11631 | 0.09508 | 0.87754 | 4.459 | 49.55 |
| 24 | ACOm | [ant colony optimization M](https://www.mql5.com/en/articles/11602) | 0.88190 | 0.66127 | 0.30377 | 1.84693 | 0.85873 | 0.58680 | 0.15051 | 1.59604 | 0.59667 | 0.37333 | 0.02472 | 0.99472 | 4.438 | 49.31 |
| 25 | BFO-GA | [bacterial foraging optimization - ga](https://www.mql5.com/en/articles/14011) | 0.89150 | 0.55111 | 0.31529 | 1.75790 | 0.96982 | 0.39612 | 0.06305 | 1.42899 | 0.72667 | 0.27500 | 0.03525 | 1.03692 | 4.224 | 46.93 |
| 26 | ABHA | [artificial bee hive algorithm](https://www.mql5.com/en/articles/15347) | 0.84131 | 0.54227 | 0.26304 | 1.64663 | 0.87858 | 0.47779 | 0.17181 | 1.52818 | 0.50923 | 0.33877 | 0.10397 | 0.95197 | 4.127 | 45.85 |
| 27 | ACMO | [atmospheric cloud model optimization](https://www.mql5.com/en/articles/15921) | 0.90321 | 0.48546 | 0.30403 | 1.69270 | 0.80268 | 0.37857 | 0.19178 | 1.37303 | 0.62308 | 0.24400 | 0.10795 | 0.97503 | 4.041 | 44.90 |
| 28 | ASBO | [adaptive social behavior optimization](https://www.mql5.com/en/articles/15347) | 0.76331 | 0.49253 | 0.32619 | 1.58202 | 0.79546 | 0.40035 | 0.26097 | 1.45677 | 0.26462 | 0.17169 | 0.18200 | 0.61831 | 3.657 | 40.63 |
| 29 | MEC | [mind evolutionary computation](https://www.mql5.com/en/articles/13432) | 0.69533 | 0.53376 | 0.32661 | 1.55569 | 0.72464 | 0.33036 | 0.07198 | 1.12698 | 0.52500 | 0.22000 | 0.04198 | 0.78698 | 3.470 | 38.55 |
| 30 | IWO | [invasive weed optimization](https://www.mql5.com/en/articles/11990) | 0.72679 | 0.52256 | 0.33123 | 1.58058 | 0.70756 | 0.33955 | 0.07484 | 1.12196 | 0.42333 | 0.23067 | 0.04617 | 0.70017 | 3.403 | 37.81 |
| 31 | Micro-AIS | [micro artificial immune system](https://www.mql5.com/en/articles/13951) | 0.79547 | 0.51922 | 0.30861 | 1.62330 | 0.72956 | 0.36879 | 0.09398 | 1.19233 | 0.37667 | 0.15867 | 0.02802 | 0.56335 | 3.379 | 37.54 |
| 32 | COAm | [cuckoo optimization algorithm M](https://www.mql5.com/en/articles/11786) | 0.75820 | 0.48652 | 0.31369 | 1.55841 | 0.74054 | 0.28051 | 0.05599 | 1.07704 | 0.50500 | 0.17467 | 0.03380 | 0.71347 | 3.349 | 37.21 |
| 33 | SDOm | [spiral dynamics optimization M](https://www.mql5.com/en/articles/12252) | 0.74601 | 0.44623 | 0.29687 | 1.48912 | 0.70204 | 0.34678 | 0.10944 | 1.15826 | 0.42833 | 0.16767 | 0.03663 | 0.63263 | 3.280 | 36.44 |
| 34 | NMm | [Nelder-Mead method M](https://www.mql5.com/en/articles/13805) | 0.73807 | 0.50598 | 0.31342 | 1.55747 | 0.63674 | 0.28302 | 0.08221 | 1.00197 | 0.44667 | 0.18667 | 0.04028 | 0.67362 | 3.233 | 35.92 |
| 35 | FAm | [firefly algorithm M](https://www.mql5.com/en/articles/11873) | 0.58634 | 0.47228 | 0.32276 | 1.38138 | 0.68467 | 0.37439 | 0.10908 | 1.16814 | 0.28667 | 0.16467 | 0.04722 | 0.49855 | 3.048 | 33.87 |
| 36 | GSA | [gravitational search algorithm](https://www.mql5.com/en/articles/12072) | 0.64757 | 0.49197 | 0.30062 | 1.44016 | 0.53962 | 0.36353 | 0.09945 | 1.00260 | 0.32667 | 0.12200 | 0.01917 | 0.46783 | 2.911 | 32.34 |
| 37 | BFO | [bacterial foraging optimization](https://www.mql5.com/en/articles/12031) | 0.61171 | 0.43270 | 0.31318 | 1.35759 | 0.54410 | 0.21511 | 0.05676 | 0.81597 | 0.42167 | 0.13800 | 0.03195 | 0.59162 | 2.765 | 30.72 |
| 38 | ABC | [artificial bee colony](https://www.mql5.com/en/articles/11736) | 0.63377 | 0.42402 | 0.30892 | 1.36671 | 0.55103 | 0.21874 | 0.05623 | 0.82600 | 0.34000 | 0.14200 | 0.03102 | 0.51302 | 2.706 | 30.06 |
| 39 | BA | [bat algorithm](https://www.mql5.com/en/articles/11915) | 0.59761 | 0.45911 | 0.35242 | 1.40915 | 0.40321 | 0.19313 | 0.07175 | 0.66810 | 0.21000 | 0.10100 | 0.03517 | 0.34617 | 2.423 | 26.93 |
| 40 | AAA | [algae adaptive algorithm](https://www.mql5.com/en/articles/15565) | 0.50007 | 0.32040 | 0.25525 | 1.07572 | 0.37021 | 0.22284 | 0.16785 | 0.76089 | 0.27846 | 0.14800 | 0.09755 | 0.52402 | 2.361 | 26.23 |
| 41 | SA | [simulated annealing](https://www.mql5.com/en/articles/13851) | 0.55787 | 0.42177 | 0.31549 | 1.29513 | 0.34998 | 0.15259 | 0.05023 | 0.55280 | 0.31167 | 0.10033 | 0.02883 | 0.44083 | 2.289 | 25.43 |
| 42 | IWDm | [intelligent water drops M](https://www.mql5.com/en/articles/13730) | 0.54501 | 0.37897 | 0.30124 | 1.22522 | 0.46104 | 0.14704 | 0.04369 | 0.65177 | 0.25833 | 0.09700 | 0.02308 | 0.37842 | 2.255 | 25.06 |
| 43 | PSO | [particle swarm optimisation](https://www.mql5.com/en/articles/11386) | 0.59726 | 0.36923 | 0.29928 | 1.26577 | 0.37237 | 0.16324 | 0.07010 | 0.60572 | 0.25667 | 0.08000 | 0.02157 | 0.35823 | 2.230 | 24.77 |
| 44 | Boids | [boids algorithm](https://www.mql5.com/en/articles/14576) | 0.43340 | 0.30581 | 0.25425 | 0.99346 | 0.35718 | 0.20160 | 0.15708 | 0.71586 | 0.27846 | 0.14277 | 0.09834 | 0.51957 | 2.229 | 24.77 |
| 45 | MA | [monkey algorithm](https://www.mql5.com/en/articles/12212) | 0.59107 | 0.42681 | 0.31816 | 1.33604 | 0.31138 | 0.14069 | 0.06612 | 0.51819 | 0.22833 | 0.08567 | 0.02790 | 0.34190 | 2.196 | 24.40 |

### **Summary**

Two versions of the algorithm are presented: original and modified. The latter involves small changes but significantly increases performance due to the exchange of information within the population. This demonstrates that even minor adjustments to the algorithm's logic can lead to significant increases in efficiency across a variety of tasks.

I really liked the idea of the algorithm, as it is aimed at avoiding getting stuck in extremes. The algorithm uses complex multi-stage logic of moving clouds from high pressure zones to low pressure zones and precipitation. However, this was not sufficient to achieve high convergence. As a result, I attempted a modification by introducing information exchange in the population, which helped improve convergence, which is one of the key aspects of any optimization algorithm.

The peculiarity of the algorithm is that no cloud stays in one place for too long. The ever-increasing pressure in the region inevitably pushes the cloud into a new, unexplored area. This mechanism was conceived by the authors as a means of counteracting getting stuck in local extremes. However, trying to improve the convergence of the algorithm increased the probability of getting stuck, which, unfortunately, partially negated the key feature that inspired me to apply this approach to other optimization algorithms. The design of any optimization algorithm is always associated with finding a compromise between resistance to getting stuck and finding an exact solution. If desired, the probability of information exchange can be reduced in the code. At the moment it is 95%, which will increase the stability.

The algorithm is a wonderful base and a set of interesting techniques (rules for the formation of humidity in regions and the distribution of pressure between them, and additionally physical laws of acceleration and inertia, depending on the mass of clouds and many other ideas can be applied) and is a real boon for researchers.

![Tab](https://c.mql5.com/2/137/Tab__3.png)

_Figure 2. Histogram of algorithm testing results (scale from 0 to 100, the higher the better,_ _where 100 is the maximum possible theoretical result, in the archive there is a script for calculating the rating table)_

![chart](https://c.mql5.com/2/137/chart__3.png)

__Figure 3. Color gradation of algorithms according to relevant tests Results greater than or equal to_ _0.99_ are highlighted in white_

**ACMO pros and cons:**

Pros:

1. Built-in mechanisms against getting stuck.

2. Relatively good convergence.
3. Relatively good scalability.


Cons:

1. A huge number of external parameters.
2. Complex implementation.
3. Difficulty in finding the right balance between getting stuck and converging.


The article is accompanied by an archive with the current versions of the algorithm codes. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

- github: [https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5](https://www.mql5.com/go?link=https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5 "https://github.com/JQSakaJoo/Population-optimization-algorithms-MQL5")

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/15921](https://www.mql5.com/ru/articles/15921)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/15921.zip "Download all attachments in the single ZIP archive")

[ACMO.zip](https://www.mql5.com/en/articles/download/15921/acmo.zip "Download ACMO.zip")(39.7 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/485463)**
(2)


![Arda Kaya](https://c.mql5.com/avatar/2025/4/6812859d-b06b.png)

**[Arda Kaya](https://www.mql5.com/en/users/fxtrader1997)**
\|
24 Apr 2025 at 16:13

Really interesting topic!


![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
28 Apr 2025 at 18:02

**Arda Kaya [#](https://www.mql5.com/en/forum/485463#comment_56541022):**

Really interesting topic!

Thank you, with pleasure.

![Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization](https://c.mql5.com/2/137/logo-17865.png)[Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization](https://www.mql5.com/en/articles/17865)

In this article, we explore the automation of the Cypher harmonic pattern in MQL5, detailing its detection and visualization on MetaTrader 5 charts. We implement an Expert Advisor that identifies swing points, validates Fibonacci-based patterns, and executes trades with clear graphical annotations. The article concludes with guidance on backtesting and optimizing the program for effective trading.

![Neural Networks in Trading: Exploring the Local Structure of Data](https://c.mql5.com/2/94/Neural_Networks_in_Trading__Studying_Local_Data_Structure____LOGO__1.png)[Neural Networks in Trading: Exploring the Local Structure of Data](https://www.mql5.com/en/articles/15882)

Effective identification and preservation of the local structure of market data in noisy conditions is a critical task in trading. The use of the Self-Attention mechanism has shown promising results in processing such data; however, the classical approach does not account for the local characteristics of the underlying structure. In this article, I introduce an algorithm capable of incorporating these structural dependencies.

![Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://c.mql5.com/2/137/websockets.png)[Websockets for MetaTrader 5: Asynchronous client connections with the Windows API](https://www.mql5.com/en/articles/17877)

This article details the development of a custom dynamically linked library designed to facilitate asynchronous websocket client connections for MetaTrader programs.

![Data Science and ML (Part 36): Dealing with Biased Financial Markets](https://c.mql5.com/2/136/Data-Science-and-ML-Part-36-logo.png)[Data Science and ML (Part 36): Dealing with Biased Financial Markets](https://www.mql5.com/en/articles/17736)

Financial markets are not perfectly balanced. Some markets are bullish, some are bearish, and some exhibit some ranging behaviors indicating uncertainty in either direction, this unbalanced information when used to train machine learning models can be misleading as the markets change frequently. In this article, we are going to discuss several ways to tackle this issue.

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/15921&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068813509942246911)

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