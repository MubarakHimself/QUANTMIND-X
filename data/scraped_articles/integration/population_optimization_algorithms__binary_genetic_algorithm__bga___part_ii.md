---
title: Population optimization algorithms: Binary Genetic Algorithm (BGA). Part II
url: https://www.mql5.com/en/articles/14040
categories: Integration, Machine Learning
relevance_score: 9
scraped_at: 2026-01-22T17:41:14.937329
---

[![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/01.png)![](https://www.mql5.com/ff/sh/qv94j0cd8n2n55z9z2/02.png)Boost your trading experienceRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=heclgjpfbvfghpmyaciuaesdtswflupo&s=4255fbe1b8cbc4d1b40afbaebf4235e5ace8b5103cba60d996897a03d588556f&uid=&ref=https://www.mql5.com/en/articles/14040&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049303629590014216)

MetaTrader 5 / Examples


### Contents

1\. [Introduction](https://www.mql5.com/en/articles/14040#tag1)

2\. [Algorithm](https://www.mql5.com/en/articles/14040#tag2)

3\. [Test results](https://www.mql5.com/en/articles/14040#tag3)

### 1\. Introduction

In the [previous article](https://www.mql5.com/en/articles/14053) we got acquainted with all the important basic concepts and methods used not only in genetic algorithms, but, one way or another, in any population optimization algorithms. Now, based on this knowledge, we can consider in detail the main subject of the current "two-volume book" - the binary genetic algorithm (BGA). Starting with general information, let's move on to the code of this remarkable search strategy and finally to the test results.

Genetic algorithm (GA) refers to evolutionary algorithms, which includes various approaches such as genetic algorithms, genetic programming, evolutionary strategies and others. They are based on the ideas of evolution and heredity, where solutions are represented as a population and genetic operators (such as crossover and mutation) are applied to create new generations of solutions.

Genetic algorithms use the principles of natural selection and genetics to solve optimization problems. The binary genetic algorithm (BGA) discussed in the article was the first among all types of genetic algorithms. Thus, BGA belongs to the class of evolutionary algorithms and is a specific variant of a genetic algorithm that uses a binary representation of data.

The work on the creation of genetic algorithms began in the middle of the 20th century. One of the founders is John Holland, who published his book "Adaptation in Natural and Artificial Systems" in 1975, where he introduced the genetic algorithm as a whole direction of approach to solving optimization problems.

The development of the genetic binary algorithm was inspired by several factors and ideas. The main ones:

- **Natural selection and principles of evolution**: BGA is based on the principles of natural selection and evolution proposed by Charles Darwin. The idea is that there is a diversity of solutions in a population, and those that are better adapted to the environment are more likely to survive and pass on their characteristics to the next generation.
- **Genetics and heredity**: BGA also uses genetics concepts such as gene, chromosome and heredity. Solutions in BGA are represented as binary strings, where individual groups of bits represent specific genes, and the gene in turn represents the parameter being optimized. Genetic operators, such as crossover and mutation, are applied to binary strings to create new generations of solutions.

Overall, the development of BGA was the result of a combination of ideas from the fields of evolutionary algorithms, genetics and optimization. It was created to solve optimization problems using the principles of natural selection and genetics, and its development continues to this day, a huge number of GA options have been created, as well as the widespread use of ideas and approaches in genetic algorithms as part of [hybrid](https://www.mql5.com/en/articles/14011) algorithms, including very complex ones.

The Binary Genetic Algorithm (BGA) uses a binary representation of data. This means that each individual (solution) is represented as a string of bits (0 and 1). Genetic operators, such as crossover and mutation, are applied to bit strings to create new generations of solutions.

Interesting fact: Genetic algorithms, including BGA, can be used to create and improve artificial intelligence. For example, they can be used to evolve neural networks, allowing the creation of more efficient machine learning models.

Genetic algorithms in general and BGA in particular are a powerful tool for solving complex optimization problems where traditional methods may not be effective enough due to the lack of analytical solution. MetaTrader 5 uses a binary GA and therefore it is even more exciting to study the operation principles of this remarkable algorithm.

### 2\. Algorithm

The binary genetic algorithm includes the following steps:

1. Population initialization - create an initial population consisting of chromosomes with random binary values.
2. Fitness assessment - evaluate the fitness of each individual (chromosome) in the daughter population.
3. Selection - select parents to create offspring using the roulette wheel method.
4. Crossover - break the parent chromosomes into sections and create new daughter chromosomes with sections of both parents.
5. Inversion - split the chromosome of a daughter individual at a randomly selected point and swap the resulting sections.

6. Mutation - randomly change bits in the genes of the offspring with a given probability of mutation.
7. Assessment of offspring fitness - assess the fitness of each new offspring.
8. Forming a new population - place the offspring population at the end of the total population and sort by fitness value.
9. Stopping criterion - repeat the process from step 3 until the stopping criterion is reached.

We will use the distance between the "max" and "min" of the optimized parameters to ensure that the BGA works only with positive numbers, simplify operations with binary strings and increase the speed of calculations. We will represent the positive distance values obtained in this way in the form of a binary Gray code and place them sequentially in one common array of chromosome symbols as shown in Figure 1. When performing the crossover method, chromosome breakpoints are located in random places on the chromosome and not necessarily at gene joining points. Breakpoints can also be inside the gene space.

![GeneCode](https://c.mql5.com/2/64/GeneCode__1.png)

Figure 1. Placing the characteristics of an individual (optimized parameters) in the chromosome

There are a significant number of external parameters for a genetic algorithm, so it is reasonable to consider them in more detail. The default settings are almost entirely consistent with the recommendations of many authors of scientific publications. I have tested and selected them to provide maximum efficiency in most types of tasks. However, deviation from these parameters in any direction can lead to achieving 100% convergence on tests with 10 or even 50 optimized parameters on individual functions, but at the same time significantly reduce efficiency on other tasks. Therefore, the default settings are optimal for use in most practical cases.

- **Population\_P (population size = 50)** \- the number of daughter individuals in each generation of the population. This population size is used in most of the algorithms covered in the tests and provides a balance between solution diversity and convergence speed.


- **ParentPopulation\_P (parent population size = 50)** \- the number of parents selected for breeding and creating the next generation. Decreasing the value of this parameter improves convergence on smooth functions (increases accuracy), increasing the value improves convergence on discrete functions (increases the diversity of genetic material).


- **CrossoverProbab\_P (crossover probability = 1.0)** \- the probability of performing a crossover operation between two selected parents. A high probability of crossover increases the combinatorial capabilities of the algorithm. Decreasing the value increases the speed of convergence, but increases the probability of getting stuck.


- **CrossoverPoints\_P (number of crossover points = 3)** \- the number of points, at which a crossover between the two parent chromosomes occurs. Increasing the points leads to more intense mixing of parameters among themselves and, in the limit, reduces to the algorithm random behavior.


- **MutationProbab\_P (mutation probability = 0.001)** \- the probability of mutation of each bit in the chromosome genotype. Mutation allows random changes to be made to genetic material and adds new solutions to a population. A low probability increases the convergence rate of the algorithm but reduces diversity, while a too high probability results in the loss of useful information.

- **InversionProbab\_P (inversion probability = 0.7)** \- the probability of performing an inversion operation on a chromosome. A high probability of inversion increases the diversity of the genetic material, but too high a probability leads to the algorithm random behavior.

- **DoubleDigitsInChromo\_P (number of decimal places in the gene)** \- the number of decimal places of the real number (optimized parameter) represented in binary form in the chromosome. Increasing the value leads to an increase in the complexity of calculations and an increase in optimization time (without using the binary form directly in solving the problem, it makes no sense to set more than 16 - the output bits will be lost when converting to double).

Let's move on to reviewing the code.

When initializing agents, it is necessary to determine the number of bits in the binary representation of the parameter being optimized. Let's say we need to consider five decimal places, which corresponds to a certain Gray code length. However, more than the required number may be encoded in this code. Therefore, we need to determine the maximum real number that can be encoded in the binary format. In the future, we can scale the encoded number to the required range of the optimized output parameter. For the fractional part of a real number, we use a given number of digits (specified in the parameters), and for the integer part - as many as required in binary form.

For example, if the input parameter of the digitsInGrayCode function is 3, then the function will return the maximum value of ulong type using the Gray code for 3 bits, that is, 7 (111 in binary).

In order to determin the maximum possible number that can be encoded by a given number of bits for the fractional and integer part of a real number, we will use the GetMaxDecimalFromGray function.

```
//——————————————————————————————————————————————————————————————————————————————
//Calculation of the maximum possible ulong number using the Gray code for a given number of bits
ulong GetMaxDecimalFromGray (int digitsInGrayCode)
{
  ulong maxValue = 1;

  for (int i = 1; i < digitsInGrayCode; i++)
  {
    maxValue <<= 1;
    maxValue |= 1;
  }

  return maxValue;
}
//——————————————————————————————————————————————————————————————————————————————
```

To represent each gene in the chromosome (the position of the gene on the chromosome), we will use the S\_BinaryGene structure, which contains fields and methods for working with genes in binary format:

- "gene" - character array representing a gene.
- "integPart" - character array of the integer gene part.
- "fractPart" - character array of the fractional gene part.
- "integGrayDigits" - number of Gray digits for the integer part of the gene.
- "fractGrayDigits" - number of Gray digits for the fractional part of the gene.
- "length" - total gene length.
- "minDoubleRange" - minimum value of the real number range.
- "maxDoubleRange" - maximum value of the real number range.
- "maxDoubleNumber" - maximum real number that can be represented using the gene.
- "doubleFract" - value to convert the fractional part of the gene to a real number.

Structure methods:

- "Init" - initializes the structure fields. Accepts the minimum and maximum values of a range of real numbers, as well as the number of digits in the fractional part of the gene. Within the method, values for the maximum real number of coding parts of the gene are calculated using Gray code.


- "ToDouble" - converts the gene to a real number. Accepts the "chromo" (chromosome) array of Gray code characters and the "indChr" gene start index. The method reads a region of a chromosome, converts the read gene into a decimal value and then scales it to a specified range of real numbers.

- "Scale" - scales the "In" input value from the "InMIN" and "InMAX" range to the "OutMIN" and "OutMAX" output range. This is an auxiliary method used in "ToDouble".

```
//——————————————————————————————————————————————————————————————————————————————
struct S_BinaryGene
{
  char   gene      [];
  char   integPart [];
  char   fractPart [];

  uint   integGrayDigits;
  uint   fractGrayDigits;
  uint   length;

  double minDoubleRange;
  double maxDoubleRange;
  double maxDoubleNumber;
  double doubleFract;

  //----------------------------------------------------------------------------
  void Init (double min, double max, int doubleDigitsInChromo)
  {
    doubleFract = pow (0.1, doubleDigitsInChromo);

    minDoubleRange = min;
    maxDoubleRange = max - min;

    ulong decInfr = 0;

    for (int i = 0; i < doubleDigitsInChromo; i++)
    {
      decInfr += 9 * (ulong)pow (10, i);
    }

    //----------------------------------------
    DecimalToGray (decInfr, fractPart);

    ulong maxDecInFr = GetMaxDecimalFromGray (ArraySize (fractPart));
    double maxDoubFr = maxDecInFr * doubleFract;


    //----------------------------------------
    DecimalToGray ((ulong)maxDoubleRange, integPart);

    ulong  maxDecInInteg = GetMaxDecimalFromGray (ArraySize (integPart));
    double maxDoubInteg  = (double)maxDecInInteg + maxDoubFr;

    maxDoubleNumber = maxDoubInteg;

    ArrayResize (gene, 0, 1000);
    integGrayDigits = ArraySize (integPart);
    fractGrayDigits = ArraySize (fractPart);
    length          = integGrayDigits + fractGrayDigits;

    ArrayCopy (gene, integPart, 0,                0, WHOLE_ARRAY);
    ArrayCopy (gene, fractPart, ArraySize (gene), 0, WHOLE_ARRAY);
  }

  //----------------------------------------------------------------------------
  double ToDouble (const char &chromo [], const int indChr)
  {
    double d;
    if(integGrayDigits > 0)d = (double)GrayToDecimal(chromo, indChr, indChr + integGrayDigits - 1);
    else                   d = 0.0;

    d +=(double)GrayToDecimal(chromo, indChr + integGrayDigits, indChr + integGrayDigits + fractGrayDigits - 1) * doubleFract;

    return minDoubleRange + Scale(d, 0.0, maxDoubleNumber, 0.0, maxDoubleRange);
  }

  //----------------------------------------------------------------------------
  double Scale (double In, double InMIN, double InMAX, double OutMIN, double OutMAX)
  {
    if (OutMIN == OutMAX) return (OutMIN);
    if (InMIN == InMAX) return (double((OutMIN + OutMAX) / 2.0));
    else
    {
      if (In < InMIN) return OutMIN;
      if (In > InMAX) return OutMAX;

      return (((In - InMIN) * (OutMAX - OutMIN) / (InMAX - InMIN)) + OutMIN);
    }
  }
};
//——————————————————————————————————————————————————————————————————————————————
```

To describe the agent, we need the S\_Agent structure, which represents the agent and contains the following data in addition to the chromosome:

- "c" - array of agent coordinate values as a real number.
- "f" - agent fitness value.
- "genes" - array of "S\_BinaryGene" structures that describe the position of each gene in the chromosome and the rules for converting the binary code into a real number.
- "chromosome" - array of agent chromosome characters.
- "calculated" - a boolean value indicating whether the agent's values have been calculated (the field is present, but not used in the code).

Structure methods:

- "Init" - initializes the structure fields. Accepts the number of "coords" coordinates, "min" and "max" arrays with minimum and maximum values for each optimized parameter, as well as doubleDigitsInChromo - the number of digits of a real number in the fractional part of the gene. The method creates and initializes genes for each coordinate, and also sets the initial fitness value and the "calculated" flag.

- "ExtractGenes" - extracts gene values from a chromosome and stores them in "c" array, uses the "ToDouble" method from the "S\_BinaryGene" structure to convert genes from a chromosome to real numbers.

```
//——————————————————————————————————————————————————————————————————————————————
struct S_Agent
{
  void Init (const int coords, const double &min [], const double &max [], int doubleDigitsInChromo)
  {
    ArrayResize(c, coords);
    f = -DBL_MAX;

    ArrayResize(genes, coords);
    ArrayResize(chromosome, 0, 1000);

    for(int i = 0; i < coords; i++)
    {
      genes [i].Init(min [i], max [i], doubleDigitsInChromo);
      ArrayCopy(chromosome, genes [i].gene, ArraySize(chromosome), 0, WHOLE_ARRAY);
    }

    calculated = false;
  }

  void ExtractGenes ()
  {
    uint pos = 0;

    for (int i = 0; i < ArraySize (genes); i++)
    {
      c [i] = genes [i].ToDouble (chromosome, pos);
      pos  += genes [i].length;

    }
  }

  double c []; //coordinates
  double f;    //fitness

  S_BinaryGene genes [];
  char chromosome    [];
  bool calculated;
};
//——————————————————————————————————————————————————————————————————————————————
```

The following code presents the definition of the S\_Roulette structure, which represents the roulette segment.

Structure fields:

- "start" - roulette segment starting point value.
- "end" - roulette segment end point.

```
//——————————————————————————————————————————————————————————————————————————————
struct S_Roulette
{
  double start;
  double end;
};
//——————————————————————————————————————————————————————————————————————————————
```

Declare the C\_AO\_BGA class that implements the genetic algorithm:

- "cB" - array of best coordinate values.
- "fB" - fitness value of the best coordinates.
- "a" - array of S\_Agent structures representing agents.
- "rangeMax" - array of maximum search range values.
- "rangeMin" - array of minimum search range values.
- "rangeStep" - array of values representing the search step.

Class methods:

- "Init" - initializes the class field. It accepts the following parameters: "coordsP" number of coordinates, "popSizeP" population size, "parentPopSizeP" parent population size, "crossoverProbabP" crossing probability, "crossoverPointsP" number of crossing points, "mutationProbabP" mutation probability, "inversionProbabP" inversion probability, "doubleDigitsInChromoP" number of decimal places in the gene. The method initializes internal variables and arrays necessary for the operation of the genetic algorithm.

- "Moving" - main method of the genetic algorithm, performs operations on the population, such as crossover, mutation, inversion and fitness assessment.

- "Revision" - the method performs a population revision, sorting agents and selecting the best.

The class's private fields and methods are used to implement the genetic algorithm internally, including roulette operations, value scaling, sorting and other operations.

```
//——————————————————————————————————————————————————————————————————————————————
class C_AO_BGA
{
  //----------------------------------------------------------------------------
  public: double  cB [];  //best coordinates
  public: double  fB;     //FF of the best coordinates
  public: S_Agent a  [];  //agent

  public: double rangeMax  []; //maximum search range
  public: double rangeMin  []; //manimum search range
  public: double rangeStep []; //step search

  public: void Init (const int    coordsP,               //coordinates number
                     const int    popSizeP,              //population size
                     const int    parentPopSizeP,        //parent population size
                     const double crossoverProbabP,      //crossover probability
                     const int    crossoverPointsP,      //crossover points
                     const double mutationProbabP,       //mutation probability
                     const double inversionProbabP,      //inversion probability
                     const int    doubleDigitsInChromoP);//number of decimal places in the gene

  public: void Moving   ();
  public: void Revision ();

  //----------------------------------------------------------------------------
  private: int    coords;               //coordinates number
  private: int    popSize;              //population size
  private: int    parentPopSize;        //parent population size
  private: double crossoverProbab;      //crossover probability
  private: int    crossoverPoints;      //crossover points
  private: double mutationProbab;       //mutation probability
  private: double inversionProbab;      //inversion probability
  private: int    doubleDigitsInChromo; //number of decimal places in the gene
  private: bool   revision;

  private: S_Agent    parents    [];  //parents
  private: int        ind        [];  //temporary array for sorting the population
  private: double     val        [];  //temporary array for sorting the population
  private: S_Agent    pTemp      [];  //temporary array for sorting the population
  private: char       tempChrome [];  //temporary chromosome for inversion surgery
  private: uint       lengthChrome;   //length of the chromosome (the length of the string of characters according to the Gray code)
  private: int        pCount;         //indices of chromosome break points
  private: uint       poRND      [];  //temporal indices of chromosome break points
  private: uint       points     [];  //final indices of chromosome break points
  private: S_Roulette roulette   [];  //roulette

  private: void   PreCalcRoulette ();
  private: int    SpinRoulette    ();
  private: double SeInDiSp  (double In, double InMin, double InMax, double Step);
  private: double RNDfromCI (double min, double max);
  private: void   Sorting   (S_Agent &p [], int size);
  private: double Scale     (double In, double InMIN, double InMAX, double OutMIN, double OutMAX);
};
//——————————————————————————————————————————————————————————————————————————————
```

The following code represents the implementation of the Init method of the C\_AO\_BGA class. This method initializes the class fields and arrays necessary for the genetic algorithm to work.

Method inputs:

- "coordsP" - number of coordinates.
- "popSizeP" - population size.
- "parentPopSizeP" - parent population size.
- "crossoverProbabP" - crossover probability.
- "crossoverPointsP" - number of crossover points.
- "mutationProbabP" - mutation probability.
- "inversionProbabP" - inversion probability.
- "doubleDigitsInChromoP" - number of decimal places in the gene.

The Init method is used to initialize class fields and arrays before executing the genetic algorithm. It sets the values of class fields, checks and adjusts the values of some parameters and also resizes arrays by allocating memory for them.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_BGA::Init (const int    coordsP,               //coordinates number
                     const int    popSizeP,              //population size
                     const int    parentPopSizeP,        //parent population size
                     const double crossoverProbabP,      //crossover probability
                     const int    crossoverPointsP,      //crossover points
                     const double mutationProbabP,       //mutation probability
                     const double inversionProbabP,      //inversion probability
                     const int    doubleDigitsInChromoP) //number of decimal places in the gene
{
  MathSrand ((int)GetMicrosecondCount ()); // reset of the generator
  fB       = -DBL_MAX;
  revision = false;

  coords               = coordsP;
  popSize              = popSizeP;
  parentPopSize        = parentPopSizeP;
  crossoverProbab      = crossoverProbabP;
  crossoverPoints      = crossoverPointsP;
  pCount               = crossoverPointsP;
  mutationProbab       = mutationProbabP;
  inversionProbab      = inversionProbabP;
  doubleDigitsInChromo = doubleDigitsInChromoP;

  if (crossoverPoints < 1) crossoverPoints = 1;
  if (pCount < 1) pCount = 1;

  ArrayResize (poRND,  pCount);
  ArrayResize (points, pCount + 2);

  ArrayResize (ind,   parentPopSize + popSize);
  ArrayResize (val,   parentPopSize + popSize);
  ArrayResize (pTemp, parentPopSize + popSize);
  ArrayResize (a,     popSize);

  ArrayResize (parents,  parentPopSize + popSize);
  ArrayResize (roulette, parentPopSize);

  ArrayResize (rangeMax,  coords);
  ArrayResize (rangeMin,  coords);
  ArrayResize (rangeStep, coords);
  ArrayResize (cB,        coords);
}
//——————————————————————————————————————————————————————————————————————————————
```

All basic operations for working with the genetic material of agents are performed using the Moving method of the C\_AO\_BGA class. The Moving method performs a genetic algorithm step, including selecting parents, crossover, inverting and mutating chromosomes and applying operations to genes and coordinates of individuals.

The method is logically divided into several parts:

1\. "if (!revision)" - if "revision" is equal to "false", the population individuals are initialized:

- The Init method is called to initialize the individual with the given parameters.
- A random chromosome is generated by filling each gene with a random value of 0 or 1.
- The ExtractGenes method is called to extract genes from the chromosome.
- The coordinates of "c" individual are reduced to a range using the SeInDiSp function.
- The fitness value of each individual "f" is set to "-DBL\_MAX".
- "lengthChrome = ArraySize (a \[0\].chromosome)" - length of an individual's chromosome (all individuals have the same length).
- "ArrayResize (tempChrome, lengthChrome)" - replaces the size of the "tempChrome" temporary array with "lengthChrome".

2\. For each individual in the population:

- A preliminary calculation of the parent selection roulette is performed using the PreCalcRoulette method.
- The parent is selected using the SpinRoulette method.
- The chromosome of the selected parent individual is copied into the chromosome of the current individual.
- The crossing operation is performed with the crossoverProbab probability.
- The second parent is selected.
- Chromosome breakpoints are determined.
- The chromosomes of the parent individuals are crossed.
- The inversion operation is performed with the inversionProbab probability.
- A random chromosome breakpoint is determined.
- Parts of the chromosome change places.
- A mutation operation is performed for each chromosome gene with the mutationProbab probability.
- The ExtractGenes method is called to extract genes from the chromosome.
- The coordinates of "c" individual are reduced to a range using the SeInDiSp function.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_BGA::Moving ()
{
  //----------------------------------------------------------------------------
  if (!revision)
  {
    for (int i = 0; i < popSize; i++)
    {
      a [i].Init (coords, rangeMin, rangeMax, doubleDigitsInChromo);

      int r = 0;

      for (int len = 0; len < ArraySize (a [i].chromosome); len++)
      {
        r  = MathRand (); //[0,32767]

        if (r > 16384) a [i].chromosome [len] = 1;
        else           a [i].chromosome [len] = 0;
      }

      a [i].ExtractGenes ();

      for (int c = 0; c < coords; c++) a [i].c [c] = SeInDiSp (a [i].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);

      a [i].f = -DBL_MAX;
      a [i].calculated = true;
    }

    lengthChrome = ArraySize (a [0].chromosome);
    ArrayResize (tempChrome, lengthChrome);

    for (int i = 0; i < parentPopSize + popSize; i++)
    {
      parents [i].Init (coords, rangeMin, rangeMax, doubleDigitsInChromo);
      parents [i].f = -DBL_MAX;
    }

    revision = true;
    return;
  }

  //----------------------------------------------------------------------------
  int    pos       = 0;
  double r         = 0;
  uint   p1        = 0;
  uint   p2        = 0;
  uint   p3        = 0;
  uint   temp      = 0;

  for (int i = 0; i < popSize; i++)
  {
    PreCalcRoulette ();

    //selection, select and copy the parent to the child------------------------
    pos = SpinRoulette ();

    ArrayCopy (a [i].chromosome, parents [pos].chromosome, 0, 0, WHOLE_ARRAY);

    //crossover-----------------------------------------------------------------
    r = RNDfromCI (0.0, 1.0);

    if (r < crossoverProbab)
    {
      //choose a second parent to breed with------------------------------------
      pos = SpinRoulette ();

      //determination of chromosome break points--------------------------------
      for (int p = 0; p < pCount; p++)
      {
        poRND [p] = (int)RNDfromCI (0.0, lengthChrome);
        if (poRND [p] >= lengthChrome) poRND [p] = lengthChrome - 1;
      }
      ArraySort (poRND);
      ArrayCopy (points, poRND, 1, 0, WHOLE_ARRAY);
      points [0] = 0;
      points [pCount + 1] = lengthChrome - 1;

      r = RNDfromCI (0.0, 1.0);

      int startPoint = r > 0.5 ? 0 : 1;

      for (int p = startPoint; p < pCount + 2; p += 2)
      {
        if (p < pCount + 1)
        {
          for (uint len = points [p]; len < points [p + 1]; len++) a [i].chromosome [len] = parents [pos].chromosome [len];
        }
      }
    }

    //perform an inversion------------------------------------------------------
    //(break the chromosome, swap the received parts, connect them together)
    r = RNDfromCI (0.0, 1.0);

    if (r < inversionProbab)
    {
      p1 = (int)RNDfromCI (0.0, lengthChrome);
      if (p1 >= lengthChrome) p1 = lengthChrome - 1;

      //copying the second part to the beginning of the temporary array
      for (uint len = p1; len < lengthChrome; len++) tempChrome [len - p1] = a [i].chromosome [len];

      //copying the first part to the end of the temporary array
      for (uint len = 0; len < p1; len++)       tempChrome [lengthChrome - p1 + len] = a [i].chromosome [len];

      //copying a temporary array back
      for (uint len = 0; len < lengthChrome; len++)   a [i].chromosome [len] = tempChrome [len];
    }

    //perform mutation---------------------------------------------------------
    for (uint len = 0; len < lengthChrome; len++)
    {
      r = RNDfromCI (0.0, 1.0);
      if (r < mutationProbab) a [i].chromosome [len] = a [i].chromosome [len] == 1 ? 0 : 1;
    }

    a [i].ExtractGenes ();

    for (int c = 0; c < coords; c++) a [i].c [c] = SeInDiSp (a [i].c [c], rangeMin [c], rangeMax [c], rangeStep [c]);

    a [i].calculated = true;
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The Revision method performs a population revision, sorts individuals by the value of their fitness function and updates the "fB" fitness of the best solution and the "cB" coordinates of the best solution. Before sorting the population, the child population is copied to the end of the total population.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_BGA::Revision ()
{
  //----------------------------------------------------------------------------
  for (int i = parentPopSize; i < parentPopSize + popSize; i++)
  {
    parents [i] = a [i - parentPopSize];
  }

  Sorting (parents, parentPopSize + popSize);

  if (parents [0].f > fB)
  {
    fB = parents [0].f;
    ArrayCopy (cB, parents [0].c, 0, 0, WHOLE_ARRAY);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The preliminary roulette layout code represents the PreCalcRoulette method. This method preliminarily computes range values for a roulette selection of individuals based on their fitness function.

```
//——————————————————————————————————————————————————————————————————————————————
void C_AO_BGA::PreCalcRoulette ()
{
  roulette [0].start = parents [0].f;
  roulette [0].end   = roulette [0].start + (parents [0].f - parents [parentPopSize - 1].f);

  for (int s = 1; s < parentPopSize; s++)
  {
    if (s != parentPopSize - 1)
    {
      roulette [s].start = roulette [s - 1].end;
      roulette [s].end   = roulette [s].start + (parents [s].f - parents [parentPopSize - 1].f);
    }
    else
    {
      roulette [s].start = roulette [s - 1].end;
      roulette [s].end   = roulette [s].start + (parents [s - 1].f - parents [s].f) * 0.1;
    }
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

The SpinRoulette method ensures the position of the parent individual is selected.

```
//——————————————————————————————————————————————————————————————————————————————
int C_AO_BGA::SpinRoulette ()
{
  double r = RNDfromCI (roulette [0].start, roulette [parentPopSize - 1].end);

  for (int s = 0; s < parentPopSize; s++)
  {
    if (roulette [s].start <= r && r < roulette [s].end) return s;
  }

  return 0;
}
//——————————————————————————————————————————————————————————————————————————————
```

### 3\. Test results

BGA test stand results:

C\_AO\_BGA:50:50:1.0:3:0.001:0.7:3

=============================

5 Hilly's; Func runs: 10000; result: 0.9999191151339382

25 Hilly's; Func runs: 10000; result: 0.994841435673127

500 Hilly's; Func runs: 10000; result: 0.5048331764136147

=============================

5 Forest's; Func runs: 10000; result: 1.0

25 Forest's; Func runs: 10000; result: 0.9997457419655973

500 Forest's; Func runs: 10000; result: 0.32054251149158375

=============================

5 Megacity's; Func runs: 10000; result: 0.9066666666666668

25 Megacity's; Func runs: 10000; result: 0.9640000000000001

500 Megacity's; Func runs: 10000; result: 0.23034999999999997

=============================

All score: 6.92090 (76.90%)

Visualization of the BGA optimization clearly demonstrates the high convergence of the algorithm. Interestingly, during the optimization process, some part of the population remains away from the global extremum, which indicates the exploration of new unknown regions of the search space, maintaining the diversity of solutions in the population. However, the algorithm faces certain difficulties when working with the Megacity discrete function, which is problematic for most algorithms. Despite some variation in results when working with this complex function, BGA performs significantly better than other algorithms.

![Hilly](https://c.mql5.com/2/64/Hilly.gif)

**BGA on the [Hilly](https://www.mql5.com/en/articles/13923#tagHilly) test function**

![Forest](https://c.mql5.com/2/64/Forest.gif)

**BGA on the [Forest](https://www.mql5.com/en/articles/11785#tag3) test function**

![Megacity](https://c.mql5.com/2/64/Megacity.gif)

**BGA on the [Megacity](https://www.mql5.com/en/articles/11785#tag3) test function**

BGA took the top spot in the table, performing best in most tests for all three test functions. BGA demonstrated especially impressive results on the Megacity discrete function, outperforming other algorithms.

| |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \# | AO | Description | Hilly | Hilly final | Forest | Forest final | Megacity (discrete) | Megacity final | Final result | % of MAX |
| 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) | 10 p (5 F) | 50 p (25 F) | 1000 p (500 F) |
| 1 | BGA | [binary genetic algorithm](https://www.mql5.com/en/articles/14040) | 0.99992 | 0.99484 | 0.50483 | 2.49959 | 1.00000 | 0.99975 | 0.32054 | 2.32029 | 0.90667 | 0.96400 | 0.23035 | 2.10102 | 6.921 | 76.90 |
| 2 | (P+O)ES | [(P+O) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.99934 | 0.91895 | 0.56297 | 2.48127 | 1.00000 | 0.93522 | 0.39179 | 2.32701 | 0.83167 | 0.64433 | 0.21155 | 1.68755 | 6.496 | 72.18 |
| 3 | SDSm | [stochastic diffusion search M](https://www.mql5.com/en/articles/13540) | 0.93066 | 0.85445 | 0.39476 | 2.17988 | 0.99983 | 0.89244 | 0.19619 | 2.08846 | 0.72333 | 0.61100 | 0.10670 | 1.44103 | 5.709 | 63.44 |
| 4 | SIA | [simulated isotropic annealing](https://www.mql5.com/en/articles/13870) | 0.95784 | 0.84264 | 0.41465 | 2.21513 | 0.98239 | 0.79586 | 0.20507 | 1.98332 | 0.68667 | 0.49300 | 0.09053 | 1.27020 | 5.469 | 60.76 |
| 5 | DE | [differential evolution](https://www.mql5.com/en/articles/13781) | 0.95044 | 0.61674 | 0.30308 | 1.87026 | 0.95317 | 0.78896 | 0.16652 | 1.90865 | 0.78667 | 0.36033 | 0.02953 | 1.17653 | 4.955 | 55.06 |
| 6 | HS | [harmony search](https://www.mql5.com/en/articles/12163) | 0.86509 | 0.68782 | 0.32527 | 1.87818 | 0.99999 | 0.68002 | 0.09590 | 1.77592 | 0.62000 | 0.42267 | 0.05458 | 1.09725 | 4.751 | 52.79 |
| 7 | SSG | [saplings sowing and growing](https://www.mql5.com/en/articles/12268) | 0.77839 | 0.64925 | 0.39543 | 1.82308 | 0.85973 | 0.62467 | 0.17429 | 1.65869 | 0.64667 | 0.44133 | 0.10598 | 1.19398 | 4.676 | 51.95 |
| 8 | (PO)ES | [(PO) evolution strategies](https://www.mql5.com/en/articles/13923) | 0.79025 | 0.62647 | 0.42935 | 1.84606 | 0.87616 | 0.60943 | 0.19591 | 1.68151 | 0.59000 | 0.37933 | 0.11322 | 1.08255 | 4.610 | 51.22 |
| 9 | ACOm | [ant colony optimization M](https://www.mql5.com/en/articles/11602) | 0.88190 | 0.66127 | 0.30377 | 1.84693 | 0.85873 | 0.58680 | 0.15051 | 1.59604 | 0.59667 | 0.37333 | 0.02472 | 0.99472 | 4.438 | 49.31 |
| 10 | BFO-GA | [bacterial foraging optimization - ga](https://www.mql5.com/en/articles/14011) | 0.89150 | 0.55111 | 0.31529 | 1.75790 | 0.96982 | 0.39612 | 0.06305 | 1.42899 | 0.72667 | 0.27500 | 0.03525 | 1.03692 | 4.224 | 46.93 |
| 11 | MEC | [mind evolutionary computation](https://www.mql5.com/en/articles/13432) | 0.69533 | 0.53376 | 0.32661 | 1.55569 | 0.72464 | 0.33036 | 0.07198 | 1.12698 | 0.52500 | 0.22000 | 0.04198 | 0.78698 | 3.470 | 38.55 |
| 12 | IWO | [invasive weed optimization](https://www.mql5.com/en/articles/11990) | 0.72679 | 0.52256 | 0.33123 | 1.58058 | 0.70756 | 0.33955 | 0.07484 | 1.12196 | 0.42333 | 0.23067 | 0.04617 | 0.70017 | 3.403 | 37.81 |
| 13 | Micro-AIS | [micro artificial immune system](https://www.mql5.com/en/articles/13951) | 0.79547 | 0.51922 | 0.30861 | 1.62330 | 0.72956 | 0.36879 | 0.09398 | 1.19233 | 0.37667 | 0.15867 | 0.02802 | 0.56335 | 3.379 | 37.54 |
| 14 | COAm | [cuckoo optimization algorithm M](https://www.mql5.com/en/articles/11786) | 0.75820 | 0.48652 | 0.31369 | 1.55841 | 0.74054 | 0.28051 | 0.05599 | 1.07704 | 0.50500 | 0.17467 | 0.03380 | 0.71347 | 3.349 | 37.21 |
| 15 | SDOm | [spiral dynamics optimization M](https://www.mql5.com/en/articles/12252) | 0.74601 | 0.44623 | 0.29687 | 1.48912 | 0.70204 | 0.34678 | 0.10944 | 1.15826 | 0.42833 | 0.16767 | 0.03663 | 0.63263 | 3.280 | 36.44 |
| 16 | NMm | [Nelder-Mead method M](https://www.mql5.com/en/articles/13805) | 0.73807 | 0.50598 | 0.31342 | 1.55747 | 0.63674 | 0.28302 | 0.08221 | 1.00197 | 0.44667 | 0.18667 | 0.04028 | 0.67362 | 3.233 | 35.92 |
| 17 | FAm | [firefly algorithm M](https://www.mql5.com/en/articles/11873) | 0.58634 | 0.47228 | 0.32276 | 1.38138 | 0.68467 | 0.37439 | 0.10908 | 1.16814 | 0.28667 | 0.16467 | 0.04722 | 0.49855 | 3.048 | 33.87 |
| 18 | GSA | [gravitational search algorithm](https://www.mql5.com/en/articles/12072) | 0.64757 | 0.49197 | 0.30062 | 1.44016 | 0.53962 | 0.36353 | 0.09945 | 1.00260 | 0.32667 | 0.12200 | 0.01917 | 0.46783 | 2.911 | 32.34 |
| 19 | BFO | [bacterial foraging optimization](https://www.mql5.com/en/articles/12031) | 0.61171 | 0.43270 | 0.31318 | 1.35759 | 0.54410 | 0.21511 | 0.05676 | 0.81597 | 0.42167 | 0.13800 | 0.03195 | 0.59162 | 2.765 | 30.72 |
| 20 | ABC | [artificial bee colony](https://www.mql5.com/en/articles/11736) | 0.63377 | 0.42402 | 0.30892 | 1.36671 | 0.55103 | 0.21874 | 0.05623 | 0.82600 | 0.34000 | 0.14200 | 0.03102 | 0.51302 | 2.706 | 30.06 |
| 21 | BA | [bat algorithm](https://www.mql5.com/en/articles/11915) | 0.59761 | 0.45911 | 0.35242 | 1.40915 | 0.40321 | 0.19313 | 0.07175 | 0.66810 | 0.21000 | 0.10100 | 0.03517 | 0.34617 | 2.423 | 26.93 |
| 22 | SA | [simulated annealing](https://www.mql5.com/en/articles/13851) | 0.55787 | 0.42177 | 0.31549 | 1.29513 | 0.34998 | 0.15259 | 0.05023 | 0.55280 | 0.31167 | 0.10033 | 0.02883 | 0.44083 | 2.289 | 25.43 |
| 23 | IWDm | [intelligent water drops M](https://www.mql5.com/en/articles/13730) | 0.54501 | 0.37897 | 0.30124 | 1.22522 | 0.46104 | 0.14704 | 0.04369 | 0.65177 | 0.25833 | 0.09700 | 0.02308 | 0.37842 | 2.255 | 25.06 |
| 24 | PSO | [particle swarm optimisation](https://www.mql5.com/en/articles/11386) | 0.59726 | 0.36923 | 0.29928 | 1.26577 | 0.37237 | 0.16324 | 0.07010 | 0.60572 | 0.25667 | 0.08000 | 0.02157 | 0.35823 | 2.230 | 24.77 |
| 25 | MA | [monkey algorithm](https://www.mql5.com/en/articles/12212) | 0.59107 | 0.42681 | 0.31816 | 1.33604 | 0.31138 | 0.14069 | 0.06612 | 0.51819 | 0.22833 | 0.08567 | 0.02790 | 0.34190 | 2.196 | 24.40 |
| 26 | SFL | [shuffled frog-leaping](https://www.mql5.com/en/articles/13366) | 0.53925 | 0.35816 | 0.29809 | 1.19551 | 0.37141 | 0.11427 | 0.04051 | 0.52618 | 0.27167 | 0.08667 | 0.02402 | 0.38235 | 2.104 | 23.38 |
| 27 | FSS | [fish school search](https://www.mql5.com/en/articles/11841) | 0.55669 | 0.39992 | 0.31172 | 1.26833 | 0.31009 | 0.11889 | 0.04569 | 0.47467 | 0.21167 | 0.07633 | 0.02488 | 0.31288 | 2.056 | 22.84 |
| 28 | RND | [random](https://www.mql5.com/en/articles/8122) | 0.52033 | 0.36068 | 0.30133 | 1.18234 | 0.31335 | 0.11787 | 0.04354 | 0.47476 | 0.25333 | 0.07933 | 0.02382 | 0.35648 | 2.014 | 22.37 |
| 29 | GWO | [grey wolf optimizer](https://www.mql5.com/en/articles/11785) | 0.59169 | 0.36561 | 0.29595 | 1.25326 | 0.24499 | 0.09047 | 0.03612 | 0.37158 | 0.27667 | 0.08567 | 0.02170 | 0.38403 | 2.009 | 22.32 |
| 30 | CSS | [charged system search](https://www.mql5.com/en/articles/13662) | 0.44252 | 0.35454 | 0.35201 | 1.14907 | 0.24140 | 0.11345 | 0.06814 | 0.42299 | 0.18333 | 0.06300 | 0.02322 | 0.26955 | 1.842 | 20.46 |
| 31 | EM | [electroMagnetism-like algorithm](https://www.mql5.com/en/articles/12352) | 0.46250 | 0.34594 | 0.32285 | 1.13129 | 0.21245 | 0.09783 | 0.10057 | 0.41085 | 0.15667 | 0.06033 | 0.02712 | 0.24412 | 1.786 | 19.85 | |

### Summary

In this article, we examined the classic version of BGA, a special case of the general class of GA genetic algorithms, and all conclusions relate specifically to it. Despite the long-standing idea of representing solutions in binary, the approach using binary code remains relevant to this day. It combines the independent spatial dimensions of an optimization problem into a single whole by encoding information in a single chromosome, which is difficult to implement using conventional real-valued feature encoding, making this algorithm stand out among other optimization algorithms.

Although the mathematics and logic of the BGA strategy are completely clear to me, I am still fascinated by what is happening with the chromosome. It can be compared to a magical kaleidoscope. As we spin the kaleidoscope, the varied shapes and colors combine into unique patterns to create a spectacular picture. Similarly, the crossover operator in BGA randomly cuts the chromosome into several parts, **including internal parameter areas**. These pieces are then put together, like shuffling the pieces of a kaleidoscope. This allows us to combine the best features of different solutions and create a new, more optimal combination. Just like a kaleidoscope, the results of BGA crossover can be surprising and unexpected, turning simple chromosomes into true "diamonds" of optimal solutions.

I am confident that the information in this two-part article on the methods and tools used in genetic algorithms will help you expand your knowledge and reach new heights in your work and research. The power of evolution is constantly manifesting itself in nature, technology and the human mind, and BGA is one of many amazing algorithms that will help us reach new heights and achievements.

BGA effectively solves a variety of problems, be it smooth surfaces, discrete problems or even large-scale problems, including a stochastic approach, opening up new possibilities where analytical solutions are limited.

![rating table](https://c.mql5.com/2/64/rating_table__6.png)

Figure 2. Color gradation of algorithms according to relevant tests Results greater than or equal to 0.99 are highlighted in white

![chart](https://c.mql5.com/2/64/chart__3.png)

Figure 3. The histogram of algorithm test results (on a scale from 0 to 100, the more the better,

where 100 is the maximum possible theoretical result, the archive features a script for calculating the rating table)

**The pros and cons of the binary genetic algorithm (BGA) apply exclusively to the presented implementation:**

Advantages:

1. High efficiency in solving a variety of problems.
2. Resistance to sticking.
3. Promising results on both smooth and complex discrete functions.
4. High convergence.

Disadvantages:

1. A large number of external parameters.

2. Quite a complex implementation.

3. High computational complexity.


The article is accompanied by an archive with updated current versions of the algorithm codes described in previous articles. The author of the article is not responsible for the absolute accuracy in the description of canonical algorithms. Changes have been made to many of them to improve search capabilities. The conclusions and judgments presented in the articles are based on the results of the experiments.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14040](https://www.mql5.com/ru/articles/14040)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14040.zip "Download all attachments in the single ZIP archive")

[31\_The\_world\_of\_AO\_BGA.zip](https://www.mql5.com/en/articles/download/14040/31_the_world_of_ao_bga.zip "Download 31_The_world_of_AO_BGA.zip")(567.93 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neuroboids Optimization Algorithm (NOA)](https://www.mql5.com/en/articles/16992)
- [Successful Restaurateur Algorithm (SRA)](https://www.mql5.com/en/articles/17380)
- [Billiards Optimization Algorithm (BOA)](https://www.mql5.com/en/articles/17325)
- [Chaos Game Optimization (CGO)](https://www.mql5.com/en/articles/17047)
- [Blood inheritance optimization (BIO)](https://www.mql5.com/en/articles/17246)
- [Circle Search Algorithm (CSA)](https://www.mql5.com/en/articles/17143)
- [Royal Flush Optimization (RFO)](https://www.mql5.com/en/articles/17063)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/467361)**
(1)


![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
26 Jan 2024 at 08:52

Thanks for the Article! Added the algorithms to the [general list](https://www.mql5.com/ru/blogs/post/755815).


![MQL5 Wizard Techniques you should know (Part 20): Symbolic Regression](https://c.mql5.com/2/78/MQL5_Wizard_Techniques_you_should_know_4Part_20x___LOGO.png)[MQL5 Wizard Techniques you should know (Part 20): Symbolic Regression](https://www.mql5.com/en/articles/14943)

Symbolic Regression is a form of regression that starts with minimal to no assumptions on what the underlying model that maps the sets of data under study would look like. Even though it can be implemented by Bayesian Methods or Neural Networks, we look at how an implementation with Genetic Algorithms can help customize an expert signal class usable in the MQL5 wizard.

![Learn how to trade the Fair Value Gap (FVG)/Imbalances step-by-step: A Smart Money concept approach](https://c.mql5.com/2/78/Learn_how_to_trade_the_Fair_Value_Gap____LOGO__1.png)[Learn how to trade the Fair Value Gap (FVG)/Imbalances step-by-step: A Smart Money concept approach](https://www.mql5.com/en/articles/14261)

A step-by-step guide to creating and implementing an automated trading algorithm in MQL5 based on the Fair Value Gap (FVG) trading strategy. A detailed tutorial on creating an expert advisor that can be useful for both beginners and experienced traders.

![Developing a multi-currency Expert Advisor (Part 1): Collaboration of several trading strategies](https://c.mql5.com/2/65/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 1): Collaboration of several trading strategies](https://www.mql5.com/en/articles/14026)

There are quite a lot of different trading strategies. So, it might be useful to apply several strategies working in parallel to diversify risks and increase the stability of trading results. But if each strategy is implemented as a separate Expert Advisor (EA), then managing their work on one trading account becomes much more difficult. To solve this problem, it would be reasonable to implement the operation of different trading strategies within a single EA.

![Triangular arbitrage with predictions](https://c.mql5.com/2/78/Triangular_arbitrage_with_predictions___LOGO___1.png)[Triangular arbitrage with predictions](https://www.mql5.com/en/articles/14873)

This article simplifies triangular arbitrage, showing you how to use predictions and specialized software to trade currencies smarter, even if you're new to the market. Ready to trade with expertise?

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14040&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049303629590014216)

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