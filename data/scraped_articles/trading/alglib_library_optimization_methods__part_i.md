---
title: ALGLIB library optimization methods (Part I)
url: https://www.mql5.com/en/articles/16133
categories: Trading, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T17:55:19.687788
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/16133&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068800363047353813)

MetaTrader 5 / Tester


### **Contents**

1. [Introduction](https://www.mql5.com/en/articles/16133#tag1)
2. ALGLIB library optimization methods:

- [BLEIC (Boundary, Linear Equality-Inequality Constraints)](https://www.mql5.com/en/articles/16133#tag3)
- [L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)](https://www.mql5.com/en/articles/16133#tag4)
- [NS (Nonsmooth Nonconvex Optimization Subject to box/linear/nonlinear - Nonsmooth Constraints)](https://www.mql5.com/en/articles/16133#tag5)

### Introduction

The standard delivery of the MetaTrader 5 terminal includes the [ALGLIB](https://www.mql5.com/en/code/1146) library, which is a powerful tool for numerical analysis that can be useful for trading system developers. The ALGLIB library offers the user a wide range of numerical analysis methods, including:

- Linear algebra - solving systems of linear equations, calculating eigenvalues and vectors, matrix decomposition.
- Optimization - methods of one-dimensional and multi-dimensional optimization.
- Interpolation and approximation - polynomial and spline interpolation, approximation of functions using least squares methods.
- Numerical integration and differentiation - integration methods (trapezoids, Simpson, etc.), numerical differentiation.
- Numerical methods for solving differential equations - ordinary differential equations and numerical methods.
- Statistical methods - parameter estimation, regression analysis, random number generation.
- Fourier analysis: fast Fourier transform.

The deterministic optimization methods presented in ALGLIB are based on various variations of gradient descent and allow the use of both analytical and numerical approaches. In this article, we will focus on numerical methods, as they are most suitable for practical tasks of traders.

It is important to note that many problems in the financial sector are discrete in nature, since price data is represented as individual points. Therefore, we are particularly interested in methods that use numerical representation of gradients. The user does not need to calculate the gradient - this task is taken over by optimization methods.

Mastering ALGLIB optimization methods is usually difficult due to the lack of unification in the names of methods and the order they are accessed in. The main goal of this article is to fill in the gaps in information about the library and provide simple usage examples. Let's define the basic terms and concepts to better understand what these methods are intended for.

**Optimization** is the process of finding the best solution under given conditions and constraints. The search implies the presence of at least two options for solving the problem, from which the best one must be selected.

**Optimization problem** is a problem, in which it is necessary to find the best solution (maximum or minimum) from a set of possible options that satisfy certain conditions. The main elements of the optimization problem:

1. Objective Function (Fitness Function). This is the function that needs to be maximized or minimized. For example, maximizing profit or minimizing costs (profit or costs are the optimization criteria).
2. Variables. These are parameters that can be changed to achieve the best result.
3. Restrictions. These are the conditions that must be met when searching for the optimal solution.

The objective function can be any function specified by the user that takes inputs (to be optimized) and produces a value that serves as an optimization criterion. For example, the target function may be testing a trading system on historical data, where the function parameters are the trading system settings, and the output value is the required quality of its operation.

Types of restrictions:

1. Boundary Constraints (Box Constraints) — restrictions placed on the values of variables, for example, the variable "x" can only be in the range from 1 to 3.
2. Linear Equality Constraints - conditions, under which an expression with variables must be exactly equal to some number. For example, (x + y = 5) is a linear equality.
3. Linear Inequality Constraints — conditions under which an expression with variables must be less than or equal to (or greater than or equal to) some number. For example, (x - y >= 1).

In the examples below we will consider only boundary constraints. So, this article will reveal effective techniques and an easy way to use ALGLIB optimization methods using simple examples.

We will use a simple objective function that needs to be maximized as an example of an optimization problem. It is a smooth monotone function with a single maximum - an inverted paraboloid. The values of this function are in the range \[0; 1\], regardless of the number of arguments, which belong to the range \[-10; 10\].

```
//——————————————————————————————————————————————————————————————————————————————
//Paraboloid, F(Xn) ∈ [0.0; 1.0], X ∈ [-10.0; 10.0], maximization
double ObjectiveFunction (double &x [])
{
  double sum = 0.0;

  for (int i = 0; i < ArraySize (x); i++)
  {
    if (x [i] < -10.0 || x [i] > 10.0) return 0.0;
    sum += (-x [i] * x [i] + 100.0) * 0.01;
  }

  return sum /= ArraySize (x);
}
//——————————————————————————————————————————————————————————————————————————————
```

### BLEIC (Boundary, Linear Equality-Inequality Constraints)

BLEIC (Boundary, Linear Equality-Inequality Constraints) is the name for a family of methods used to solve optimization problems with equalities and inequalities. The name of the method comes from the classification of constraints, which divides them into active and inactive ones at the current point. The method reduces a problem with constraints in the form of equalities and inequalities to a sequence of subproblems limited only by equalities. Active inequalities are treated as equalities, and inactive ones are temporarily ignored (although we continue to monitor them). Informally speaking, the current point moves along the feasible set, "sticking" or "unsticking" from the boundaries.

1\. What it does? It looks for the minimum or maximum of some function (for example, it may look for the highest profit or the lowest cost), while taking into account various constraints:

- Boundaries of variable values (for example, price cannot be negative)
- Equalities (e.g. the sum of the weights in a portfolio should equal 100%)
- Inequalities (e.g. risk must not exceed a certain level)

2\. How does it work? It uses limited memory, which means that it does not store all intermediate calculations, but only the most important ones. Moves towards a solution gradually, step by step:

- Assesses the current situation
- Determines the direction of movement
- Takes a step in that direction
- Checks if restrictions are violated
- Adjusts the movement if necessary

3\. Features:

- The function should be "smooth" (without sharp jumps)
- Able to find a local minimum instead of a global one
- Sensitive to initial approximation (starting point)

A simple example: imagine you are looking for a place to build a house on a plot of land. You have restrictions: the house must be a certain size (equality), must be no closer than X meters from the property line (inequality), and must be located within the property line (boundary conditions). You want the view to be the best (objective function). BLEIC will gradually "move" the imaginary house around the site, respecting all restrictions, until it finds the best location in terms of a view. Find more details about this, as well as the following algorithms, on the [library author page](https://www.mql5.com/go?link=https://www.alglib.net/optimization/ "https://www.alglib.net/optimization/").

To use the BLEIC method and the rest in the ALGLIB library, we need to include the file (the library is supplied with the MetaTrader 5 terminal, the user does not need to install anything additionally).

```
#include <Math\Alglib\alglib.mqh>
```

Let's develop a script - an example for effective work using ALGLIB methods. I will highlight the main steps that are typical when working with ALGLIB methods. Identical blocks of code are also highlighted in the appropriate color.

1\. define the boundary conditions of the problem, such as the number of launches of the fitness function (objective function), the ranges of the parameters to be optimized and their step. For ALGLIB methods, it is necessary to assign starting values of the optimized parameters "x" (the methods are deterministic and the results depend entirely on the initial values, so we will apply random number generation in the range of the problem parameters), as well as the scale "s" (the methods are sensitive to the scale of the parameters relative to each other, in this case we set the scale to "1").

2\. Declare the objects necessary for the algorithm to work.

3\. Set the external parameters of the algorithm (settings).

4\. initialize the algorithm by passing the ranges and steps of the parameters to be optimized, as well as the external parameters of the algorithm, to the method.

5\. Perform optimization.

6\. Obtain the optimization results for further use.

Keep in mind that the user has no way to influence the optimization process or stop it at any time. The method performs all operations independently, calling the fitness function inside its process. The algorithm can call the fitness function an arbitrary number of times (there is no way to specify this stop parameter for the BLEIC method), and the user can only control the maximum allowed number of calls by himself, by forcing an attempt to pass a stop command to the method.

```
//——————————————————————————————————————————————————————————————————————————————
void OnStart ()
{
  // Initialization of optimization parameters---------------------------------------
  int numbTestFuncRuns = 10000;
  int params           = 1000;

  // Create and initialize arrays for range bounds---------------------
  double rangeMin [], rangeMax [], rangeStep;
  ArrayResize (rangeMin,  params);
  ArrayResize (rangeMax,  params);

  for (int i = 0; i < params; i++)
  {
    rangeMin  [i] = -10;
    rangeMax  [i] =  10;
  }
  rangeStep =  DBL_EPSILON;

  double x [];
  double s [];
  ArrayResize (x, params);
  ArrayResize (s, params);
  ArrayInitialize (s, 1);

  // Generate random initial parameter values in given ranges----
  for (int i = 0; i < params; i++)
  {
    x [i] = rangeMin [i] + ((rangeMax [i] - rangeMin [i]) * rand () / 32767.0);
  }

  // Create objects for optimization------------------------------------------
  C_OptimizedFunction  fFunc; fFunc.Init (params, numbTestFuncRuns);
  CObject              obj;
  CNDimensional_Rep    frep;
  CMinBLEICReportShell rep;

  // Set the parameters of the BLEIC optimization algorithm---------------------------
  double diffStep = 0.00001;
  double epsg     = 1e-16;
  double epsf     = 1e-16;
  double epsi     = 0.00001;

  CAlglib::MinBLEICCreateF      (x, diffStep, fFunc.state);
  CAlglib::MinBLEICSetBC        (fFunc.state, rangeMin, rangeMax);
  CAlglib::MinBLEICSetInnerCond (fFunc.state, epsg, epsf, rangeStep);
  CAlglib::MinBLEICSetOuterCond (fFunc.state, rangeStep, epsi);
  CAlglib::MinBLEICOptimize     (fFunc.state, fFunc, frep, 0, obj);
  CAlglib::MinBLEICResults      (fFunc.state, x, rep);

  // Output of optimization results-----------------------------------------------
  Print ("BLEIC, best result: ", fFunc.fB, ", number of function launches: ", fFunc.numberLaunches);
}
//——————————————————————————————————————————————————————————————————————————————
```

Since the method calls the fitness function itself (and not from the user program), it will be necessary to wrap the call to the fitness function in a class that inherits from a parent class in ALGLIB (these parent classes are different for different methods). Declare the wrapper class as **C\_OptimizedFunction** and set the following methods in the class:

1\. **Func ()** is a virtual method that is overridden in derived classes.

2\. **Init ()**  — initialize class parameters. Inside the method:

- Variables related to the number of runs and the best value found for the function are initialized.
- The **c** and **cB** arrays are reserved for storing coordinates.

Variables:

- **state** — **CMinBLEICStateShell** type object specific to the BLEIC method is used when calling static methods of the algorithm and calling the stop method.

- **numberLaunches** — current number of launches (necessary to prevent uncontrolled or too long execution of the fitness function).

- **maxNumbLaunchesAllowed**  — maximum permissible number of launches.
- **fB** — best found value of the fitness function.

- **c \[\]**  — array of current coordinates.
- **cB \[\]**  — array for storing the best search coordinates.

```
//——————————————————————————————————————————————————————————————————————————————
// Class for function optimization, inherits from CNDimensional_Func
class C_OptimizedFunction : public CNDimensional_Func
{
  public:
  C_OptimizedFunction (void) { }
  ~C_OptimizedFunction (void) { }

  // A virtual function to contain the function being optimized--------
  virtual void Func (CRowDouble &x, double &func, CObject &obj);

  // Initialization of optimization parameters---------------------------------------
  void Init (int coords,
             int maxNumberLaunchesAllowed)
  {
    numberLaunches         = 0;
    maxNumbLaunchesAllowed = maxNumberLaunchesAllowed;
    fB = -DBL_MAX;

    ArrayResize (c,  coords);
    ArrayResize (cB, coords);
  }

  //----------------------------------------------------------------------------
  CMinBLEICStateShell state;          // State
  int                 numberLaunches; // Launch counter

  double fB;                          // Best found value of the objective function (maximum)
  double cB [];                       // Coordinates of the point with the best function value

  private: //-------------------------------------------------------------------
  double c  [];                       // Array for storing current coordinates
  int    maxNumbLaunchesAllowed;      // Maximum number of function calls allowed
};
//——————————————————————————————————————————————————————————————————————————————
```

The **Func** method of the **C\_OptimizedFunction** is intended to access the user's fitness function. It takes the **x** vector as arguments (one of the variants of the optimized parameters of the problem, proposed by the optimization method), the **func** argument to accept the calculated value of the fitness function being returned and the **obj** object (the purpose of which is unclear to me, perhaps reserved for the ability to pass additional information to/from the method). The main stages of the method:

1\. The **numberLaunches** counter is increased. Its objective is to track the number of **Func** method calls.

2\. If the number of launches exceeds the permissible value of **maxNumbLaunchesAllowed**, the function sets the **func** value in **DBL\_MAX** (the maximum value of the "double" type, ALGLIB methods are designed to minimize functions, this value means the worst possible solution). Then we call the **MinBLEICRequestTermination** function. It signals the BLEIC method to stop the optimization.

3\. Next in the loop, the values are copied from the **x** vector to the **c** array. This is necessary in order to use these values to pass to the user's fitness function.

4\. The **ObjectiveFunction** function is called. It calculates the value of the objective function for the current values in the **c** array. The result is saved in **ffVal**, while the **func** value is set to the negative **ffVal** value (we are optimizing an inverted paraboloid, which needs to be maximized, and BLEIC minimizes the function, so we flip the value).

5\. If the current value of **ffVal** exceeds the previous best value of **fB**, then **fB** is updated and the **cB** array copies the current state of **c**. This allows us to keep track of the best found value of the objective function with the corresponding parameters and refer to them later if necessary.

The **Func** function implements a call to a custom fitness function and tracks the number of times it has been launched, updating the best results. It also controls the stop conditions if the number of starts exceeds a set limit.

```
//——————————————————————————————————————————————————————————————————————————————
// Implementation of the function to be optimized
void C_OptimizedFunction::Func (CRowDouble &x, double &func, CObject &obj)
{
  // Increase the function launch counter and limitation control----------------
  numberLaunches++;
  if (numberLaunches >= maxNumbLaunchesAllowed)
  {
    func = DBL_MAX;
    CAlglib::MinBLEICRequestTermination (state);
    return;
  }

  // Copy input coordinates to internal array-------------------------
  for (int i = 0; i < x.Size (); i++) c [i] = x [i];

  // Calculate objective function value----------------------------------------
  double ffVal = ObjectiveFunction (c);
  func = -ffVal;

  // Update the best solution found--------------------------------------
  if (ffVal > fB)
  {
    fB = ffVal;
    ArrayCopy (cB, c);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

After running the test script with the BLEIC algorithm to optimize the paraboloid function, we get the following result in print:

BLEIC, best result: 0.6861786206145579, number of function launches: 84022

It is worth noting that despite requests to stop optimization using the MinBLEICRequestTermination method, the algorithm continued the process and attempted to access the fitness function another 74,022 times, exceeding the limit of 10,000 runs.

Now let's try not to restrain BLEIC and let it act at its own discretion. The results were as follows:

BLEIC, best result: 1.0, number of function launches: 72017

As we can see, BLEIC is able to completely converge on the paraboloid function, but in this case it is impossible to estimate in advance the required number of runs of the target function. We will conduct full-scale testing and analysis of the results later.

The differentiation step is important in the algorithm. For example, if we use a very small step, for example 1e-16 instead of 0.00001, then the algorithm stops prematurely, essentially getting stuck, with the following result:

BLEIC, best result: 0.6615878186651468, number of function launches: 4002

### L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)

The L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) algorithm is an efficient optimization method specifically designed for solving large-scale problems where the number of variables exceeds 1,000. It is a quasi-Newton method and uses limited memory to store information about the curvature of the function, which avoids the need to explicitly store and calculate the full Hessian matrix.

The principle of the algorithm is that it constructs and refines a quadratic model of the function being optimized using the last M pairs of values and gradients. Typically M is a moderate number, from 3 to 10, which significantly reduces the computational cost to O(N·M) operations. At each iteration, the algorithm computes the gradient at the current point, determines the search direction using the stored vectors, and performs a linear search to determine the step length. If a step of the quasi-Newton method does not result in a sufficient decrease in the function value or gradient, a repeated direction adjustment occurs.

The main feature of L-BFGS is the positive definiteness of the approximate Hessian, which guarantees that the direction of the quasi-Newton method will always be the direction of descent, regardless of the curvature of the function.

How LBFGS works in basic terms:

1\. The main idea: the algorithm tries to find the minimum of the function, gradually "descending" to it, while it builds a simplified (quadratic) model of the function, which it constantly refines.

2\. How exactly does it accomplish that? It remembers the last M steps (usually 310 steps) and at each step it stores two things:

- where we were (meaning)
- where we can move (gradient)

Based on this data, the algorithm constructs an approximation of the function curvature (the Hessian matrix) and uses this approximation to determine the next step.

3\. Features: always moves "down" (towards decreasing function value), uses memory economically (stores only the last M steps) and works quickly (costs are proportional to the problem size × M).

4\. Practical example. Imagine you are walking down a mountain in the fog:

- you can only determine the direction of descent at the current point
- remember the last few steps and how the slope changed
- Based on this information, you predict where it is best to take the next step

5\. Limitations:

- it may work slower for very complex "landscapes"
- may require additional configuration to improve performance

Unlike the BLEIC method, L-BFGS allows you to directly set the limitation on the number of runs of the target function, but there is no possibility to specify boundary conditions for the parameters being optimized. The M value in the example below is set to "1", using other values did not lead to noticeable changes in the performance and behavior of the algorithm.

```
//——————————————————————————————————————————————————————————————————————————————
void OnStart ()
{
  // Initialization of optimization parameters---------------------------------------
  int numbTestFuncRuns = 10000;
  int params           = 1000;

  // Create and initialize arrays for search range bounds--------------
  double rangeMin [], rangeMax [], rangeStep;
  ArrayResize (rangeMin,  params);
  ArrayResize (rangeMax,  params);

  for (int i = 0; i < params; i++)
  {
    rangeMin  [i] = -10;
    rangeMax  [i] =  10;
  }
  rangeStep =  DBL_EPSILON;

  double x [];
  double s [];
  ArrayResize (x, params);
  ArrayResize (s, params);
  ArrayInitialize (s, 1);

  // Generate random initial parameter values in given ranges-----
  for (int i = 0; i < params; i++)
  {
    x [i] = rangeMin [i] + ((rangeMax [i] - rangeMin [i]) * rand () / 32767.0);
  }

  // Create objects for optimization-------------------------------------------
  C_OptimizedFunction  fFunc; fFunc.Init (params, numbTestFuncRuns);
  CObject              obj;
  CNDimensional_Rep    frep;
  CMinLBFGSReportShell rep;

  // Set the parameters of the L-BFGS optimization algorithm---------------------------
  double diffStep = 0.00001;
  double epsg     = 1e-16;
  double epsf     = 1e-16;

  CAlglib::MinLBFGSCreateF  (1, x, diffStep, fFunc.state);
  CAlglib::MinLBFGSSetCond  (fFunc.state, epsg, epsf, rangeStep, numbTestFuncRuns);
  CAlglib::MinLBFGSSetScale (fFunc.state, s);
  CAlglib::MinLBFGSOptimize (fFunc.state, fFunc, frep, 0, obj);
  CAlglib::MinLBFGSResults  (fFunc.state, x, rep);

  //----------------------------------------------------------------------------
  Print ("L-BFGS, best result: ", fFunc.fB, ", number of function launches: ", fFunc.numberLaunches);
}
//——————————————————————————————————————————————————————————————————————————————
```

In L-BFGS, the "state" variable type is set by CMinLBFGSStateShell.

```
//——————————————————————————————————————————————————————————————————————————————
// Class for function optimization, inherits from CNDimensional_Func
class C_OptimizedFunction : public CNDimensional_Func
{
  public: //--------------------------------------------------------------------
  C_OptimizedFunction (void) { }
  ~C_OptimizedFunction (void) { }

  // A virtual function to contain the function being optimized---------
  virtual void Func (CRowDouble &x, double &func, CObject &obj);

  // Initialization of optimization parameters----------------------------------------
  void Init (int coords,
             int maxNumberLaunchesAllowed)
  {
    numberLaunches         = 0;
    maxNumbLaunchesAllowed = maxNumberLaunchesAllowed;
    fB = -DBL_MAX;

    ArrayResize (c,  coords);
    ArrayResize (cB, coords);
  }

  //----------------------------------------------------------------------------
  CMinLBFGSStateShell state;          // State
  int                 numberLaunches; // Launch counter

  double fB;                          // Best found value of the objective function (maximum)
  double cB [];                       // Coordinates of the point with the best function value

  private: //-------------------------------------------------------------------
  double c  [];                       // Array for storing current coordinates
  int    maxNumbLaunchesAllowed;      // Maximum number of function calls allowed
};
//——————————————————————————————————————————————————————————————————————————————
```

Request the stop of the optimization process using the MinLBFGSRequestTermination command.

```
//——————————————————————————————————————————————————————————————————————————————
// Implementation of the function to be optimized
void C_OptimizedFunction::Func (CRowDouble &x, double &func, CObject &obj)
{
  //Increase the function launch counter and limitation control-------------------
  numberLaunches++;
  if (numberLaunches >= maxNumbLaunchesAllowed)
  {
    func = DBL_MAX;
    CAlglib::MinLBFGSRequestTermination (state);
    return;
  }

  // Copy input coordinates to internal array-------------------------
  for (int i = 0; i < x.Size (); i++) c [i] = x [i];

  // Calculate objective function value----------------------------------------
  double ffVal = ObjectiveFunction (c);
  func = -ffVal;

  // Update the best solution found--------------------------------------
  if (ffVal > fB)
  {
    fB = ffVal;
    ArrayCopy (cB, c);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

After running the test script with the L-BFGS algorithm to optimize the paraboloid function, we get the following result in print:

L-BFGS, best result: 0.6743844728276278, number of function launches: 24006

It is very likely that the parameter for the maximum number of runs of the objective function does not work, because the algorithm actually performed more than 2 times more runs.

Now let it perform optimization without restrictions on the number of runs:

L-BFGS, best result: 1.0, number of function launches: 52013

Like BLEIC, L-BFGS is capable of fully converging on the paraboloid function, but the number of launches becomes uncontrollable. In the review of the next algorithm, we will show that this can be a really big problem if this nuance is not taken into account.

For L-BFGS, the differentiation step is also important. If you use a very small step of 1e-16, the algorithm stops prematurely getting stuck with this result:

L-BFGS, best result: 0.6746423814003036, number of function launches: 4001

### NS (Nonsmooth Nonconvex Optimization Subject to box / linear / nonlinear - Nonsmooth Constraints)

NS (Nonsmooth Nonconvex Optimization Subject to box/linear/nonlinear - Nonsmooth Constraints) is a non-smooth non-convex optimization algorithm designed to solve problems where the objective function is not smooth and convex. This means that the function may have sharp changes, angles, or other features. The main characteristics of such problems are that the objective function may contain discontinuities or abrupt changes that make analysis difficult using gradient-based methods.

The principles of the algorithm include gradient estimation, which uses a gradient sampling method that involves estimating the gradient at several random points around the current solution. This helps to avoid problems related to the peculiarities of the function. Based on the obtained gradient estimates, a limited quadratic programming (QP) problem is formed. The solution allows us to determine the direction, in which to move to improve the current solution. The algorithm works iteratively, updating the current solution at each iteration based on the calculated gradients and the solution to the QP problem.

Let's consider what this description of optimization means in simple terms:

1\. NONSMOOTH (non-smooth optimization):

- The function may have breaks or "fissures"
- No requirement for continuous differentiability
- There may be sharp transitions and jumps

2\. NONCONVEX (non-convex):

- The function may have several local minima and maxima
- The "landscape" of the function may feature "hills" and "valleys"

3\. Constraint types (CONSTRAINTS): BOX, LINEAR, NONLINEAR-NONSMOOTH (described above).

The peculiarity of this method is the need to specify and set parameters for the AGS solver (adaptive gradient sampling method). This solver is designed to solve non-smooth problems with boundary, linear and non-linear constraints. The AGS solver includes several important features: special constraint handling, variable scaling (to handle poorly scalable problems), and built-in support for numerical differentiation.

The most important limitation of the AGS solver is that it is not designed for high-dimensional problems. One step of AGS requires approximately 2·N gradient evaluations at randomly chosen points (in comparison, L-BFGS requires O(1) evaluations per step). Typically you will need O(N) iterations to converge, resulting in O(N²) gradient estimates per optimization session.

Unlike the previous methods, NS requires the use of CRowDouble type for boundary condition variables instead of double type and initial values of the optimized parameters of the problem. Additionally, you need to specify the parameters for the AGS solver.

```
//——————————————————————————————————————————————————————————————————————————————
void OnStart ()
{
  // Initialization of optimization parameters---------------------------------------
  int numbTestFuncRuns = 10000;
  int params           = 1000;

  // Additionally, you need to specify --------------
  CRowDouble rangeMin, rangeMax;
  rangeMin.Resize (params);
  rangeMax.Resize (params);
  double rangeStep;

  for (int i = 0; i < params; i++)
  {
    rangeMin.Set (i, -10);
    rangeMax.Set (i,  10);
  }
  rangeStep = DBL_EPSILON;

  CRowDouble x, s;
  x.Resize (params);
  s.Resize (params);
  s.Fill (1);

  // Generate random initial parameter values in given ranges----
  for (int i = 0; i < params; i++)
  {
    x.Set (i, rangeMin [i] + ((rangeMax [i] - rangeMin [i]) * rand () / 32767.0));
  }

  // Create objects for optimization------------------------------------------
  C_OptimizedFunction fFunc; fFunc.Init (params, numbTestFuncRuns);
  CObject             obj;
  CNDimensional_Rep   frep;
  CMinNSReport        rep;

  // Set the parameters of the NS optimization algorithm------------------------------
  double diffStep = 0.00001;
  double radius   = 0.8;
  double rho      = 50.0;

  CAlglib::MinNSCreateF    (x, diffStep, fFunc.state);
  CAlglib::MinNSSetBC      (fFunc.state, rangeMin, rangeMax);
  CAlglib::MinNSSetScale   (fFunc.state, s);
  CAlglib::MinNSSetCond    (fFunc.state, rangeStep, numbTestFuncRuns);

  CAlglib::MinNSSetAlgoAGS (fFunc.state, radius, rho);

  CAlglib::MinNSOptimize   (fFunc.state, fFunc, frep, obj);
  CAlglib::MinNSResults    (fFunc.state, x, rep);

  // Output of optimization results-----------------------------------------------
  Print ("NS, best result: ", fFunc.fB, ", number of function launches: ", fFunc.numberLaunches);
}
//——————————————————————————————————————————————————————————————————————————————
```

For the NS method, a wrapper class must be created, now inherited from another parent class — CNDimensional\_FVec. We will also need to change the virtual method to FVec. The notable feature of the method is the fact that it is impossible to return the fitness function value equal to DBL\_MAX, since in this case the method will end with an error, unlike the previous optimization methods. So we will add an additional field to the class (fW) to track the worst-case solution during the optimization.

```
//——————————————————————————————————————————————————————————————————————————————
// Class for function optimization, inherits from CNDimensional_FVec
class C_OptimizedFunction : public CNDimensional_FVec
{
  public: //--------------------------------------------------------------------
  C_OptimizedFunction (void) { }
  ~C_OptimizedFunction (void) { }

  // A virtual function to contain the function being optimized--------
  virtual void FVec (CRowDouble &x, CRowDouble &fi, CObject &obj);

  // Initialization of optimization parameters---------------------------------------
  void Init (int coords,
             int maxNumberLaunchesAllowed)
  {
    numberLaunches         = 0;
    maxNumbLaunchesAllowed = maxNumberLaunchesAllowed;
    fB = -DBL_MAX;
    fW =  DBL_MAX;

    ArrayResize (c,  coords);
    ArrayResize (cB, coords);
  }

  //----------------------------------------------------------------------------
  CMinNSState state;             // State
  int         numberLaunches;    // Launch counter

  double fB;                     // Best found value of the objective function (maximum)
  double fW;                     // Worst found value of the objective function (maximum)
  double cB [];                  // Coordinates of the point with the best function value

  private: //-------------------------------------------------------------------
  double c  [];                  // Array for storing current coordinates
  int    maxNumbLaunchesAllowed; // Maximum number of function calls allowed
};
//——————————————————————————————————————————————————————————————————————————————
```

Incorrect actions are shown in red. Instead, we will return the worst solution found during optimization (with a minus sign, because the method works for minimization). And of course, we need to change the method stop call method to MinNSRequestTermination.

```
//——————————————————————————————————————————————————————————————————————————————
void C_OptimizedFunction::FVec (CRowDouble &x, CRowDouble &fi, CObject &obj)
{
  // Increase the function launch counter and limitation control----------------
  numberLaunches++;
  if (numberLaunches >= maxNumbLaunchesAllowed)
  {
    //fi.Set (0, DBL_MAX);  //Cannot return DBL_MAX value
    fi.Set (0, -fW);
    CAlglib::MinNSRequestTermination (state);
    return;
  }

  // Copy input coordinates to internal array-------------------------
  for (int i = 0; i < x.Size (); i++) c [i] = x [i];

  // Calculate objective function value----------------------------------------
  double ffVal = ObjectiveFunction (c);
  fi.Set (0, -ffVal);

  // Update the best and worst solutions found-----------------------------
  if (ffVal < fW) fW = ffVal;
  if (ffVal > fB)
  {
    fB = ffVal;
    ArrayCopy (cB, c);
  }
}
//——————————————————————————————————————————————————————————————————————————————
```

After running the test script with the NS algorithm to optimize the paraboloid function, we get the following result in print:

NS, best result: 0.6605212238333136, number of function launches: 1006503

It seems that the parameter of the maximum allowed number of runs of the objective function does not work for NS either, because the algorithm actually performed more than 1 million launches.

Now let's try not to limit NS and let it perform optimization without restrictions on the number of runs. Unfortunately, I could not wait for the script to finish running and I had to forcefully stop its work by closing the chart:

No result

The differentiation step is also important for NS. If we use a very small step of 1e-16, the algorithm stops prematurely - it gets stuck without using the number of runs of the target function allocated for its work, with the following result:

NS, best result: 0.6784901840822722, number of function launches: 96378

### Conclusion

In this article, we got acquainted with optimization methods from the ALGLIB library. We have considered the key features of these methods, knowledge of which allows not only to carry out the optimization itself, but also helps to avoid unpleasant situations, such as an uncontrolled number of calls to the target function.

In the next, final article on ALGLIB optimization methods, we will examine three more methods in detail. We will test all the methods considered, which will allow us to identify their strengths and weaknesses in practical application, as well as summarize the results of our research. In addition, we traditionally visualize the operation of algorithms to clearly demonstrate their characteristic behavior on complex test problems. This will give us a better understanding of how each method handles different optimization challenges.

The text of the article presents fully working script codes for running ALGLIB methods. In addition, in the archive you will find everything you need to start using the methods discussed to optimize your trading strategies and other tasks right now. Thus, the goal of the article - to show simple and clear examples of the use of methods - has been achieved.

I would like to express my special gratitude to [Evgeniy Chernish](https://www.mql5.com/en/users/vp999369), who helped me understand the specifics of accessing ALGLIB library methods.

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Simple test ALGLIB BLEIC.mq5 | Script | Test script for working with BLEIC |
| 2 | Simple test ALGLIB LBFGS.mq5 | Script | Test script for working with L-BFGS |
| 3 | Simple test ALGLIB NS.mq5 | Script | Test script for working with NS |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16133](https://www.mql5.com/ru/articles/16133)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16133.zip "Download all attachments in the single ZIP archive")

[ALGLIB\_optimization\_methods.zip](https://www.mql5.com/en/articles/download/16133/alglib_optimization_methods.zip "Download ALGLIB_optimization_methods.zip")(4.36 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/488028)**
(1)


![Rorschach](https://c.mql5.com/avatar/2022/3/6244E941-EF6F.jpg)

**[Rorschach](https://www.mql5.com/en/users/rorschach)**
\|
25 Oct 2024 at 18:03

Thanks, that's interesting.


![From Basic to Intermediate: Array (III)](https://c.mql5.com/2/99/Do_b4sico_ao_intermedierio__Array_III__LOGO.png)[From Basic to Intermediate: Array (III)](https://www.mql5.com/en/articles/15473)

In this article, we will look at how to work with arrays in MQL5, including how to pass information between functions and procedures using arrays. The purpose is to prepare you for what will be demonstrated and explained in future materials in the series. Therefore, I strongly recommend that you carefully study what will be shown in this article.

![Automating Trading Strategies in MQL5 (Part 18): Envelopes Trend Bounce Scalping - Core Infrastructure and Signal Generation (Part I)](https://c.mql5.com/2/146/18269-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 18): Envelopes Trend Bounce Scalping - Core Infrastructure and Signal Generation (Part I)](https://www.mql5.com/en/articles/18269)

In this article, we build the core infrastructure for the Envelopes Trend Bounce Scalping Expert Advisor in MQL5. We initialize envelopes and other indicators for signal generation. We set up backtesting to prepare for trade execution in the next part.

![Price Action Analysis Toolkit Development (Part 25): Dual EMA Fractal Breaker](https://c.mql5.com/2/147/18297-price-action-analysis-toolkit-logo.png)[Price Action Analysis Toolkit Development (Part 25): Dual EMA Fractal Breaker](https://www.mql5.com/en/articles/18297)

Price action is a fundamental approach for identifying profitable trading setups. However, manually monitoring price movements and patterns can be challenging and time-consuming. To address this, we are developing tools that analyze price action automatically, providing timely signals whenever potential opportunities are detected. This article introduces a robust tool that leverages fractal breakouts alongside EMA 14 and EMA 200 to generate reliable trading signals, helping traders make informed decisions with greater confidence.

![Neural Networks in Trading: Market Analysis Using a Pattern Transformer](https://c.mql5.com/2/97/Market_Situation_Analysis_Using_Pattern_Transformer___LOGO.png)[Neural Networks in Trading: Market Analysis Using a Pattern Transformer](https://www.mql5.com/en/articles/16130)

When we use models to analyze the market situation, we mainly focus on the candlestick. However, it has long been known that candlestick patterns can help in predicting future price movements. In this article, we will get acquainted with a method that allows us to integrate both of these approaches.

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/16133&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068800363047353813)

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