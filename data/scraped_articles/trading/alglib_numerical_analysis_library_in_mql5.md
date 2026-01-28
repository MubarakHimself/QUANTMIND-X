---
title: ALGLIB numerical analysis library in MQL5
url: https://www.mql5.com/en/articles/13289
categories: Trading
relevance_score: 3
scraped_at: 2026-01-23T18:03:40.426109
---

[![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/01.png)![](https://www.mql5.com/ff/sh/20jc81m23z78s5z9z2/02.png)Create your own AI for tradingRead our book "Neural Networks in Algo Trading with MQL5"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/neurobook%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.neurobook%26utm_content=visit.page%26utm_campaign=neurobook.promo.04.2024&a=elbyupbppbqpzzvzhxtydvlupfcbmnmb&s=0d2f8feb92df3772a11aca1f195d2996b59d6539e283cdf4a18ccff02e5ad43d&uid=&ref=https://www.mql5.com/en/articles/13289&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068998850665971658)

MetaTrader 5 / Examples


### Introduction

Financial markets generate data with a huge amount of complex relationships. To analyze them, we need to use the most modern methods of applied mathematics. Successfully combining the high complexity of financial data with the simplicity and efficiency of analysis is a challenging task. ALGLIB is a high-performance library designed specifically for working with numerical methods and data analysis algorithms. It is a reliable assistant in the analysis of financial markets.

**ALGLIB versatility**

Today, ALGLIB is recognized as one of the best libraries for working with numerical methods. It supports several programming languages (C++, C#, Java, Python, Delphi) and operating systems (Windows and POSIX, including Linux).

Among the many features of ALGLIB, the following stand out:

- Linear algebra: Includes direct algorithms, eigenvalue (EVD) and singular value (SVD) methods, which are important for data analysis.
- Solving equations: Supports both linear and nonlinear systems of equations, which is useful when modeling complex financial processes.
- Interpolation and approximation: There is support for various methods of data interpolation and approximation, which is useful in analyzing and forecasting market trends.
- Optimization: ALGLIB provides algorithms to find optimal solutions, which is important for optimizing an investment portfolio and other financial decisions.
- Numerical integration: Allows the calculation of definite integrals, which is useful in assessing financial instruments.
- Machine learning: Involves data analysis, classification, regression and even the use of neural networks, which opens up new possibilities for predicting market trends.

**ALGLIB advantages**

Why choose ALGLIB when working with financial data?

Here are the key benefits of the library:

- Portability: ALGLIB compiles easily on a variety of platforms using a variety of compilers making it accessible to developers of varying backgrounds.
- Ease of use: Support for multiple programming languages so you can choose the language you are most comfortable with, without having to learn new syntax.
- Open source: ALGLIB is open source and can be used under GPL 2+ terms. This makes it accessible for both scientific research and commercial projects.
- Commercial user support: Commercial users can purchase a license that provides them with legal protection when using ALGLIB.

Besides, the library contains the large collection of test cases covering the major part of the proposed methods' functionality. That will allow you to carry the tests and [report detected errors](https://www.mql5.com/go?link=http://bugs.alglib.net/my_view_page.php "http://bugs.alglib.net/my_view_page.php") to the project's authors. More detailed information about the library can be found on the project website [https://www.alglib.net/](https://www.mql5.com/go?link=https://www.alglib.net/ "https://www.alglib.net/")

ALGLIB was first adapted for use in the MQL5 language in 2012. This adaptation was a conversion of the library from version 3.5. More than 10 years have passed since then. During this time, ALGLIB has become [widely known among developers and analysts](https://www.mql5.com/en/search#!keyword=Alglib&module=mql5_module_articles) in the field of trading in financial markets. Over the years, the development team has worked actively to improve the library, making more than 70 changes, including the addition of new classes, functions and improvements.

It should also be noted that the existing library classes have been redesigned to use [matrices and vectors](https://www.mql5.com/en/docs/matrix), and new functionality introduced in ALGLIB 3.19 has been added. ['Complex' data type](https://www.mql5.com/en/docs/basis/types/complex) is now used to work with complex numbers. All source code has been revised and structured in accordance with the unified design style.

Unfortunately, the changes made to the ALGLIB library in version 3.19 for MQL5 were quite significant, and therefore backward compatibility is not provided. Users applying ALGLIB version 3.5 in their projects are advised to thoroughly review their programs and make any necessary adjustments.

In addition to the libraries themselves, the test scripts have also been updated. Now there are 91 of them for classes and 152 for interfaces. This facilitates more reliable and comprehensive testing of the library functionality.

The new version of ALGLIB is available here: [ALGLIB - Numerical Analysis Library - library for MetaTrader 5](https://www.mql5.com/en/code/1146), as well as part of the MetaTrader 5 platform (\\MQL5\\Include\\Math\\Alglib\\, including test cases in \\MQL5\\Scripts\\UnitTests\\Alglib\\).

### 1\. ALGLIB 3.19 what's new (list of changes since version 3.5)

> 3.19:
>
> - The most important feature in ALGLIB 3.19.0 is the new RBF (Radial Basis Function) solver for handling large data sets, which supports thin-plate splines, biharmonic splines and multiquadrics. This solver supports both interpolation and fitting (smoothing) problems;
> - The new RBF algorithm uses the domain decomposition method to solve linear systems. As a result, it has O(N) memory requirements and O(N2) execution time, which is a significant improvement over the O(N2) memory requirements and O(N3) execution time required by other open source implementations. It can be used for datasets with more than 100,000 points.
>
> 3.18:
>
> - Added Sparse GMRES solver for linear equations;
> - Improved performance of the AMD algorithm (when optimizing the processing of matrices with dense rows);
> - Improved the speed of the interior point solvers for linear programming (LP) and quadratic programming (QP) thanks to the new Cholesky decomposition and extensive optimization of the solver code.
>
> 3.17:
>
> - Added Sparse Supernodal Cholesky decomposition (with expert functions and user-friendly wrappers) and corresponding Sparse Direct Linear Solver. Enabled permutation padding reduction and undefined factorizations;
> - Added a solver for large-scale interior point linear programming (LP) problems;
> - Added a solver for large-scale interior point semidefinite quadratic programming (QP) problems.
>
> 3.16:
>
> - Implemented a solver for interior point quadratic programming (QP) problems with dense and sparse versions;
> - Added a new subroutine for fast fitting of penalized cubic splines with O(N\*logN) execution time;
> - Added a new SQP solver for nonlinear programming;
> - Introduced a compressed binary storage format for large random forests (reducing memory usage by 3.7-5.7 times);
> - Added sparsegemv() function for CRS and Skyline matrices;
> - Implemented CDF and PDF bivariate normal functions.
> - QP solvers now report Lagrange multipliers;
> - QP solvers now support two-way linear constraints;
> - Improved stability of the SLP solver;
> - Improved reference element selection in the LP solver for a more accurate solution.
>
> 3.15:
>
> - Implemented the Singular Spectrum Analysis (SSA, "caterpillar") algorithm for time series. The implementation is optimized and includes trend extraction, prediction, averaging prediction and fast incremental model updates;
> - Added direct solver for sparse linear systems stored in Skyline (SKS) format;
> - Improved the performance of the QP-DENSE-AUL solver for quadratic problems with a huge number of inequalities;
> - Significantly increased the speed of BLEIC and QP-BLEIC solvers (up to four times). Revised the internal code of the solvers resulting in significantly improved constraint handling;
> - Added thermal triggering support for sub-dimensional eigenvalue solvers. When solving a sequence of related eigenvalue problems, it is possible to reuse a previously found basic solution as a starting point for a new solving session;
> - Simplified the creation of striped matrices in the SKS format (sparsecreatesksband() function);
> - Added a new set of BLAS level 2 functions for real numbers: GEMV, SYMV, TRSV;
> - The sparseset() function now supports SKS matrices;
> - The minqp solver now supports auto calculation of variable scales based on the diagonal of the quadratic term.
>
> 3.12:
>
> - The rbfsetpoints() function now checks for the presence of NAN/INF in the input data;
> - k-means clustering and eigenvalue back iteration algorithms now use deterministic seed values for initialization making results reproducible across different runs;
> - Fixed a small bug in the QQP solver - incorrect automatic scaling of the quadratic term.
>
> 3.11:
>
> - Added the ability to perform linearly constrained nonlinear least squares (MinLM and LSFit). Now it is possible to carry out nonlinear approximation with linear constraints on the parameters;
> - Added support for approximate smallest circumscribing, minimum area and maximum inscribed N-spheres for data (in 2D - circumscribing circles, in 3D - inscribed spheres);
> - Improved the stability of the MinNLC solver and added another preprocessing mode - "precise stable";
> - Added a new optimizer - MinBC with restrictions only on variables in the "active" state. These restrictions allow activation strategies that are not possible with general linear constraints;
> - Added streaming serialization/deserialization in ALGLIB versions for C# and C++;
> - Implemented direct/sparse/out-of-order eigenvalue solvers using subspace method and fast truncated principal component analysis (PCA) using subspace method;
> - Improved hierarchical RBFs with parallelism support - several orders of magnitude faster on some data sets and able to process more than 100,000 points;
> - Added linearly constrained quadratic programming (QP) problem solver;
> - Improved kd trees - with queries over rectangular areas and thread-safe versions of the query functions.
>
> 3.10:
>
> - Added CSV import functionality - now it is possible to read 2D matrices from CSV files;
> - Introduced the AGS (Adaptive Gradient Sampling) algorithm to optimize nonlinear, unsmoothed and inconsistent constrained problems making ALGLIB one of the few commercial packages that supports unsmoothed optimization;
> - Added the Ward method for hierarchical clustering;
> - Implemented lightweight linear solvers without condition number estimation and iterative correction - they are many times faster than their "functionality-rich" analogues.
>
> 3.9:
>
> - Significant improvements in sparse/dense linear algebra support: SKS sparse matrix storage format, linear algebra operations for SKS-based matrices, Cholesky factorizer for SKS, many additional functions for sparse matrices;
> - Improvements in solvers and optimizers: a new solver for a limited quadratic programming problem with constraints on variables - QuickQP, nonlinear Augmented Lagrangian optimizer, improved BLEIC optimizer, polynomial solver and many other minor improvements;
> - Added additional interpolation/fitting functions: logistic curve fitting with 4/5 parameters, Ramer-Douglas-Peucker (RDP) algorithm;
> - Improved the speed of the linear discriminant analysis (LDA) algorithm.
>
> 3.8:
>
> - Added ranking functionality (descriptive statistics) - the function that replaces data with their ranks;
> - Introduced a new solver, QP-BLEIC, capable of solving sparse and inconsistent quadratic programming problems with boundary and linear constraints;
> - Improved FFT performance (more performance, but still single threaded);
> - Multiple minor improvements (steps up in BLEIC optimizer, better weights initialization for MLP, Akima spline for less than five points).
>
> 3.7:
>
> - Significantly redesigned the BLEIC optimizer. First, it uses a new three-stage active set algorithm proposed by Elvira Illarionova, which combines gradient projection with iterations of L-BFGS constraint equality. Second, since the algorithm no longer has nested outer/inner iterations, it is possible to set more transparent stopping criteria for the algorithm. Third, it uses a new constraint activation/deactivation strategy that handles degenerate constraints correctly;
> - Significantly improved support for neural networks in ALGLIB. Introduced a new training interface that greatly simplifies the training of multiple networks with the same settings and data. It is now possible to specify the training set using a sparse matrix;
> - Improved clustering support - the new version of ALGLIB includes the hierarchical cluster analysis algorithm from the clustering subpackage. This algorithm includes several distance metrics (Euclidean, 1-norm, infinity-norm, Pearson-Spearman correlation-based metrics, cosine distance) and several link types (single link, full link, average link). The K-means clustering functionality (which existed long before the new algorithm) was combined with the new clustering algorithm;
> - Sparse linear solvers (CG and LSQR) now support automatic diagonal preprocessor;
> - Linear/nonlinear least squares solvers (lsfit subpackages) now report errors in ratios;
> - The sparse matrix functionality now includes new functions for converting between hash table and CRS views, as well as performance improvements to the SparseRewriteExisting function.
>
> 3.6:
>
> - The quadratic optimizer now supports an arbitrary combination of boundary and linear equalities/inequalities. The new version of the optimizer uses a combination of the augmented Lagrange method and the active set method;
> - The Spline1D module now supports interpolation with monotonic cubic splines;
> - Added support for vector bilinear and bicubic splines;
> - Added support for scalar and vector trilinear (3D) splines;
> - Improved support for sparse matrices: efficient enumeration of non-zero elements using the SparseEnumerate() function, faster SparseGet() for matrices stored in CRS format;
> - Optimization and nonlinear approximation algorithms (subpackages LSFit, MinLM, MinBLEIC, MinLBFGS, MinCG) can check the validity of the user-supplied gradient (the most common error in numerical programs).

### 2\. ALGLIB library in scientific research

With open source availability and free use for non-commercial projects, ALGLIB has become an important tool in the world of scientific research. It is successfully used to solve diverse and complex problems.

ALGLIB's significant impact is evident in the development of custom software, where many algorithms from ALGLIB become the basis for creating innovative solutions.

In addition, ALGLIB's single-threaded algorithms serve as a benchmark and reference point for the development of parallel versions of iterative algorithms designed to solve huge systems of equations containing about 2 million equations. This demonstrates the reliability and outstanding efficiency of the algorithms provided by the ALGLIB library.

Researchers have noted the high quality and efficiency of ALGLIB's algorithms, making it the preferred choice for solving complex research problems in a variety of fields, including basic research and engineering.

**2.1. ALGLIB in optimization problems**

> 2014: Hugo J. Kuijf, Susanne J. van Veluw, Mirjam I. Geerlings, Max A. Viergever, Geert Jan Biessels & Koen L. Vincken _._ [Automatic Extraction of the Midsagittal Surface from Brain MR Images using the Kullback–Leibler Measure](https://www.mql5.com/go?link=https://link.springer.com/article/10.1007/s12021-013-9215-0 "https://link.springer.com/article/10.1007/s12021-013-9215-0"). _Neuroinform_ **12**, 395–403 (2014)
>
> > The midsagittal plane computed in the previous section was used to initialize the computation of the midsagittal surface. The midsagittal surface was represented as a bicubic spline, as implemented in ALGLIB (Bochkanov and Bystritsky (2012)). Control points for the spline were placed in a regular grid on the computed midsagittal plane, with distance m between the control points.
>
> 2017: Vadim Bulavintsev, [A GPU-enabled Black-box Optimization in Application to Dispersion-based Geoacoustic Inversion](https://www.mql5.com/go?link=https://ceur-ws.org/Vol-1987/paper15.pdf "https://ceur-ws.org/Vol-1987/paper15.pdf"), 2017, in Yu. G. Evtushenko, M. Yu. Khachay, O. V. Khamisov, Yu. A. Kochetov, V.U. Malkova, M.A. Posypkin (eds.): Proceedings of the OPTIMA-2017 Conference, Petrovac, Montenegro, 02-Oct-2017
>
> > For reference, we provide the double-precision version based on the AlgLib library \[Bochkanov & Bystritsky, 2016\], that we used in our previous work \[Zaikin et al., press\]. The AlgLib library includes a state-of-the-art implementation of the bisection algorithm, thoroughly tuned to produce the most accurate results possible with the modern CPUs floating point units (FPUs). Our CPU-based implementation of bisection algorithm can not boast such accuracy.However, it is notably faster than AlgLib due to its simplicity. That is why, our and AlgLib-based algorithm’s results are different. The discrepancy between the outputs (residue) of the same algorithm on the CPU and GPU is the result of the different implementations of floating-point units on these platforms.

**2.2. ALGLIB in interpolation problems**

> 2021: Jasek K., Pasternak M., Miluski W., Bugaj J., Grabka M, [Application of Gaussian Radial Basis Functions for Fast Spatial Imaging of Ground Penetration Radar Data Obtained on an Irregular Grid.](https://www.mql5.com/go?link=https://www.mdpi.com/2079-9292/10/23/2965 "https://www.mdpi.com/2079-9292/10/23/2965")Electronics 2021, 10, 2965.
>
> > The ALGLIB package implements both kinds of RBFs: the global Gaussian function and compactly supported. The classical Gaussian function takes small values already at a distance of about 3R0 from the center and can easily be modified to be compactly supported.
> >
> > In this paper, the RBF-ML algorithm implemented in the ALGLIB package was used. It has three parameters: the initial radius, R0, the number of layers, NL, and the regularisation coefficient, l. This algorithm builds a hierarchy of models with decreasing radii \[13\]. In the initial (optional) iteration, the algorithm builds a linear least squares model. The values predicted by the linear model are subtracted from the function values at the nodes, and the residual vector is passed to the next iteration. In the first iteration, a traditional RBF model with a radius equal to R0 is built. However, it does not use a dense solver and does not try to solve the problem exactly. It solves the least squares problem by performing a fixed number (approximately 50) of LSQR \[22\] iterations. Usually, the first iteration is sufficient. Additional iterations will not improve the situation because with such a large radius, the linear system is ill-conditioned. The values predicted by the first layer of the RBF model are subtracted from the function values at the nodes, and, again, the residual vector is passed to the next iteration. With each successive iteration, the radius is halved by performing the same constant number of LSQR iterations, and the forecasts of the new models are subtracted from the residual vector.
> >
> > In all subsequent iterations, a fine regularisation can be applied to improve the convergence of the LSQR solver. Larger values for the regularisation factor can help reduce data noise. Another way of controlled smoothing is to select the appropriate number of layers. Figure 2 shows an example of B-scan and a method of approximation by a hierarchical model. Subsequent layers have a radius that is twice as small and explain the residues after the previous layer. As the radius decreases, the finer details of the B-scan are reproduced.
> >
> > The hierarchical algorithm has several significant advantages:
> >
> > • Gaussian CS-RBFs produce linear systems with sparse matrices, enabling the use of the sparse LSQR solver, which can work with the rank defect matrix;
> >
> > • The time of the model building depends on the number of points, N, as N logN in contrast to simple RBF’s implementations with O(N3) efficiency;
> >
> > • An iterative algorithm (successive layers correct the errors of the previous ones) creates a robust model, even with a very large initial radius. Successive layers have smaller radii and correct the inaccuracy introduced by the previous layer;
> >
> > • The multi-layer model allows for control smoothing both by changing the regularisation coefficient and by a different number of layers.
> >
> > The presented hierarchical RBF approximation algorithm is one of the most efficient algorithms for processing large, scattered data sets. Its implementation, located in the ALGLIB library, enables simple software development, which can be successfully used to analyse GPR images.
>
> 2022: Ruiz M., Nieto J., Costa V., Craciunescu T., Peluso E., Vega J, Murari A, JET Contributors, [Acceleration of an Algorithm Based on the Maximum Likelihood Bolometric Tomography for the Determination of Uncertainties in the Radiation Emission on JET Using Heterogeneous Platforms](https://www.mql5.com/go?link=https://www.mdpi.com/2076-3417/12/13/6798 "https://www.mdpi.com/2076-3417/12/13/6798"). Appl. Sci. 2022, 12, 6798\.
>
> > ALGLIB is a numeric library focused on solving general numerical problems. It can be used with different programming languages such as C++, C#, and Delphi. It offers a great variety of functions for different science fields. In this specific application, it is required to interpolate the 2D data arrays that can or cannot be equally spaced (nonuniformly distributed). The development of the function implementing the equivalent to griddata requires the use of ALGLIB 2D interpolation functions for sparse/non-uniform data. For the fitting part, the least square solver function is used, for which two options are available: BlockLLS or FastDDDM. The FastDDDM option was chosen to achieve the best possible performance.
> >
> > ArrayFire provides a complete API that solves the most common functionalities implemented with MATLAB language. Therefore, it can be considered that porting MATLAB to C++ using ArrayFire API is relatively straightforward, and some parts of the code are even equivalent line by line. Nevertheless, ArrayFire does not include some powerful functions available in MATLAB. For example, the function “griddata” allows different types of interpolations using uniform and not-uniform input data distribution. This function in MATLAB has some parts of the internal code visible to the user, but other parts are not available, making it impossible to reproduce its calculations. While ArrayFire version 3.8.0 includes a function for interpolation, it expects that input data will be uniformly organized. To solve this problem, we chose the open-source library ALGLIB, which provides a set of functions for 2D interpolation that can be used to circumvent the problem.
> >
> > While most of the code and functions of the algorithm in MATLAB were translated into C++ and optimized, others could not be translated directly. The reason is that there is no information about the internal calculations of some of the functions in MATLAB. This implies that the results obtained in both implementations are slightly different. These differences are mainly evident in the implementation of the griddata function.For this application, it has been used with the “bicubic splines” interpolation method, incorporated in the FastDDM solver belonging to the ALGLIB library.

**2.3. ALGLIB algorithms as a benchmark for comparison**

> 2015: Tarek Ibn Ziad, M. & Alkabani, Yousra & El-Kharashi, M. W. & Salah, Khaled & Abdelsalam, Mohamed. (2015). [Accelerating electromagnetic simulations: A hardware emulation approach](https://www.mql5.com/go?link=https://ieeexplore.ieee.org/document/7440386 "https://ieeexplore.ieee.org/document/7440386"). 10.1109/ICECS.2015.7440386.
>
> > ...Ibn Ziad et al. implemented a Jacobi iterative solver on a physical hardware emulation platform to accelerate the finite element solver of an EM simulator \[5\]. They demonstrated the efficiency of their solution via implementing a twodimensional (2D) edge element code for solving Maxwell’s equations for metamaterials using FEM. Their design achieved 101x speed-up over the same pure software implementation on MATLAB \[13\]and 35x over the best iterative software solver from ALGLIB C++ library \[14\] in case of solving 2 million equations.
> >
> > In this paper, we present a scalable architecture that can efficiently accelerate the solver core of an EM simulator. The architecture is implemented on a physical hardware emulation platform and is compared to the state-of-the-art solvers. Experimental results show that the proposed solver is capable of 522x speed-up over the same pure software implementation on Matlab, 184x speed-up over the best iterative software solver from the ALGLIB C++ library, and 5x speed-up over another emulation-based hardware implementation from the literature, solving 2 million equations.
>
> 2016: Liu, Yongchao & Pan, Tony & Aluru, Srinivas (2016), [Parallel Pairwise Correlation Computation On Intel Xeon Phi Clusters](https://www.mql5.com/go?link=https://ieeexplore.ieee.org/document/7789334 "https://ieeexplore.ieee.org/document/7789334").
>
> > Using both artificial and real gene expression datasets, we have compared LightPCC to two CPU-based counterparts: a sequential C++ implementation in ALGLIB (http://www.alglib.net) and an implementation based on a parallel GEMM routine in Intel Math Kernel Library(MKL). Our experimental results showed that by using one 5110P Phi and 16 Phis, LightPCC is able to run up to 20.6× and 218.2× faster than ALGLIB, and up to 6.8× and 71.4× faster than singled-threaded MKL, respectively.

**2.4. ALGLIB algorithms as part of specialized software**

> 2015: Kraff S, Lindauer A, Joerger M, Salamone SJ, Jaehde U. [Excel-Based Tool for Pharmacokinetically Guided Dose Adjustment of Paclitaxel. Ther Drug Monit.](https://www.mql5.com/go?link=https://pubmed.ncbi.nlm.nih.gov/25774704/ "https://pubmed.ncbi.nlm.nih.gov/25774704/") 2015 Dec;37(6):725-32
>
> > Methods: Population PK parameters of paclitaxel were taken from a published PK model. An Alglib VBA code was implemented in Excel 2007 to compute differential equations for the paclitaxel PK model. Maximum a posteriori Bayesian estimates of the PK parameters were determined with the Excel Solver using individual drug concentrations. Concentrations from 250 patients were simulated receiving 1 cycle of paclitaxel chemotherapy. Predictions of paclitaxel Tc > 0.05 μmol/L as calculated by the Excel tool were compared with NONMEM, whereby maximum a posteriori Bayesian estimates were obtained using the POSTHOC function.
>
> 2017: Hogland, John & Anderson, Nathaniel. (2017). [Function Modeling Improves the Efficiency of Spatial Modeling Using Big Data from Remote Sensing](https://www.mql5.com/go?link=https://www.mdpi.com/2504-2289/1/1/3 "https://www.mdpi.com/2504-2289/1/1/3"). Big Data and Cognitive Computing.1.3.
>
> > While the statistical and machine learning transformations can be used to build surfaces and calculate records within a tabular field, they do not in themselves define the relationships between response and explanatory variables like a predictive model. To define these relationships, we built a suite of classes that perform a wide variety of statistical testing and predictive modeling using many of the optimization algorithms and mathematical procedures found within ALGLIB and Accord.net \[15,16\]
> >
> > Furthermore, we introduce a new coding library that integrates Accord.NET and ALGLIB numeric libraries and uses lazy evaluation to facilitate a wide range of spatial, statistical, and machine learning procedures within a new GIS modeling framework called function modeling. Results from simulations show a 64.3% reduction in processing time and an 84.4% reduction in storage space attributable to function modeling. In an applied case study, this translated to a reduction in processing time from 2247 h to 488 h and a reduction is storage space from 152 terabytes to 913 gigabytes.

### 3\. ALGLIB library for financial market analysis

The first version of the library for MQL5 (ALGLIB 3.5) has become widely used in the analysis of financial data, solving various problems using modern algorithms.

Below is a list of articles actively applying classes and functions from the ALGLIB library:

- [Grokking market "memory" through differentiation and entropy analysis](https://www.mql5.com/en/articles/6351)

Logistic regression algorithm (CLogitModel class)

- [Statistical receipts for a trader: Hypotheses](https://www.mql5.com/en/articles/1240)

Hypotheses testing:

Wilcoxon signed rank test (CWilcoxonSignedRank class)

Mann-Whitney U test (CMannWhitneyU class)

Spearman rank correlation coefficient significance test (SpearmanRankCorrelationSignificance)

- [The Role of statistical distributions in trader's work](https://www.mql5.com/en/articles/257)

Hypothesis testing: Jarque-Bera test (CJarqueBera class)

- [Neural network: Self-optimizing Expert Advisor](https://www.mql5.com/en/articles/2279)

Algorithms for working with neural networks

CMLPBase class — multilayer perceptron

CMLPTrain class — multilayer perceptron training

CMLPE class — neural network sets

- [Optimize the strategy by the balance value and compare the results with the "Balance + max Sharpe Ratio criterion](https://www.mql5.com/en/articles/3642)

CLinReg, CLinearModel and CLRReport classes for working with linear regression

- [R-squared as an estimation of quality of the strategy balance curve](https://www.mql5.com/en/articles/2358)

CLinReg class for working with linear regression

Calculation of Pearson linear correlation and Spearman rank correlation coefficients

- [Visual evaluation of optimization results](https://www.mql5.com/en/articles/9922)

Linear regression model;

Calculation of Pearson linear correlation coefficient

- [An Example of Spread Strategy Development for Moscow Exchange Futures](https://www.mql5.com/en/articles/2739)

Linear regression model

- [Guide to writing a DLL for MQL5 in Delphi](https://www.mql5.com/en/articles/96)

Linear regression model

- [Studying the CCanvas Class. Anti-aliasing and shadows](https://www.mql5.com/en/articles/1612)
- [Graphics in DoEasy library (part 77): Shadow object class](https://www.mql5.com/en/articles/9575)
- [DoEasy. Controls (Part 27): Working on ProgressBar WinForms object](https://www.mql5.com/en/articles/11764)

Gaussian shadow shaping and blur

- [Portfolio trading in MetaTrader 4](https://www.mql5.com/en/articles/2646)

Principal Component Analysis method (LRBuildZ, LSFitLinearC and PCABuildBasis functions)

- [Controlled optimization: Simulated annealing](https://www.mql5.com/en/articles/4150)

Random number generator (CHighQualityRandStateShell class, HQRndRandomize, HQRndNormal, HQRndNormal2 and HQRndUniformR functions)

- [Practical Use of Kohonen Neural Networks in Algorithmic Trading. Part I. Tools](https://www.mql5.com/en/articles/5472)

K-means clustering algorithm

- [MQL5 Cookbook - Creating a ring buffer for fast calculation of indicators in a sliding window](https://www.mql5.com/en/articles/3047)

Calculation of basic statistical parameters

\- Mean

\- standard deviation (StdDev)

\- bell-shaped distribution asymmetry (Skewness)

\- Kurtosis

- [MQL5 Wizard techniques you should know (Part 06): Fourier Transform](https://www.mql5.com/en/articles/12599)

Fast Fourier transformation (FFTR1D)

- [Frequency domain representations of time series: The Power Spectrum](https://www.mql5.com/en/articles/12701)

Fast Fourier transformation (FFTR1D)

- [Random Decision Forest in reinforcement learning](https://www.mql5.com/en/articles/3856)

Random Decision Forest (RDF) algorithm

- [MQL5 Wizard techniques you should know (Part 5): Markov Chains](https://www.mql5.com/en/articles/11930)

markov chains (CMarkovCPD class)

- [Forecasting Time Series (Part 2): Least-Square Support-Vector Machine (LS-SVM)](https://www.mql5.com/en/articles/7603)

Linear "solvers" of the ALGLIB library

- [MQL5 Wizard techniques you should know (Part 04): Linear discriminant analysis](https://www.mql5.com/en/articles/11687) Linear discriminant analysis (CLDA class)

- [Forecasting market movements using the Bayesian classification and indicators based on Singular Spectrum Analysis](https://www.mql5.com/en/articles/3172)

ALGLIB matrix functions


Thus, the functionality of the ALGLIB mathematical library turned out to be a useful tool for analyzing financial data.

### 4\. Singular Spectrum Analysis in ALGLIB

In addition to the already existing methods, the new version of the ALGLIB library now includes the [Singular Spectrum Analysis](https://en.wikipedia.org/wiki/Singular_spectrum_analysis "https://ru.wikipedia.org/wiki/SSA_(%D0%BC%D0%B5%D1%82%D0%BE%D0%B4)") method (SSA, also known as "caterpillar"). This method significantly expands the capabilities of analyzing financial time series, especially in problems of their forecasting. The SSA algorithm has been available since version 3.15, and its implementation has been optimized. It provides functionality for trend extraction, time series prediction, averaging prediction, and has fast incremental model updates.

We invite you to familiarize yourself with how this method works in practice and share your experience of using it when developing trading strategies.

Below is a test script with examples of using the SSA method. Find additional information about examples and details of using the SSA method in the ALGLIB library in the ["Singular Spectrum Analysis"](https://www.mql5.com/go?link=https://www.alglib.net/time-series/singular-spectrum-analysis.php "https://www.alglib.net/time-series/singular-spectrum-analysis.php") section of the official library help.

```
//+------------------------------------------------------------------+
//|                                                     SSA_Test.mq5 |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <Math\Alglib\alglib.mqh>

// Examples of SSA usage with ALGLIB
// https://www.alglib.net/translator/man/manual.cpp.html#example_ssa_d_basic
// https://www.alglib.net/translator/man/manual.cpp.html#example_ssВa_d_realtime
// https://www.alglib.net/translator/man/manual.cpp.html#example_ssa_d_forecast

//+------------------------------------------------------------------+
//| PrintMatrix                                                      |
//+------------------------------------------------------------------+
void PrintMatrix(CMatrixDouble &x)
  {
//--- print matrix
   string str="[";\
   for(int i=0; i<x.Rows(); i++)\
     {\
      str+="[";\
      for(int j=0; j<x.Cols(); j++)\
        {\
         str+=StringFormat("%f",x.Get(i,j));\
         if(j<x.Cols()-1)\
            str+=",";\
        }\
      str+="]";\
      if(i<x.Rows()-1)\
         str+=",";\
     }\
   str+="]";
   printf("%s",str);
  }
//+------------------------------------------------------------------+
//| PrintVector                                                      |
//+------------------------------------------------------------------+
void PrintVector(CRowDouble &x)
  {
//--- print vector
   string str="[";\
   for(int i=0; i<x.Size(); i++)\
     {\
      str+=StringFormat("%f",x[i]);\
      if(i<x.Size()-1)\
         str+=",";\
     }\
   str+="]";
   printf("%s",str);
  }
//+------------------------------------------------------------------+
//| ssa_d_basic                                                      |
//+------------------------------------------------------------------+
//| Here we demonstrate SSA trend/noise separation for some toy      |
//| problem: small monotonically growing series X are analyzed with  |
//| 3-tick window and "top-K" version of SSA, which selects K largest|
//| singular vectors for analysis, with K=1.                         |
//+------------------------------------------------------------------+
int ssa_d_basic(void)
  {
//--- prepare input data
   double input_data[]= {0,0.5,1,1,1.5,2};
   CRowDouble x=input_data;
//--- first, we create SSA model, set its properties and add dataset.
//--- we use window with width=3 and configure model to use direct SSA algorithm which runs exact O(N*W^2) analysis - to extract one top singular vector.
//--- NOTE: SSA model may store and analyze more than one sequence (say, different sequences may correspond to data collected from different devices)
//--- ALGLIB wrapper class
   CAlglib alglib;
//--- SSA model
   CSSAModel ssa_model;
//--- create SSA model
   alglib.SSACreate(ssa_model);
//--- set window width for SSA model
   alglib.SSASetWindow(ssa_model,3);
//--- adds data sequence to SSA model
   alglib.SSAAddSequence(ssa_model,x);
//--- set SSA algorithm to "direct top-K" algorithm
   alglib.SSASetAlgoTopKDirect(ssa_model,1);
//--- now we begin analysis. Internally SSA model stores everything it needs:
//--- data, settings, solvers and so on. Right after first call to analysis-related function it will analyze dataset, build basis and perform analysis.
//--- subsequent calls to analysis functions will reuse previously computed basis, unless you invalidate it by changing model settings (or dataset).
//--- trend and noise
   CRowDouble trend,noise;
//--- build SSA basis using internally stored (entire) dataset and return reconstruction for the sequence being passed to this function
   alglib.SSAAnalyzeSequence(ssa_model,x,x.Size(),trend,noise);
//--- print result
   PrintVector(trend);
//--- output:   [0.381548,0.558290,0.781016,1.079470,1.504191,2.010505]
//--- EXPECTED: [0.3815,0.5582,0.7810,1.0794,1.5041,2.0105]
   return 0;
  }

//+------------------------------------------------------------------+
//| ssa_d_forecast                                                   |
//+------------------------------------------------------------------+
//| Here we demonstrate SSA forecasting on some toy problem with     |
//| clearly visible linear trend and small amount of noise.          |
//+------------------------------------------------------------------+
int ssa_d_forecast(void)
  {
//--- ALGLIB wrapper
   CAlglib alglib;
//--- model
   CSSAModel ssa_model;
//--- prepare input data
   double input_data[] = {0.05,0.96,2.04,3.11,3.97,5.03,5.98,7.02,8.02};
   CRowDouble x=input_data;
//--- first, we create SSA model, set its properties and add dataset.
//--- we use window with width=3 and configure model to use direct SSA algorithm - one which runs exact O(N*W^2) analysis-to extract two top singular vectors
//--- NOTE: SSA model may store and analyze more than one sequence (say, different sequences may correspond to data collected from different devices)
//--- create SSA model
   alglib.SSACreate(ssa_model);
//--- set window width for SSA model
   alglib.SSASetWindow(ssa_model,3);
//--- set window width for SSA model
   alglib.SSAAddSequence(ssa_model,x);
//--- set SSA algorithm to "direct top-K" algorithm
   alglib.SSASetAlgoTopKDirect(ssa_model,2);
//--- now we begin analysis. Internally SSA model stores everything it needs:
//--- data, settings, solvers and so on. Right after first call to analysis-related function it will analyze dataset, build basis and perform analysis.
//--- subsequent calls to analysis functions will reuse previously computed basis, unless you invalidate it by changing model settings (or dataset).
//--- in this example we show how to use ssaforecastlast() function, which predicts changed in the last sequence of the dataset.
//--- if you want to perform prediction for some other sequence, use alglib.SSAForecastSequence().
//--- trend
   CRowDouble trend;
   alglib.SSAForecastLast(ssa_model,3,trend);
//--- print result
   PrintVector(trend);
//--- output:   [9.000587,9.932294,10.805125]
//--- EXPECTED: [9.0005,9.9322,10.8051]
//--- well, we expected it to be [9,10,11]. There exists some difference, which can be explained by the artificial noise in the dataset.
   return 0;
  }
//+------------------------------------------------------------------+
//| ssa_d_realtime                                                   |
//+------------------------------------------------------------------+
//| Suppose that you have a constant stream of incoming data, and    |
//| you want to regularly perform singular spectral analysis         |
//| of this stream.                                                  |
//|                                                                  |
//| One full run of direct algorithm costs O(N*Width^2) operations,  |
//| so the more points you have, the more it costs to rebuild basis  |
//| from scratch.                                                    |
//|                                                                  |
//| Luckily we have incremental SSA algorithm which can perform      |
//| quick updates of already computed basis in O(K*Width^2) ops,     |
//| where K is a number of singular vectors extracted. Usually it    |
//| is orders of magnitude faster than full update of the basis.     |
//|                                                                  |
//| In this example we start from some initial dataset x0. Then we   |
//| start appending elements one by one to the end of the last       |
//| sequence                                                         |
//|                                                                  |
//| NOTE: direct algorithm also supports incremental updates, but    |
//|       with O(Width^3) cost. Typically K<<Width, so specialized   |
//|       incremental algorithm is still faster.                     |
//+------------------------------------------------------------------+
int ssa_d_realtime(void)
  {
//--- ALGLIB wrapper
   CAlglib alglib;
//--- model
   CSSAModel ssa_model1;
//---
   CMatrixDouble a1;
   CRowDouble sv1;
   int w,k;
//--- prepare input data
   double input_data[]= {0.009,0.976,1.999,2.984,3.977,5.002};
   CRowDouble x0=input_data;
//--- create SSA model
   alglib.SSACreate(ssa_model1);
//--- set window width for SSA model
   alglib.SSASetWindow(ssa_model1,3);
//--- adds data sequence to SSA model
   alglib.SSAAddSequence(ssa_model1,x0);
//--- set algorithm to the real-time version of top-K, K=2
   alglib.SSASetAlgoTopKRealtime(ssa_model1,2);
//--- one more interesting feature of the incremental algorithm is "power-up" cycle.
//--- even with incremental algorithm initial basis calculation costs O(N*Width^2) ops.
//--- if such startup cost is too high for your real-time app, then you may divide initial basis calculation
//--- across several model updates. It results in better latency at the price of somewhat lesser precision during first few updates.
   alglib.SSASetPowerUpLength(ssa_model1,3);
//--- now, after we prepared everything, start to add incoming points one by one;
//--- in the real life, of course, we will perform some work between subsequent update (analyze something, predict, and so on).
//--- after each append we perform one iteration of the real-time solver. Usually
//--- one iteration is more than enough to update basis. If you have REALLY tight performance constraints,
//--- you may specify fractional amount of iterations, which means that iteration is performed with required probability.
   double updateits = 1.0;
//--- append single point to last data sequence stored in the SSA model and update model in the incremental manner
   alglib.SSAAppendPointAndUpdate(ssa_model1,5.951,updateits);
//--- execute SSA on internally stored dataset and get basis found by current method
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,7.074,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,7.925,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,8.992,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,9.942,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,11.051,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,11.965,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,13.047,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- append single point to last data sequence
   alglib.SSAAppendPointAndUpdate(ssa_model1,13.970,updateits);
//--- execute SSA
   alglib.SSAGetBasis(ssa_model1,a1,sv1,w,k);
//--- ok, we have our basis in a1[] and singular values at sv1[]. But is it good enough? Let's print it.
   PrintMatrix(a1);
//--- output:   [[0.510607,0.753611],[0.575201,0.058445],[0.639081,-0.654717]]
//--- EXPECTED: [[0.510607,0.753611],[0.575201,0.058445],[0.639081,-0.654717]]
//--- ok, two vectors with 3 components each. but how to understand that is it really good basis? let's compare it with direct SSA algorithm on the entire sequence.
   CSSAModel ssa_model2;
   CMatrixDouble a2;
   CRowDouble sv2;
//--- input data
   double input_data2[]= {0.009,0.976,1.999,2.984,3.977,5.002,5.951,7.074,7.925,8.992,9.942,11.051,11.965,13.047,13.970};
   CRowDouble x2=input_data2;
//--- create SSA model
   alglib.SSACreate(ssa_model2);
//--- set window width for SSA model
   alglib.SSASetWindow(ssa_model2,3);
//--- add data sequence to SSA model
   alglib.SSAAddSequence(ssa_model2,x2);
//--- set SSA algorithm to "direct top-K" algorithm
   alglib.SSASetAlgoTopKDirect(ssa_model2,2);
//--- execute SSA on internally stored dataset and get basis found by current method
   alglib.SSAGetBasis(ssa_model2,a2,sv2,w,k);
//--- it is exactly the same as one calculated with incremental approach!
   PrintMatrix(a2);
//--- output:   [[0.510607,0.753611],[0.575201,0.058445],[0.639081,-0.654717]]
//--- EXPECTED: [[0.510607,0.753611],[0.575201,0.058445],[0.639081,-0.654717]]
   return 0;
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//--- the very simple example of trend/noise separation
   ssa_d_basic();
//--- forecasting with SSA
   ssa_d_forecast();
//--- real-time analysis of constantly arriving data
   ssa_d_realtime();
  }
//+------------------------------------------------------------------+
```

### Conclusion

ALGLIB is a powerful tool for analyzing data in financial markets. Multilinguality, cross-platform nature, rich functionality and open source make it an attractive choice for researchers and developers in the field of financial analysis and modeling. There is an ongoing need for reliable data analytics tools, and ALGLIB successfully meets this challenge by supporting continuous development and improvement.

For their part, the MetaTrader 5 platform developers provide traders with the best solutions:

- MQL5 language, which is as good as C++ in terms of speed;
- Built-in processing of [SQLite](https://www.mql5.com/en/docs/database) databases, the ability to carry out calculations using [OpenCL](https://www.mql5.com/en/docs/opencl), [DirectX](https://www.mql5.com/en/docs/directx) support, using [ONNX](https://www.mql5.com/en/docs/onnx) models and integration with [Python](https://www.mql5.com/en/docs/python_metatrader5);
- Mathematical libraries, including [fuzzy logic](https://www.mql5.com/en/docs/standardlibrary/mathematics/fuzzy_logic), [statistics](https://www.mql5.com/en/docs/standardlibrary/mathematics/stat) and updated [ALGLIB](https://www.mql5.com/en/code/1146) version.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/13289](https://www.mql5.com/ru/articles/13289)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/13289.zip "Download all attachments in the single ZIP archive")

[SSA\_Test.mq5](https://www.mql5.com/en/articles/download/13289/ssa_test.mq5 "Download SSA_Test.mq5")(26.56 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/461192)**
(10)


![MetaQuotes](https://c.mql5.com/avatar/2009/11/4AF883AB-83DE.jpg)

**[Renat Fatkhullin](https://www.mql5.com/en/users/renat)**
\|
7 Oct 2023 at 11:36

**Maxim Kuznetsov [#](https://www.mql5.com/ru/forum/455275#comment_49788600):**

the last two lines about test-case of the original AlgLIB. There are no tests in the MQL5 adaptation.

All extensive Alglib test-cases have always been from the very first ported version of the MQL5 library [(October 2012](https://www.mql5.com/ru/forum/8265)):

```
\MQL5\Scripts\UnitTests\Alglib\
                               TestClasses.mq5
                               TestInterfaces.mq5
                               TestClasses.mqh
                               TestInterfaces.mqh
```

Now it is 3,850 kb of tests in source code and 105,000 lines of code covering almost all functionality.

Anyone can compile the unit tests TestClasses.mq5 / TestInterfaces.mq5 and run them in the terminal.

![MetaQuotes](https://c.mql5.com/avatar/2009/11/4AF883AB-83DE.jpg)

**[Renat Fatkhullin](https://www.mql5.com/en/users/renat)**
\|
7 Oct 2023 at 11:41

In addition to Alglib, there are testcases for other maths libraries:

![](https://c.mql5.com/3/419/5218511540072.png)

![Alexey Topounov](https://c.mql5.com/avatar/2011/9/4E62AB20-013F.jpg)

**[Alexey Topounov](https://www.mql5.com/en/users/alextp)**
\|
16 Oct 2023 at 13:25

Colleagues, where (in which file) can I see the version number of the library?


![Alexey Topounov](https://c.mql5.com/avatar/2011/9/4E62AB20-013F.jpg)

**[Alexey Topounov](https://www.mql5.com/en/users/alextp)**
\|
17 Oct 2023 at 10:27

After the update the [neural network](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") stopped working.

I rolled back to the old version of ALGLIB. If you need it - attached.

![Evgeniy Chernish](https://c.mql5.com/avatar/2024/3/65eac9b5-9233.png)

**[Evgeniy Chernish](https://www.mql5.com/en/users/vp999369)**
\|
19 Mar 2024 at 14:56

Afternoon!

Has anyone been able to figure out how to use non-linear ISC optimisation ?

Here is an example from Alglib site [https://www.alglib.net/translator/man/manual.cpp.html#example\_lsfit\_d\_nlf](https://www.mql5.com/go?link=https://www.alglib.net/translator/man/manual.cpp.html%23example_lsfit_d_nlf "https://www.alglib.net/translator/man/manual.cpp.html#example_lsfit_d_nlf")

Could you please tell me what I'm doing wrong?

```
//+------------------------------------------------------------------+
//|Optim.mq5 |
//|vp |
//| https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "vp"
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <Math\Alglib\alglib.mqh>

 void function_cx_1_func(double &c[],double &x[],double &func,CObject &obj)
{
    // this callback calculates f(c,x)=exp(-c0*sqr(x0))
    // where x is a position on X-axis and c is adjustable parameter
    func = MathExp(-c[0]*MathPow(x[0],2));
}

void OnStart()
  {
int info;
CObject  obj;
vector v = {-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0};
double y[] = {0.223130, 0.382893, 0.582748, 0.786628, 0.941765, 1.000000, 0.941765, 0.786628, 0.582748, 0.382893, 0.223130};
double c[] = {0.3};
CMatrixDouble x;
x.Col(0,v);
double epsx = 0.000001;
int maxits = 0;
double diffstep = 0.0001;

//
// Fitting without weights
//
CLSFitStateShell state;
CAlglib::LSFitCreateF(x,y,c,diffstep,state);
CAlglib::LSFitSetCond(state,epsx,maxits);
CNDimensional_Rep rep;
CNDimensional_PFunc function_cx_1_func;
CAlglib::LSFitFit(state,function_cx_1_func,rep,0,obj);

CLSFitReportShell grep;
CAlglib::LSFitResults(state,info,c,grep);

ArrayPrint(c); // EXPECTED: [1.5]
Print(grep.GetIterationsCount());
Print(grep.GetRMSError());

  }
```

![Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5](https://c.mql5.com/2/65/Introduction_to_MQL5_rPart_38_Mastering_the_Core_Elements_of_MQL5____LOGO___small-transformed.png)[Introduction to MQL5 (Part 3): Mastering the Core Elements of MQL5](https://www.mql5.com/en/articles/14099)

Explore the fundamentals of MQL5 programming in this beginner-friendly article, where we demystify arrays, custom functions, preprocessors, and event handling, all explained with clarity making every line of code accessible. Join us in unlocking the power of MQL5 with a unique approach that ensures understanding at every step. This article sets the foundation for mastering MQL5, emphasizing the explanation of each line of code, and providing a distinct and enriching learning experience.

![How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://c.mql5.com/2/64/rj-article-image-60x60.png)[How to create a simple Multi-Currency Expert Advisor using MQL5 (Part 6): Two RSI indicators cross each other's lines](https://www.mql5.com/en/articles/14051)

The multi-currency expert advisor in this article is an expert advisor or trading robot that uses two RSI indicators with crossing lines, the Fast RSI which crosses with the Slow RSI.

![Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost](https://c.mql5.com/2/65/Data_Science_and_Machine_Learning_4Part_19y_Supercharge_Your_AI_models_with_AdaBoost___LOGO.png)[Data Science and Machine Learning (Part 19): Supercharge Your AI models with AdaBoost](https://www.mql5.com/en/articles/14034)

AdaBoost, a powerful boosting algorithm designed to elevate the performance of your AI models. AdaBoost, short for Adaptive Boosting, is a sophisticated ensemble learning technique that seamlessly integrates weak learners, enhancing their collective predictive strength.

![Building and testing Aroon Trading Systems](https://c.mql5.com/2/64/Building_and_testing_Aroon_Trading_Systems___LOGO.png)[Building and testing Aroon Trading Systems](https://www.mql5.com/en/articles/14006)

In this article, we will learn how we can build an Aroon trading system after learning the basics of the indicators and the needed steps to build a trading system based on the Aroon indicator. After building this trading system, we will test it to see if it can be profitable or needs more optimization.

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/13289&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068998850665971658)

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