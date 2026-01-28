---
title: Population optimization algorithms: Resistance to getting stuck in local extrema (Part I)
url: https://www.mql5.com/en/articles/14352
categories: Integration
relevance_score: 7
scraped_at: 2026-01-22T17:51:24.662893
---

[![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/01.png)![](https://www.mql5.com/ff/sh/vzatb6m64gt8yfc4z2/02.png)Powerful analytics for traders of any levelAll the necessary trading reports for beginners and professionals](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=muccpajyfystoakuukdobwigjejzmpqn&s=52daad60fa795e635264e6f94898f05493bca3b5124d4cca8eb7e82333c2ef12&uid=&ref=https://www.mql5.com/en/articles/14352&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049430807866616683)

MetaTrader 5 / Examples


### Contents

1\. [Introduction](https://www.mql5.com/en/articles/14352#tag1)

2\. [Setting a task](https://www.mql5.com/en/articles/14352#tag2)

3\. [Code changes](https://www.mql5.com/en/articles/14352#tag3)

4\. [Algorithms](https://www.mql5.com/en/articles/14352#tag4)

5\. [Preliminary conclusions](https://www.mql5.com/en/articles/14352#tag5)

### 1\. Introduction

This is a unique research, the idea for which came to me while answering questions that arose during the discussion of one of my articles. I am hopeful that readers will appreciate the value and originality of this work.

My thoughts and ideas leading to this research are the result of deep immersion in the topic and passion for scientific research. I believe that this work may become an important contribution to the field of algorithmic optimization attracting the attention of researchers and practitioners.

In this experiment, I propose to conduct a test aimed at assessing the resistance of algorithms to getting stuck in local extrema and, instead of randomly placing agents at the first iteration throughout the entire field of the search space, place them in the global minimum. The objective of the experiment is to search for a global maximum.

In such a scenario, where all the search agents of the algorithm are located at one point, we are faced with an interesting phenomenon - a degenerate population. This is like a moment of freezing time, where diversity in the population is reduced to a minimum. Although this scenario is artificial, it allows us to obtain interesting conclusions and evaluate the impact of reducing diversity in the population on the outcome. The algorithm should be able to get out of such a bottleneck and achieve a global maximum.

In this kind of stress test for optimization algorithms, we can reveal the secrets of agent interaction, their cooperation or competition, and understand how these factors affect the speed of achieving the optimum. Such analysis opens new horizons in understanding the importance of diversity in a population for the efficient operation of algorithms and allows us to develop strategies for maintaining this diversity to achieve better results.

To carry out the experiment, we need to first initialize the coordinates of the agents forcibly outside the algorithm, using the coordinates of the global minimum, before measuring the fitness function at the first epoch.

Such an experiment will allow us to evaluate resistance to extremely difficult conditions and the ability to overcome limitations.

### 2\. Setting a task

Imagine that you are in a deep valley, surrounded by dense darkness. Your eyes are blindfolded, preventing you from seeing the world around you. You feel a cold wind blow across your face and your heart beats hard, as if trying to burst out of your chest, emphasizing your anxiety and uncertainty.

You start moving by taking the first step forward. Your feet feel the ground, and you gradually move along the valley, with each step closer to freedom, and your mind is filled with determination. Finally, when you reach the edge, a breathtaking view opens up in front of you. The glass-flat plain stretches out before you like an endless canvas. The isolated steep cliffs that rise in the distance evoke mixed feelings of amazement and challenge. However, despite this beautiful panorama, you still cannot see it.

You decide to rely on your senses and intuition to determine the direction, in which to move and reach the top of the highest cliff. Now imagine that this difficult task is faced by an algorithm based on the Megacity test discrete function. This algorithm, like you, is limited by its capabilities and cannot see or sense the environment directly. It must use its computing skills and logic to decide on the best path to the top of the tallest cliff. Thus, this problem on the Megacity test discrete function is a difficult challenge for an algorithm that, like you, faces uncertainty.

However, the Hilly and Forest functions also pose a difficult problem in such circumstances. Let's take a look at these familiar functions to appreciate the complexity of the upcoming task for the algorithms.

![Hilly road](https://c.mql5.com/2/71/Hilly_road.png)

**Hilly function**

![Forest road](https://c.mql5.com/2/71/Forest_road.png)

**Forest function**

![Megacity road](https://c.mql5.com/2/71/Megacity_road.png)

**Megacity function**

Imagine that you are an algorithm, the embodiment of computing power and logical thinking. You are not just dry strings of code or equations on paper, you come to life and take on the role of a researcher. Your mind is filled with questions and hypotheses, and you strive to solve the mystery that stands before you. You are immersed in a world of data and information, analyzing, comparing and highlighting key factors, while your computing abilities are firing on all cylinders. You start building different models and scenarios, exploring possible paths to solving a problem and trying different combinations and options, and your creativity leads to new discoveries. But the path to a solution is not always easy. You face challenges that may seem insurmountable, but you do not give up. You overcome obstacles and distractions, one by one, getting closer to your goal with each step. And finally, when you find a solution, it is like a moment of enlightenment. Your mind is illuminated with the bright light of understanding and you see all the pieces falling into place.

In the context of this picture, the optimization algorithm is faced with the task of maximizing a function that has a complex topology with many local maxima. Here is a more detailed description of the problem:

- **Local maxima**. On the way from the starting point (labeled "min") to the ending point (labeled "max"), the algorithm encounters many local maxima. This can lead to the algorithm getting stuck in a local trap without reaching the global extremum. This is a major problem in optimization problems. For example, gradient descent based algorithms tend to "climb" to the nearest maximum and if it is a local maximum then they "get stuck".

- **High dimension**. If we consider a function in a multidimensional space, then the number of local maxima increases sharply, which complicates the problem. In high dimensions, the space becomes "empty", which makes the problem even more difficult due to the fact that the algorithm has nothing to cling on.


- **Function surface complexity**. The function surface can be very complex with many peaks and valleys, making the optimization process more time-consuming. This requires the algorithm to be able to jump over local maxima and climb to the global maximum.

- **Convergence and convergence speed.** The convergence of the algorithm to the function maximum may be slowed down due to the distance of the finishing point from the starting point, which may require additional iterations to achieve the goal.

- **Area research.** The algorithm may need to additionally explore the region in search of a global maximum, which can lead to increased computational complexity.

To summarize the description of the problem, we can highlight the difference between a typical optimization situation, where agents begin their movement in the search space uniformly distributed, and a situation where it is necessary not just to "reveal" existing information, but rather to explore and expand their knowledge and experience into uncharted areas. Setting the problem in this way becomes especially valuable in cases where we already have some solutions obtained in previous optimization sessions, and we want to start from this solution "point", rather than simply crystallize existing information.

### 3\. Code changes

In order to place agents at one point, assign the coordinates of the test function global minimum to the appropriate population agents. In general, this procedure looks like this:

```
if (epochCNT == 1)
{
  for (int set = 0; set < ArraySize (AO.a); set++)
  {
    for (int i = 0; i < funcCount; i++)
    {
      AO.a [set].c [i * 2]     = f.GetMinFuncX ();
      AO.a [set].c [i * 2 + 1] = f.GetMinFuncY ();
    }
  }
}
```

For some algorithms, it is not enough to simply place agents at one point in the search space. Additional information should be entered. For example, in the case of the SDSm algorithm, it is also necessary to specify the corresponding sector for each coordinate. On the other hand, when using the BGA algorithm, it is necessary to convert coordinate values from a real representation to a binary one, which entails additional changes in the algorithm code. For BGA, the location of agents at one point will look like this:

```
//==================
if (epochCNT == 1)
{
  for (int set = 0; set < ArraySize (AO.a); set++)
  {
    for (int i = 0; i < funcCount; i++)
    {
      AO.a [set].DoubleToGene (f.GetMinFuncX (), i * 2);
      AO.a [set].DoubleToGene (f.GetMinFuncY (), i * 2 + 1);
    }

    AO.a [set].ExtractGenes ();
  }
}
//==================
```

As can be seen from this code, the necessary transformation of coordinates into the binary code of genes occurs here. I am currently working on unifying the process of placing agents in the desired coordinates. This study presents the source codes in their current state. Almost every algorithm required a custom testbed due to the nature of their architecture. Stay tuned for new articles to receive updates on the algorithms. Simplification of placing coordinates in the population, as well as their unification, will be brought to a common standard.

### 4\. Algorithms

Moving on to analyzing the behavior of the algorithms in our unique testing, we begin to discuss the algorithms that demonstrated the worst results. What is especially surprising is that some of them, which had previously occupied quite high positions in the rankings in standard tests, performed poorly in our unusual test. This suggests that the success of algorithms may depend not only on their overall efficiency, but also on their ability to adapt to specific conditions and characteristics of the problem. Such unexpected results highlight the importance of conducting a variety of tests and studies to gain a deeper understanding of the performance of optimization algorithms in different contexts.

Below are reports of the algorithms operation, which should be read as follows:

- C\_AO\_FSS:50;0.01;0.8             - algorithm name and external parameters

- 5 Hilly's                                   - name of the test function and its number in the test

-  Func runs: 10000                   - number of runs

- result: 0.32457068874346456  - obtained result, where 0.0 is the minimum of the test function, and 1.0 is the maximum. The higher the value, the better

- All score: 1.33084                   - total value of the points scored. The higher the value, the better


**Differential evolution ( [DE](https://www.mql5.com/en/articles/13781))**

C\_AO\_DE:50;0.2;0.8

=============================

5 Hilly's; Func runs: 10000; result: 0.0

25 Hilly's; Func runs: 10000; result: 0.0

500 Hilly's; Func runs: 10000; result: 0.0

=============================

5 Forest's; Func runs: 10000; result: 0.0

25 Forest's; Func runs: 10000; result: 0.0

500 Forest's; Func runs: 10000; result: 0.0

=============================

5 Megacity's; Func runs: 10000; result: 0.0

25 Megacity's; Func runs: 10000; result: 0.0

500 Megacity's; Func runs: 10000; result: 0.0

=============================

All score: 0.00000

Unfortunately, even one of the most powerful algorithms in the ranking table completely failed our test. In this case, the agents found themselves stagnant, unable to move. The reason for this failure is that each new position of an agent depends on the positions of three other agents, and if they all end up at the same point, then none of them will be able to update their coordinates. This situation highlights the importance of carefully analyzing the interactions between agents and adequately controlling their movements to successfully complete the optimization problem. Such unexpected failures may motivate further research and improvements in the development of algorithms that can effectively deal with such complexities.

I will not provide a printout of the test bench for other algorithms that completely failed the test.

**Electromagnetic algorithm ( [EM](https://www.mql5.com/en/articles/12352))**

The EM algorithm encountered the problem of the inability to update the coordinates of agents during optimization. In this case, the particles collapsed under the influence of electromagnetic attraction, which made the agents combine into a lump.

**Gravity search algorithm ( [GSA](https://www.mql5.com/en/articles/12072))**

The forces of gravity led to the fact that all objects were at one point and remained there - they were attracted to the center, similar to a black hole.

**Artificial ant colony algorithm ( [ACOm](https://www.mql5.com/en/articles/11602))**

The problem with this algorithm was the lack of paths for the movement of ants that would move based on the smell of pheromones. In this case, since the ants started from one point, the paths between them were not formed, which led to difficulties in the movement and coordination of the ants' actions.

**Fish School Search ( [FSS](https://www.mql5.com/en/articles/11841))**

C\_AO\_FSS:50;0.01;0.8

=============================

5 Hilly's; Func runs: 10000; result: 0.32457068874346456

25 Hilly's; Func runs: 10000; result: 0.27938488291267094

500 Hilly's; Func runs: 10000; result: 0.2343201202260512

=============================

5 Forest's; Func runs: 10000; result: 0.18964347858030822

25 Forest's; Func runs: 10000; result: 0.16146315945349987

500 Forest's; Func runs: 10000; result: 0.14145987387955847

=============================

5 Megacity's; Func runs: 10000; result: 0.0

25 Megacity's; Func runs: 10000; result: 0.0

500 Megacity's; Func runs: 10000; result: 0.0

=============================

All score: 1.33084

In this algorithm, fish use the difference in fitness between the last and previous iterations to determine their direction of movement. While on the Hilly and Forest functions the fish sense changes in the landscape, running the algorithm on the flat surface of Megacity confuses the fish, depriving them of orientation. Amazing fish behavior occurs when starting on smooth surfaces that have a gradient. However, if the starting point is at the top of the local extremum, and not in the hole, the fish are unlikely to move even on the Hilly and Forest functions.

**Simulated Isotropic Annealing ( [SIA](https://www.mql5.com/en/articles/13870))**

C\_AO\_SIA:100:0.01:0.1

=============================

5 Hilly's; Func runs: 10000; result: 0.32958446477979136

25 Hilly's; Func runs: 10000; result: 0.32556359155723036

500 Hilly's; Func runs: 10000; result: 0.27262289744765306

=============================

5 Forest's; Func runs: 10000; result: 0.1940720887058382

25 Forest's; Func runs: 10000; result: 0.1935893813273654

500 Forest's; Func runs: 10000; result: 0.16409411642496857

=============================

5 Megacity's; Func runs: 10000; result: 0.0

25 Megacity's; Func runs: 10000; result: 0.0

500 Megacity's; Func runs: 10000; result: 0.0

=============================

All score: 1.47953

The simulated isotropic annealing algorithm exhibits a unique combination of features that are reminiscent of FSS operation, but with significant differences. In this algorithm, movement in different directions from the starting point occurs more energetically and actively than in FSS, creating a feeling of vigorous creativity in finding the optimal solution. Similar to FSS, the isotropic annealing simulation algorithm uses differences in fitness function values to guide motion. However, here the movement is influenced by a gradual decrease in temperature, which over time leads to the "freezing" of particles at certain points in space.

**Evolutionary strategies ( [(PO)ES](https://www.mql5.com/en/articles/13923))**

C\_AO\_(PO)ES:100:10:0.025:8.0

=============================

5 Hilly's; Func runs: 10000; result: 0.32231823718105856

25 Hilly's; Func runs: 10000; result: 0.3228736374003839

500 Hilly's; Func runs: 10000; result: 0.2797261292300971

=============================

5 Forest's; Func runs: 10000; result: 0.19410491957153192

25 Forest's; Func runs: 10000; result: 0.1875135077472832

500 Forest's; Func runs: 10000; result: 0.15801830580073034

=============================

5 Megacity's; Func runs: 10000; result: 0.1292307692307692

25 Megacity's; Func runs: 10000; result: 0.12553846153846154

500 Megacity's; Func runs: 10000; result: 0.08198461538461577

=============================

All score: 1.80131

This algorithm was the first on the list that was able to successfully complete all tests. Although the use of the term "successfully completed" seems pretty judgmental, in fact it passed each of them with at least some result other than zero. Remarkably, there is a tendency of the population to divide into separate groups. However, the initial enthusiasm of the algorithm lasts only for a short time - the agents quickly stop searching at the first nearest hill, probably assuming that they have already achieved success.

It is especially interesting that at the beginning the agents, having divided into groups, inspire optimism in the observer by moving in different directions, but disappointment comes quickly: as soon as one of the groups achieves a noticeable improvement in their position, all other groups immediately change direction, rushing towards the leader. These moments evoke mixed feelings and emotions - from delight to disappointment. The interaction of agents in this algorithm resembles life with all its surprises and changeability.

**Monkey Algorithm ( [MA](https://www.mql5.com/en/articles/12212))**

C\_AO\_MA:50;0.01;0.9;50

=============================

5 Hilly's; Func runs: 10000; result: 0.32874856274894027

25 Hilly's; Func runs: 10000; result: 0.30383823957660194

500 Hilly's; Func runs: 10000; result: 0.2475564907358033

=============================

5 Forest's; Func runs: 10000; result: 0.20619304546795353

25 Forest's; Func runs: 10000; result: 0.1733511102614089

500 Forest's; Func runs: 10000; result: 0.14786586882293234

=============================

5 Megacity's; Func runs: 10000; result: 0.17538461538461542

25 Megacity's; Func runs: 10000; result: 0.1436923076923077

500 Megacity's; Func runs: 10000; result: 0.09555384615384681

=============================

All score: 1.82218

In the context of this algorithm, the monkeys continue to move in a chosen direction in a fairly isolated manner, even if that direction turns out to be the wrong one. This unique behavior allows agents to explore space more efficiently, spreading further from their starting point. They make long "leaps into the unknown", especially impressive on the discrete Megacity function, where there is no fitness increment on horizontal surfaces, which facilitates reaching distant areas.

However, despite this exploration ability, it proves insufficient to achieve the main goal as the number of available iterations comes to an end. It is important to note the fascinating visual behavior of the algorithm, which truly resembles the chaotic movement of monkeys in a flock, creating an amazing spectacle and arousing interest in its further research and improvement.

**Simulated Annealing ( [SA](https://www.mql5.com/en/articles/13851))**

C\_AO\_SA:50:1000.0:0.1:0.2

=============================

5 Hilly's; Func runs: 10000; result: 0.3266993983850477

25 Hilly's; Func runs: 10000; result: 0.30166692301946135

500 Hilly's; Func runs: 10000; result: 0.2545648344562219

=============================

5 Forest's; Func runs: 10000; result: 0.1939959116807614

25 Forest's; Func runs: 10000; result: 0.17721159702946082

500 Forest's; Func runs: 10000; result: 0.15159936395874307

=============================

5 Megacity's; Func runs: 10000; result: 0.2584615384615384

25 Megacity's; Func runs: 10000; result: 0.15292307692307697

500 Megacity's; Func runs: 10000; result: 0.10135384615384675

=============================

All score: 1.91848

In the simulated annealing algorithm, unlike its "relative" SIA (which, by the way, occupies a much higher position in the ranking), the behavior is more chaotic, which is even noticeable to the naked eye in visualizations. This chaotic nature of the "annealing" simulation, however, helps achieving slightly better results. However, these achievements are not great enough to immortalize this algorithm in the hall of fame of outstanding algorithms, but the improvement is noticeable and deserves recognition.

**Firefly algorithm ( [FAm](https://www.mql5.com/en/articles/11873))**

C\_AO\_FAm:50;0.1;0.3;0.1

=============================

5 Hilly's; Func runs: 10000; result: 0.32461162859403175

25 Hilly's; Func runs: 10000; result: 0.31981492599317524

500 Hilly's; Func runs: 10000; result: 0.25932958993768923

=============================

5 Forest's; Func runs: 10000; result: 0.2124297717365277

25 Forest's; Func runs: 10000; result: 0.21595138588924906

500 Forest's; Func runs: 10000; result: 0.1577543024576405

=============================

5 Megacity's; Func runs: 10000; result: 0.2246153846153846

25 Megacity's; Func runs: 10000; result: 0.1987692307692308

500 Megacity's; Func runs: 10000; result: 0.12084615384615457

=============================

All score: 2.03412

FA is one of my favorite algorithms. Its appeal is evident not only in the beautiful name itself, but also in the elegant idea behind it, as well as in the elegant behavior of the fireflies. These mystical luminous creatures are capable of instantly approaching the nearest local extremes at such a speed that it is truly difficult to keep track of them. However, this great show is followed by stagnation when agents find themselves stuck in local maxima, unable to explore further and reach a global optimum.

Although it may seem frustrating, this moment of stagnation opens up an opportunity for improving the algorithm. By introducing mechanisms to overcome local pitfalls, FA can achieve new horizons of efficiency and accuracy. Thus, even in the moments when the fireflies stop, we see not just the end, but a new beginning - the opportunity to improve and develop this amazing algorithm.

**Bacterial Foraging Optimization ( [BFO](https://www.mql5.com/en/articles/12031))**

C\_AO\_BFO:50;0.01;0.3;100

=============================

5 Hilly's; Func runs: 10000; result: 0.3226339934200066

25 Hilly's; Func runs: 10000; result: 0.2925193012197403

500 Hilly's; Func runs: 10000; result: 0.2554221763445149

=============================

5 Forest's; Func runs: 10000; result: 0.2111053636851011

25 Forest's; Func runs: 10000; result: 0.20536292110181784

500 Forest's; Func runs: 10000; result: 0.15743855819242952

=============================

5 Megacity's; Func runs: 10000; result: 0.27999999999999997

25 Megacity's; Func runs: 10000; result: 0.19415384615384618

500 Megacity's; Func runs: 10000; result: 0.11735384615384695

=============================

All score: 2.03599

The ability of bacteria to maintain their mobility even without the need to increase their fitness is a key factor that allows them to spread efficiently over long distances, outperforming the algorithms discussed above in this aspect. This amazing phenomenon is especially evident in the Megacity environment, where bacteria demonstrate amazing mobility and survival capabilities, allowing them to successfully adapt to diverse and complex environments. In this context, bacteria become real pioneers, exploring and colonizing new territories, which emphasizes their unique capabilities and importance in the world of living organisms.

**Charge System Search ( [CSS](https://www.mql5.com/en/articles/13662))**

C\_AO\_CSS:50;0.1;0.7;0.01

=============================

5 Hilly's; Func runs: 10000; result: 0.38395827586082376

25 Hilly's; Func runs: 10000; result: 0.3048219687002418

500 Hilly's; Func runs: 10000; result: 0.2895158695448419

=============================

5 Forest's; Func runs: 10000; result: 0.2699906934238054

25 Forest's; Func runs: 10000; result: 0.19451237087137088

500 Forest's; Func runs: 10000; result: 0.18498127715987073

=============================

5 Megacity's; Func runs: 10000; result: 0.16923076923076924

25 Megacity's; Func runs: 10000; result: 0.13846153846153847

500 Megacity's; Func runs: 10000; result: 0.12276923076923094

=============================

All score: 2.05824

Surprisingly, this algorithm showed itself completely unexpectedly in this test, surpassing its usual indicators, where it ranks second to last among outsiders in the rating. This time, CSS finished somewhere in the middle (twelfth from bottom). The mystery of this transformation has its own explanation: electrostatic charges, obeying the equations of the algorithm, begin to interact with repulsive forces when they fall within the radius of the charge, which allows them to spread explosively in the surrounding search space. This process is not only visually appealing, but also has potential for practical applications.

The ability to fire like a firecracker opens up new possibilities for CSS. For example, we can consider this algorithm as a source of solution "ideas" for determining the optimal position of agents for other optimization algorithms, or we can successfully integrate it into hybrid solutions where CSS will help avoid population degeneration. Thus, the unexpected success of CSS in this testing is not only inspiring, but also opens up new perspectives for its application.

**Saplings Sowing and Growing up algorithm ( [SSG](https://www.mql5.com/en/articles/12268))**

C\_AO\_SSG:50;0.3;0.5;0.4;0.1

=============================

5 Hilly's; Func runs: 10000; result: 0.3284133103606342

25 Hilly's; Func runs: 10000; result: 0.3246280774155864

500 Hilly's; Func runs: 10000; result: 0.2808547975998361

=============================

5 Forest's; Func runs: 10000; result: 0.194115963123826

25 Forest's; Func runs: 10000; result: 0.19754974771110584

500 Forest's; Func runs: 10000; result: 0.17111478002239264

=============================

5 Megacity's; Func runs: 10000; result: 0.25846153846153846

25 Megacity's; Func runs: 10000; result: 0.23353846153846156

500 Megacity's; Func runs: 10000; result: 0.14158461538461614

=============================

All score: 2.13026

This algorithm reveals its potential on gradient functions, such as Hilly or Forest, and ranks high in the standard ranking. However, its efficiency is fully manifested only in the presence of a positive change in the gradient. Otherwise, the population quickly degrades, and individuals converge at one best local point, which opens up the possibility of using the SSG method to refine the results of optimization algorithms.

### 5\. Preliminary conclusions

In this unique research experiment where the algorithms were subjected to stringent initial conditions, we discovered many fascinating features of the various algorithms that remained hidden under the standard random uniform placement of agents in the search space. Just as it happens in real life, organisms reveal their inner potential in extreme conditions.

We also witnessed unexpected test results for some algorithms, including falling from the top to the bottom. This allows us to better understand how to use algorithms based on their capabilities in specialized optimization problems, as well as gain a deeper understanding of their strengths and weaknesses. It also more clearly revealed both the positive and negative sides of each algorithm, which allows them to more effectively use their strengths and compensate for their weaknesses. In addition, this research contributes to a better understanding of the creation of hybrid algorithms, allowing the strengths of different methods to be combined to achieve optimal results.

In the next article, we will continue to consider the properties and behavior of algorithms, and draw conclusions.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14352](https://www.mql5.com/ru/articles/14352)

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
**[Go to discussion](https://www.mql5.com/en/forum/469270)**
(2)


![Andrey Khatimlianskii](https://c.mql5.com/avatar/2022/10/63528ae6-0b8c.jpg)

**[Andrey Khatimlianskii](https://www.mql5.com/en/users/komposter)**
\|
10 Mar 2024 at 11:18

Interesting research!

What came to mind for some reason was dividing the search space into 4/9/16/... parts and running the algorithm on each subspace (but with fewer iterations) and then selecting the best result.

![Andrey Dik](https://c.mql5.com/avatar/2024/8/66be0662-3c24.png)

**[Andrey Dik](https://www.mql5.com/en/users/joo)**
\|
10 Mar 2024 at 12:50

**Andrey Khatimlianskii [#](https://www.mql5.com/ru/forum/463518#comment_52674147):**

That's an interesting study!

What came to mind for some reason was dividing the search space into 4/9/16/... parts and running the algorithm on each subspace (but with fewer iterations) and then selecting the best result.

It's great to see that the study has generated interest from readers.

Yes, dividing the space into zones, exploring the zones separately and then analysing the results makes practical sense.

![Developing a Replay System (Part 39): Paving the Path (III)](https://c.mql5.com/2/64/Desenvolvendo_um_sistema_de_Replay_dParte_39w_Pavimentando_o_Terreno_nIIIu_LOGO.png)[Developing a Replay System (Part 39): Paving the Path (III)](https://www.mql5.com/en/articles/11599)

Before we proceed to the second stage of development, we need to revise some ideas. Do you know how to make MQL5 do what you need? Have you ever tried to go beyond what is contained in the documentation? If not, then get ready. Because we will be doing something that most people don't normally do.

![Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://c.mql5.com/2/71/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 4): Pending virtual orders and saving status](https://www.mql5.com/en/articles/14246)

Having started developing a multi-currency EA, we have already achieved some results and managed to carry out several code improvement iterations. However, our EA was unable to work with pending orders and resume operation after the terminal restart. Let's add these features.

![Creating Time Series Predictions using LSTM Neural Networks: Normalizing Price and Tokenizing Time](https://c.mql5.com/2/82/Creating_Time_Series_Predictions_using_LSTM_Neural_Networks___LOGO.png)[Creating Time Series Predictions using LSTM Neural Networks: Normalizing Price and Tokenizing Time](https://www.mql5.com/en/articles/15063)

This article outlines a simple strategy for normalizing the market data using the daily range and training a neural network to enhance market predictions. The developed models may be used in conjunction with an existing technical analysis frameworks or on a standalone basis to assist in predicting the overall market direction. The framework outlined in this article may be further refined by any technical analyst to develop models suitable for both manual and automated trading strategies.

![Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)](https://c.mql5.com/2/82/Building_A_Candlestick_Trend_Constraint_Model_Part_5__NEXT_LOGO_2.png)[Building A Candlestick Trend Constraint Model (Part 5): Notification System (Part II)](https://www.mql5.com/en/articles/14968)

Today, we are discussing a working Telegram integration for MetaTrader 5 Indicator notifications using the power of MQL5, in partnership with Python and the Telegram Bot API. We will explain everything in detail so that no one misses any point. By the end of this project, you will have gained valuable insights to apply in your projects.

[![](https://www.mql5.com/ff/si/q0vxp9pq0887p07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fvps%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Duse.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=rktadgjlwhobyedohbrepzshvpcqrlpo&s=a93cef75a53eb5da24c98e0068b3c2b96015191a0af0d1857f5b4dd22e55e7bf&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=wdtszuheneyfzxexqcogjtscsvudxghz&ssn=1769093483542365426&ssn_dr=0&ssn_sr=0&fv_date=1769093483&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14352&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Population%20optimization%20algorithms%3A%20Resistance%20to%20getting%20stuck%20in%20local%20extrema%20(Part%20I)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909348371689336&fz_uniq=5049430807866616683&sv=2552)

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