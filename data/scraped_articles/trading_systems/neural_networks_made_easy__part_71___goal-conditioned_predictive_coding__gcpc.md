---
title: Neural networks made easy (Part 71): Goal-Conditioned Predictive Coding (GCPC)
url: https://www.mql5.com/en/articles/14012
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 6
scraped_at: 2026-01-23T11:41:52.413492
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/14012&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062650201807693390)

MetaTrader 5 / Trading systems


### Introduction

Behavior Cloning (BC) is a promising approach for solving various offline reinforcement learning problems. Instead of assessing the value of states and actions, BC directly trains the Agent behavior policy, building dependencies between the set goal, the analyzed environment state and the Agent's action. This is achieved using supervised learning methods on pre-collected offline trajectories. The familiar Decision Transformer method and its derivative algorithms have demonstrated the effectiveness of sequence modeling for offline reinforcement learning.

Previously, when using the above algorithms, we experimented with various options for setting goals to stimulate the Agent actions we needed. However, how the model learns the previously passed trajectory remained outside our attention. Now, the question arises about the applicability of studying the trajectory as a whole. This question was addressed by the authors of the paper " [Goal-Conditioned Predictive Coding for Offline Reinforcement Learning](https://www.mql5.com/go?link=https://arxiv.org/abs/2307.03406 "https://arxiv.org/abs/2307.03406")". In their paper, they explore several key questions:

1. Are offline trajectories useful for sequence modeling or do they simply provide more data for supervised policy learning?

2. What would be the most effective learning goal for trajectory representation to support policy learning? Should sequence models be trained to encode historical experience, future dynamics, or both?

3. Since the same sequence model can be used for both trajectory representation learning and policy learning, should we have the same learning goals or not?


The article presents the results of experiments in 3 artificial environments, which allow the authors to draw the following conclusions:

- Sequence modeling, if properly designed, can effectively aid decision making when the resulting trajectory representation is used as input for policy learning.

- There is a discrepancy between the optimal trajectory representation learning goal and the policy learning goal.


Based on these observations the authors of the paper have created a two-stage framework that compresses trajectory information into compact compressed representations using sequence modeling pre-training. The compressed representation is then used to train the Agent behavior policy using a simple multilayer perceptron (MLP) based model. Their proposed Goal-Conditioned Predictive Coding (GCPC) method is the most effective goal for learning trajectory representation. It provides competitive performance in all of their benchmark tests. The authors especially note its effectiveness for solving long-horizon tasks. The strong empirical performance of GCPC comes from the latent representation of past and predicted states. In this case, state prediction is performed with a focus on set goals, which provide decisive guidance for decision-making.

### 1\. Goal-Conditioned Predictive Coding algorithm

The authors of the GCPC method use the sequence modeling for offline reinforcement learning. To solve the problem of offline reinforcement learning, they use conditional, filtered or weighted imitation learning. It is assumed that there is a pre-collected set of training data. But the policies used to collect data may not be known. The training data contains a set of trajectories. Each trajectory is represented as a set of states and actions ( _St, At_). A trajectory may optionally contain a reward _Rt_, obtained at the time step _t_.

Since the trajectories are collected using unknown policies, they may not be optimal or have a sufficient level of expertise. We have already discussed that proper use of offline trajectories containing suboptimal data can lead to more effective behavioral policies. Because suboptimal trajectories may contain sub-trajectories that demonstrate useful "skills", which can be combined to solve given tasks.

The method authors believe that the Agent behavior policy should be able to accept any form of information about the state or trajectory as the input and predict the next action:

- When only the current observed state _St_ and the goal _G_ are used, the Agent policy ignores the history observations.
- When the Agent policy is a sequence model, it can employ the entire observed trajectory to predict the next action _At_.

To optimize the Agent behavior policy, a maximum likelihood objective function is usually used.

Sequence modeling can be used for decision making from two perspectives: learning trajectory representations and learning behavior policies. The first direction seeks to obtain useful representations from raw input trajectories in the form of a condensed latent representation or the pre-trained network weights. The second direction seeks to transform the observation and goal into the optimal action for completing the task.

Learning the trajectory function and policy function can be implemented using Transformer models. The authors of the GCPC method suggest that for the trajectory function, it can be useful to compress the original data into a condensed representation using sequence modeling techniques. It is also desirable to decouple trajectory representation learning from policy learning. The decoupling not only provides flexibility in choosing the goals of representation learning, but also allows us to study the impact of sequence modeling on trajectory representation learning and policy learning independently. Therefore, GCPC uses a two-stage structure with TrajNet (trajectory model) and PolicyNet (policy model). To train TrajNet, unsupervised learning methods, such as masked autoencoder or next token prediction, are used for sequence modeling. PolicyNet aims to derive effective policies using a supervised learning objective function from collected offline trajectories.

The first stage of trajectory representation training uses masked autoencoding. TrajNet receives the trajectory _ꚍ_ and, if necessary, the goal _G_, and learns to restore _τ_ from a masked view of the same trajectory. Optionally, TrajNet also generates a condensed representation of the trajectory _B_, which can be used by PolicyNet for subsequent policy training. In their paper, the authors of the GCPC method propose to feed a masked representation of the traversed trajectory as input to the autoencoder model. At the output of the Decoder, they strive to obtain an unmasked representation of the traversed trajectory and subsequent states.

In the second stage, TrajNet is applied to the unmasked observed trajectory _ꚍ_ to get a condensed representation of the trajectory _B_. PolicyNet then predicts the action _A_ given the observed trajectory (or the current state of the environment), the goal _G_ and the condensed trajectory representation _B_.

The proposed framework provides a unified view for comparing different designs for implementing representation learning and policy learning. Many existing methods can be considered as special cases of the proposed structure. For example, for the DT implementation, the trajectory representation function is set to be the identity mapping function of the input trajectory, and the policy is trained to autoregressively generate actions.

Authors' [visualization](https://www.mql5.com/go?link=https://brown-palm.github.io/GCPC/ "https://brown-palm.github.io/GCPC/") method is presented below.

[![](https://c.mql5.com/2/63/gcpct1c.gif)](https://www.mql5.com/go?link=https://brown-palm.github.io/GCPC/ "https://brown-palm.github.io/GCPC/")

### 2\. Implementation using MQL5

We have considered the theoretical aspects of the Goal-Conditioned Predictive Coding method. Next, let's move on to its implementation using MQL5. Here you should primarily pay attention to the different number of models used at different stages of training and operation of the model.

#### 2.1 Model architecture

In the first stage, the authors of the method propose to train a trajectory representation model. The model architecture uses a Transformer. To train it, we need to build an Autoencoder. In the second stage, we will use only the trained Encoder. Therefore, in order not to "drag" an unnecessary Decoder to the second stage of training, we will divide the Autoencoder into 2 models: Encoder and Decoder. The architecture of the models is presented in the CreateTrajNetDescriptions method. In the parameters, the method receives pointers to 2 dynamic arrays to indicate the architecture of the specified models.

In the body of the method, we check the received pointers and, if necessary, create new dynamic array objects.

```
bool CreateTrajNetDescriptions(CArrayObj *encoder, CArrayObj *decoder)
  {
//---
   CLayerDescription *descr;
//---
   if(!encoder)
     {
      encoder = new CArrayObj();
      if(!encoder)
         return false;
     }
   if(!decoder)
     {
      decoder = new CArrayObj();
      if(!decoder)
         return false;
     }
```

First, let's describe the architecture of the Encoder. We feed only historical price movement and analyzed indicators data into the model.

```
//--- Encoder
   encoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = (HistoryBars * BarDescr);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Please note that, unlike the previously discussed models, at this stage we do not use either data on the account status or information about the actions previously performed by the Agent. There is an opinion that in some cases information about previous actions can have a negative impact. Therefore, the authors of the GCPC method excluded it from the source data. Information about the account state does not affect the environment state. Thus, it is not important for predicting subsequent environmental states.

We always feed the unprocessed source data into the model. Therefore, in the next layer we use batch normalization to bring the source data into a comparable form.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

After preprocessing the data, we need to implement random data masking, which is provided by the GCPC algorithm. To implement this functionality, we will use the [DropOut](https://www.mql5.com/en/articles/9112) layer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronDropoutOCL;
   descr.count = prev_count;
   descr.probability = 0.8f;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Please note that in general practice, it is not recommended to use the batch normalization layer and DropOut together in one model. This is due to the fact that excluding some information and replacing it with zero values distorts the original data distribution and has a negative impact on the operation of the batch normalization layer. For this reason, we first normalize the data and only then mask it. This way the batch normalization layer works with the full data set and minimizes the impact of the DropOut layer on its operation. At the same time, we implement a masking functionality to train our model to recover missing data and ignore outliers inherent in a stochastic environment.

Next in the model of our Encoder comes a convolutional block to reduce the dimension of the data and identify stable patterns.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = HistoryBars;
   descr.window = BarDescr;
   descr.step = BarDescr;
   int prev_wout = descr.window_out = BarDescr / 2;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = prev_count;
   descr.step = prev_wout;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count;
   descr.window = prev_wout;
   descr.step = prev_wout;
   prev_wout = descr.window_out = 8;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = prev_count;
   descr.step = prev_wout;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The result of the above-described processing of the source data is fed into a block of fully connected layers, which allows us to obtain an embedding of the initial state.

```
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = EmbeddingSize;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

In addition to historical data, the authors of the GCPC method propose to feed the Encoder with goal embedding and _Slot tokens_ (results of previous Encoder passes). Our global goal of obtaining the maximum possible profit does not affect the environment and we omit it. Instead, we add the results of the last pass of our Encoder to the model using a concatenation layer.

```
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = 2 * EmbeddingSize;
   descr.window = prev_count;
   descr.step = EmbeddingSize;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Further data processing is performed using [GPT](https://www.mql5.com/en/articles/9025) models. To implement it, we first create a data stack using an embedding layer.

```
//--- layer 10
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronEmbeddingOCL;
   prev_count = descr.count = GPTBars;
     {
      int temp[] = {EmbeddingSize, EmbeddingSize};
      ArrayCopy(descr.windows, temp);
     }
   prev_wout = descr.window_out = EmbeddingSize;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

It is followed by the attention block. Previously, we already created a data sparse process using the DropOut layer, so in this model I **didn't use** the sparse attention layer.

```
//--- layer 11
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMLMHAttentionOCL;
   descr.count = prev_count * 2;
   descr.window = prev_wout;
   descr.step = 4;
   descr.window_out = 16;
   descr.layers = 4;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

At the output of the Encoder, we reduce the data dimension with a fully connected layer and normalize the data with the SoftMax function.

```
//--- layer 12
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = EmbeddingSize;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 13
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = prev_count;
   descr.step = 1;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

We will feed a condensed representation of the trajectory into the Decoder input.

```
//--- Decoder
   decoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The initial data of the Decoder was obtained from the previous model and already has a comparable form. This means that we do not need the batch normalization layer in this case. The resulting data is processed in the fully connected layer.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (HistoryBars + PrecoderBars) * EmbeddingSize;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Then we process it in the attention layer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMLMHAttentionOCL;
   prev_count = descr.count = prev_count / EmbeddingSize;
   prev_wout = descr.window = EmbeddingSize;
   descr.step = 4;
   descr.window_out = 16;
   descr.layers = 2;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The architecture of our Decoder is built in such a way that at the output of the attention block we have an embedding for each candlestick of the analyzed and predicted environment states. Here we need to understand the purpose of data. Let's consider the following.

Why do we analyze indicators? Trend indicators show us the direction of the trend. Oscillators are designed to indicate overbought and oversold zones, thereby indicating points of possible market reversal. All this is valuable at the current moment in time. Would such prediction be valuable with some depth? My personal opinion is that, taking into account the data prediction error, the indicator forecasting value is close to zero. Ultimately, we receive profit and loss from changes in the price of the instrument, and not from the indicator values. Therefore, we will predict price movement data at the output of the Decoder.

Let's recall what information about price movement we save in the experience replay buffer. The information includes 3 deviations:

- candlestick body Close - Open
- High - Open
- Low - Open

So, we will predict these values. To independently restore values from candlestick embeddings, we will use the layer of the [ensemble of models](https://www.mql5.com/en/articles/12508#para3).

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronMultiModels;
   descr.count = 3;
   descr.window = prev_wout;
   descr.step = prev_count;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!decoder.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

This concludes the description of the autoencoder architecture for the first stage of training of trajectory presentation TrajNet. But before moving on to model training Expert Advisors, I propose to complete the work on describing the architecture of the models. Let's look at the architecture of the second-stage policy training models PolicyNet. The architecture is provided in the CreateDescriptions method.

Contrary to expectations, at the second stage we will train not one model of the Actor behavior policy, but three models.

The first one is a small model of the current state encoder. Do not confuse it with the Autoencoder Encoder trained in the first stage. This model will combine a condensed representation of the trajectory from the Autoencoder Encoder with information about the state of the account into a single representation.

The second one is Actor policy model, which we discussed above.

And the third is a model of goal setting based on the analysis of the condensed trajectory representation.

As usual, in the method parameters, we pass pointers to dynamic arrays describing the model architecture. In the body of the method, we check the relevance of the received pointers and, if necessary, create new instances of dynamic array objects.

```
bool CreateDescriptions(CArrayObj *actor, CArrayObj *goal, CArrayObj *encoder)
  {
//---
   CLayerDescription *descr;
//---
   if(!actor)
     {
      actor = new CArrayObj();
      if(!actor)
         return false;
     }
   if(!goal)
     {
      goal = new CArrayObj();
      if(!goal)
         return false;
     }
   if(!encoder)
     {
      encoder = new CArrayObj();
      if(!encoder)
         return false;
     }
```

As mentioned above, we feed a condensed representation of the trajectory into the Encoder.

```
//--- State Encoder
   encoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

The received data is combined with information about the account state in the concatenation layer.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = AccountDescr;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

At this point, the tasks assigned to the Encoder are considered completed and we move on to the architecture of the Actor, which receives the results of the work of the previous model as input.

```
//--- Actor
   actor.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = LatentCount;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

We combine the received data with the set goal.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = NRewards;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

Process it with fully connected layers.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

At the Actor output, we add stochasticity to the policy of its behavior.

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronVAEOCL;
   descr.count = NActions;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

And last but not least is the goal generation model. I think it's no secret that the ability to generate profit strongly depends on various aspects of the environment state. Therefore, based on past experience, I decided to add a separate model for generating goals depending on the environment state.

We will feed into the model input a condensed representation of the observed trajectory. Here we speak about trajectories without considering the account state. Our reward function is built to operate on relative values without being tied to a specific deposit size. Therefore, to set goals, we proceed only from an analysis of the environment without considering the state of the account.

```
//--- Goal
   goal.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!goal.Add(descr))
     {
      delete descr;
      return false;
     }
```

The received data is analyzed by 2 fully connected layers.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!goal.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!goal.Add(descr))
     {

      delete descr;
      return false;
     }
```

At the model output, we use a fully parameterized [quantile function](https://www.mql5.com/en/articles/11804). The advantage of this solution is that it returns the most probable result, rather than the average value typical of a fully connected layer. The differences in results are most noticeable for distributions with 2 or more vertices.

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFQF;
   descr.count = NRewards;
   descr.window_out = 32;
   descr.optimization = ADAM;
   descr.activation = None;
   if(!goal.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

#### 2.2 Model of interaction with the environment

We continue the implementation of the Goal-Conditioned Predictive Coding method. After describing the model architectures, we move on to the implementation of the algorithms. First we will implement an Expert Advisor for interacting with the environment and collecting data for the training sample. The authors of the method did not focus on the method for collecting training data. In fact, the training dataset can be collected in any available way, including the algorithms we discussed earlier: [ExORL](https://www.mql5.com/en/articles/13819) and [Real-ORL](https://www.mql5.com/en/articles/13854). It is only necessary to match data recording and presentation formats. But to optimize pre-trained models, we need an EA that, in the process of interacting with the environment, would use the behavioral policy we have learned and save the results of interaction into a trajectory. We implement this functionality in the EA ..\\Experts\\GCPC\\Research.mq5. The basic principles of constructing the EA algorithm match those used in previous works. However, the number of models leaves its mark. Let's focus specifically on some of the EA's methods.

In this Expert Advisor, we will use 4 models.

```
CNet                 Encoder;
CNet                 StateEncoder;
CNet                 Actor;
CNet                 Goal;
```

Pre-trained models are loaded in the OnInit EA initialization method. Find the complete code of the method in the attachment. I will only mention the changes here.

First we load the AutoEncoder Encoder model. If there is a loading error, we initialize a new model with random parameters.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
........
........
//--- load models
   float temp;
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreateTrajNetDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
      //---
     }
```

Then we load the 3 remaining models. If necessary, we also initialize them with random parameters.

```
   if(!StateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !Goal.Load(FileName + "Goal.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *goal = new CArrayObj();
      CArrayObj *encoder = new CArrayObj();
      if(!CreateDescriptions(actor, goal, encoder))
        {
         delete actor;
         delete goal;
         delete encoder;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !StateEncoder.Create(encoder) || !Goal.Create(goal))
        {
         delete actor;
         delete goal;
         delete encoder;
         return INIT_FAILED;
        }
      delete actor;
      delete goal;
      delete encoder;
      //---
     }
```

Transfer all models into a single OpenCL context.

```
   StateEncoder.SetOpenCL(Actor.GetOpenCL());
   Encoder.SetOpenCL(Actor.GetOpenCL());
   Goal.SetOpenCL(Actor.GetOpenCL());
```

Be sure to turn off the Encoder model training mode.

```
   Encoder.TrainMode(false);
```

Please note that although we do not plan to use backpropagation methods within this EA, we use the DropOut layer in the Encoder. Therefore, we need to change the training mode to disable masking under operating conditions of the model.

Next, we check the consistency of the architecture of the loaded models.

```
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   Encoder.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the Encoder does not match the embedding size (%d <> %d)", EmbeddingSize,
                                                                                                  Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(),
                                                                                        (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   PrevBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   PrevEquity = AccountInfoDouble(ACCOUNT_EQUITY);
//---
   return(INIT_SUCCEEDED);
  }
```

Interaction with the environment is implemented in the OnTick method. At the beginning of the method, we check for the occurrence of a new bar opening event and, if necessary, load historical data. The received information is transferred to data buffers. These operations have been copied from previous implementations without changes, so we will not dwell on them. Let us consider only the sequence of calling methods for the feed-forward model pass. As provided by the GCPC algorithm, we first call the Encoder's feed-forward method.

```
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
........
........
//---
   if(!Encoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CNet*)GetPointer(Encoder)) ||
```

Note that the model uses itself recurrently as the data source for the second flow of information.

Next we call the feed-forward method of the State Encoder and Goal Model. Both models use a condensed representation of the observed trajectory as input data.

```
      !StateEncoder.feedForward((CNet *)GetPointer(Encoder), -1, (CBufferFloat *)GetPointer(bAccount)) ||
      !Goal.feedForward((CNet *)GetPointer(Encoder), -1, (CBufferFloat *)NULL) ||
```

The results of these models are fed into the input of the Actor policy model to generate a subsequent action.

```
      !Actor.feedForward((CNet *)GetPointer(StateEncoder), -1, (CNet *)GetPointer(Goal)))
      return;
```

We should not forget to check the results of the operations.

Next, the results of the Actor model are decoded and actions are performed in the environment, followed by saving the experience gained into a trajectory. The algorithm of these operations is used without changes. You can find the complete code of the EA for interacting with the environment in the attachment.

#### 2.3 Training the trajectory function

After collecting the training dataset, we move on to building model training EAs. According to the GCPC algorithm, the first step is to train the TrajNet trajectory function model. We implement this functionality in the EA ...\\Experts\\GCPC\\StudyEncoder.mq5.

As we discussed in the theoretical part of this article, at the first stage we train a masked autoencoder model, which in our case consists of 2 models: Encoder and Decoder.

```
//+------------------------------------------------------------------+
//| Input parameters                                                 |
//+------------------------------------------------------------------+
input int                  Iterations     = 1e4;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
STrajectory          Buffer[];
CNet                 Encoder;
CNet                 Decoder;
```

Please pay attention to the following moment. The encoder recursively uses its own results from the previous pass as the initial data of the second flow of information. For a feed-forward pass we can simply use a pointer to the model itself. However, for a backpropagation pass, this approach is not acceptable. Because the model's results buffer will contain the data from the last pass, not the previous one. This is not acceptable for our model training process. Therefore, we need an additional data buffer to store the results of the previous pass.

```
CBufferFloat         LastEncoder;
```

In the EA initialization method, we first load the training dataset and check the results of the operations. IF there is no data for training the models, all subsequent operations are meaningless.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
```

After successfully loading the training dataset, we try to open the pre-trained models. If an error occurs, we initialize new models with random parameters.

```
//--- load models
   float temp;
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !Decoder.Load(FileName + "Dec.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Init new models");
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *decoder = new CArrayObj();
      if(!CreateTrajNetDescriptions(encoder, decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      if(!Encoder.Create(encoder) || !Decoder.Create(decoder))
        {
         delete encoder;
         delete decoder;
         return INIT_FAILED;
        }
      delete encoder;
      delete decoder;
      //---
     }
```

We place both models in a single OpenCL context.

```
   OpenCL = Encoder.GetOpenCL();
   Decoder.SetOpenCL(OpenCL);
```

Check the compatibility of the model architectures.

```
   Encoder.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the Encoder does not match the embedding size count (%d <> %d)", EmbeddingSize,
                                                                                                 Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(),
                                                                                       (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   Decoder.GetLayerOutput(0, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("Input size of Decoder doesn't match Encoder output (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
```

After successfully passing the check block, we initialize the auxiliary buffers in the same OpenCL context.

```
   if(!LastEncoder.BufferInit(EmbeddingSize,0) ||
      !Gradient.BufferInit(EmbeddingSize,0) ||
      !LastEncoder.BufferCreate(OpenCL) ||
      !Gradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
      return INIT_FAILED;
     }
```

At the end of the EA initialization method, we generate a custom event for the start of the learning process.

```
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
```

The actual training process is implemented in the Train method. In this method, we traditionally combine the Goal-Conditioned Predictive Coding algorithm with our developments from previous articles. At the beginning of the method, we create a vector of probabilities of using trajectories to train models.

```
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
```

However, note that in this case there is no practical effect from weighing trajectories. In the process of training the Autoencoder, we use only historical data of price movement and analyzed indicators. All our trajectories are collected on one historical interval of one instrument. Therefore, for our Autoencoder, all trajectories contain identical data. Nevertheless, I will leave this functionality for the future to enable the possibility of training models on trajectories of various time intervals and instruments.

Next we initialize local variables and vectors. Let's pay attention to the vector of standard deviations. Its size is equal to the vector of Decoder results. The principles of its use will be discussed a little later.

```
   vector<float> result, target;
   matrix<float> targets;
   STD = vector<float>::Zeros((HistoryBars + PrecoderBars) * 3);
   int std_count = 0;
   uint ticks = GetTickCount();
```

After the preparatory work, implement a system of model training cycles. The Encoder uses a GPT block with a stack of latent states, which is sensitive to the sequence of the source data. Therefore, when training models, we will use entire batches of sequential states from each sampled trajectory.

In the body of the outer loop, taking into account the previously generated probabilities, we sample one trajectory and randomly select an initial state on it.

```
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 50;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 3 - PrecoderBars - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
```

Then we clear the model stacks and the buffer of previous Encoder results.

```
      Encoder.Clear();
      Decoder.Clear();
      LastEncoder.BufferInit(EmbeddingSize,0);
```

Now everything is ready to begin the nested learning loop on the selected trajectory.

```
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      for(int i = state; i < end; i++)
        {
         State.AssignArray(Buffer[tr].States[i].state);
```

In the body of the loop, we fill the initial data buffer from the training dataset and sequentially call the feed-forward pass methods of our models. First the Encoder.

```
         if(!LastEncoder.BufferWrite() || !Encoder.feedForward((CBufferFloat*)GetPointer(State), 1, false,
                                                               (CBufferFloat*)GetPointer(LastEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
```

Then the Decoder.

```
         if(!Decoder.feedForward(GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            break;
           }
```

After successfully completing the feed-forward passes of our models, we need to run a backpropagation pass and adjust the model parameters. But first we need to prepare the target values of the Decoder results. As you remember, at the output of the decoder we plan to receive the reconstructed values and results of price change predictions for several candlesticks, which are indicated in the first three elements of the array describing the state of each candlestick. To obtain this data, we will create a matrix, in each row of which we will store descriptions of the environment state in the desired time range. And then we take only the first 3 columns of the resulting matrix. These will be our target values.

```
         target.Assign(Buffer[tr].States[i].state);
         ulong size = target.Size();
         targets = matrix<float>::Zeros(1, size);
         targets.Row(target, 0);
         if(size > BarDescr)
            targets.Reshape(size / BarDescr, BarDescr);
         ulong shift = targets.Rows();
         targets.Resize(shift + PrecoderBars, 3);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target.Assign(Buffer[tr].States[i + t].state);
            if(size > BarDescr)
              {
               matrix<float> temp(1, size);
               temp.Row(target, 0);
               temp.Reshape(size / BarDescr, BarDescr);
               temp.Resize(size / BarDescr, 3);
               target = temp.Row(temp.Rows() - 1);
              }
            targets.Row(target, shift + t);
           }
         targets.Reshape(1, targets.Rows()*targets.Cols());
         target = targets.Row(0);
```

Inspired by the results of the previous [article](https://www.mql5.com/en/articles/13982) describing the use of closed-form operators, I decided to slightly change the learning process and place more emphasis on large deviations. So, I simply ignore minor deviations considering them to be forecast errors. Therefore, at this stage, I calculate the moving standard deviation of the model's results from the target values.

```
         Decoder.getResults(result);
         vector<float> error = target - result;
         std_count = MathMin(std_count, 999);
         STD = MathSqrt((MathPow(STD, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
```

It should be noted here that we control the deviation for each parameter separately.

We then check whether the current prediction error exceeds a threshold value. The backpropagation pass is only performed if there is a prediction error above the threshold value **_for at least one parameter_**.

```
         vector<float> check = MathAbs(error) - STD * STD_Multiplier;
         if(check.Max() > 0)
           {
            //---
            Result.AssignArray(CAGrad(error) + result);
            if(!Decoder.backProp(Result, (CNet *)NULL) ||
               !Encoder.backPropGradient(GetPointer(LastEncoder), GetPointer(Gradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               break;
              }
           }
```

Please note that this approach has several nuances. The average error of the model is calculated only when performing a backpropagation pass. Therefore, in this case, the current error affects the average error only when the threshold value is exceeded. As a result, the small errors that we ignore do not affect the value of the average error of the model. Thus, we get an overestimation of this metric. This is not critical, since the value is purely informative.

The "other side of the coin" is that by focusing only on significant deviations, we help the model to identify the main drivers that influence certain performance values. The use of the moving standard deviation as a guideline for the threshold value allows us to reduce the threshold of permissible error during the learning process. Which enables finer tuning of the model.

At the end of the loop iterations, we save the results of the Encoder to an auxiliary buffer and inform the user about the progress of the model training process.

```
         Encoder.getResults(result);
         LastEncoder.AssignArray(result);
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / ((end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Decoder", percent, Decoder.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
```

After completing all iterations of the training loop system, we clear the comments field on the chart, display information about the training results in the log, and initiate the EA shutdown.

```
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Decoder", Decoder.getRecentAverageError());
   ExpertRemove();
//---
  }
```

Be sure to remember to save the trained models and clear the memory in the EA deinitialization method.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
     {
      Encoder.Save(FileName + "Enc.nnw", 0, 0, 0, TimeCurrent(), true);
      Decoder.Save(FileName + "Dec.nnw", Decoder.getRecentAverageError(), 0, 0, TimeCurrent(), true);
     }
   delete Result;
   delete OpenCL;
  }
```

#### 2.4 Policy Training

The next step is to train the Agent behavior policy, which is implemented in the EA ...\\Experts\\GCPC\\Study.mq5. Here we will train a state encoder model, which is essentially an integral part of our Agent model. We will also train the goal setting model.

Although it is functionally possible to separate the process of training the Agent behavior policy and the goal setting model into 2 separate programs, I decided to combine them within one EA. As will be seen from the implementation algorithm, these 2 processes are closely intertwined and use a large amount of common data. In this case, it would hardly be efficient to divide model training into 2 parallel processes with a large share of duplicated operations.

This EA, similar to the EA for interaction with the environment, uses 4 models, 3 of which are trained in it.

```
CNet                 Actor;
CNet                 StateEncoder;
CNet                 Encoder;
CNet                 Goal;
```

In the OnInit EA initialization method, as in the EA discussed above, we load the training dataset.

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   ResetLastError();
   if(!LoadTotalBase())
     {
      PrintFormat("Error of load study data: %d", GetLastError());
      return INIT_FAILED;
     }
```

Next we load the models. First, we try to open a pre-trained Encoder. It must be trained in the first stage of the Goal-Conditioned Predictive Coding algorithm. If this model is not available, we cannot move to the next stage.

```
//--- load models
   float temp;
   if(!Encoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true))
     {
      Print("Cann't load Encoder model");
      return INIT_FAILED;
     }
```

After successfully reading the Encoder model, we try to open the remaining models. All of them are trained in this EA. Therefore, when any error occurs, we create new models and initialize them with random parameters.

```
   if(!StateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !Goal.Load(FileName + "Goal.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *goal = new CArrayObj();
      CArrayObj *encoder = new CArrayObj();
      if(!CreateDescriptions(actor, goal, encoder))
        {
         delete actor;
         delete goal;
         delete encoder;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) || !StateEncoder.Create(encoder) || !Goal.Create(goal))
        {
         delete actor;
         delete goal;
         delete encoder;
         return INIT_FAILED;
        }
      delete actor;
      delete goal;
      delete encoder;
      //---
     }
```

We then move all models into a single OpenCL context. We also set the Encoder training mode to false to disable masking of the source data.

The next step is to check the compatibility of the architectures of all loaded models to eliminate possible errors when transferring data between models.

```
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)", NActions, Result.Total());
      return INIT_FAILED;
     }
   Encoder.getResults(Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("The scope of the Encoder does not match the embedding size (%d <> %d)", EmbeddingSize, Result.Total());
      return INIT_FAILED;
     }
//---
   Encoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)", Result.Total(),
                                                                                               (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
//---
   StateEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != EmbeddingSize)
     {
      PrintFormat("Input size of State Encoder doesn't match Bottleneck (%d <> %d)", Result.Total(), EmbeddingSize);
      return INIT_FAILED;
     }
//---
   StateEncoder.getResults(Result);
   int latent_state = Result.Total();
   Actor.GetLayerOutput(0, Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Actor doesn't match output State Encoder (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Goal.GetLayerOutput(0, Result);
   latent_state = Result.Total();
   Encoder.getResults(Result);
   if(Result.Total() != latent_state)
     {
      PrintFormat("Input size of Goal doesn't match output Encoder (%d <> %d)", Result.Total(), latent_state);
      return INIT_FAILED;
     }
//---
   Goal.getResults(Result);
   if(Result.Total() != NRewards)
     {
      PrintFormat("The scope of Goal doesn't match rewards count (%d <> %d)", Result.Total(), NRewards);
      return INIT_FAILED;
     }
```

After successfully passing all the necessary controls, we create auxiliary buffers in the OpenCL context.

```
   if(!bLastEncoder.BufferInit(EmbeddingSize, 0) ||
      !bGradient.BufferInit(MathMax(EmbeddingSize, AccountDescr), 0) ||
      !bLastEncoder.BufferCreate(OpenCL) ||
      !bGradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
      return INIT_FAILED;
     }
```

Generate a custom event for the start of the learning process.

```
   if(!EventChartCustom(ChartID(), 1, 0, 0, "Init"))
     {
      PrintFormat("Error of create study event: %d", GetLastError());
      return INIT_FAILED;
     }
//---
   return(INIT_SUCCEEDED);
  }
```

In the EA deinitialization method, we save the trained models and remove the dynamic objects used.

```
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if(!(reason == REASON_INITFAILED || reason == REASON_RECOMPILE))
     {
      Actor.Save(FileName + "Act.nnw", 0, 0, 0, TimeCurrent(), true);
      StateEncoder.Save(FileName + "StEnc.nnw", 0, 0, 0, TimeCurrent(), true);
      Goal.Save(FileName + "Goal.nnw", 0, 0, 0, TimeCurrent(), true);
     }
   delete Result;
   delete OpenCL;
  }
```

The process of training models is implemented in the Train method. In the body of the method, we first generate a buffer of probabilities for choosing trajectories to train the models. We weigh all trajectories in the training set by their profitability. The most profitable passes have a greater likelihood of participating in the learning process.

```
//+------------------------------------------------------------------+
//| Train function                                                   |
//+------------------------------------------------------------------+
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
```

Then we initialize the local variables. Here you can notice two vectors of standard deviations, which we will use for policy models and goal setting.

```
   vector<float> result, target;
   matrix<float> targets;
   STD_Actor = vector<float>::Zeros(NActions);
   STD_Goal = vector<float>::Zeros(NRewards);
   int std_count = 0;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
```

Although none of the trained models have recurrent blocks and stacks in their architecture, we still create a loop system to train the models. Because the initial data for the trained models is generated by the Encoder, which operates the GPT architecture.

In the body of the outer loop, we sample the trajectory and the initial state on it.

```
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 50;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (Buffer[tr].Total - 2 - PrecoderBars - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
```

We clear the Encoder stack and the buffer of its latest results.

```
      Encoder.Clear();
      bLastEncoder.BufferInit(EmbeddingSize, 0);
```

Note that we use a buffer to record the last state of the Encoder, although we are not going to execute backpropagation passes for this model. For feed-forward passes, we could use a pointer to the model, as was implemented in the environment interaction EA. However, when moving to a new trajectory, we need to reset not only the stack of latent states, but also the model's result buffer. This is easier to do using an additional buffer.

In the body of the nested loop, we load the analyzed state data from the training dataset and generate a condensed representation of it using the Encoder model.

```
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
      for(int i = state; i < end; i++)
        {
         bState.AssignArray(Buffer[tr].States[i].state);
         //---
         if(!bLastEncoder.BufferWrite() ||
            !Encoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false, (CBufferFloat*)GetPointer(bLastEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Next, we fill the account state description buffer, which is supplemented with timestamp harmonics.

```
         float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
         bAccount.Clear();
         bAccount.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
         bAccount.Add(Buffer[tr].States[i].account[1] / PrevBalance);
         bAccount.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
         bAccount.Add(Buffer[tr].States[i].account[2]);
         bAccount.Add(Buffer[tr].States[i].account[3]);
         bAccount.Add(Buffer[tr].States[i].account[4] / PrevBalance);
         bAccount.Add(Buffer[tr].States[i].account[5] / PrevBalance);
         bAccount.Add(Buffer[tr].States[i].account[6] / PrevBalance);
         double time = (double)Buffer[tr].States[i].account[7];
         double x = time / (double)(D'2024.01.01' - D'2023.01.01');
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_MN1);
         bAccount.Add((float)MathCos(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_W1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         x = time / (double)PeriodSeconds(PERIOD_D1);
         bAccount.Add((float)MathSin(x != 0 ? 2.0 * M_PI * x : 0));
         if(bAccount.GetIndex() >= 0)
            bAccount.BufferWrite();
```

The condensed representation of the analyzed state of the environment is combined with a vector describing the state of the account.

```
         //--- State embedding
         if(!StateEncoder.feedForward((CNet *)GetPointer(Encoder), -1, (CBufferFloat*)GetPointer(bAccount)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

And now, to run a feed-forward pass of the Actor, we need to indicate the goal. As in the case of Decision Transformer, at this stage we use the actual results obtained from interaction with the environment as goals. The actual actions of the Agent are used as the target results of the policy. In this way, we build connections between goal and action in a specific environmental state. But there is one point. When training the Autoencoder, we aimed to obtain forecast data several candles ahead. Therefore, we now expect to have predictive information about several subsequent candles in a condensed representation of the current state. It is logical to assume that the Agent actions performed at this stage should be designed to obtain a result in the predicted time period. We could take the total reward for the forecast period as the goal for the action taken. However, why should a currently open deal be closed only after the expiration of the forecast period? It can be closed either earlier or later. For the "later" case, we cannot look beyond the forecast values. Therefore, we can only take the result at the end of the forecast period. But if the direction of price movement changes within the forecast period, the transaction should be closed earlier. Therefore, our potential goal should be the maximum value over the forecast period, taking into account the discount factor.

The problem is that the experience replay buffer stores cumulative rewards until the end of the episode. However, we need the total amount of rewards from the analyzed state over the forecast data horizon. Therefore, we first restore the reward at each step without considering the discount factor.

```
         targets = matrix<float>::Zeros(PrecoderBars, NRewards);
         result.Assign(Buffer[tr].States[i + 1].rewards);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target = result;
            result.Assign(Buffer[tr].States[i + t + 2].rewards);
            target = target - result * DiscFactor;
            targets.Row(target, t);
           }
```

Then we sum them up in reverse order, taking into account the discount factor.

```
         for(int t = 1; t < PrecoderBars; t++)
           {
            target = targets.Row(t - 1) + targets.Row(t) * MathPow(DiscFactor, t);
            targets.Row(target, t);
           }
```

From the resulting matrix, we select the row with the maximum reward, which will be our goal.

```
         result = targets.Sum(1);
         ulong row = result.ArgMax();
         target = targets.Row(row);
         bGoal.AssignArray(target);
```

I quite agree with the observation that the profit (or loss) obtained at subsequent time steps can be associated with trades that the Agent made earlier or later. There are two points here.

Mentioning previously performed deals is not entirely correct. Because the fact that the Agent left them open is an action of the current moment. Therefore, their subsequent result is a consequence of this action.

As for subsequent actions, in the framework of trajectory analysis we analyze not individual actions, but the behavior policy of the Actor as a whole. Consequently, the goal is set for the policy for the foreseeable future, and not for a separate action. From this point of view, setting a maximum goal for the forecast period is quite relevant.

Taking into account the prepared goal, we have enough data to execute a feed-forward pass of the Actor.

```
         //--- Actor
         if(!Actor.feedForward((CNet *)GetPointer(StateEncoder), -1, (CBufferFloat*)GetPointer(bGoal)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Next, we need to adjust the model parameters to minimize the error between the predicted actions and those actually performed in the process of interaction with the environment. Here we use supervised learning methods supplemented with an emphasis on maximum deviations. As in the algorithm described above, we first calculate the moving standard deviation error of the forecasts for each parameter.

```
         target.Assign(Buffer[tr].States[i].action);
         target.Clip(0, 1);
         Actor.getResults(result);
         vector<float> error = target - result;
         std_count = MathMin(std_count, 999);
         STD_Actor = MathSqrt((MathPow(STD_Actor, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
```

Then we compare the current error with the threshold value. A backpropagation pass is executed only if there is a deviation above the threshold in at least one parameter.

```
         check = MathAbs(error) - STD_Actor * STD_Multiplier;
         if(check.Max() > 0)
           {
            Result.AssignArray(CAGrad(error) + result);
            if(!Actor.backProp(Result, (CBufferFloat *)GetPointer(bGoal), (CBufferFloat *)GetPointer(bGradient)) ||
               !StateEncoder.backPropGradient(GetPointer(bAccount), (CBufferFloat *)GetPointer(bGradient)))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
```

After updating the Actor parameters, we move on to training the goal setting model. Unlike the Actor, it uses only a condensed representation of the analyzed state received from the Encoder as initial data. Also, we don't need to prepare additional data before performing a feed-forward pass.

```
         //--- Goal
         if(!Goal.feedForward((CNet *)GetPointer(Encoder), -1, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

For the target values for model training, we will use the goals that were set above for the Actor policy. But with a small addition. In many works, it is recommended to use an increasing factor to the actual results obtained when forming goals for trained policies. This should stimulate behavior policy to choose more optimal actions. We will immediately train the goal setting model for better results. To do this, when forming a vector of target values, we will increase actual achievements by 2 times. However, please note the following. We cannot simply multiply the vector of actual rewards by 2. Since among the rewards received there may also be negative values, and multiplying them by 2 will only worsen expectations. Therefore, we first determine the sign of the reward.

```
         target=targets.Row(row);
         result = target / (MathAbs(target) + FLT_EPSILON);
```

As a result of this operation, we expect to obtain a vector containing "-1" for negative values and "1" for positive values. Raising the vector from "2" to the power of the resulting vector, we get "2" for positive values and "½" for negative ones.

```
        result = MathPow(vector<float>::Full(NRewards, 2), result);
```

Now we can multiply the vector of actual results by the vector of coefficients obtained above to double the expected reward. We will use this as target values for training our goal setting model.

```
         target = target * result;
         Goal.getResults(result);
         error = target - result;
         std_count = MathMin(std_count, 999);
         STD_Goal = MathSqrt((MathPow(STD_Goal, 2) * std_count + MathPow(error, 2)) / (std_count + 1));
         std_count++;
         check = MathAbs(error) - STD_Goal * STD_Multiplier;
         if(check.Max() > 0)
           {
            Result.AssignArray(CAGrad(error) + result);
            if(!Goal.backProp(Result, (CBufferFloat *)NULL, (CBufferFloat *)NULL))
              {
               PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
               Stop = true;
               break;
              }
           }
```

Here we also exploit the idea of using closed-form expressions to optimize the model with an emphasis on maximum deviations.

At this stage, we optimized the parameters of all trained models. We save the results of the Encoder to the appropriate buffer.

```
         Encoder.getResults(result);
         bLastEncoder.AssignArray(result);
```

Inform the user about the progress of the learning process and move on to the next iteration of the loop system.

```
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / ((end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent, Actor.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Goal", percent, Goal.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
```

After completing all iterations of the model training loop system, we clear the comments field on the symbol chart. Print the training results to the log and complete the EA operation.

```
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Actor", Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__, "Goal", Goal.getRecentAverageError());
   ExpertRemove();
//---
  }
```

This concludes the description of the programs used by the algorithm. Find the complete code of all programs used in the article in the attachment. This attachment also contains the EA for testing trained models, which we will not dwell on now.

### 3\. Test

We have done quite a lot of work to implement the Goal-Conditioned Predictive Coding method using MQL5. The size of this article confirms the amount of work done. Now it is time to move on to testing its results.

As usual, we train and test models using historical data for EURUSD, H1. The models are trained on historical data for the first 7 months of 2023. To test the trained model, we use historical data from August 2023, which immediately follows the training historical period.

Training was performed iteratively. First, a training dataset was collected, which we collected in 2 stages. At the first stage, we saved passes based on real signal data into the training set, as was proposed in the [Real-ORL](https://www.mql5.com/en/articles/13854) method. Then the training dataset was supplemented with passes using the EA ...\\Experts\\GCPC\\Research.mq5 and random policies.

The Autoencoder was trained on this data using the EA ...\\Experts\\GCPC\\StudyEncoder.mq5. As mentioned above, for the purposes of training this EA, all passes are identical. Model training does not require additional updating of the training dataset. Therefore, we train a masked Autoencoder until acceptable results are obtained.

At the second stage, we train the Agent behavior policy and goal setting model. Here we use an iterative approach, in which we train models and then update the training data. I must say that at this stage I was surprised. The training process turned out to be quite stable and with good dynamics of results. During the training process, a policy was obtained that was capable of generating profit both in the training and test time periods.

### Conclusion

In this article we got acquainted with a rather interesting Goal-Conditioned Predictive Coding method. Its main contribution is the division of the model training process into 2 sub-processes: trajectory learning and separate policy learning. When learning a trajectory, attention is focused on the possibility of projecting observed trends onto future states, which generally increases the information content of the data transmitted to the Agent for decision-making.

In the practical part of this article, we implemented our vision of the proposed method using MQL5 and in practice confirmed the effectiveness of the proposed approach.

However, once again, I would like to pay your attention to the fact that all the programs presented in the article are intended only for technology demonstration purposes. They are not ready for use in real financial markets.

### References

- [Goal-Conditioned Predictive Coding for Offline Reinforcement Learning](https://www.mql5.com/go?link=https://arxiv.org/abs/2307.03406 "https://arxiv.org/abs/2205.10484")
- [Neural networks made easy (Part 70): Closed-form policy improvement operators (CFPI)](https://www.mql5.com/en/articles/13982)

### Programs used in the article

| # | Issued to | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Example collection EA |
| 2 | ResearchRealORL.mq5 | Expert Advisor | EA for collecting examples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Policy training EA |
| 4 | StudyEncoder.mq5 | Expert Advisor | Autoencoder training EA |
| 5 | Test.mq5 | Expert Advisor | Model testing EA |
| 6 | Trajectory.mqh | Class library | System state description structure |
| 7 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 8 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14012](https://www.mql5.com/ru/articles/14012)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14012.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/14012/mql5.zip "Download MQL5.zip")(757.07 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)
- [Neural Networks in Trading: Memory Augmented Context-Aware Learning for Cryptocurrency Markets (Final Part)](https://www.mql5.com/en/articles/16993)

**[Go to discussion](https://www.mql5.com/en/forum/467797)**

![Population optimization algorithms: Evolution of Social Groups (ESG)](https://c.mql5.com/2/68/Population_optimization_algorithms_Evolution_of_Social_Groups_rESGw___LOGO.png)[Population optimization algorithms: Evolution of Social Groups (ESG)](https://www.mql5.com/en/articles/14136)

We will consider the principle of constructing multi-population algorithms. As an example of this type of algorithm, we will have a look at the new custom algorithm - Evolution of Social Groups (ESG). We will analyze the basic concepts, population interaction mechanisms and advantages of this algorithm, as well as examine its performance in optimization problems.

![Integrate Your Own LLM into EA (Part 3): Training Your Own LLM with CPU](https://c.mql5.com/2/79/Integrate_Your_Own_LLM_into_EA__Part_3_-_Training_Your_Own_LLM_with_CPU_____LOGO.png)[Integrate Your Own LLM into EA (Part 3): Training Your Own LLM with CPU](https://www.mql5.com/en/articles/13920)

With the rapid development of artificial intelligence today, language models (LLMs) are an important part of artificial intelligence, so we should think about how to integrate powerful LLMs into our algorithmic trading. For most people, it is difficult to fine-tune these powerful models according to their needs, deploy them locally, and then apply them to algorithmic trading. This series of articles will take a step-by-step approach to achieve this goal.

![Population optimization algorithms: Artificial Multi-Social Search Objects (MSO)](https://c.mql5.com/2/69/Population_optimization_algorithms___Artificial_Multi-Social_Search_Objects_dMSOb____LOGO.png)[Population optimization algorithms: Artificial Multi-Social Search Objects (MSO)](https://www.mql5.com/en/articles/14162)

This is a continuation of the previous article considering the idea of social groups. The article explores the evolution of social groups using movement and memory algorithms. The results will help to understand the evolution of social systems and apply them in optimization and search for solutions.

![Bill Williams Strategy with and without other indicators and predictions](https://c.mql5.com/2/79/Bill_Williams_Strategy_with_and_without_other_Indicators_and_Predictions__LOGO.png)[Bill Williams Strategy with and without other indicators and predictions](https://www.mql5.com/en/articles/14975)

In this article, we will take a look to one the famous strategies of Bill Williams, and discuss it, and try to improve the strategy with other indicators and with predictions.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14012&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062650201807693390)

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