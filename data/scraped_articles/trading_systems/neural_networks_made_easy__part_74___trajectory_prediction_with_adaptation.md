---
title: Neural networks made easy (Part 74): Trajectory prediction with adaptation
url: https://www.mql5.com/en/articles/14143
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T19:14:28.468679
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/14143&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070126193681829985)

MetaTrader 5 / Trading systems


### Introduction

Building a trading strategy is inseparable from analyzing the market situation and forecasting the most likely movement of a financial instrument. This movement often correlated with other financial assets and macroeconomic indicators. This can be compared with the movement of transport, where each vehicle follows its own individual destination. However, their actions on the road are interconnected to a certain extent and are strictly regulated by traffic rules. Also, due to the individual perception of the road situation by vehicle drivers, a share of stochasticity remains on the roads.

Similarly, in the world of finance, price formation is subject to certain rules. However, the stochasticity of supply and demand created by market participants leads to stochasticity in price. This may be why many trajectory forecasting methods used in the navigation field perform well in predicting future price movements.

In this article I want to introduce you to a method for effectively jointly predicting the trajectories of all agents on the scene with dynamic learning of weights _ADAPT_, which was proposed to solve problems in the field of navigation of autonomous vehicles. The method was first presented in the article " [ADAPT: Efficient Multi-Agent Trajectory Prediction with Adaptation](https://www.mql5.com/go?link=https://arxiv.org/abs/2307.14187 "https://arxiv.org/abs/2307.14187")".

### 1\. ADAPT algorithm

The _ADAPT_ method analyzes the past trajectories of all agents in the scene map and predicts their future trajectories. A vectorized scene representation models different types of interactions between agents and the map to obtain the best possible representation of the agents. Similar to goal setting approaches, the algorithm first predicts a possible set of endpoints. Each endpoint is then refined to account for the agent's displacement in the scene. After that the full trajectory determined at the end points is predicted.

The authors of the method stabilize model training by separating endpoint and trajectory prediction with gradient stopping. The model presented by the authors uses small multilayer perceptrons to predict endpoints and trajectories to keep model complexity low.

The authors' proposed method uses a vectorized representation to encode the map and agents in a structured manner. This representation creates a connected graph for each scene element independently, given the past trajectories of the agents and the scene map. The authors of the method propose to use two separate subgraphs for agents and map objects.

_ADAPT_ allows you to simulate various types of interactions between scene elements. The authors proposed to model four types of relationships: agent-to-lane ( _AL_), lane-to-lane ( _LL_), lane-to-agent ( _LA_) and agent-to-agent ( _AA_).

Interdependencies are analyzed using multi-headed attention blocks, similarly to [AutoBots](https://www.mql5.com/en/articles/14095). However, self-attention blocks ( _AA_, _LL_) are complemented with cross-relation blocks ( _AL_, _LA_) using a cross-attention encoder. Each interaction is modeled sequentially, and the process is repeated _L_ times.

In this way, intermediate features can be updated at each iteration, and then the updated features are used to calculate attention at the next iteration. Each scene element can be informed by different types of interactions _L_ times.

To predict the endpoint in the case of using an agent-centric representation, it is possible to use _MLP_, which may be preferable due to its advantages in single-agent prediction. But when using a scene-centric representation, it is recommended to use an adaptive head with dynamic weights, which is more effective in multi-agent prediction of trajectory end points.

After receiving the end point for each agent, the algorithm interpolates future coordinates between the start point and the end point using _MLP_. Here we "decouple" the endpoints to ensure that weight updates for full trajectory prediction are decoupled from the endpoint prediction. We similarly predict the probability for each trajectory using decoupled endpoints.

To train models, we predict _K_ trajectories and apply variety loss to capture multi-modal future scenarios. The error gradient is backpropagated only through the most accurate trajectory. Since we predict the full trajectories conditioned on the endpoints, the accuracy of endpoint prediction is essential for full trajectory prediction. Therefore, the authors of the method apply a separate loss function to improve endpoint prediction. The final element of the original loss function is the classification loss to guide the probabilities assigned to trajectories.

The original [visualization](https://www.mql5.com/go?link=https://kuis-ai.github.io/adapt/ "https://kuis-ai.github.io/adapt/") of the method presented by the paper authors is provided below.

![Authors' visualization of the method](https://c.mql5.com/2/65/pipelinev1w.png)

### 2\. Implementation using MQL5

Above is a fairly condensed theoretical description of the _ADAPT_ method, which is due to the large amount of work ahead and limitations of the article format. Some aspects will be discussed in more detail during our implementation of the proposed approaches. Please note that our implementation will differ in many ways from the original method. Here are the differences.

First, we will not use separate tensors for encoding agents and polylines. The agents in our case are the analyzed features. Each feature is characterized by 2 parameters: value and time. During the analyzed time period, it moves in a certain trajectory. Although each indicator has its own range of values, we actually do not have a map of the scene. However, we have a snapshot of the scene at a single point in time with all the agents in it. Technically, we can replace one entity with another. It seems there is no need to create a separate tensor for this, since this is a look at the same data in another dimension. Therefore, we will use one tensor with different accents.

#### 2.1 Cross-Relationship Block

Further, thinking about the way to implement the proposed approaches, I realized that I did not have the implementation of the cross-relationship block. Previously, our tasks were more autoregressive in nature. For such tasks, the use of a [self-attention](https://www.mql5.com/en/articles/8909) block was quite adequate. This time we need to analyze the relationship between various entities. So, we will implement a new neural layer CNeuronMH2AttentionOCL. The class implementation algorithms are largely borrowed from the self-attention block. The difference is that the Query, Key and Value entities will be formed from different dimensions of the source data tensor. This required substantial modifications. Therefore, I decided to create a new class rather than modernize the existing one.

```
class CNeuronMH2AttentionOCL       :  public CNeuronBaseOCL
  {
protected:
   uint              iHeads;                                      ///< Number of heads
   uint              iWindow;                                     ///< Input window size
   uint              iUnits;                                      ///< Number of units
   uint              iWindowKey;                                  ///< Size of Key/Query window
   //---
   CNeuronConvOCL    Q_Embedding;
   CNeuronConvOCL    KV_Embedding;
   CNeuronTransposeOCL Transpose;
   int               ScoreIndex;
   CNeuronBaseOCL    MHAttentionOut;
   CNeuronConvOCL    W0;
   CNeuronBaseOCL    AttentionOut;
   CNeuronConvOCL    FF[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      attentionOut(void);
   //---
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      AttentionInsideGradients(void);
public:
   /** Constructor */
                     CNeuronMH2AttentionOCL(void);
   /** Destructor */~CNeuronMH2AttentionOCL(void) {};
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint heads,
                          uint units_count, ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);
   //---
   virtual int       Type(void)   const   {  return defNeuronMH2AttentionOCL;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual CLayerDescription* GetLayerInfo(void);
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau);
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
```

In the class constructor, we only set initial values for local variables.

```
CNeuronMH2AttentionOCL::CNeuronMH2AttentionOCL(void)  :  iHeads(0),
                                                         iWindow(0),
                                                         iUnits(0),
                                                         iWindowKey(0)
  {
   activation = None;
  }
```

The class destructor remains empty.

Initialization of the _CNeuronMH2AttentionOCL_ class objects is implemented in the _Init_ method. At the beginning of the method, we call a relevant method of the parent class, in which the data received from the external program is checked and inherited objects are initialized.

```
bool CNeuronMH2AttentionOCL::Init(uint numOutputs, uint myIndex,
                                  COpenCLMy *open_cl, uint window,
                                  uint window_key, uint heads,
                                  uint units_count,
                                  ENUM_OPTIMIZATION optimization_type,
                                  uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count,
                                                       optimization_type, batch))
      return false;
```

We save the values of the main parameters.

```
   iWindow = fmax(window, 1);
   iWindowKey = fmax(window_key, 1);
   iUnits = fmax(units_count, 1);
   iHeads = fmax(heads, 1);
   activation = None;
```

Since we will be analyzing the source data in different dimensions, we will need to transpose the tensor of the source data.

```
   if(!Transpose.Init(0, 0, OpenCL, iUnits, iWindow, optimization_type, batch))
      return false;
   Transpose.SetActivationFunction(None);
```

To generate the _Query_, _Key_ and _Value_ entities we will use convolutional layers. The number of filters is equal to the dimension of the vector of one entity. _Query_ will be generated from one dimension of the original data tensor, while _Key_ and _Value_ will be generated from another. Therefore, we will create 2 layers (one for each dimension).

```
   if(!Q_Embedding.Init(0, 0, OpenCL, iWindow, iWindow, iWindowKey * iHeads, iUnits,
                                                                     optimization_type, batch))
      return false;
   Q_Embedding.SetActivationFunction(None);

   if(!KV_Embedding.Init(0, 0, OpenCL, iUnits, iUnits, 2 * iWindowKey * iHeads, iWindow,
                                                                     optimization_type, batch))
      return false;
   KV_Embedding.SetActivationFunction(None);
```

We only need the dependency coefficient matrix on the OpenCL context side. To save resources used, we create a buffer only in the context. On the side of the main program, only a pointer to the buffer is stored.

```
   ScoreIndex = OpenCL.AddBuffer(sizeof(float) * iUnits * iWindow * iHeads, CL_MEM_READ_WRITE);
   if(ScoreIndex == INVALID_HANDLE)
      return false;
```

Next come objects similar to the self-attention block. Here we create a multi-headed attention output layer.

```
//---
   if(!MHAttentionOut.Init(0, 0, OpenCL, iWindowKey * iUnits * iHeads, optimization_type, batch))
      return false;
   MHAttentionOut.SetActivationFunction(None);
```

Dimensionality reduction layer.

```
   if(!W0.Init(0, 0, OpenCL, iWindowKey * iHeads, iWindowKey * iHeads, iWindow, iUnits,
                                                                      optimization_type, batch))
      return false;
   W0.SetActivationFunction(None);
```

At the output of the attention block, we summarize the results obtained with the original data in a separate layer.

```
   if(!AttentionOut.Init(0, 0, OpenCL, iWindow * iUnits, optimization_type, batch))
      return false;
   AttentionOut.SetActivationFunction(None);
```

It is followed by a block of linear _MLPs_.

```
   if(!FF[0].Init(0, 0, OpenCL, iWindow, iWindow, 4 * iWindow, iUnits, optimization_type, batch))
      return false;
   if(!FF[1].Init(0, 0, OpenCL, 4 * iWindow, 4 * iWindow, iWindow, iUnits, optimization_type,
                                                                                          batch))
      return false;
   for(int i = 0; i < 2; i++)
      FF[i].SetActivationFunction(None);
```

In order to avoid unnecessary copying of error gradients from the buffer of the parent class to the buffer of the internal layer during the backpropagation pass, we will replace pointers to objects.

```
   Gradient.BufferFree();
   delete Gradient;
   Gradient = FF[1].getGradient();
//---
   return true;
  }
```

Moving on to the description of the feed-forward pass, please note that despite the large number of internal layers that implement certain functionality, we need to directly analyze the relationships. Although mathematically this functionality is completely identical to the self-attention block, we are faced with the fact that the number of _Query_ entities will most likely differ from the number of _Key_ and _Value_ entities, which results in a rectangular _Score_ matrix and violates the logic of previously created kernels. Therefore, we will create new kernels.

For the feed-forward pass, we create the _MH2AttentionOut_ kernel. The kernel will receive in parameters 4 pointers to data buffers and the vector dimension of one entity element. All our entities have the same size of elements.

```
__kernel void MH2AttentionOut(__global float *q,      ///<[in] Matrix of Querys
                              __global float *kv,     ///<[in] Matrix of Keys
                              __global float *score,  ///<[out] Matrix of Scores
                              __global float *out,    ///<[out] Matrix of Scores
                              int dimension           ///< Dimension of Key
                             )
  {
//--- init
   const int q_id = get_global_id(0);
   const int k = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int kunits = get_global_size(1);
   const int heads = get_global_size(2);
```

We will launch the kernel in a task space of as many as 3 dimensions for elements  _Query,_ _Key_ and attention heads. Moreover, all threads within one _Query_ element and one attention head will be combined into groups, which is due to the need to normalize the _Score_ matrix with the _SoftMax_ function within the specified groups.

In the kernel body, we first identify each thread and determine the offset in the global data buffers.

```
   const int shift_q = dimension * (q_id + qunits * h);
   const int shift_k = dimension * (k + kunits * h);
   const int shift_v = dimension * (k + kunits * (heads + h));
   const int shift_s = q_id * kunits * heads + h * kunits + k;
```

We also define other constants and declare a local array.

```
   const uint ls = min((uint)get_local_size(1), (uint)LOCAL_ARRAY_SIZE);
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
   __local float temp[LOCAL_ARRAY_SIZE];
```

After that we calculate the dependency coefficient matrix.

```
//--- sum of exp
   uint count = 0;
   if(k < ls)
      do
        {
         if((count * ls) < (kunits - k))
           {
            float sum = 0;
            for(int d = 0; d < dimension; d++)
               sum = q[shift_q + d] * kv[shift_k + d];
            sum = exp(sum / koef);
            if(isnan(sum))
               sum = 0;
            temp[k] = (count > 0 ? temp[k] : 0) + sum;
           }
         count++;
        }
      while((count * ls + k) < kunits);
   barrier(CLK_LOCAL_MEM_FENCE);
```

```
   count = min(ls, (uint)kunits);
//---
   do
     {
      count = (count + 1) / 2;
      if(k < ls)
         temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
      if(k + count < ls)
         temp[k + count] = 0;
      barrier(CLK_LOCAL_MEM_FENCE);
     }
   while(count > 1);
```

```
//--- score
   float sum = temp[0];
   float sc = 0;
   if(sum != 0)
     {
      for(int d = 0; d < dimension; d++)
         sc = q[shift_q + d] * kv[shift_k + d];
      sc = exp(sc / koef);
      if(isnan(sc))
         sc = 0;
     }
   score[shift_s] = sc;
   barrier(CLK_LOCAL_MEM_FENCE);
```

We also calculate new values of the _Query_ entity taking into account the dependence coefficients for each element of the vector separately.

```
//--- out
   for(int d = 0; d < dimension; d++)
     {
      uint count = 0;
      if(k < ls)
         do
           {
            if((count * ls) < (kunits - k))
              {
               float sum = q[shift_q + d] * kv[shift_v + d] *
                                (count == 0 ? sc : score[shift_s + count * ls]);
               if(isnan(sum))
                  sum = 0;
               temp[k] = (count > 0 ? temp[k] : 0) + sum;
              }
            count++;
           }
         while((count * ls + k) < kunits);
      barrier(CLK_LOCAL_MEM_FENCE);
      //---
      count = min(ls, (uint)kunits);
      do
        {
         count = (count + 1) / 2;
         if(k < ls)
            temp[k] += (k < count && (k + count) < kunits ? temp[k + count] : 0);
         if(k + count < ls)
            temp[k + count] = 0;
         barrier(CLK_LOCAL_MEM_FENCE);
        }
      while(count > 1);
      //---
      out[shift_q + d] = temp[0];
     }
  }
```

Next, we create a new kernel to implement the backpropagation functionality _MH2AttentionInsideGradients_. We will also run this kernel in a 3-dimensional task space.

In the kernel parameters, we pass 6 pointers to data buffers. These include error gradient buffers for all entities.

```
__kernel void MH2AttentionInsideGradients(__global float *q, __global float *q_g,
                                          __global float *kv, __global float *kv_g,
                                          __global float *scores,
                                          __global float *gradient,
                                          int kunits)
  {
//--- init
   const int q_id = get_global_id(0);
   const int d = get_global_id(1);
   const int h = get_global_id(2);
   const int qunits = get_global_size(0);
   const int dimension = get_global_size(1);
   const int heads = get_global_size(2);
```

In the kernel body, we, as always, identify the thread and create the necessary constants.

```
   const int shift_q = dimension * (q_id + qunits * h) + d;
   const int shift_k = dimension * (q_id + kunits * h) + d;
   const int shift_v = dimension * (q_id + kunits * (heads + h)) + d;
   const int shift_s = q_id * kunits * heads + h * kunits;
   const int shift_g = h * qunits * dimension + d;
   float koef = sqrt((float)dimension);
   if(koef < 1)
      koef = 1;
```

First we calculate the error gradients for the _Value_ entity. To do this, we simply multiply the vector of error gradients from the output of the attention block by the corresponding dependence coefficients.

```
//--- Calculating Value's gradients
   int step_score = q_id * kunits * heads;
   for(int v = q_id; v < kunits; v += qunits)
     {
      int shift_score = h * kunits + v;
      float grad = 0;
      for(int g = 0; g < qunits; g++)
         grad += gradient[shift_g + g * dimension] * scores[shift_score + g * step_score];
      kv_g[shift_v + v * dimension]=grad;
     }
```

We then calculate the error gradients for the _Query_ entity. This time we first need to calculate the error gradient on the elements of the dependence coefficient matrix, taking into account the derivative of the _SoftMax_ function. Then it should be multiplied by the corresponding element of the _Key_ tensor.

```
//--- Calculating Query's gradients
   float grad = 0;
   float out_g = gradient[shift_g + q_id * dimension];
   int shift_val = (heads + h) * kunits * dimension + d;
   int shift_key = h * kunits * dimension + d;
   for(int k = 0; k < kunits; k++)
     {
      float sc_g = 0;
      float sc = scores[shift_s + k];
      for(int v = 0; v < kunits; v++)
         sc_g += scores[shift_s + v] * out_g * kv[shift_val + v * dimension] *
                                                        ((float)(k == v) - sc);
      grad += sc_g * kv[shift_key + k * dimension];
     }
   q_g[shift_q] = grad / koef;
```

Similarly, we calculate the error gradient for the _Key_ entity. However, this time we calculate the error gradients of the dependence coefficients along the corresponding tensor column.

```
//--- Calculating Key's gradients
   for(int k = q_id; k < kunits; k += qunits)
     {
      int shift_score = h * kunits + k;
      int shift_val = (heads + h) * kunits * dimension + d;
      grad = 0;
      float val = kv[shift_v];
      for(int scr = 0; scr < qunits; scr++)
        {
         float sc_g = 0;
         int shift_sc = scr * kunits * heads;
         float sc = scores[shift_sc + k];
         for(int v = 0; v < kunits; v++)
            sc_g += scores[shift_sc + v] * gradient[shift_g + scr * dimension] * val *
                                                                ((float)(k == v) - sc);
         grad += sc_g * q[shift_q + scr * dimension];
        }
      kv_g[shift_k + k * dimension] = grad / koef;
     }
  }
```

After building the algorithm on the OpenCL context side, we return to our class to organize the process on the main program side. First, let's look at the feedForward method. Similar to the relevant methods for other neural layers, in the parameters we receive a pointer to the previous neural layer, which provides the source data.

```
bool CNeuronMH2AttentionOCL::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//---
   if(!Q_Embedding.FeedForward(NeuronOCL))
      return false;
```

However, we do not check the relevance of the received pointer. Instead, we call the feed-forward method of the _Q\_Embedding_ inner layer to create a tensor of _Query_ entities, passing the resulting pointer to it. In the body of the specified method, all the necessary controls are already implemented and we do not need to implement them again.

Next, we will generate the _Key_ and _Value_ entities. As mentioned earlier, for these we use a different dimension of the original data tensor. Therefore, we first transpose the source data matrix, and then call the feed-forward method of the corresponding inner layer.

```
   if(!Transpose.FeedForward(NeuronOCL) || !KV_Embedding.FeedForward(NeuronOCL))
      return false;
```

_MH2AttentionOut_ kernel calls will be implemented in a separate method _attentionOut_.

```
   if(!attentionOut())
      return false;
```

We compress the multi-head attention results tensor to the size of the original data.

```
   if(!W0.FeedForward(GetPointer(MHAttentionOut)))
      return false;
```

Then we add the obtained values to the original data and normalize them. The _SumAndNormilize_ method is inherited from the parent class.

```
//---
   if(!SumAndNormilize(W0.getOutput(), NeuronOCL.getOutput(), AttentionOut.getOutput(), iWindow))
      return false;
```

At the end of the attention block, we pass the data through MLP.

```
   if(!FF[0].FeedForward(GetPointer(AttentionOut)))
      return false;
   if(!FF[1].FeedForward(GetPointer(FF[0])))
      return false;
```

Add the values again and mormalize.

```
   if(!SumAndNormilize(FF[1].getOutput(), AttentionOut.getOutput(), Output, iWindow))
      return false;
//---
   return true;
  }
```

To complete the picture of the feed-forward algorithm, let's consider the attentionOut method. The method does not receive parameters and works only with internal class objects. Therefore, in the body of the method we only check the relevance of the pointer to the _OpenCL_ context.

```
bool CNeuronMH2AttentionOCL::attentionOut(void)
  {
   if(!OpenCL)
      return false;
```

Next, we'll create the task space and offset arrays. As discussed when building the kernel, we create a 3-dimensional problem space with a local group along the second dimension.

```
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iUnits, iWindow, iHeads};
   uint local_work_size[3] = {1, iWindow, 1};
```

We pass the necessary parameters to the kernel.

```
   ResetLastError();
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionOut, def_k_mh2ao_q,
                                                       Q_Embedding.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__,
                                                            GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionOut, def_k_mh2ao_kv,
                                                       KV_Embedding.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__,
                                                             GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionOut, def_k_mh2ao_score, ScoreIndex))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__,
                                                             GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionOut, def_k_mh2ao_out,
                                                       MHAttentionOut.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__,
                                                              GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgument(def_k_MH2AttentionOut, def_k_mh2ao_dimension, (int)iWindowKey))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__,
                                                              GetLastError(), __LINE__);
      return false;
     }
```

Then put the kernel in the execution queue.

```
   if(!OpenCL.Execute(def_k_MH2AttentionOut, 3, global_work_offset, global_work_size,
                                                                    local_work_size))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      return false;
     }
//---
   return true;
  }
```

We have implemented the feed-forward pass process on both the main program side and the _OpenCL_ context side. Next, we need to arrange the backpropagation process. To implement the algorithm on the _OpenCL_ context side, we have already created the _MH2AttentionInsideGradients_ kernel. Now we need to create the _AttentionInsideGradients_ method for calling this kernel. We will not pass anything in the parameters to the method, similar to the relevant feed-forward method.

```
bool CNeuronMH2AttentionOCL::AttentionInsideGradients(void)
  {
   if(!OpenCL)
      return false;
```

In the body of the method we check the relevance of the pointer to the _OpenCL_ context. After that, we create arrays indicating the dimension of the task space and the offsets in it.

```
   uint global_work_offset[3] = {0};
   uint global_work_size[3] = {iUnits, iWindowKey, iHeads};
```

Pass the parameters necessary to the kernel.

```
   ResetLastError();
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_q,
                                                            Q_Embedding.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                 __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_qg,
                                                            Q_Embedding.getGradientIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                 __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_kv,
                                                            KV_Embedding.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                  __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_kvg,
                                                           KV_Embedding.getGradientIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                  __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_score,
                                                                                ScoreIndex))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                  __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_MH2AttentionInsideGradients, def_k_mh2aig_outg,
                                                         MHAttentionOut.getGradientIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                  __LINE__);
      return false;
     }
   if(!OpenCL.SetArgument(def_k_MH2AttentionInsideGradients, def_k_mh2aig_kunits, (int)iWindow))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(),
                                                                                  __LINE__);
      return false;
     }
```

Put the kernel in the execution queue.

```
   if(!OpenCL.Execute(def_k_MH2AttentionInsideGradients, 3, global_work_offset,
                                                             global_work_size))
     {
      printf("Error of execution kernel %s: %d", __FUNCTION__, GetLastError());
      return false;
     }
//---
   return true;
  }
```

In general, this is a standard algorithm for such tasks. And the entire algorithm for distributing the error gradient inside our layer is described by the _calcInputGradients_ method. In the parameters, the method receives a pointer to the object of the previous layer to which the error gradient must be passed.

```
bool CNeuronMH2AttentionOCL::calcInputGradients(CNeuronBaseOCL *prevLayer)
  {
   if(!FF[1].calcInputGradients(GetPointer(FF[0])))
      return false;
```

In the body of the method, we alternately propagate the error gradient from the block output to the previous layer. As you remember, when initializing the class, we replaced the pointer with the error gradient buffer. And the subsequent layer wrote the error gradient directly to the last layer of the inner _MLP_. From there we will propagate the error gradient to the output level of the attention block.

```
   if(!FF[0].calcInputGradients(GetPointer(AttentionOut)))
      return false;
```

At this level, we added the results of the attention block to the initial data. Similarly, we collect a gradient from 2 directions.

```
   if(!SumAndNormilize(FF[1].getGradient(), AttentionOut.getGradient(), W0.getGradient(),
                                                                           iWindow, false))
      return false;
```

Next, we propagate the error gradient across the heads of attention.

```
   if(!W0.calcInputGradients(GetPointer(MHAttentionOut)))
      return false;
```

Propagate the error gradient to the entities.

```
   if(!AttentionInsideGradients())
      return false;
```

We propagate the error gradient from _Key_ and _Value_ to the transpose layer. In the feed-forward pass, we transposed the source data matrix. With the error gradient, we have to do the opposite operation.

```
   if(!KV_Embedding.calcInputGradients(GetPointer(Transpose)))
      return false;
```

Next we have to transfer the error gradient from all entities to the previous layer.

```
   if(!Q_Embedding.calcInputGradients(prevLayer))
      return false;
```

Please note here that the error gradient goes to the previous layer from 4 threads:

- Query
- Key
- Value
- Bypassing the attention block.

However, our inner layer methods, when passing the error gradient, delete previously recorded data. Therefore, having received the error gradient from _Query_, we add it to the error gradient at the output of the attention block in the inner layer buffer.

```
   if(!SumAndNormilize(prevLayer.getGradient(), W0.getGradient(), AttentionOut.getGradient(),
                                                                              iWindow, false))
      return false;
```

And after receiving data from _Key_ and _Value_, we add up all the threads.

```
   if(!Transpose.calcInputGradients(prevLayer))
      return false;
   if(!SumAndNormilize(prevLayer.getGradient(), AttentionOut.getGradient(),
                                                      prevLayer.getGradient(), iWindow, false))
      return false;
//---
   return true;
  }
```

The weight updating method is quite simple. We simply call the relevant methods on the inner layers.

```
bool CNeuronMH2AttentionOCL::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!Q_Embedding.UpdateInputWeights(NeuronOCL))
      return false;
   if(!KV_Embedding.UpdateInputWeights(GetPointer(Transpose)))
      return false;
   if(!W0.UpdateInputWeights(GetPointer(MHAttentionOut)))
      return false;
   if(!FF[0].UpdateInputWeights(GetPointer(AttentionOut)))
      return false;
   if(!FF[1].UpdateInputWeights(GetPointer(FF[0])))
      return false;
//---
   return true;
  }
```

This concludes our consideration of methods for organizing the cross-relationship process. You can find the complete code of the class and all its methods in the attachment. We are moving on to building Expert Advisors for training and testing the models.

#### 2.2 Model architecture

As can be seen from the theoretical description of the _ADAPT_ method, the proposed approach has a rather complex hierarchical structure. For us, this translates into a large number of trained models. We will divide the description of their architecture into 2 methods. First, we will create 3 models that are related to the endpoint prediction process.

```
bool CreateTrajNetDescriptions(CArrayObj *encoder, CArrayObj *endpoints, CArrayObj *probability)
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
   if(!endpoints)
     {
      endpoints = new CArrayObj();
      if(!endpoints)
         return false;
     }
   if(!probability)
     {
      probability = new CArrayObj();
      if(!probability)
         return false;
     }
```

The environmental state encoder receives raw input data describing 1 state.

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

As always, we normalize the received data.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = MathMax(1000, GPTBars);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

We also generate an embedding, which we add to the historical sequence accumulation buffer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronEmbeddingOCL;
     {
      int temp[] = {prev_count};
      ArrayCopy(descr.windows, temp);
     }
   prev_count = descr.count = GPTBars;
   int prev_wout = descr.window_out = EmbeddingSize;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Then we introduce positional coding.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronPEOCL;
   descr.count = prev_count;
   descr.window = prev_wout;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

This is followed by the blocks of comprehensive attention. For the convenience of managing the model architecture, we will create a loop based on the number of the block iterations.

```
   for(int l = 0; l < Lenc; l++)
     {
      //--- layer 4
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronTransposeOCL;
      descr.count = prev_count;
      descr.window = prev_wout;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
```

According to the algorithm proposed by the authors of the _ADAPT_ method, we first check the relationships between polylines (in our case, states) and agents. Before using our cross-relationship block in this direction, we need to transpose the resulting amount of information. Then we add our new layer.

```
      //--- layer 5
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMH2AttentionOCL;
      descr.count = prev_wout;
      descr.window = prev_count;
      descr.step = 8;
      descr.window_out = 16;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
```

Then comes the trajectory self-attention block.

```
      //--- layer 6
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMLMHAttentionOCL;
      descr.count = prev_wout;
      descr.window = prev_count;
      descr.step = 8;
      descr.window_out = 16;
      descr.layers = 1;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
```

Next, we analyze the relationship on a different plane. For this, we transpose the data and repeat the attention blocks.

```
      //--- layer 7
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronTransposeOCL;
      descr.count = prev_wout;
      descr.window = prev_count;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
```

```
      //--- layer 8
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMH2AttentionOCL;
      descr.count = prev_count;
      descr.window = prev_wout;
      descr.step = 8;
      descr.window_out = 16;
      descr.layers = 1;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
```

```
      //--- layer 9
      if(!(descr = new CLayerDescription()))
         return false;
      descr.type = defNeuronMLMHAttentionOCL;
      descr.count = prev_count;
      descr.window = prev_wout;
      descr.step = 8;
      descr.window_out = 16;
      descr.layers = 1;
      descr.optimization = ADAM;
      if(!encoder.Add(descr))
        {
         delete descr;
         return false;
        }
     }
```

As mentioned above, we wrapped the Encoder block into a loop. The number of loop iterations is provided in constants.

```
#define        Lenc                    3             //Number ADAPT Encoder blocks
```

Thus, changing one constant allows us to quickly change the number of attention blocks in the Encoder.

The Encoder results are used to predict multiple sets of endpoints. The number of such sets is determined by the NForecast constant.

```
#define        NForecast               5             //Number of forecast
```

We will use a simple _MLP_ for the endpoint prediction model. In this model, the data received from the Encoder passes through fully connected layers.

```
//--- Endpoints
   endpoints.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = (prev_count * prev_wout);
   descr.activation = None;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
```

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
```

The latent state is normalized by the _SoftMax_ function.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = LatentCount;
   descr.step = 1;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
```

Next, we generate endpoints in the fully connected layer.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 3 * NForecast;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!endpoints.Add(descr))
     {
      delete descr;
      return false;
     }
```

The model for predicting the probabilities of choosing trajectories also uses the results of the Encoder as input data.

```
//--- Probability
   probability.Clear();
//--- Input layer
   if(!probability.Add(endpoints.At(0)))
      return false;
```

But in it, they are analyzed taking into account predicted endpoints.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = 3 * NForecast;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
```

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
```

Operations with probabilistic quantities allow us to use the _SoftMax_ layer at the output of the model.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = NForecast;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
```

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronSoftMaxOCL;
   descr.count = NForecast;
   descr.step = 1;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!probability.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

Now we come to the point where we made fundamental changes to the _ADAPT_ method algorithm. Our changes are required by the specifics of the financial markets. However, in my opinion, they absolutely do not contradict the approaches proposed by the authors of the method.

The authors proposed their own algorithm for solving problems related to the navigation of autonomous vehicles. Here the quality of trajectory prediction is of great importance. Because a collision of 2 or more vehicles on any part of the trajectory can lead to critical consequences.

In the case of financial market trading, more attention is paid to control points. We are not that interested in the trajectory of the price movement and its small fluctuations in the range of the general trend. What's more important to us is the extremes of the maximum possible profits and drawdowns within the framework of this movement.

Therefore, we excluded the trajectory prediction block and replaced it with an Actor model, which will generate the parameters of the transaction. At the same time, we retained the general approach to training the models. We will get back to it a little later.

Our Actor uses 4 data sources to make a decision:

- State embedding
- Account status descriptions
- Predicted endpoint sets
- Probabilities of each predicted set of endpoints

Previously, we created a mechanism for combining only 2 streams of information. To combine 4 streams, we will build a cascade of models.

```
bool CreateDescriptions(CArrayObj *actor, CArrayObj *end_encoder, CArrayObj *state_encoder)
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
   if(!end_encoder)
     {
      end_encoder = new CArrayObj();
      if(!end_encoder)
         return false;
     }
   if(!state_encoder)
     {
      state_encoder = new CArrayObj();
      if(!state_encoder)
         return false;
     }
```

We combine sets of predicted endpoints and their probabilities into endpoint embedding.

```
//--- Endpoints Encoder
   end_encoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = 3 * NForecast;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!end_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = NForecast;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!end_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

We combine environmental state embedding with balance and open positions parameters.

```
//--- State Encoder
   state_encoder.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   prev_count = descr.count = GPTBars * EmbeddingSize;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!state_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = AccountDescr;
   descr.optimization = ADAM;
   descr.activation = SIGMOID;
   if(!state_encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

We pass the results of the work of the 2 specified models to the Actor for decision-making.

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
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = prev_count;
   descr.step = LatentCount;
   descr.optimization = ADAM;
   descr.activation = LReLU;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

Inside the Actor, we use fully connected layers.

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
```

We generate its stochastic behavior.

```
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
//---
   return true;
  }
```

As you can see, we plan to use the simplest possible model architectures. This is one of the advantages of the _ADAPT_ method.

In this article, I decided not to dwell on a detailed description of Expert Advisors for interaction with the environment. The structure of the collected data and the methods of interaction with the environment have not changed. Of course, changes have been made to the sequence of calling models for decision making. I suggest that you study the code to see the sequence. The full EA code can be found in the attachment. But the Model Training EA has several unique aspects.

#### 2.3 Model Training

Unlike the last few articles, this time we will train all models within one EA "...\\Experts\\ADAPT\\Study.mq5". This is because we need to transfer the error gradient from almost all models to the Environmental Encoder.

The EA initialization method is built according to a standard scheme. First we load the training dataset.

```
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

Then, in 2 stages, we load the previously created models and, if necessary, create new ones.

```
//--- load models
   float temp;
   if(!ADAPTEncoder.Load(FileName + "Enc.nnw", temp, temp, temp, dtStudied, true) ||
      !ADAPTEndpoints.Load(FileName + "Endp.nnw", temp, temp, temp, dtStudied, true) ||
      !ADAPTProbability.Load(FileName + "Prob.nnw", temp, temp, temp, dtStudied, true)
     )
     {
      CArrayObj *encoder = new CArrayObj();
      CArrayObj *endpoint = new CArrayObj();
      CArrayObj *prob = new CArrayObj();
      if(!CreateTrajNetDescriptions(encoder, endpoint, prob))
        {
         delete endpoint;
         delete prob;
         delete encoder;
         return INIT_FAILED;
        }
      if(!ADAPTEncoder.Create(encoder) ||
         !ADAPTEndpoints.Create(endpoint) ||
         !ADAPTProbability.Create(prob))
        {
         delete endpoint;
         delete prob;
         delete encoder;
         return INIT_FAILED;
        }
      delete endpoint;
      delete prob;
      delete encoder;
     }
```

```
   if(!StateEncoder.Load(FileName + "StEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !EndpointEncoder.Load(FileName + "EndEnc.nnw", temp, temp, temp, dtStudied, true) ||
      !Actor.Load(FileName + "Act.nnw", temp, temp, temp, dtStudied, true))
     {
      CArrayObj *actor = new CArrayObj();
      CArrayObj *endpoint = new CArrayObj();
      CArrayObj *encoder = new CArrayObj();
      if(!CreateDescriptions(actor, endpoint, encoder))
        {
         delete actor;
         delete endpoint;
         delete encoder;
         return INIT_FAILED;
        }
      if(!Actor.Create(actor) ||
         !StateEncoder.Create(encoder) ||
         !EndpointEncoder.Create(endpoint))
        {
         delete actor;
         delete endpoint;
         delete encoder;
         return INIT_FAILED;
        }
      delete actor;
      delete endpoint;
      delete encoder;
      //---
     }
```

We transfer all models into a single _OpenCL_ context.

```
   OpenCL = Actor.GetOpenCL();
   StateEncoder.SetOpenCL(OpenCL);
   EndpointEncoder.SetOpenCL(OpenCL);
   ADAPTEncoder.SetOpenCL(OpenCL);
   ADAPTEndpoints.SetOpenCL(OpenCL);
   ADAPTProbability.SetOpenCL(OpenCL);
```

Control the model architecture.

```
   Actor.getResults(Result);
   if(Result.Total() != NActions)
     {
      PrintFormat("The scope of the actor does not match the actions count (%d <> %d)",
                                                                NActions, Result.Total());
      return INIT_FAILED;
     }
//---
   ADAPTEndpoints.getResults(Result);
   if(Result.Total() != 3 * NForecast)
     {
      PrintFormat("The scope of the Endpoints does not match forecast endpoints (%d <> %d)",
                                                            3 * NForecast, Result.Total());
      return INIT_FAILED;
     }
//---
   ADAPTEncoder.GetLayerOutput(0, Result);
   if(Result.Total() != (HistoryBars * BarDescr))
     {
      PrintFormat("Input size of Encoder doesn't match state description (%d <> %d)",
                                                Result.Total(), (HistoryBars * BarDescr));
      return INIT_FAILED;
     }
```

Create an auxiliary buffer.

```
   if(!bGradient.BufferInit(MathMax(AccountDescr, NForecast), 0) ||
      !bGradient.BufferCreate(OpenCL))
     {
      PrintFormat("Error of create buffers: %d", GetLastError());
      return INIT_FAILED;
     }
```

Generate a custom event for the start of model training.

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

The training process itself is organized using the _Train_ method.

```
void Train(void)
  {
//---
   vector<float> probability = GetProbTrajectories(Buffer, 0.9);
```

In the body of the method, we first create a vector of probabilities for choosing trajectories from the experience replay buffer. Then we create the required local variables.

```
   vector<float> result, target;
   matrix<float> targets, temp_m;
   bool Stop = false;
//---
   uint ticks = GetTickCount();
```

Training, as usual, is implemented in a system of nested loops. In the body of the outer loop, we sample the trajectory and the packet of learning states on it.

```
   for(int iter = 0; (iter < Iterations && !IsStopped() && !Stop); iter ++)
     {
      int tr = SampleTrajectory(probability);
      int batch = GPTBars + 48;
      int state = (int)((MathRand() * MathRand() / MathPow(32767, 2)) *
                             (Buffer[tr].Total - 2 - PrecoderBars - batch));
      if(state <= 0)
        {
         iter--;
         continue;
        }
      ADAPTEncoder.Clear();
      int end = MathMin(state + batch, Buffer[tr].Total - PrecoderBars);
```

The process of training models on a sequence of historical data is built in the nested loop.

```
      for(int i = state; i < end; i++)
        {
         bState.AssignArray(Buffer[tr].States[i].state);
```

We take one environmental state and pass it to the Encoder.

```
         //--- Trajectory
         if(!ADAPTEncoder.feedForward((CBufferFloat*)GetPointer(bState), 1, false,
                                                              (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Then we generate sets of predicted endpoints and their probabilities.

```
         if(!ADAPTEndpoints.feedForward((CNet*)GetPointer(ADAPTEncoder), -1,
                                                             (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

```
         if(!ADAPTProbability.feedForward((CNet*)GetPointer(ADAPTEncoder), -1,
                                               (CNet*)GetPointer(ADAPTEndpoints)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Next, to organize the endpoint training process, we need to generate target values. We take subsequent states from the experience replay buffer to a given planning depth.

```
         targets = matrix<float>::Zeros(PrecoderBars, 3);
         for(int t = 0; t < PrecoderBars; t++)
           {
            target.Assign(Buffer[tr].States[i + 1 + t].state);
            if(target.Size() > BarDescr)
              {
               matrix<float> temp(1, target.Size());
               temp.Row(target, 0);
               temp.Reshape(target.Size() / BarDescr, BarDescr);
               temp.Resize(temp.Rows(), 3);
               target = temp.Row(temp.Rows() - 1);
              }
            targets.Row(target, t);
           }
```

But we do not use the last state in them, as one might think from the definition of endpoints. Instead, we look for the nearest extrema. First, we calculate the cumulative total of the deviation of the closing price of each candle from the analyzed state. And to the obtained values we add intervals up to _High_ and _Low_ of every bar. We save the calculation results in a matrix.

```
         target = targets.Col(0).CumSum();
         targets.Col(target, 0);
         targets.Col(target + targets.Col(1), 1);
         targets.Col(target + targets.Col(2), 2);
```

In the resulting matrix we find the nearest extremum.

```
         int extr = 1;
         if(target[0] == 0)
            target[0] = target[1];
         int direct = (target[0] > 0 ? 1 : -1);
         for(int i = 1; i < PrecoderBars; i++)
           {
            if((target[i]*direct) < 0)
               break;
            extr++;
           }
```

Form a vector from the found nearest extrema.

```
         targets.Resize(extr, 3);
         if(direct >= 0)
           {
            target = targets.Max(AXIS_HORZ);
            target[2] = targets.Col(2).Min();
           }
         else
           {
            target = targets.Min(AXIS_HORZ);
            target[1] = targets.Col(1).Max();
           }
```

Among the sets of predicted endpoints, we determine the vector with the minimum deviation and replace it with target values.

```
         ADAPTEndpoints.getResults(result);
         targets.Reshape(1, result.Size());
         targets.Row(result, 0);
         targets.Reshape(NForecast, 3);
         temp_m = targets;
         for(int i = 0; i < 3; i++)
            temp_m.Col(temp_m.Col(i) - target[i], i);
         temp_m = MathPow(temp_m, 2.0f);
         ulong pos = temp_m.Sum(AXIS_VERT).ArgMin();
         targets.Row(target, pos);
```

We use the resulting matrix to train a model for predicting target points.

```
         Result.AssignArray(targets);
         //---
         if(!ADAPTEndpoints.backProp(Result, (CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

We propagate the error gradient to the Encoder model and update its parameters.

```
         if(!ADAPTEncoder.backPropGradient((CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Here we also train a model for predicting trajectory probabilities. But its error gradients are not propagated to other models.

```
         bProbs.AssignArray(vector<float>::Zeros(NForecast));
         bProbs.Update((int)pos, 1);
         bProbs.BufferWrite();
         if(!ADAPTProbability.backProp(GetPointer(bProbs), GetPointer(ADAPTEndpoints)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

After updating the parameters of the endpoint prediction models, we move on to training our Actor's policy. To execute our Actor's feed-forward operations Actor at this stage, we only need a tensor to describe the state of the account and open positions. Let's form this tensor.

```
         //--- Policy
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

Next, we sequentially call the feed-forward methods of our cascade of Actor models.

```
         //--- State embedding
         if(!StateEncoder.feedForward((CNet *)GetPointer(ADAPTEncoder), -1,
                                      (CBufferFloat*)GetPointer(bAccount)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

It should be noted here that instead of predictive values of endpoint sets and their probabilities, we use tensors of target values, which we used above to train the corresponding models.

```
         //--- Endpoint embedding
         if(!EndpointEncoder.feedForward(Result, -1, false, (CBufferFloat*)GetPointer(bProbs)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

```
         //--- Actor
         if(!Actor.feedForward((CNet *)GetPointer(StateEncoder), -1,
                                                          (CNet*)GetPointer(EndpointEncoder)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

After the feed-forward pass, we need to update model parameters. For this we need target values. According to the _ADAPT_ method a model should be trained to predict trajectories on real data from the experience replay buffer. We could, as before, take Agent actions from the experience replay buffer. But in this case, we do not have a mechanism for assessing and prioritizing such actions.

In this situation, I decided to take a different approach. Since we already have target endpoint values based on real data of subsequent price movements from the training dataset, why don't we use them to generate the "optimal" trade under the analyzed conditions. We determine the direction and trading levels of the "optimal" trade. We take the position volume taking into account the risk of 1% of Equity per trade.

```
         result = vector<float>::Zeros(NActions);
         double value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS);
         double risk = AccountInfoDouble(ACCOUNT_EQUITY) * 0.01;
         if(direct > 0)
           {
            float tp = float(target[1] / _Point / MaxTP);
            result[1] = tp;
            int sl = int(MathMax(MathMax(target[1] / 3, -target[2]) / _Point, MaxSL/10));
            result[2] = float(sl) / MaxSL;
            result[0] = float(MathMax(risk / (value * sl), 0.01))+FLT_EPSILON;
           }
         else
           {
            float tp = float((-target[2]) / _Point / MaxTP);
            result[4] = tp;
            int sl = int(MathMax(MathMax((-target[2]) / 3, target[1]) / _Point, MaxSL/10));
            result[5] = float(sl) / MaxSL;
            result[3] = float(MathMax(risk / (value * sl), 0.01))+FLT_EPSILON;
           }
```

When calculating the position volume, we use Equity, since at the time of the trade the account may already have open positions, the profit (loss) of which is not taken into account in the Account Balance.

The "optimal" position generated in this way is used to train Actor models.

```
         Result.AssignArray(result);
         if(!Actor.backProp(Result, (CNet *)GetPointer(EndpointEncoder)) ||
            !StateEncoder.backPropGradient(GetPointer(bAccount),
                                  (CBufferFloat *)GetPointer(bGradient)) ||
            !EndpointEncoder.backPropGradient(GetPointer(bProbs),
                                  (CBufferFloat *)GetPointer(bGradient))
           )
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

We use the error gradient from Actor model training to update the Encoder parameters.

```
         if(!ADAPTEncoder.backPropGradient((CBufferFloat*)NULL))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            Stop = true;
            break;
           }
```

Please note that we are not updating the endpoint prediction model parameters at this stage. This limitation was introduced by the authors of the _ADAPT_ method and is designed to increase the model training stability.

After updating the parameters of all models, all we need to do is inform the user about the progress of the training process and move on to the next iteration of the loop system.

```
         //---
         if(GetTickCount() - ticks > 500)
           {
            double percent = (double(i - state) / ((end - state)) + iter) * 100.0 / (Iterations);
            string str = StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Actor", percent,
                                                                  Actor.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Endpoints", percent,
                                                         ADAPTEndpoints.getRecentAverageError());
            str += StringFormat("%-14s %6.2f%% -> Error %15.8f\n", "Probability", percent,
                                                       ADAPTProbability.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
```

At the end of the method, we clear the comments field on the chart. Log the model training results into the journal. Then initiate the EA termination.

```
   Comment("");
//---
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__,
                                                       "Actor", Actor.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__,
                                          "Endpoints", ADAPTEndpoints.getRecentAverageError());
   PrintFormat("%s -> %d -> %-15s %10.7f", __FUNCTION__, __LINE__,
                                      "Probability", ADAPTProbability.getRecentAverageError());
   ExpertRemove();
//---
  }
```

This concludes the description of the MQL5 implementation of our vision of the algorithm. You can find the complete code of all programs used in the article in the attachment.

### 3\. Testing

We have done quite a lot of work to implement the _ADAPT_ method using MQL5. Our implementation is far from the original algorithm. Nevertheless, it is in the spirit of the proposed approaches and exploits the original idea related to the comprehensive analysis of the relationships between the objects of the analyzed scene. Now it's time to test the results of our work on real historical data in the strategy tester.

The models were trained using historical data of EURUSD, H1, for the first 7 months of 2023. All indicators are used with default parameters.

The trained models were tested in full compliance with the training parameters. We only changed the time interval of the historical data. At this stage we used historical data from August 2023.

Since the structure of the data collected in the process of interaction with the environment has not changed, I did not collect new training data in my experiment. To train the models, I use the passes collected when training previous models. Moreover, the proposed approach to calculating the "optimal trade" allows us to avoid the calculation of additional passes that refine and supplement the training data space.

Here it may seem that one pass is enough to train the model. However, during the training process, we need to provide as much diverse information as possible to the model, including information about the state of the account and open positions.

Based on the results of the tests, we can make a conclusion about the effectiveness of the considered method. The simplicity of the models allows faster training of the models. The effectiveness of the proposed approaches is confirmed by the results of the trained model, which showed the capability to generate profits on both the training and test datasets.

### Conclusion

The _ADAPT_ method discussed in this article is an innovative approach to predicting agent trajectories in various complex scenarios. This approach is efficient, requires a small amount of computing resources and provides high quality predictions for each agent in the scene.

Improvements made to the _ADAPT_ method include an adaptive head that increases the capacity of the model without increasing its size, and the use of dynamic learning of weights to better adapt to each agent's individual situations. These innovations greatly contribute to effective trajectory prediction.

In the practical part of the article, we implemented our vision of the proposed approaches using MQL5. We trained and tested models using real historical data. Based on the results obtained, we can make a conclusion about the effectiveness of the _ADAPT_ method and the possibility of using its variations to build a model and operate it in financial markets.

However, I would like to remind you that any programs presented in the article are only intended to demonstrate the technology and are not ready for use in real-world financial trading.

### References

[ADAPT: Efficient Multi-Agent Trajectory Prediction with Adaptation](https://www.mql5.com/go?link=https://arxiv.org/abs/2307.14187 "https://arxiv.org/abs/2205.10484")
[Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

### Programs used in the article

| # | Issued to | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Example collection EA |
| 2 | ResearchRealORL.mq5 | Expert Advisor | EA for collecting examples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Model training EA |
| 4 | Test.mq5 | Expert Advisor | Model testing EA |
| 5 | Trajectory.mqh | Class library | System state description structure |
| 6 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 7 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14143](https://www.mql5.com/ru/articles/14143)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14143.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/14143/mql5.zip "Download MQL5.zip")(3615.14 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/468570)**
(1)


![](https://c.mql5.com/avatar/avatar_na2.png)

**\[Deleted\]**
\|
29 Jan 2024 at 10:13

Dmitry, can you give model testing more attention? Maybe in separate articles. The material is interesting, but it is impossible to draw any conclusions from the given tests. It is also difficult to reproduce (especially for those who don't have GPU or have a macbook at all).


![Neural networks made easy (Part 75): Improving the performance of trajectory prediction models](https://c.mql5.com/2/68/Neural_Networks_Made_Easy_dPart_751_Improving_the_Performance_of_Trajectory_Prediction_Models____LOG.png)[Neural networks made easy (Part 75): Improving the performance of trajectory prediction models](https://www.mql5.com/en/articles/14187)

The models we create are becoming larger and more complex. This increases the costs of not only their training as well as operation. However, the time required to make a decision is often critical. In this regard, let us consider methods for optimizing model performance without loss of quality.

![Developing a multi-currency Expert Advisor (Part 3): Architecture revision](https://c.mql5.com/2/70/Developing_a_multi-currency_advisor_0Part_1g___LOGO__3.png)[Developing a multi-currency Expert Advisor (Part 3): Architecture revision](https://www.mql5.com/en/articles/14148)

We have already made some progress in developing a multi-currency EA with several strategies working in parallel. Considering the accumulated experience, let's review the architecture of our solution and try to improve it before we go too far ahead.

![Neural networks made easy (Part 76): Exploring diverse interaction patterns with Multi-future Transformer](https://c.mql5.com/2/69/Neural_networks_made_easy_zPart_765_Exploring_various_modes_of_interaction_Multi-future_Transformer_.png)[Neural networks made easy (Part 76): Exploring diverse interaction patterns with Multi-future Transformer](https://www.mql5.com/en/articles/14226)

This article continues the topic of predicting the upcoming price movement. I invite you to get acquainted with the Multi-future Transformer architecture. Its main idea is to decompose the multimodal distribution of the future into several unimodal distributions, which allows you to effectively simulate various models of interaction between agents on the scene.

![MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library](https://c.mql5.com/2/80/MQL5_Trading_Toolkit_Part_1___LOGO.png)[MQL5 Trading Toolkit (Part 1): Developing A Positions Management EX5 Library](https://www.mql5.com/en/articles/14822)

Learn how to create a developer's toolkit for managing various position operations with MQL5. In this article, I will demonstrate how to create a library of functions (ex5) that will perform simple to advanced position management operations, including automatic handling and reporting of the different errors that arise when dealing with position management tasks with MQL5.

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/14143&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070126193681829985)

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