---
title: Neural networks made easy (Part 46): Goal-conditioned reinforcement learning (GCRL)
url: https://www.mql5.com/en/articles/12816
categories: Trading Systems, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:43:46.078307
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/12816&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5062673068213577406)

MetaTrader 5 / Trading systems


### Introduction

"Goal-conditioned reinforcement learning" sounds a little unusual or even strange. After all, the basic principle of reinforcement learning is aimed at maximizing the total reward during the interaction of the agent with the environment. But in this context, we are looking at achieving a specific goal at a specific stage or within a specific scenario.

We have already discussed the benefits of breaking down an overall goal into subtasks and explored methods for teaching an agent different skills that contribute to achieving the overall outcome. In this article, I propose to look at this problem from a different angle. Namely, we should train an agent to independently choose a strategy and skill to achieve a specific subtask.

### 1\. GCRL features

Goal-conditioned reinforcement learning (GCRL) is a set of complex reinforcement learning problems. We train the agent to achieve different goals in certain scenarios. Previously, we trained the agent to choose one action or another depending on the current state of the environment. In case of GCRL, we want to train the agent in such a way that its action is determined not only by the current state, but also by a specific subtask at this stage. In other words, in addition to the vector describing the current state, we should somehow indicate to the agent a subtask to achieve at each specific moment. It is very similar to the task of training skills, when we indicated a skill to the agent at each moment of time. After all, indicating to use the “open a position” skill or “open a position” task seems like a play on words. But behind these words lie differences in the approaches to training agents.

In reinforcement learning, the bottleneck is always the reward function. Just like in conventional reinforcement training, a single objective reward function is used in skill training tasks. Indicating the skill to use should complement the state of the environment and help the agent navigate it.

When using GCRL approaches, we introduce specific subtasks. Their achievement should be reflected in the reward received by the agent. It is similar to the internal reward of a discriminator, but is based on clear measurable indicators aimed at achieving a specific goal (solving a subtask).

To understand this fine line, let's look at an example of opening a position in both approaches. When training skills, we passed the current state of the environment and the vector of the account state with missing open positions to the scheduler. This allowed the scheduler to determine the skill description vector to be passed on to the agent to make a decision. As you remember, we used a change in the account balance as a reward. It is worth noting that we apply the same reward throughout the agent's training. Moreover, opening a position does not immediately affect the balance change. The exception is possible commissions for opening a position. But in general, we receive a reward with a delay to open a position.

In case of GCRL, we introduce an additional reward for achieving a specific subtask along with the global goal reward. For example, we can introduce some reward for opening a position or, conversely, impose fines until the agent opens a position. Here we need to take a balanced approach to the formation of such a reward. It should not exceed the possible profits and losses from the trading operation itself. Otherwise, the agent will simply open positions and "gain points", while the account balance will tend to 0.

Besides, the reward should depend on the task at hand. We will reward for opening a position and penalize for the absence of such an action only when setting the "opening a position" task. When searching for an exit point from a position, we, on the contrary, can introduce a penalty for an additional open position, as well as for holding a position for a long time.

When forming a vector for describing the task at hand for GCRL, it is important to take into account certain requirements. The vector should explicitly indicate the subtask the agent should achieve at a specific point in time.

The task description vector can include various elements, depending on the context and specifics of the task. For example, in case of opening a position, the description vector may contain information about the target asset, trading volume, price limits, or other parameters associated with opening a position. These elements should be clear and understandable for the agent so that he can correctly interpret the given subtask.

In addition, the task description vector should be sufficiently informative so that the agent can make decisions that are maximally focused on achieving this subtask. This may require the inclusion of additional data or contextual information that will help the agent more accurately understand how to act for achieving the goal.

There should be a pronounced logical, but not mathematical, relationship between the subtask description vector and the desired result. We can use a regular one-hot vector. Each element of the vector will correspond to a separate subtask. The vector will be passed to the agent along with a description of the current state of the environment. The main thing is that the agent can clearly interpret the subtask and build its internal connections between the subtask and the reward. In this regard, we should pay attention to reward. The additional reward introduced should be matched to a specific subtask.

But there are other approaches to forming a subtask description vector. If a combination of many factors is required to describe a separate subtask, we can use a separate model to form such a vector by analogy with methods for training skills. Such a model can be trained using various auto encoders or any other available method.

As you can see, both approaches are quite powerful and allow us to solve different problems. However, each of them has its shortcomings. It is no coincidence that various synergies between the two approaches appear, which makes it possible to build an even more stable algorithm. Indeed, while training skills, we built dependencies between the current state of the environment and the agent skill (action policy). Using additional tools aimed at achieving a specific subtask will help adjust the agent strategy to obtain the optimal result.

One such approach is adaptive variational GCRL (aVGCRL). The idea is that in a stochastic environment, the distribution of each skill representation will not be uniform. Moreover, it may change depending on the state of the environment. In certain states, there will be a dependence with some skills for which the dispersion of the distribution will be minimal. At the same time, the likelihood of using other skills in the same states will not be so clear and their distribution dispersion will be significantly higher. In other environmental states, the variance of skill distributions is likely to be dramatically different. This effect can be observed if we look at the latent representation of the variances of the variational auto encoder we used in the previous [article](https://www.mql5.com/en/articles/12783) to train the scheduler. A logical solution would be to focus on explicit dependencies. The authors of the aVGCRL method propose dividing the deviation error for each skill from the target value by the dispersion of the distribution. Obviously, the smaller the variance, the greater the influence of the error and the more the corresponding weighting coefficients change during the training process. At the same time, the randomness of other skills does not introduce a significant imbalance into the general model.

### 2\. Implementation using MQL5

Let's move on to the GCRL method implementation to get acquainted with it even better. We will create a kind of symbiosis of the two considered methods, although we will combine everything into a single model.

In the previous article, we created 2 models: a scheduler in the form of a variational auto encoder and an agent. Unlike previous approaches, the agent received only the latent state of the autoe nctoder, which, according to our logic, should have contained all the necessary information. The [test](https://www.mql5.com/en/articles/12783#para4) showed that training the agent to achieve the state predicted by the auto encoder did not provide the desired result. This may be due to the insufficient quality of forecast conditions.

At the same time, the use of classical approaches to reward made it possible to improve the agent training process using a previously trained scheduler.

In this work, we decided to abandon separate training of the variational auto encoder and included its encoder directly in the Agent model. It should be said that this approach somewhat violates the principles of training an auto encoder. After all, the main idea of using any auto encoder is data compression without reference to a specific task. But now we are not faced with the task of training an encoder to solve several problems from the same source data.

Besides, we only supply the current state of the environment to the encoder input. In our case, these are historical data on the movement of the instrument price and parameters of the analyzed indicators. In other words, we exclude information about the account status. We assume that the scheduler (in this case, the encoder) will form the skill to be used based on historical data. This can be a policy of working in a rising, falling or flat market.

Based on information about the account status, we will create a subtask for the Agent to search for an entry or exit point.

Dividing the model into Scheduler and Agent is absolutely arbitrary. After all, we will form one model. However, as mentioned above, we supply only historical data to the encoder input. This means that we have to add information about the assigned subtask to the middle of the model. We have not done this before. This is not a completely new solution. We have encountered this before. In such cases, we created 2 models.

The first part was solved by one model, then we combined the output of the first model with new data and fed it into the input of the second model. This solution is easier to arrange, but it has one significant drawback. It causes redundant communication between the main program and the OpenCL context. We have to get the results of the first model from the context and reload them for the second model. The same goes to the error gradient during the reverse pass. Using a single model eliminates these operations. But the question arises of adding new information at a separate stage of the model operation.

To solve this problem, we will create a new type of neural layer CNeuronConcatenate. As before, we begin working on each new neural layer class by creating the necessary kernels in the OpenCL program. First we created the Concat\_FeedForward forward pass kernel. All kernels were created on the basis of similar kernels of the base fully connected neural layer. The main difference is the addition of additional buffers and parameters for the second stream of information.

In the Concat\_FeedForward kernel parameters, we see a single weight matrix, 2 source data tensors, a vector of results and 3 numeric parameters (sizes of source data tensors and activation function ID)

```
__kernel void Concat_FeedForward(__global float *matrix_w,
                                 __global float *matrix_i1,
                                 __global float *matrix_i2,
                                 __global float *matrix_o,
                                 int inputs1,
                                 int inputs2,
                                 int activation
                                )
```

As before, we will launch the kernel in a one-dimensional task space based on the number of neurons in our layer, which is identical to the size of the results buffer. In the kernel body, we define the thread ID and declare the necessary local variables. Here we determine the offset in the weight coefficients buffer. Please note that for each neuron at the output of the layer we define the number of weights equal to the total size of 2 source data buffers and 1 Bayesian bias neuron.

```
  {
   int i = get_global_id(0);
   float sum = 0;
   float4 inp, weight;
   int shift = (inputs1 + inputs2 + 1) * i;
```

Next, we arrange a cycle for calculating the weighted sum of 1 source data buffer. This process is completely identical to that in the kernel of a fully connected neural layer.

```
   for(int k = 0; k < inputs1; k += 4)
     {
      switch(inputs1 - k)
        {
         case 1:
            inp = (float4)(matrix_i1[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i1[k], matrix_i1[k + 1], matrix_i1[k + 2], matrix_i1[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
```

After completing the loop iterations, we adjust the bias in the weight matrix by the size of 1 source data buffer. Besides, we create a similar cycle for 2 source data buffers.

```
   shift += inputs1;
   for(int k = 0; k < inputs2; k += 4)
     {
      switch(inputs2 - k)
        {
         case 1:
            inp = (float4)(matrix_i2[k], 0, 0, 0);
            weight = (float4)(matrix_w[shift + k], 0, 0, 0);
            break;
         case 2:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], 0, 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], 0, 0);
            break;
         case 3:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2], 0);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], 0);
            break;
         default:
            inp = (float4)(matrix_i2[k], matrix_i2[k + 1], matrix_i2[k + 2], matrix_i2[k + 3]);
            weight = (float4)(matrix_w[shift + k], matrix_w[shift + k + 1], matrix_w[shift + k + 2], matrix_w[shift + k + 3]);
            break;
        }
      float d = dot(inp, weight);
      if(isnan(sum + d))
         continue;
      sum += d;
     }
```

At the end of the kernel, we add a Bayesian bias element and activate the resulting sum. Then we save the resulting value in the corresponding element of the result buffer.

```
   sum += matrix_w[shift + inputs2];
//---
   if(isnan(sum))
      sum = 0;
   switch(activation)
     {
      case 0:
         sum = tanh(sum);
         break;
      case 1:
         sum = 1 / (1 + exp(-sum));
         break;
      case 2:
         if(sum < 0)
            sum *= 0.01f;
         break;
      default:
         break;
     }
   matrix_o[i] = sum;
  }
```

Exactly the same approach was used when modifying the backpass kernels and updating the weight matrix. You can familiarize yourself with them in NeuroNet\_DNG\\NeuroNet.cl (added to the article).

After creating the kernels, we move on to working on the code for the CNeuronConcatenate class in the main program. The set of class methods is quite standard:

- CNeuronConcatenate constructor and ~CNeuronConcatenate destructor
- initializing the Init neural layer
- feedForward forward pass
- calcHiddenGradients error gradient distribution
- updating updateInputWeights weight matrix
- Type object identification
- working with Save and Load files.

```
class CNeuronConcatenate   :  public CNeuronBaseOCL
  {
protected:
   int               i_SecondInputs;
   CBufferFloat     *ConcWeights;
   CBufferFloat     *ConcDeltaWeights;
   CBufferFloat     *ConcFirstMomentum;
   CBufferFloat     *ConcSecondMomentum;

public:
                     CNeuronConcatenate(void);
                    ~CNeuronConcatenate(void);
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons,
                          uint inputs1, uint inputs2, ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   virtual bool      calcHiddenGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput, CBufferFloat *SecondGradient);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput);
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual int       Type(void)        const                      {  return defNeuronConcatenate; }
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
```

Additionally, in the class, we declare one variable to record the size of additional source data and 4 data buffers: weight and moment matrices for various methods for optimizing weight coefficients. The new buffers will be used to arrange the communication process with the previous neural layer and new source data. Data transfer to the subsequent neural layer is arranged by means of the parent class of the fully connected CNeuronBaseOCL neural layer.

We initialize the data buffers in the class constructor.

```
CNeuronConcatenate::CNeuronConcatenate(void) : i_SecondInputs(0)
  {
   ConcWeights = new CBufferFloat();
   ConcDeltaWeights = new CBufferFloat();
   ConcFirstMomentum = new CBufferFloat();
   ConcSecondMomentum = new CBufferFloat;
  }
```

In the class destructor, we clear data and delete objects.

```
CNeuronConcatenate::~CNeuronConcatenate()
  {
   if(!!ConcWeights)
      delete ConcWeights;
   if(!!ConcDeltaWeights)
      delete ConcDeltaWeights;
   if(!!ConcFirstMomentum)
      delete ConcFirstMomentum;
   if(!!ConcSecondMomentum)
      delete ConcSecondMomentum;
  }
```

The indication of the size of all necessary data buffers is arranged in the Init object initialization method. The method receives the necessary initial data in the parameters:

- numOutputs — number of neurons in the next layer
- open\_cl  —  pointer to the OpenCL context handling object
- numNeurons  —  number of neurons in the current layer
- numInputs1  —  number of elements in the previous layer
- numInputs2  —  number of elements in the additional source data buffer
- optimization\_type  — parameter optimization method ID.

```
bool CNeuronConcatenate::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl, uint numNeurons,
                              uint numInputs1, uint numInputs2, ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, numNeurons, optimization_type, batch))
      return false;
```

In the body of the method, instead of a control block, we call a similar method of the parent class and check the result of the operations. The parent class already implements the basic controls, so we do not need to repeat them. In addition, the method of the parent class implements the initialization of all inherited objects and variables. Therefore, we only have to arrange the process of initializing the added objects in the body of this method.

First, we will create and initialize a matrix of weighting coefficients with random values to arrange data exchange with the previous neural layer. Please note that the size of the weight matrix is set sufficient to arrange work with the previous layer and the additional source data buffer. This is exactly the approach we envisioned when creating the forward pass kernel. Now we adhere to it when creating class methods on the side of the main program.

```
   i_SecondInputs = (int)numInputs2;
   if(!ConcWeights)
     {
      ConcWeights = new CBufferFloat();
      if(!ConcWeights)
         return false;
     }
   int count = (int)((numInputs1 + numInputs2 + 1) * numNeurons);
   if(!ConcWeights.Reserve(count))
      return false;
   float k = (float)(1.0 / sqrt(numNeurons + 1.0));
   for(int i = 0; i < count; i++)
     {
      if(!ConcWeights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
         return false;
     }
   if(!ConcWeights.BufferCreate(OpenCL))
      return false;
```

Next, depending on the weight coefficient update method specified in the parameters, we initialize the moment buffers. As you might remember, we use one moment buffer for SGD. In case of using the Adam method, 2 moment buffers will be initialized. We delete unused objects, which will allow us to use available resources more efficiently.

```
   if(optimization == SGD)
     {
      if(!ConcDeltaWeights)
        {
         ConcDeltaWeights = new CBufferFloat();
         if(!ConcDeltaWeights)
            return false;
        }
      if(!ConcDeltaWeights.BufferInit(count, 0))
         return false;
      if(!ConcDeltaWeights.BufferCreate(OpenCL))
         return false;
      if(!!ConcFirstMomentum)
         delete ConcFirstMomentum;
      if(!!ConcSecondMomentum)
         delete ConcSecondMomentum;
     }
   else
     {
      if(!!ConcDeltaWeights)
         delete ConcDeltaWeights;
      //---
      if(!ConcFirstMomentum)
        {
         ConcFirstMomentum = new CBufferFloat();
         if(CheckPointer(ConcFirstMomentum) == POINTER_INVALID)
            return false;
        }
      if(!ConcFirstMomentum.BufferInit(count, 0))
         return false;
      if(!ConcFirstMomentum.BufferCreate(OpenCL))
         return false;
      //---
      if(!ConcSecondMomentum)
        {
         ConcSecondMomentum = new CBufferFloat();
         if(!ConcSecondMomentum)
            return false;
        }
      if(!ConcSecondMomentum.BufferInit(count, 0))
         return false;
      if(!ConcSecondMomentum.BufferCreate(OpenCL))
         return false;
     }
//---
   return true;
  }
```

We finish working with class initialization methods and move on to organizing the main functionality. First we will create the feedForward pass method. Unlike the direct pass methods of all previously considered classes, this method receives 2 pointers to objects in its parameters: the previous neural layer and an additional source data buffer. There is nothing surprising here, because this is the main distinguishing feature of the class being created. But this approach requires additional work on the side of the main program outside the created class. We will talk about this a little later.

```
bool CNeuronConcatenate::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!OpenCL || !NeuronOCL || !SecondInput)
      return false;
```

In the body of the method, we first check the relevance of the received pointers. Besides, we will check the presence of the pointer to an object for working with the OpenCL context. If at least one pointer is missing, we terminate the method with a negative result.

Next, we check the size of the additional data buffer. It should contain a sufficient number of elements. Please note that we can specify a larger buffer size. But during the work, only the first elements from the buffer will be used in the amount specified when the class was initialized.

```
   if(SecondInput.Total() < i_SecondInputs)
      return false;
   if(SecondInput.GetIndex() < 0 && !SecondInput.BufferCreate(OpenCL))
      return false;
```

Then we check for the pointer to the data buffer in the OpenCL context and create a new buffer if necessary.

Note that we only create a new buffer if there is no pointer to the data buffer in the context. If it is present, we do not reload the data into the context. We believe that the presence of a pointer indicates the presence of data in the context. Therefore, when the contents of the buffer change on the side of the main program, it will be necessary to copy the data into the context. It is the user's responsibility to ensure that the data in the context memory is up to date.

Next, we pass pointers to the data buffers and the necessary constants to the kernel parameters. This procedure is identical for all kernels. Only the identifiers of kernels, parameters and pointers to the corresponding data buffers change. All mathematical operations should be specified in the body of the kernel itself on the OpenCL program side.

```
   if(!OpenCL.SetArgumentBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_w, ConcWeights.GetIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_i1, NeuronOCL.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_i2, SecondInput.GetIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(def_k_ConcatFeedForward, def_k_cff_matrix_o, Output.GetIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgument(def_k_ConcatFeedForward, def_k_cff_inputs1, (int)NeuronOCL.Neurons()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgument(def_k_ConcatFeedForward, def_k_cff_inputs2, (int)i_SecondInputs))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgument(def_k_ConcatFeedForward, def_k_cff_activation, (int)activation))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
```

At the end of the method operations, we specify the task spaces to run the kernel and put it in the execution queue.

```
   uint global_work_offset[1] = {0};
   uint global_work_size[1];
   global_work_size[0] = Output.Total();
   if(!OpenCL.Execute(def_k_ConcatFeedForward, 1, global_work_offset, global_work_size))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
//---
   return true;
  }
```

Here it is very important to control the correctness of the specification of the called kernel at each stage, as well as the buffer ID and its contents. Of course, we should not forget to control the correctness of operations at every step.

The methods for distributing error gradients and updating the weight matrix are based on a similar algorithm, and you can get acquainted with them in the attachment. It should only be noted that when distributing the error gradient, a buffer of error gradients will be added at the level of additional source data. In this work, we will not download and use its data. But it may be required in the future if the vector of additional initial data is generated by the second model.

After creating the methods of our CNeuronConcatenate class, we should take care of arranging the process of transferring an additional buffer of user's source data from the main program to a specific neural layer. Generally, the process is organized in such a way that after creating a model, the user works only with 2 methods: forward and reverse pass of the model as a whole. Users do not control data transfer between neural layers. The whole process takes place “under the hood” of our library. Therefore, the user should be able to call one forward pass method and specify 2 data buffers in its parameters. After that, the model should independently distribute the data into the appropriate information flows.

At this stage, we plan to use only one layer with data addition. In order not to complicate the process with additional tracking of which neural layer to transfer additional source data to, it was decided to pass the pointer to the buffer to all neural layers. The decision on the usage is made at the level of the class itself.

We will not consider in detail adding one parameter in several methods along the chain. The complete code of all methods and functions is available in the attachment. Let's dwell on just one detail: despite the fact that the direct pass methods of all classes have identical names and are declared virtual, adding a parameter in some and not having it in others does not allow you to fully redefine methods in inherited classes. To preserve heredity, we would have to redo the forward and backward pass methods of all previously created classes. We did not do this. Instead, we just added additional control to the dispatch methods of the underlying neural layer. Let's look at the example of the direct pass method.

In the parameters of the CNeuronBaseOCL::FeedForward dispatch method, we add a pointer to the data buffer and assign a default value to it. This trick will allow us to still use the method with only a pointer to the previous neural layer. This will be useful when using the library for previously created models and allow compiling previously created programs without any changes.

Next we check the type of the current neural layer. If we are in a class for combining data from two threads, then we call the corresponding forward pass method. Otherwise, we use the previously created algorithm. Below is only part of the method code with changes. Further, the method code did not change. The full code of the CNeuronBaseOCL::FeedForward method can be found in the attachment. There you will also find modified reverse pass dispatch methods. Additional buffers with null default pointers were added to them as well.

```
bool CNeuronBaseOCL::FeedForward(CObject *SourceObject, CBufferFloat *SecondInput = NULL)
  {
   if(CheckPointer(SourceObject) == POINTER_INVALID)
      return false;
//---
   CNeuronBaseOCL *temp = NULL;
   if(Type() == defNeuronConcatenate)
     {
      temp = SourceObject;
      CNeuronConcatenate *concat = GetPointer(this);
      return concat.feedForward(temp, SecondInput);
     }
```

There is a lot of information, but the article size is limited. Therefore, I rather briefly went through the methods of the new CNeuronConcatenate class. I hope this will not have a negative impact on the understanding of ideas and approaches. In any case, their algorithm is not much different from similar methods of the previously discussed classes. The complete code of all methods and classes is given in the attachment. If you have any questions, I am ready to answer them in the forum and personal messages on the website. Choose any communication channel convenient for you.

We move closer to the GCRL reinforcement learning method under consideration and consider the processes of building and training the model. As before, we will create 3 EAs:

- primary collection of examples "GCRL\\Research.mq5"
- agent training "GCRL\\StudyActor.mq5"
- testing the model operation "GCRL\\Test.mq5"

We will indicate the model architecture in the GCRL\\Trajectory.mqh include file.

As mentioned above, we will assemble the entire model within one agent. Consequently, we will only have a description of the architecture of one model. In the body of the CreateDescriptions method, we will first check the relevance of the pointer to the dynamic array object and, if necessary, create a new object. Be sure to clear the dynamic array before adding new objects for describing neural layers.

```
bool CreateDescriptions(CArrayObj *actor)
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
//--- Actor
   actor.Clear();
```

As always, we create the source data layer first. It is followed by the normalization layer. We already mentioned above that the initial data for the encoder will be only historical data and indicator parameters. This is reflected in the size of these neural layers.

```
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   int prev_count = descr.count = (HistoryBars * BarDescr);
   descr.window = 0;
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
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1000;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

Next, we completely repeat the encoder architecture from the previous article. It consists of a block of convolutions. It is followed by 3 fully connected layers and ends with the encoder layers of the latent representation of the variational auto encoder. This is a slightly unusual solution for a complete model. We have already talked about the conventions of dividing algorithms and models. Let's look at the practical results.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count - 1;
   descr.window = 2;
   descr.step = 1;
   descr.window_out = 4;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronProofOCL;
   prev_count = descr.count = prev_count;
   descr.window = 4;
   descr.step = 4;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvOCL;
   prev_count = descr.count = prev_count - 1;
   descr.window = 2;
   descr.step = 1;
   descr.window_out = 4;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronProofOCL;
   prev_count = descr.count = prev_count;
   descr.window = 4;
   descr.step = 4;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 128;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 2 * NSkills;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 9
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronVAEOCL;
   descr.count = NSkills;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

The encoder description is complete. Let's move on to creating our Agent. Its architecture begins with a layer of combining 2 data streams. The first stream is equal to the size of the encoder results. The second is equal to the size of the vector describing the task. We will use the description of the balance state as a vector for describing the task at hand.

In the theoretical part, we talked about the need for separability of subtasks. In our simplified scheme, we will use only 2 subtasks:

- searching for the entry point into a position
- searching for an exit point from a position

We indicated open positions in the structure of the account status description. Therefore, if the volume of open positions is "0", then the task is to open a position. Otherwise, we are looking for an exit point. The idea is simple and reminiscent of using a one-hot vector. The only difference is the volume of open positions. It will rarely be equal to "1" because we use the minimum lot and allow the simultaneous opening of several positions.

```
//--- layer 10
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = 256;
   descr.window=prev_count;
   descr.step=AccountDescr;
   descr.optimization = ADAM;
   descr.activation = TANH;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

We use relative units when describing the state of an account. We expect their value to be close to the normalized data. Therefore, we will not use the batch normalization layer here.

Next comes the decision-making block of 2 fully connected layers and the block of fully parameterized FQF quantile function. As you can see, we used a similar decision-making block in the agent from the previous article. There we already discussed the main properties and features of the solutions of each neural layer.

```
//--- layer 11
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 12
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseOCL;
   descr.count = 256;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 13
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFQF;
   descr.count = NActions;
   descr.window_out = 32;
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

After describing the architecture of the model, we move on to creating a robot for collecting the primary database of examples "GCRL\\Research.mq5". The algorithm of this EA moves practically unchanged from one article to another. Let me leave its detailed consideration beyond the scope of this article. The full EA code can be found in the attachment. We will only briefly discuss the changes caused by the use of the GCRL method.

First of all, let us remember that one of the disadvantages of the latest models was the long-term retention of open positions. We can notice that our vector for describing the account state contains the volume of open positions and accumulated profit in each direction. But there is no indication of the timing of opening positions. If we want to train an agent to control this process, then we should provide it with an appropriate reference point.

In the range of actions of our agent, there is only the option of closing all positions. Therefore, I do not see the need to separate the time of open long and short positions. Let's introduce one common parameter for all positions. At the same time, we wanted to create a parameter that would depend not only on time, but also on the volume of the position, accumulated profit or loss.

As such an indicator, we propose to use the sum of the absolute values of the accumulated profit/loss weighted by the duration of the open position. This will allow us to adapt the indicator to the time of opening a position, volume and market volatility (indirectly through profit). Using the absolute value of profit will allow us to eliminate the mutually absorbing influence of profitable and unprofitable positions.

![](https://c.mql5.com/2/55/2228941625071.png)

Taking into account the above, we will adjust the process of describing the account state, which is carried out in the OnTick method of the EA.

We will store the account balance and equity indicators in the first 2 elements of the account status description. To reduce the amount of information and improve its quality, we abandoned the indication of margin indicators due to their low information content in the context of the current task. However, I do not exclude their possible addition in subsequent works.

The time for opening positions is taken into account in seconds, and we work with the H1 timeframe. Let’s immediately determine the multiplier for adjusting the position validity time in hours. Here we will add a variable to calculate the penalty for holding a position using the above equation. However, we do not want the holding penalty to exceed the income from the position. For this purpose, we will determine that every hour we will impose the fine of 1/10 of the accumulated profit. Using the absolute value of profit in the equation above will allow us to penalize both profitable and unprofitable positions.

We save the current time into a local variable and start the loop of searching through open positions. In the loop body, we will calculate the volume of open positions and the accumulated profit/loss in each direction, as well as a total penalty for holding a position.

```
   sState.account[0] = (float)AccountInfoDouble(ACCOUNT_BALANCE);
   sState.account[1] = (float)AccountInfoDouble(ACCOUNT_EQUITY);
//---
   double buy_value = 0, sell_value = 0, buy_profit = 0, sell_profit = 0;
   double position_discount = 0;
   double multiplyer = 1.0 / (60.0 * 60.0 * 10.0);
   int total = PositionsTotal();
   datetime current = TimeCurrent();
   for(int i = 0; i < total; i++)
     {
      if(PositionGetSymbol(i) != Symb.Name())
         continue;
      switch((int)PositionGetInteger(POSITION_TYPE))
        {
         case POSITION_TYPE_BUY:
            buy_value += PositionGetDouble(POSITION_VOLUME);
            buy_profit += PositionGetDouble(POSITION_PROFIT);
            break;
         case POSITION_TYPE_SELL:
            sell_value += PositionGetDouble(POSITION_VOLUME);
            sell_profit += PositionGetDouble(POSITION_PROFIT);
            break;
        }
      position_discount -= (current - PositionGetInteger(POSITION_TIME)) * multiplyer*MathAbs(PositionGetDouble(POSITION_PROFIT));
     }
   sState.account[2] = (float)buy_value;
   sState.account[3] = (float)sell_value;
   sState.account[4] = (float)buy_profit;
   sState.account[5] = (float)sell_profit;
   sState.account[6] = (float)position_discount;
```

After completing the loop iterations, we will save the resulting values into the appropriate array elements for writing to the example database.

Before passing the data to our model, we will convert it into a relative units field.

```
   State.AssignArray(sState.state);
   Account.Clear();
   float PrevBalance = (Base.Total <= 0 ? sState.account[0] : Base.States[Base.Total - 1].account[0]);
   float PrevEquity = (Base.Total <= 0 ? sState.account[1] : Base.States[Base.Total - 1].account[1]);
   Account.Add((sState.account[0] - PrevBalance) / PrevBalance);
   Account.Add(sState.account[1] / PrevBalance);
   Account.Add((sState.account[1] - PrevEquity) / PrevEquity);
   Account.Add(sState.account[2]);
   Account.Add(sState.account[3]);
   Account.Add(sState.account[4] / PrevBalance);
   Account.Add(sState.account[5] / PrevBalance);
   Account.Add(sState.account[6] / PrevBalance);
```

Let me remind you that when describing the direct pass method, we emphasized that the responsibility for the relevance of the data of the additional source data buffer in the OpenCL context memory lies with the user. Therefore, after updating the account information buffer, we will transfer its contents to the context memory. Only after that, we call the direct pass method of our agent passing the pointers to both data buffers.

```
   if(Account.GetIndex()>=0)
      if(!Account.BufferWrite())
         return;
   if(!Actor.feedForward(GetPointer(State), 1, false, GetPointer(Account)))
      return;
```

The block for sampling and performing agent actions was transferred from similar EAs without changes and we will omit its description here.

At the end of the description of changes to the OnTick function of the EA for collecting examples, it is necessary to say a few words about the reward function. As before, the basis of our reward function is the relative value of changes in account balance. But the GCRL method provides additional rewards for achieving local goals. In our case, we will use penalties. For the task of closing positions, we will each time subtract the above calculated indicator of the weighted sum of the absolute values of accumulated profits and losses. By doing so, we will penalize holding positions with accumulated significant profits or losses as much as possible. This should encourage the agent to close positions. At the same time, positions with small accumulated profits will not generate a large penalty. This will allow the agent to expect profits to accumulate.

```
   float reward = Account[0];
   if((buy_value+sell_value)>0)
     reward+=(float)position_discount;
   else
     reward-=atr;
   if(!Base.Add(sState, act, reward))
      ExpertRemove();
//---
  }
```

If there are no open positions, we will encourage the agent to make trades. In this case, a penalty is provided in the amount of the current value of the ATR indicator.

Otherwise, the EA's algorithm has not undergone any changes. You can find its full code in the attachment.

After completing work on the EA for collecting the example database "GCRL\\Research.mq5", we launch it in the slow optimization mode of the strategy tester. Let's move on to the "GCRL\\StudyActor.mq5" Agent training EA.

In this work, we will train the agent only on actions and rewards stored in the example database. We will not calculate predictive rewards for other actions, as we did in the previous article. Instead, we will focus on teaching the agent to build a policy depending on the task at hand. We will take advantage of the fact that our database of examples contains passes for one historical period of time. But due to a number of randomly selected actions at the stage of collecting a database of examples, in each pass for one historical moment we will receive a different set of open positions and accumulated profits/losses with different actions of the agent and subsequent reward. This means we can carry out several forward and backward passes of the model from one historical moment with setting of various local tasks for the agent. This will give us the effect of replaying one moment several times and exploring the environment.

We will not waste resources and time searching for identical historical states. Let’s simply take advantage of the stationarity of historical data. After all, it is easy to notice that all our test agents started from one historical moment and "passed" the same number of steps (candles). An exception may be when the test is stopped due to a stop-out. But every N step in all passes will always correspond to one historical moment. This is what we will build our agent training on.

As always, model training is carried out in the Train function of the "GCRL\\StudyActor.mq5" EA. At the beginning of the function, we quantify it by passing through our example database. Then we organize the first loop, in which we find the pass with the maximum number of steps. We do not save a specific passage, but only the number of steps. We will use it when sampling a specific historical moment for training.

```
void Train(void)
  {
   int total_tr = ArraySize(Buffer);
   int total_steps = 0;
   for(int tr = 0; tr < total_tr; tr++)
     {
      if(Buffer[tr].Total > total_steps)
         total_steps = Buffer[tr].Total;
     }
```

Next, we will arrange a system of 2 nested loops. The first one is based on the number of model training iterations. In the body of this loop, we sample one historical moment for this training iteration. In a nested loop, we will iterate through all the passes available to us and check for the presence of sampled state in them.

```
   uint ticks = GetTickCount();
//---
   for(int iter = 0; (iter < Iterations && !IsStopped()); iter ++)
     {
      int i = (int)((MathRand() * MathRand() / MathPow(32767, 2)) * (total_steps - 2));
      for(int tr = 0; tr < total_tr; tr++)
        {
         if(i >= (Buffer[tr].Total - 1))
            continue;
```

If this condition exists, we train the Agent using the stored data and move on to the next pass.

```
         State.AssignArray(Buffer[tr].States[i].state);
         float PrevBalance = Buffer[tr].States[MathMax(i - 1, 0)].account[0];
         float PrevEquity = Buffer[tr].States[MathMax(i - 1, 0)].account[1];
         Account.Clear();
         Account.Add((Buffer[tr].States[i].account[0] - PrevBalance) / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[1] / PrevBalance);
         Account.Add((Buffer[tr].States[i].account[1] - PrevEquity) / PrevEquity);
         Account.Add(Buffer[tr].States[i].account[2]);
         Account.Add(Buffer[tr].States[i].account[3]);
         Account.Add(Buffer[tr].States[i].account[4] / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[5] / PrevBalance);
         Account.Add(Buffer[tr].States[i].account[6] / PrevBalance);
         //---
         if(Account.GetIndex()>=0)
            Account.BufferWrite();
         if(!Actor.feedForward(GetPointer(State), 1, false,GetPointer(Account)))
           {
            PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
            ExpertRemove();
            break;
           }
         //---
      ActorResult = vector<float>::Zeros(NActions);
      ActorResult[Buffer[tr].Actions[i]] = Buffer[tr].Revards[i];
      Result.AssignArray(ActorResult);
      if(!Actor.backProp(Result, 0, NULL, 1, false,GetPointer(Account),GetPointer(Gradient)))
        {
         PrintFormat("%s -> %d", __FUNCTION__, __LINE__);
         ExpertRemove();
         break;
        }
         if(GetTickCount() - ticks > 500)
           {
            string str = StringFormat("%-15s %5.2f%% -> Error %15.8f\n", "Actor",
                                       iter * 100.0 / (double)(Iterations),
                                       Actor.getRecentAverageError());
            Comment(str);
            ticks = GetTickCount();
           }
        }
     }
```

Thus, each individual state will be replayed by our agent in terms of the number of passes with a different formulation of the local subtask. Thus, we want to show the agent that its actions should take into account not only the state of the environment, but also the local subtask. As you may remember, when collecting the database of examples, we added a penalty for failure to complete a local task at each step. Now in each pass, we will have different rewards for one historical moment, which will correspond to the local subtasks of the passes.

The rest of the Expert Advisor code remained unchanged. Find the full code of all programs used in the article in the attachment.

### 3\. Test

After completing work on EAs, we move on to training the model and testing the results obtained. We do not change the model training parameters. As before, the model is trained on EURUSD H1 historical data. The indicator parameters are used by default. Our agent was trained on 4 months of 2023. We checked the quality of training and the ability of the Agent to work on new data within June 1-18, 2023.

The test results are presented in the screenshots below. As you can see, we managed to make a profit testing the model. On the balance chart, there are stages of growth and there is flat movement. I am glad there are no falls. In general, over 12 trading days, the profit factor was 2.2 and the recovery factor was 1.47. The EA made 220 trades. More than 53% of them were closed with a profit. Moreover, the average profitable position is almost 2 times higher than the average unprofitable one. Unfortunately, the EA only opened long positions. We have already encountered a similar effect. The applied approach did not solve this problem.

![Test graph](https://c.mql5.com/2/55/graph.png)

![Test results](https://c.mql5.com/2/55/tanle.png)

![Position holding time](https://c.mql5.com/2/55/PositionTime.png)

The positive aspects of using the GCRL method include a reduction in the time it takes to hold a position. During the test, the maximum position holding time was 21 hours and 15 minutes. The average time of holding a position is 5 hours 49 minutes. As you might remember, we set a penalty in the amount of 1/10 of the accumulated profit for each hour of holding for failure to complete the task of closing a position. In other words, after 10 hours of holding, the penalty exceeded the income from the position.

### Conclusion

In this article, we introduced the method of Goal-conditioned reinforcement learning (GCRL). A special feature of this method is the introduction of local subtasks and rewards for their achievement. This allows us to divide one global task into several smaller ones and move towards achieving it step by step.

This approach has a number of advantages. It reduces learning complexity by breaking down a task into smaller, more manageable components. This simplifies the decision-making process and improves the agent training speed.

In addition, GCRL helps improve the generalization ability of the agent. As the agent learns to solve different local subtasks, it develops a set of skills and strategies that can be applied in different contexts.

Finally, GCRL provides flexibility in defining goals and objectives for the agent. We can select and change local subtasks depending on our needs and environmental conditions. This allows the agent to adapt to different situations and effectively use their skills to achieve their goals.

We implemented the presented method using MQL5. We also trained the model and checked the training results on data outside the training set. The test results showed that there were still unresolved issues. In particular, the EA opened positions in only one direction. At the same time, this did not prevent it from making a profit during the test.

It should also be noted that the position holding time has decreased. This confirms the Agent work on solving 2 local tasks: opening and closing a position.

Generally, the test results are positive and allow the method to be used to find new solutions.

### List of references

[Variational Empowerment as Representation Learning for Goal-Based Reinforcement Learning](https://www.mql5.com/go?link=https://arxiv.org/pdf/2106.01404.pdf "https://arxiv.org/pdf/2106.01404.pdf")
[Neural networks made easy (Part 43): Mastering skills without the reward function](https://www.mql5.com/en/articles/12698)
[Neural networks made easy (Part 44): Learning skills with dynamics in mind](https://www.mql5.com/en/articles/12750)
[Neural networks made easy (Part 45): Training state exploration skills](https://www.mql5.com/en/articles/12783)

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Example collection EA |
| 2 | StudyActor.mq5 | Expert Advisor | Agent training EA |
| 3 | Test.mq5 | Expert Advisor | Model testing EA |
| 4 | Trajectory.mqh | Class library | System state description structure |
| 5 | FQF.mqh | Class library | Class library for arranging the work of a fully parameterized model |
| 6 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 7 | NeuroNet.cl | Code Base | OpenCL program code library |
| 8 | VAE.mqh | Class library | Variational auto encoder latent layer class library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/12816](https://www.mql5.com/ru/articles/12816)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/12816.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/12816/mql5.zip "Download MQL5.zip")(615.73 KB)

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

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/456644)**
(4)


![Nigel Philip J Stephens](https://c.mql5.com/avatar/avatar_na2.png)

**[Nigel Philip J Stephens](https://www.mql5.com/en/users/nigpig2)**
\|
1 Nov 2023 at 23:07

I could not reproduce your results, based on the mql5 download files and the historic and test data date ranges.


![Chris](https://c.mql5.com/avatar/avatar_na2.png)

**[Chris](https://www.mql5.com/en/users/nodon)**
\|
1 Mar 2024 at 00:55

Nice article.

Nigel, you are not the only one.

It's been presented enough to prevent reproducibility unless you spend pretty long time to debug the code or discover its proper usage.

For example:

"After completing work on the EA for collecting the example database "GCRL\\Research.mq5", we launch it in the slow optimization mode of the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ")"

Simple question is actually, what parameters are to be optimized?

![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
1 Mar 2024 at 01:56

**Chris [#](https://www.mql5.com/en/forum/456644#comment_52572090):**

Nice article.

Nigel, you are not the only one.

It's been presented enough to prevent reproducibility unless you spend pretty long time to debug the code or discover its proper usage.

For example:

"After completing work on the EA for collecting the example database "GCRL\\Research.mq5", we launch it in the slow optimization mode of the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ")"

Simple question is actually, what parameters are to be optimized?

All parameters are default. You must set only Agent number for optimize. It use to set number of tester iterations.

![Chris](https://c.mql5.com/avatar/avatar_na2.png)

**[Chris](https://www.mql5.com/en/users/nodon)**
\|
2 Mar 2024 at 12:22

**Dmitriy Gizlyk [#](https://www.mql5.com/en/forum/456644#comment_52572324):**

All parameters are default. You must set only Agent number for optimize. It use to set number of tester iterations.

Hi Dmitriy,

There must be something wrong with your library. In several tests I obtained the same results, having the same drawbacks.

The Test strategy generates two series of orders separated in time. First buy orders, then sell orders.

Sell orders are never being closed except the moment the testing period is over.

The same behaviour can be observed when testing your other strategies, so the bug must be in a class common to your strategies.

Another potential reason is some susceptibility to initial state of tests.

Find attached a report of my test.

![MQL5 Wizard Techniques you should know (Part 07): Dendrograms](https://c.mql5.com/2/59/Dendrograms_Logo.png)[MQL5 Wizard Techniques you should know (Part 07): Dendrograms](https://www.mql5.com/en/articles/13630)

Data classification for purposes of analysis and forecasting is a very diverse arena within machine learning and it features a large number of approaches and methods. This piece looks at one such approach, namely Agglomerative Hierarchical Classification.

![Neural networks made easy (Part 45): Training state exploration skills](https://c.mql5.com/2/55/Neural_Networks_Part_45_avatar.png)[Neural networks made easy (Part 45): Training state exploration skills](https://www.mql5.com/en/articles/12783)

Training useful skills without an explicit reward function is one of the main challenges in hierarchical reinforcement learning. Previously, we already got acquainted with two algorithms for solving this problem. But the question of the completeness of environmental research remains open. This article demonstrates a different approach to skill training, the use of which directly depends on the current state of the system.

![Developing an MQTT client for Metatrader 5: a TDD approach — Part 4](https://c.mql5.com/2/59/mechanism_in_MQTT_logo.png)[Developing an MQTT client for Metatrader 5: a TDD approach — Part 4](https://www.mql5.com/en/articles/13651)

This article is the fourth part of a series describing our development steps of a native MQL5 client for the MQTT protocol. In this part, we describe what MQTT v5.0 Properties are, their semantics, how we are reading some of them, and provide a brief example of how Properties can be used to extend the protocol.

![Neural networks made easy (Part 44): Learning skills with dynamics in mind](https://c.mql5.com/2/55/Neural_Networks_are_Just_a_Part_Avatar.png)[Neural networks made easy (Part 44): Learning skills with dynamics in mind](https://www.mql5.com/en/articles/12750)

In the previous article, we introduced the DIAYN method, which offers the algorithm for learning a variety of skills. The acquired skills can be used for various tasks. But such skills can be quite unpredictable, which can make them difficult to use. In this article, we will look at an algorithm for learning predictable skills.

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/12816&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5062673068213577406)

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