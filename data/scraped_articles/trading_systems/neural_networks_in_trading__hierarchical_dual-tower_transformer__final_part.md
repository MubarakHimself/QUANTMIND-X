---
title: Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)
url: https://www.mql5.com/en/articles/17104
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:27:18.260604
---

[Best articles and CodeBase updates in MQL5.community channelsFollow us to ensure you never miss out on important updates![](https://www.mql5.com/ff/sh/n9yf51p2srwzfqh5z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=dgazvhktsxqakdvarucjbvmvzenwlyje&s=98a038fe082e458df8c4a1d8e116e3a6646fd5517f06e48b2356b7ee005817d6&uid=&ref=https://www.mql5.com/en/articles/17104&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069452953263211860)

MetaTrader 5 / Trading systems


### Introduction

In the previous [article](https://www.mql5.com/en/articles/17069), we reviewed the theoretical aspects of the _Hidformer_ framework, developed specifically for analyzing and forecasting complex multivariate time series. The model demonstrates high effectiveness in processing dynamic and volatile data thanks to its unique architecture.

A key element of _Hidformer_ is the use of advanced attention mechanisms that make it possible not only to identify explicit dependencies in the data but also to uncover deep, latent relationships. To achieve this, the model employs a two-tower encoder, with each tower performing an independent analysis of the raw data. One tower specializes in analyzing the temporal structure, identifying trends and patterns, while the second examines the data in the frequency domain. This approach provides a comprehensive understanding of market dynamics, enabling the model to account for both short-term and long-term changes in price series.

An innovative aspect of the model is the use of a recursive attention mechanism to analyze temporal dependencies, allowing it to sequentially accumulate information about complex dynamic patterns of the financial instrument being studied. Combined with the linear attention mechanism used for analyzing the frequency spectrum of the input data, this approach optimizes computational costs and ensures training stability. This allows the _Hidformer_ framework to effectively adapt to the multidimensionality and nonlinearity of the input data, providing more reliable forecasts in conditions of high market volatility.

The model's decoder, built on a multilayer perceptron, enables prediction of the entire sequence of prices in a single step, minimizing the accumulation of errors typical for step-by-step forecasting. This significantly improves the quality of long-term forecasts, making the model especially valuable for practical applications in financial analysis.

The original visualization of the _Hidformer_ framework is provided below.

![](https://c.mql5.com/2/116/2554792780707.png)

In the [practical section](https://www.mql5.com/en/articles/17069#para3) of the previous article, we completed the preparatory work and implemented our own versions of the recursive and linear attention algorithms. Today, we continue the development of the approaches proposed by the authors of the _Hidformer_ framework.

### Time Series Analysis

The authors of the _Hidformer_ framework proposed a two-tower encoder architecture, which we adopted as our foundation. In our implementation, each encoder tower is represented as a separate object, allowing for flexible adaptation of the model to various tasks. However, unlike the original framework, we introduced several modifications driven by the specifics of the problem our model aims to solve. Initially, the framework was designed to forecast the continuation of the analyzed time series, but we took it a step further.

Drawing on the experience gained from implementing the _[MacroHFT](https://www.mql5.com/en/articles/16975)_ and _[FinCon](https://www.mql5.com/en/articles/16916)_ frameworks, we repurposed the encoder towers as independent agents that generate possible scenarios for upcoming trading operations. This significantly expands the system’s functional capabilities.

As in the original _Hidformer_ architecture, our agents analyze market data in the form of multivariate time series and their frequency characteristics. The recursive attention mechanism allows the model to capture dependencies within multivariate time series, while frequency spectrum analysis is performed using linear attention modules. This approach enables a deeper understanding of structural patterns in the data and allows the model to adapt to changing market conditions in real time - particularly important in high-frequency and algorithmic trading.

Additionally, each agent is equipped with a module for recurrent analysis of previously made decisions, allowing it to evaluate them in the context of the evolving market situation. This module provides the ability to analyze past decisions, identify the most effective strategies, and adapt the model to changing market conditions.

The time-series analysis agent is implemented as the _CNeuronHidformerTSAgent_ object. Its structure is shown below.

```
class CNeuronHidformerTSAgent    :  public CResidualConv
  {
protected:
   CNeuronBaseOCL                caRole[2];
   CNeuronRelativeCrossAttention cStateToRole;
   CNeuronS3                     cShuffle;
   CNeuronRecursiveAttention     cRecursiveState;
   CResidualConv                 cResidualState;
   CNeuronRecursiveAttention     cRecursiveAction;
   CNeuronMultiScaleRelativeCrossAttention   cActionToState;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHidformerTSAgent(void) {};
                    ~CNeuronHidformerTSAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint stack_size, uint action_space,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHidformerTSAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
```

As the parent class, we use a convolutional block with feedback, which serves as the _FeedForward_ block of one of the internal attention modules.

It is worth noting that the presented structure includes a wide spectrum of diverse components, each performing its own unique function in the organization of this new class of algorithms. These elements ensure a multifaceted approach, enabling the model to adapt to various scenarios of information processing and analysis of complex patterns. We will examine each of these components in more detail during the construction of the class methods.

All objects are declared statically, allowing us to leave the class constructor and destructor empty. Initialization of all inherited and declared objects is implemented in the _Init_ method. This method accepts several constant parameters that clearly define the architecture of the created object.

```
bool CNeuronHidformerTSAgent::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint window, uint window_key, uint units_count,
                                   uint heads, uint stack_size, uint action_space,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CResidualConv::Init(numOutputs, myIndex, open_cl, 3, 3, (action_space + 2) / 3, optimization_type, batch))
      return false;
```

Initialization begins with a call to the corresponding method of the parent class, which already contains the necessary control points and initialization procedures for inherited objects. It should be kept in mind that the interfaces of the parent object must produce outputs consistent with the intended behavior of the agent. In this case, the agent is expected to output a tensor of trading operations, where each operation is represented by three key parameters: trade volume, stop-loss level, and take-profit level. Buy and sell operations are represented as separate rows of this matrix. Thus, when calling the initialization method of the parent class, we set the window size of both the input data and the output results equal to 3, and set the sequence length to one third of the agent's action vector.

After the successful execution of the parent class operations, we initialize the newly introduced internal objects. First, we initialize the structures responsible for forming the agent-role tensor. We adapted this concept from the _FinCon_ framework and adjusted it to the current task. The main advantage of this concept lies in dividing the responsibilities for analyzing the input data among several parallel agents, enabling them to focus on specific aspects of the analyzed sequence.

```
//--- Role
   int index = 0;
   if(!caRole[0].Init(10 * window_key, index, OpenCL, 1, optimization, iBatch))
      return false;
   caRole[0].getOutput().Fill(1);
   index++;
   if(!caRole[1].Init(0, index, OpenCL, 10 * window_key, optimization, iBatch))
      return false;
```

Next, we initialize the relative cross-attention module, which highlights key properties of the input data in accordance with the agent’s assigned role.

```
//--- State to Role
   index++;
   if(!cStateToRole.Init(0, index, OpenCL, window, window_key, units_count, heads, window_key, 10,
                                                                            optimization, iBatch))
      return false;
```

Following the initial processing of the raw data, we return to the original _Hidformer_ architecture, which includes a segmentation step before feeding data into the encoder. It is important to note that segmentation is performed independently in each tower, which helps avoid unwanted correlations between different data streams and improves the model's adaptability to heterogeneous input sequences.

In our modified version, we expanded the agent’s functionality by replacing the classical segmentation mechanism with a specialized _[S3](https://www.mql5.com/en/articles/15074)_ module. This module not only performs segmentation but also implements a mechanism of learnable segment shuffling. Such an approach makes it possible to better identify latent relationships between different parts of the sequence. As a result, the agent can form more robust and generalized representations.

```
//--- State
   index++;
   if(!cShuffle.Init(0, index, OpenCL, window, window * units_count, optimization, iBatch))
      return false;
```

The data prepared in the previous steps is fed into the encoder, consisting of a recursive attention module and a convolutional block with feedback.

```
   index++;
   if(!cRecursiveState.Init(0, index, OpenCL, window, window_key, units_count, heads, stack_size, optimization, iBatch))
      return false;
   index++;
   if(!cResidualState.Init(0, index, OpenCL, window, window, units_count, optimization, iBatch))
      return false;
```

Such an encoder allows us to analyze the input sequence in the context of the latest price, identifying likely support and resistance levels or areas of stable pattern formation.

At the next stage, we again deviate from the original _Hidformer_ version and add a module for analyzing the agent's previously taken actions. First, we recursively analyze the latest action in the context of its historical sequence.

```
//--- Action
   index++;
   if(!cRecursiveAction.Init(0, index, OpenCL, 3, window_key, (action_space + 2) / 3, heads, stack_size,
                                                                                   optimization, iBatch))
      return false;
```

Then, we use a multi-scale cross-attention module to analyze the agent's policy in the context of dynamic market conditions.

```
   index++;
   if(!cActionToState.Init(0, index, OpenCL, 3, window_key, (action_space + 2) / 3, heads, window,
                                                                units_count, optimization, iBatch))
      return false;
//---
   return true;
  }
```

The functionality of the _FeedForward_ block is implemented through the capabilities of the parent class.

After successful initialization of all internal objects, we return the logical result of the operations to the calling program and complete the method.

We now proceed to designing the forward-pass algorithm, implemented within the _feedForward_ method. The method parameters include a pointer to the object containing the input data.

```
bool CNeuronHidformerTSAgent::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(bTrain && !caRole[1].FeedForward(caRole[0].AsObject()))
      return false;
```

Inside the method, we begin by generating the tensor describing the agent's current role. However, this operation is performed only during model training. It is easy to see that during model inference, a fixed role tensor will be generated at each iteration of the feed-forward pass. This this step is redundant. Therefore, we first check the current operating mode and only then call the forward pass of the internal fully connected layer responsible for generating the role tensor. This approach eliminates unnecessary operations and reduces decision-making latency.

We then proceed to processing the received input data. First, we extract the elements relevant to the agent's role. This is done through the cross-attention module.

```
//--- State to Role
   if(!cStateToRole.FeedForward(NeuronOCL, caRole[1].getOutput()))
      return false;
```

Next, the enhanced environment state is segmented and shuffled.

```
//--- State
   if(!cShuffle.FeedForward(cStateToRole.AsObject()))
      return false;
```

It is then processed by the recursive attention module, which enriches the representation of the environment state with information about prior price-movement dynamics.

```
   if(!cRecursiveState.FeedForward(cShuffle.AsObject()))
      return false;
   if(!cResidualState.FeedForward(cRecursiveState.AsObject()))
      return false;
```

At the next stage, we perform an in-depth analysis of the agent's behavior policy. First, the latest decision is analyzed in the context of previous actions stored in the memory of the recursive attention module.

```
//--- Action
   if(!cRecursiveAction.FeedForward(AsObject()))
      return false;
```

Then, we analyze the agent's policy in the context of the evolving market environment using the multi-scale cross-attention module.

```
   if(!cActionToState.FeedForward(cRecursiveAction.AsObject(), cResidualState.getOutput()))
      return false;
```

You can see that the architecture of the action-analysis module is borrowed from the classical _Transformer_ decoder. A classical decoder sequentially uses the modules _Self-Attention_ → _Cross-Attention_ → _FeedForward_. In our case, the _Self-Attention_ module was replaced by a recursive attention module in accordance with the _Hidformer_ framework. Following the same logic, we replaced multi-head _Cross-Attention_ with multi-scale attention. The remaining component is the _FeedForward_ block. It is implemented via the parent class. However, before using it, we must note that the input to this decoder-like structure consists of the results of the previous feed-forward pass of our method. For the correct execution of the backpropagation pass, we need to store this information. Therefore, we temporarily redirect the inherited data-buffer pointers and only then call the feed-forward method of the parent class.

```
   if(!SwapBuffers(Output, PrevOutput))
      return false;
//---
   return CResidualConv::feedForward(cActionToState.AsObject());
  }
```

The logical result of these operations is returned to the calling program, and the method concludes.

The next step is constructing the backpropagation algorithms. The backpropagation pass in our objects is represented by two methods: _calcInputGradients_ and _updateInputWeights_. The former ensures correct distribution of the error gradient among all objects involved in the decision-making process, proportionally to their influence on the final output. The latter performs optimization of the model's trainable parameters to minimize the total error. The updateInputWeights method is usually straightforward. It typically consists of calling the corresponding methods of internal objects containing trainable parameters, passing along the data saved during the feed-forward pass. The gradient-distribution method, however, is closely related to the feed-forward information flows and requires a more detailed explanation.

The parameters of the _calcInputGradients_ method include a pointer to the input-data object. This is the same object that was passed during the feed-forward pass. This time, however, we must send into it the error gradient corresponding to the influence of the input data on the model's output. Obviously, a valid pointer is required to transfer this information. Therefore, inside the method, we immediately check the pointer's validity.

```
bool CNeuronHidformerTSAgent::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
```

After this small control block, we proceed to constructing the gradient-distribution algorithm.

The information flows of gradient distribution mirror the flows of the feed-forward pass but in reverse. The feed-forward pass ended with a call to the parent class method. Accordingly, gradient distribution begins by using inherited mechanisms. At this stage, we call the relevant method of the parent class, passing the error downward to the module responsible for multi-scale cross-attention over both the agent's policy and market dynamics.

```
   if(!CResidualConv::calcInputGradients(cActionToState.AsObject()))
      return false;
```

Next, we must divide the resulting gradient between two information flows: analysis of the agent's policy and analysis of the environment state represented by the multivariate time series.

```
   if(!cRecursiveAction.calcHiddenGradients(cActionToState.AsObject(),
         cResidualState.getOutput(),
         cResidualState.getGradient(),
         (ENUM_ACTIVATION)cResidualState.Activation()))
      return false;
```

First, we distribute the gradient along the policy-analysis branch. To do this, we must pass it through the recursive attention module responsible for processing the agent's previous actions. Note that the inputs to this block were the results of the previous feed-forward pass of our object. We previously saved them in a separate data buffer. For proper gradient distribution, we must temporarily restore these values to the result buffer while preserving the current results. Therefore, we once again substitute the buffer pointers.

Moreover, during gradient distribution, the values in the corresponding interface buffers will be overwritten. This is undesirable because these values are still needed for parameter updates. Thus, we also redirect the error-gradient buffer.

Only after ensuring the preservation of all necessary data do we perform the gradient-distribution operations through the recursive attention module. After successful execution, the buffer pointers are restored to their original state.

```
//--- Action
   CBufferFloat *temp = Gradient;
   if(!SwapBuffers(Output, PrevOutput) ||
      !SetGradient(cRecursiveAction.getPrevOutput(), false))
      return false;
   if(!calcHiddenGradients(cRecursiveAction.AsObject()))
      return false;
   if(!SwapBuffers(Output, PrevOutput))
      return false;
   Gradient = temp;
```

We then proceed to gradient distribution along the multivariate time-series analysis path. First, we propagate the gradients to the level of the recursive attention module analyzing the environment state.

```
//--- State
   if(!cRecursiveState.calcHiddenGradients(cResidualState.AsObject()))
      return false;
```

Next, the gradient is passed to the segmentation and shuffling block.

```
   if(!cShuffle.calcHiddenGradients(cRecursiveState.AsObject()))
      return false;
```

Following this branch further, we transmit the gradient to the cross-attention module analyzing the raw data in the context of the agent's role.

```
   if(!cStateToRole.calcHiddenGradients(cShuffle.AsObject()))
      return false;
```

From this point, the gradient again splits into two flows: toward the input data object and toward the agent role formation branch.

```
   if(!NeuronOCL.calcHiddenGradients(cStateToRole.AsObject(),
                                     caRole[1].getOutput(),
                                     caRole[1].getGradient(),
                                     (ENUM_ACTIVATION)caRole[1].Activation()))
      return false;
//---
   return true;
  }
```

It is worth noting that no further gradient propagation occurs along the role formation branch. The first layer of this _MLP_ is fixed, and only the second neural layer contains trainable parameters, to which we have already passed the error signal.

Finally, we return the logical result of the execution to the calling program and complete the method.

This concludes our discussion of the algorithms used to build the methods of the time-series analysis agent for the environment state. You can find the full code of this object and all its methods in the attachments.

### Working with the Frequency Domain

The next stage is to construct an agent for analyzing the frequency characteristics of the analyzed signal. It should be noted that the structure of this agent is quite similar to the previously created time-series analysis agent. At the same time, it has distinct features associated with transforming the input signal into the frequency domain. To isolate the high- and low-frequency components of the environment state signal, we implemented a discrete wavelet transform, borrowed from the _[Multitask-Stockformer](https://www.mql5.com/en/articles/16747)_ framework.

The frequency-domain agent algorithms are implemented as the _CNeuronHidformerFreqAgent_ object. Its structure is shown below.

```
class CNeuronHidformerFreqAgent    :  public CResidualConv
  {
protected:
   CNeuronTransposeOCL           cTranspose;
   CNeuronLegendreWaveletsHL     cLegendre;
   CNeuronTransposeRCDOCL        cHLState;
   CNeuronLinerAttention         cAttentionState;
   CResidualConv                 cResidualState;
   CNeuronS3                     cShuffle;
   CNeuronRecursiveAttention     cRecursiveAction;
   CNeuronMultiScaleRelativeCrossAttention cActionToState;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHidformerFreqAgent(void) {};
                    ~CNeuronHidformerFreqAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint filters, uint units_count,
                          uint heads, uint stack_size, uint action_space,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHidformerFreqAgent; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
```

In the presented structure, one can easily notice similarities in the names of internal objects, indicating a related structure between the time-domain and frequency-domain agents. However, there are also differences, which we will examine in more detail when constructing the methods for this new class.

All internal objects are declared statically, which allows us to leave the constructor and destructor of the object empty. Initialization of all newly declared and inherited objects is performed within the _Init_ method.

```
bool CNeuronHidformerFreqAgent::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                     uint window, uint filters, uint units_count,
                                     uint heads, uint stack_size, uint action_space,
                                     ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CResidualConv::Init(numOutputs, myIndex, open_cl, 3, 3, (action_space + 2) / 3, optimization_type, batch))
      return false;
```

The method parameters include a set of constants that allow unambiguous interpretation of the architecture of the object being created. Within the method, we first call the relevant method of the parent class, which already implements the necessary control points and initialization of inherited objects and interfaces. It should be noted that, despite the difference in data domains, the agent is expected to output the same trading-operations tensor. Therefore, the approaches described above for calling the parent class initialization method in the time-series agent are equally applicable here.

Next, we initialize the newly declared objects. Pay attention to a distinction in the design of agents for different domains. The frequency-domain agent does not have a role-generation module. In our implementation, we do not plan to use a large number of frequency-domain agents.

Additionally, the segmentation block has been replaced by a discrete wavelet transform module. The transformation from the time domain to the frequency domain is performed on unit sequences. For convenient handling of these sequences, we first transpose the input-data matrix.

```
   int index = 0;
   if(!cTranspose.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      return false;
```

The univariate time series are divided into equal segments. A discrete wavelet transform is applied to each segment, allowing us to extract significant structural components of the temporal dependencies. The minimum segment size is limited to five elements, striking a balance between analysis accuracy and computational cost.

```
   index++;
   uint wind = (units_count>=20 ? (units_count + 3) / 4 : units_count);
   uint units = (units_count + wind - 1) / wind;
   if(!cLegendre.Init(0, index, OpenCL, wind, wind, units, filters, window, optimization, batch))
      return false;
```

It should be noted that the output of the discrete wavelet transform module is a tensor containing both the high- and low-frequency components of the signal. The high-frequency component immediately follows the low-frequency component of each segment, and the data can be represented as a three-dimensional tensor \[ _Segment_, \[ _Low_, _High_\], _Filters_\].

For further analysis, it is important to separate the data into the respective components. However, since identical operations are applied to both signal types, it is more efficient to process them in parallel. Therefore, we do not explicitly split the signal into separate objects; instead, we transpose the tensor, which allows more efficient use of computational resources and accelerates data processing.

```
   index++;
   if(!cHLState.Init(0, index, OpenCL, units * window, 2, filters, optimization, iBatch))
      return false;
```

As provided by the authors of the _Hidformer_ framework, we then apply a linear attention algorithm. In our case, we perform separate analyses of the high- and low-frequency components, enabling the identification of the most significant patterns and adaptive adjustment of the signal-processing strategy according to their frequency characteristics.

```
   index++;
   if(!cAttentionState.Init(0, index, OpenCL, filters, filters, units* window, 2, optimization, iBatch))
      return false;
```

The resulting outputs are passed through a convolutional block with feedback, serving as the _FeedForward_ module of our frequency-domain encoder.

```
   index++;
   if(!cResidualState.Init(0, index, OpenCL, filters, filters, 2 * units * window, optimization, iBatch))
      return false;
```

Next, we initialize the agent-policy analysis block, analogous to that used in the construction of the time-series agent. But there is one caveat. For the linear attention module, the order of segments is irrelevant, since the analysis is applied to the entire sequence at once. When using the multi-scale cross-attention module, however, we must address the prioritization of segments, as this module was designed for time-series sequences and prioritizes the most recent elements.

To solve this problem, we use the segmenting and shuffling object. In this case, our data is already segmented, and the key focus is the learnable shuffling of segments. This allows the model to independently learn segment priority based on the training data.

```
   index++;
   if(!cShuffle.Init(0, index, OpenCL, filters, cResidualState.Neurons(), optimization, iBatch))
      return false;
```

We will not elaborate further on the functionality of objects used by the agent-policy analysis module, as the approaches described for the time-series agent are preserved.

```
//--- Action
   index++;
   if(!cRecursiveAction.Init(0, index, OpenCL, 3, filters, (action_space + 2) / 3, heads, stack_size,
                                                                               optimization, iBatch))
      return false;
   index++;
   if(!cActionToState.Init(0, index, OpenCL, 3, filters, (action_space + 2) / 3, heads, filters,
                                                           2 * units * window, optimization, iBatch))
      return false;
//---
   return true;
  }
```

Once all internal objects are successfully initialized, we complete the method, returning a logical result to the calling program.

To reduce article length, the methods for feed-forward and backpropagation passes are left for independent study. Their algorithms follow the same principles as those described for the time-series agent. The full code for both agents and their methods is provided in the attachment.

### Top-Level Object

After constructing the objects of the multivariate time-series and frequency-domain towers, the next step in building the complete _Hidformer_ framework is to combine them into a single structure and add a decoder. The authors of _Hidformer_ used an _MLP_ as a decoder for forecasting the expected continuation of the analyzed time series. Despite our modification of the task, a perceptron can still be used to generate the final decision. However, we went a step further, borrowing the concept of a _[hyper-agent](https://www.mql5.com/en/articles/16975#para31)_ from the _MacroHFT_ framework. Inspired by this idea, we created the _CNeuronHidformer_ object having the following structure.

```
class CNeuronHidformer  :  public CNeuronBaseOCL
  {
protected:
   CNeuronTransposeOCL        cTranspose;
   CNeuronHidformerTSAgent    caTSAgents[4];
   CNeuronHidformerFreqAgent  caFreqAgents[2];
   CNeuronMacroHFTHyperAgent  cHyperAgent;
   CNeuronBaseOCL             cConcatenated;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronHidformer(void) {};
                    ~CNeuronHidformer(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint layers, uint stack_size, uint nactions,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronHidformer; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      Clear(void) override;
  };
```

In this architecture, one can clearly see the structural similarity to the _[CNeuronMacroHFT](https://www.mql5.com/en/articles/16975#para32)_ class from the _MacroHFT_ framework. Essentially, the new structure represents a modified version of it, retaining the core design principles while introducing targeted changes to improve data-processing efficiency.

A key difference lies in the configuration of environment-analysis agents. In this version, six specialized agents are used: four for analyzing multivariate time series and two for processing input data in the frequency domain. To ensure a balanced analysis, all agents are evenly distributed between processing the direct and transposed representations of the input data. This architecture allows for a more detailed exploration of various aspects of the input data, revealing hidden patterns and adaptively adjusting the processing strategy.

Overall, the modifications to the agent structure introduce only minor adjustments to the algorithms of the object's methods. The main logic remains unchanged, and all key operational principles of the model are preserved. Therefore, we leave it to the reader to study the construction algorithms of the methods independently. The full code for this object and all its methods is provided in the attachment.

### Model Architecture

A few words about the architecture of the trainable model. As you may have noticed, our constructed architecture is a synergy of the _Hidformer_ and _MacroHFT_ frameworks. The architecture of the trainable model and its training methods are no exception. We replicated the model architecture from the _MacroHFT_ framework, modifying only a single layer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronHidformer;
//--- Windows
     {
      int temp[] = {BarDescr, 120, NActions}; //Window, Stack Size, N Actions
      if(ArrayCopy(descr.windows, temp) < int(temp.Size()))
         return false;
     }
   descr.count = HistoryBars;
   descr.window_out = 32;
   descr.step = 4;                              // Scales
   descr.layers =3;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

Otherwise, the operational architecture remained unchanged, including the use of the risk-management agent. A complete description of the model architecture, as well as the full code for the training and testing routines, is provided in the attachment, transferred unchanged from the previous article.

### Testing

We have completed substantial work in implementing our interpretation of the approaches proposed by the _Hidformer_ authors. We now arrive at the crucial stage: evaluating the effectiveness of our solutions on real historical data. In our implementation, we borrowed extensively from the _MacroHFT_ framework. Therefore, it's logical to compare the new model performance with it. So, we train the new model on the [training dataset](https://www.mql5.com/en/articles/16993#para5) previously compiled for training the _MacroHFT_-based implementation.

That training dataset was collected from historical data for the entire year of 2024 for the _EURUSD_ currency pair on the _M1_ timeframe. All indicator parameters were set to their default values.

The same Expert Advisors are used for training and testing the model. Testing was conducted on historical data from January 2025, maintaining all other parameters. The test results are presented below.

![](https://c.mql5.com/2/116/2841821014537.png)![](https://c.mql5.com/2/116/1682972589067.png)

The results show that the model achieved profit on historical data outside the training dataset. Overall, during the calendar month, the model executed 29 trades. This makes slightly more than one trade per trading day, which is not enough for high-frequency trading. At the same time, more than 60% of the trades were profitable, and the average profitable trade exceeded the average losing trade by 60%.

### Conclusion

We explored the _Hidformer_ framework, designed for analyzing and forecasting complex multivariate time series. The model demonstrates high efficiency thanks to its unique dual-tower encoder architecture. One tower analyzes the temporal structure of the input data, while the other operates in the frequency domain. The recursive attention mechanism uncovers complex price-change patterns, while linear attention reduces the computational complexity of analyzing long sequences.

In the practical part of this work, we implemented our own interpretation of the proposed approaches using _MQL5_. We trained the model on real historical data and tested it on out-of-sample data. The test results demonstrate the model's potential. However, before deploying it in live trading, the model must be trained on a more representative dataset, followed by comprehensive testing.

#### References

- [Hidformer: Transformer-Style Neural Network in Stock Price Forecasting](https://www.mql5.com/go?link=https://arxiv.org/abs/2412.19932 "Hidformer: Transformer-Style Neural Network in Stock Price Forecasting")
- [Hidformer: Hierarchical dual-tower transformer using multi-scale mergence for long-term time series forecasting](https://www.mql5.com/go?link=https://www.sciencedirect.com/science/article/abs/pii/S0957417423029147?via=ihub "Hidformer: Hierarchical dual-tower transformer using multi-scale mergence for long-term time series forecasting")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Expert Advisor for collecting examples |
| 2 | ResearchRealORL.mq5 | Expert Advisor | Expert Advisor for collecting samples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Model training Expert Advisor |
| 4 | Test.mq5 | Expert Advisor | Model testing Expert Advisor |
| 5 | Trajectory.mqh | Class library | System state and model architecture description structure |
| 6 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 7 | NeuroNet.cl | Library | OpenCL program code |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17104](https://www.mql5.com/ru/articles/17104)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17104.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/17104/mql5.zip "Download MQL5.zip")(2406.07 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model](https://www.mql5.com/en/articles/17142)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**[Go to discussion](https://www.mql5.com/en/forum/500736)**

![Market Simulation (Part 07): Sockets (I)](https://c.mql5.com/2/117/Simula92o_de_mercado_Parte_07__LOGO2.png)[Market Simulation (Part 07): Sockets (I)](https://www.mql5.com/en/articles/12621)

Sockets. Do you know what they are for or how to use them in MetaTrader 5? If the answer is no, let's start by studying them. In today's article, we'll cover the basics. Since there are several ways to do the same thing, and we are always interested in the result, I want to show that there is indeed a simple way to transfer data from MetaTrader 5 to other programs, such as Excel. However, the main idea is not to transfer data from MetaTrader 5 to Excel, but the opposite, that is, to transfer data from Excel or any other program to MetaTrader 5.

![Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://c.mql5.com/2/183/20387-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 52): Master Market Structure with Multi-Timeframe Visual Analysis](https://www.mql5.com/en/articles/20387)

This article presents the Multi‑Timeframe Visual Analyzer, an MQL5 Expert Advisor that reconstructs and overlays higher‑timeframe candles directly onto your active chart. It explains the implementation, key inputs, and practical outcomes, supported by an animated demo and chart examples showing instant toggling, multi‑timeframe confirmation, and configurable alerts. Read on to see how this tool can make chart analysis faster, clearer, and more efficient.

![Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://c.mql5.com/2/122/Developing_a_Multicurrency_Advisor_Part_23___LOGO_2.png)[Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

We aim to create a system for automatic periodic optimization of trading strategies used in one final EA. As the system evolves, it becomes increasingly complex, so it is necessary to look at it as a whole from time to time in order to identify bottlenecks and suboptimal solutions.

![Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://c.mql5.com/2/183/19035-implementing-practical-modules-logo.png)[Implementing Practical Modules from Other Languages in MQL5 (Part 04): time, date, and datetime modules from Python](https://www.mql5.com/en/articles/19035)

Unlike MQL5, Python programming language offers control and flexibility when it comes to dealing with and manipulating time. In this article, we will implement similar modules for better handling of dates and time in MQL5 as in Python.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/17104&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069452953263211860)

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