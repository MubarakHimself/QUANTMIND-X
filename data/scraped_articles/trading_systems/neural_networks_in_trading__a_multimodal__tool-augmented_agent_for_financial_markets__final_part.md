---
title: Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (Final Part)
url: https://www.mql5.com/en/articles/16867
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:29:20.952647
---

[![](https://www.mql5.com/ff/sh/9nb0c8df2rmwfn89z2/01.png) MetaTrader VPS vs regular cloud hosting services8 reasons why our solution is the best option for automated tradingRead](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=dgmsfszgoedimaicrqqmagvqzpwuxkur&s=c59e3617ccf44fd54d4c50a03b44fd689ff7507b8fe4990c83772cc5419e627d&uid=&ref=https://www.mql5.com/en/articles/16867&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069481583515207119)

MetaTrader 5 / Trading systems


### Introduction

In the previous [article](https://www.mql5.com/en/articles/16850), we started exploring the _FinAgent_ framework - an advanced tool designed for data analysis and decision support in financial markets. Its development focuses on creating an efficient mechanism for building trading strategies and minimizing risks in a complex and rapidly changing market environment. The architecture of _FinAgent_ consists of five interconnected modules, each performing specialized functions to ensure the overall adaptability of the system.

The Market Analysis Module is responsible for extracting and processing data from diverse sources, including price charts, market news, and reports. Within this module, the system identifies stable patterns that can be used to forecast price dynamics.

The Reflection Modules play a crucial role in the model's adaptation and learning process. The Low-Level Reflection Module analyzes interdependencies among current market signals, improving the accuracy of short-term forecasts. The High-Level Reflection Module, by contrast, works with long-term trends - incorporating historical data and the results of past trading decisions - to adjust the strategy based on accumulated experience.

The Memory Module provides long-term storage for large volumes of market data. Using modern vector similarity technologies, it minimizes noise and enhances information retrieval accuracy, which is especially important for developing long-term strategies and uncovering complex relationships.

At the core of the system lies the Decision-Making Module, which integrates the results from all other components. Based on both current and historical data, it generates optimal trading recommendations. Moreover, through the integration of expert knowledge and traditional indicators, the module is capable of producing balanced and well-founded suggestions.

The original visualization of the _FinAgent_ framework is provided below.

![](https://c.mql5.com/2/164/x2d1q__1.png)

In the [previous article](https://www.mql5.com/en/articles/16850#para3), we began implementing the approaches proposed by the authors of the _FinAgent_ framework using MQL5. We introduced algorithms for the low-level and high-level reflection modules, implemented as the objects _CNeuronLowLevelReflection_ and _CNeuronHighLevelReflection_. These modules analyze market signals, the history of trading decisions, and the actual financial results achieved, allowing the agent to adapt its behavior policy to changing market conditions. They also enable flexible responses to dynamic trend shifts and help identify key patterns within the data.

A distinctive feature of our implementation is the integration of memory blocks directly into the reflection objects. This approach differs from the original framework's architecture, where memory for all information streams was implemented as a separate module. By embedding memory into the reflection components themselves, we simplify the construction of data flows and interactions between different elements of the framework.

Continuing this work, we will now examine the implementation of several key modules, each playing a unique role within the overall system architecture:

- _Market Analysis Module_ is designed to process data from a wide variety of sources, including financial reports, news feeds, and stock quotations. It brings multimodal data into a unified format and extracts stable patterns that can be used to forecast future market dynamics.
- _Auxiliary Tools_, based on prior knowledge, support analysis and decision-making through historical patterns, statistical data, and expert evaluations. They also provide logical interpretability for the system decisions.
- _Decision Support System_, which consolidates the results of all modules to generate an adaptive and optimal trading strategy. This system offers real-time action recommendations, enabling traders and analysts to respond promptly to changing market conditions and make better-informed decisions.

The Market Analysis Module plays a central role in the system, as it is responsible for the preprocessing and unification of data. This step is particularly important for uncovering hidden patterns that are difficult to detect using traditional data analysis methods. The authors of _FinAgent_ employed large language models ( _LLMs_) to extract key aspects of data and perform dimensional compression. In our implementation, however, we chose not to use _LLMs_, instead focusing on specialized models for time series analysis, which provide higher precision and performance. In this article series, we have presented several frameworks for analyzing and forecasting multivariate time series. Any of them could be applied here. For the purposes of this article, we selected a transformer model with segmented attention, implemented as the _[CNeuronPSformer](https://www.mql5.com/en/articles/16483)_ class.

That said, this is by no means the only possible solution. In fact, the _FinAgent_ framework supports multimodal input data. This enable us not only to experiment with different algorithms for representing and analyzing time series, but also to combine them. This approach significantly expands the system capabilities, allowing for a more detailed understanding of market processes and contributing to the development of highly effective, adaptive trading strategies.

The Auxiliary Tools Module integrates prior knowledge about the analyzed environment into the overall model architecture. This component generates analytical signals based on classical indicators such as moving averages, oscillators, and volume-based indicators. All of them have long proven their effectiveness in algorithmic trading. However, the module is not limited to standard tools alone.

Furthermore, generating signals through well-defined rules based on technical indicator readings enhances the interpretability of the model's decisions and improves their reliability and effectiveness. This is a crucial factor for strategic planning and risk management.

### Auxiliary Tools Module

Developing a signal generation module based on the outputs of classical indicators within a neural model is a far more complex task than it may initially appear. The main difficulty lies not in interpreting the signals, but in evaluating the metrics fed into the model's input.

In traditional strategies, signal descriptions directly depend on the actual readings of the indicators. However, these values often belong to entirely unrelated and incomparable distributions, which creates significant challenges for model construction. This factor greatly reduces training efficiency, as algorithms must adapt to analyzing heterogeneous data. The result is longer processing times, reduced forecasting accuracy, and other adverse effects. For this reason, we previously decided to use only normalized data in our models.

The normalization process allows all analyzed features to be scaled to a common, comparable range, which in turn substantially improves model training quality. This approach minimizes the risk of distortions caused by differences in measurement units or time-dependent variability. An important advantage of normalization is that it enables deeper data analysis because in this form inputs become far more predictable and manageable for machine learning algorithms.

However, it should be noted that normalization significantly complicates signal generation in classical strategies. These strategies were originally designed to work with raw data and assume fixed threshold values for interpreting indicators. During normalization, the data are transformed, which causes undefined shifts in threshold levels. Moreover, normalization makes it impossible to generate signals based on the crossing of two lines in a classical indicator, since there is no guarantee that both lines will shift synchronously. As a result, the generated signals become distorted, or may not appear at all. This leads us to the necessity of developing new approaches to interpreting indicator outputs.

Here, I believe, we found a simple yet conceptually sound solution. The essence lies in the fact that, during normalization, all analyzed features are transformed to have a zero mean and unit variance. As a result, each variable becomes comparable to others and can be interpreted as a kind of oscillator. This provides a universal signal interpretation scheme: values above 0 are treated as buy signals, and values below 0 as sell signals. It is also possible to introduce threshold levels, creating "corridors" that filter out weak or ambiguous signals. This minimizes false positives, increases analysis accuracy, and supports more well-grounded decision-making.

We also account for the possibility of inverted signals for certain features. This issue can be resolved through the use of trainable parameters that adapt to historical data.

Applying this approach establishes a foundation for building models capable of effectively adapting to changing conditions and generating more accurate, reliable signals.

To implement this signal-generation method, we begin by constructing the _MoreLessEqual_ kernel on the _OpenCL_ side. In this case, a simple algorithm with a fixed threshold value was implemented.

The kernel parameters include pointers to two data buffers of equal size. One contains the input data, while the second will store the generated signals represented by one of three numerical values:

- -1 — sell
- 0 — no signal
- 1 — buy

We plan to execute the kernel in a one-dimensional task space corresponding to the size of the analyzed data buffer.

```
__kernel void MoreLessEqual(__global const float * input,
                            __global float * output)
  {
   const size_t i = get_global_id(0);
   const float value = IsNaNOrInf(input[i], 0);
   float result = 0;
```

Within the kernel body, we identify the current operation thread and immediately read the corresponding input value into a local variable. A mandatory step is to validate the input: any invalid data are automatically replaced with 0 to prevent downstream errors during further processing.

A local variable is then introduced to store intermediate results. Initially this variable is assigned a value indicating the absence of any signal.

Next, we check the absolute value of the analyzed variable. To generate a signal, this value must exceed the specified threshold.

```
   if(fabs(value) > 1.2e-7)
     {
      if(value > 0)
         result = 1;
      else
         result = -1;
     }
   output[i] = result;
  }
```

Positive values above the threshold produce a buy signal, while negative values below the threshold indicate a sell signal. The corresponding flag is stored in the local variable. And before the kernel completes, this flag is written into the results buffer.

The algorithm described above is a sequential forward-pass procedure, where data are processed without any trainable parameters. This method relies on strictly deterministic computations aimed at minimizing computational costs and avoiding unnecessary complexity - this is especially important when processing large volumes of information. It is also worth noting that error gradient propagation is not applied in this data flow, since our goal is to identify stable signals derived from indicator values rather than to "fit" them to a target output. This makes the algorithm particularly attractive for systems that demand both high speed and high accuracy of processing.

Once the algorithm is implemented on the _OpenCL_ side, we must organize the management and invocation of the kernel from the main program. To implement this functionality, we create a new object _CNeuronMoreLessEqual_ shown below.

```
class CNeuronMoreLessEqual :  public CNeuronBaseOCL
  {
protected:
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override {return true; }

public:
                     CNeuronMoreLessEqual(void) {};
                    ~CNeuronMoreLessEqual(void) {};
  };
```

The structure of this new object is very simple. It does not even include an initialization method. The parent class handles nearly all functionality. We only override the feed-forward and backpropagation methods.

In the feed-forward pass, pointers to the data buffers are passed to the parameters of the previously described kernel, which is then queued for execution.

```
bool CNeuronMoreLessEqual::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!OpenCL || !NeuronOCL)
      return false;
   uint global_work_offset[1] = { 0 };
   uint global_work_size[1] = { Neurons() };
   ResetLastError();
   const int kernel = def_k_MoreLessEqual;
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_mle_inputs, NeuronOCL.getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
   if(!OpenCL.SetArgumentBuffer(kernel, def_k_mle_outputs, getOutputIndex()))
     {
      printf("Error of set parameter kernel %s: %d; line %d", __FUNCTION__, GetLastError(), __LINE__);
      return false;
     }
//---
   if(!OpenCL.Execute(kernel, 1, global_work_offset, global_work_size))
     {
      printf("Error of execution kernel %s: %d; line %d", OpenCL.GetKernelName(kernel), GetLastError(), __LINE__);
      return false;
     }
//---
   return true;
  }
```

At first glance, the functionality of the backpropagation methods may seem unclear, given the earlier statement about the absence of trainable parameters and gradient propagation. However, it is important to note that, within a neural network architecture, these methods are mandatory for all layers. Otherwise, the corresponding parent-class method would be called, which might behave incorrectly in our specific architecture. To avoid such issues, the parameter-update method is overridden with a stub that simply returns _true_.

As for the omission of gradient propagation, logically this is equivalent to passing zero values. Thus, in the gradient distribution method, we simply reset the corresponding buffer in the source data object to zero, ensuring the model operates correctly and minimizing the risk of runtime errors.

```
bool CNeuronMoreLessEqual::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL || !NeuronOCL.getGradient())
      return false;
   return NeuronOCL.getGradient().Fill(0);
  }
```

This concludes our work on the Auxiliary Tools Module. The full code for the _CNeuronMoreLessEqual_ class and all its methods is provided in the attachment.

At this stage, we have covered nearly all key modules of the _FinAgent_ framework. The remaining component to discuss is the Decision-Making Module, which serves as the core element of the overall architecture. This module ensures the synthesis of information from multiple data streams - often more than two. We decided to integrate the decision-making module directly into the composite framework object rather than implementing it as a separate entity. This design choice has improved the interoperability of all system components.

### Building the FinAgent Framework

And now, the time has come to bring together all the previously created modules into a unified, comprehensive structure - the _FinAgent_ framework, ensuring their integration and synergistic interaction. Modules of different functional types are combined to achieve a common goal: creating an efficient and flexible system for analyzing complex market data and developing strategies that take into account the dynamics and specific characteristics of financial markets. This functionality is implemented by a new object _CNeuronFinAgent_. Its structure is shown below.

```
class CNeuronFinAgent   :  public CNeuronRelativeCrossAttention
  {
protected:
   CNeuronTransposeOCL  cTransposeState;
   CNeuronLowLevelReflection  cLowLevelReflection[2];
   CNeuronHighLevelReflection cHighLevelReflection;
   CNeuronMoreLessEqual cTools;
   CNeuronPSformer      cMarketIntelligence;
   CNeuronMemory        cMarketMemory;
   CNeuronRelativeCrossAttention cCrossLowLevel;
   CNeuronRelativeCrossAttention cMarketToLowLevel;
   CNeuronRelativeCrossAttention cMarketToTools;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                       CBufferFloat *SecondGradient, ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;

public:
                     CNeuronFinAgent(void) {};
                    ~CNeuronFinAgent(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint account_descr, uint nactions, uint segments,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronFinAgent; }
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

In this structure, we see a familiar set of overridable methods and several internal objects, among which one can easily identify the modules we have already implemented within the _FinAgent_ framework. The construction of the information flows that define their interaction will be discussed as we examine the implementation algorithms of this class methods.

All internal objects are declared statically, allowing us to leave the class constructor and destructor empty. Initialization of all declared and inherited objects is performed in the _Init_ method. The parameters of this method include several constants that define the architecture of the created object.

```
bool CNeuronFinAgent::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                           uint window, uint window_key, uint units_count, uint heads,
                           uint account_descr, uint nactions, uint segments,
                           ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl, 3,
                                           window_key, nactions / 3, heads,
                                           window, units_count,
                                           optimization_type, batch))
      return false;
```

Looking slightly ahead, note that our decision-making block will consist of several sequential cross-attention layers. The last of these is implemented through the parent object, which, not by coincidence, is based on the _[CNeuronRelativeCrossAttention](https://www.mql5.com/en/articles/16163#para3)_ class.

At the output of our _FinAgent_ framework implementation, we expect to obtain a tensor representation of the agent's actions in the form of a matrix, where each row is a vector describing a separate action. The buy and sell operations are represented by distinct rows of this matrix. Each operation is described by three parameters: the trade volume and two price levels - stop-loss and take-profit. Consequently, our action matrix will contain three columns.

Therefore, when calling the initialization method of the parent class, we specify the data analysis window on the main pipeline as 3, and the number of elements in the analyzed sequence as three times smaller than the vector size provided in the parameters describing the agent's actions. This configuration enables the model to evaluate the effectiveness of each individual operation within the context of a secondary information stream, through which the system transmits processed information about the surrounding environment. Therefore, we transfer the corresponding parameters.

After successfully executing the parent class initialization procedures, we proceed to prepare the newly declared internal objects. We begin by initializing the components of the Market Analysis Module. In our implementation, this module consists of two objects: a segmented-attention transformer for detecting stable patterns in multivariate time series data, and a memory block.

```
   int index = 0;
   if(!cMarketIntelligence.Init(0, index, OpenCL, window, units_count, segments, 0.2f, optimization, iBatch))
      return false;
   index++;
   if(!cMarketMemory.Init(0, index, OpenCL, window, window_key, units_count, heads, optimization, iBatch))
      return false;
```

To achieve a comprehensive analysis of the environment, we employ two low-level reflection modules, which operate in parallel on tensors of the input data represented in different projections. To obtain the second projection of the analyzed data, we use a transposition object.

```
   index++;
   if(!cTransposeState.Init(0, index, OpenCL, units_count, window, optimization, iBatch))
      return false;
```

Next, we initialize two low-level reflection objects. Their analysis of data from different projections is indicated by the interchange of the window and sequence length dimensions of the analyzed tensor.

```
   index++;
   if(!cLowLevelReflection[0].Init(0, index, OpenCL, window, window_key, units_count, heads, optimization, iBatch))
      return false;
   index++;
   if(!cLowLevelReflection[1].Init(0, index, OpenCL, units_count, window_key, window, heads, optimization, iBatch))
      return false;
```

In the first case, we analyze a multivariate time series, where each time step is represented by a data vector, and compare these vectors to uncover interdependencies among them. In the second case, we analyze individual univariate sequences to detect dependencies and regularities in their dynamics.

We then initialize the high-level reflection module, which examines the agent's recent actions in the context of market changes and financial results.

```
   index++;
   if(!cHighLevelReflection.Init(0, index, OpenCL, window, window_key, units_count, heads, account_descr, nactions,
                                                                                             optimization, iBatch))
      return false;
```

At this stage, we also prepare the Auxiliary Tools Module object.

```
   index++;
   if(!cTools.Init(0, index, OpenCL, window * units_count, optimization, iBatch))
      return false;
   cTools.SetActivationFunction(None);
```

The results of all the initialized modules are aggregated in the Decision-Making Module, which, as mentioned earlier, consists of several sequential cross-attention blocks. The first stage integrates information from the two low-level reflection modules.

```
   index++;
   if(!cCrossLowLevel.Init(0, index, OpenCL, window, window_key, units_count, heads, units_count, window,
                                                                                   optimization, iBatch))
      return false;
```

Next, we enrich the output of the Market Analysis Module with information derived from the low-level reflection modules.

```
   index++;
   if(!cMarketToLowLevel.Init(0, index, OpenCL, window, window_key, units_count, heads, window, units_count,
                                                                                       optimization, iBatch))
      return false;
```

Then, we add a layer of prior knowledge.

```
   index++;
   if(!cMarketToTools.Init(0, index, OpenCL, window, window_key, units_count, heads, window, units_count,
                                                                                   optimization, iBatch))
      return false;
//---
   return true;
  }
```

The final layer of the Decision-Making Module has already been initialized earlier. It is represented by the parent object.

After successfully initializing all nested objects, we return the logical result of the operations to the calling program and complete the method.

The next stage of our work is constructing the forward-pass algorithm for our _FinAgent_ framework implementation, in the _feedForward_ method. In the method parameters, we receive pointers to two input data objects. The method parameters include pointers to two input data objects: the first carries information about the current environmental state, while the second represents the account status and current financial results.

```
bool CNeuronFinAgent::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!cMarketIntelligence.FeedForward(NeuronOCL))
      return false;
   if(!cMarketMemory.FeedForward(cMarketIntelligence.AsObject()))
      return false;
```

Information about the analyzed market environment undergoes initial processing in the Market Analysis Module, which identifies patterns using the segmented-attention transformer and detects their stable combinations within the Memory Module.

The discovered patterns in two projections are then passed to the low-level reflection modules for a comprehensive analysis of market dynamics.

```
   if(!cTransposeState.FeedForward(cMarketIntelligence.AsObject()))
      return false;
   if(!cLowLevelReflection[0].FeedForward(cMarketIntelligence.AsObject()))
      return false;
   if(!cLowLevelReflection[1].FeedForward(cTransposeState.AsObject()))
      return false;
```

Note that the low-level reflection modules operate exclusively on the patterns detected in the current market environment, without using data from the memory block of the Market Analysis Module. This approach focuses on the immediate market reaction to the discovered patterns, allowing for a more precise assessment of current changes and trends without relying on historical data.

The same logic applies to the high-level reflection module.

```
   if(!cHighLevelReflection.FeedForward(cMarketIntelligence.AsObject(), SecondInput))
      return false;
```

As a reminder, the input to the high-level reflection module includes both the current market environment data (as output from the Market Analysis Module) and the financial results vector. The tensor of the agent's previous actions is used recursively from the high-level reflection module's results buffer.

The Auxiliary Tools Module, however, works directly with the raw input data, as it seeks signals based on prior knowledge contained in the analyzed indicator readings.

```
   if(!cTools.FeedForward(NeuronOCL))
      return false;
```

Next, we move on to organizing the processes of the Decision-Making Module. Initially, we enrich the results of the low-level reflection analysis by integrating dependencies identified in the dynamics of univariate sequences. This enhances analytical accuracy and deepens the model's understanding of system interactions, providing a more comprehensive evaluation of the market environment.

```
   if(!cCrossLowLevel.FeedForward(cLowLevelReflection[0].AsObject(), cLowLevelReflection[1].getOutput()))
      return false;
```

At the following stage, we integrate the information obtained from low-level reflection into the representation of stable patterns produced by the Memory Block in the Market Analysis Module. This step refines and reinforces the discovered relationships, yielding a more precise and comprehensive perception of the market's current dynamics and interactions.

```
   if(!cMarketToLowLevel.FeedForward(cMarketMemory.AsObject(), cCrossLowLevel.getOutput()))
      return false;
```

It is important to emphasize that the low-level reflection modules analyze the current market state, identifying the market's response to individual patterns. However, some of these patterns may occur infrequently, making their corresponding market reactions statistically insignificant. In such cases, information is stored in the low-level reflection module's memory, as similar patterns may appear in the future. This allows the model to gather additional data about market responses.

Nonetheless, unconfirmed information cannot be used for decision-making. Therefore, in the Decision-Making Module, we rely only on stable patterns, requesting corresponding reaction data from the low-level reflection module for a more accurate and well-founded assessment.

We then enhance the results of the market analysis by incorporating prior knowledge.

```
   if(!cMarketToTools.FeedForward(cMarketToLowLevel.AsObject(), cTools.getOutput()))
      return false;
```

Note that we did not introduce trainable parameters for interpreting the flags generated by the Auxiliary Tools Module, although this was discussed earlier. Instead, this functionality is delegated to the _Key_ and _Value_ formation parameters within the cross-attention module. Thus, interpretation and processing of these flags are directly integrated into the cross-attention mechanism. This makes explicit additional parameters unnecessary.

At the end of the feed-forward method, we analyze the results of the high-level reflection module in the context of the identified stable patterns and the market reactions to them. This operation is performed using the tools of the parent class.

```
   return CNeuronRelativeCrossAttention::feedForward(cHighLevelReflection.AsObject(), cMarketToTools.getOutput());
  }
```

The logical result of the operations is then returned to the calling program, completing the feed-forward method.

Following the forward pass, we proceed to organize the backpropagation processes. In this section, we will examine in detail the algorithm for the error gradient distribution method ( _calcInputGradients_), while leaving the trainable parameter optimization method ( _updateInputWeights_) for independent study.

```
bool CNeuronFinAgent::calcInputGradients(CNeuronBaseOCL *NeuronOCL,
                                         CBufferFloat *SecondInput,
                                         CBufferFloat *SecondGradient,
                                         ENUM_ACTIVATION SecondActivation = -1)
  {
   if(!NeuronOCL || !SecondGradient)
      return false;
```

The method parameters again include pointers to the same input data objects - this time, we must pass the error gradients according to the influence of the input data on the model's final output. The method body begins with a validation of these pointers, since further operations are meaningless if they are invalid.

As you know, error gradient distribution fully mirrors the information flow of the feed-forward pass, only in reverse. The forward pass concluded with a call to the s corresponding method of the parent class. Therefore, the backpropagation algorithm starts by calling the gradient distribution method of the parent class. It distributes the model's error between the high-level reflection module and the preceding cross-attention block of the Decision-Making Module.

```
   if(!CNeuronRelativeCrossAttention::calcInputGradients(cHighLevelReflection.AsObject(),
                                       cMarketToTools.getOutput(),
                                       cMarketToTools.getGradient(),
                                       (ENUM_ACTIVATION)cMarketToTools.Activation()))
      return false;
```

We then sequentially propagate the error gradients through all cross-attention blocks of the Decision-Making Module, distributing the errors across all information flows of the framework according to their influence on the model's output.

```
   if(!cMarketToLowLevel.calcHiddenGradients(cMarketToTools.AsObject(),
                                         cTools.getOutput(),
                                         cTools.getGradient(),
                                         (ENUM_ACTIVATION)cTools.Activation()))
      return false;
//---
   if(!cMarketMemory.calcHiddenGradients(cMarketToLowLevel.AsObject(),
                                         cCrossLowLevel.getOutput(),
                                         cCrossLowLevel.getGradient(),
                                         (ENUM_ACTIVATION)cCrossLowLevel.Activation()))
      return false;
```

Next, we propagate the error gradients through the low-level reflection modules.

```
   if(!cLowLevelReflection[0].calcHiddenGradients(cCrossLowLevel.AsObject(),
         cLowLevelReflection[1].getOutput(),
         cLowLevelReflection[1].getGradient(),
         (ENUM_ACTIVATION)cLowLevelReflection[1].Activation()))
      return false;
   if(!cTransposeState.calcHiddenGradients(cLowLevelReflection[1].AsObject()))
      return false;
```

At this point, the error gradients have been distributed across all framework modules. We must now collect data from all information streams at the level of the original input data. Recall that all reflection modules and the memory block of the Market Analysis Module operate on the preprocessed data generated by the segmented-attention transformer Therefore, we first collect the error gradient at the output level of that transformer.

The first step is transferring the error gradient from the memory block.

```
   if(!((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).calcHiddenGradients(cMarketMemory.AsObject()))
      return false;
```

Next, we replace the pointer to the error gradient buffer of our input data preprocessing object, allowing us to store the accumulated gradient values.

```
   CBufferFloat *temp = cMarketIntelligence.getGradient();
   if(!cMarketIntelligence.SetGradient(cMarketIntelligence.getPrevOutput(), false) ||
      !((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).calcHiddenGradients(cHighLevelReflection.AsObject(),
                                                            SecondInput, SecondGradient, SecondActivation) ||
      !SumAndNormilize(temp, cMarketIntelligence.getGradient(), temp, 1, false, 0, 0, 0, 1))
      return false;
```

We then invoke the gradient distribution method of the high-level reflection module. After that we must sum the results obtained from both data streams.

It should be noted that the high-level reflection module operates on two data streams. Thus, during gradient propagation, this module simultaneously processes errors from both the main stream and the financial results stream. This allows the model to account for errors in both crucial aspects of the analysis, ensuring more precise tuning of the system.

The low-level reflection modules handle gradient propagation in a similar manner. However, unlike the high-level reflection module, these operate on a single source of input data, simplifying the error gradient distribution process.

```
   if(!((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).calcHiddenGradients(cLowLevelReflection[0].AsObject()) ||
      !SumAndNormilize(temp, cMarketIntelligence.getGradient(), temp, 1, false, 0, 0, 0, 1))
      return false;
   if(!((CNeuronBaseOCL*)cMarketIntelligence.AsObject()).calcHiddenGradients(cTransposeState.AsObject()) ||
      !SumAndNormilize(temp, cMarketIntelligence.getGradient(), temp, 1, false, 0, 0, 0, 1) ||
      !cMarketIntelligence.SetGradient(temp, false))
      return false;
```

Do not forget that after each iteration, the newly obtained gradient values must be added to the previously accumulated gradients. This ensures that all model errors are correctly accounted for. After processing all information flows, it is important to restore the original buffer pointers to their initial state.

Finally, we pass the error gradient back to the input level of the main information stream and complete the method, returning the logical result of the operations to the calling program.

```
   if(!NeuronOCL.calcHiddenGradients(cMarketIntelligence.AsObject()))
      return false;
//---
   return true;
  }
```

Note that the Auxiliary Tools Module does not participate in the error gradient distribution algorithm. As discussed earlier, we do not plan to propagate gradients through this information flow. Moreover, clearing the gradient buffer for the data source object in this context would be harmful, since the same object also receives gradients through the main information stream.

This concludes our discussion of the _FinAgent_ framework implementation algorithms in _MQL5_. The full source code for all presented objects and methods is available in the attachments for your reference and further experimentation. There you will also find the complete program code and the architecture of the trainable model used in preparing this article. All components were transferred almost unchanged from the previous article on building an [agent with layered memory](https://www.mql5.com/en/articles/16816). The only modifications concern the model architecture, where we replaced a single neural layer with the integrated FinAgent framework described above.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFinAgent;
//--- Windows
     {
      int temp[] = {BarDescr, AccountDescr, 2 * NActions, Segments}; //Window, Account description, N Actions, Segments
      if(ArrayCopy(descr.windows, temp) < int(temp.Size()))
         return false;
     }
   descr.count = HistoryBars;
   descr.window_out = 32;
   descr.step = 4;                              // Heads
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

The architecture of all remaining layers has been preserved without alteration. And now, we move on to the final stage of our work - evaluating the effectiveness of the implemented approaches on real historical data.

### Testing

In the last two articles, we examined the _FinAgent_ framework in detail. During this process, we implemented our own interpretation of the approaches proposed by its authors. We adapted the framework algorithms to meet our specific requirements. We have now reached another important stage: evaluating the effectiveness of the implemented solutions on real historical data.

Please note that during development, we introduced significant modifications to the core algorithms of the _FinAgent_ framework. These changes affect key aspects of the model operation. Therefore, in this evaluation, we are assessing our adapted version, not the original framework.

The model was trained on historical data for the _EURUSD_ currency pair for 2023 using the H1 timeframe. All indicator settings used by the model were left at their default values, allowing us to focus on evaluating the algorithm itself and its ability to work with raw data without additional tuning.

For the initial training stage, we used a dataset prepared in previous studies. We applied a training algorithm that generates "almost ideal" target actions for the Agent, allowing us to train the model without continuously updating the training dataset. However, while this approach worked effectively, we believe that regular updates to the training set would improve accuracy and broaden the coverage of different account states.

After several training cycles, the model demonstrated stable profitability on both training and test data. Final testing was conducted using historical data for _January_ 2024, with all model parameters and indicator settings preserved. This approach provides an objective assessment of the model performance under conditions as close as possible to real market environments. The results are presented below.

![](https://c.mql5.com/2/164/3783279264844.png)![](https://c.mql5.com/2/164/4388848951190.png)

During the testing period, the model executed 95 trades, significantly exceeding the performance of previous models over a similar period. Over 42% of trades were closed with a profit. Since the average profitable trade was 1.5 times larger than the average losing trade, the model was profitable overall. The profit factor was recorded at 1.09.

Interestingly, the majority of profits were realized during the first half of the month, when prices fluctuated within a relatively narrow range. When a bearish trend developed, the balance line moved sideways, and some drawdown was observed.

![Symbol chart for the testing period](https://c.mql5.com/2/164/EURUSD_H1_Jan2024.png)

In my opinion, the observed behavior can likely be attributed to the algorithms within the Market Analysis Module and Auxiliary Tools Module. This area remains open for further investigation.

### Conclusion

We have explored the _FinAgent_ framework, an advanced solution for comprehensive market analysis and historical data evaluation. By integrating textual and visual information, the framework significantly expands the possibilities for making well-informed trading decisions. With its five key architectural components, _FinAgent_ demonstrates both accuracy and high adaptability, which are critical for trading in financial markets characterized by frequently changing conditions.

Notably, the framework is not limited to a single type of analysis. It provides a wide range of tools capable of working effectively with both textual and graphical data. This versatility allows the model to account for multiple market factors, providing a deeper understanding of market dynamics. These features make _FinAgent_ a promising tool for developing trading strategies that can adapt to changing market conditions and consider even minor market fluctuations.

In the practical part of our work, we implemented our interpretation of the framework approaches in _MQL5_. We trained the model by integrating these approaches and tested it on real historical data. The results demonstrated the model's ability to generate profits. However, profitability was found to be dependent on market conditions. Also, there's the need for further experiments to enhance the model's adaptability to dynamically changing market environments.

#### References

- [A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist](https://www.mql5.com/go?link=https://arxiv.org/abs/2402.18485 "A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Expert Advisor for collecting samples |
| 2 | ResearchRealORL.mq5 | Expert Advisor | Expert Advisor for collecting samples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Model training EA |
| 4 | Test.mq5 | Expert Advisor | Model Testing Expert Advisor |
| 5 | Trajectory.mqh | Class library | System state and model architecture description structure |
| 6 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 7 | NeuroNet.cl | Code library | OpenCL program code |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16867](https://www.mql5.com/ru/articles/16867)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16867.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16867/mql5.zip "Download MQL5.zip")(2327.64 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/498297)**

![Overcoming The Limitation of Machine Learning (Part 6): Effective Memory Cross Validation](https://c.mql5.com/2/176/20010-overcoming-the-limitation-of-logo.png)[Overcoming The Limitation of Machine Learning (Part 6): Effective Memory Cross Validation](https://www.mql5.com/en/articles/20010)

In this discussion, we contrast the classical approach to time series cross-validation with modern alternatives that challenge its core assumptions. We expose key blind spots in the traditional method—especially its failure to account for evolving market conditions. To address these gaps, we introduce Effective Memory Cross-Validation (EMCV), a domain-aware approach that questions the long-held belief that more historical data always improves performance.

![Mastering Quick Trades: Overcoming Execution Paralysis](https://c.mql5.com/2/176/19576-mastering-quick-trades-overcoming-logo.png)[Mastering Quick Trades: Overcoming Execution Paralysis](https://www.mql5.com/en/articles/19576)

The UT BOT ATR Trailing Indicator is a personal and customizable indicator that is very effective for traders who like to make quick decisions and make money from differences in price referred to as short-term trading (scalpers) and also proves to be vital and very effective for long-term traders (positional traders).

![From Basic to Intermediate: Template and Typename (V)](https://c.mql5.com/2/116/Do_b8sico_ao_intermediurio_Template_e_Typename____LOGO.png)[From Basic to Intermediate: Template and Typename (V)](https://www.mql5.com/en/articles/15671)

In this article, we'll explore one last simple use case for templates, and discuss the benefits and necessity of using typename in your code. Although this article may seem a bit complicated at first, it is important to understand it properly in order to use templates and typename later.

![Market Simulation (Part 04): Creating the C_Orders Class (I)](https://c.mql5.com/2/112/Simulat6o_de_mercado_Parte_01_Cross_Order_I_LOGO.png)[Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)

In this article, we will start creating the C\_Orders class to be able to send orders to the trading server. We'll do this little by little, as our goal is to explain in detail how this will happen through the messaging system.

[![](https://www.mql5.com/ff/si/3p2yc19r7qvs297n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F618%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dsignal.advantage%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=bewozmaxwejekdopjicjtsbzmjgfjyvt&s=e49ac7e84b713650e3af82ec3c6b4d02fdf06617c5821011b1e499af5edd01f4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=kgnyxanwwxzqaeqrfuhrxyuhxouhsjsl&ssn=1769182159623965505&ssn_dr=0&ssn_sr=0&fv_date=1769182159&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16867&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20in%20Trading%3A%20A%20Multimodal%2C%20Tool-Augmented%20Agent%20for%20Financial%20Markets%20(Final%20Part)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=17691821594616918&fz_uniq=5069481583515207119&sv=2552)

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