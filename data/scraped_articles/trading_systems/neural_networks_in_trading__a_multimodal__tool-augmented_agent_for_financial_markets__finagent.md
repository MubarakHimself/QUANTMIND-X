---
title: Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)
url: https://www.mql5.com/en/articles/16850
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:29:40.000688
---

[![](https://www.mql5.com/ff/sh/a27a2kwmtszm2m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
MQL5 Channels - Messenger for traders\\
\\
Subscribe to traders' channels or create your own.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=vpcudokyepxfrcxrpjcktglhsjlemtza&s=f08ad2c1289e29bd5630f1ef977aef297d5cdbfcb686faed4a4b0f1e276d3c4a&uid=&ref=https://www.mql5.com/en/articles/16850&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069486153360410082)

MetaTrader 5 / Trading systems


### Introduction

Financial markets play an important role in maintaining economic stability, participating in capital allocation and risk management. Modern financial trading systems, which widely use technical analysis, enhance these processes; however, in conditions of high market volatility and variability, they often face significant limitations. Rule-based trading systems are rigid and difficult to adapt to rapidly changing market conditions, frequently resulting in reduced effectiveness. Reinforcement learning ( _RL_)–based systems demonstrate greater adaptability but also have their shortcomings:

- High demand for extensive training data;
- Decisions can be difficult to explain;
- Difficult to generalize across different market conditions;
- Sensitive to market noise;
- Restricted integration of multimodal market information.

In recent years, large language models ( _LLMs_) have shown remarkable potential in decision-making, expanding their application beyond natural language processing. The integration of memory and planning modules allows _LLMs_ to adapt to dynamically changing environments. Multimodal _LLMs_ further enhance these capabilities by processing both textual and visual information, while the addition of external tools broadens the range of tasks these models can tackle, including complex financial scenarios.

Despite successes in financial data analysis, _LLM_ agents face several limitations:

- Limited multimodal processing of numerical, textual, and visual data;
- The need for precise integration of data from multiple sources;
- Low adaptability to rapidly changing markets;
- Difficulty in using expert knowledge and traditional methods;
- Insufficient transparency in decision-making.

The authors of the study " _[A Multimodal Foundation Agent for Financial Trading: Tool-Augmented, Diversified, and Generalist](https://www.mql5.com/go?link=https://arxiv.org/abs/2402.18485 "https://arxiv.org/abs/2402.18485")_" attempted to solve these limitations by introducing the _FinAgent_ framework — a multimodal foundation agent that integrates textual and visual information for analyzing market dynamics and historical data. The key components of _FinAgent_ include multimodal data processing to identify critical market trends, a two-tiered reflection module analyzing both short- and long-term decisions, a memory system to minimize noise in analysis, and a decision-making module that incorporates expert knowledge and advanced trading strategies.

### The FinAgent Algorithm

The _FinAgent_ framework is a tool for data analysis and informed decision-making in financial markets. It provides users with a set of instruments for understanding market processes, forecasting dynamics, and optimizing trading strategies. _FinAgent_ comprises five primary modules that interact to form an ecosystem for data processing and decision-making.

The Market Analysis Module is responsible for collecting, processing, and interpreting various data sources, including market news, price changes, and company reports. Using advanced methods, the module identifies hidden patterns, enabling the agent to adapt its actions to current market conditions.

To achieve maximum efficiency, _FinAgent_ analyzes both current market data and accumulated historical information. Daily updates, such as news, price fluctuations, and other operational data, form the basis for short-term decision-making. Simultaneously, analyzing past events helps uncover long-term patterns, enabling the development of resilient strategies for future operations. This dual approach ensures high adaptability and flexibility.

_FinAgent_'s data acquisition process uses a large language model ( _LLM_) that transforms market information into textual queries. These queries are then used to search for similar data within the memory module's historical database. The application of vector similarity methods enhances search accuracy and focuses on the most relevant information. Additionally, the system uses specialized text fields, improving data handling and preventing the loss of critical details. The resulting data are structured and summarized, simplifying analysis and minimizing the influence of secondary information.

The two-tiered Reflection Module functions similarly to human learning processes. The low-level reflection module identifies correlations between price changes and market dynamics, enabling short-term market fluctuation forecasts, which is especially important for traders operating on smaller time scales. Meanwhile, the high-level reflection module analyzes more complex and deeper connections based on historical data and prior trading outcomes. This allows for error detection and the development of corrective strategies. The process includes visualizing key points, such as buy and sell moments, and evaluating their effectiveness. An iterative learning approach enables the agent to accumulate experience and leverage it to improve future actions.

The Memory Module plays a central role in ensuring _FinAgent_'s stable operation. It is responsible for storing data and enabling efficient retrieval. Vector search methods allow rapid access to relevant information within vast datasets, reducing noise and enhancing analytical precision. The memory mechanism in _FinAGent_ provided important context and cognitive capabilities. In financial trading, memory is particularly important for accuracy, adaptability, and learning from past experiences. It allows the agent to use current news and reports to forecast future market changes, adapt to volatile conditions, and continually refine strategies.

The Decision-Making Module integrates and processes key data, including market summaries, price dynamics analysis from low-level reflection, and insights from previous decisions. It also incorporates tools complemented by professional investment recommendations and time-tested trading strategies. A critical function of this module is market sentiment analysis, predicting bullish and bearish trends based on current price movements, and reflecting on lessons learned. Additionally, it considers expert advice and evaluates the effectiveness of traditional indicators.

Based on the combination of these analyses and the current financial context, the module generates the final decision: whether to buy, sell, or hold an asset. Contextual learning principles are applied to create a logical decision structure. This ensures that every trading action is justified, grounded in a comprehensive understanding of market dynamics within the relevant context. Such an approach allows for a better adaptation to market conditions and enables informed, strategically sound decisions.

Thus, _FinAgent_ represents a comprehensive tool that integrates data analysis, reflection, and process automation. It enables traders and analysts to effectively adapt to the market, minimize risks, and enhance profitability, opening new opportunities for strategic planning.

The original visualization of the _FinAgent_ framework is as follows:

![Author's visualization of the FinAgent framework](https://c.mql5.com/2/176/x2d1q__1.png)

### Implementation in MQL5

After examining the theoretical aspects of the _FinAgent_ framework, we now move to the practical part of this article, where we implement our vision of the proposed approaches using _MQL5_.

It is important to note that, similar to previous works, we exclude the use of large language models and aim to implement the proposed approaches using the tools available to us.

We begin our work by creating the low- and high-level reflection modules.

**Low-Level Reflection Module**

When starting the development of the low-level reflection module, it is worth drawing attention to the memory module architecture proposed by the authors of the _FinAgent_ framework. The memory module can be conceptually divided into three objects that collect information from the market analysis module and the two reflection modules of different levels. This allows us to restructure the model's modules without altering the overall information flow and to integrate individual memory blocks into the corresponding modules. Taking advantage of this property, we incorporate a memory block for this information stream into the low-level reflection module.

The modified low-level reflection module is implemented as the _CNeuronLowLevelReflection_ object - its structure is shown below.

```
class CNeuronLowLevelReflection :   public CNeuronMemory
  {
protected:
   CNeuronLSTMOCL    cChangeLSTM;
   CNeuronMambaOCL   cChangeMamba;
   CNeuronRelativeCrossAttention cCrossAttention[2];
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronLowLevelReflection(void) {};
                    ~CNeuronLowLevelReflection(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronLowLevelReflection; }
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

Using the memory object as a parent class enables seamless integration of market information analysis processes with cognitive memory functions into a unified information pipeline. This allows the system not only to efficiently process raw data but also to account for historical information, providing context for current analysis.

The class structure includes recurrent objects of various architectures and cross-attention blocks. These components play a key role in information processing and decision-making. Their purpose and functionality are explained in detail during the implementation of the object’s methods, allowing a deeper understanding of their influence on system performance.

All internal objects are declared as static, allowing us to keep the class constructor and destructor empty. The initialization of these declared and inherited objects is performed in the _Init_ method.

```
bool CNeuronLowLevelReflection::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                            uint window, uint window_key, uint units_count, uint heads,
                                       ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMemory::Init(numOutputs, myIndex, open_cl, window, window_key, units_count, heads,
                                                                      optimization_type, batch))
      return false;
```

The parameter structure of this method is fully inherited from the parent class. In the method body, we immediately call the parent's method of the same name, passing all received parameters.

The parent class method already implements the algorithm for controlling received parameters and initializing inherited objects.

Next, we initialize the newly declared objects. We first initialize two recurrent objects designed to identify the dynamics of the analyzed parameters. Using different architectural solutions for these objects ensures deeper analysis, allowing the system to adapt effectively to both short-term and long-term changes, detecting trends across multiple time scales.

```
   int index = 0;
   if(!cChangeLSTM.Init(0, index, OpenCL, window, units_count, optimization, iBatch))
      return false;
   index++;
   if(!cChangeMamba.Init(0, index, OpenCL, window, 2 * window, units_count, optimization, iBatch))
      return false;
```

The analysis results are integrated into a unified solution via cross-attention blocks, which effectively combine information from different sources and levels, focusing on key relationships and dependencies between parameters. Cross-attention facilitates the discovery of hidden patterns and interconnections, improving decision quality and providing more accurate and coherent information perception.

```
   for(int i = 0; i < 2; i++)
     {
      index++;
      if(!cCrossAttention[i].Init(0, index, OpenCL, window, window_key, units_count, heads,
                                                window, units_count, optimization, iBatch))
         return false;
     }
//---
   return true;
  }
```

After initializing all internal objects, we return the logical result of the operations to the calling program and complete the method execution.

The next stage is building the feed-forward algorithms for the low-level reflection module, implemented in the _feedForward_ method.

```
bool CNeuronLowLevelReflection::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cChangeLSTM.FeedForward(NeuronOCL))
      return false;
   if(!cChangeMamba.FeedForward(NeuronOCL))
      return false;
```

The method receives a pointer to the raw data object, which contains a description of the current state of the environment. This data is passed to the corresponding methods of our recurrent objects to identify parameter dynamics, enabling the system to track environmental changes and adapt its decisions accordingly.

We then use cross-attention blocks to enrich the description of the current environmental state with information about detected changes. This allows integration of new data with existing data, enhancing context and improving the perception of environmental dynamics. This approach creates a "trajectory trace" that reflects changes in the analyzed state.

```
   if(!cCrossAttention[0].FeedForward(NeuronOCL, cChangeLSTM.getOutput()))
      return false;
   if(!cCrossAttention[1].FeedForward(cCrossAttention[0].AsObject(), cChangeMamba.getOutput()))
      return false;
```

The analysis results are passed to the memory module, implemented by the parent class. This module identifies persistent trends, forming the basis for predicting upcoming price movements.

```
   return CNeuronMemory::feedForward(cCrossAttention[1].AsObject());
  }
```

As can be seen, three internal objects process the raw data describing the environment during the feed-forward pass. Consequently, during the backpropagation pass, we must gather error gradients from all three information streams. The algorithm for distributing error gradients is implemented in the _calcInputGradients_ method.

```
bool CNeuronLowLevelReflection::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
```

The method receives a pointer to the same source data object. However, this time, we pass an error gradient reflecting the influence of the source data on the model's output. The method first checks the validity of the received pointer, as data transfer is impossible otherwise.

Using the parent class, we propagate the error gradient through the memory module down to the cross-attention blocks.

```
   if(!CNeuronMemory::calcInputGradients(cCrossAttention[1].AsObject()))
      return false;
   if(!cCrossAttention[0].calcHiddenGradients(cCrossAttention[1].AsObject(),
                       cChangeMamba.getOutput(), cChangeMamba.getGradient(),
                                (ENUM_ACTIVATION)cChangeMamba.Activation()))
      return false;
```

Then we distribute the resulting error among the recurrent blocks.

Next, we propagate the error gradient through the three information streams back to the raw input level. Here we first compute the gradient along the cross-attention block pipeline.

```
   if(!NeuronOCL.calcHiddenGradients(cCrossAttention[0].AsObject(), cChangeLSTM.getOutput(),
                      cChangeLSTM.getGradient(), (ENUM_ACTIVATION)cChangeLSTM.Activation()))
      return false;
```

Then we substitute the pointer to the raw input data object's gradient buffer, propagating the error through the recurrent object pipelines and summing the values obtained from different information streams.

```
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cChangeMamba.getPrevOutput(), false) ||
      !NeuronOCL.calcHiddenGradients(cChangeMamba.AsObject()) ||
      !SumAndNormilize(NeuronOCL.getGradient(), temp, temp, iWindow, false, 0, 0, 0, 1))
      return false;
```

```
   if(!NeuronOCL.calcHiddenGradients(cChangeLSTM.AsObject()) ||
      !SumAndNormilize(NeuronOCL.getGradient(), temp, temp, iWindow, false, 0, 0, 0, 1))
      return false;
```

Finally, the pointers to the data buffers are restored to their original state, ensuring proper preparation for further use.

```
   if(!NeuronOCL.SetGradient(temp, false))
      return false;
//---
   return true;
  }
```

At the end of the method, the logical result of the operations is returned to the calling program.

This concludes the discussion of the algorithms for constructing the methods of the low-level reflection module. You can find the complete code of this class and all its methods in the attachment.

**High-Level Reflection Module**

Next, we need to create the high-level reflection module, which focuses on deeper and more comprehensive analysis of interdependencies. Unlike the low-level reflection module, which concentrates on short-term changes and the dynamics of the current state, the high-level reflection module examines long-term trends and relationships identified from previous trading operations.

The primary objective of this module is a thorough assessment of the validity of previously made decisions, as well as an analysis of the actual results. This process allows us to evaluate how effectively the current trading strategy was applied and how well it aligns with the set goals. A key part of this analysis is identifying the strengths and weaknesses of the strategy.

Additionally, the high-level reflection module is expected to provide specific recommendations for optimizing the trading strategy. These recommendations aim to improve trade profitability and/or reduce risk.

Obviously, to fully implement the functionality of the high-level reflection module, significantly more input data is required. First of all, we need the agent's actual actions for analysis.

Furthermore, we need a description of the environment at the moment decisions were made. This enables the evaluation of how justified each decision was under specific market conditions and how these conditions may have affected the results.

Equally important is profit-and-loss information, which helps assess how decisions impacted financial results. This data not only enables evaluation of trade success but also identifies potential weaknesses in the strategy that may require adjustment.

Moreover, for effective functioning of the high-level reflection module, the results of the analysis must be stored for future use. This enables the system to consider past conclusions and improve decision-making based on accumulated experience. The authors of the _FinAgent_ framework accounted for this requirement and incorporated a corresponding memory module block into the architecture.

Thus, we arrive at four streams of input information:

- Agent actions;
- Environment state;
- Financial results (account status);
- Memory.

In the current implementation of our data exchange interfaces between objects, only two information streams are allowed.

Similar to the low-level reflection module, the high-level reflection module uses the memory module as the parent class for the new object. This design allows the memory stream to be formed directly within the object, eliminating the need for additional input sources. As a result, the module becomes more autonomous and can efficiently process and store data within its own structure.

Additionally, the outputs of the high-level reflection module can likely be interpreted as a latent representation of the agent's action tensor. This is because the agent's actions are essentially a function of the results provided by our block. Consequently, changes in the analysis outcomes directly influence the adjustment of agent actions. This approach enables the creation of a more adaptive behavioral model, where the agent’s actions are dynamically optimized based on information obtained from high-level reflection.

So, using the above assumptions and architectural decisions, we construct a basic model of the new object's interfaces, operating with two sources of input data. These provide the necessary information for analyzing current conditions and the results of previous actions.

However, there is one more important aspect. To analyze the agent's overall behavioral policy, rather than focusing solely on individual trades, the _FinAgent_ framework authors suggest including an analysis of the account balance chart with markers for executed trades. This approach provides a generalized view of the agent's strategy outcomes. In our implementation, we reproduce this solution by adding recurrent objects for processing tensors representing account state and agent actions. These objects model the relationship between balance dynamics and trading actions, enabling a deeper analysis of the applied policy.

The solutions described above are implemented in the new object _CNeuronHighLevelReflection_. The structure of this object is shown below.

```
class CNeuronHighLevelReflection :  public CNeuronMemory
  {
protected:
   CNeuronBaseOCL    cAccount;
   CNeuronLSTMOCL    cHistoryAccount;
   CNeuronRelativeCrossAttention cActionReason;
   CNeuronLSTMOCL    cHistoryActions;
   CNeuronRelativeCrossAttention cActionResult;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override { return false; }
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override { return false; }
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput,
                                        CBufferFloat *SecondGradient,
                                        ENUM_ACTIVATION SecondActivation = None) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override {return false; }
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL,
                                        CBufferFloat *SecondInput) override;

public:
                     CNeuronHighLevelReflection(void) {};
                    ~CNeuronHighLevelReflection(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count, uint heads,
                          uint desc_account, uint actions_state,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   //---
   virtual int       Type(void) override   const   {  return defNeuronHighLevelReflection; }
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

The object structure includes the familiar set of overridable methods, providing flexibility and adaptability in functionality without disrupting the overall system architecture. In addition, the new class contains several internal objects, the functionality of which will be detailed during the implementation of the methods.

All internal objects are declared as static, allowing us to keep the class constructor and destructor empty. Initialization of all declared and inherited objects is performed in the _Init_ method.

```
bool CNeuronHighLevelReflection::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                      uint window, uint window_key, uint units_count,
                                      uint heads, uint desc_account, uint actions_state,
                                      ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronMemory::Init(numOutputs, myIndex, open_cl, 3, window_key, actions_state / 3,
                           heads, optimization_type, batch))
      return false;
```

The initialization method receives a set of constants that define the architecture of the object being created. One of these parameters specifies the dimensionality of the agent's action vector.

As noted above, the output of this object is expected to produce a latent representation of the agent's action tensor. Each trading operation of the agent is characterized by three parameters: trade volume, stop loss level, and take profit level. Buy and Sell trades are represented as separate tensor rows, eliminating the need for an additional parameter to indicate trade direction.

In the method body, we call the parent class method of the same name (in this case, we use the memory module). The parameters include the agent's action tensor data, structured according to the previously described assumptions. This approach maintains functional continuity and leverages the parent class capabilities for data handling. All these allows you to integrate the analysis results from the current object into the overall model structure, ensuring consistency and accessibility for subsequent objects.

After successfully executing the parent class operations, we initialize the nested objects. First, we initialize a basic fully connected layer, which is used to handle the second information stream correctly.

```
   int index = 0;
   if(!cAccount.Init(0, index, OpenCL, desc_account, optimization, iBatch))
      return false;
```

Next, we initialize the recurrent block for tracking account state changes.

```
   index++;
   if(!cHistoryAccount.Init(0, index, OpenCL, desc_account, 1, optimization, iBatch))
      return false;
```

To analyze the validity of the most recent trade decision, we use a cross-attention block to examine dependencies between agent actions and the environment state tensor.

```
   index++;
   if(!cActionReason.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits, iHeads,
                          window, units_count, optimization, iBatch))
      return false;
```

Action dynamics are collected in the corresponding recurrent block.

```
   index++;
   if(!cHistoryActions.Init(0, index, OpenCL, iWindow, iUnits, optimization, iBatch))
      return false;
```

We then evaluate the effectiveness of the agent's policy by comparing trading activity dynamics with account state within the cross-attention block.

```
   index++;
   if(!cActionResult.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits, iHeads,
                          desc_account, 1, optimization, iBatch))
      return false;;
//---
   return true;
  }
```

After initializing all internal objects, we return the logical result of the operations to the calling program and complete the method execution.

Next, we construct the feed-forward pass algorithm for the high-level reflection module within the _feedForward_ method.

```
bool CNeuronHighLevelReflection::feedForward(CNeuronBaseOCL *NeuronOCL, CBufferFloat *SecondInput)
  {
   if(!NeuronOCL || !SecondInput)
      return false;
```

The method receives pointers to the input objects of the two information streams. We immediately verify the pointer validity.

It should be noted that the second information stream is represented by a data buffer. Before further operations, we use the pointer to replace the buffer with a specially prepared internal object. This allows us to use the base interfaces of the internal objects to process the received data.

```
   if(cAccount.getOutput() != SecondInput)
     {
      if(cAccount.Neurons() != SecondInput.Total())
         if(!cAccount.Init(0, 0, OpenCL, SecondInput.Total(), optimization, iBatch))
            return false;
      if(!cAccount.SetOutput(SecondInput, true))
         return false;
     }
```

We then evaluate changes in account state using the recurrent block.

```
   if(!cHistoryAccount.FeedForward(cAccount.AsObject()))
      return false;
```

The validity of the most recent trade decision is checked against the current environment state using the cross-attention block.

```
   if(!cActionReason.FeedForward(this.AsObject(), NeuronOCL.getOutput()))
      return false;
```

It is important to distinguish between the tensor describing the current environment state and the tensor used for decision-making. The model receives the environment state tensor after the action has been executed and the system has transitioned to the new state. However, this difference is not expected to significantly affect analysis results.

Each new bar triggers a new iteration of the model. The depth of the analyzed history significantly exceeds one bar, providing detailed and comprehensive analysis. With each state transition, data shifts by one element in the multimodal time series, with the earliest bar excluded. This bar is expected to have minimal impact on the current action, so its removal will not have a noticeable effect on the analysis.

The inclusion of a new bar, unknown at the time the decision was made, provides extra information to assess the correctness of the trade. This mechanism helps evaluate both the consequences of the trade and the correspondence between actual market changes and prior forecasts.

The analysis results are passed to the recurrent block for monitoring the agent's policy.

```
   if(!cHistoryActions.FeedForward(cActionReason.AsObject()))
      return false;
```

We then analyze the policy in the context of financial results using the cross-attention block.

```
   if(!cActionResult.FeedForward(cHistoryActions.AsObject(), cHistoryAccount.getOutput()))
      return false;
```

Next, the result buffer is updated to preserve the agent's latest actions, enabling a proper backward pass, and the parent class method is called to perform memory module functions.

```
   if(!SwapBuffers(Output, PrevOutput))
      return false;
//---
   return CNeuronMemory::feedForward(cActionResult.AsObject());
  }
```

The logical result of these operations is returned to the calling program, and the method concludes.

After completing the implementation of the feed-forward pass methods, we proceed to organize the backpropagation pass. Backpropagation operations use linear algorithms, which should not cause difficulties in implementation. So, I think that detailed discussion is unnecessary. The full code for the high-level reflection object and all its methods is available in the attachments.

We have reached the end of this article, but our work related to the implementation of the _FinAgent_ framework is not yet complete. Let's take a short break, and in the next article, we will bring the project to its logical conclusion.

### Conclusion

In this article, we explored the _FinAgent_ framework, an innovative solution that integrates textual and visual information for comprehensive analysis of market dynamics and historical data. Using five key components, _FinAgent_ provides high accuracy and adaptability in trading decision-making. This makes it a promising tool for developing effective and flexible trading strategies capable of operating in volatile market conditions.

In the practical section, we implemented our version of the two reflection modules using _MQL5_. In the next article, we will continue this work by completing the framework and testing the effectiveness of the implemented solutions on real historical data.

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

Original article: [https://www.mql5.com/ru/articles/16850](https://www.mql5.com/ru/articles/16850)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16850.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16850/mql5.zip "Download MQL5.zip")(2327.7 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/498167)**
(2)


![Dominic Michael Frehner](https://c.mql5.com/avatar/2024/11/672504f5-a016.jpg)

**[Dominic Michael Frehner](https://www.mql5.com/en/users/cryptonist)**
\|
9 Jan 2025 at 13:19

It would be great if you could also [write articles](https://www.mql5.com/en/articles/408 "Article: New Article Publishing System at the MQL5.community ") in English.

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
21 Oct 2025 at 14:06

**[@Dominic Michael Frehner](https://www.mql5.com/en/users/cryptonist) [#](https://www.mql5.com/en/forum/498167#comment_58322692):** It would be great if you could also [write articles](https://www.mql5.com/en/articles/408 "Article: New Article Publishing System at the MQL5.community ") in English.

In which forum language did you post? The article is available in English too ... [Neural Networks in Trading: A Multimodal, Tool-Augmented Agent for Financial Markets (FinAgent)](https://www.mql5.com/en/articles/16850 "Click to change text")

This is a multilingual discussion topic, precisely because the article is translated into multiple languages.

![Introduction to MQL5 (Part 25): Building an EA that Trades with Chart Objects (II)](https://c.mql5.com/2/176/19968-introduction-to-mql5-part-25-logo__1.png)[Introduction to MQL5 (Part 25): Building an EA that Trades with Chart Objects (II)](https://www.mql5.com/en/articles/19968)

This article explains how to build an Expert Advisor (EA) that interacts with chart objects, particularly trend lines, to identify and trade breakout and reversal opportunities. You will learn how the EA confirms valid signals, manages trade frequency, and maintains consistency with user-selected strategies.

![Dynamic Swing Architecture: Market Structure Recognition from Swings to Automated Execution](https://c.mql5.com/2/176/19793-dynamic-swing-architecture-logo.png)[Dynamic Swing Architecture: Market Structure Recognition from Swings to Automated Execution](https://www.mql5.com/en/articles/19793)

This article introduces a fully automated MQL5 system designed to identify and trade market swings with precision. Unlike traditional fixed-bar swing indicators, this system adapts dynamically to evolving price structure—detecting swing highs and swing lows in real time to capture directional opportunities as they form.

![Market Simulation (Part 04): Creating the C_Orders Class (I)](https://c.mql5.com/2/112/Simulat6o_de_mercado_Parte_01_Cross_Order_I_LOGO.png)[Market Simulation (Part 04): Creating the C\_Orders Class (I)](https://www.mql5.com/en/articles/12589)

In this article, we will start creating the C\_Orders class to be able to send orders to the trading server. We'll do this little by little, as our goal is to explain in detail how this will happen through the messaging system.

![The MQL5 Standard Library Explorer (Part 2): Connecting Library Components](https://c.mql5.com/2/176/19834-the-mql5-standard-library-explorer-logo.png)[The MQL5 Standard Library Explorer (Part 2): Connecting Library Components](https://www.mql5.com/en/articles/19834)

Today, we take an important step toward helping every developer understand how to read class structures and quickly build Expert Advisors using the MQL5 Standard Library. The library is rich and expandable, yet it can feel like being handed a complex toolkit without a manual. Here we share and discuss an alternative integration routine—a concise, repeatable workflow that shows how to connect classes reliably in real projects.

[![](https://www.mql5.com/ff/si/mbxx5fzr169cx07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F498%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.buy.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=yiuacrhbffqmmulobpsgnypolteeimpt&s=949562ee5e6aca93c0231542844344e241ce4a26ab488f494b70624c190b74d7&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=ezsofrrhqrjepyprxvcrpedskpyynmwv&ssn=1769182178061152067&ssn_dr=0&ssn_sr=0&fv_date=1769182178&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16850&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20in%20Trading%3A%20A%20Multimodal%2C%20Tool-Augmented%20Agent%20for%20Financial%20Markets%20(FinAgent)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918217868149137&fz_uniq=5069486153360410082&sv=2552)

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