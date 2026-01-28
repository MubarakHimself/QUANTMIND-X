---
title: Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (FinCon)
url: https://www.mql5.com/en/articles/16916
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:29:10.723429
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=yydycdlbjielvecuxxfclfhfwtoluajf&ssn=1769182149023192199&ssn_dr=0&ssn_sr=0&fv_date=1769182149&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16916&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20in%20Trading%3A%20A%20Multi-Agent%20System%20with%20Conceptual%20Reinforcement%20(FinCon)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918214964886927&fz_uniq=5069478946405287365&sv=2552)

MetaTrader 5 / Trading systems


### Introduction

Financial markets are characterized by high volatility and complexity, creating significant challenges for making optimal investment decisions. Traders and portfolio managers must take into account a variety of multimodal data, such as macroeconomic indicators, technical signals, and behavioral factors of market participants. The primary goal of these efforts is to maximize returns while minimizing risks.

In traditional financial institutions, data processing and decision-making involve various specialists: analysts conduct market research, risk managers assess potential threats, and executives make strategic decisions. However, despite the hierarchical structure, human factors and limited resources often hinder the ability to quickly adapt to rapid market changes. As a result, the use of automated systems has become increasingly relevant: systems that not only accelerate the analytical process but also reduce the likelihood of human error.

Modern research in artificial intelligence and financial technology focuses on developing adaptive software solutions. Such systems can learn from historical data, identify market patterns, and make more informed decisions. One of the most promising recent directions is the integration of natural language processing ( _NLP_) methods, which enable the analysis of financial news, expert forecasts, and other text-based data to improve prediction accuracy and risk assessment.

The effectiveness of such systems largely depends on two key aspects: interaction among system components and their capacity for continuous self-learning. Studies have shown that systems modeling collaborative teamwork among specialists demonstrate superior performance, and with the adoption of new approaches, these models are becoming increasingly adaptable to changing market conditions.

Existing solutions, such as _[FinMem](https://www.mql5.com/en/articles/16804)_ and _[FinAgent](https://www.mql5.com/en/articles/16850)_, demonstrate significant progress in automating financial operations. However, these systems also have limitations: they tend to focus on short-term market dynamics and often lack comprehensive mechanisms for long-term risk management. Moreover, constraints on computational resources and limited algorithmic flexibility can reduce the quality of their recommendations.

These challenges are addressed in the paper " _[FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://www.mql5.com/go?link=https://arxiv.org/abs/2407.06567 "https://arxiv.org/abs/2407.06567")_". The authors propose _FinCon_, a multi-agent system specifically designed to integrate stock trading and portfolio management processes.

The _FinCon_ framework simulates the workflow of a professional investment team. Analyst agents gather and analyze data from various sources, including market indicators, news feeds, and historical data, while manager agents synthesize these insights and make final decisions. This approach minimizes redundant communication among participants and optimizes computational resource usage.

The authors designed _FinCon_ to operate both with individual financial assets and diversified portfolios. This makes the system highly versatile.

_FinCon_ performs real-time risk assessments using analytical models and machine learning algorithms. After each trading episode, the system conducts a post-operation analysis, identifying errors and updating its models based on newly acquired data.

Experimental results demonstrate that _FinCon_ significantly improves risk management and enhances overall portfolio performance.

### FinCon Architecture

The _FinCon_ architecture features a two-tier hierarchy consisting of two main components: the Manager–Analyst agent group, and the Risk Control subsystem. This structure ensures efficient information processing, reduces error probability, minimizes decision-making costs, and enables deep market analysis.

The _FinCon_ framework resembles a well-organized investment firm, where all resources are optimized for maximum efficiency. Its main objective is to improve information perception and analysis while minimizing communication and data-processing overhead.

Analyst agents play a key role in _FinCon_ by extracting primary investment ideas from massive volumes of market data. Each agent has a narrow specialization, preventing data duplication and cognitive overload. In the reference implementation, the framework uses seven analyst agents. Text agents analyze news articles, press releases, and financial reports to identify potential risks and opportunities. Audio agents interpret recorded earnings calls from company executives, detecting emotional nuances and critical discussion points. Data analysis and stock selection agents calculate quantitative metrics, enabling the manager to forecast market trends with high precision.

The analyst agents provide their outputs to the manager agent, the sole decision-making entity. For trading decisions, the manager performs 4 key functions: consolidating analytical results, monitoring risk in real time, continuously reviewing past decisions, and refining its investment beliefs.

_FinCon_ implements a two-level risk control mechanism. Intra-episode risk assessment allows for real-time corrective actions to minimize short-term losses. Inter-episode risk evaluation compares results across episodes to identify mistakes and improve strategies. This dual-layered approach ensures system resilience to external changes and promotes its continuous improvement.

Updating investment beliefs plays a critical role in model adaptation. It allows FinCon to adjust the focus of its analyst agents during data extraction and refine the manager's decision-making logic. The _Actor–Critic_ mechanism enables _FinCon_ to periodically optimize its trading strategy based on the given trading goals. This approach uses the analysis of both successful and unsuccessful actions, contributing to ongoing refinement of its decision-making policies.

_FinCon_'s episodic reflection is powered by a unique mechanism known as Conceptual Verbal Reinforcement ( _CVRF_). This component evaluates the effectiveness of sequential trading episodes by comparing analysts' insights with the manager's decisions. _CVRF_ links key findings to specific strategic aspects of trading. By contrasting conceptual insights from more and less successful episodes, the model generates recommendations for belief adjustment. These insights help agents focus on the most relevant market information, thereby improving overall portfolio performance.

Belief adjustment recommendations are first delivered to the manager, then selectively shared with analysts. This minimizes redundant communication and prevents information overload. The method measures the proportion of overlapping trading actions between consecutive learning trajectories, enhancing system efficiency. This is particularly noticeable in environments where each agent has a clearly defined, specialized role, promoting synergy across the model.

_FinCon_ features an advanced memory module divided into three key components: working memory, procedural memory, and episodic memory.

_Working memory_ temporarily stores data needed for ongoing operations. This enables rapid processing of large information volumes without losing context.

_Procedural memory_ retains algorithms and strategies successfully applied in previous episodes. This memory enables fast adaptation to recurring tasks and the reuse of proven methods.

_Episodic memory_ records key events, actions, and outcomes, which are especially important for high-level policy refinement. This memory type plays a key role in the model's learning process, helping it leverage past successes and failures to improve future performance.

The layered structure of _FinCon_ makes it a highly effective and adaptive system. The original visualization of the framework is provided below.

![Author's visualization of the FinCon framework](https://c.mql5.com/2/110/multi_agent_Architecturei1c.jpg)

### Implementation in MQL5

After exploring the theoretical aspects of the _FinCon_ framework, we now move to the practical section of our article, where we implement our interpretation of the proposed concepts using _MQL5_.

It is important to note that the original _FinCon_ framework is built upon the use of pre-trained large language models (LLMs). In our projects, we do not use language models. Instead, we implement the proposed principles using only the tools available within the MQL5 environment. So, let's begin by modernizing the memory module.

#### Approaches to Memory Module Modernization

As you may recall, our previously developed [memory block](https://www.mql5.com/en/articles/16804#para31) consists of two recurrent modules with different architectural designs. This design enables the creation of a two-level memory system with varying rates of information decay, making it more flexible and adaptive for diverse tasks. This approach is particularly useful in situations where both short-term and long-term dependencies need to be taken into account.

Note that in our model, the decay rate is not explicitly defined. Instead, it emerges naturally due to architectural differences between the recurrent modules. One module focuses on short-term memory for rapid information updates. The other is responsible for long-term dependencies, retaining important data over extended periods. These architectural distinctions create a natural balance between processing speed and analytical depth.

Information within the memory block is stored in the hidden states of the recurrent modules. This approach allows for a significant reduction in memory usage, but it also introduces challenges when it becomes necessary to extract and compare specific episodes. This capability, however, is critical for implementing the conceptual verification mechanism, which requires the possibility to compare the current context to previously stored episodes.

Furthermore, the authors of the _FinCon_ framework highlight the importance of employing a three-level memory system for more precise analysis and prediction. Consequently, it becomes necessary to upgrade the existing memory block by introducing an additional episodic memory layer.

The process of optimizing the memory module involves several important steps. First of all, it minimizes computational resource usage. This can be achieved by integrating data compression algorithms. Such algorithms eliminate redundant information, reducing memory requirements per episode while preserving essential characteristics of the data.

Another important task is to ensure fast and accurate access to relevant information. To achieve this, it is advisable to use vector similarity algorithms. These algorithms enable the system to quickly locate episodes most similar to the current one — a critical feature for real-time applications.

As a result, the modernized memory block will become a cornerstone of the model's improved overall efficiency. It will ensure not only compact data storage, but also rapid access to relevant information, significantly enhancing the decision-making process.

#### Implementation of the New Memory Module

We implement the proposed methods within a new object called _CNeuronMemoryDistil_. Its structure is presented below.

```
class CNeuronMemoryDistil  :  public CNeuronMemory
  {
protected:
   CNeuronBaseOCL       cConcatenated;
   CNeuronTransposeOCL  cTransposeConc;
   CNeuronConvOCL       cConvolution;
   CNeuronEmbeddingOCL  cDistilMemory;
   CNeuronRelativeCrossAttention cCrossAttention;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronMemoryDistil(void) {};
                    ~CNeuronMemoryDistil(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key, uint units_count,
                          uint heads, uint stack_size,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override
 const   {  return defNeuronMemoryDistil; }
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

In our implementation, the parent class is the previously developed [memory object](https://www.mql5.com/en/articles/16804#para31), from which we inherit all core interfaces as well as the two recurrent modules for short-term and long-term memory. The internal objects declared within our new class form the foundation for episodic memory. This allows the storage and processing of data corresponding to specific episodes. Each object performs a specialized role within the overall memory structure, enabling more precise and comprehensive data analysis. Their functionality will be described in detail during the implementation of the memory module methods.

All internal objects are declared statically, allowing us to leave the constructor and destructor empty. This approach optimizes memory usage and enhances system stability by minimizing the risk of errors during object creation or destruction.

Initialization of both inherited and newly declared objects occurs within the _Init_ method. The method parameters include a set of constants that define the architecture of the object and provide the necessary flexibility to configure the model for specific tasks while maintaining integrity and functionality.

```
bool CNeuronMemoryDistil::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint window, uint window_key, uint units_count,
                               uint heads, uint stack_size,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronRelativeCrossAttention::Init(numOutputs, myIndex, open_cl, window,
                                         window_key, units_count, heads, window,
                                       2 * stack_size, optimization_type, batch))
      return false;
```

Within the method body, we typically call the parent class Init method to initialize inherited objects, passing the corresponding parameters. However, due to the unique characteristics of our new object, the base memory module's initialization method cannot be directly used. Instead, we employ an analogous method from the cross-attention object, which serves as the parent for the base memory module. This ensures consistency and proper functioning across all elements of the new architecture.

Looking ahead, the output of our memory module is expected to analyze the current state within the context of episodic memory. Here, the cross-attention block plays a critical role, performing two primary functions. The first one is to find the most relevant episodes. To implement this task, the framework uses vector similarity algorithms, which underpin the attention mechanism's dependency coefficients.

The second function of the cross-attention block is to enrich the original state with context from selected episodes. This creates a more complete and informative representation, improving data processing and decision accuracy.

During initialization, we pass the source data parameters to the cross-attention object, and the context sequence length is set to twice the episodic memory buffer size. This accommodates storing results from both short-term and long-term memory modules.

After successfully initializing the cross-attention object, we call the initialization methods of the inherited short-term and long-term recurrent modules. We take this process from the parent class method.

```
   uint index = 0;
   if(!cLSTM.Init(0, index, OpenCL, iWindow, iUnits, optimization, iBatch))
      return false;
   index++;
   if(!cMamba.Init(0, index, OpenCL, iWindow, 2 * iWindow, iUnits, optimization,
                                                                        iBatch))
      return false;
```

The outputs of these modules are concatenated into a single buffer, enabling simultaneous access to both memory types. A base object of sufficient size is created for this purpose.

```
   index++;
   if(!cConcatenated.Init(0, index, OpenCL, 2 * iWindow * iUnits, optimization,
                                                                       iBatch))
      return false;
   cConcatenated.SetActivationFunction(None);
```

Next, the concatenated information is compressed and integrated into the episodic memory storage object.

Since we are working with a multimodal time series representing a single environmental state, it is important to preserve key characteristics of unit sequences during compression. A two-stage compression approach is implemented. The first step is the preliminary compression of individual unit sequences to extract essential features and reduce data volume without loss of integrity.

The concatenated tensor is transposed to represent data as univariate sequences.

```
   index++;
   if(!cTransposeConc.Init(0, index, OpenCL, iUnits, 2 * iWindow, optimization,
                                                                       iBatch))
      return false;
```

Data are then compressed using a convolutional layer.

```
   index++;
   if(!cConvolution.Init(0, index, OpenCL, iUnits, iUnits, iWindowKey, iWindow,
                                                      1, optimization, iBatch))
      return false;
   cConvolution.SetActivationFunction(GELU);
```

To store episodic memory, we apply an [embedding](https://www.mql5.com/en/articles/13347#para3) layer, where partially compressed states are projected into a compact latent representation and added to a fixed-length _FIFO_ ( _First In, First Out_) memory stack.

Each tensor block is projected into the latent space using trainable projection matrices. These matrices transform multimodal data from the raw format into a unified representation. This approach simplifies further analysis and ensures data consistency when integrating information from different sources. In this case, these are short-term and long-term memory modules.

```
   index++;
   uint windows[] = {iWindowKey * iWindow, iWindowKey * iWindow};
   if(!cDistilMemory.Init(0, index, OpenCL, iUnitsKV / 2, iWindow, windows))
      return false;
```

As mentioned earlier, the analysis of raw data in the context of episodic memory is performed using the cross-attention base object. For a more comprehensive analysis, an additional cross-attention object is introduced to process source data within the context of short-term and long-term memory. This approach helps capture synergistic effects across memory types.

The previously created concatenation object simplifies this integration. As a result, a single cross-attention object can enrich data while accounting for both memory modules. This optimizes computational resources and enhances analysis precision.

```
   index++;
   if(!cCrossAttention.Init(0, index, OpenCL, iWindow, iWindowKey, iUnits, iHeads,
                                       iWindow, 2 * iUnits, optimization, iBatch))
      return false;
//---
   return true;
  }
```

After all internal objects are initialized, the method returns a logical result to the calling program.

Next, we move to constructing the feed-forward pass for the our new block within the _feedForward_ method.

```
bool CNeuronMemoryDistil::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cLSTM.FeedForward(NeuronOCL))
      return false;
   if(!cMamba.FeedForward(NeuronOCL))
      return false;
```

In the method parameters, a pointer to the source data object is received and immediately passed to the corresponding short-term and long-term recurrent module methods. Their outputs are concatenated into a single data buffer.

```
   if(!Concat(cLSTM.getOutput(), cMamba.getOutput(), cConcatenated.getOutput(),
                                                     iWindow, iWindow, iUnits))
      return false;
   if(!cTransposeConc.FeedForward(cConcatenated.AsObject()))
      return false;
```

Note that we perform data concatenation in terms of individual time steps. Such step-wise concatenation preserves the structure of all unit sequences.

The concatenated tensor is transposed and compressed with a convolutional layer.

```
   if(!cConvolution.FeedForward(cTransposeConc.AsObject()))
      return false;
```

The compressed representation is projected via the embedding layer into episodic memory.

```
   if(!cDistilMemory.FeedForward(cConvolution.AsObject()))
      return false;
```

At this stage, we have saved the required amount of information in three levels of our memory block. Next, we need to enrich the current state of the environment with the context of key events. First, we analyze source data in the context from both short-term/long-term memory.

```
   if(!cCrossAttention.FeedForward(NeuronOCL, cConcatenated.getOutput()))
      return false;
```

Then we enrich data using the context from episodic memory.

```
   return CNeuronRelativeCrossAttention::feedForward(cCrossAttention.AsObject(),
                                                     cDistilMemory.getOutput());
  }
```

The logical result is returned to the calling program.

After implementing the feed-forward method, we proceed to the backpropagation algorithms. The backpropagation pass consists of two key methods:

- _calcInputGradients_ — propagates error gradients.
- _updateInputWeights_ — updates model parameters.

The data flow in the error gradient distribution method mirrors the feed-forward pass algorithm in reverse order.

In the parameters of the _calcInputGradients_ method, we receive a pointer to the same source data object as in the feed-forward pass. However, this time we propagate the error gradient corresponding to the data influence on the model's final output.

```
bool CNeuronMemoryDistil::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
```

At the beginning of the method, we immediately verify the validity of the received pointer, as if it is invalid, further operations within the method become meaningless.

The feed-forward pass concluded with a call to the cross-attention parent class method. Accordingly, the error gradient distribution process begins with the corresponding method of the same object. In this stage, the gradient is propagated through the information flows of the memory hierarchy.

```
  if(!CNeuronRelativeCrossAttention::calcInputGradients(cCrossAttention.AsObject(),
                                                         cDistilMemory.getOutput(),
                                                       cDistilMemory.getGradient(),
                                      (ENUM_ACTIVATION)cDistilMemory.Activation()))
      return false;
```

The episodic memory gradient is passed through the data compression layers down to the level of the short-term and long-term memory concatenation object.

```
   if(!cConvolution.calcHiddenGradients(cDistilMemory.AsObject()))
      return false;
   if(!cTransposeConc.calcHiddenGradients(cConvolution.AsObject()))
      return false;
   if(!cConcatenated.calcHiddenGradients(cTransposeConc.AsObject()))
      return false;
```

However, this concatenation object is also used for analyzing input data within the context of both memory types. Therefore, the error gradient must be propagated to it via the second information flow as well.

```
   if(!NeuronOCL.calcHiddenGradients(cCrossAttention.AsObject(),
                                     cConcatenated.getOutput(),
                                     cConcatenated.getPrevOutput(),
                                     (ENUM_ACTIVATION)cConcatenated.Activation()))
      return false;
```

It is important to note that, in this case, we use the free buffer of the concatenation object rather than its specialized one to obtain the error gradient. This approach preserves previously computed data, avoiding overwriting valuable information.

Next, we sum the values obtained from both information flows and distribute the resulting gradients among the corresponding memory objects.

```
   if(!SumAndNormilize(cConcatenated.getGradient(), cConcatenated.getPrevOutput(),
                       cConcatenated.getGradient(), iWindow, false, 0, 0, 0, 1) ||
      !DeConcat(cLSTM.getGradient(), cMamba.getGradient(),
                cConcatenated.getGradient(), iWindow, iWindow, iUnits))
      return false;
```

Here, it is crucial to point out that we intentionally avoid applying an activation function to the concatenation object in order to prevent data distortion. That's because the short-term and long-term memory modules may use different activation functions. However, after distributing the error gradients among the memory modules, we must check whether each object includes an activation function and, if so, adjust the gradient values using the derivatives of those activation functions.

```
   if(cLSTM.Activation() != None)
      if(!DeActivation(cLSTM.getOutput(), cLSTM.getGradient(),
                       cLSTM.getGradient(), cLSTM.Activation()))
         return false;
   if(cMamba.Activation() != None)
      if(!DeActivation(cMamba.getOutput(), cMamba.getGradient(),
                       cMamba.getGradient(), cMamba.Activation()))
         return false;
```

At this point, we proceed to propagate the error gradient down through the short-term and long-term memory pipelines to the level of the original input data. However, recall that the gradient buffer of this object already contains data obtained during the context-level information analysis. To preserve these previously computed values, we must temporarily swap the buffer pointers before performing further operations.

```
   CBufferFloat *temp = NeuronOCL.getGradient();
   if(!NeuronOCL.SetGradient(cConcatenated.getPrevOutput(), false) ||
      !NeuronOCL.calcHiddenGradients(cLSTM.AsObject()) ||
      !SumAndNormilize(temp, NeuronOCL.getGradient(), temp, iWindow,
                                                  false, 0, 0, 0, 1))
      return false;
```

Once this substitution is complete, we sequentially propagate the error gradients through the information flows of the corresponding memory modules, adding the new values to the previously accumulated data.

```
   if(!NeuronOCL.calcHiddenGradients(cMamba.AsObject()) ||
      !SumAndNormilize(temp, NeuronOCL.getGradient(), temp, iWindow,
                                               false, 0, 0, 0, 1) ||
      !NeuronOCL.SetGradient(temp, false))
      return false;
//---
   return true;
  }
```

After this step, we restore the buffer pointers to their original state. We return the logical result of the operations to the calling program, and complete the execution of the method.

This concludes our examination of the methods implemented in the modernized memory block _CNeuronMemoryDistil_. The full code of this object and all its methods is provided in the attachment for further study.

The next stage of our work involves constructing the Analyst Agent object. However, since we are nearing the article format limit, we will take a short break and continue this work in the next part.

### Conclusion

In this article, we explored the _FinCon_ framework, which is an innovative multi-agent system designed to enhance financial decision-making. The framework integrates a hierarchical "manager–analyst" interaction structure with conceptual verbal reinforcement mechanisms, enabling coordinated agent collaboration and effective risk management. These features allow the model to successfully handle a wide range of financial tasks, demonstrating high adaptability and performance in a dynamic market environment.

In the practical section, we began implementing the proposed approaches using _MQL5_. This work will continue in the next article, where we will evaluate the performance and efficiency of the implemented solutions using real historical market data.

#### References

- [FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making](https://www.mql5.com/go?link=https://arxiv.org/abs/2407.06567 "FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making")
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

Original article: [https://www.mql5.com/ru/articles/16916](https://www.mql5.com/ru/articles/16916)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16916.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16916/mql5.zip "Download MQL5.zip")(2352.32 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/499116)**

![Self Optimizing Expert Advisors in MQL5 (Part 16): Supervised Linear System Identification](https://c.mql5.com/2/178/20023-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 16): Supervised Linear System Identification](https://www.mql5.com/en/articles/20023)

Linear system identifcation may be coupled to learn to correct the error in a supervised learning algorithm. This allows us to build applications that depend on statistical modelling techniques without necessarily inheriting the fragility of the model's restrictive assumptions. Classical supervised learning algorithms have many needs that may be supplemented by pairing these models with a feedback controller that can correct the model to keep up with current market conditions.

![From Novice to Expert: Revealing the Candlestick Shadows (Wicks)](https://c.mql5.com/2/178/19919-from-novice-to-expert-revealing-logo.png)[From Novice to Expert: Revealing the Candlestick Shadows (Wicks)](https://www.mql5.com/en/articles/19919)

In this discussion, we take a step forward to uncover the underlying price action hidden within candlestick wicks. By integrating a wick visualization feature into the Market Periods Synchronizer, we enhance the tool with greater analytical depth and interactivity. This upgraded system allows traders to visualize higher-timeframe price rejections directly on lower-timeframe charts, revealing detailed structures that were once concealed within the shadows.

![Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (Final Part)](https://c.mql5.com/2/111/Neural_Networks_in_Trading____FinCon____LOGO2__1.png)[Neural Networks in Trading: A Multi-Agent System with Conceptual Reinforcement (Final Part)](https://www.mql5.com/en/articles/16937)

We continue to implement the approaches proposed by the authors of the FinCon framework. FinCon is a multi-agent system based on Large Language Models (LLMs). Today, we will implement the necessary modules and conduct comprehensive testing of the model on real historical data.

![Introduction to MQL5 (Part 27): Mastering API and WebRequest Function in MQL5](https://c.mql5.com/2/178/17774-introduction-to-mql5-part-27-logo.png)[Introduction to MQL5 (Part 27): Mastering API and WebRequest Function in MQL5](https://www.mql5.com/en/articles/17774)

This article introduces how to use the WebRequest() function and APIs in MQL5 to communicate with external platforms. You’ll learn how to create a Telegram bot, obtain chat and group IDs, and send, edit, and delete messages directly from MT5, building a strong foundation for mastering API integration in your future MQL5 projects.

[![](https://www.mql5.com/ff/si/d9hnbkyp2d47h07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fsignals%2Fmt5%2Fpage1%3Fpreset%3D2%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dmax.profit.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=hgyovyikvykcdukcncnktswvlctghemf&s=545653d14172edfb3c9c02ca8e948778c29f9c1b70be9a587e8d4b040fb23539&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=awctrvzhkkqwvythwmyvjntzbklznijy&ssn=1769182149023192199&ssn_dr=0&ssn_sr=0&fv_date=1769182149&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16916&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20in%20Trading%3A%20A%20Multi-Agent%20System%20with%20Conceptual%20Reinforcement%20(FinCon)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918214964798270&fz_uniq=5069478946405287365&sv=2552)

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