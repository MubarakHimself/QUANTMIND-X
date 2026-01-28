---
title: Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model
url: https://www.mql5.com/en/articles/17142
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:26:57.664927
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/17142&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069447344035923263)

MetaTrader 5 / Trading systems


### Introduction

The rapid development of artificial intelligence has led to the active integration of deep learning methods into data analysis, including the financial sector. Financial inputs are characterized by high dimensionality, heterogeneity, and temporal structure, which complicates the application of traditional processing methods. At the same time, deep learning has demonstrated high efficiency in analyzing complex and unstructured data.

Among modern convolutional architectures, there's the one that stands out: _ResNeXt_ introduced in " _[Aggregated Residual Transformations for Deep Neural Networks](https://www.mql5.com/go?link=https://arxiv.org/abs/1611.05431 "https://arxiv.org/abs/1611.05431")_". _ResNeXt_ is capable of capturing both local and global dependencies and effectively handling multidimensional data while reducing computational complexity through grouped convolutions.

A key area of financial analysis using deep learning is multi-task learning ( _MTL_). This approach allows simultaneous solutions to multiple related tasks, improving model accuracy and generalization capability. Unlike classical approaches where each model addresses a single task, MTL leverages shared data representations, making the model more robust to market fluctuations and enhancing the training process. This approach is particularly valuable for market trend forecasting, risk assessment, and asset valuation, as financial markets are dynamic and influenced by numerous factors.

The study " _[Collaborative Optimization in Financial Data Mining Through Deep Learning and ResNeXt](https://www.mql5.com/go?link=https://arxiv.org/abs/2412.17314 "https://arxiv.org/abs/2412.17314")_" introduced a framework for integrating the _ResNeXt_ architecture into multi-task models. This solution opens new possibilities for processing time series, identifying spatiotemporal patterns, and generating accurate forecasts. _ResNeXt_'s grouped convolutions and residual blocks accelerate training and reduce the risk of losing critical features, making this method especially relevant for financial analysis.

Another significant advantage of the proposed approach is the automation of extracting meaningful features from raw data. Traditional financial analysis often requires extensive feature engineering, whereas deep neural networks can autonomously identify key patterns. This is particularly important when analyzing multimodal financial data, which involves multiple sources such as market indicators, macroeconomic reports, and news publications. MTL's flexibility allows dynamic adjustment of task weights and loss functions, improving the model's adaptability to market changes and improving forecasting accuracy.

### ResNeXt Architecture

The _ResNeXt_ architecture is based on a modular design and grouped convolutions. At its core are convolutional blocks with residual connections, governed by two key principles:

- If the output feature maps have the same size, blocks use identical hyperparameters (width and filter sizes).
- If the feature map size is reduced, the block width is proportionally increased.

Adhering to these principles maintains roughly constant computational complexity across all model layers, simplifying design. It is enough to defining a single template module, while other blocks are generated automatically, ensuring standardization, easier tuning, and streamlined architectural analysis.

Standard neurons in artificial neural networks perform a weighted sum of inputs, the primary operation in convolutional and fully connected layers. This process can be divided into three stages: splitting, transformation, and aggregation. _ResNeXt_ introduces a more flexible approach, allowing transformation functions to be more complex or even mini-models themselves. This gives rise to the _Network-in-Neuron_ concept, expanding the architecture's capabilities through a new dimension: _cardinality_. Unlike width or depth, cardinality determines the number of independent, complex transformations within each block. Experiments show increasing cardinality can be more effective than increasing depth or width, providing a better balance between performance and computational efficiency.

All _ResNeXt_ blocks share a uniform _bottleneck_ structure. It consists of:

- An initial 1×1 convolutional layer that reduces feature dimensionality,
- A main 3×3 convolutional layer performing core data processing,
- A final 1×1 convolutional layer restoring the original dimensionality.

This design reduces computational complexity while maintaining high model expressiveness. Additionally, residual connections preserve gradients during training, preventing vanishing gradients, which is a key factor for deep networks.

A major enhancement in _ResNeXt_ is the use of _Grouped Convolutions_. Here, input data are divided into multiple independent groups, each processed by separate convolutional filters, with the results subsequently aggregated. This reduces model parameters, maintains high network throughput, and improves computational efficiency without significant accuracy loss.

To maintain stable computational complexity when changing the number of groups, _ResNeXt_ adjusts the width of _bottleneck_ blocks, controlling the number of channels in internal layers. This allows scalable models without excessive computational overhead.

The multi-task learning framework based on _ResNeXt_ represents a progressive approach to financial data processing, addressing shared feature utilization and cooperative modeling across various analytical tasks. It consists of three key structural components:

- Feature extraction module,
- Shared learning module,
- Task-specific output layers.

This approach integrates efficient deep learning mechanisms with financial time series, delivering high forecasting accuracy and adapting the model to dynamic market conditions.

The feature extraction module relies on the _ResNeXt_ architecture, effectively capturing both local and global characteristics of financial data. In multidimensional financial data, the critical parameter is the number of groups in the model. It balances detailed feature representation with computational cost. Each grouped convolution in _ResNeXT_ identifies specific patterns in channel groups, which are then aggregated into a unified representation.

After passing through nonlinear transformation layers, extracted features form the basis for subsequent multi-task learning and task-specific adaptation. The shared learning module employs a weight-sharing mechanism that projects common features into task-specific spaces. This ensures the model can generate individual representations for each task while avoiding interference and simultaneously achieving high feature-sharing efficiency. Clustering tasks based on correlations further enhances the shared learning mechanism.

Task-specific output layers consist of fully connected perceptrons that project specialized features into the final prediction space. Output layers can adapt to the nature of each task. In particular, classification tasks use cross-entropy loss, while regression tasks use mean squared error ( _MSE_). For multi-task learning, the final loss function is represented as a weighted sum of individual task losses.

Training occurs in multiple stages. Initially, models are pre-trained on individual tasks to ensure effective convergence of specialized _MLPs_. The model is then fine-tuned in the multi-task architecture to improve overall performance. Optimization is performed using the _Adam_ algorithm with dynamic learning rate adjustment.

### Implementation Using MQL5

After reviewing the theoretical aspects of the _ResNeXt_-based multi-task learning framework, we proceed to implement our interpretation using _MQL5_. We'll begin with the construction of the basic _Res_ _NeXt_ architectural blocks - _bottleneck modules_.

#### Bottleneck Module

The _bottleneck_ module consists of three convolutional layers, each performing a critical role in processing raw data. The first layer reduces feature dimensionality, lowering the computational complexity of subsequent processing.

The second convolutional layer performs the main convolution, extracting complex high-level features necessary for accurate interpretation of the data. It analyzes interdependencies among elements, identifying patterns critical for later stages. This approach enables the model to adapt to nonlinear dependencies in financial data, improving forecast accuracy.

The final layer restores the original tensor dimensionality, preserving all significant information. Dimensionality reduction along the temporal axis in earlier feature extraction is compensated by expanding the feature space, consistent with _ResNeXt_ principles.

To stabilize training, each convolutional layer is followed by batch normalization. It reduces internal covariate shift and accelerates convergence. _ReLU_ activation enhances model nonlinearity, improving its ability to capture complex dependencies and generalization quality.

The above architecture is implemented within the _CNeuronResNeXtBottleneck_ object that has the following structure.

```
class CNeuronResNeXtBottleneck :  public CNeuronConvOCL
  {
protected:
   CNeuronConvOCL          cProjectionIn;
   CNeuronBatchNormOCL     cNormalizeIn;
   CNeuronTransposeRCDOCL  cTransposeIn;
   CNeuronConvOCL          cFeatureExtraction;
   CNeuronBatchNormOCL     cNormalizeFeature;
   CNeuronTransposeRCDOCL  cTransposeOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResNeXtBottleneck(void){};
                    ~CNeuronResNeXtBottleneck(void){};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint window,
                          uint step, uint units_count, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResNeXtBottleneck;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual CLayerDescription* GetLayerInfo(void) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
```

As the parent class, we use a convolutional layer object, which performs the function of restoring the feature space. Additionally, the structure includes a number of internal objects, each assigned a key role in the algorithms we are constructing. Their functionality will be detailed as we build the methods of the new class.

All internal objects are declared as static, allowing us to keep the class constructor and destructor empty. The initialization of these declared and inherited objects is performed in the _Init_ method. This method receives a set of constants that unambiguously define the architecture of the module being created.

```
bool CNeuronResNeXtBottleneck::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                   uint chanels_in, uint chanels_out, uint window,
                                   uint step, uint units_count, uint group_size, uint groups,
                                   ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   int units_out = ((int)units_count - (int)window + (int)step - 1) / (int)step + 1;
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, group_size * groups, group_size * groups,
                                              chanels_out, units_out, 1, optimization_type, batch))
      return false;
```

In the method body, we usually call the parent class method having the same name, which contains initialization algorithms for inherited objects and interfaces. However, in this case, the parent class functions as the final convolutional layer of the module. Its input receives data after feature extraction, during which the dimensionality of the processed tensor may have changed. Therefore, we first determine the sequence length at the module output and only then call the parent class method.

After successfully initializing inherited objects, we proceed to the newly declared objects. Work begins with the data projection block. The first convolutional layer prepares projections of the raw data for the required number of working groups.

```
//--- Projection In
   uint index = 0;
   if(!cProjectionIn.Init(0, index, OpenCL, chanels_in, chanels_in, group_size * groups,
                                                    units_count, 1, optimization, iBatch))
      return false;
   index++;
   if(!cNormalizeIn.Init(0, index, OpenCL, cProjectionIn.Neurons(), iBatch, optimization))
      return false;
   cNormalizeIn.SetActivationFunction(LReLU);
```

These projections are then normalized and activated using the _LReLU_ function.

Note that the result of these operations is a three-dimensional tensor \[ _Time, Group, Dimension_\]. To implement independent processing of individual groups, we move the group identifier dimension to the first position using a 3D tensor transposition object.

```
   index++;
   if(!cTransposeIn.Init(0, index, OpenCL, units_count, groups, group_size, optimization, iBatch))
      return false;
   cTransposeIn.SetActivationFunction((ENUM_ACTIVATION)cNormalizeIn.Activation());
```

Next is the feature extraction block. Here, we use a convolutional layer specifying the number of groups as the number of independent sequences. This ensures that the values of individual groups are not "mixed". Each group uses its own matrix of trainable parameters.

```
//--- Feature Extraction
   index++;
   if(!cFeatureExtraction.Init(0, index, OpenCL, group_size * window, group_size * step, group_size,
                                                           units_out, groups, optimization, iBatch))
      return false;
```

Additionally, note that the method receives parameters for the convolution window size and its step along the temporal dimension. When passing these parameters to the internal convolutional layer initialization, we multiply the corresponding values by the group size.

After the convolutional layer, we add batch normalization with the _LReLU_ activation function.

```
   index++;
   if(!cNormalizeFeature.Init(0, index, OpenCL, cFeatureExtraction.Neurons(), iBatch, optimization))
      return false;
   cNormalizeFeature.SetActivationFunction(LReLU);
```

The final feature space backward projection block consists solely of a 3D tensor transposition object that merges the groups into a single sequence. The actual data projection, as mentioned earlier, is performed using the inherited methods of the parent class.

```
//--- Projection Out
   index++;
   if(!cTransposeOut.Init(0, index, OpenCL, groups, units_out, group_size, optimization, iBatch))
      return false;
   cTransposeOut.SetActivationFunction((ENUM_ACTIVATION)cNormalizeFeature.Activation());
//---
   return true;
  }
```

We now just need to return the logical result of the operations to the caller and complete the object initialization method.

The next step is building the feed-forward pass algorithm, implemented in the _feedForward_ method.

```
bool CNeuronResNeXtBottleneck::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Projection In
   if(!cProjectionIn.FeedForward(NeuronOCL))
      return false;
```

The method receives a pointer to the raw data object, which is immediately passed to the method of the same name in the first internal convolutional layer of the data projection block. We do not check the validity of the pointer, as this verification is already handled within the internal layer, making an additional check redundant.

Projection results are normalized and transposed into separate group representations.

```
   if(!cNormalizeIn.FeedForward(cProjectionIn.AsObject()))
      return false;
   if(!cTransposeIn.FeedForward(cNormalizeIn.AsObject()))
      return false;
```

In the feature extraction block, grouped convolution operations are performed, and the results are normalized.

```
//--- Feature Extraction
   if(!cFeatureExtraction.FeedForward(cTransposeIn.AsObject()))
      return false;
   if(!cNormalizeFeature.FeedForward(cFeatureExtraction.AsObject()))
      return false;
```

Extracted features from individual groups are transposed back into a single multidimensional sequence and projected into the designated feature space using the parent class.

```
//--- Projection Out
   if(!cTransposeOut.FeedForward(cNormalizeFeature.AsObject()))
      return false;
   return CNeuronConvOCL::feedForward(cTransposeOut.AsObject());
  }
```

The logical result of these operations is returned to the calling program, and the method concludes.

As you can see, the feed-forward algorithm is linear. Error gradients propagate linearly as well. Therefore, backpropagation methods are left for independent study. The complete code for this object and all its methods is provided in the attachment.

#### Residual Connections Module

The _ResNeXt_ architecture features residual connections for each _Bottleneck_ module, which facilitate efficient error gradient propagation during backpropagation. These connections allow the model to reuse previously extracted features, improving convergence and reducing the risk of gradient vanishing. As a result, the model can be trained to greater depths without significantly increasing computational costs.

It is important to note that the output tensor of a _Bottleneck_ module maintains approximately the same overall size, but individual dimensions may change. A reduction in temporal steps is compensated by an increase in feature space dimensionality, preserving critical information and capturing long-term dependencies. A separate projection module ensures proper integration of residual connections by mapping input data to the required dimensions. This prevents mismatches and maintains training stability even in deep architectures.

In our implementation, we created this module as the _CNeuronResNeXtResidual_ object, whose structure is shown below.

```
class CNeuronResNeXtResidual:  public CNeuronConvOCL
  {
protected:
   CNeuronTransposeOCL     cTransposeIn;
   CNeuronConvOCL          cProjectionTime;
   CNeuronBatchNormOCL     cNormalizeTime;
   CNeuronTransposeOCL     cTransposeOut;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResNeXtResidual(void){};
                    ~CNeuronResNeXtResidual(void){};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out,
                          uint units_in, uint units_out,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResNeXtResidual;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual CLayerDescription* GetLayerInfo(void) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
```

When developing this object, we used approaches similar to those used in constructing _Bottleneck_ modules but adapted to the different dimensions of the input tensor.

In the presented object structure, you can see several nested objects - their functionality will be described during the implementation of the new class methods. All internal objects are declared statically. This allows us to leave the class's constructor and destructor empty. Initialization of all objects, including inherited ones, is performed in the _Init_ module, which receives a set of constants defining the object architecture.

```
bool CNeuronResNeXtResidual::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                                  uint chanels_in, uint chanels_out,
                                  uint units_in, uint units_out,
                                  ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, chanels_in, chanels_in, chanels_out,
                            units_out, 1, optimization_type, batch))
      return false;
```

Within the method, we first call the parent class method of the same name, passing the necessary parameters. As with _Bottleneck_ modules, we use a convolutional layer as the parent class. It also handles the projection of data into the new feature space.

After successful initialization of inherited objects and interfaces, we move on to working with the newly declared objects. To convenient working with temporal dimensions, we first transpose the input data.

```
   int index=0;
   if(!cTransposeIn.Init(0, index, OpenCL, units_in, chanels_in, optimization, iBatch))
      return false;
```

Next, a convolutional layer projects individual unit sequences into the specified dimensionality.

```
   index++;
   if(!cProjectionTime.Init(0, index, OpenCL, units_in, units_in, units_out, chanels_in, 1, optimization, iBatch))
      return false;
```

The results are normalized, similar to the _Bottleneck_ module. However, no activation function is applied, as the residual module must pass all information without loss.

```
   index++;
   if(!cNormalizeTime.Init(0, index, OpenCL, cProjectionTime.Neurons(), iBatch, optimization))
      return false;
```

We then adjust the feature space. For this, we perform inverse transposition. The projection is then handled using the parent class.

```
   index++;
   if(!cTransposeOut.Init(0, index, OpenCL, chanels_in, units_out, optimization, iBatch))
      return false;
//---
   return true;
  }
```

All that remains for us to do is return the logical result of the operations to the calling program and complete the work of the new object initialization method.

Next, we move on to constructing the feed-forward pass algorithm in the _feedForward_ method.

```
bool CNeuronResNeXtResidual::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
//--- Projection Timeline
   if(!cTransposeIn.FeedForward(NeuronOCL))
      return false;
```

The method receives a pointer to the object containing the input data. This pointer is passed to the internal data transposition layer, which converts the data into unit sequences.

Next, we adjust the dimensionality of the unit sequences to the target size using a convolutional layer.

```
   if(!cProjectionTime.FeedForward(cTransposeIn.AsObject()))
      return false;
```

The results are normalized.

```
   if(!cNormalizeTime.FeedForward(cProjectionTime.AsObject()))
      return false;
```

Finally, inverse transposition is applied, and the data is projected into the feature space.

```
//--- Projection Chanels
   if(!cTransposeOut.FeedForward(cNormalizeTime.AsObject()))
      return false;
   return CNeuronConvOCL::feedForward(cTransposeOut.AsObject());
  }
```

The final projection is performed using the parent class. The logical result of the operations is then returned to the calling program, completing the feed-forward method.

The feed-forward pass algorithm is linear. Therefore, during backpropagation, we have a linear gradient flow. So, backpropagation methods are left for independent study, similar to the _CNeuronResNeXtBottleneck_ object. The complete code for these objects and all their modules is provided in the attachment.

#### The _ResNeXt_ Block

Above, we created separate objects representing the two information streams of the _ResNeXt_ framework. Now it is time to combine these objects into a single structure, allowing more efficient data processing. For this purpose, we create the _CNeuronResNeXtBlock_ object, which will serve as the main block for subsequent data processing. The structure of this object is presented below.

```
class CNeuronResNeXtBlock :  public CNeuronBaseOCL
  {
protected:
   uint                     iChanelsOut;
   CNeuronResNeXtBottleneck cBottleneck;
   CNeuronResNeXtResidual   cResidual;
   CBufferFloat             cBuffer;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronResNeXtBlock(void){};
                    ~CNeuronResNeXtBlock(void){};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint chanels_in, uint chanels_out, uint window,
                          uint step, uint units_count, uint group_size, uint groups,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   override const   {  return defNeuronResNeXtBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual CLayerDescription* GetLayerInfo(void) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
  };
```

In this structure, we see familiar objects and a standard set of virtual methods, which we will need to override.

All internal objects are declared as static, allowing us to keep the class constructor and destructor empty. The initialization of these declared and inherited objects is performed in the _Init_ method. Its parameter structure is entirely inherited from the _CNeuronResNeXtBottleneck_ object.

```
bool CNeuronResNeXtBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint chanels_in, uint chanels_out, uint window,
                               uint step, uint units_count, uint group_size, uint groups,
                               ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   int units_out = ((int)units_count - (int)window + (int)step - 1) / (int)step + 1;
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, units_out * chanels_out, optimization_type, batch))
      return false;
```

Within the method body, we first determine the sequence dimensionality at the block output, then initialize the base interfaces inherited from the parent object.

After successfully executing the parent class initialization method, we save the necessary parameters into the object's variables.

```
   iChanelsOut = chanels_out;
```

We initialize the internal objects of the previously constructed information streams.

```
   int index = 0;
   if(!cBottleneck.Init(0, index, OpenCL, chanels_in, chanels_out, window, step, units_count,
                       group_size, groups, optimization, iBatch))
      return false;
   index++;
   if(!cResidual.Init(0, index, OpenCL, chanels_in, chanels_out, units_count, units_out, optimization, iBatch))
      return false;
```

At the block output, we expect to receive the sum of the values from the two information streams. Therefore, the received error gradient can be fully propagated to both data streams. To avoid unnecessary data copying, pointers to the corresponding data buffers are swapped.

```
   if(!cResidual.SetGradient(cBottleneck.getGradient(), true))
      return false;
   if(!SetGradient(cBottleneck.getGradient(), true))
      return false;
//---
   return true;
  }
```

Finally, we return a boolean result to the calling program and complete the initialization method.

Next, we build the feed-forward pass algorithms in the _feedForward_ method. Everything is quite simple here.

```
bool CNeuronResNeXtBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!cBottleneck.FeedForward(NeuronOCL))
      return false;
   if(!cResidual.FeedForward(NeuronOCL))
      return false;
```

The method receives a pointer to the input data object, which is immediately passed to the methods of the two information stream objects. The results are then summed and normalized.

```
   if(!SumAndNormilize(cBottleneck.getOutput(), cResidual.getOutput(), Output, iChanelsOut, true, 0, 0, 0, 1))
      return false;
//--- result
   return true;
  }
```

The logical result of the operations is then returned to the calling program, completing the feed-forward method.

Although the structure may appear simple at first glance, it actually includes two information streams, which adds complexity to error gradient distribution. The algorithm responsible for this process is implemented in the _calcInputGradients_ method.

```
bool CNeuronResNeXtBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
      return false;
```

The method receives a pointer to the input data object used during the feed-forward pass. In this case, the error gradient must be propagated according to the influence of the input data on the final model output. Data can only be passed to a valid object. Therefore, before continuing operations, we first check the relevance of the received pointer.

After passing this verification, the error gradient is propagated through the first information stream.

```
   if(!NeuronOCL.calcHiddenGradients(cBottleneck.AsObject()))
      return false;
```

Before propagating the gradient through the second stream, the previously obtained data must be preserved. Rather than fully copying the data, we use a pointer-swapping mechanism. The pointer to the error gradient buffer of the input data is saved in a local variable.

```
   CBufferFloat *temp = NeuronOCL.getGradient();
```

Next, we check if the auxiliary buffer matches the gradient buffer dimensions. Adjust, if necessary.

```
   if(cBuffer.GetOpenCL() != OpenCL ||
      cBuffer.Total() != temp.Total())
     {
      if(!cBuffer.BufferInitLike(temp))
         return false;
     }
```

Its pointer is then passed to the input data object.

```
   if(!NeuronOCL.SetGradient(GetPointer(cBuffer), false))
      return false;
```

Now the error gradient can be safely propagated through the second information stream without risk of data loss.

```
   if(!NeuronOCL.calcHiddenGradients(cResidual.AsObject()))
      return false;
```

The values from the two streams are summed, and the buffer pointers are restored to their original state.

```
   if(!SumAndNormilize(temp, NeuronOCL.getGradient(), temp, 1, false, 0, 0, 0, 1))
      return false;
   if(!NeuronOCL.SetGradient(temp, false))
      return false;
//---
   return true;
  }
```

Then we return the logical result of the operation to the caller and complete the error gradient distribution method.

This concludes the overview of the algorithmic construction of the _ResNeXt_ block object methods. The complete code for this object and all its methods is provided in the attachment.

We have now reached the end of this article, but our work is not yet complete. We will pause briefly and continue in the next article.

### Conclusion

In this article, we introduced a multi-task learning framework based on the _ResNeXt_ architecture, designed for processing financial data. This framework enables efficient feature extraction and processing, optimizing classification and regression tasks in high-dimensional and time-series data environments.

In the practical section, we constructed the main elements of the _ResNeXt_ architecture. In the next article, we will build the multi-task learning framework and evaluate the effectiveness of the implemented approaches on real historical data.

#### References

- [Aggregated Residual Transformations for Deep Neural Networks](https://www.mql5.com/go?link=https://arxiv.org/abs/1611.05431 "Aggregated Residual Transformations for Deep Neural Networks")
- [Collaborative Optimization in Financial Data Mining Through Deep Learning and ResNeXt](https://www.mql5.com/go?link=https://arxiv.org/abs/2412.17314 "Collaborative Optimization in Financial Data Mining Through Deep Learning and ResNeXt")
- [Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

#### Programs used in the article

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

Original article: [https://www.mql5.com/ru/articles/17142](https://www.mql5.com/ru/articles/17142)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17142.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/17142/mql5.zip "Download MQL5.zip")(2430.99 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Neural Networks in Trading: Hybrid Graph Sequence Models (GSM++)](https://www.mql5.com/en/articles/17279)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Final Part)](https://www.mql5.com/en/articles/17241)
- [Neural Networks in Trading: Two-Dimensional Connection Space Models (Chimera)](https://www.mql5.com/en/articles/17210)
- [Neural Networks in Trading: Multi-Task Learning Based on the ResNeXt Model (Final Part)](https://www.mql5.com/en/articles/17157)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Final Part)](https://www.mql5.com/en/articles/17104)
- [Neural Networks in Trading: Hierarchical Dual-Tower Transformer (Hidformer)](https://www.mql5.com/en/articles/17069)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/501006)**
(9)


![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
16 Jan 2026 at 21:25

**Alain Verleyen [#](https://www.mql5.com/ru/forum/481083#comment_58959297):**

In my experience, traders who can share something really useful never share anything.

Yes, they know (like me since 1998) that a working strategy quickly stops working after distribution.

That's why forum programmers share individual solutions, while a working (profitable) strategy has never been published. Or sold.

![lynxntech](https://c.mql5.com/avatar/2022/7/62CF9DBF-A3CD.png)

**[lynxntech](https://www.mql5.com/en/users/lynxntech)**
\|
16 Jan 2026 at 21:29

**Edgar Akhmadeev [#](https://www.mql5.com/ru/forum/481083#comment_58959460):**

Yes, they know (as I have since 1998) that a strategy that works quickly stops working once it is disseminated.

That's why forum programmers share individual solutions, and a working (profitable) strategy has never been published. Or sold.

and the need to transfer funds between countries doesn't count anymore?)

How can you be such a system?

A trading robot will always work if you buy on a pullback, the question is where is the pullback?

![lynxntech](https://c.mql5.com/avatar/2022/7/62CF9DBF-A3CD.png)

**[lynxntech](https://www.mql5.com/en/users/lynxntech)**
\|
16 Jan 2026 at 21:47

I've seen the translation, I'm definitely not translatable.


![Edgar Akhmadeev](https://c.mql5.com/avatar/avatar_na2.png)

**[Edgar Akhmadeev](https://www.mql5.com/en/users/dali)**
\|
16 Jan 2026 at 22:21

**lynxntech [#](https://www.mql5.com/ru/forum/481083#comment_58959510):**

I've seen the translation, I'm definitely not translatable

I have to admit, I wasn't smart enough to understand the original.

"I've been talking to myself all night, and they didn't understand me!" (Zhvanetsky

![Vitaly Muzichenko](https://c.mql5.com/avatar/2025/11/691d3a3a-b70b.png)

**[Vitaly Muzichenko](https://www.mql5.com/en/users/mvs)**
\|
17 Jan 2026 at 00:40

**Edgar Akhmadeev [#](https://www.mql5.com/ru/forum/481083#comment_58959460):**

Yes, they know (as I have since 1998) that a strategy that works quickly ceases to work once it is disseminated.

This applies to exchanges with limited liquidity, it does not apply to forex, there is enough liquidity there for everyone

P.S. I remembered Mikhail, he has a system of hedging on the Moscow Exchange, he shared it and it works, and it should work in the future. Everything depends on personal capital, and there is nothing to do there with 100 dollars.

Here, everyone is looking for a system for a hundred quid, and profitability of 10% per day. That's why such results of searches.

![Automating Trading Strategies in MQL5 (Part 44): Change of Character (CHoCH) Detection with Swing High/Low Breaks](https://c.mql5.com/2/184/20355-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 44): Change of Character (CHoCH) Detection with Swing High/Low Breaks](https://www.mql5.com/en/articles/20355)

In this article, we develop a Change of Character (CHoCH) detection system in MQL5 that identifies swing highs and lows over a user-defined bar length, labels them as HH/LH for highs or LL/HL for lows to determine trend direction, and triggers trades on breaks of these swing points, indicating a potential reversal, and trades the breaks when the structure changes.

![The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://c.mql5.com/2/183/20289-the-mql5-standard-library-explorer-logo.png)[The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)

In this session, we will build a sophisticated, multi-signal Expert Advisor using the MQL5 Standard Library. This approach allows us to seamlessly blend built-in signals with our own custom logic, demonstrating how to construct a powerful and flexible trading algorithm. For more, click to read further.

![Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://c.mql5.com/2/184/20425-introduction-to-mql5-part-30-logo.png)[Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

Discover a step-by-step tutorial that simplifies the extraction, conversion, and organization of candle data from API responses within the MQL5 environment. This guide is perfect for newcomers looking to enhance their coding skills and develop robust strategies for managing market data efficiently.

![The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://c.mql5.com/2/140/View_Component_for_Tables_in_MVC_Paradigm_in_MQL5_Basic_Graphic_Element___LOGO.png)[The View component for tables in the MQL5 MVC paradigm: Base graphical element](https://www.mql5.com/en/articles/17960)

The article covers the process of developing a base graphical element for the View component as part of the implementation of tables in the MVC (Model-View-Controller) paradigm in MQL5. This is the first article on the View component and the third one in a series of articles on creating tables for the MetaTrader 5 client terminal.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/17142&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069447344035923263)

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