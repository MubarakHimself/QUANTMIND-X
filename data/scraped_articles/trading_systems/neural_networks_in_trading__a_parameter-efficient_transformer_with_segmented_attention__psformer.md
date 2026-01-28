---
title: Neural Networks in Trading: A Parameter-Efficient Transformer with Segmented Attention (PSformer)
url: https://www.mql5.com/en/articles/16439
categories: Trading Systems, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:33:27.713199
---

[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/16439&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5069544491901191878)

MetaTrader 5 / Trading systems


### Introduction

Multivariate time series forecasting is an important task in deep learning, with practical applications in meteorology, energy, anomaly detection, and financial analysis. With the rapid advancement of artificial intelligence, significant efforts have been made to design innovative models that improve forecasting accuracy. _Transformer_-based architectures, in particular, have attracted considerable attention due to their proven effectiveness in natural language processing and computer vision. Moreover, large-scale, pre-trained _Transformer_ models have demonstrated strong performance in time series forecasting, showing that increasing model parameters and training data can substantially enhance predictive capabilities.

At the same time, many simple linear models achieve competitive results compared to more complex _Transformer_-based architectures. Their success in time series forecasting is likely due to their lower complexity, which reduces the risk of overfitting to noisy or irrelevant data. Even with limited datasets, these models can effectively capture stable, reliable patterns.

To address the challenges of modeling long-term dependencies and capturing complex temporal relationships, the _[PatchTST](https://www.mql5.com/en/articles/14798)_ approach processes data using patching techniques to extract local semantics, delivering strong performance. However, PatchTST uses channel-independent structures and has significant potential for further improvement in modeling efficiency. Furthermore, the unique nature of multivariate time series, where temporal and spatial dimensions differ significantly from other data types, offers many unexplored opportunities.

One way to reduce model complexity in deep learning is parameter sharing ( _PS_), which significantly decreases the number of parameters while improving computational efficiency. In convolutional networks, filters share weights across spatial positions, extracting local features with fewer parameters. Similarly, _LSTM_ models share weight matrices across time steps, managing memory and information flow. In natural language processing, parameter sharing has been extended to _Transformers_ by reusing weights across layers, reducing redundancy without compromising performance.

In multitask learning, the _Task-Adaptive Parameter Sharing_ ( _TAPS_) method selectively fine-tunes task-specific layers, maximizing parameter sharing while enabling efficient learning with minimal task-specific adjustments. Research indicates that parameter sharing can reduce model size, improve generalization, and lower overfitting risk across diverse tasks.

The authors of " _[PSformer: Parameter-efficient Transformer with Segment Attention for Time Series Forecasting](https://www.mql5.com/go?link=https://arxiv.org/abs/2411.01419 "https://arxiv.org/abs/2411.01419")_" propose an innovative _Transformer_-based model for multivariate time series forecasting that incorporates parameter sharing principles.

They introduce a _Transformer_ encoder with a two-level segment-based attention mechanism, where each encoder layer includes a shared-parameter block. This block contains three fully connected layers with residual connections, enabling a low overall parameter count while maintaining effective information exchange across model components. To focus attention within segments, they apply a patching method that splits variable sequences into separate patches. Patches occupying the same position across different variables are then grouped into segments. Each segment becomes a spatial extension of a single-variable patch, effectively dividing the multivariate time series into multiple segments.

Within each segment, attention mechanisms enhance the capture of local spatio-temporal relationships, while cross-segment information integration improves overall forecasting accuracy. The authors also incorporate the _SAM_ optimization method to further reduce overfitting without degrading learning performance. Extensive experiments on long-term time series forecasting datasets show that _PSformer_ delivers strong results. _PSformer_ outperforms state-of-the-art models in 6 out of 8 key forecasting benchmarks.

### The PSformer Algorithm

A multivariate time series _X_ ∈ _R_ _M×L_ contains _M_ variables ans a look-back window of length _L_. The sequence length _L_ is evenly divided into _N_ non-overlapping patches of size _P_. Then, _P_( _i_) from the _M_ variables forms the _i_-th segment, representing a cross-section of length _C_ (where _C=M×_ _P_).

The key components of _PSformer_ are Segment Attention ( _SegAtt_) and the Parameter-Sharing Block ( _PS_). The _PSformer_ encoder serves as the model's core, containing both the SegAtt module and the _PS_ block. The _PS_ block provides parameters for all encoder layers through parameter sharing.

As in other time series forecasting architectures, the _PSformer_ authors use the _[RevIN](https://www.mql5.com/en/articles/14673)_ method to effectively address distribution shift issues.

Segment spatio-temporal attention ( _SegAtt_) merges patches from different channels at the same position into a segment and establishes spatio-temporal relationships across segments. Specifically, the original time series _X_ ∈ _R_ _M_ _×L_ is first divided into patches where _L=P×N_, then reshaped into _X_ ∈ _R__(_ _M×P)×N_, by merging dimensions _M_ and _P_. This produces _X_ ∈ _R_ _C×N_ (where _C=M×P_), enabling cross-channel information fusion.

In this transformed space, the data is processed by two consecutive modules with identical architecture, separated by a _ReLU_ activation. Each module contains a parameter-sharing block and a _Self-Attention_ mechanism which is already familiar to us. While computing 𝑸 _uery_ ∈ _RC×N_, 𝑲 _ey_ ∈ _RC×N_ and 𝑽 _alue_ ∈ _RC×N_ matrices involves non-linear transformations of input _Xin_ along segments into _N_-dimensional representations, the scaled dot-product attention primarily distributes focus across the entire _C_ dimension. This allows the model to learn dependencies between spatio-temporal segments across both channels and time.

This mechanism integrates information from different segments by computing _Q_, _K_, and _V_. It also captures local spatio-temporal dependencies within each segment while also modeling long-term inter-segment relationships across extended time steps. The final output is _Xou_ _t_ ∈ _R_ _C_ _×N_, completing the attention process.

_PSformer_ introduces a new _Parameter Shared Block_ ( _PS Block_), consisting of three fully connected layers with residual connections. Specifically, it uses three learnable linear mappings _W_ _j_ ∈ _RN×N_ с _j_ ∈ {1, 2, 3}. The outputs of the first two layers are computed as follows:

![](https://c.mql5.com/2/163/1423111755845__1.png)

This structure is analogous to a _FeedForward_ block with residual connections. The intermediate output 𝑿_o_ _u_ _t_ then serves as the input for the third transformation:

![](https://c.mql5.com/2/163/5239562505326__1.png)

Overall, the _PS_ block can be expressed as:

![](https://c.mql5.com/2/163/2951769139331__1.png)

The _PS_ block stucture enables non-linear transformations while preserving the trajectory of a linear mapping. Although three layers in the _PS_ block have different parameters, the entire _PS_ block is reused across multiple positions in the _PSformer_ encoder, ensuring that the same 𝑾_S_ block parameters are common to all these positions. Specifically, the _PS_ block's parameters are shared in three parts of each _PSformer_ encoder: including the two SegAtt modules and the final _PS_ block. This parameter-sharing strategy reduces the total parameter count while maintaining model expressiveness.

The two-stage _SegAtt_ mechanism can be compared to a _FeedForward_ block in a vanilla _Transformer_, where the _MLP_ is replaced with attention operations. Residual connections are added between the input and output, and the result is passed to the final _PS_ block.

A dimensional transformation is then applied to obtain 𝑿_o_ _⁢u_ _t_ ∈ _R_ _M_ _×_ _L_, where _C=M×_ _P_ and _L=P×N_.

After passing through _n_ layers of _PSformer_, a final transformation is applied to project the output onto the forecasting horizon _F_.

![](https://c.mql5.com/2/163/4906244484532__1.png)

where 𝑿_p_ _re_ _d_ ∈ _R_ _M_ _×_ _F_ and 𝑾_F_ ∈ _R_ _L_ _×_ _F_ represent linear mapping.

The original visualization of the _PSformer_ framework is provided below.

![](https://c.mql5.com/2/163/6270162761143__1.png)

### Implementation in MQL5

After covering the theoretical aspects of the _PSformer_ framework, we now move on to the practical implementation of our vision of the proposed approaches using _MQL5_. Of particular interest to us is the algorithm for implementing the Parameter-Sharing Block ( _PS_ Block).

#### Parameter Shared Block

As noted earlier, in the authors' original implementation the PS Block consists of three fully connected layers whose parameters are applied to all analyzed segments. From our perspective, there's nothing complicated. We have repeatedly employed convolutional layers with non-overlapping analysis windows in similar situations. The real challenge lies elsewhere: in the mechanism for sharing parameters across multiple blocks.

On the one hand, we could certainly reuse the same block multiple times within a single layer. However, this introduces the problem of preserving data for the backpropagation pass. When an object is reused for multiple feed-forward passes, the result buffer will store new outputs, overwriting those from previous feed-forward passes. In a typical neural layer workflow, this is not an issue, since we consistently alternate between feed-forward and backpropagation passes. After each backward pass, the results from the preceding feed-forward pass are no longer needed and can safely be overwritten. But when this alternation is disrupted, we face the problem of retaining all the data required for a correct backpropagation pass.

In such cases, we must store not only the final outputs of the block, but also all intermediate values. Or we must recompute them, which increases the model's computational complexity. Additionally, a mechanism is needed to synchronize buffers at specific points to correctly compute the error gradient.

Clearly, implementing these requirements would require changes to our data exchange interfaces between neural layers. This, in turn, would trigger broader modifications to our library functionality.

The second option is to establish a mechanism for full-fledged sharing of a single parameter buffer among several identical neural layers. This approach, however, is not without its own "hidden pitfalls".

Recall that when we explored the [_Deep Deterministic Policy Gradient_](https://www.mql5.com/en/articles/12853) framework, we implemented a soft parameter update algorithm for the target model. Copying parameters after each update, however, is computationally expensive. Ideally, we would replace the parameter buffers in the relevant objects with shared parameter matrices.

Here, in addition to the parameter matrix itself, we must also share the momentum buffers used during parameter updates. Using separate momentum buffers at different stages can bias the parameter update vector toward one of the internal layers.

There is another critical point. Шn this implementation, parameters used in the backpropagation pass may differ from those used in the feed-forward pass. This may sound unusual, but let's illustrate it with a simple example involving two consecutive layers that share parameters. During the feed-forward pass, both layers use parameters _W_ and produce outputs _O1_ and _O2_ respectively. At the gradient distribution stage, we compute error gradients _G1_ and _G2_ respectively. SO, the error gradient propagation process is correct. At this stage, the model parameters remain unchanged, and all error gradients correctly correspond to the feed-forward parameters _W_. However, if we update the parameters in one of the layers, for example, the second, we get adjusted parameters _W'_. We immediately encounter a mismatch: the error gradients no longer correspond to the updated parameters. Directly applying a mismatched gradient can distort the training process.

One solution to this problem is to determine the target values for a given layer based on the outputs from the last feed-forward pass and the corresponding error gradients, then perform a new feed-forward pass with the updated parameters to compute a corrected error gradient. If this sounds familiar, it is because this approach closely resembles the _SAM_ optimization algorithm we discussed in previous articles. Indeed, by adding parameter updates before executing the repeated forward pass, we obtain the full _SAM_ optimization procedure.

This is precisely why the authors of the _PSformer_ framework recommend using _SAM_ optimization. It allows us to tolerate the risk of gradient–parameter mismatches, since the gradients are recomputed before parameter updates. In other scenarios, however, such mismatches could pose a serious problem.

Considering all the above, we decided to adopt the second approach - sharing parameter buffers between identical layers.

As noted earlier, the _PS_ Block in the original paper employs three fully connected layers, which we replace with convolutional layers. We therefore begin our parameter-sharing implementation with the _CNeuronConvSAMOCL_ convolutional layer object.

In our convolutional parameter-sharing layer, we substitute only the pointers to the parameter and momentum buffers. All other buffers and internal variables must still match the dimensions of the parameter matrix. Naturally, this requires adjustments to the object's initialization method. Before doing so, we create two auxiliary methods: _InitBufferLike_ and _ReplaceBuffer_.

InitBufferLike creates a new buffer filled with zero values based on a given reference buffer. Its algorithm is quite simple. It accepts two pointers to data buffer objects as parameters. First, it checks whether the reference buffer pointer ( _master_) is valid. The presence of a valid reference pointer is critical for subsequent operations. If this check fails, the method terminates and returns _false_.

```
bool CNeuronConvSAMOCL::InitBufferLike(CBufferFloat *&buffer, CBufferFloat *master)
  {
   if(!master)
      return false;
```

If the first checkpoint is passed successfully, we check the relevance of the pointer to the created buffer. But here, if we get a negative result, we simply create a new instance of the object.

```
   if(!buffer)
     {
      buffer = new CBufferFloat();
      if(!buffer)
         return false;
     }
```

And don't forget to check if the new buffer has been created correctly.

Next, we initialize the buffer of the required size with zero values.

```
   if(!buffer.BufferInit(master.Total(), 0))
      return false;
```

Then we create its copy in the _OpenCL_ context.

```
   if(!buffer.BufferCreate(master.GetOpenCL()))
      return false;
//---
   return true;
  }
```

The method then concludes by returning the logical result of the operation to the caller.

Second method _ReplaceBuffer_ substitutes the pointer to the specified buffer. At first glance, we don't need a whole method to assign a pointer to an internal variable object. However, in the method body we check and, if necessary, remove excess data buffers. This allows us to use both RAM and _OpenCL_-context memory more efficiently.

```
void CNeuronConvSAMOCL::ReplaceBuffer(CBufferFloat *&buffer, CBufferFloat *master)
  {
   if(buffer==master)
     return;
   if(!!buffer)
     {
      buffer.BufferFree();
      delete buffer;
     }
//---
   buffer = master;
  }
```

After creating the auxiliary methods, we proceed to building a new initialization algorithm for the convolutional layer object based on a reference instance _InitPS_. In this method, instead of receiving a full set of constants defining the object architecture, we accept only a pointer to a reference object.

```
bool CNeuronConvSAMOCL::InitPS(CNeuronConvSAMOCL *master)
  {
   if(!master ||
      master.Type() != Type()
     )
      return false;
```

In the method body, we check if the received pointer is correct and the object types match.

Next, instead of build a whole set of parent class methods, we simply transfer the values of all inherited parameters from the reference object.

```
   alpha = master.alpha;
   iBatch = master.iBatch;
   t = master.t;
   m_myIndex = master.m_myIndex;
   activation = master.activation;
   optimization = master.optimization;
   iWindow = master.iWindow;
   iStep = master.iStep;
   iWindowOut = master.iWindowOut;
   iVariables = master.iVariables;
   bTrain = master.bTrain;
   fRho = master.fRho;
```

Next, we create result and error gradient buffers similar to those in the reference object.

```
   if(!InitBufferLike(Output, master.Output))
      return false;
   if(!!master.getPrevOutput())
      if(!InitBufferLike(PrevOutput, master.getPrevOutput()))
         return false;
   if(!InitBufferLike(Gradient, master.Gradient))
      return false;
```

After that, we transfer the pointers first to the weight and moment buffers inherited from the basic fully connected layer.

```
   ReplaceBuffer(Weights, master.Weights);
   ReplaceBuffer(DeltaWeights, master.DeltaWeights);
   ReplaceBuffer(FirstMomentum, master.FirstMomentum);
   ReplaceBuffer(SecondMomentum, master.SecondMomentum);
```

We repeat a similar operation for the buffers of convolutional layer parameters and their moments.

```
   ReplaceBuffer(WeightsConv, master.WeightsConv);
   ReplaceBuffer(DeltaWeightsConv, master.DeltaWeightsConv);
   ReplaceBuffer(FirstMomentumConv, master.FirstMomentumConv);
   ReplaceBuffer(SecondMomentumConv, master.SecondMomentumConv);
```

Next, we need to create buffers of adjusted parameters. However, both adjusted parameter buffers may not be created under certain conditions. The buffer of adjusted parameters of a fully connected layer is created only if there are outgoing connections. Therefore, we first check the size of this buffer in the reference object. We create the relevant buffer only when necessary.

```
   if(master.cWeightsSAM.Total() > 0)
     {
      CBufferFloat *buf = GetPointer(cWeightsSAM);
      if(!InitBufferLike(buf, GetPointer(master.cWeightsSAM)))
         return false;
     }
```

Otherwise, we clear this buffer, reducing memory consumption.

```
   else
     {
      cWeightsSAM.BufferFree();
      cWeightsSAM.Clear();
     }
```

The buffer of adjusted parameters of incoming connections is created when the blur area coefficient is greater than 0.

```
   if(fRho > 0)
     {
      CBufferFloat *buf = GetPointer(cWeightsSAMConv);
      if(!InitBufferLike(buf, GetPointer(master.cWeightsSAMConv)))
         return false;
     }
```

Otherwise, we clear this buffer.

```
   else
     {
      cWeightsSAMConv.BufferFree();
      cWeightsSAMConv.Clear();
     }
```

Technically, instead of using the blur coefficient, we could check the size of the buffer containing the adjusted parameters of the incoming connections of the reference object - just as we do for the buffer of adjusted parameters for the outgoing connections. However, we know that if the blur coefficient is greater than zero, this buffer must exist. Thus, we include an additional control. If an attempt is made to create a zero-length buffer, the process will fail, throwing an error and halting initialization. This helps prevent more serious issues later in execution.

At the end of the initialization method, we transfer all objects into a single _OpenCL_ context and return the logical result of the operation to the calling program.

```
   SetOpenCL(master.OpenCL);
//---
   return true;
  }
```

After modifying the convolutional layer object, we proceed to the next stage of our work. Now we will create the Parameter-Sharing Block ( _PS_ Block) itself. For this, we introduce a new object: _CNeuronPSBlock_. As outlined in the theoretical section, the PS Block consists of three sequential data transformation layers. Each has a square parameter matrix, ensuring that the input and output tensor dimensions remain consistent for both the block as a whole and its internal layers. Between the first two layers, a [_GELU_](https://www.mql5.com/en/articles/16360#para33) activation function is applied. After the second layer, a residual connection is added to the original input.

To implement this architecture, the new object will contain two internal convolutional layers, while the final convolutional layer will be represented directly by the structure of our class itself, inheriting the base functionality from the convolutional layer class. Since we will be using _SAM_ optimization during training, all convolutional layers in the architecture will be SAM-compatible. The structure of the new class is shown below.

```
class CNeuronPSBlock :  public CNeuronConvSAMOCL
  {
protected:
   CNeuronConvSAMOCL acConvolution[2];
   CNeuronBaseOCL    cResidual;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronPSBlock(void) {};
                    ~CNeuronPSBlock(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint units_count,
                          uint variables, float rho,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   virtual bool      InitPS(CNeuronPSBlock *master);
   //---
   virtual int       Type(void)   const         {  return defNeuronPSBlock;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual CLayerDescription* GetLayerInfo(void);
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
```

As seen in the structure, the new object declares two initialization methods. This is done on purpose. _Init_ – the standard initialization method, where the architecture of the object is explicitly defined by the parameters passed to the method. _InitPS_ – analogous to the method of the same name in the convolutional layer class, it creates a new object based on the structure of a reference object. During this process, pointers to parameter and momentum buffers are copied from the reference. Let's consider in more detail the algorithm for constructing the specified methods.

As mentioned above, the _Init_ method receives a set of constants in its parameters, allowing the architecture of the object to be fully determined.

```
bool CNeuronPSBlock::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint units_count,
                          uint variables, float rho,
                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvSAMOCL::Init(numOutputs, myIndex, open_cl, window, window, window_out,
                               units_count, variables, rho, optimization_type, batch))
      return false;
```

In the method body, we immediately forward all received parameters to the identically named method of the parent class. As you know, the parent method already contains the necessary parameter validation points and initialization logic for inherited objects.

Since all convolutional layers inside the PS Block have the same dimensions, the initialization of the first internal convolutional layer uses the exact same parameters.

```
   if(!acConvolution[0].Init(0, 0, OpenCL, iWindow, iWindow, iWindowOut, units_count,
                             iVariables, fRho, optimization, iBatch))
      return false;
   acConvolution[0].SetActivationFunction(GELU);
```

We then add the GELU activation function, as suggested by the PSformer authors.

However, we also allow the user to modify the tensor dimensions at the block output. Therefore, when initializing the second internal convolutional layer, which is followed by a residual connection, we swap the parameters for the analysis window size and the number of filters. This ensures that the output dimensions match those of the original input data.

```
   if(!acConvolution[1].Init(0, 1, OpenCL, iWindowOut, iWindowOut, iWindow, units_count,
                             iVariables, fRho, optimization, iBatch))
      return false;
   acConvolution[1].SetActivationFunction(None);
```

We do not use the activation function here.

Next, we add a base neural layer to store the residual connection data. Its size corresponds to the result buffer of the second nested convolutional layer.

```
   if(!cResidual.Init(0, 2, OpenCL, acConvolution[1].Neurons(), optimization, iBatch))
      return false;
   if(!cResidual.SetGradient(acConvolution[1].getGradient(), true))
      return false;
   cResidual.SetActivationFunction(None);
```

Immediately after creating the object based on the reference instance, we replace the error-gradient buffer. This optimization allows us to reduce the number of data-copy operations during the backpropagation pass.

Next, we explicitly disable the activation function for our parameter-sharing block and complete the method, returning a logical result to the caller.

```
   SetActivationFunction(None);
//---
   return true;
  }
```

The second initialization method is somewhat simpler. It receives a pointer to a reference object and directly passes it to the identically named method of the parent class.

It is important to note that the parameter types in the current method differ from those in the parent class. So, we explicitly specify the type of the object being passed.

```
bool CNeuronPSBlock::InitPS(CNeuronPSBlock *master)
  {
   if(!CNeuronConvSAMOCL::InitPS((CNeuronConvSAMOCL*)master))
      return false;
```

The parent class method already contains the necessary validation checks, as well as the logic for copying constants, creating new buffers, and storing pointers to the parameter and momentum buffers.

We then iterate through the internal convolutional layers, calling their corresponding initialization methods and copying data from the respective reference objects.

```
   for(int i = 0; i < 2; i++)
      if(!acConvolution[i].InitPS(master.acConvolution[i].AsObject()))
         return false;
```

The residual-connection layer does not contain trainable parameters, and its size matches the result buffer of the second internal convolutional layer. Therefore, its initialization logic is taken entirely from the main initialization method.

```
   if(!cResidual.Init(0, 2, OpenCL, acConvolution[1].Neurons(), optimization, iBatch))
      return false;
   if(!cResidual.SetGradient(acConvolution[1].getGradient(), true))
      return false;
   cResidual.SetActivationFunction(None);
//---
   return true;
  }
```

As before, we replace the pointers to the error-gradient buffer.

With the initialization methods complete, we move on to the feed-forward algorithms. This part is relatively straightforward. The method receives a pointer to the input data object, which we pass directly to the feed-forward method of the first internal convolutional layer.

```
bool CNeuronPSBlock::feedForward(CNeuronBaseOCL *NeuronOCL)
  {
   if(!acConvolution[0].FeedForward(NeuronOCL))
      return false;
```

The results are then passed sequentially to the next convolutional layer. Afterward, we sum the resulting values with the original input. We save the sum in the residual-connection buffer.

```
   if(!acConvolution[1].FeedForward(acConvolution[0].AsObject()))
      return false;
   if(!SumAndNormilize(NeuronOCL.getOutput(), acConvolution[1].getOutput(), cResidual.getOutput(),
                                                                       iWindow, true, 0, 0, 0, 1))
      return false;
```

Here, we diverge slightly from the original PSformer algorithm: we normalize the residual tensor before passing it to the final convolutional layer, whose functionality is inherited from the parent class.

```
   if(!CNeuronConvSAMOCL::feedForward(cResidual.AsObject()))
      return false;
//---
   return true;
  }
```

The method concludes by returning the logical result of the operation to the caller.

The error-gradient distribution method _calcInputGradients_ is also simple but has important nuances. It receives a pointer to the source-data layer object, into which we must propagate the error gradient.

```
bool CNeuronPSBlock::calcInputGradients(CNeuronBaseOCL *NeuronOCL)
  {
   if(!NeuronOCL)
     return false;
```

First, we check the validity of the received pointer - if invalid, further processing is meaningless.

We then pass the gradients backward through all convolutional layers in reverse order.

```
   if(!CNeuronConvSAMOCL::calcInputGradients(cResidual.AsObject()))
      return false;
   if(!acConvolution[0].calcHiddenGradients(acConvolution[1].AsObject()))
      return false;
   if(!NeuronOCL.calcHiddenGradients(acConvolution[0].AsObject()))
      return false;
```

Note that no explicit error-gradient transfer is coded from the residual-connection object to the second internal convolutional layer. However, thanks to our earlier pointer substitution for data buffers, information is still transferred in full.

After sending the gradient back to the source-data layer through the convolutional layer pipeline, we also add the gradient from the residual-connection branch. There are two possible cases, depending on whether the source-data object has an activation function.

I want to remind you that we pass the error gradient to the residual connections object without adjusting for the derivative of the activation function. We explicitly indicated its absence for this object.

Therefore, given the absence of an activation function for the source data object, we only need to add the corresponding values of the two buffers.

```
   if(NeuronOCL.Activation() == None)
     {
      if(!SumAndNormilize(NeuronOCL.getGradient(), cResidual.getGradient(), NeuronOCL.getGradient(),
                                                                        iWindow, false, 0, 0, 0, 1))
         return false;
     }
```

Otherwise, we first adjust the obtained error gradient using the derivative of the activation function into a free buffer. And then, we sum the obtained results with those previously accumulated in the source-data object buffer.

```
   else
     {
      if(!DeActivation(NeuronOCL.getOutput(), cResidual.getGradient(), cResidual.getPrevOutput(),
                                                                           NeuronOCL.Activation()) ||
         !SumAndNormilize(NeuronOCL.getGradient(), cResidual.getPrevOutput(), NeuronOCL.getGradient(),
                                                                          iWindow, false, 0, 0, 0, 1))
         return false;
     }
//---
   return true;
  }
```

Then we complete the method.

A few words should be said about the _updateInputWeights_ method where we update block parameters. This block is straightforward – we just call the corresponding methods in the parent class and in the internal objects containing trainable parameters.

```
bool CNeuronPSBlock::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(!CNeuronConvSAMOCL::updateInputWeights(cResidual.AsObject()))
      return false;
   if(!acConvolution[1].UpdateInputWeights(acConvolution[0].AsObject()))
      return false;
   if(!acConvolution[0].UpdateInputWeights(NeuronOCL))
      return false;
//---
   return true;
  }
```

However, using _SAM_ optimization imposes strict requirements on the order of operations. During _SAM_ optimization, we perform a second forward pass with adjusted parameters. This updates the result buffer While this is harmless for updating the current layer parameters, it can disrupt parameter updates in subsequent layers. This is because they use the previous layer's feed-forward results. To prevent this, we must update parameters in reverse order through the internal objects. This will ensure each layer's parameters are adjusted before its input buffer is altered by another layer.

This concludes our discussion of the _CNeuronPSBlock_ parameter sharing block algorithms. The full source code for this class and its methods can be found in the provided attachment.

Our work is not yet complete, but the article turned out to be long. Therefore, we will take a short break and continue the work in the next article.

### Conclusion

In this article, we explored the _PSformer_ framework, whose authors emphasize its high accuracy in time series forecasting and efficient use of computational resources. The _PSformer's_ key architectural components include the Parameter-Sharing Block ( _PS_) and Segment-Based Spatio-Temporal Attention ( _SegAtt_). They allow for effective modeling of both local and global time series dependencies while reducing parameter count without sacrificing forecast quality.

In the practical section, we began implementing our own interpretation of the proposed methods in _MQL5_. Our work is not yet complete. In the next article, we will continue development and evaluate the effectiveness of these approaches on real historical datasets relevant to our specific tasks.

#### References

[PSformer: Parameter-efficient Transformer with Segment Attention for Time Series Forecasting](https://www.mql5.com/go?link=https://arxiv.org/abs/2411.01419 "PSformer: Parameter-efficient Transformer with Segment Attention for Time Series Forecasting")
[Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

#### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | Research.mq5 | Expert Advisor | Expert Advisor for collecting examples |
| 2 | ResearchRealORL.mq5 | Expert Advisor | Expert Advisor for collecting examples using the Real-ORL method |
| 3 | Study.mq5 | Expert Advisor | Model training Expert Advisor |
| 4 | StudyEncoder.mq5 | Expert Advisor | Encoder training Expert Advisor |
| 5 | Test.mq5 | Expert Advisor | Model testing Expert Advisor |
| 6 | Trajectory.mqh | Class library | System state description structure |
| 7 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 8 | NeuroNet.cl | Library | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/16439](https://www.mql5.com/ru/articles/16439)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16439.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16439/mql5.zip "Download MQL5.zip")(2171.23 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/493210)**
(1)


![youwei_qing](https://c.mql5.com/avatar/avatar_na2.png)

**[youwei\_qing](https://www.mql5.com/en/users/youwei_qing)**
\|
20 Apr 2025 at 11:18

I observed that the second parameter 'SecondInput' is unused, as CNeuronBaseOCL's feedForward method with two parameters internally calls the single-parameter version. Can you verify if this is a bug?

class CNeuronBaseOCL : public CObject

{

...

virtual bool feedForward(CNeuronBaseOCL \*NeuronOCL);

virtual bool feedForward(CNeuronBaseOCL \*NeuronOCL, CBufferFloat \*SecondInput) { return feedForward(NeuronOCL); }

..

}

Actor.feedForward((CBufferFloat\* [)GetPointer](https://www.mql5.com/en/docs/common/getpointer "MQL5 documentation: GetPointer function")(bAccount), 1, false, GetPointer(Encoder),LatentLayer)； ？？

Encoder.feedForward((CBufferFloat\*)GetPointer(bState), 1, false, GetPointer(bAccount))； ？？？

![Developing a Replay System (Part 76): New Chart Trade (III)](https://c.mql5.com/2/103/Desenvolvendo_um_sistema_de_Replay_Parte_76___LOGO.png)[Developing a Replay System (Part 76): New Chart Trade (III)](https://www.mql5.com/en/articles/12443)

In this article, we'll look at how the code of DispatchMessage, missing from the previous article, works. We will laso introduce the topic of the next article. For this reason, it is important to understand how this code works before moving on to the next topic. The content presented here is intended solely for educational purposes. Under no circumstances should the application be viewed for any purpose other than to learn and master the concepts presented.

![Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation](https://c.mql5.com/2/162/19077-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation](https://www.mql5.com/en/articles/19077)

In this article, we develop a trendline trader program that uses least squares fit to detect support and resistance trendlines, generating dynamic buy and sell signals based on price touches and open positions based on generated signals.

![Price Action Analysis Toolkit Development (Part 36): Unlocking Direct Python Access to MetaTrader 5 Market Streams](https://c.mql5.com/2/162/19065-price-action-analysis-toolkit-logo__1.png)[Price Action Analysis Toolkit Development (Part 36): Unlocking Direct Python Access to MetaTrader 5 Market Streams](https://www.mql5.com/en/articles/19065)

Harness the full potential of your MetaTrader 5 terminal by leveraging Python’s data-science ecosystem and the official MetaTrader 5 client library. This article demonstrates how to authenticate and stream live tick and minute-bar data directly into Parquet storage, apply sophisticated feature engineering with Ta and Prophet, and train a time-aware Gradient Boosting model. We then deploy a lightweight Flask service to serve trade signals in real time. Whether you’re building a hybrid quant framework or enhancing your EA with machine learning, you’ll walk away with a robust, end-to-end pipeline for data-driven algorithmic trading.

![Integrating MQL5 with data processing packages (Part 5): Adaptive Learning and Flexibility](https://c.mql5.com/2/162/18761-integrating-mql5-with-data-logo.png)[Integrating MQL5 with data processing packages (Part 5): Adaptive Learning and Flexibility](https://www.mql5.com/en/articles/18761)

This part focuses on building a flexible, adaptive trading model trained on historical XAUUSD data, preparing it for ONNX export and potential integration into live trading systems.

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/16439&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5069544491901191878)

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