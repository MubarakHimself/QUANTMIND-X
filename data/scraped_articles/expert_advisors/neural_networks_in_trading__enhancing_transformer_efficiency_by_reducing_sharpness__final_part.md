---
title: Neural Networks in Trading: Enhancing Transformer Efficiency by Reducing Sharpness (Final Part)
url: https://www.mql5.com/en/articles/16403
categories: Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T21:22:53.676563
---

[![](https://www.mql5.com/ff/sh/5z040u47jcv59943z2/6c76c03a8b37e08b8655a1a085770b7a.jpg)\\
MetaTrader 5 for iOS and Android\\
\\
Fully featured platform for any devices and web browsers\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=ddonqpipxfqlnsvzlwuowsuwlejpyjxk&s=9daba65b69f40afc3c35f95b1f84ef5824d68c47f29ce96a6dc5b164a2727baa&uid=&ref=https://www.mql5.com/en/articles/16403&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071790119846882939)

MetaTrader 5 / Examples


### Introduction

In the previous [article](https://www.mql5.com/en/articles/16388) we got acquainted with the theoretical aspects of the _SAMformer_ ( _Sharpness-Aware Multivariate Transformer_) framework. It is an innovative model designed to address the inherent limitations of traditional _Transformers_ in long-term forecasting tasks for multivariate time series data. Some of the core issues with vanilla _Transformers_ include high training complexity, poor generalization on small datasets, and a tendency to fall into suboptimal local minima. These limitations hinder the applicability of _Transformer_-based models in scenarios with limited input data and high demands for predictive accuracy.

The key idea behind _SAMformer_ lies in its use of shallow architecture, which reduces computational complexity and helps prevent overfitting. One of its central components is the Sharpness-Aware Minimization ( _SAM_) optimization mechanism, which enhances the model's robustness to slight parameter variations, thereby improving its generalization capability and the quality of the final predictions.

Thanks to these features, _SAMformer_ delivers outstanding forecast performance on both synthetic and real-world time series datasets. The model achieves high accuracy while significantly reducing the number of parameters, making it more efficient and suitable for deployment in resource-constrained environments. These advantages open the door to _SAMformer's_ broad application across domains such as finance, healthcare, supply chain management, and energy—where long-term forecasting plays a crucial role.

The original visualization of the framework is provided below.

![](https://c.mql5.com/2/151/1945916559011__2.png)

We have already begun implementing the proposed approaches. In the previous [article](https://www.mql5.com/en/articles/16388), we introduced new kernels on the _OpenCL_ side. We also discussed enhancements to the fully connected layer. Today, we will continue this work.

### 1\. Convolutional Layer with SAM Optimization

We continue the work we started. As the next step, we are extending the convolutional layer with _SAM_ optimization capabilities. As you might expect, our new class _CNeuronConvSAMOCL_ is implemented as a subclass of the existing convolutional layer _CNeuronConvOCL_. The structure of the new object is presented below.

```
class CNeuronConvSAMOCL    :  public CNeuronConvOCL
  {
protected:
   float             fRho;
   //---
   CBufferFloat      cWeightsSAM;
   CBufferFloat      cWeightsSAMConv;
   //---
   virtual bool      calcEpsilonWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      feedForwardSAM(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);

public:
                     CNeuronConvSAMOCL(void) {  activation = GELU;   }
                    ~CNeuronConvSAMOCL(void) {};
//---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out,
                          uint units_count, uint variables,
                          ENUM_OPTIMIZATION optimization_type, uint batch) override;
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint step, uint window_out,
                          uint units_count, uint variables, float rho,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void)   const         {  return defNeuronConvSAMOCL;                         }
   virtual int       Activation(void)  const    {  return (fRho == 0 ? (int)None : (int)activation);   }
   virtual int       getWeightsSAMIndex(void)   {  return cWeightsSAM.GetIndex();                      }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   //---
   virtual CLayerDescription* GetLayerInfo(void);
   virtual void      SetOpenCL(COpenCLMy *obj);
  };
```

Take note that the presented structure includes two buffers for storing adjusted parameters. One buffer is for outgoing connections, similar to the fully connected layer ( _cWeightsSAM_). And another is for incoming connections ( _cWeightsSAMConv_). Note that the parent class does not explicitly include such duplication of parameter buffers. In fact, the buffer for the outgoing connection weights is defined in the parent fully connected layer.

Here, we faced a design dilemma: whether to inherit from the fully connected layer with _SAM_ functionality or from the existing convolutional layer. In the first case, we wouldn't need to define a new buffer for adjusted outgoing connections, as it would be inherited. However, this would require us to completely re-implement the convolutional layer's methods.

In the second scenario, by inheriting from the convolutional layer, we retain all of its existing functionality. However, this approach lacks the buffer for adjusted outgoing weights, which is necessary for the proper operation of the subsequent fully connected SAM-optimized layer.

We chose the second inheritance option, as it required less effort to implement the functionality needed.

As before, we declare additional internal objects statically, allowing us to keep the constructor and destructor empty. Nevertheless, within the class constructor, we set _[GELU](https://www.mql5.com/en/articles/16360#para33)_ as the default activation function. All remaining initialization steps for both inherited and newly declared objects are carried out in the _Init_ method. Here, you'll notice the overriding of two methods with the same name but different parameter sets. We'll first examine the version with the most comprehensive parameter list.

```
bool CNeuronConvSAMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                             uint window_in, uint step, uint window_out,
                             uint units_count, uint variables, float rho,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronConvOCL::Init(numOutputs, myIndex, open_cl, window_in, step, window_out,
                                    units_count, variables, optimization_type, batch))
      return false;
```

In the method parameters, we receive the main constants that allow us to uniquely determine the architecture of the object being created. We immediately pass nearly all of these parameters to the parent class method of the same name, where all necessary control points and initialization algorithms for inherited objects have already been implemented.

After the successful execution of the parent class method, we store the blur region coefficient in an internal variable. This is the only parameter we do not pass to the parent method.

```
   fRho = fabs(rho);
   if(fRho == 0)
      return true;
```

We then immediately check the stored value. If the blur coefficient is zero, the _SAM_ optimization algorithm degenerates into the base parameter optimization method. In that case, all required components have already been initialized by the parent class. So, we can return a successful result.

Otherwise, we first initialize the buffer for the adjusted incoming connections with zero values.

```
   cWeightsSAMConv.BufferFree();
   if(!cWeightsSAMConv.BufferInit(WeightsConv.Total(), 0) ||
      !cWeightsSAMConv.BufferCreate(OpenCL))
      return false;
```

Next, if necessary, we similarly initialize the buffer for adjusted outgoing parameters.

```
   cWeightsSAM.BufferFree();
   if(!Weights)
     return true;
   if(!cWeightsSAM.BufferInit(Weights.Total(), 0) ||
      !cWeightsSAM.BufferCreate(OpenCL))
      return false;
//---
   return true;
  }
```

Note that this last buffer is initialized only if outgoing connection parameters are present. This occurs when the convolutional layer is followed by a fully connected layer.

After successfully initializing all internal components, the method returns the logical result of the operation back to the calling program.

The second initialization method in our class completely overrides the parent class method and has identical parameters. However, as you may have guessed, it omits the blur coefficient parameter, which is critical for _SAM_ optimization. In the method body, we assign a default blur coefficient of 0.7. This coefficient was mentioned in the original paper introducing the _SAMformer_ framework. We then call the previously described class initialization method.

```
bool CNeuronConvSAMOCL::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,

                             uint window_in, uint step, uint window_out,
                             uint units_count, uint variables,
                             ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   return CNeuronConvSAMOCL::Init(numOutputs, myIndex, open_cl, window_in, step, window_out, units_count,
                                                                variables, 0.7f, optimization_type, batch);
  }
```

This approach allows us to easily swap a regular convolutional layer with its _SAM_-optimized counterpart in nearly any of the previously discussed architectural configurations, simply by changing the object type.

As with the fully connected layer, all forward-pass and gradient distribution functionality is inherited from the parent class. However, we introduce two wrapper methods for calling OpenCL program kernels: _calcEpsilonWeights_ and _feedForwardSAM_. The first method calls the kernel responsible for computing the adjusted parameters. The second one mirrors the parent forward-pass method but uses the adjusted parameter buffer instead. We will not go into the detailed logic of these methods here. They follow the same kernel-queuing algorithms discussed earlier. You can explore their full implementations in the attached source code.

The parameter optimization method of this class closely resembles its counterpart in the fully connected SAM-optimized layer. However, in this case, we don't check the type of the preceding layer. Unlike fully connected layers, a convolutional layer contains its own internal parameter matrix applied to input data. Thus, it uses its own adjusted parameter buffer. All it needs from the previous layer is the input data buffer, which all of our objects provide.

Nonetheless, we check the blur coefficient value. When it is zero, SAM optimization is effectively bypassed. In this case, we simply use the parent class method.

```
bool CNeuronConvSAMOCL::updateInputWeights(CNeuronBaseOCL *NeuronOCL)
  {
   if(fRho <= 0)
      return CNeuronConvOCL::updateInputWeights(NeuronOCL);
```

If SAM optimization is enabled, we first combine the error gradient with the feed-forward pass results to produce the current object's target tensor:

```
   if(!SumAndNormilize(Gradient, Output, Gradient, iWindowOut, false, 0, 0, 0, 1))
      return false;
```

Next, we update the model parameters using the blur coefficient. This involves calling the wrapper that enqueues the appropriate kernel. Note that both convolutional and fully connected layers use methods with identical names. But they are queued to different kernels specific to their internal architectures.

```
   if(!calcEpsilonWeights(NeuronOCL))
      return false;
```

The same applies to the feed-forward methods using adjusted parameters.

```
   if(!feedForwardSAM(NeuronOCL))
      return false;
```

After a successful second feed-forward pass, we calculate the deviation from the target values.

```
   float error = 1;
   if(!calcOutputGradients(Gradient, error))
      return false;
```

We then call the parent class’s method to update the model’s parameters.

```
//---
   return CNeuronConvOCL::updateInputWeights(NeuronOCL);
  }
```

Finally, the logical result is returned to the calling program, completing the method.

A few words need to be said about saving the parameters of the trained model. When saving the trained model, we follow the same approach discussed in the context of the fully connected _SAM_ layer. We do not save the buffers containing adjusted parameters. Instead, we only add the blur coefficient to the data saved by the parent class.

```
bool CNeuronConvSAMOCL::Save(const int file_handle)
  {
   if(!CNeuronConvOCL::Save(file_handle))
      return false;
   if(FileWriteFloat(file_handle, fRho) < INT_VALUE)
      return false;
//---
   return true;
  }
```

When loading a pre-trained model, we need to prepare the necessary buffers. It's important to note that the criteria for creating buffers for adjusted incoming and outgoing parameters are different.

First, we load the data saved by the parent class.

```
bool CNeuronConvSAMOCL::Load(const int file_handle)
  {
   if(!CNeuronConvOCL::Load(file_handle))
      return false;
```

Next, we check whether the file contains more data, then read the blur coefficient.

```
   if(FileIsEnding(file_handle))
      return false;
   fRho = FileReadFloat(file_handle);
```

A positive blur coefficient is the key condition for initializing the adjusted parameter buffers. So, we check the value of the loaded parameter. If this condition is not met, we clear any unused buffers in the _OpenCL_ context and in the main memory. After that we complete the method with a positive result.

```
   cWeightsSAMConv.BufferFree();
   cWeightsSAM.BufferFree();
   cWeightsSAMConv.Clear();
   cWeightsSAM.Clear();
   if(fRho <= 0)
      return true;
```

This is one of those cases where the control point is non-critical to program execution. As noted earlier, a zero blur coefficient reduces _SAM_ to a basic optimization method. So, in that case, our object falls back to the functionality of the parent class.

If the condition is satisfied, we proceed to initialize and allocate memory in the _OpenCL_ context for the adjusted incoming parameters.

```
   if(!cWeightsSAMConv.BufferInit(WeightsConv.Total(), 0) ||
      !cWeightsSAMConv.BufferCreate(OpenCL))
      return false;
```

To create the buffer for adjusted outgoing parameters, an additional condition must be met: the presence of such connections. Therefore, we check the pointer validity before initialization.

```
   if(!Weights)
     return true;
```

Again, lack of a valid pointer is not a critical error. It simply reflects the architecture of the model. Therefore, if there is no current pointer, we terminate the method with a positive result.

In an outgoing connection buffer is found, we initialize and create a similarly sized buffer for the adjusted parameters.

```
   if(!cWeightsSAM.BufferInit(Weights.Total(), 0) ||
      !cWeightsSAM.BufferCreate(OpenCL))
      return false;
//---
   return true;
  }
```

Then we return the logical result of the operation to the caller and complete the method execution.

With this, we complete our examination of the convolutional layer methods implementing _SAM_ optimization in _CNeuronConvSAMOCL_. The full code of this class and all its methods can be found in the attachment.

### 2\. Adding SAM to the Transformer

At this stage, we have created both fully connected and convolutional layer objects that incorporate _SAM_-based parameter optimization. It is now time to integrate these approaches into the _Transformer_ architecture. This is exactly as proposed by the authors of the _SAMformer_ framework. To objectively evaluate the impact of these techniques on model performance, we decided not to create entirely new classes. Instead, we integrated the SAM-based approaches directly into the structure of an existing class For the base architecture, we chose the Transformer with relative attention _[R-MAT](https://www.mql5.com/en/articles/16097)_.

As you know, the _CNeuronRMAT_ class implements a linear sequence of alternating _CNeuronRelativeSelfAttention_ and _CResidualConv_ objects. The first implements the relative attention mechanism with feedback, while the second contains a feedback-based convolutional block. To integrate _SAM_ optimization, it is sufficient to replace all convolutional layers in these objects with their _SAM_-enabled counterparts. The updated class structure is shown below.

```
class CNeuronRelativeSelfAttention   :  public CNeuronBaseOCL
  {
protected:
   uint                    iWindow;
   uint                    iWindowKey;
   uint                    iHeads;
   uint                    iUnits;
   int                     iScore;
   //---
   CNeuronConvSAMOCL          cQuery;
   CNeuronConvSAMOCL          cKey;
   CNeuronConvSAMOCL          cValue;
   CNeuronTransposeOCL     cTranspose;
   CNeuronBaseOCL          cDistance;
   CLayer                  cBKey;
   CLayer                  cBValue;
   CLayer                  cGlobalContentBias;
   CLayer                  cGlobalPositionalBias;
   CLayer                  cMHAttentionPooling;
   CLayer                  cScale;
   CBufferFloat            cTemp;
   //---
   virtual bool      AttentionOut(void);
   virtual bool      AttentionGradient(void);

   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      calcInputGradients(CNeuronBaseOCL *NeuronOCL) override;
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL) override;

public:
                     CNeuronRelativeSelfAttention(void) : iScore(-1) {};
                    ~CNeuronRelativeSelfAttention(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_key,
                          uint units_count, uint heads,
                          ENUM_OPTIMIZATION optimization_type, uint batch);
   //---
   virtual int       Type(void) override   const   {  return defNeuronRelativeSelfAttention; }
   //---
   virtual bool      Save(int const file_handle) override;
   virtual bool      Load(int const file_handle) override;
   //---
   virtual bool      WeightsUpdate(CNeuronBaseOCL *source, float tau) override;
   virtual void      SetOpenCL(COpenCLMy *obj) override;
   //---
   virtual uint      GetWindow(void) const { return iWindow; }
   virtual uint      GetUnits(void) const { return iUnits; }
  };
```

```
class CResidualConv  :  public CNeuronBaseOCL
  {
protected:
   int               iWindowOut;
   //---
   CNeuronConvSAMOCL    cConvs[3];
   CNeuronBatchNormOCL cNorm[3];
   CNeuronBaseOCL    cTemp;
   //---
   virtual bool      feedForward(CNeuronBaseOCL *NeuronOCL);
   virtual bool      updateInputWeights(CNeuronBaseOCL *NeuronOCL);
   virtual bool      calcInputGradients(CNeuronBaseOCL *prevLayer);

public:
                     CResidualConv(void) {};
                    ~CResidualConv(void) {};
   //---
   virtual bool      Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                          uint window, uint window_out, uint count,
                          ENUM_OPTIMIZATION optimization_type,
                          uint batch);
   //---
   virtual int       Type(void)   const   {  return defResidualConv;   }
   //--- methods for working with files
   virtual bool      Save(int const file_handle);
   virtual bool      Load(int const file_handle);
   virtual CLayerDescription* GetLayerInfo(void);
   virtual void      SetOpenCL(COpenCLMy *obj);
   virtual void      TrainMode(bool flag);
  };
```

Note that for the feedback convolutional module, we only modify the object type in the class structure. No changes are required to the class methods. This is possible because of our overloaded convolutional layer initialization methods with _SAM_ initialization. Recall that the _CNeuronConvSAMOCL_ class provides two initialization methods: one with the blur coefficient as a parameter and one without it. The method without the blur coefficient overrides the parent class method previously used to initialize convolutional layers. As a result, when initializing _CResidualConv_ objects, the program calls our overridden initialization method, which automatically assigns a default blur coefficient and triggers full convolutional layer initialization with _SAM_ optimization.

The situation with the relative attention module is slightly more complex. The _CNeuronRelativeSelfAttention_ module has a more complex architecture that includes additional nested trainable bias models. Their architecture is defined in the object initialization method. Therefore, to enable _SAM_ optimization for these internal models, we must modify the initialization method of the relative attention module itself.

The method parameters remain unchanged, and the initial steps of its algorithm are also preserved. The object types for generating the _Query_, _Key_, and _Value_ entities have already been updated in the class structure.

```
bool CNeuronRelativeSelfAttention::Init(uint numOutputs, uint myIndex, COpenCLMy *open_cl,
                               uint window, uint window_key, uint units_count, uint heads,
                                          ENUM_OPTIMIZATION optimization_type, uint batch)
  {
   if(!CNeuronBaseOCL::Init(numOutputs, myIndex, open_cl, window * units_count, optimization_type, batch))
      return false;
//---
   iWindow = window;
   iWindowKey = window_key;
   iUnits = units_count;
   iHeads = heads;
//---
   int idx = 0;
   if(!cQuery.Init(0, idx, OpenCL, iWindow, iWindow, iWindowKey * iHeads, iUnits, 1, optimization, iBatch))
      return false;
   cQuery.SetActivationFunction(GELU);
   idx++;
   if(!cKey.Init(0, idx, OpenCL, iWindow, iWindow, iWindowKey * iHeads, iUnits, 1, optimization, iBatch))
      return false;
   cKey.SetActivationFunction(GELU);
   idx++;
   if(!cValue.Init(0, idx, OpenCL, iWindow, iWindow, iWindowKey * iHeads, iUnits, 1, optimization, iBatch))
      return false;
   cKey.SetActivationFunction(GELU);
   idx++;
   if(!cTranspose.Init(0, idx, OpenCL, iUnits, iWindow, optimization, iBatch))
      return false;
   idx++;
   if(!cDistance.Init(0, idx, OpenCL, iUnits * iUnits, optimization, iBatch))
      return false;
```

Further, in the _BKey_ and _BValue_ bias generation models, we substitute convolutional object types while maintaining other parameters.

```
   idx++;
   CNeuronConvSAMOCL *conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iUnits, iUnits, iWindow, iUnits, 1, optimization, iBatch) ||
      !cBKey.Add(conv))
      return false;
   idx++;
   conv.SetActivationFunction(TANH);
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iWindow, iWindow, iWindowKey * iHeads, iUnits, 1, optimization, iBatch) ||
      !cBKey.Add(conv))
      return false;
```

```
   idx++;
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iUnits, iUnits, iWindow, iUnits, 1, optimization, iBatch) ||
      !cBValue.Add(conv))
      return false;
   idx++;
   conv.SetActivationFunction(TANH);
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iWindow, iWindow, iWindowKey * iHeads, iUnits, 1, optimization, iBatch) ||
      !cBValue.Add(conv))
      return false;
```

In the models for generating global context and position biases, we use fully connected layers with _SAM_ optimization.

```
   idx++;
   CNeuronBaseOCL *neuron = new CNeuronBaseSAMOCL();
   if(!neuron ||
      !neuron.Init(iWindowKey * iHeads * iUnits, idx, OpenCL, 1, optimization, iBatch) ||
      !cGlobalContentBias.Add(neuron))
      return false;
   idx++;
   CBufferFloat *buffer = neuron.getOutput();
   buffer.BufferInit(1, 1);
   if(!buffer.BufferWrite())
      return false;
   neuron = new CNeuronBaseSAMOCL();
   if(!neuron ||
      !neuron.Init(0, idx, OpenCL, iWindowKey * iHeads * iUnits, optimization, iBatch) ||
      !cGlobalContentBias.Add(neuron))
      return false;
```

```
   idx++;
   neuron = new CNeuronBaseSAMOCL();
   if(!neuron ||
      !neuron.Init(iWindowKey * iHeads * iUnits, idx, OpenCL, 1, optimization, iBatch) ||
      !cGlobalPositionalBias.Add(neuron))
      return false;
   idx++;
   buffer = neuron.getOutput();
   buffer.BufferInit(1, 1);
   if(!buffer.BufferWrite())
      return false;
   neuron = new CNeuronBaseSAMOCL();
   if(!neuron ||
      !neuron.Init(0, idx, OpenCL, iWindowKey * iHeads * iUnits, optimization, iBatch) ||
      !cGlobalPositionalBias.Add(neuron))
      return false;
```

For pooling operation _MLP_, we again use convolutional layers using _SAM_ optimization approaches.

```
   idx++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, idx, OpenCL, iWindowKey * iHeads * iUnits, optimization, iBatch) ||
      !cMHAttentionPooling.Add(neuron)
     )
      return false;
   idx++;
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iWindowKey * iHeads, iWindowKey * iHeads, iWindow, iUnits, 1, optimization, iBatch) ||
      !cMHAttentionPooling.Add(conv)
     )
      return false;
   idx++;
   conv.SetActivationFunction(TANH);
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iWindow, iWindow, iHeads, iUnits, 1, optimization, iBatch) ||
      !cMHAttentionPooling.Add(conv)
     )
      return false;
   idx++;
   conv.SetActivationFunction(None);
   CNeuronSoftMaxOCL *softmax = new CNeuronSoftMaxOCL();
   if(!softmax ||
      !softmax.Init(0, idx, OpenCL, iHeads * iUnits, optimization, iBatch) ||
      !cMHAttentionPooling.Add(softmax)
     )
      return false;
   softmax.SetHeads(iUnits);
```

Note that for the first layer, we still use the base fully connected layer. Because it is used solely to store the output of the multi-head attention block.

A similar situation occurs in the scaling block. The first layer remains a base fully connected layer, as it stores the result of multiplying attention weights by the outputs of the multi-head attention block. This is then followed by convolutional layers with _SAM_ optimization.

```
   idx++;
   neuron = new CNeuronBaseOCL();
   if(!neuron ||
      !neuron.Init(0, idx, OpenCL, iWindowKey * iUnits, optimization, iBatch) ||
      !cScale.Add(neuron)
     )
      return false;
   idx++;
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL, iWindowKey, iWindowKey, 2 * iWindow, iUnits, 1, optimization, iBatch) ||
      !cScale.Add(conv)
     )
      return false;
   conv.SetActivationFunction(LReLU);
   idx++;
   conv = new CNeuronConvSAMOCL();
   if(!conv ||
      !conv.Init(0, idx, OpenCL,
2  * iWindow, 2 * iWindow, iWindow, iUnits, 1, optimization, iBatch) ||
      !cScale.Add(conv)
     )
      return false;
   conv.SetActivationFunction(None);
//---
   if(!SetGradient(conv.getGradient(), true))
      return false;
//---
   SetOpenCL(OpenCL);
//---
   return true;
  }
```

With that, we conclude the integration of SAM optimization approaches into the _Transformer_ with relative attention. The full code for the updated objects is provided in the attachment.

### 3\. Model Architecture

We have created new objects and updated certain existing ones. The next step is to adjust the overall model architecture. Unlike in some recent articles, today's architectural changes are more extensive. We begin with the architecture of the environment Encoder, implemented in the _CreateEncoderDescriptions_ method. As before, this method receives a pointer to a dynamic array where the sequence of model layers will be recorded.

```
bool CreateEncoderDescriptions(CArrayObj *&encoder)
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
```

In the method body, we check the relevance of the received pointer and, if necessary, create a new instance of the dynamic array.

We leave the first 2 layers unchanged. These are the source data and batch normalization layers. The size of these layers is identical and must be sufficient to record the original data tensor.

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
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBatchNormOCL;
   descr.count = prev_count;
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Next, the authors of the SAMformer framework propose using attention by channels. Therefore, we use a data transposition layer that helps us represent the original data as a sequence of attention channels.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronTransposeOCL;
   descr.count = HistoryBars;
   descr.window= BarDescr;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Then we use the relative attention block, into which we have already added _SAM_ optimization approaches.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronRMAT;
   descr.window=HistoryBars;
   descr.count=BarDescr;
   descr.window_out = EmbeddingSize/2;                // Key Dimension
   descr.layers = 1;                                  // Layers
   descr.step = 2;                                    // Heads
   descr.batch = 1e4;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Two important points should be noted here. First, we use channel attention. Therefore, the analysis window equals the depth of the analyzed history, and the number of elements matches the number of independent channels. Second, as proposed by the authors of the SAMformer framework, we use only one attention layer. However, unlike the original implementation, we employ two attention heads. We have also retained the _FeedForward_ block. Although, the framework authors used only one attention head and removed the _FeedForward_ component.

Next, we must reduce the dimensionality of the output tensor to the desired size. This will be done in two stages. First, we apply a convolutional layer with SAM optimization to reduce the dimensionality of the individual channels.

```
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvSAMOCL;
   descr.count = BarDescr;
   descr.window = HistoryBars;
   descr.step = HistoryBars;
   descr.window_out = LatentCount/BarDescr;
   descr.probability = 0.7f;
   descr.activation = GELU;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
```

Then we use a fully connected layer with _SAM_ optimization to obtain a general embedding of the current environmental state of a given size.

```
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.probability = 0.7f;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!encoder.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

In both cases we use _descr.probability_ to specify the blur area coefficient.

The method concludes by returning the logical result of the operation to the caller. The model architecture itself is returned via the dynamic array pointer provided as a parameter.

After defining the architecture of the environment Encoder, we proceed to describe the layers of the _Actor_ and _Critic_ layers. The descriptions of both models are generated in the _CreateDescriptions_ method. Since this method builds two separate model descriptions, its parameters include two pointers to dynamic arrays.

```
bool CreateDescriptions(CArrayObj *&actor, CArrayObj *&critic)
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
   if(!critic)
     {
      critic = new CArrayObj();
      if(!critic)
         return false;
     }
```

Inside the method, we verify the validity of the provided pointers and, if necessary, create new dynamic arrays.

We start with the architecture of the _Actor_. The first layer of this model is implemented as a fully connected layer with _SAM_ optimization. Its size matches the state description vector of the trading account.

```
//--- Actor
   actor.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   int prev_count = descr.count = AccountDescr;
   descr.activation = None;
   descr.probability=0.7f;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

It is worth noting that here, we use a _SAM_-optimized fully connected layer to record the input data. In the environment _Encoder_, a base fully connected layer was used in a similar position. This difference is due to the presence of a subsequent fully connected layer with _SAM_ optimization, which requires the preceding layer to provide a buffer of adjusted parameters for correct operation.

```
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   prev_count = descr.count = EmbeddingSize;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

As in the environment encoder, we use _descr.probability_ to set the blur region coefficient. For all models, we apply a unified coefficient of 0.7.

Two consecutive _SAM_-optimized fully connected layers create embeddings of the current trading account state, which are then concatenated with the corresponding environmental state embedding. This concatenation is performed by a dedicated data concatenation layer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = EmbeddingSize;
   descr.step = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

The result is passed to a decision-making block consisting of three _SAM_-optimized fully connected layers.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = 2 * NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

At the output of the final layer, we generate a tensor that is twice the size of the Actor’s target action vector. This design allows us to incorporate stochasticity into the actions. As before, we achieve this using the latent state layer of an autoencoder.

```
//--- layer 6
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

Recall that the latent layer of an autoencoder splits the input tensor into two parts: the first part contains the mean values of the distributions for each element of the output sequence, and the second part contains the variances of the corresponding distributions. Training these means and variances within the decision-making module enables us to constrain the range of generated random values via the latent layer of the autoencoder, thus introducing stochasticity into the Actor's policy.

It is worth adding that the autoencoder's latent layer generates independent values for each element of the output sequence. However, in our case, we expect a coherent set of parameters for executing a trade: position size, take-profit levels, and stop-loss levels. To ensure consistency among these trade parameters, we employ a _SAM_-optimized convolutional layer that separately analyzes the parameters for long and short trades.

```
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConvSAMOCL;
   descr.count = NActions / 3;
   descr.window = 3;
   descr.step = 3;
   descr.window_out = 3;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

To limit the output domain of this layer, we use a sigmoid activation function.

And the final touch of our Actor model is a frequency-boosted feed-forward prediction layer ( _CNeuronFreDFOCL_), which allows the results of the model to be matched with the target values in the frequency domain.

```
//--- layer 8
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFreDFOCL;
   descr.window = NActions;
   descr.count =  1;
   descr.step = int(false);
   descr.probability = 0.7f;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!actor.Add(descr))
     {
      delete descr;
      return false;
     }
```

The _Critic_ model has a similar architecture. However, instead of describing the state of the account passed to the _Actor_, we feed the model with the parameters of the trading operation generated by the _Actor_. We also use 2 fully connected layers with SAM optimization to obtain trading operation embedding.

```
//--- Critic
   critic.Clear();
//--- Input layer
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   prev_count = descr.count = NActions;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 1
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   prev_count = descr.count = EmbeddingSize;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
```

The trading operation embedding is combined with the environment state embedding in the data concatenation layer.

```
//--- layer 2
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronConcatenate;
   descr.count = LatentCount;
   descr.window = EmbeddingSize;
   descr.step = LatentCount;
   descr.activation = LReLU;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
```

And then we use a decision block of 3 consecutive fully connected layers with _SAM_ optimization. But unlike _Actor_, in this case the stochastic nature of the results is not used.

```
//--- layer 3
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 4
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 5
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = LatentCount;
   descr.activation = SIGMOID;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//--- layer 6
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronBaseSAMOCL;
   descr.count = NRewards;
   descr.activation = None;
   descr.optimization = ADAM;
   descr.probability=0.7f;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
```

On top of the Critic model, we add a forward prediction layer with frequency gain.

```
//--- layer 7
   if(!(descr = new CLayerDescription()))
      return false;
   descr.type = defNeuronFreDFOCL;
   descr.window = NRewards;
   descr.count =  1;
   descr.step = int(false);
   descr.probability = 0.7f;
   descr.activation = None;
   descr.optimization = ADAM;
   if(!critic.Add(descr))
     {
      delete descr;
      return false;
     }
//---
   return true;
  }
```

After completing the generation of the model architecture descriptions, the method terminates by returning the logical result of the operations to the caller. The architecture descriptions themselves are returned via the dynamic array pointers received in the method parameters.

This concludes our work on model construction. The complete architecture can be found in the attachments. There you will also find the full source code for the environment interaction and model training programs, which have been carried over from previous work without modification.

### 4\. Testing

We have carried out a substantial amount of work implementing the approaches proposed by the authors of the _SAMformer_ framework. It is now time to evaluate the effectiveness of our implementation on real historical data. As before, model training was conducted on actual historical data for the EURUSD instrument covering the entire year of 2023. Throughout the experiments, we used the H1 timeframe. All indicator parameters were set to their default values.

As mentioned earlier, the programs responsible for environment interaction and model training remained unchanged. This allows us to reuse the training datasets created earlier for the initial training of our models. Moreover, since the R-MAT framework was chosen as the baseline for incorporating SAM optimization, we decided not to update the training set during model training. Naturally, we expect this choice to have a negative impact on model performance. However, it enables a more direct comparison with the baseline model by removing any influence from changes in the training dataset.

Training for all three models was conducted simultaneously. The results of testing the trained _Actor_ policy are presented below. The testing was performed on real historical data for January 2024, with all other training parameters unchanged.

Before examining the results, I would like to mention several points regarding model training. First, _SAM_ optimization inherently smooths the loss landscape. This, in turn, allows us to consider higher learning rates. While in earlier works we primarily used a learning rate of 3.0e-04, in this case we increased it to 1.0e-03.

Second, the use of only a single attention layer reduced the total number of trainable parameters, helping to offset the computational overhead introduced by the additional feed-forward pass required by _SAM_ optimization.

![](https://c.mql5.com/2/151/2665815134337.png)![](https://c.mql5.com/2/151/1007593159160.png)

As a result of training, we obtained a policy capable of generating profit outside the training dataset. During the testing period, the model executed 19 trades, 11 of which were profitable (57.89%). By comparison, our previously implemented R-MAT model executed 15 trades over the same period, with 9 profitable trades (60.0%). Notably, the total return of the new model was nearly double that of the baseline.

### Conclusion

The _SAMformer_ framework provides an effective solution to the key limitations of the _Transformer_ architecture in the context of long-term forecasting for multivariate time series. A conventional _Transformer_ faces significant challenges, including high training complexity and poor generalization capability, particularly when working with small training datasets.

The core strengths of _SAMformer_ lie in its shallow architecture and the integration of _Sharpness-Aware Minimization_ (SAM). These approaches help the model avoid poor local minima, improve training stability and accuracy, and deliver superior generalization performance.

In the practical portion of our work, we implemented our own interpretation of these methods in _MQL5_ and trained the models on real historical data. The testing results validate the effectiveness of the proposed approaches, showing that their integration can enhance the performance of baseline models without incurring additional training costs. And in some cases, it even allows you to reduce such training costs.

### References

[SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://www.mql5.com/go?link=https://arxiv.org/abs/2402.10198 "SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention")
[Sharpness-Aware Minimization for Efficiently Improving Generalization](https://www.mql5.com/go?link=https://arxiv.org/abs/2010.01412 "Sharpness-Aware Minimization for Efficiently Improving Generalization")
[Other articles from this series](https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles "https://www.mql5.com/en/search#!keyword=Neural%20networks%20made%20easy&module=mql5_module_articles")

### Programs used in the article

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

Original article: [https://www.mql5.com/ru/articles/16403](https://www.mql5.com/ru/articles/16403)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16403.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/16403/mql5.zip "Download MQL5.zip")(2147 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/493056)**
(7)


![Evgeny Belyaev](https://c.mql5.com/avatar/2025/8/68a633a5-c7d1.png)

**[Evgeny Belyaev](https://www.mql5.com/en/users/genino)**
\|
24 Nov 2024 at 23:03

**Dmitriy Gizlyk [#](https://www.mql5.com/ru/forum/476818#comment_55209584):**

Annual income of Russian banks in dolars. Divide by 12 and compare.

In yuan 6, in yuan bonds more than 10.

![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
25 Nov 2024 at 01:46

**Evgeny Belyaev [#](https://www.mql5.com/ru/forum/476818#comment_55210695):**

In renminbi 6, in renminbi bonds more than 10.

But the results of testing on EURUSD and the result in USD are given in the article. At the same time, the load on the deposit is 1-2%. And nobody wrote that it is a grail.

![Evgeny Belyaev](https://c.mql5.com/avatar/2025/8/68a633a5-c7d1.png)

**[Evgeny Belyaev](https://www.mql5.com/en/users/genino)**
\|
26 Nov 2024 at 23:42

**Dmitriy Gizlyk [#](https://www.mql5.com/ru/forum/476818#comment_55211473):**

But the article gives the results of testing on EURUSD and the result in USD. At the same time, the load on the deposit is 1-2%. And no one wrote that it is a grail.

ok. cap in quid in banks give 5%.

![Khaled Ali E Msmly](https://c.mql5.com/avatar/2020/12/5FE5FF28-4741.jpg)

**[Khaled Ali E Msmly](https://www.mql5.com/en/users/kamforex9496)**
\|
12 Aug 2025 at 11:07

Great article, thank you.


![Ivan Butko](https://c.mql5.com/avatar/2017/1/58797422-0BFA.png)

**[Ivan Butko](https://www.mql5.com/en/users/capitalplus)**
\|
13 Aug 2025 at 09:25

**dsplab [#](https://www.mql5.com/ru/forum/476818#comment_55207429):**

Total profit of 0.35% per month? Wouldn't it be more profitable to just put the money in the bank?

![](https://c.mql5.com/3/472/4153861065801.png).

[![](https://c.mql5.com/3/472/4513245387222__1.png)](https://c.mql5.com/3/472/4513245387222.png "https://c.mql5.com/3/472/4513245387222.png")

[![](https://c.mql5.com/3/472/187894868024__1.png)](https://c.mql5.com/3/472/187894868024.png "https://c.mql5.com/3/472/187894868024.png")

![Integrating MQL5 with data processing packages (Part 5): Adaptive Learning and Flexibility](https://c.mql5.com/2/162/18761-integrating-mql5-with-data-logo.png)[Integrating MQL5 with data processing packages (Part 5): Adaptive Learning and Flexibility](https://www.mql5.com/en/articles/18761)

This part focuses on building a flexible, adaptive trading model trained on historical XAUUSD data, preparing it for ONNX export and potential integration into live trading systems.

![Formulating Dynamic Multi-Pair EA (Part 4): Volatility and Risk Adjustment](https://c.mql5.com/2/162/18165-formulating-dynamic-multi-pair-logo__1.png)[Formulating Dynamic Multi-Pair EA (Part 4): Volatility and Risk Adjustment](https://www.mql5.com/en/articles/18165)

This phase fine-tunes your multi-pair EA to adapt trade size and risk in real time using volatility metrics like ATR boosting consistency, protection, and performance across diverse market conditions.

![Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation](https://c.mql5.com/2/162/19077-automating-trading-strategies-logo__1.png)[Automating Trading Strategies in MQL5 (Part 25): Trendline Trader with Least Squares Fit and Dynamic Signal Generation](https://www.mql5.com/en/articles/19077)

In this article, we develop a trendline trader program that uses least squares fit to detect support and resistance trendlines, generating dynamic buy and sell signals based on price touches and open positions based on generated signals.

![Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://c.mql5.com/2/160/19014-mastering-log-records-part-logo.png)[Mastering Log Records (Part 10): Avoiding Log Replay by Implementing a Suppression](https://www.mql5.com/en/articles/19014)

We created a log suppression system in the Logify library. It details how the CLogifySuppression class reduces console noise by applying configurable rules to avoid repetitive or irrelevant messages. We also cover the external configuration framework, validation mechanisms, and comprehensive testing to ensure robustness and flexibility in log capture during bot or indicator development.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=twrxsxszbjlcwmdaismueazsbvifgdda&ssn=1769192572909134400&ssn_dr=0&ssn_sr=0&fv_date=1769192572&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F16403&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20Networks%20in%20Trading%3A%20Enhancing%20Transformer%20Efficiency%20by%20Reducing%20Sharpness%20(Final%20Part)%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176919257229062792&fz_uniq=5071790119846882939&sv=2552)

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