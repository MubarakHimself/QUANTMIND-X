---
title: Working with ONNX models in float16 and float8 formats
url: https://www.mql5.com/en/articles/14330
categories: Integration, Machine Learning
relevance_score: 0
scraped_at: 2026-01-24T14:04:00.980665
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/14330&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5083312245791791389)

MetaTrader 5 / Integration


### Contents

- [1\. New Data Types for Working with ONNX Models](https://www.mql5.com/en/articles/14330#ch1)
- [1.1. FP16 format](https://www.mql5.com/en/articles/14330#ch1_1)
- [1.1.1. Execution Tests of the ONNX Cast Operator for FLOAT16](https://www.mql5.com/en/articles/14330#ch1_1_1)
- [1.1.2. Execution Tests of the ONNX Cast Operator for BFLOAT16](https://www.mql5.com/en/articles/14330#ch1_1_2)
- [1.2. FP8 format](https://www.mql5.com/en/articles/14330#ch1_2)
- [1.2.1. FP8 formats fp8\_e5m2 and fp8\_e4m3](https://www.mql5.com/en/articles/14330#ch1_2_1)
- [1.2.2. Execution Tests of the ONNX Cast Operator for FLOAT8](https://www.mql5.com/en/articles/14330#ch1_2_2)

- [2\. Using ONNX for Image Super-Resolution](https://www.mql5.com/en/articles/14330#ch2)
- [2.1. Executing ONNX Model with float32](https://www.mql5.com/en/articles/14330#ch2_1)
- [2.2. Executing an ONNX Model with float16](https://www.mql5.com/en/articles/14330#ch2_2)

- [Conclusions](https://www.mql5.com/en/articles/14330#summary)


With the advancement of machine learning and artificial intelligence technologies, there is a growing need to optimize processes for working with models. The efficiency of model operation directly depends on the data formats used to represent them. In recent years, several new data types have emerged, specifically designed for working with deep learning models.

In this article, we will focus on two such new data formats - float16 and float8, which are beginning to be actively used in modern ONNX models. These formats represent alternative options to more precise but resource-intensive floating-point data formats. They provide an optimal balance between performance and accuracy, making them particularly attractive for various machine learning tasks. We will explore the key characteristics and advantages of float16 and float8 formats, as well as introduce functions for converting them to standard float and double formats.

This will help developers and researchers better understand how to effectively use these formats in their projects and models. As an example, we will examine the operation of the ESRGAN ONNX model, which is used for image quality enhancement.

### 1\. New Data Types for Working with ONNX Models

To speed up computations, some models utilize data types with lower precision, such as Float16 and even Float8.

Support for these new data types has been added to work with ONNX models in the MQL5 language, allowing for the manipulation of 8-bit and 16-bit floating-point representations.

The script outputs the full list of elements of the ENUM\_ONNX\_DATA\_TYPE enumeration.

```
//+------------------------------------------------------------------+
//|                                              ONNX_Data_Types.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
   for(int i=0; i<21; i++)
      PrintFormat("%2d %s",i,EnumToString(ENUM_ONNX_DATA_TYPE(i)));
  }
```

Output:

```
 0: ONNX_DATA_TYPE_UNDEFINED
 1: ONNX_DATA_TYPE_FLOAT
 2: ONNX_DATA_TYPE_UINT8
 3: ONNX_DATA_TYPE_INT8
 4: ONNX_DATA_TYPE_UINT16
 5: ONNX_DATA_TYPE_INT16
 6: ONNX_DATA_TYPE_INT32
 7: ONNX_DATA_TYPE_INT64
 8: ONNX_DATA_TYPE_STRING
 9: ONNX_DATA_TYPE_BOOL
10: ONNX_DATA_TYPE_FLOAT16
11: ONNX_DATA_TYPE_DOUBLE
12: ONNX_DATA_TYPE_UINT32
13: ONNX_DATA_TYPE_UINT64
14: ONNX_DATA_TYPE_COMPLEX64
15: ONNX_DATA_TYPE_COMPLEX128
16: ONNX_DATA_TYPE_BFLOAT16
17: ONNX_DATA_TYPE_FLOAT8E4M3FN
18: ONNX_DATA_TYPE_FLOAT8E4M3FNUZ
19: ONNX_DATA_TYPE_FLOAT8E5M2
20: ONNX_DATA_TYPE_FLOAT8E5M2FNUZ
```

Thus, it is now possible to execute ONNX models working with such data.

Moreover, in MQL5, additional functions for data conversion have been added:

```
bool ArrayToFP16(ushort &dst_array[],const float &src_array[],ENUM_FLOAT16_FORMAT fmt);
bool ArrayToFP16(ushort &dst_array[],const double &src_array[],ENUM_FLOAT16_FORMAT fmt);
bool ArrayToFP8(uchar &dst_array[],const float &src_array[],ENUM_FLOAT8_FORMAT fmt);
bool ArrayToFP8(uchar &dst_array[],const double &src_array[],ENUM_FLOAT8_FORMAT fmt);

bool ArrayFromFP16(float &dst_array[],const ushort &src_array[],ENUM_FLOAT16_FORMAT fmt);
bool ArrayFromFP16(double &dst_array[],const ushort &src_array[],ENUM_FLOAT16_FORMAT fmt);
bool ArrayFromFP8(float &dst_array[],const uchar &src_array[],ENUM_FLOAT8_FORMAT fmt);
bool ArrayFromFP8(double &dst_array[],const uchar &src_array[],ENUM_FLOAT8_FORMAT fmt);
```

Since the floating-point formats for 16 and 8 bits may differ, the 'fmt' parameter in the conversion functions must specify which format of number needs to be processed.

For 16-bit versions, a new ENUM\_FLOAT16\_FORMAT enumeration is used, which currently has the following values:

- FLOAT\_FP16 — standard 16-bit format, also known as half float.

- FLOAT\_BFP16 — special [brain float point](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format "https://en.wikipedia.org/wiki/Bfloat16_floating-point_format") floating-point format.


For 8-bit versions, a new ENUM\_FLOAT8\_FORMAT enumeration is used, which currently has the following values:

- FLOAT\_FP8\_E4M3FN — 8-bit floating-point number, 4-bit exponent and 3-bit mantissa. Usually used as coefficients.

- FLOAT\_FP8\_E4M3FNUZ — 8-bit floating-point number, 4-bit exponent and 3-bit mantissa. Supports NaN, does not support negative zero and Inf. Usually used as coefficients.
- FLOAT\_FP8\_E5M2FN — 8-bit floating-point number, 5-bit exponent and 2-bit mantissa. Supports NaN and Inf. Usually used for [gradients](https://www.mql5.com/ru/articles/11200).

- FLOAT\_FP8\_E5M2FNUZ — 8-bit floating-point number, 5-bit exponent and 2-bit mantissa. Supports NaN and Inf, does not support negative zero. Also used for gradients.

### 1.1. FP16 Format

FLOAT16 and BFLOAT16 formats are data types used to represent floating-point numbers.

FLOAT16, also known as ["half-precision floating point"](https://en.wikipedia.org/wiki/Half-precision_floating-point_format "https://en.wikipedia.org/wiki/Half-precision_floating-point_format") format, uses 16 bits to represent a floating-point number. This format provides a balance between precision and computational efficiency. FLOAT16 is widely used in deep learning and neural networks, where high performance is required when processing large volumes of data. This format allows for accelerated computations by reducing the size of numbers, which is particularly important when training deep neural networks on graphics processing units (GPUs).

BFLOAT16 (or [Brain Floating Point 16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format "https://en.wikipedia.org/wiki/Bfloat16_floating-point_format")) also uses 16 bits, but it differs from FLOAT16 in the way numbers are represented. In this format, 8 bits are allocated for representing the exponent, and the remaining 7 bits are used to represent the mantissa. This format was developed for use in deep learning and artificial intelligence, especially in Google Tensor Processing Unit (TPU) processors. BFLOAT16 provides good performance when training neural networks and can be effectively used to accelerate computations.

Both of these formats have their advantages and limitations. FLOAT16 provides higher precision but requires more resources for storage and computations. BFLOAT16, on the other hand, provides higher performance and efficiency when processing data but may be less precise.

![Fig.1. Formats of the bit representation of floating-point numbers FLOAT16 and BFLOAT16](https://c.mql5.com/2/70/float32_2__1.png)

Fig.1. Formats of the bit representation of floating-point numbers FLOAT16 and BFLOAT16

![](https://c.mql5.com/2/70/float16_info.png)

Table 1. Floating-point numbers in FLOAT16 format

**1.1.1. Execution Tests of the ONNX Cast Operator for FLOAT16**

As an illustration, let's consider the task of converting data of type FLOAT16 to types float and double.

ONNX models with the Cast operation:

- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_FLOAT16\_to\_FLOAT](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT16_to_FLOAT")
- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_FLOAT16\_to\_DOUBLE](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT16_to_DOUBLE "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT16_to_DOUBLE")

![Fig.2. Input and output parameters of the model test_cast_FLOAT16_to_DOUBLE.onnx](https://c.mql5.com/2/70/onnx_test_cast_float16_to_double.png)

Fig.2. Input and output parameters of the model test\_cast\_FLOAT16\_to\_DOUBLE.onnx

![Fig.3. Input and output parameters of the model test_cast_FLOAT16_to_FLOAT.onnx](https://c.mql5.com/2/70/onnx_test_cast_float16_to_float.png)

Fig.3. Input and output parameters of the model test\_cast\_FLOAT16\_to\_FLOAT.onnx

As seen from the description of the properties of ONNX models, the input requires data of type ONNX\_DATA\_TYPE\_FLOAT16, and the model will return output data in the ONNX\_DATA\_TYPE\_FLOAT format..

To convert the values, we will use the functions ArrayToFP16() and ArrayFromFP16() with the FLOAT\_FP16 parameter.

Example:

```
//+------------------------------------------------------------------+
//|                                              TestCastFloat16.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#resource "models\\test_cast_FLOAT16_to_DOUBLE.onnx" as const uchar ExtModel1[];
#resource "models\\test_cast_FLOAT16_to_FLOAT.onnx" as const uchar ExtModel2[];

//+------------------------------------------------------------------+
//| union for data conversion                                        |
//+------------------------------------------------------------------+
template<typename T>
union U
  {
   uchar uc[sizeof(T)];
   T value;
  };
//+------------------------------------------------------------------+
//| ArrayToString                                                    |
//+------------------------------------------------------------------+
template<typename T>
string ArrayToString(const T &data[],uint length=16)
  {
   string res;

   for(uint n=0; n<MathMin(length,data.Size()); n++)
      res+="," + StringFormat("%.2x",data[n]);

   StringSetCharacter(res,0,'[');\
   return res+"]";
  }

//+------------------------------------------------------------------+
//| PatchONNXModel                                                   |
//+------------------------------------------------------------------+
void PatchONNXModel(const uchar &original_model[],uchar &patched_model[])
  {
   ArrayCopy(patched_model,original_model,0,0,WHOLE_ARRAY);
//--- special ONNX model patch(IR=9,Opset=20)
   patched_model[1]=0x09;
   patched_model[ArraySize(patched_model)-1]=0x14;
  }
//+------------------------------------------------------------------+
//| CreateModel                                                      |
//+------------------------------------------------------------------+
bool CreateModel(long &model_handle,const uchar &model[])
  {
   model_handle=INVALID_HANDLE;
   ulong flags=ONNX_DEFAULT;
//ulong flags=ONNX_DEBUG_LOGS;
//---
   model_handle=OnnxCreateFromBuffer(model,flags);
   if(model_handle==INVALID_HANDLE)
      return(false);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| PrepareShapes                                                    |
//+------------------------------------------------------------------+
bool PrepareShapes(long model_handle)
  {
   ulong input_shape1[]= {3,4};
   if(!OnnxSetInputShape(model_handle,0,input_shape1))
     {
      PrintFormat("error in OnnxSetInputShape for input1. error code=%d",GetLastError());
      //--
      OnnxRelease(model_handle);
      return(false);
     }
//---
   ulong output_shape[]= {3,4};
   if(!OnnxSetOutputShape(model_handle,0,output_shape))
     {
      PrintFormat("error in OnnxSetOutputShape for output. error code=%d",GetLastError());
      //--
      OnnxRelease(model_handle);
      return(false);
     }
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| RunCastFloat16ToDouble                                           |
//+------------------------------------------------------------------+
bool RunCastFloat16ToDouble(long model_handle)
  {
   PrintFormat("test=%s",__FUNCTION__);

   double test_data[12]= {1,2,3,4,5,6,7,8,9,10,11,12};
   ushort data_uint16[12];
   if(!ArrayToFP16(data_uint16,test_data,FLOAT_FP16))
     {
      Print("error in ArrayToFP16. error code=",GetLastError());
      return(false);
     }
   Print("test array:");
   ArrayPrint(test_data);
   Print("ArrayToFP16:");
   ArrayPrint(data_uint16);

   U<ushort> input_float16_values[3*4];
   U<double> output_double_values[3*4];

   float test_data_float[];
   if(!ArrayFromFP16(test_data_float,data_uint16,FLOAT_FP16))
     {
      Print("error in ArrayFromFP16. error code=",GetLastError());
      return(false);
     }

   for(int i=0; i<12; i++)
     {
      input_float16_values[i].value=data_uint16[i];
      PrintFormat("%d input value =%f  Hex float16 = %s  ushort value=%d",i,test_data_float[i],ArrayToString(input_float16_values[i].uc),input_float16_values[i].value);
     }

   Print("ONNX input array:");
   ArrayPrint(input_float16_values);

   bool res=OnnxRun(model_handle,ONNX_NO_CONVERSION,input_float16_values,output_double_values);
   if(!res)
     {
      PrintFormat("error in OnnxRun. error code=%d",GetLastError());
      return(false);
     }

   Print("ONNX output array:");
   ArrayPrint(output_double_values);
//---
   double sum_error=0.0;
   for(int i=0; i<12; i++)
     {
      double delta=test_data[i]-output_double_values[i].value;
      sum_error+=MathAbs(delta);
      PrintFormat("%d output double %f = %s  difference=%f",i,output_double_values[i].value,ArrayToString(output_double_values[i].uc),delta);
     }
//---
   PrintFormat("test=%s   sum_error=%f",__FUNCTION__,sum_error);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| RunCastFloat16ToFloat                                            |
//+------------------------------------------------------------------+
bool RunCastFloat16ToFloat(long model_handle)
  {
   PrintFormat("test=%s",__FUNCTION__);

   double test_data[12]= {1,2,3,4,5,6,7,8,9,10,11,12};
   ushort data_uint16[12];
   if(!ArrayToFP16(data_uint16,test_data,FLOAT_FP16))
     {
      Print("error in ArrayToFP16. error code=",GetLastError());
      return(false);
     }
   Print("test array:");
   ArrayPrint(test_data);
   Print("ArrayToFP16:");
   ArrayPrint(data_uint16);

   U<ushort> input_float16_values[3*4];
   U<float>  output_float_values[3*4];

   float test_data_float[];
   if(!ArrayFromFP16(test_data_float,data_uint16,FLOAT_FP16))
     {
      Print("error in ArrayFromFP16. error code=",GetLastError());
      return(false);
     }

   for(int i=0; i<12; i++)
     {
      input_float16_values[i].value=data_uint16[i];
      PrintFormat("%d input value =%f  Hex float16 = %s  ushort value=%d",i,test_data_float[i],ArrayToString(input_float16_values[i].uc),input_float16_values[i].value);
     }

   Print("ONNX input array:");
   ArrayPrint(input_float16_values);

   bool res=OnnxRun(model_handle,ONNX_NO_CONVERSION,input_float16_values,output_float_values);
   if(!res)
     {
      PrintFormat("error in OnnxRun. error code=%d",GetLastError());
      return(false);
     }

   Print("ONNX output array:");
   ArrayPrint(output_float_values);
//---
   double sum_error=0.0;
   for(int i=0; i<12; i++)
     {
      double delta=test_data[i]-(double)output_float_values[i].value;
      sum_error+=MathAbs(delta);
      PrintFormat("%d output float %f = %s difference=%f",i,output_float_values[i].value,ArrayToString(output_float_values[i].uc),delta);
     }
//---
   PrintFormat("test=%s   sum_error=%f",__FUNCTION__,sum_error);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| TestCastFloat16ToFloat                                           |
//+------------------------------------------------------------------+
bool TestCastFloat16ToFloat(const uchar &res_model[])
  {
   uchar model[];
   PatchONNXModel(res_model,model);
//--- get model handle
   long model_handle=INVALID_HANDLE;
//--- get model handle
   if(!CreateModel(model_handle,model))
      return(false);
//--- prepare input and output shapes
   if(!PrepareShapes(model_handle))
      return(false);
//--- run ONNX model
   if(!RunCastFloat16ToFloat(model_handle))
      return(false);
//--- release model handle
   OnnxRelease(model_handle);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| TestCastFloat16ToDouble                                          |
//+------------------------------------------------------------------+
bool TestCastFloat16ToDouble(const uchar &res_model[])
  {
   uchar model[];
   PatchONNXModel(res_model,model);
//---
   long model_handle=INVALID_HANDLE;
//--- get model handle
   if(!CreateModel(model_handle,model))
      return(false);
//--- prepare input and output shapes
   if(!PrepareShapes(model_handle))
      return(false);
//--- run ONNX model
   if(!RunCastFloat16ToDouble(model_handle))
      return(false);
//--- release model handle
   OnnxRelease(model_handle);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   if(!TestCastFloat16ToDouble(ExtModel1))
      return 1;

   if(!TestCastFloat16ToFloat(ExtModel2))
      return 1;
//---
   return 0;
  }
//+------------------------------------------------------------------+
```

Output:

```
TestCastFloat16 (EURUSD,H1)     test=RunCastFloat16ToDouble
TestCastFloat16 (EURUSD,H1)     test array:
TestCastFloat16 (EURUSD,H1)      1.00000  2.00000  3.00000  4.00000  5.00000  6.00000  7.00000  8.00000  9.00000 10.00000 11.00000 12.00000
TestCastFloat16 (EURUSD,H1)     ArrayToFP16:
TestCastFloat16 (EURUSD,H1)     15360 16384 16896 17408 17664 17920 18176 18432 18560 18688 18816 18944
TestCastFloat16 (EURUSD,H1)     0 input value =1.000000  Hex float16 = [00,3c]  ushort value=15360
TestCastFloat16 (EURUSD,H1)     1 input value =2.000000  Hex float16 = [00,40]  ushort value=16384
TestCastFloat16 (EURUSD,H1)     2 input value =3.000000  Hex float16 = [00,42]  ushort value=16896
TestCastFloat16 (EURUSD,H1)     3 input value =4.000000  Hex float16 = [00,44]  ushort value=17408
TestCastFloat16 (EURUSD,H1)     4 input value =5.000000  Hex float16 = [00,45]  ushort value=17664
TestCastFloat16 (EURUSD,H1)     5 input value =6.000000  Hex float16 = [00,46]  ushort value=17920
TestCastFloat16 (EURUSD,H1)     6 input value =7.000000  Hex float16 = [00,47]  ushort value=18176
TestCastFloat16 (EURUSD,H1)     7 input value =8.000000  Hex float16 = [00,48]  ushort value=18432
TestCastFloat16 (EURUSD,H1)     8 input value =9.000000  Hex float16 = [80,48]  ushort value=18560
TestCastFloat16 (EURUSD,H1)     9 input value =10.000000  Hex float16 = [00,49]  ushort value=18688
TestCastFloat16 (EURUSD,H1)     10 input value =11.000000  Hex float16 = [80,49]  ushort value=18816
TestCastFloat16 (EURUSD,H1)     11 input value =12.000000  Hex float16 = [00,4a]  ushort value=18944
TestCastFloat16 (EURUSD,H1)     ONNX input array:
TestCastFloat16 (EURUSD,H1)          [uc] [value]
TestCastFloat16 (EURUSD,H1)     [ 0]  ...   15360
TestCastFloat16 (EURUSD,H1)     [ 1]  ...   16384
TestCastFloat16 (EURUSD,H1)     [ 2]  ...   16896
TestCastFloat16 (EURUSD,H1)     [ 3]  ...   17408
TestCastFloat16 (EURUSD,H1)     [ 4]  ...   17664
TestCastFloat16 (EURUSD,H1)     [ 5]  ...   17920
TestCastFloat16 (EURUSD,H1)     [ 6]  ...   18176
TestCastFloat16 (EURUSD,H1)     [ 7]  ...   18432
TestCastFloat16 (EURUSD,H1)     [ 8]  ...   18560
TestCastFloat16 (EURUSD,H1)     [ 9]  ...   18688
TestCastFloat16 (EURUSD,H1)     [10]  ...   18816
TestCastFloat16 (EURUSD,H1)     [11]  ...   18944
TestCastFloat16 (EURUSD,H1)     ONNX output array:
TestCastFloat16 (EURUSD,H1)          [uc]  [value]
TestCastFloat16 (EURUSD,H1)     [ 0]  ...  1.00000
TestCastFloat16 (EURUSD,H1)     [ 1]  ...  2.00000
TestCastFloat16 (EURUSD,H1)     [ 2]  ...  3.00000
TestCastFloat16 (EURUSD,H1)     [ 3]  ...  4.00000
TestCastFloat16 (EURUSD,H1)     [ 4]  ...  5.00000
TestCastFloat16 (EURUSD,H1)     [ 5]  ...  6.00000
TestCastFloat16 (EURUSD,H1)     [ 6]  ...  7.00000
TestCastFloat16 (EURUSD,H1)     [ 7]  ...  8.00000
TestCastFloat16 (EURUSD,H1)     [ 8]  ...  9.00000
TestCastFloat16 (EURUSD,H1)     [ 9]  ... 10.00000
TestCastFloat16 (EURUSD,H1)     [10]  ... 11.00000
TestCastFloat16 (EURUSD,H1)     [11]  ... 12.00000
TestCastFloat16 (EURUSD,H1)     0 output double 1.000000 = [00,00,00,00,00,00,f0,3f]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     1 output double 2.000000 = [00,00,00,00,00,00,00,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     2 output double 3.000000 = [00,00,00,00,00,00,08,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     3 output double 4.000000 = [00,00,00,00,00,00,10,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     4 output double 5.000000 = [00,00,00,00,00,00,14,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     5 output double 6.000000 = [00,00,00,00,00,00,18,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     6 output double 7.000000 = [00,00,00,00,00,00,1c,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     7 output double 8.000000 = [00,00,00,00,00,00,20,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     8 output double 9.000000 = [00,00,00,00,00,00,22,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     9 output double 10.000000 = [00,00,00,00,00,00,24,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     10 output double 11.000000 = [00,00,00,00,00,00,26,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     11 output double 12.000000 = [00,00,00,00,00,00,28,40]  difference=0.000000
TestCastFloat16 (EURUSD,H1)     test=RunCastFloat16ToDouble   sum_error=0.000000
TestCastFloat16 (EURUSD,H1)     test=RunCastFloat16ToFloat
TestCastFloat16 (EURUSD,H1)     test array:
TestCastFloat16 (EURUSD,H1)      1.00000  2.00000  3.00000  4.00000  5.00000  6.00000  7.00000  8.00000  9.00000 10.00000 11.00000 12.00000
TestCastFloat16 (EURUSD,H1)     ArrayToFP16:
TestCastFloat16 (EURUSD,H1)     15360 16384 16896 17408 17664 17920 18176 18432 18560 18688 18816 18944
TestCastFloat16 (EURUSD,H1)     0 input value =1.000000  Hex float16 = [00,3c]  ushort value=15360
TestCastFloat16 (EURUSD,H1)     1 input value =2.000000  Hex float16 = [00,40]  ushort value=16384
TestCastFloat16 (EURUSD,H1)     2 input value =3.000000  Hex float16 = [00,42]  ushort value=16896
TestCastFloat16 (EURUSD,H1)     3 input value =4.000000  Hex float16 = [00,44]  ushort value=17408
TestCastFloat16 (EURUSD,H1)     4 input value =5.000000  Hex float16 = [00,45]  ushort value=17664
TestCastFloat16 (EURUSD,H1)     5 input value =6.000000  Hex float16 = [00,46]  ushort value=17920
TestCastFloat16 (EURUSD,H1)     6 input value =7.000000  Hex float16 = [00,47]  ushort value=18176
TestCastFloat16 (EURUSD,H1)     7 input value =8.000000  Hex float16 = [00,48]  ushort value=18432
TestCastFloat16 (EURUSD,H1)     8 input value =9.000000  Hex float16 = [80,48]  ushort value=18560
TestCastFloat16 (EURUSD,H1)     9 input value =10.000000  Hex float16 = [00,49]  ushort value=18688
TestCastFloat16 (EURUSD,H1)     10 input value =11.000000  Hex float16 = [80,49]  ushort value=18816
TestCastFloat16 (EURUSD,H1)     11 input value =12.000000  Hex float16 = [00,4a]  ushort value=18944
TestCastFloat16 (EURUSD,H1)     ONNX input array:
TestCastFloat16 (EURUSD,H1)          [uc] [value]
TestCastFloat16 (EURUSD,H1)     [ 0]  ...   15360
TestCastFloat16 (EURUSD,H1)     [ 1]  ...   16384
TestCastFloat16 (EURUSD,H1)     [ 2]  ...   16896
TestCastFloat16 (EURUSD,H1)     [ 3]  ...   17408
TestCastFloat16 (EURUSD,H1)     [ 4]  ...   17664
TestCastFloat16 (EURUSD,H1)     [ 5]  ...   17920
TestCastFloat16 (EURUSD,H1)     [ 6]  ...   18176
TestCastFloat16 (EURUSD,H1)     [ 7]  ...   18432
TestCastFloat16 (EURUSD,H1)     [ 8]  ...   18560
TestCastFloat16 (EURUSD,H1)     [ 9]  ...   18688
TestCastFloat16 (EURUSD,H1)     [10]  ...   18816
TestCastFloat16 (EURUSD,H1)     [11]  ...   18944
TestCastFloat16 (EURUSD,H1)     ONNX output array:
TestCastFloat16 (EURUSD,H1)          [uc]  [value]
TestCastFloat16 (EURUSD,H1)     [ 0]  ...  1.00000
TestCastFloat16 (EURUSD,H1)     [ 1]  ...  2.00000
TestCastFloat16 (EURUSD,H1)     [ 2]  ...  3.00000
TestCastFloat16 (EURUSD,H1)     [ 3]  ...  4.00000
TestCastFloat16 (EURUSD,H1)     [ 4]  ...  5.00000
TestCastFloat16 (EURUSD,H1)     [ 5]  ...  6.00000
TestCastFloat16 (EURUSD,H1)     [ 6]  ...  7.00000
TestCastFloat16 (EURUSD,H1)     [ 7]  ...  8.00000
TestCastFloat16 (EURUSD,H1)     [ 8]  ...  9.00000
TestCastFloat16 (EURUSD,H1)     [ 9]  ... 10.00000
TestCastFloat16 (EURUSD,H1)     [10]  ... 11.00000
TestCastFloat16 (EURUSD,H1)     [11]  ... 12.00000
TestCastFloat16 (EURUSD,H1)     0 output float 1.000000 = [00,00,80,3f] difference=0.000000
TestCastFloat16 (EURUSD,H1)     1 output float 2.000000 = [00,00,00,40] difference=0.000000
TestCastFloat16 (EURUSD,H1)     2 output float 3.000000 = [00,00,40,40] difference=0.000000
TestCastFloat16 (EURUSD,H1)     3 output float 4.000000 = [00,00,80,40] difference=0.000000
TestCastFloat16 (EURUSD,H1)     4 output float 5.000000 = [00,00,a0,40] difference=0.000000
TestCastFloat16 (EURUSD,H1)     5 output float 6.000000 = [00,00,c0,40] difference=0.000000
TestCastFloat16 (EURUSD,H1)     6 output float 7.000000 = [00,00,e0,40] difference=0.000000
TestCastFloat16 (EURUSD,H1)     7 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat16 (EURUSD,H1)     8 output float 9.000000 = [00,00,10,41] difference=0.000000
TestCastFloat16 (EURUSD,H1)     9 output float 10.000000 = [00,00,20,41] difference=0.000000
TestCastFloat16 (EURUSD,H1)     10 output float 11.000000 = [00,00,30,41] difference=0.000000
TestCastFloat16 (EURUSD,H1)     11 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat16 (EURUSD,H1)     test=RunCastFloat16ToFloat   sum_error=0.000000
```

**1.1.2. Execution Tests of the ONNX Cast Operator for BFLOAT16**

This example examines the conversion from BFLOAT16 to float.

ONNX model with the Cast operation:

- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_BFLOAT16\_to\_FLOAT](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_BFLOAT16_to_FLOAT "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_BFLOAT16_to_FLOAT")

![Fig.4. Input and output parameters of model test_cast_BFLOAT16_to_FLOAT.onnx](https://c.mql5.com/2/70/onnx_test_cast_bfloat16_to_float.png)

Fig.4. Input and output parameters of model test\_cast\_BFLOAT16\_to\_FLOAT.onnx

Input data of type ONNX\_DATA\_TYPE\_BFLOAT16 is required, and the model will return output data in the format of ONNX\_DATA\_TYPE\_FLOAT.

To convert the values, we will use the functions ArrayToFP16() and ArrayFromFP16() with the parameter BFLOAT\_FP16.

```
//+------------------------------------------------------------------+
//|                                             TestCastBFloat16.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#resource "models\\test_cast_BFLOAT16_to_FLOAT.onnx" as const uchar ExtModel1[];

//+------------------------------------------------------------------+
//| union for data conversion                                        |
//+------------------------------------------------------------------+
template<typename T>
union U
  {
   uchar uc[sizeof(T)];
   T value;
  };
//+------------------------------------------------------------------+
//| ArrayToString                                                    |
//+------------------------------------------------------------------+
template<typename T>
string ArrayToString(const T &data[],uint length=16)
  {
   string res;

   for(uint n=0; n<MathMin(length,data.Size()); n++)
      res+="," + StringFormat("%.2x",data[n]);

   StringSetCharacter(res,0,'[');\
   return res+"]";
  }

//+------------------------------------------------------------------+
//| PatchONNXModel                                                   |
//+------------------------------------------------------------------+
void PatchONNXModel(const uchar &original_model[],uchar &patched_model[])
  {
   ArrayCopy(patched_model,original_model,0,0,WHOLE_ARRAY);
//--- special ONNX model patch(IR=9,Opset=20)
   patched_model[1]=0x09;
   patched_model[ArraySize(patched_model)-1]=0x14;
  }
//+------------------------------------------------------------------+
//| CreateModel                                                      |
//+------------------------------------------------------------------+
bool CreateModel(long &model_handle,const uchar &model[])
  {
   model_handle=INVALID_HANDLE;
   ulong flags=ONNX_DEFAULT;
//ulong flags=ONNX_DEBUG_LOGS;
//---
   model_handle=OnnxCreateFromBuffer(model,flags);
   if(model_handle==INVALID_HANDLE)
      return(false);
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| PrepareShapes                                                    |
//+------------------------------------------------------------------+
bool PrepareShapes(long model_handle)
  {
   ulong input_shape1[]= {3,4};
   if(!OnnxSetInputShape(model_handle,0,input_shape1))
     {
      PrintFormat("error in OnnxSetInputShape for input1. error code=%d",GetLastError());
      //--
      OnnxRelease(model_handle);
      return(false);
     }
//---
   ulong output_shape[]= {3,4};
   if(!OnnxSetOutputShape(model_handle,0,output_shape))
     {
      PrintFormat("error in OnnxSetOutputShape for output. error code=%d",GetLastError());
      //--
      OnnxRelease(model_handle);
      return(false);
     }
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| RunCastBFloat16ToFloat                                           |
//+------------------------------------------------------------------+
bool RunCastBFloat16ToFloat(long model_handle)
  {
   PrintFormat("test=%s",__FUNCTION__);

   double test_data[12]= {1,2,3,4,5,6,7,8,9,10,11,12};
   ushort data_uint16[12];
   if(!ArrayToFP16(data_uint16,test_data,FLOAT_BFP16))
     {
      Print("error in ArrayToFP16. error code=",GetLastError());
      return(false);
     }
   Print("test array:");
   ArrayPrint(test_data);
   Print("ArrayToFP16:");
   ArrayPrint(data_uint16);

   U<ushort> input_float16_values[3*4];
   U<float>  output_float_values[3*4];

   float test_data_float[];
   if(!ArrayFromFP16(test_data_float,data_uint16,FLOAT_BFP16))
     {
      Print("error in ArrayFromFP16. error code=",GetLastError());
      return(false);
     }

   for(int i=0; i<12; i++)
     {
      input_float16_values[i].value=data_uint16[i];
      PrintFormat("%d input value =%f  Hex float16 = %s  ushort value=%d",i,test_data_float[i],ArrayToString(input_float16_values[i].uc),input_float16_values[i].value);
     }

   Print("ONNX input array:");
   ArrayPrint(input_float16_values);

   bool res=OnnxRun(model_handle,ONNX_NO_CONVERSION,input_float16_values,output_float_values);
   if(!res)
     {
      PrintFormat("error in OnnxRun. error code=%d",GetLastError());
      return(false);
     }

   Print("ONNX output array:");
   ArrayPrint(output_float_values);
//---
   double sum_error=0.0;
   for(int i=0; i<12; i++)
     {
      double delta=test_data[i]-(double)output_float_values[i].value;
      sum_error+=MathAbs(delta);
      PrintFormat("%d output float %f = %s difference=%f",i,output_float_values[i].value,ArrayToString(output_float_values[i].uc),delta);
     }
//---
   PrintFormat("test=%s   sum_error=%f",__FUNCTION__,sum_error);
//---
   return(true);
  }

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
   uchar model[];
   PatchONNXModel(ExtModel1,model);
//--- get model handle
   long model_handle=INVALID_HANDLE;
//--- get model handle
   if(!CreateModel(model_handle,model))
      return 1;
//--- prepare input and output shapes
   if(!PrepareShapes(model_handle))
      return 1;
//--- run ONNX model
   if(!RunCastBFloat16ToFloat(model_handle))
      return 1;
//--- release model handle
   OnnxRelease(model_handle);
//---
   return 0;
  }
//+------------------------------------------------------------------+
```

Output:

```
TestCastBFloat16 (EURUSD,H1)    test=RunCastBFloat16ToFloat
TestCastBFloat16 (EURUSD,H1)    test array:
TestCastBFloat16 (EURUSD,H1)     1.00000  2.00000  3.00000  4.00000  5.00000  6.00000  7.00000  8.00000  9.00000 10.00000 11.00000 12.00000
TestCastBFloat16 (EURUSD,H1)    ArrayToFP16:
TestCastBFloat16 (EURUSD,H1)    16256 16384 16448 16512 16544 16576 16608 16640 16656 16672 16688 16704
TestCastBFloat16 (EURUSD,H1)    0 input value =1.000000  Hex float16 = [80,3f]  ushort value=16256
TestCastBFloat16 (EURUSD,H1)    1 input value =2.000000  Hex float16 = [00,40]  ushort value=16384
TestCastBFloat16 (EURUSD,H1)    2 input value =3.000000  Hex float16 = [40,40]  ushort value=16448
TestCastBFloat16 (EURUSD,H1)    3 input value =4.000000  Hex float16 = [80,40]  ushort value=16512
TestCastBFloat16 (EURUSD,H1)    4 input value =5.000000  Hex float16 = [a0,40]  ushort value=16544
TestCastBFloat16 (EURUSD,H1)    5 input value =6.000000  Hex float16 = [c0,40]  ushort value=16576
TestCastBFloat16 (EURUSD,H1)    6 input value =7.000000  Hex float16 = [e0,40]  ushort value=16608
TestCastBFloat16 (EURUSD,H1)    7 input value =8.000000  Hex float16 = [00,41]  ushort value=16640
TestCastBFloat16 (EURUSD,H1)    8 input value =9.000000  Hex float16 = [10,41]  ushort value=16656
TestCastBFloat16 (EURUSD,H1)    9 input value =10.000000  Hex float16 = [20,41]  ushort value=16672
TestCastBFloat16 (EURUSD,H1)    10 input value =11.000000  Hex float16 = [30,41]  ushort value=16688
TestCastBFloat16 (EURUSD,H1)    11 input value =12.000000  Hex float16 = [40,41]  ushort value=16704
TestCastBFloat16 (EURUSD,H1)    ONNX input array:
TestCastBFloat16 (EURUSD,H1)         [uc] [value]
TestCastBFloat16 (EURUSD,H1)    [ 0]  ...   16256
TestCastBFloat16 (EURUSD,H1)    [ 1]  ...   16384
TestCastBFloat16 (EURUSD,H1)    [ 2]  ...   16448
TestCastBFloat16 (EURUSD,H1)    [ 3]  ...   16512
TestCastBFloat16 (EURUSD,H1)    [ 4]  ...   16544
TestCastBFloat16 (EURUSD,H1)    [ 5]  ...   16576
TestCastBFloat16 (EURUSD,H1)    [ 6]  ...   16608
TestCastBFloat16 (EURUSD,H1)    [ 7]  ...   16640
TestCastBFloat16 (EURUSD,H1)    [ 8]  ...   16656
TestCastBFloat16 (EURUSD,H1)    [ 9]  ...   16672
TestCastBFloat16 (EURUSD,H1)    [10]  ...   16688
TestCastBFloat16 (EURUSD,H1)    [11]  ...   16704
TestCastBFloat16 (EURUSD,H1)    ONNX output array:
TestCastBFloat16 (EURUSD,H1)         [uc]  [value]
TestCastBFloat16 (EURUSD,H1)    [ 0]  ...  1.00000
TestCastBFloat16 (EURUSD,H1)    [ 1]  ...  2.00000
TestCastBFloat16 (EURUSD,H1)    [ 2]  ...  3.00000
TestCastBFloat16 (EURUSD,H1)    [ 3]  ...  4.00000
TestCastBFloat16 (EURUSD,H1)    [ 4]  ...  5.00000
TestCastBFloat16 (EURUSD,H1)    [ 5]  ...  6.00000
TestCastBFloat16 (EURUSD,H1)    [ 6]  ...  7.00000
TestCastBFloat16 (EURUSD,H1)    [ 7]  ...  8.00000
TestCastBFloat16 (EURUSD,H1)    [ 8]  ...  9.00000
TestCastBFloat16 (EURUSD,H1)    [ 9]  ... 10.00000
TestCastBFloat16 (EURUSD,H1)    [10]  ... 11.00000
TestCastBFloat16 (EURUSD,H1)    [11]  ... 12.00000
TestCastBFloat16 (EURUSD,H1)    0 output float 1.000000 = [00,00,80,3f] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    1 output float 2.000000 = [00,00,00,40] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    2 output float 3.000000 = [00,00,40,40] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    3 output float 4.000000 = [00,00,80,40] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    4 output float 5.000000 = [00,00,a0,40] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    5 output float 6.000000 = [00,00,c0,40] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    6 output float 7.000000 = [00,00,e0,40] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    7 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    8 output float 9.000000 = [00,00,10,41] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    9 output float 10.000000 = [00,00,20,41] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    10 output float 11.000000 = [00,00,30,41] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    11 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastBFloat16 (EURUSD,H1)    test=RunCastBFloat16ToFloat   sum_error=0.000000
```

### 1.2. FP8 Format

Modern language models can contain billions of parameters. Training models using FP16 numbers has already proven to be effective. Transitioning from 16-bit floating-point numbers to FP8 allows to halve the memory requirements and accelerate training and model execution.

The FP8 format (8-bit floating-point number) is one of the data types used to represent floating-point numbers. In FP8, each number is represented by 8 bits of data, which are typically divided into three components: sign, exponent, and mantissa. This format provides a compromise between precision and storage efficiency, making it attractive for use in applications where memory and computational resources need to be conserved.

One of the key advantages of FP8 is its efficiency in processing large volumes of data. Thanks to its compact representation of numbers, FP8 reduces memory requirements and accelerates calculations. This is particularly important in machine learning and artificial intelligence applications where processing large datasets is common.

Additionally, FP8 can be useful for implementing low-level operations such as arithmetic calculations and signal processing. Its compact format makes it suitable for use in embedded systems and applications where resources are limited. However, it is worth noting that FP8 has its limitations due to its limited precision. In some applications where high precision calculations are required, such as scientific computing or financial analytics, the use of FP8 may be insufficient.

**1.2.1. FP8 formats fp8\_e5m2 and fp8\_e4m3**

In 2022, two articles were published introducing floating-point numbers stored in one byte, unlike float32 numbers, which are stored in 4 bytes.

In the article ["FP8 Formats for Deep Learning"](https://www.mql5.com/go?link=https://arxiv.org/abs/2209.05433 "https://arxiv.org/abs/2209.05433") (2022) by NVIDIA, Intel, and ARM, two types are introduced following IEEE specifications. The first type is E4M3, with 1 bit for the sign, 4 bits for the exponent, and 3 bits for the mantissa. The second type is E5M2, with 1 bit for the sign, 5 bits for the exponent, and 2 bits for the mantissa. The first type is usually used for weights, and the second one for gradients.

The second article, ["8-bit Numerical Formats For Deep Neural Networks"](https://www.mql5.com/go?link=https://arxiv.org/pdf/2206.02915.pdf "https://arxiv.org/pdf/2206.02915.pdf"), presents similar types. The IEEE standard assigns the same value to +0 (or integer 0) and -0 (or integer 128). The article proposes assigning different float values to these two numbers. Additionally, various divisions between the exponent and mantissa are explored, showing that E4M3 and E5M2 are the best.

As a result, ONNX introduced 4 new types (starting from version 1.15.0):

- E4M3FN: 1 bit for the sign, 4 bits for the exponent, 3 bits for the mantissa, only NaN values and no infinite values (FN).
- E4M3FNUZ: 1 bit for the sign, 4 bits for the exponent, 3 bits for the mantissa, only NaN values and no infinite values (FN), no negative zero (UZ).
- E5M2: 1 bit for the sign, 5 bits for the exponent, 2 bits for the mantissa.
- E5M2FNUZ: 1 bit for the sign, 5 bits for the exponent, 2 bits for the mantissa, only NaN values and no infinite values (FN), no negative zero (UZ).

The implementation usually depends on the hardware. NVIDIA, Intel, and Arm implement E4M3FN, while E5M2 is implemented in modern graphics processing units. GraphCore does the same but with E4M3FNUZ and E5M2FNUZ.

Let's briefly summarize the main information about the FP8 type according to the article [NVIDIA Hopper: H100 and FP8 Support](https://www.mql5.com/go?link=https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support "https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support").

![Fig.5. Bit representation of FP8 formats](https://c.mql5.com/2/70/Float8_formats.png)

Fig.5. Bit representation of FP8 formats

![Table 3. Floating point numbers in E5M2 format](https://c.mql5.com/2/70/float8_e5m2.png)

Table 3. Floating point numbers in E5M2 format

![Table 4. Floating point numbers in E4M3 format](https://c.mql5.com/2/70/float8_e4m3.png)

Table 4. Floating point numbers in E4M3 format

Comparison of the ranges of positive values ​​of FP8\_E4M3 and FP8\_E5M2 numbers is shown in the figure 6.

![Fig.6. Comparison of the ranges for positive FP8 numbers](https://c.mql5.com/2/70/float8_ranges.png)

Fig.6. Comparison of the ranges for positive FP8 numbers ( [reference](https://www.mql5.com/go?link=https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support "https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support"))

Comparison of the accuracy of arithmetic operations (Add, Mul, Div) for numbers in FP8\_E5M2 and FP8\_E4M3 formats is shown in the figure 7.

![Fig.7. Comparison of the accuracy of arithmetic operations for numbers in the float8_e5m2 and float8_e4m3 formats](https://c.mql5.com/2/70/float8_compare2.png)

Fig.7. Comparison of the accuracy of arithmetic operations for numbers in the float8\_e5m2 and float8\_e4m3 formats ( [reference](https://www.mql5.com/go?link=https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support "https://lambdalabs.com/blog/nvidia-hopper-h100-and-fp8-support"))

Recommended usage of numbers in the FP8 format:

- E4M3 for weight and activation tensors;
- E5M2 for gradient tensors.

**1.2.2. Execution tests of the ONNX operator Cast for FLOAT8**

This example considers the conversion from various types of FLOAT8 to float.

ONNX models with the Cast operation:

- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_FLOAT8E4M3FN\_to\_FLOAT.onnx](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FN_to_FLOAT.onnx "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FN_to_FLOAT.onnx")
- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_FLOAT8E4M3FNUZ\_to\_FLOAT.onnx](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT.onnx "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E4M3FNUZ_to_FLOAT.onnx")
- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_FLOAT8E5M2\_to\_FLOAT.onnx](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E5M2_to_FLOAT.onnx "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E5M2_to_FLOAT.onnx")
- [https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test\_cast\_FLOAT8E5M2FNUZ\_to\_FLOAT.onnx](https://www.mql5.com/go?link=https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT.onnx "https://github.com/onnx/onnx/tree/main/onnx/backend/test/data/node/test_cast_FLOAT8E5M2FNUZ_to_FLOAT.onnx")

![Fig.8. Input and output parameters of the model test_cast_FLOAT8E4M3FN_to_FLOAT.onnx in MetaEditor](https://c.mql5.com/2/70/test_cast_FLOAT8E4M3FN_to_FLOAT.png)

Fig.8. Input and output parameters of the model test\_cast\_FLOAT8E4M3FN\_to\_FLOAT.onnx in MetaEditor

![Fig.9. Input and output parameters of the model test_cast_FLOAT8E4M3FNUZ_to_FLOAT.onnx in MetaEditor](https://c.mql5.com/2/70/test_cast_FLOAT8E4M3FNUZ_to_FLOAT.png)

Fig.9. Input and output parameters of the model test\_cast\_FLOAT8E4M3FNUZ\_to\_FLOAT.onnx in MetaEditor

![Fig.10. Input and output parameters of the model test_cast_FLOAT8E5M2_to_FLOAT.onnx in MetaEditor](https://c.mql5.com/2/70/test_cast_FLOAT8E5M2_to_FLOAT.png)

Fig.10. Input and output parameters of the model test\_cast\_FLOAT8E5M2\_to\_FLOAT.onnx in MetaEditor

![Fig.11. Input and output parameters of the model test_cast_FLOAT8E5M2FNUZ_to_FLOAT.onnx in MetaEditor](https://c.mql5.com/2/70/test_cast_FLOAT8E5M2FNUZ_to_FLOAT.png)

Fig.11. Input and output parameters of the model test\_cast\_FLOAT8E5M2FNUZ\_to\_FLOAT.onnx in MetaEditor

Example:

```
//+------------------------------------------------------------------+
//|                                              TestCastBFloat8.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#resource "models\\test_cast_FLOAT8E4M3FN_to_FLOAT.onnx" as const uchar ExtModel_FLOAT8E4M3FN_to_FLOAT[];
#resource "models\\test_cast_FLOAT8E4M3FNUZ_to_FLOAT.onnx" as const uchar ExtModel_FLOAT8E4M3FNUZ_to_FLOAT[];
#resource "models\\test_cast_FLOAT8E5M2_to_FLOAT.onnx" as const uchar ExtModel_FLOAT8E5M2_to_FLOAT[];
#resource "models\\test_cast_FLOAT8E5M2FNUZ_to_FLOAT.onnx" as const uchar ExtModel_FLOAT8E5M2FNUZ_to_FLOAT[];

#define TEST_PASSED 0
#define TEST_FAILED 1
//+------------------------------------------------------------------+
//| union for data conversion                                        |
//+------------------------------------------------------------------+
template<typename T>
union U
  {
   uchar uc[sizeof(T)];
   T value;
  };
//+------------------------------------------------------------------+
//| ArrayToHexString                                                 |
//+------------------------------------------------------------------+
template<typename T>
string ArrayToHexString(const T &data[],uint length=16)
  {
   string res;

   for(uint n=0; n<MathMin(length,data.Size()); n++)
      res+="," + StringFormat("%.2x",data[n]);

   StringSetCharacter(res,0,'[');\
   return(res+"]");
  }
//+------------------------------------------------------------------+
//| ArrayToString                                                    |
//+------------------------------------------------------------------+
template<typename T>
string ArrayToString(const U<T> &data[],uint length=16)
  {
   string res;

   for(uint n=0; n<MathMin(length,data.Size()); n++)
      res+="," + (string)data[n].value;

   StringSetCharacter(res,0,'[');\
   return(res+"]");
  }
//+------------------------------------------------------------------+
//| PatchONNXModel                                                   |
//+------------------------------------------------------------------+
long CreatePatchedModel(const uchar &original_model[])
  {
   uchar patched_model[];
   ArrayCopy(patched_model,original_model);
//--- special ONNX model patch(IR=9,Opset=20)
   patched_model[1]=0x09;
   patched_model[ArraySize(patched_model)-1]=0x14;

   return(OnnxCreateFromBuffer(patched_model,ONNX_DEFAULT));
  }
//+------------------------------------------------------------------+
//| PrepareShapes                                                    |
//+------------------------------------------------------------------+
bool PrepareShapes(long model_handle)
  {
//--- configure input shape
   ulong input_shape[]= {3,5};

   if(!OnnxSetInputShape(model_handle,0,input_shape))
     {
      PrintFormat("error in OnnxSetInputShape for input1. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(false);
     }
//--- configure output shape
   ulong output_shape[]= {3,5};

   if(!OnnxSetOutputShape(model_handle,0,output_shape))
     {
      PrintFormat("error in OnnxSetOutputShape for output. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(false);
     }

   return(true);
  }
//+------------------------------------------------------------------+
//| RunCastFloat8Float                                               |
//+------------------------------------------------------------------+
bool RunCastFloat8ToFloat(long model_handle,const ENUM_FLOAT8_FORMAT fmt)
  {
   PrintFormat("TEST: %s(%s)",__FUNCTION__,EnumToString(fmt));
//---
   float test_data[15]   = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
   uchar data_float8[15] = {};

   if(!ArrayToFP8(data_float8,test_data,fmt))
     {
      Print("error in ArrayToFP8. error code=",GetLastError());
      OnnxRelease(model_handle);
      return(false);
     }

   U<uchar> input_float8_values[3*5];
   U<float> output_float_values[3*5];
   float    test_data_float[];
//--- convert float8 to float
   if(!ArrayFromFP8(test_data_float,data_float8,fmt))
     {
      Print("error in ArrayFromFP8. error code=",GetLastError());
      OnnxRelease(model_handle);
      return(false);
     }

   for(uint i=0; i<data_float8.Size(); i++)
     {
      input_float8_values[i].value=data_float8[i];
      PrintFormat("%d input value =%f  Hex float8 = %s  ushort value=%d",i,test_data_float[i],ArrayToHexString(input_float8_values[i].uc),input_float8_values[i].value);
     }

   Print("ONNX input array: ",ArrayToString(input_float8_values));
//--- execute model (convert float8 to float using ONNX)
   if(!OnnxRun(model_handle,ONNX_NO_CONVERSION,input_float8_values,output_float_values))
     {
      PrintFormat("error in OnnxRun. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(false);
     }

   Print("ONNX output array: ",ArrayToString(output_float_values));
//--- calculate error (compare ONNX and ArrayFromFP8 results)
   double sum_error=0.0;

   for(uint i=0; i<test_data.Size(); i++)
     {
      double delta=test_data_float[i]-(double)output_float_values[i].value;
      sum_error+=MathAbs(delta);
      PrintFormat("%d output float %f = %s difference=%f",i,output_float_values[i].value,ArrayToHexString(output_float_values[i].uc),delta);
     }
//---
   PrintFormat("%s(%s): sum_error=%f\n",__FUNCTION__,EnumToString(fmt),sum_error);
   return(true);
  }
//+------------------------------------------------------------------+
//| TestModel                                                        |
//+------------------------------------------------------------------+
bool TestModel(const uchar &model[],const ENUM_FLOAT8_FORMAT fmt)
  {
//--- create patched model
   long model_handle=CreatePatchedModel(model);

   if(model_handle==INVALID_HANDLE)
      return(false);
//--- prepare input and output shapes
   if(!PrepareShapes(model_handle))
      return(false);
//--- run ONNX model
   if(!RunCastFloat8ToFloat(model_handle,fmt))
      return(false);
//--- release model handle
   OnnxRelease(model_handle);

   return(true);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
//--- run ONNX model
   if(!TestModel(ExtModel_FLOAT8E4M3FN_to_FLOAT,FLOAT_FP8_E4M3FN))
      return(TEST_FAILED);

//--- run ONNX model
   if(!TestModel(ExtModel_FLOAT8E4M3FNUZ_to_FLOAT,FLOAT_FP8_E4M3FNUZ))
      return(TEST_FAILED);

//--- run ONNX model
   if(!TestModel(ExtModel_FLOAT8E5M2_to_FLOAT,FLOAT_FP8_E5M2FN))
      return(TEST_FAILED);

//--- run ONNX model
   if(!TestModel(ExtModel_FLOAT8E5M2FNUZ_to_FLOAT,FLOAT_FP8_E5M2FNUZ))
      return(TEST_FAILED);

   return(TEST_PASSED);
  }
//+------------------------------------------------------------------+
```

Output:

```
TestCastFloat8 (EURUSD,H1)      TEST: RunCastFloat8ToFloat(FLOAT_FP8_E4M3FN)
TestCastFloat8 (EURUSD,H1)      0 input value =1.000000  Hex float8 = [38]  ushort value=56
TestCastFloat8 (EURUSD,H1)      1 input value =2.000000  Hex float8 = [40]  ushort value=64
TestCastFloat8 (EURUSD,H1)      2 input value =3.000000  Hex float8 = [44]  ushort value=68
TestCastFloat8 (EURUSD,H1)      3 input value =4.000000  Hex float8 = [48]  ushort value=72
TestCastFloat8 (EURUSD,H1)      4 input value =5.000000  Hex float8 = [4a]  ushort value=74
TestCastFloat8 (EURUSD,H1)      5 input value =6.000000  Hex float8 = [4c]  ushort value=76
TestCastFloat8 (EURUSD,H1)      6 input value =7.000000  Hex float8 = [4e]  ushort value=78
TestCastFloat8 (EURUSD,H1)      7 input value =8.000000  Hex float8 = [50]  ushort value=80
TestCastFloat8 (EURUSD,H1)      8 input value =9.000000  Hex float8 = [51]  ushort value=81
TestCastFloat8 (EURUSD,H1)      9 input value =10.000000  Hex float8 = [52]  ushort value=82
TestCastFloat8 (EURUSD,H1)      10 input value =11.000000  Hex float8 = [53]  ushort value=83
TestCastFloat8 (EURUSD,H1)      11 input value =12.000000  Hex float8 = [54]  ushort value=84
TestCastFloat8 (EURUSD,H1)      12 input value =13.000000  Hex float8 = [55]  ushort value=85
TestCastFloat8 (EURUSD,H1)      13 input value =14.000000  Hex float8 = [56]  ushort value=86
TestCastFloat8 (EURUSD,H1)      14 input value =15.000000  Hex float8 = [57]  ushort value=87
TestCastFloat8 (EURUSD,H1)      ONNX input array: [56,64,68,72,74,76,78,80,81,82,83,84,85,86,87]
TestCastFloat8 (EURUSD,H1)      ONNX output array: [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]
TestCastFloat8 (EURUSD,H1)      0 output float 1.000000 = [00,00,80,3f] difference=0.000000
TestCastFloat8 (EURUSD,H1)      1 output float 2.000000 = [00,00,00,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      2 output float 3.000000 = [00,00,40,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      3 output float 4.000000 = [00,00,80,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      4 output float 5.000000 = [00,00,a0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      5 output float 6.000000 = [00,00,c0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      6 output float 7.000000 = [00,00,e0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      7 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      8 output float 9.000000 = [00,00,10,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      9 output float 10.000000 = [00,00,20,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      10 output float 11.000000 = [00,00,30,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      11 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      12 output float 13.000000 = [00,00,50,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      13 output float 14.000000 = [00,00,60,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      14 output float 15.000000 = [00,00,70,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      RunCastFloat8ToFloat(FLOAT_FP8_E4M3FN): sum_error=0.000000
TestCastFloat8 (EURUSD,H1)
TestCastFloat8 (EURUSD,H1)      TEST: RunCastFloat8ToFloat(FLOAT_FP8_E4M3FNUZ)
TestCastFloat8 (EURUSD,H1)      0 input value =1.000000  Hex float8 = [40]  ushort value=64
TestCastFloat8 (EURUSD,H1)      1 input value =2.000000  Hex float8 = [48]  ushort value=72
TestCastFloat8 (EURUSD,H1)      2 input value =3.000000  Hex float8 = [4c]  ushort value=76
TestCastFloat8 (EURUSD,H1)      3 input value =4.000000  Hex float8 = [50]  ushort value=80
TestCastFloat8 (EURUSD,H1)      4 input value =5.000000  Hex float8 = [52]  ushort value=82
TestCastFloat8 (EURUSD,H1)      5 input value =6.000000  Hex float8 = [54]  ushort value=84
TestCastFloat8 (EURUSD,H1)      6 input value =7.000000  Hex float8 = [56]  ushort value=86
TestCastFloat8 (EURUSD,H1)      7 input value =8.000000  Hex float8 = [58]  ushort value=88
TestCastFloat8 (EURUSD,H1)      8 input value =9.000000  Hex float8 = [59]  ushort value=89
TestCastFloat8 (EURUSD,H1)      9 input value =10.000000  Hex float8 = [5a]  ushort value=90
TestCastFloat8 (EURUSD,H1)      10 input value =11.000000  Hex float8 = [5b]  ushort value=91
TestCastFloat8 (EURUSD,H1)      11 input value =12.000000  Hex float8 = [5c]  ushort value=92
TestCastFloat8 (EURUSD,H1)      12 input value =13.000000  Hex float8 = [5d]  ushort value=93
TestCastFloat8 (EURUSD,H1)      13 input value =14.000000  Hex float8 = [5e]  ushort value=94
TestCastFloat8 (EURUSD,H1)      14 input value =15.000000  Hex float8 = [5f]  ushort value=95
TestCastFloat8 (EURUSD,H1)      ONNX input array: [64,72,76,80,82,84,86,88,89,90,91,92,93,94,95]
TestCastFloat8 (EURUSD,H1)      ONNX output array: [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0]
TestCastFloat8 (EURUSD,H1)      0 output float 1.000000 = [00,00,80,3f] difference=0.000000
TestCastFloat8 (EURUSD,H1)      1 output float 2.000000 = [00,00,00,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      2 output float 3.000000 = [00,00,40,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      3 output float 4.000000 = [00,00,80,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      4 output float 5.000000 = [00,00,a0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      5 output float 6.000000 = [00,00,c0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      6 output float 7.000000 = [00,00,e0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      7 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      8 output float 9.000000 = [00,00,10,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      9 output float 10.000000 = [00,00,20,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      10 output float 11.000000 = [00,00,30,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      11 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      12 output float 13.000000 = [00,00,50,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      13 output float 14.000000 = [00,00,60,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      14 output float 15.000000 = [00,00,70,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      RunCastFloat8ToFloat(FLOAT_FP8_E4M3FNUZ): sum_error=0.000000
TestCastFloat8 (EURUSD,H1)
TestCastFloat8 (EURUSD,H1)      TEST: RunCastFloat8ToFloat(FLOAT_FP8_E5M2FN)
TestCastFloat8 (EURUSD,H1)      0 input value =1.000000  Hex float8 = [3c]  ushort value=60
TestCastFloat8 (EURUSD,H1)      1 input value =2.000000  Hex float8 = [40]  ushort value=64
TestCastFloat8 (EURUSD,H1)      2 input value =3.000000  Hex float8 = [42]  ushort value=66
TestCastFloat8 (EURUSD,H1)      3 input value =4.000000  Hex float8 = [44]  ushort value=68
TestCastFloat8 (EURUSD,H1)      4 input value =5.000000  Hex float8 = [45]  ushort value=69
TestCastFloat8 (EURUSD,H1)      5 input value =6.000000  Hex float8 = [46]  ushort value=70
TestCastFloat8 (EURUSD,H1)      6 input value =7.000000  Hex float8 = [47]  ushort value=71
TestCastFloat8 (EURUSD,H1)      7 input value =8.000000  Hex float8 = [48]  ushort value=72
TestCastFloat8 (EURUSD,H1)      8 input value =8.000000  Hex float8 = [48]  ushort value=72
TestCastFloat8 (EURUSD,H1)      9 input value =10.000000  Hex float8 = [49]  ushort value=73
TestCastFloat8 (EURUSD,H1)      10 input value =12.000000  Hex float8 = [4a]  ushort value=74
TestCastFloat8 (EURUSD,H1)      11 input value =12.000000  Hex float8 = [4a]  ushort value=74
TestCastFloat8 (EURUSD,H1)      12 input value =12.000000  Hex float8 = [4a]  ushort value=74
TestCastFloat8 (EURUSD,H1)      13 input value =14.000000  Hex float8 = [4b]  ushort value=75
TestCastFloat8 (EURUSD,H1)      14 input value =16.000000  Hex float8 = [4c]  ushort value=76
TestCastFloat8 (EURUSD,H1)      ONNX input array: [60,64,66,68,69,70,71,72,72,73,74,74,74,75,76]
TestCastFloat8 (EURUSD,H1)      ONNX output array: [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,8.0,10.0,12.0,12.0,12.0,14.0,16.0]
TestCastFloat8 (EURUSD,H1)      0 output float 1.000000 = [00,00,80,3f] difference=0.000000
TestCastFloat8 (EURUSD,H1)      1 output float 2.000000 = [00,00,00,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      2 output float 3.000000 = [00,00,40,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      3 output float 4.000000 = [00,00,80,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      4 output float 5.000000 = [00,00,a0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      5 output float 6.000000 = [00,00,c0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      6 output float 7.000000 = [00,00,e0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      7 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      8 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      9 output float 10.000000 = [00,00,20,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      10 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      11 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      12 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      13 output float 14.000000 = [00,00,60,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      14 output float 16.000000 = [00,00,80,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      RunCastFloat8ToFloat(FLOAT_FP8_E5M2FN): sum_error=0.000000
TestCastFloat8 (EURUSD,H1)
TestCastFloat8 (EURUSD,H1)      TEST: RunCastFloat8ToFloat(FLOAT_FP8_E5M2FNUZ)
TestCastFloat8 (EURUSD,H1)      0 input value =1.000000  Hex float8 = [40]  ushort value=64
TestCastFloat8 (EURUSD,H1)      1 input value =2.000000  Hex float8 = [44]  ushort value=68
TestCastFloat8 (EURUSD,H1)      2 input value =3.000000  Hex float8 = [46]  ushort value=70
TestCastFloat8 (EURUSD,H1)      3 input value =4.000000  Hex float8 = [48]  ushort value=72
TestCastFloat8 (EURUSD,H1)      4 input value =5.000000  Hex float8 = [49]  ushort value=73
TestCastFloat8 (EURUSD,H1)      5 input value =6.000000  Hex float8 = [4a]  ushort value=74
TestCastFloat8 (EURUSD,H1)      6 input value =7.000000  Hex float8 = [4b]  ushort value=75
TestCastFloat8 (EURUSD,H1)      7 input value =8.000000  Hex float8 = [4c]  ushort value=76
TestCastFloat8 (EURUSD,H1)      8 input value =8.000000  Hex float8 = [4c]  ushort value=76
TestCastFloat8 (EURUSD,H1)      9 input value =10.000000  Hex float8 = [4d]  ushort value=77
TestCastFloat8 (EURUSD,H1)      10 input value =12.000000  Hex float8 = [4e]  ushort value=78
TestCastFloat8 (EURUSD,H1)      11 input value =12.000000  Hex float8 = [4e]  ushort value=78
TestCastFloat8 (EURUSD,H1)      12 input value =12.000000  Hex float8 = [4e]  ushort value=78
TestCastFloat8 (EURUSD,H1)      13 input value =14.000000  Hex float8 = [4f]  ushort value=79
TestCastFloat8 (EURUSD,H1)      14 input value =16.000000  Hex float8 = [50]  ushort value=80
TestCastFloat8 (EURUSD,H1)      ONNX input array: [64,68,70,72,73,74,75,76,76,77,78,78,78,79,80]
TestCastFloat8 (EURUSD,H1)      ONNX output array: [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,8.0,10.0,12.0,12.0,12.0,14.0,16.0]
TestCastFloat8 (EURUSD,H1)      0 output float 1.000000 = [00,00,80,3f] difference=0.000000
TestCastFloat8 (EURUSD,H1)      1 output float 2.000000 = [00,00,00,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      2 output float 3.000000 = [00,00,40,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      3 output float 4.000000 = [00,00,80,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      4 output float 5.000000 = [00,00,a0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      5 output float 6.000000 = [00,00,c0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      6 output float 7.000000 = [00,00,e0,40] difference=0.000000
TestCastFloat8 (EURUSD,H1)      7 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      8 output float 8.000000 = [00,00,00,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      9 output float 10.000000 = [00,00,20,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      10 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      11 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      12 output float 12.000000 = [00,00,40,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      13 output float 14.000000 = [00,00,60,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      14 output float 16.000000 = [00,00,80,41] difference=0.000000
TestCastFloat8 (EURUSD,H1)      RunCastFloat8ToFloat(FLOAT_FP8_E5M2FNUZ): sum_error=0.000000
TestCastFloat8 (EURUSD,H1)
```

### 2\. Using ONNX for Image Super-Resolution

In this section, we will explore an example of using SRGAN models for enhancing image resolution.

ESRGAN, or Enhanced Super-Resolution Generative Adversarial Networks, is a powerful neural network architecture designed to address the task of image super-resolution. ESRGAN is developed to enhance image quality by increasing their resolution to a higher level. This is achieved by training a deep neural network on a large dataset of low-resolution images and their corresponding high-quality images. ESRGAN employs the architecture of Generative Adversarial Networks (GANs), which consists of two main components: a generator and a discriminator. The generator is responsible for creating high-resolution images, while the discriminator is trained to distinguish between the generated images and real ones.

At the core of the ESRGAN architecture are residual blocks, which help extract and preserve important image features at different levels of abstraction. This enables the network to efficiently restore details and textures in high-quality images.

To achieve high quality and universality in solving the super-resolution task, ESRGAN requires extensive training datasets. This allows the network to learn various styles and characteristics of images, making it more adaptable to different types of input data. ESRGAN can be used to improve image quality in many fields, including photography, medical diagnostics, film and video production, graphic design, and more. Its flexibility and efficiency make it one of the leading methods in the field of image super-resolution.

ESRGAN represents a significant advancement in the field of image processing and artificial intelligence, opening up new possibilities for creating and enhancing images.

**2.1. Executing an ONNX Model with float32**

To execute the example, you need to download the file [https://github.com/amannm/super-resolution-service/blob/main/models/esrgan.onnx](https://www.mql5.com/go?link=https://github.com/amannm/super-resolution-service/blob/main/models/esrgan.onnx "https://github.com/amannm/super-resolution-service/blob/main/models/esrgan.onnx") and copy it to the folder \\MQL5\\Scripts\\models.

The ESRGAN.onnx model contains ~1200 ONNX operations, the initial ones of which are presented in Fig.12.

![Fig.12.](https://c.mql5.com/2/70/ESRGAN_onnx_details.png)

Fig.12. ESRGAN.onnx model description in MetaEditor

![Fig.13. ESRGAN.ONNX model in Netron](https://c.mql5.com/2/70/ESRGAN_onnx_details-Netron.png)

Fig.13. ESRGAN.ONNX model in Netron

The code provided below demonstrates image upscaling by a factor of 4 using ESRGAN.onnx.

It starts by loading the esrgan.onnx model, then selecting and loading the original image in BMP format. After that, the image is converted into separate RGB channels, which are then fed into the model as input. The model performs the process of upscaling the image by a factor of 4, after which the resulting upscaled image undergoes inverse transformation and is prepared for display.

The Canvas library is used for display, and the ONNX Runtime library is used for model execution. Upon program execution, the upscaled image is saved to a file with "\_upscaled" appended to the original file name. Key functions include image preprocessing and postprocessing, as well as model execution for image upscaling.

```
//+------------------------------------------------------------------+
//|                                                       ESRGAN.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| 4x image upscaling demo using ESRGAN                             |
//| esrgan.onnx model from                                           |
//| https://github.com/amannm/super-resolution-service/              |
//+------------------------------------------------------------------+
//| Xintao Wang et al (2018)                                         |
//| ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks|
//| https://arxiv.org/abs/1809.00219                                 |
//+------------------------------------------------------------------+
#resource "models\\esrgan.onnx" as uchar ExtModel[];
#include <Canvas\Canvas.mqh>
//+------------------------------------------------------------------+
//| clamp                                                            |
//+------------------------------------------------------------------+
float clamp(float value, float minValue, float maxValue)
  {
   return MathMin(MathMax(value, minValue), maxValue);
  }
//+------------------------------------------------------------------+
//| Preprocessing                                                    |
//+------------------------------------------------------------------+
bool Preprocessing(float &data[],uint &image_data[],int &image_width,int &image_height)
  {
//--- checkup
   if(image_height==0 || image_width==0)
      return(false);
//--- prepare destination array with separated RGB channels for ONNX model
   int data_count=3*image_width*image_height;

   if(ArrayResize(data,data_count)!=data_count)
     {
      Print("ArrayResize failed");
      return(false);
     }
//--- converting
   for(int y=0; y<image_height; y++)
      for(int x=0; x<image_width; x++)
        {
         //--- load source RGB
         int   offset=y*image_width+x;
         uint  clr   =image_data[offset];
         uchar r     =GETRGBR(clr);
         uchar g     =GETRGBG(clr);
         uchar b     =GETRGBB(clr);
         //--- store RGB components as separated channels
         int offset_ch1=0*image_width*image_height+offset;
         int offset_ch2=1*image_width*image_height+offset;
         int offset_ch3=2*image_width*image_height+offset;

         data[offset_ch1]=r/255.0f;
         data[offset_ch2]=g/255.0f;
         data[offset_ch3]=b/255.0f;
        }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| PostProcessing                                                   |
//+------------------------------------------------------------------+
bool PostProcessing(const float &data[], uint &image_data[], const int &image_width, const int &image_height)
  {
//--- checks
   if(image_height == 0 || image_width == 0)
      return(false);

   int data_count=image_width*image_height;

   if(ArraySize(data)!=3*data_count)
      return(false);
   if(ArrayResize(image_data,data_count)!=data_count)
      return(false);
//---
   for(int y=0; y<image_height; y++)
      for(int x=0; x<image_width; x++)
        {
         int offset    =y*image_width+x;
         int offset_ch1=0*image_width*image_height+offset;
         int offset_ch2=1*image_width*image_height+offset;
         int offset_ch3=2*image_width*image_height+offset;
         //--- rescale to [0..255]
         float r=clamp(data[offset_ch1]*255,0,255);
         float g=clamp(data[offset_ch2]*255,0,255);
         float b=clamp(data[offset_ch3]*255,0,255);
         //--- set color image_data
         image_data[offset]=XRGB(uchar(r),uchar(g),uchar(b));
        }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| ShowImage                                                        |
//+------------------------------------------------------------------+
bool ShowImage(CCanvas &canvas,const string name,const int x0,const int y0,const int image_width,const int image_height, const uint &image_data[])
  {
   if(ArraySize(image_data)==0 || name=="")
      return(false);
//--- prepare canvas
   canvas.CreateBitmapLabel(name,x0,y0,image_width,image_height,COLOR_FORMAT_XRGB_NOALPHA);
//--- copy image to canvas
   for(int y=0; y<image_height; y++)
      for(int x=0; x<image_width; x++)
         canvas.PixelSet(x,y,image_data[y*image_width+x]);
//--- ready to draw
   canvas.Update(true);
   return(true);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
//--- select BMP from <data folder>\MQL5\Files
   string image_path[1];

   if(FileSelectDialog("Select BMP image",NULL,"Bitmap files (*.bmp)|*.bmp",FSD_FILE_MUST_EXIST,image_path,"lenna-original4.bmp")!=1)
     {
      Print("file not selected");
      return(-1);
     }
//--- load BMP into array
   uint image_data[];
   int  image_width;
   int  image_height;

   if(!CCanvas::LoadBitmap(image_path[0],image_data,image_width,image_height))
     {
      PrintFormat("CCanvas::LoadBitmap failed with error %d",GetLastError());
      return(-1);
     }
//--- convert RGB image to separated RGB channels
   float input_data[];
   Preprocessing(input_data,image_data,image_width,image_height);
   PrintFormat("input array size=%d",ArraySize(input_data));
//--- load model
   long model_handle=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);

   if(model_handle==INVALID_HANDLE)
     {
      PrintFormat("OnnxCreate error %d",GetLastError());
      return(-1);
     }

   PrintFormat("model loaded successfully");
   PrintFormat("original:  width=%d, height=%d  Size=%d",image_width,image_height,ArraySize(image_data));
//--- set input shape
   ulong input_shape[]={1,3,image_height,image_width};

   if(!OnnxSetInputShape(model_handle,0,input_shape))
     {
      PrintFormat("error in OnnxSetInputShape. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(-1);
     }
//--- upscaled image size
   int   new_image_width =4*image_width;
   int   new_image_height=4*image_height;
   ulong output_shape[]= {1,3,new_image_height,new_image_width};

   if(!OnnxSetOutputShape(model_handle,0,output_shape))
     {
      PrintFormat("error in OnnxSetOutputShape. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(-1);
     }
//--- run the model
   float output_data[];
   int new_data_count=3*new_image_width*new_image_height;
   if(ArrayResize(output_data,new_data_count)!=new_data_count)
     {
      OnnxRelease(model_handle);
      return(-1);
     }

   if(!OnnxRun(model_handle,ONNX_DEBUG_LOGS,input_data,output_data))
     {
      PrintFormat("error in OnnxRun. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(-1);
     }

   Print("model successfully executed, output data size ",ArraySize(output_data));
   OnnxRelease(model_handle);
//--- postprocessing
   uint new_image[];
   PostProcessing(output_data,new_image,new_image_width,new_image_height);
//--- show images
   CCanvas canvas_original,canvas_scaled;
   ShowImage(canvas_original,"original_image",new_image_width,0,image_width,image_height,image_data);
   ShowImage(canvas_scaled,"upscaled_image",0,0,new_image_width,new_image_height,new_image);
//--- save upscaled image
   StringReplace(image_path[0],".bmp","_upscaled.bmp");
   Print(ResourceSave(canvas_scaled.ResourceName(),image_path[0]));
//---
   while(!IsStopped())
      Sleep(100);

   return(0);
  }
//+------------------------------------------------------------------+
```

Output:

[![Fig.14. The result of the ESRGAN.onnx model execution](https://c.mql5.com/2/70/lenna-ESRGAN-example.png)](https://c.mql5.com/2/70/lenna-ESRGAN-example.png "https://c.mql5.com/2/70/lenna-ESRGAN-example.png")

Fig.14. The result of the ESRGAN.onnx model execution (160x200 -> 640x800)

In this example, the 160x200 image was enlarged four times (to 640x800) using the ESRGAN.onnx model.

**2.2. Example of executing an ONNX model with float16**

To convert models to float16, we will use the method described in [Create Float16 and Mixed Precision Models](https://www.mql5.com/go?link=https://onnxruntime.ai/docs/performance/model-optimizations/float16.html "https://onnxruntime.ai/docs/performance/model-optimizations/float16.html").

```
# Copyright 2024, MetaQuotes Ltd.
# https://www.mql5.com

import onnx
from onnxconverter_common import float16

from sys import argv

# Define the path for saving the model
data_path = argv[0]
last_index = data_path.rfind("\\") + 1
data_path = data_path[0:last_index]

# convert the model to float16
model_path = data_path+'\\models\\esrgan.onnx'
modelfp16_path = data_path+'\\models\\esrgan_float16.onnx'

model = onnx.load(model_path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, modelfp16_path)
```

After conversion, the file size decreased by half (from 64MB to 32MB).

The changes in the code are minimal.

```
//+------------------------------------------------------------------+
//|                                               ESRGAN_float16.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| 4x image upscaling demo using ESRGAN                             |
//| esrgan.onnx model from                                           |
//| https://github.com/amannm/super-resolution-service/              |
//+------------------------------------------------------------------+
//| Xintao Wang et al (2018)                                         |
//| ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks|
//| https://arxiv.org/abs/1809.00219                                 |
//+------------------------------------------------------------------+
#resource "models\\esrgan_float16.onnx" as uchar ExtModel[];
#include <Canvas\Canvas.mqh>
//+------------------------------------------------------------------+
//| clamp                                                            |
//+------------------------------------------------------------------+
float clamp(float value, float minValue, float maxValue)
  {
   return MathMin(MathMax(value, minValue), maxValue);
  }
//+------------------------------------------------------------------+
//| Preprocessing                                                    |
//+------------------------------------------------------------------+
bool Preprocessing(float &data[],uint &image_data[],int &image_width,int &image_height)
  {
//--- checkup
   if(image_height==0 || image_width==0)
      return(false);
//--- prepare destination array with separated RGB channels for ONNX model
   int data_count=3*image_width*image_height;

   if(ArrayResize(data,data_count)!=data_count)
     {
      Print("ArrayResize failed");
      return(false);
     }
//--- converting
   for(int y=0; y<image_height; y++)
      for(int x=0; x<image_width; x++)
        {
         //--- load source RGB
         int   offset=y*image_width+x;
         uint  clr   =image_data[offset];
         uchar r     =GETRGBR(clr);
         uchar g     =GETRGBG(clr);
         uchar b     =GETRGBB(clr);
         //--- store RGB components as separated channels
         int offset_ch1=0*image_width*image_height+offset;
         int offset_ch2=1*image_width*image_height+offset;
         int offset_ch3=2*image_width*image_height+offset;

         data[offset_ch1]=r/255.0f;
         data[offset_ch2]=g/255.0f;
         data[offset_ch3]=b/255.0f;
        }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| PostProcessing                                                   |
//+------------------------------------------------------------------+
bool PostProcessing(const float &data[], uint &image_data[], const int &image_width, const int &image_height)
  {
//--- checks
   if(image_height == 0 || image_width == 0)
      return(false);

   int data_count=image_width*image_height;

   if(ArraySize(data)!=3*data_count)
      return(false);
   if(ArrayResize(image_data,data_count)!=data_count)
      return(false);
//---
   for(int y=0; y<image_height; y++)
      for(int x=0; x<image_width; x++)
        {
         int offset    =y*image_width+x;
         int offset_ch1=0*image_width*image_height+offset;
         int offset_ch2=1*image_width*image_height+offset;
         int offset_ch3=2*image_width*image_height+offset;
         //--- rescale to [0..255]
         float r=clamp(data[offset_ch1]*255,0,255);
         float g=clamp(data[offset_ch2]*255,0,255);
         float b=clamp(data[offset_ch3]*255,0,255);
         //--- set color image_data
         image_data[offset]=XRGB(uchar(r),uchar(g),uchar(b));
        }
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| ShowImage                                                        |
//+------------------------------------------------------------------+
bool ShowImage(CCanvas &canvas,const string name,const int x0,const int y0,const int image_width,const int image_height, const uint &image_data[])
  {
   if(ArraySize(image_data)==0 || name=="")
      return(false);
//--- prepare canvas
   canvas.CreateBitmapLabel(name,x0,y0,image_width,image_height,COLOR_FORMAT_XRGB_NOALPHA);
//--- copy image to canvas
   for(int y=0; y<image_height; y++)
      for(int x=0; x<image_width; x++)
         canvas.PixelSet(x,y,image_data[y*image_width+x]);
//--- ready to draw
   canvas.Update(true);
   return(true);
  }
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
int OnStart(void)
  {
//--- select BMP from <data folder>\MQL5\Files
   string image_path[1];

   if(FileSelectDialog("Select BMP image",NULL,"Bitmap files (*.bmp)|*.bmp",FSD_FILE_MUST_EXIST,image_path,"lenna.bmp")!=1)
     {
      Print("file not selected");
      return(-1);
     }
//--- load BMP into array
   uint image_data[];
   int  image_width;
   int  image_height;

   if(!CCanvas::LoadBitmap(image_path[0],image_data,image_width,image_height))
     {
      PrintFormat("CCanvas::LoadBitmap failed with error %d",GetLastError());
      return(-1);
     }
//--- convert RGB image to separated RGB channels
   float input_data[];
   Preprocessing(input_data,image_data,image_width,image_height);
   PrintFormat("input array size=%d",ArraySize(input_data));

   ushort input_data_float16[];
   if(!ArrayToFP16(input_data_float16,input_data,FLOAT_FP16))
     {
      Print("error in ArrayToFP16. error code=",GetLastError());
      return(false);
     }
//--- load model
   long model_handle=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(model_handle==INVALID_HANDLE)
     {
      PrintFormat("OnnxCreate error %d",GetLastError());
      return(-1);
     }

   PrintFormat("model loaded successfully");
   PrintFormat("original:  width=%d, height=%d  Size=%d",image_width,image_height,ArraySize(image_data));
//--- set input shape
   ulong input_shape[]={1,3,image_height,image_width};

   if(!OnnxSetInputShape(model_handle,0,input_shape))
     {
      PrintFormat("error in OnnxSetInputShape. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(-1);
     }
//--- upscaled image size
   int   new_image_width =4*image_width;
   int   new_image_height=4*image_height;
   ulong output_shape[]= {1,3,new_image_height,new_image_width};

   if(!OnnxSetOutputShape(model_handle,0,output_shape))
     {
      PrintFormat("error in OnnxSetOutputShape. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(-1);
     }
//--- run the model
   float output_data[];
   ushort output_data_float16[];
   int new_data_count=3*new_image_width*new_image_height;
   if(ArrayResize(output_data_float16,new_data_count)!=new_data_count)
     {
      OnnxRelease(model_handle);
      return(-1);
     }

   if(!OnnxRun(model_handle,ONNX_NO_CONVERSION,input_data_float16,output_data_float16))
     {
      PrintFormat("error in OnnxRun. error code=%d",GetLastError());
      OnnxRelease(model_handle);
      return(-1);
     }

   Print("model successfully executed, output data size ",ArraySize(output_data));
   OnnxRelease(model_handle);

   if(!ArrayFromFP16(output_data,output_data_float16,FLOAT_FP16))
     {
      Print("error in ArrayFromFP16. error code=",GetLastError());
      return(false);
     }
//--- postprocessing
   uint new_image[];
   PostProcessing(output_data,new_image,new_image_width,new_image_height);
//--- show images
   CCanvas canvas_original,canvas_scaled;
   ShowImage(canvas_original,"original_image",new_image_width,0,image_width,image_height,image_data);
   ShowImage(canvas_scaled,"upscaled_image",0,0,new_image_width,new_image_height,new_image);
//--- save upscaled image
   StringReplace(image_path[0],".bmp","_upscaled.bmp");
   Print(ResourceSave(canvas_scaled.ResourceName(),image_path[0]));
//---
   while(!IsStopped())
      Sleep(100);

   return(0);
  }
//+------------------------------------------------------------------+
```

Changes in the code required to execute the model converted to float16 format are highlighted in color.

Output:

![Fig.15. The result of the ESRGAN_float16.onnx model execution](https://c.mql5.com/2/70/lenna-ESRGAN_float16_example__3.png)

Fig.15. The result of the ESRGAN\_float16.onnx model execution (160x200 -> 640x800)

Thus, using float16 numbers instead of float32 allows reducing the size of the ONNX model file by half (from 64MB to 32MB).

When executing models with float16 numbers, the image quality remained the same, making it visually difficult to find differences:

![Fig. 16. Comparison of the results of ESRGAN model operation for float and float16](https://c.mql5.com/2/71/lenna-ESRGAN-compare2.png)

Fig.16. Comparison of the results of ESRGAN model operation for float and float16

The changes in the code are minimal, requiring only attention to the conversion of input and output data.

In this case, after conversion to float16, the model's performance quality did not change significantly. However, when analyzing financial data, it is essential to strive for calculations with the highest possible accuracy.

### Conclusions

The use of new data types for floating-point numbers allows reducing the size of ONNX models without significant loss of quality.

Preprocessing and post-processing of data are significantly simplified by using conversion functions ArrayToFP16/ArrayFromFP16 and ArrayToFP8/ArrayFromFP8.

Minimal changes in the code are required to work with converted ONNX models.

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/14330](https://www.mql5.com/ru/articles/14330)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14330.zip "Download all attachments in the single ZIP archive")

[Codes-float16-float8-MQL5.zip](https://www.mql5.com/en/articles/download/14330/codes-float16-float8-mql5.zip "Download Codes-float16-float8-MQL5.zip")(92.52 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)
- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/463252)**
(6)


![Quantum](https://c.mql5.com/avatar/2012/6/4FEEAD86-80B2.jpg)

**[Quantum](https://www.mql5.com/en/users/quantum)**
\|
27 Feb 2024 at 16:49

**fxsaber [#](https://www.mql5.com/ru/forum/463198#comment_52539026):**

Please add another image of the same size on the right - a quadrupled (instead of one pixel - four (2x2) of the same colour) original image.

![Lenna-ESRGAN-ESRGAN_float and original 4-x-scaled](https://c.mql5.com/3/430/lenna-ESRGAN-compare2.png)

You can replace the code to display it:

```
   //ShowImage(canvas_original,"original_image",new_image_width,0,image_width,image_height,image_data);
   ShowImage4(canvas_original,"original_image",new_image_width,0,image_width,image_height,image_data);
```

```
//+------------------------------------------------------------------+
//| ShowImage4                                                        |
//+------------------------------------------------------------------+
bool ShowImage4(CCanvas &canvas,const string name,const int x0,const int y0,const int image_width,const int image_height, const uint &image_data[])
  {
   if(ArraySize(image_data)==0 || name=="")
      return(false);
//--- prepare canvas
   canvas.CreateBitmapLabel(name,x0,y0,4*image_width,4*image_height,COLOR_FORMAT_XRGB_NOALPHA);
//--- copy image to canvas
   for(int y=0; y<4*image_height-1; y++)
      for(int x=0; x<4*image_width-1; x++)
      {
         uint  clr =image_data[(y/4)*image_width+(x/4)];
         canvas.PixelSet(x,y,clr);
         }
//--- ready to draw
   canvas.Update(true);
   return(true);
  }
```

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
27 Feb 2024 at 17:05

**Quantum [#](https://www.mql5.com/ru/forum/463198#comment_52540426):**

You can replace the code to output it:

Thanks! Reduced by a factor of two at each coordinate, getting the right image as the original.

![](https://c.mql5.com/3/430/lenna-ESRGAN-compare2-2.png)

I thought that float16/32 would become close to the original with this transformation. But they are noticeably better! I.e. UpScale+DownScale >> Original.

ZY Surprised. It seems reasonable to run all old images/videos through such onnx-model.

![fxsaber](https://c.mql5.com/avatar/2019/8/5D67260D-44C9.png)

**[fxsaber](https://www.mql5.com/en/users/fxsaber)**
\|
27 Feb 2024 at 17:11

If the onnx-model input is given the same data, will the output always be the same?

Is there an element of randomness within the onnx-model?

![Quantum](https://c.mql5.com/avatar/2012/6/4FEEAD86-80B2.jpg)

**[Quantum](https://www.mql5.com/en/users/quantum)**
\|
27 Feb 2024 at 17:54

**fxsaber [#](https://www.mql5.com/ru/forum/463198#comment_52540931):**

If the onnx model is fed the same data as input, but the output will always have the same result?

Is there an element of randomness within the onnx-model?

In general, it depends on what operators are used inside the ONNX model.

For this model the result should be the same, it contains deterministic operations (1195 in total)

- [Add](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Add.html "https://onnx.ai/onnx/operators/onnx__Add.html") (93)
- [Cast](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Cast.html "https://onnx.ai/onnx/operators/onnx__Cast.html") (8)
- [Concat](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Concat.html "https://onnx.ai/onnx/operators/onnx__Concat.html") (276)
- [Constant](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Constant.html "https://onnx.ai/onnx/operators/onnx__Constant.html") (94)
- [Conv](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Conv.html "https://onnx.ai/onnx/operators/onnx__Conv.html") (351)
- [LeakyRelu](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__LeakyRelu.html "https://onnx.ai/onnx/operators/onnx__LeakyRelu.html") (279)
- [Mul](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Mul.html "https://onnx.ai/onnx/operators/onnx__Mul.html") (92)
- [Resize](https://www.mql5.com/go?link=https://onnx.ai/onnx/operators/onnx__Resize.html "https://onnx.ai/onnx/operators/onnx__Resize.html") (2).

![Aleksei Kuznetsov](https://c.mql5.com/avatar/2013/10/52601B64-7C6E.jpg)

**[Aleksei Kuznetsov](https://www.mql5.com/en/users/elibrarius)**
\|
28 Feb 2024 at 11:10

Описание float16

[https://ru.wikipedia.org/wiki/%D0%A7%D0%B8%D1%81%D0%BB%D0%BE\_%D0%BF%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BD%D0%BE%D0%B9\_%D1%82%D0%BE%D1%87%D0%BD%D0%BE%D1%81%D1%82%D0%B8](https://ru.wikipedia.org/wiki/%D0%A7%D0%B8%D1%81%D0%BB%D0%BE_%D0%BF%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BD%D0%BE%D0%B9_%D1%82%D0%BE%D1%87%D0%BD%D0%BE%D1%81%D1%82%D0%B8 "https://ru.wikipedia.org/wiki/%D0%A7%D0%B8%D1%81%D0%BB%D0%BE_%D0%BF%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BD%D0%BE%D0%B9_%D1%82%D0%BE%D1%87%D0%BD%D0%BE%D1%81%D1%82%D0%B8")

## Примеры чисел половинной точности

In these examples, floating point numbers are represented in binary. They include the sign bit, exponent, and mantissa.

0 01111 0000000000 = +1 \*215-15 = 1

0 01111 0000000001 = +1.0000000001 2 \*215-15=1\+  2-10 = 1.0009765625 (the next higher number after 1)

I.e. for numbers with 5 decimal places (most currencies) only 1.00098 can be applied after 1.00000.

Cool! But not for trading and working with quotes.

![Quantization in machine learning (Part 1): Theory, sample code, analysis of implementation in CatBoost](https://c.mql5.com/2/59/Quantization_in_machine_learning_logo.png)[Quantization in machine learning (Part 1): Theory, sample code, analysis of implementation in CatBoost](https://www.mql5.com/en/articles/13219)

The article considers the theoretical application of quantization in the construction of tree models and showcases the implemented quantization methods in CatBoost. No complex mathematical equations are used.

![Experiments with neural networks (Part 7): Passing indicators](https://c.mql5.com/2/59/Experiments_with__networks_logoup.png)[Experiments with neural networks (Part 7): Passing indicators](https://www.mql5.com/en/articles/13598)

Examples of passing indicators to a perceptron. The article describes general concepts and showcases the simplest ready-made Expert Advisor followed by the results of its optimization and forward test.

![Creating multi-symbol, multi-period indicators](https://c.mql5.com/2/59/multi-period_indicators_logo.png)[Creating multi-symbol, multi-period indicators](https://www.mql5.com/en/articles/13578)

In this article, we will look at the principles of creating multi-symbol, multi-period indicators. We will also see how to access the data of such indicators from Expert Advisors and other indicators. We will consider the main features of using multi-indicators in Expert Advisors and indicators and will see how to plot them through custom indicator buffers.

![Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://c.mql5.com/2/59/Online_Decision_Transformer_logo_up.png)[Neural networks made easy (Part 60): Online Decision Transformer (ODT)](https://www.mql5.com/en/articles/13596)

The last two articles were devoted to the Decision Transformer method, which models action sequences in the context of an autoregressive model of desired rewards. In this article, we will look at another optimization algorithm for this method.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=misniactixhbxjpaapbtbnazeaccfwvm&ssn=1769252638561580249&ssn_dr=0&ssn_sr=0&fv_date=1769252638&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14330&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Working%20with%20ONNX%20models%20in%20float16%20and%20float8%20formats%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176925263840647343&fz_uniq=5083312245791791389&sv=2552)

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