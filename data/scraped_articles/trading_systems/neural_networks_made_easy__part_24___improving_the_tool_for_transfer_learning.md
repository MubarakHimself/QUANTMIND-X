---
title: Neural networks made easy (Part 24): Improving the tool for Transfer Learning
url: https://www.mql5.com/en/articles/11306
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 3
scraped_at: 2026-01-23T19:28:49.346226
---

[![](https://www.mql5.com/ff/sh/dcfwvnr2j2662m6kz2/c0d1e95edf776bf88908b398733d0997.jpg)\\
Trading chats in MQL5 Channels\\
\\
Dozens of channels with market analytics in different languages.\\
\\
Download](https://www.mql5.com/ff/go?link=https://www.metatrader5.com/en/news/2270%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=messenger.for.traders%26utm_content=download.app%26utm_campaign=0524.mql5.channels&a=fbkqsrihzrcaspjwpzqwvwhuwytvekmw&s=58ba7bd7d20708f42b52a0a9fb72b3cddf13cbc212e4450461952955dfcc433c&uid=&ref=https://www.mql5.com/en/articles/11306&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5070319093547996095)

MetaTrader 5 / Trading systems


### Contents

- [Introduction](https://www.mql5.com/en/articles/11306#para1)
- [1\. Displaying complete information about the neural layer](https://www.mql5.com/en/articles/11306#para2)
- [2\. Activation of used/deactivation of unused input fields](https://www.mql5.com/en/articles/11306#para3)
- [3\. Adding keyboard event handling](https://www.mql5.com/en/articles/11306#para4)
- [Conclusion](https://www.mql5.com/en/articles/11306#para5)
- [List of references](https://www.mql5.com/en/articles/11306#para6)
- [Programs used in the article](https://www.mql5.com/en/articles/11306#para7)

### Introduction

In the previous article in this series, we have created a tool to take advantage of the Transfer Learning technology. As a result of the work done, we got a tool that allows the editing of already trained models. With this tool, we can take any number of neural layers from a pre-rained model. Of course, there are limiting conditions. We take only consecutive layers starting from the initial data layer. The reason for this approach lies in the nature of neural networks. They work well only of the initial data is similar to that used when training the model.

Furthermore, the created tool allows not only editing trained models. It also allows creating completely new ones. This will allow to avoid describing the model architecture in the program code. We will only need to describe a model using the tool. Then we will trail and use the model by uploading the created neural network from a file. This enables experimenting with different architectures without changing the program code. This does not even require the recompilation of the program. You will simply need to change the model file.

Such a useful toll should also be as user friendly as possible. Thus, in this article, we will try to improve its usability.

### 1\. Displaying complete information about the neural layer

Let us start improving the tool usability by increasing the amount of information about each neural layer. As you remember, in the last article we collected all possible information about the architecture of each neural layer of the trained model. But the tool showed the user only the neural layer type and the number of output neurons. This is ok when we work with one model and remember its architecture. But when you come to experimenting with a large number of models, this amount of information will obviously not be enough.

On the other hand, more information requires more space on the information board. Probably, it would not be good to add horizontal scrolling to the model info window. Therefore, I decided to display information about each neural layer in several lines. The output information must be easy to read. It should not look like one huge text block difficult to understand. To divide a text into blocks, let us insert visual separators between descriptions of two successive neural layers.

The decision to split the text into several lines seems a simple solution but its implementation process also required non-standard approaches. The point is that we use the CListView list class to display information about the architecture of the model. Each line in it represents a separate element of the list. Also, there is no possibility of both displaying one element in several lines and grouping several elements into one entity. Adding such functionality will require changes to the algorithm and class architecture. In practice, this will result in the creation of a new class of control object. In this case, one could inherit from the CListView class or create a completely new element. But this requires too much effort which I was not planning.

Therefore, I decided to use an already existing class, but with a few tweaks, without making changes to the class code. As mentioned above, we will use separators to visually divide text into blocks for individual neural layers. The separators will split the whole text with the model architecture description into separate neural layer blocks. We will also visually group information for each neural layer.

But in addition to visual grouping, we also need an understanding at the program level to which neural layer a list element belongs. In the previous article, we implemented changes in the number of copied neural layers by selecting a separate neural layer of the trained model with the mouse and deleting the selected layer from the list of neural layers added to the new model. In both cases, we need a clear understanding of the correspondence between the selected element and the specific neural layer.

When adding each element to the list, we specified its text and a numeric value. Usually, a numeric value is used for quick identification of a selected element. Previously, we specified an individual value for each element. But it is also possible to use one value for several elements. Of course, this approach will make it difficult to identify each element of the list. However, we do not this right now. We only need to identify a group of elements. Therefore, using this capability, we can identify not a single element, but a whole group of elements.

```
bool  AddItem(
   const string  item,     // text
   const long    value     // value
   )
```

Actually, this solution provides another advantage. The CListView class has the SelectByValue method. The main purpose of this method is to select an element by its numerical value. Its algorithm finds the first element with the specified numerical value among all elements of the list and selects it. By organizing the handling of the list selection change event, we can read the value of the element selected by the user and ask the class to select the first element from the list with this value. This will visualize the beginning of the group. I think this is a pretty handy feature.

```
bool  SelectByValue(
   const long  value     // value
   )
```

Now, let us loot at the implementation of the described approaches. First of all, we need to implement a textual representation of the neural layer architecture description to display it on the panel. For this purpose, let us create the LayerDescriptionToString method. The method receives in parameters a pointer to the object of the neural layer architecture description and a pointer to the dynamic array of strings to which the textual description of the neural layer will be written. Each element of the array is a separate line in the model architecture description list. In the above terms, each element is a separate group of elements in the list for describing one neural layer. By using a dynamic array, we can organize element groups of different sizes, depending on the need to describe a particular neural layer.

The method will receive the number of elements in the array.

```
int CNetCreatorPanel::LayerDescriptionToString(const CLayerDescription *layer, string& result[])
  {
   if(!layer)
      return -1;
```

In the body of the method, we first check the validity of the received pointer to the description of the neural layer architecture.

Next, we will prepare a local variable and clear the resulting dynamic array.

```
   string temp;
   ArrayFree(result);
```

Next we will create a text description of the neural layer depending on its type. We will not work with the dynamic array of strings right away. Instead, the entire description will be written in one string. But we will insert a separator where the string should be split. In this example I used a backslash "\\". I used the StringFormat function to properly compose text with this markup. The function generates formatted text with minimal effort.

After creating a formatted string description of the neural layer architecture, we will use the StringSplit function and split our text into lines. This function divides the text into lines according to the separator elements which were carefully added to the text in the previous step. The convenience of using this function also lies in the fact that it increases the size of the dynamic array to the required size. SO, we do not need to control this part.

```
   switch(layer.type)
     {
      case defNeuronBaseOCL:
         temp = StringFormat("Dense (outputs %d, \activation %s, \optimization %s)",
                layer.count, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronConvOCL:
         temp = StringFormat("Convolution (outputs %d, \window %d, step %d, window out %d, \activation %s, \optimization %s)",
                layer.count * layer.window_out, layer.window, layer.step, layer.window_out, EnumToString(layer.activation),

                EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronProofOCL:
         temp = StringFormat("Proof (outputs %d, \window %d, step %d, \optimization %s)",
                layer.count, layer.window, layer.step, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronAttentionOCL:
         temp = StringFormat("Self Attention (outputs %d, \units %s, window %d, \optimization %s)",
                layer.count * layer.window, layer.count, layer.window, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronMHAttentionOCL:
         temp = StringFormat("Multi-Head Attention (outputs %d, \units %s, window %d, heads %s, \optimization %s)",
                layer.count * layer.window, layer.count, layer.window, layer.step, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronMLMHAttentionOCL:
         temp = StringFormat("Multi-Layer MH Attention (outputs %d, \units %s, window %d, key size %d, \heads %s, layers %d,
                              \optimization %s)",
                layer.count * layer.window, layer.count, layer.window, layer.window_out, layer.step, layer.layers,

                EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronDropoutOCL:
         temp = StringFormat("Dropout (outputs %d, \probability %d, \optimization %s)",
                layer.count, layer.probability, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronBatchNormOCL:
         temp = StringFormat("Batchnorm (outputs %d, \batch size %d, \optimization %s)",
                layer.count, layer.batch, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronVAEOCL:
         temp = StringFormat("VAE (outputs %d)", layer.count);
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      case defNeuronLSTMOCL:
         temp = StringFormat("LSTM (outputs %d, \optimization %s)", layer.count, EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;

      default:
         temp = StringFormat("Unknown type %#x (outputs %d, \activation %s, \optimization %s)",
                layer.type, layer.count, EnumToString(layer.activation), EnumToString(layer.optimization));
         if(StringSplit(temp, '\\', result) < 0)
            return -1;
         break;
     }
```

After creating descriptions of all known neural layers, do not forget to add one standard description for unknown types. Thus, we can inform the user about the detection of an unknown neural layer and protect from unintentional violation of the model integrity.

At the end of the method, return the size of the array of results to the caller.

```
//---
   return ArraySize(result);
  }
```

Next, we move on to the LoadModel method, which we already discussed met in the previous [article](https://www.mql5.com/en/articles/11273#para33). We will not change the entire method, but only change the body of the loop which adds elements to the list. As before, in the loop body, we first get a pointer to the next layer description object from the dynamic array. Immediately check the validity of the received pointer.

```
   for(int i = 0; i < total; i++)
     {
      CLayerDescription* temp = m_arPTModelDescription.At(i);
      if(!temp)
         return false;
```

Then we will prepare a dynamic array of strings and call the above described LayerDescriptionToString method to generate a text description of the neural layer. After the method completes, we get an array of string descriptions and the number of elements in it. If an error occurs, the method will return an empty array and -1 instead of the array size. Inform the user about the error and complete the method.

```
      string items[];
      int total_items = LayerDescriptionToString(temp, items);
      if(total_items < 0)
        {
         printf("%s %d Error at layer %d: %d", __FUNCSIG__, __LINE__, i, GetLastError());
         return false;
        }
```

If the description text is successfully generated, first add the block separator element. Then, in a nested loop, output the entire content of the text array describing the neural layer.

```
      if(!m_lstPTModel.AddItem(StringFormat("____ Layer %d ____", i + 1), i + 1))
         return false;
      for(int it = 0; it < total_items; it++)
         if(!m_lstPTModel.AddItem(items[it], i + 1))
            return false;
     }
```

Pay attention that that when specifying the group id, we add 1 to the ordinal number of the neural layer in the dynamic array with the model description. This is required because indexing in the array starts with 0. If we specify 0 as a numeric identifier, the CListView class will automatically replace it with the total number of elements in the list. We wouldn't want to receive a random value instead of a group ID.

The rest of the LoadModel method code has not changed. Its full code is provided in the attachment. Also, the attachment contains the codes of all methods and classes used in the program. In particular, you can see similar additions to the method for displaying the description of the new ChangeNumberOfLayers model.

Please note that in the ChangeNumberOfLayers method, the information about the model is collected from two dynamic arrays containing model architecture descriptions. The first one describes the architecture of the donor model. We take the description of copied neural layer from it. The second array contains a description of the neural networks we are adding.

After outputting the model architecture descriptions, move on to the methods processing events of changes in created lists states.

```
ON_EVENT(ON_CHANGE, m_lstPTModel, OnChangeListPTModel)
ON_EVENT(ON_CHANGE, m_lstNewModel, OnChangeListNewModel)
```

As described above, when the user selects any line in the list, we will move the selection to the first line of the specified block. For this, we simply get the group ID of the element selected by the user and instruct the program to select the first element with the given ID. This operation is implemented by the SelectByValue method.

```
bool CNetCreatorPanel::OnChangeListNewModel(void)
  {
   long value = m_lstNewModel.Value();
//---
   return m_lstNewModel.SelectByValue(value);
  }
```

This expands the displayed information about the model architecture. The amount of information is minimally sufficient and specific to a neural layer type. So, the user will only see the relevant information about a specific neural layer. Furthermore, the there is no extra information cluttering up the window.

### 2\. Activation of used/deactivation of unused input fields

The next modification concerns data input fields. Strange as it may seem, but they provide a rather large field for imagination. Probably the first thing that catches your eye is the amount of input information. The panel provides input fields foe all elements of the CLayerDescription class describing the neural layer architecture. I do not say that it's bad. The user can see all the specified data and change it in any order and whenever needed, before adding a layer. But we know that not all of these fields are relevant for all neural layers.

For example, for a fully connected neural layer, it is enough to specify only three parameters: the number of neurons, the activation function, and the parameter optimization method. The rest of the parameters are irrelevant to it. When dealing with the convolutional neural layer, you need to specify the size of the input data window and its step. The number of output elements will depend on the source data buffer size and the two specified parameters.

In the recurrent LSTM block, the activation functions are defined by the block architecture and thus there is no need to specify them.

Well, the user might know all these features. But a well-designed tool should warn the user against possible "mechanical" errors. There are two preventive options possible. We can remove irrelevant elements from the panel or simply make them uneditable.

Each option has its pros and cons. The advantages of the first option include the reduced the number of input fields on the panel. So, the panel can be more compact. The downside is a more complex implementation. Since we will need to rearrange the elements on the panel each time. At the same time, the constant rearrangement of objects can confuse the user and lead to errors.

In my opinion, the use of this method is justified when you need to enter a large amount of data. Then removing unnecessary objects will make the panel more compact and neater.

The second option is acceptable if we have a small number of elements. We can easily arrange all the elements on the panel at once. Furthermore, we do not confuse the user by moving them around the panel unnecessarily. The user will visually remember their location, which improves overall performance.

We have already placed all the input fields on the interface panel. Therefore, I consider the second implementation option acceptable.

We already have an architectural solution. But we will go a little further. The panel has fields with drop-down lists and direct input fields. The drop-down field allows selecting only one of available options. But in value input fields, the user can physically enter any text.

However, we expect to get an integer value there. Logically, we should add a check of the entered information before passing it to the object describing the architecture of the created neural layer. To share the correctness of information with the user, the entered information will be validated immediately after the user enters the text. After validation, we will replace the information entered in the field by the user with the information accepted by the tool. Thus, the user can see the difference between the entered and read information. If necessary, the user can further correct the data.

And one moment. When describing the neural layer architecture in the CLayerDescription class, we have dual-purpose elements. For example, **_step_** for the convolutional and subsample layers specifies a step of the source data window. But the same parameter is used to specify the number of attention heads when describing attention neural layers.

The window\_out parameter specifies the number of filters in the convolutional layer and the size of the internal key layer in the attention block.

To make the interface more user-friendly, it is better to change the text labels when choosing the appropriate type of neural layer.

The user will not be confused with the problem of rearrangement in the interface window. The field itself does not change. Only information next to it changes. If the user does not pay attention to the new data and automatically enters information into the corresponding field, this will not lead to any errors in the organization of the model. The data in any case will be sent into the desired element of the description of the layer architecture.

To implement the above solutions, we need to take a step back and do some preparatory work.

First of all, when creating text labels on the interface panel, we were not saving pointers to the corresponding objects. Now, when we need to change the text of some of them, we will have to look for them in the general array of objects. To avoid this, let us get back to the CreateLabel text label creation method. Upon completion of the method operations, instead of a logical result, let us return a pointer to the created object.

```
CLabel* CNetCreatorPanel::CreateLabel(const int id, const string text,
                                      const int x1, const int y1,
                                      const int x2, const int y2
                                     )
  {
   CLabel *tmp_label = new CLabel();
   if(!tmp_label)
      return NULL;
   if(!tmp_label.Create(m_chart_id, StringFormat("%s%d", LABEL_NAME, id), m_subwin, x1, y1, x2, y2))
     {
      delete tmp_label;
      return NULL;
     }
   if(!tmp_label.Text(text))
     {
      delete tmp_label;
      return NULL;
     }
   if(!Add(tmp_label))
     {
      delete tmp_label;
      return NULL;
     }
//---
   return tmp_label;
  }
```

Of course, we won't store pointers to all labels. We will save only two objects. To do this, we will declare two additional variables. Although we use dynamic pointers to objects, we will not add them to the destructor of our tool class. These objects will still be deleted in the array of all tool objects. But at the same time, we will get direct access to the objects we need.

```
   CLabel*           m_lbWindowOut;
   CLabel*           m_lbStepHeads;
```

We will write pointers to new variables in the Create method of our class. The method needs small changes which are shown below. The rest of the method code remains unchanged. The full code of the method is provided in the attachment.

```
bool CNetCreatorPanel::Create(const long chart, const string name,
                              const int subwin, const int x1, const int y1)
  {
   if(!CAppDialog::Create(chart, name, subwin, x1, y1, x1 + PANEL_WIDTH, y1 + PANEL_HEIGHT))
      return false;
//---
...............
...............
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   m_lbStepHeads = CreateLabel(8, "Step", lx1, ly1, lx1 + EDIT_WIDTH, ly2);
   if(!m_lbStepHeads)
      return false;
//---
...............
...............
//---
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   m_lbWindowOut = CreateLabel(9, "Window Out", lx1, ly1, lx1 + EDIT_WIDTH, ly2);
   if(!m_lbWindowOut)
      return false;
//---
...............
...............
//---
   return true;
  }
```

The next step in our preparatory work is to create a method for changing the status of the input field. The standard CEdit class already has the ReadOnly structure to change the object status. But this method does not provide visualization of the status. It only locks the possibility to enter data. However, we need a visual separation of objects available and not available for input. We will not invent anything new. Let us highlight the objects with a background color. Editable fields will have the white background, and uneditable ones will have the background color matching the panel color.

This functionality will be implemented in the  EditReedOnly method. In the method parameter, pass a pointer to the object and a new status flag. In the method body, pass the received flag to the ReadOnly method of the input object and set the background of the object according to the specified flag.

```
bool CNetCreatorPanel::EditReedOnly(CEdit& object, const bool flag)
  {
   if(!object.ReadOnly(flag))
      return false;
   if(!object.ColorBackground(flag ? CONTROLS_DIALOG_COLOR_CLIENT_BG : CONTROLS_EDIT_COLOR_BG))
      return false;
//---
   return true;
  }
```

Now pay attention to activation functions. Or rather, to the drop-down list of available activation functions. Not all neural layer types require drop down lists. Some architectures provide a pre-defined activation function type which cannot be changed by the list. An example of this is the LSTM block, subsample layer, attention blocks. However, the CComboBox class does not provide a method that would block the functionality of the class in any way. Therefore, we will use a workaround and will change the list of available activation functions on a case-by-case basis. We will create separate methods for populating the list of available activation functions.

In fact, there are only two such methods. One of them is general, indicating the activation functions — ActivationListMain. The second one is empty — ActivationListEmpty which has only one choice "None".

To understand the method construction algorithm, let's consider the code of the ActivationListMain method. At the beginning of the method, clear the existing list of elements of the available activation functions. Then fill the list in a loop using the ItemAdd method and the EnumToString function.

Note here that the encoding of the elements in the activation function enumeration starts with -1 for None. The next function — the hyperbolic tangent TANH — has the index 0. This is not good for the reason indicated above when describing the filling of the list of descriptions. Because the drop-down list is the CListView class. Therefore, to exclude the null value of the list identifier, we simply add a small constant to the enum identifier.

After populating the list of available activation functions, set the default value and exit the method.

```
bool CNetCreatorPanel::ActivationListMain(void)
  {
   if(!m_cbActivation.ItemsClear())
      return false;
   for(int i = -1; i < 3; i++)
      if(!m_cbActivation.ItemAdd(EnumToString((ENUM_ACTIVATION)i), i + 2))
         return false;
   if(!m_cbActivation.SelectByValue((int)DEFAULT_ACTIVATION + 2))
      return false;
//---
   return true;
  }
```

Another method that we need will help us automate the user's work a little. As mentioned above, in the case of convolutional models or attention blocks, the number of elements at the output of the model is dependent on the size of the window of the analyzed initial data and its movement step. In order to eliminate possible errors and reduce the user's manual labor, I decided to close the input field for the number of blocks and fill it with a separate SetCounts method.

In the parameters of this method, we pass the type of the created neural layer. The method will return the bool result of the operations.

```
bool CNetCreatorPanel::SetCounts(const uint position, const uint type)
  {
   const uint position = m_arAddLayers.Total();
```

And in the method body, we first determine the number of elements in the output of the previous layer. Please note that the previous layer can be in one of two dynamic arrays: descriptions of the architecture of the donor model or descriptions of the architecture for adding new neural layers. We can easily determine where to take the last neural layer from. A neural layer will always be added to the end of the list. Therefore, we will take a layer from the donor model only if the array of new neural layers is empty. Following this logic, we check the size of the dynamic array of new neural layers. Depending on its size, request from the corresponding array a pointer to the previous neural layer.

```
   CLayerDescription *prev;
   if(position <= 0)
     {
      if(!m_arPTModelDescription || m_spPTModelLayers.Value() <= 0)
         return false;
      prev = m_arPTModelDescription.At(m_spPTModelLayers.Value() - 1);
      if(!prev)
         return false;
     }
   else
     {
      if(m_arAddLayers.Total() < (int)position)
         return false;
      prev = m_arAddLayers.At(position - 1);
     }
   if(!prev)
      return false;
```

Next, count the number of elements in the result buffer of the previous layer according to its type. If the buffer size is not greater than 0, exit the method with false.

```
   int outputs = prev.count;
   switch(prev.type)
     {
      case defNeuronAttentionOCL:
      case defNeuronMHAttentionOCL:
      case defNeuronMLMHAttentionOCL:
         outputs *= prev.window;
         break;
      case defNeuronConvOCL:
         outputs *= prev.window_out;
         break;
     }
//---
   if(outputs <= 0)
      return false;
```

Then read from the interface the values of the analyzed initial data window size and its step. And also prepare a variable to record the result of the calculation.

```
   int counts = 0;
   int window = (int)StringToInteger(m_edWindow.Text());
   int step = (int)StringToInteger(m_edStep.Text());
```

The number of elements will be calculated depending on the type of neural layer being created. To calculate the number of elements of the convolutional and subsample layers, we need the size of the analyzed input data window and its step.

```
   switch(type)
     {
      case defNeuronConvOCL:
      case defNeuronProofOCL:
         if(step <= 0)
            break;
         counts = (outputs - window - 1 + 2 * step) / step;
         break;
```

When using attention blocks, the step size is equal to the window size. Using mathematical rules, reduce the formula.

```
      case defNeuronAttentionOCL:
      case defNeuronMHAttentionOCL:
      case defNeuronMLMHAttentionOCL:
         if(window <= 0)
            break;
         counts = (outputs + window - 1) / window;
         break;
```

When using the latent layer of the variational autoencoder, the layer size will be exactly two times smaller than the previous one.

```
      case defNeuronVAEOCL:
         counts = outputs / 2;
         break;
```

For all other cases, we will set the size of the neural layer to be equal to the size of the previous layer. This can be used when declaring a batch normalization or Dropout layer.

```
      default:
         counts = outputs;
         break;
     }
//---
   return m_edCount.Text((string)counts);
  }
```

Transfer the received value to the corresponding interface element.

Now we have enough means to organize interface changes depending on the type of the neural layer to be created. So, let's see how we can do it. This functionality is implemented in the OnChangeNeuronType method. The name is called so because we will call it every time the user changes the type of the neural layer.

The specified method does not contain parameters and returns the logical result of the operation. In the method body, we first define the type of neural layer selected by the user.

```
bool CNetCreatorPanel::OnChangeNeuronType(void)
  {
   long type = m_cbNewNeuronType.Value();
```

Further, the algorithm branches depending on the selected neural layer type. The algorithm for each neural layer will be similar. But almost every neural layer has its own nuances. For a fully connected neural layer, we leave only one active input field for the number of neurons and load the full list of possible activation functions.

```
   switch((int)type)
     {
      case defNeuronBaseOCL:
         if(!EditReedOnly(m_edCount, false) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!ActivationListMain())
            return false;
         break;
```

For a convolutional layer, three more input fields will be active. These include the size of the analyzed source data window and its step, as well as the size of the result window (the number of filters). We also update the values of two text labels and restart the recalculation of the number of elements in a neural layer depending on the source data window size and step. Note that we count the number of elements for one filter. Thus, the result does not depend on the number of filters used.

```
      case defNeuronConvOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, false))
            return false;
         if(!m_lbStepHeads.Text("Step"))
            return false;
         if(!m_lbWindowOut.Text("Window Out"))
            return false;
         if(!ActivationListMain())
            return false;
         if(!SetCounts(defNeuronConvOCL))
            return false;
         break;
```

For the subsampling layer, we do not specify the number of filters and the activation function. In our implementation, we always use the maximum value as the activation function of the subsample layer. Therefore, clear the list of available activation functions. But, as with the convolutional layer, we start calculating the number of elements of the created layer.

```
      case defNeuronProofOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbStepHeads.Text("Step"))
            return false;
         if(!SetCounts(defNeuronProofOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
```

When declaring the LSTM block, the list of activation functions is also not used, so clear it. Only one input field is available — the number of elements in the neural layer.

```
      case defNeuronLSTMOCL:
         if(!EditReedOnly(m_edCount, false) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
```

To initialize the Dropout layer, we need to specify only the values of the neuron dropout probability. No activation function is used. The number of elements is equal to the size of the previous neural layer.

```
      case defNeuronDropoutOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, false) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!SetCounts(defNeuronDropoutOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
```

The similar approach applies to the batch normalization layer. However, here we specify the batch size.

```
      case defNeuronBatchNormOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, false) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!SetCounts(defNeuronBatchNormOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
```

Depending on the attention method, we make active the input fields for the number of attention heads and neural layers in the block. The text labels for the corresponding input fields are changed.

```
      case defNeuronAttentionOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!SetCounts(defNeuronAttentionOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;

      case defNeuronMHAttentionOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!m_lbStepHeads.Text("Heads"))
            return false;
         if(!SetCounts(defNeuronMHAttentionOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;

      case defNeuronMLMHAttentionOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, false) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, false) ||
            !EditReedOnly(m_edWindow, false) ||
            !EditReedOnly(m_edWindowOut, false))
            return false;
         if(!m_lbStepHeads.Text("Heads"))
            return false;
         if(!m_lbWindowOut.Text("Keys size"))
            return false;
         if(!SetCounts(defNeuronMLMHAttentionOCL))
            return false;
         if(!ActivationListEmpty())
            return false;
         break;
```

For the latent layer of the variational autoencoder, there is no need to enter any data. Only select the layer type and add it to the model.

```
      case defNeuronVAEOCL:
         if(!EditReedOnly(m_edCount, true) ||
            !EditReedOnly(m_edBatch, true) ||
            !EditReedOnly(m_edLayers, true) ||
            !EditReedOnly(m_edProbability, true) ||
            !EditReedOnly(m_edStep, true) ||
            !EditReedOnly(m_edWindow, true) ||
            !EditReedOnly(m_edWindowOut, true))
            return false;
         if(!ActivationListEmpty())
            return false;
         if(!SetCounts(defNeuronVAEOCL))
            return false;
         break;
```

If the neural layer type specified in the parameters is not found, then complete the method with 'false'.

```
      default:
         return false;
         break;
     }
//---
   return true;
  }
```

If all the operations of the method are successfully completed, exit with a positive result.

Now we need to organize the start of the described method at the right time. We will use the event related to a change in the value of the layer type selection element and will add an appropriate event handler.

```
EVENT_MAP_BEGIN(CNetCreatorPanel)
ON_EVENT(ON_CLICK, m_edPTModel, OpenPreTrainedModel)
ON_EVENT(ON_CLICK, m_btAddLayer, OnClickAddButton)
ON_EVENT(ON_CLICK, m_btDeleteLayer, OnClickDeleteButton)
ON_EVENT(ON_CLICK, m_btSave, OnClickSaveButton)
ON_EVENT(ON_CHANGE, m_spPTModelLayers, ChangeNumberOfLayers)
ON_EVENT(ON_CHANGE, m_lstPTModel, OnChangeListPTModel)
ON_EVENT(ON_CHANGE, m_lstNewModel, OnChangeListNewModel)
ON_EVENT(ON_CHANGE, m_cbNewNeuronType, OnChangeNeuronType)
EVENT_MAP_END(CAppDialog)
```

By implementing the methods described above, we organized the activation and deactivation of input fields depending on the type of the selected neural layer. But we have also discussed data entry control.

In all input fields we expect integers greater than zero. The only exception is the value of the probability of element dropout in the Dropout layer. This can be a real value between 0 and 1. Therefore, we need two methods to validate the entered data. One for probability and one for all other elements.

The algorithm of both methods is quite simple. First, we read the text value entered by the user, convert it to a numeric value and check if it is within the range of valid values. Enter the received value back to the corresponding window of the interface. The user will only need to check if the data has been interpreted correctly.

```
bool CNetCreatorPanel::OnEndEditProbability(void)
  {
   double value = StringToDouble(m_edProbability.Text());
   return m_edProbability.Text(DoubleToString(fmax(0, fmin(1, value)), 2));
  }

bool CNetCreatorPanel::OnEndEdit(CEdit& object)
  {
   long value = StringToInteger(object.Text());
   return object.Text((string)fmax(1, value));
  }
```

Note that when checking the correctness of the probability value, we clearly identify the input field. But to identify an object in the second method, we will pass the relevant object pointer in method parameters. Here lies another challenge. The proposed event handling macros do not have a suitable macro to pass a pointer of the caller object to the event handling method. So, we need to add such a macro.

```
#define ON_EVENT_CONTROL(event,control,handler)          if(id==(event+CHARTEVENT_CUSTOM) && lparam==control.Id()) \
                                                              { handler(control); return(true); }
```

Among the input fields there can be the size of the analyzed source data window and its step. These parameters affect the number of elements in the neural layer. So, when changing their values, we need to recalculate the size of the created neural layer. But the event handling model we use allows only one handler for each event. At the same time, we can use one handler to serve different events. Therefore, let us create another method that will first check the values in the input fields for the window size and its step. And then we call the method for recalculating the size of the neural layer, taking into account the selected neural layer type.

```
bool CNetCreatorPanel::OnChangeWindowStep(void)
  {
   if(!OnEndEdit(m_edWindow) || !OnEndEdit(m_edStep))
      return false;
   return SetCounts((uint)m_cbNewNeuronType.Value());
  }
```

Now we just have to complete our event handler map. That will allow you to run the right event handler at the right time.

```
EVENT_MAP_BEGIN(CNetCreatorPanel)
ON_EVENT(ON_CLICK, m_edPTModel, OpenPreTrainedModel)
ON_EVENT(ON_CLICK, m_btAddLayer, OnClickAddButton)
ON_EVENT(ON_CLICK, m_btDeleteLayer, OnClickDeleteButton)
ON_EVENT(ON_CLICK, m_btSave, OnClickSaveButton)
ON_EVENT(ON_CHANGE, m_spPTModelLayers, ChangeNumberOfLayers)
ON_EVENT(ON_CHANGE, m_lstPTModel, OnChangeListPTModel)
ON_EVENT(ON_CHANGE, m_lstNewModel, OnChangeListNewModel)
ON_EVENT(ON_CHANGE, m_cbNewNeuronType, OnChangeNeuronType)
ON_EVENT(ON_END_EDIT, m_edWindow, OnChangeWindowStep)
ON_EVENT(ON_END_EDIT, m_edStep, OnChangeWindowStep)
ON_EVENT(ON_END_EDIT, m_edProbability, OnEndEditProbability)
ON_EVENT_CONTROL(ON_END_EDIT, m_edCount, OnEndEdit)
ON_EVENT_CONTROL(ON_END_EDIT, m_edWindowOut, OnEndEdit)
ON_EVENT_CONTROL(ON_END_EDIT, m_edLayers, OnEndEdit)
ON_EVENT_CONTROL(ON_END_EDIT, m_edBatch, OnEndEdit)
EVENT_MAP_END(CAppDialog)
```

### 3\. Adding keyboard event handling

We've done a good job to make our Transfer Learning tool much more convenient and user friendly. But all these improvements have focused on the interface, to make it easier to use with a mouse or touch pad. But we have not implemented any possibility of using the keyboard to work with the tool. For example, it can be convenient to use the up and down arrows to change the number of neural layers to copy. Pressing the Delete key can call a method to delete the selected neural layer from the model being created.

I will not dive deep into this topic now. I'll just show you how to add key processing with existing event handlers in just a few lines of code.

All the three features proposed above are already implemented in our tool code. They are executed when a certain event occurs. To delete the selected neural layer, there is a separate button on the panel. The number of neural layers to be copied is changed using buttons of the CSpinEdit object.

Technically, pressing the keyboard buttons is the same event as pressing the mouse buttons or moving it. It is also processed by the OnChartEvent function. So, the ChartEvent method of the class is called.

When a keystroke event occurs, we will receive the CHARTEVENT\_KEYDOWN event ID. The lparam variable will store the ID of the pressed key.

Using this property, we can play around with the keyboard and determine the identifiers of all the keys we are interested in. For example, here are the codes of the keys mentioned above.

```
#define KEY_UP                               38
#define KEY_DOWN                             40
#define KEY_DELETE                           46
```

Now let's get back to the ChartEvent method of our class. In it, we called a similar method of the parent class. Now, we need to add the check for the event ID and the visibility of our tool. The event handler will only run when the tool interface is visible. The user should be able to see what is happening on the panel and visually control the process.

If the first stage of verification is passed, check the code of the pressed key. If there is a corresponding key in the list, generate a custom event that corresponds to a similar action on the panel of our interface.

For example, when the _Delete_ is pressed, generate a button click event _DELETE_ on the interface panel.

```
void CNetCreatorPanel::ChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   CAppDialog::ChartEvent(id, lparam, dparam, sparam);
   if(id == CHARTEVENT_KEYDOWN && m_spPTModelLayers.IsVisible())
     {
      switch((int)lparam)
        {
         case KEY_UP:
            EventChartCustom(CONTROLS_SELF_MESSAGE, ON_CLICK, m_spPTModelLayers.Id() + 2, 0.0, m_spPTModelLayers.Name() + "Inc");
            break;
         case KEY_DOWN:
            EventChartCustom(CONTROLS_SELF_MESSAGE, ON_CLICK, m_spPTModelLayers.Id() + 3, 0.0, m_spPTModelLayers.Name() + "Dec");
            break;
         case KEY_DELETE:
            EventChartCustom(CONTROLS_SELF_MESSAGE, ON_CLICK, m_btDeleteLayer.Id(), 0.0, m_btDeleteLayer.Name());
            break;
        }
     }
  }
```

After that we exit the method. Next, we let the program handle the generated event using the already existing event handlers and methods.

Of course, this approach is possible only if there are appropriate handlers in the program. But you can create new event handlers and generate unique events for them.

### Conclusion

In this article, we looked at various options for improving the usability of the user interface. You can evaluate the quality of the approaches by testing the tool attached to the article. I hope you find this tool useful. I will be grateful if you share your impressions and wishes for improving the tool in the related forum thread.

### List of references

1. [Neural networks made easy (Part 20): Autoencoders](https://www.mql5.com/en/articles/11172)
2. [Neural networks made easy (Part 21): Variational autoencoders (VAE)](https://www.mql5.com/en/articles/11206)
3. [Neural networks made easy (Part 22): Unsupervised learning of recurrent models](https://www.mql5.com/en/articles/11245)
4. [Neural networks made easy (Part 23): Building a tool for Transfer Learning](https://www.mql5.com/en/articles/11273)

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | NetCreator.mq5 | EA | Model building tool |
| 2 | NetCreatotPanel.mqh | Class library | Class library for creating the tool |
| 3 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 4 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/11306](https://www.mql5.com/ru/articles/11306)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11306.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/11306/mql5.zip "Download MQL5.zip")(74.22 KB)

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
**[Go to discussion](https://www.mql5.com/en/forum/435256)**
(4)


![Dmitry Iglakov](https://c.mql5.com/avatar/2015/10/5619F390-33F4.png)

**[Dmitry Iglakov](https://www.mql5.com/en/users/cjdmitri)**
\|
23 Sep 2022 at 19:26

[Neural networks](https://www.mql5.com/en/articles/497 "Article: Neural networks - from theory to practice ") are simple (part of 500988939928177231827361823461827631827361827361827361827361284762834762834762). By the time you read the last part, you will most likely be 89 years old and neural networks will no longer be relevant.

And seriously, "neural networks are simple" is when there are at most two articles of this length. Not when I think that mt5 is hung up and I have been receiving notifications about the article "neural networks are  simple "  for a year already


![Dmitriy Gizlyk](https://c.mql5.com/avatar/2014/8/53E8CB77-1C48.png)

**[Dmitriy Gizlyk](https://www.mql5.com/en/users/dng)**
\|
23 Sep 2022 at 19:48

**Dmitry Iglakov [#](https://www.mql5.com/ru/forum/430434#comment_42258736):**

Neural networks are simple (part of 500988939928177231827361823461827631827361827361827361827361284762834762834762). By the time you read the last part, you will most likely be 89 years old and neural networks will no longer be relevant.

And seriously, "neural networks are simple" is when there are at most two articles of this size. Not when I think that mt5 is hung up and I have been receiving notifications about the article "neural networks are  simple  " for a year already

The idea of "Neural Networks are Simple" is to show the accessibility of the technology to everyone. Yes, the series is quite long. But the practical use is available to the reader from the second article. And each separate article tells about new possibilities. And it is possible to include them in your developments right after reading the article. Whether to use them or not is a personal matter for each reader. There is no need to wait for the next article. As for the volume of the topic, I can say that science is developing and new algorithms appear every day. It is quite possible that their application can bear fruit in trading.

![Ivan Butko](https://c.mql5.com/avatar/2017/1/58797422-0BFA.png)

**[Ivan Butko](https://www.mql5.com/en/users/capitalplus)**
\|
23 Sep 2022 at 22:45

A UFO flew in and published this article.

And so it can be said about each of the articles in the series. The more you dive into the topic, the more you realise the depth and value of such articles. The main thing for beginners is not to give up when you don't see beautiful graphs of balance growth in the article. The author smoothly leads to this. Although, the working copy is attached to several articles, you can take it and use it.

Dmitry thank you.


![Fajar Hidayat](https://c.mql5.com/avatar/2022/1/61EA4305-D2AC.jpg)

**[Fajar Hidayat](https://www.mql5.com/en/users/fajarhida)**
\|
24 Feb 2023 at 07:32

**MetaQuotes:**

New article [Neural networks made easy (Part 24): Improving the tool for Transfer Learning](https://www.mql5.com/en/articles/11306) has been published:

Author: [Dmitriy Gizlyk](https://www.mql5.com/en/users/DNG "DNG")

im stuck here..  when i load previous model also always say "the file is damaged".. but its work when i use tool from part 23.. can you give me a hint to solve the problem?  thanks


![DoEasy. Controls (Part 17): Cropping invisible object parts, auxiliary arrow buttons WinForms objects](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__5.png)[DoEasy. Controls (Part 17): Cropping invisible object parts, auxiliary arrow buttons WinForms objects](https://www.mql5.com/en/articles/11408)

In this article, I will create the functionality for hiding object sections located beyond their containers. Besides, I will create auxiliary arrow button objects to be used as part of other WinForms objects.

![DoEasy. Controls (Part 16): TabControl WinForms object — several rows of tab headers, stretching headers to fit the container](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__4.png)[DoEasy. Controls (Part 16): TabControl WinForms object — several rows of tab headers, stretching headers to fit the container](https://www.mql5.com/en/articles/11356)

In this article, I will continue the development of TabControl and implement the arrangement of tab headers on all four sides of the control for all modes of setting the size of headers: Normal, Fixed and Fill To Right.

![Developing a trading Expert Advisor from scratch (Part 28): Towards the future (III)](https://c.mql5.com/2/48/development__4.png)[Developing a trading Expert Advisor from scratch (Part 28): Towards the future (III)](https://www.mql5.com/en/articles/10635)

There is still one task which our order system is not up to, but we will FINALLY figure it out. The MetaTrader 5 provides a system of tickets which allows creating and correcting order values. The idea is to have an Expert Advisor that would make the same ticket system faster and more efficient.

![Data Science and Machine Learning (Part 08): K-Means Clustering in plain MQL5](https://c.mql5.com/2/50/k-means_clustering_small.png)[Data Science and Machine Learning (Part 08): K-Means Clustering in plain MQL5](https://www.mql5.com/en/articles/11615)

Data mining is crucial to a data scientist and a trader because very often, the data isn't as straightforward as we think it is. The human eye can not understand the minor underlying pattern and relationships in the dataset, maybe the K-means algorithm can help us with that. Let's find out...

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/11306&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5070319093547996095)

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