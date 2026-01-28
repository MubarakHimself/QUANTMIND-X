---
title: Neural networks made easy (Part 23): Building a tool for Transfer Learning
url: https://www.mql5.com/en/articles/11273
categories: Trading Systems, Integration, Expert Advisors
relevance_score: 6
scraped_at: 2026-01-23T11:45:39.129236
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=arrvihnorxsweshfqflpaxodmecuzcww&ssn=1769157937820545564&ssn_dr=0&ssn_sr=0&fv_date=1769157937&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11273&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20networks%20made%20easy%20(Part%2023)%3A%20Building%20a%20tool%20for%20Transfer%20Learning%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915793733646590&fz_uniq=5062695633971750704&sv=2552)

MetaTrader 5 / Trading systems


### Contents

- [Introduction](https://www.mql5.com/en/articles/11273#para1)
- [1\. The purpose of Transfer Learning](https://www.mql5.com/en/articles/11273#para2)
- [2\. Creating a tool](https://www.mql5.com/en/articles/11273#para3)

  - [2.1 Design](https://www.mql5.com/en/articles/11273#para31)
  - [2.2 Implementing the user interface](https://www.mql5.com/en/articles/11273#para32)
  - [2.3 Implementing the tool functionality](https://www.mql5.com/en/articles/11273#para33)

- [3\. Testing](https://www.mql5.com/en/articles/11273#para4)
- [Conclusion](https://www.mql5.com/en/articles/11273#para5)
- [List of references](https://www.mql5.com/en/articles/11273#para6)
- [Programs used in the article](https://www.mql5.com/en/articles/11273#para7)

### Introduction

We continue our immersion in the world of artificial intelligence. Today I invite you to get acquainted with the Transfer Learning technology. We have already mentioned this technology in various articles but have never used it. Meanwhile, this is a powerful tool which increases the efficiency of developing neural networks and reduces the cost of training them.

### 1\. The purpose of Transfer Learning

What is Transfer Learning and why do we need it? Transfer Learning is a machine learning method in which the knowledge of a model trained to solve one problem is reused as a basis for solving new problems. Of course, to solve new problems, the model is preliminarily additionally trained on new data. In the general case, with a properly selected donor model, additional training runs much faster and with better results than training a similar model from scratch.

It is possible to use the full donor model or part of it.

Similar to this technology is the case when we used clustering and data compression results to pre-process source data for the neural network. In this case, we used the entire pre-trained model. But when building a model for solving new problems, we did not carry out additional training of the donor model. We only used it to pre-process the "raw" source data and trained a new model using this data.

When we started studying autoencoders, we also talked about the possibility of using Transfer Learning after model training. But in this case, we cannot use the complete autoencoder completely as a donor model, because we trained it to compress the original data and then restore it from the compressed representation. Therefore, there is no point in using the entire autoencoder as a donor model. For data pre-processing, it would be much more efficient to use only the encoder. In this case, the overall model will be smaller, and the efficiency of further layers will be higher, since fewer trainable weights will be required to process the same amount of information.

But the use of Transfer Learning is not limited to unsupervised learning results. Think back to how many times you started training your model all over again when you needed to add or remove even one neural layer. In this case, part of neural layers could be reused.

There is another area of application of this technology. Due to the fading gradient problem, it is nearly impossible to fully train a deep model. The use of Transfer Learning allows training neural layers in blocks and gradually increasing the size of the model.

Of course, there are many other possible uses for this technology, which you can explore. Now, let us proceed to considering an instrument which would allow it use.

### 2\. Creating a tool

Let us first decide on the purpose of the tool which we will be creating. First of all, let's get back to how we save our trained models. All of them are saved in a single binary file. Each model object has its own strict data recording structure. Therefore, it will be difficult to simply remove part of the data from the file in the editor. So, we need to load the entire trained model from the file, perform the necessary manipulations and save the new model to a new file or to overwrite the previous one. A new file is more preferable, as the donor model can further be used to solve the problems on which it was trained.

Also, our neural networks work well only with the data on which they were trained. The result can be unpredictable on completely new data. This also applies to individual neuronal layers. Therefore, for Transfer Learning, we can use only successive neural layers, starting from the input data layer. You cannot extract a block from the middle or end of the model. That is, we can use the entire donor model or several of its first layers. Then we add to it several different neural layers and save the new model.

At the same time, we need to ensure the full functionality of the new model both in the training mode and in operation. Of course, the model must first be trained.

Please pay attention to the following. Neural layers from the donor model retain their weights. They also retain all their knowledge gained at the model pre-training stage. The new neural layers will receive random weights, just like when the model was initialized. If we start training a new model as we did before, then along with training new layers, we will unbalance the previously trained neural layers. Therefore, we must first block the training of the donor model neural layers. This way we ensure training of only new layers.

#### 2.1 Design

We need not only program that will use the source donor model. We need to somehow process and resave it to a new file. The number of copied layers, as well as the model architecture are always individual. Therefore, we need a tool that will allow the user to quickly and conveniently configure each model individually. I.e., we need a tool with a convenient user interface. SO, we will start with the UI design.

So, I see three clear blocks. In the first block we will work with the donor model. Here we need the ability to select a file with a trained model. After loading a model from a file, the tool must provide a description of the architecture of the loaded model. This is because the user should understand which model is loaded and which neural layers it will copy. We will also inform the tool about the number of copied layers. As mentioned above, we will sequentially copy the neural layers starting from the source data layer.

In the second block, neural layers will be added. Here we will create fields for entering information about the neural layer being created. As with the program code, we will sequentially describe each neural layer one by one and will add it to the architecture of the new model.

The third block will display the holistic architecture of the created model with the ability to specify a file to save it. An example design of the tool is presented below.

![Tool design](https://c.mql5.com/2/48/TransferLearning__2.png)

Both the design of the tool and its implementation are presented for demonstration purposes only. You can always change them to best meet your needs.

#### 2.2 Implementing the user interface

Now we can proceed to the implementation of the design. For this, let us create a new class CNetCreatorPanel that inherits the CAppDialog dialog application base class.

Each control in the panel will be created as a separate object. Therefore, we will declare quite a lot of objects in our new class. For convenience, we will divide them into blocks.

The first block will contain objects related to the visualization of the pre-trained model:

- m\_edPTModel — an element for specifying the file name of the pre-trained model
- m\_edPTModelLayers — display of the total number of neural layers in the pre-trained model
- m\_spPTModelLayers — the number of neural layers that will be copied to the new model
- m\_lstPTMode — display of the architecture of the pre-trained model

```
class CNetCreatorPanel : protected CAppDialog
  {
protected:
   //--- pre-trained model
   CEdit             m_edPTModel;
   CEdit             m_edPTModelLayers;
   CSpinEdit         m_spPTModelLayers;
   CListView         m_lstPTModel;
   CNetModify        m_Model;
   CArrayObj*        m_arPTModelDescription;
```

Also, we will here declare the objects to work with the pre-trained model:

- m\_Model — the object of the pre-trained model
- m\_arPTModelDescription — a dynamic array with the description of the architecture of the pre-trained model

Pay attention to the following two moments. All objects are declared as static, except for the dynamic array of the model architecture description. The use of static objects enables transferring of memory operations to the system. This is because static objects are created and deleted together with the object in which they are contained and do not require any additional work from the programmer. But this way, it is only possible to create objects in the structure of our class. The description of the architecture will be obtained from the pre-trained model. Therefore, this object was declared through a dynamic pointer.

And the second moment. To declare a pre-trained model object, we used the CNetModify class. But previously we created the CNet class for neural network models. This is because we need additional functionality from our neural network. To implement it, we will create a new class CNetModify derived from the CNet class. But we will get back to this part when describing the tool functionality.

The next block contains objects for describing the new neural layer being created. The objects are in line with the elements of the CLayerDescription class describing the neural layer architecture. That is why we will not examine each of the element in detail. But I would like to mention the creation of two buttons to add a new neural layer and to delete a created one. Only added neural layers can be deleted. To control the number of copied neural layers, we will use the elements of the previous block.

```
   //--- add layers
   CComboBox         m_cbNewNeuronType;
   CEdit             m_edCount;
   CEdit             m_edWindow;
   CEdit             m_edWindowOut;
   CEdit             m_edStep;
   CEdit             m_edLayers;
   CEdit             m_edBatch;
   CEdit             m_edProbability;
   CComboBox         m_cbActivation;
   CComboBox         m_cbOptimization;
   CButton           m_btAddLayer;
   CButton           m_btDeleteLayer;
```

The last block of objects of the new model contains only 3 elements. These are an object for displaying the general architecture of the model, a button for saving the new model and a dynamic array describing the architecture of neural layers we are adding. In this case, we have created a static object of the dynamic array describing the architecture of neural layers being added m\_arAddLayers. The architecture of the neural layers will be created inside the tool. This object can also be created as static.

```
   //--- new model
   CListView         m_lstNewModel;
   CButton           m_btSave;
   CArrayObj         m_arAddLayers;
```

We will use a basic list of public methods of the class. These include a class constructor and destructor, a panel creation method, and an event handler.

Three methods of the parent class have been overridden. This could have been avoided by public inheritance.

```
public:
                     CNetCreatorPanel();
                    ~CNetCreatorPanel();
   //--- main application dialog creation and destroy
   virtual bool      Create(const long chart, const string name, const int subwin, const int x1, const int y1);
   //--- chart event handler
   virtual bool      OnEvent(const int id, const long &lparam, const double &dparam, const string &sparam);

   virtual void      Destroy(const int reason = REASON_PROGRAM) override { CAppDialog::Destroy(reason); }
   bool              Run(void) { return CAppDialog::Run();}
   void              ChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
     {               CAppDialog::ChartEvent(id, lparam, dparam, sparam); }
  };
```

Because we use static objects, the constructor and destructor of our class are practically empty.

The main part of work related to the creation and arrangement of the interface elements is implemented in the dialog window creation method Create. But before we move on to method description, let us perform a little preparatory work.

First, we need to define the number of constants that will help us properly organize the internal space of the interface. The full list is provided in the attachment.

It should also be noted that in addition to the input elements, our interface contains a number of text labels. But we haven't declared objects for them. This is done intentionally to simplify the structure of our class. We need them only for visualization, so they are not used to create the functionality of our tool. However, we will need to create these objects. The procedure for creating such objects will be repeated, except for some data. This may include object text and its location. In order to structure our code, we will create a separate CreateLabel method for creating such labels.

In the method parameters, we will pass the object identifier, the text of the label and its coordinates on the panel.

In the method body, we first create a new label object and check the operation result. Then we create an object on the chart, pass the necessary content to it, and add the created object pointer to a dynamic array with the collection of the interface objects.

We have created a new object with a pointer in a private variable. During the execution of method operations, check the result of each operation and, in case of an error, delete the created object. But after exiting the method, we do not leave a pointer to the created object in our class for its further removal when the program is closed. This is because we passed a pointer to the created object to the collection of dialog box objects, the full functionality of which is already implemented in the parent class. This functionality includes the deletion of all objects of the collection when the program is closed. So, for now we can pass the pointer to the collection and forget about it.

```
bool CNetCreatorPanel::CreateLabel(const int id, const string text, const int x1, const int y1, const int x2, const int y2)
  {
   CLabel *tmp_label = new CLabel();
   if(!tmp_label)
      return false;
   if(!tmp_label.Create(m_chart_id, StringFormat("%s%d", LABEL_NAME, id), m_subwin, x1, y1, x2, y2))
     {
      delete tmp_label;
      return false;
     }
   if(!tmp_label.Text(text))
     {
      delete tmp_label;
      return false;
     }
   if(!Add(tmp_label))
     {
      delete tmp_label;
      return false;
     }
//---
   return true;
  }
```

Similarly, we will create a method for creating input objects. But instead of creating new objects, we use those previously created in the class. The relevant pointers are passed in method parameters.

```
bool CNetCreatorPanel::CreateEdit(const int id,
                                  CEdit& object,
                                  const int x1,
                                  const int y1,
                                  const int x2,
                                  const int y2,
                                  bool read_only)
  {
   if(!object.Create(m_chart_id, StringFormat("%s%d", EDIT_NAME, id), m_subwin, x1, y1, x2, y2))
      return false;
   if(!object.TextAlign(ALIGN_RIGHT))
      return false;
   if(!object.ReadOnly(read_only))
      return false;
   if(!Add(object))
      return false;
//---
   return true;
  }
```

In addition, we use enumerations and constants to describe the architecture of the created neural layers. To avoid the entering of incorrect values by users into such elements, let is create special controls. The user will be able to select only one element from the proposed list. We need several such elements. Let us start by creating an element to indicate the type of neural layer. This functionality will be implemented in the CreateComboBoxType method. Since this method is designed to create a specific element, we do not need to pass a pointer to an object in the parameters. Here we only need to specify the coordinates of the element being created.

In the method body, we create an element on the chart at the specified coordinates and check the result.

Next, we need to fill the element with a text description and the numeric ID. We can use the identifier of the neural layer type as an ID. But we do not have a text description. Therefore, to translate a numeric identifier into a text description, we will create a separate LayerTypeToString method. Its algorithm is quite simple. You can view it in the attachment. Here we will only call this method for each type of neural layer.

At the end of the method, we will add the object pointer to the collection of our interface objects.

Note that we add both dynamic and static objects to the collection. This is because the collection functionality is much broader than the control over the removal of objects after program completion. At the same time, the collection elements participate in determining the coordinates of objects on the chart and in processing events. The general purpose of the specified collection is in the functioning of all objects as a single whole organism.

```
bool CNetCreatorPanel::CreateComboBoxType(const int x1, const int y1, const int x2, const int y2)
  {
   if(!m_cbNewNeuronType.Create(m_chart_id, "cbNewNeuronType", m_subwin, x1, y1, x2, y2))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronBaseOCL), defNeuronBaseOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronConvOCL), defNeuronConvOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronProofOCL), defNeuronProofOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronLSTMOCL), defNeuronLSTMOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronAttentionOCL), defNeuronAttentionOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronMHAttentionOCL), defNeuronMHAttentionOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronMLMHAttentionOCL), defNeuronMLMHAttentionOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronDropoutOCL), defNeuronDropoutOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronBatchNormOCL), defNeuronBatchNormOCL))
      return false;
   if(!m_cbNewNeuronType.ItemAdd(LayerTypeToString(defNeuronVAEOCL), defNeuronVAEOCL))
      return false;
   if(!Add(m_cbNewNeuronType))
      return false;
//---
   return true;
  }
```

Similarly, create objects for enumerations of activation functions and parameter optimization methods. To convert the enumeration into a text form, we will use the standard EnumToString function. Therefore, we can add elements to the list in a loop. The full code of the methods is available in the attachment.

This completes the preparatory work, and we can proceed to creating the user interface. This functionality is executed in the Create method. In parameters, we receive only the coordinates of the location of the upper right corner of the panel on the chart. However, to create objects, we will also need the dimensions of our panel. To enable convenient operations and future modifications (if necessary), I set the dimensions of the panel through predefined constants. The panel is created by a similar method of the parent class. It is the first to be called in the method body.

```
bool CNetCreatorPanel::Create(const long chart, const string name, const int subwin, const int x1, const int y1)
  {
   if(!CAppDialog::Create(chart, name, subwin, x1, y1, x1 + PANEL_WIDTH, y1 + PANEL_HEIGHT))
      return false;
```

Next, add interface objects to the created panel. The objects will be added sequentially, starting from the upper left corner. The coordinates of each new object will be linked to the coordinates of the previous object. This approach will allow us to build objects into an even structure.

According to the above logic, let us start creating pre-trained model work group objects. The first one is a group label. To create it, determine the coordinates of the label and call the previously created CreateLabel method. Th label text and coordinates are passed to this method. Do not forget to add a unique label ID.

```
   int lx1 = INDENT_LEFT;
   int ly1 = INDENT_TOP;
   int lx2 = lx1 + LIST_WIDTH;
   int ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(0, "PreTrained model", lx1, ly1, lx2, ly2))
      return false;
```

Next we create an input field that will be used to select the name of the file with a pre-trained model. To do this, shift the coordinates of the created object vertically and leave the horizontal coordinates unchanged. Thus, 2 objects will be located strictly under each other.

The user will not be able to specify the file name manually. Instead, we prompt the user to select a file from the existing ones. We will get back to the functionality of this action a little later. For now, we make the file name field read-only. The object is created by calling the previously created CreateEdit method. After creating the field, add an informational message to it.

```
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateEdit(0, m_edPTModel, lx1, ly1, lx2, ly2, true))
      return false;
   if(!m_edPTModel.Text("Select file"))
      return false;
```

Below that we will specify the number of neural fields of the trained model. To do this, create a text label and an input field (output in this case) for the number of neural layers. This field will also be read-only.

```
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(1, "Layers Total", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(1, m_edPTModelLayers, lx2 - EDIT_WIDTH, ly1, lx2, ly2, true))
      return false;
   if(!m_edPTModelLayers.Text("0"))
      return false;
```

Similarly, create a label and fields for entering the number of neural layers to copy. We need to implement here a mechanism that limits the user in choosing the number of neural layers. It must not be less than 0 or greater than the total number of neural layers in the model. This can be easily done by using an instance of the CSpinEdit class object. This class allows us to specify a range of valid values. The rest is already implemented in the class.

```
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(2, "Transfer Layers", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!m_spPTModelLayers.Create(m_chart_id, "spPTMCopyLayers", m_subwin, lx2 - 100, ly1, lx2, ly2))
      return false;
   m_spPTModelLayers.MinValue(0);
   m_spPTModelLayers.MaxValue(0);
   m_spPTModelLayers.Value(0);
   if(!Add(m_spPTModelLayers))
      return false;
```

Next we should only display a window with the description of the pre-trained model architecture. Please note that before this we always shifted the coordinates of the created objects one level lower. In this case, we only shifted the upper border from the previous object towards the bottom. The lower border of the object is set at an indent from the height of our window. Thus, we stretch the object to the size of the window and get a smooth edge at the bottom of the created interface.

```
   lx1 = INDENT_LEFT;
   lx2 = lx1 + LIST_WIDTH;
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ClientAreaHeight() - INDENT_BOTTOM;
   if(!m_lstPTModel.Create(m_chart_id, "lstPTModel", m_subwin, lx1, ly1, lx2, ly2))
      return false;
   if(!m_lstPTModel.VScrolled(true))
      return false;
   if(!Add(m_lstPTModel))
      return false;
```

This completes operations with the pre-trained model block and proceeds to the second block of objects to describe the architecture of the added neural layer. The block objects are also created from top to bottom. When defining the coordinates for the new object, we will shift the coordinates horizontally and define the top border at the level of the indent from the top edge of the window.

```
   lx1 = lx2 + CONTROLS_GAP_X;
   lx2 = lx1 + ADDS_WIDTH;
   ly1 = INDENT_TOP;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(3, "Add layer", lx1, ly1, lx2, ly2))
      return false;
```

Below, at the indent distance, create a combo box to select the type of neural layer to create. This is done by using the previously created method. The width of this object will be equal to the width of the entire block.

```
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateComboBoxType(lx1, ly1, lx2, ly2))
      return false;
```

This is followed by the elements describing the architectures of the created neural layer. For each element from the CLayerDescription neural layer architecture description class, we will create 2 objects: a text label with the name of the element and a value input field. To position the elements on the interface panel in a strict order, we will align the text labels to the left, and the input fields to the right of the block. The size of all input fields will be the same. This approach will create a kind of table.

I will not provide identical code for all 9 elements now. Below is an example of code for creating 2 rows from our table. The full code is available in the attachment.

```
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(4, "Neurons", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateEdit(2, m_edCount, lx2 - EDIT_WIDTH, ly1, lx2, ly2, false))
      return false;
   if(!m_edCount.Text((string)DEFAULT_NEURONS))
      return false;

   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + EDIT_HEIGHT;
   if(!CreateLabel(5, "Activation", lx1, ly1, lx1 + EDIT_WIDTH, ly2))
      return false;
//---
   if(!CreateComboBoxActivation(lx2 - EDIT_WIDTH, ly1, lx2, ly2))
      return false;
```

After creating elements to describe the architecture of the added neural layer, let us add 2 buttons: for adding and removing a neural layer. Arrange the buttons in one row, dividing the width of the block between them in half.

```
   ly1 = ly2 + CONTROLS_GAP_Y;
   ly2 = ly1 + BUTTON_HEIGHT;
   if(!m_btAddLayer.Create(m_chart_id, "btAddLayer", m_subwin, lx1, ly1, lx1 + ADDS_WIDTH / 2, ly2))
      return false;
   if(!m_btAddLayer.Text("ADD LAYER"))
      return false;
   m_btAddLayer.Locking(false);
   if(!Add(m_btAddLayer))
      return false;
//---
   if(!m_btDeleteLayer.Create(m_chart_id, "btDeleteLayer", m_subwin, lx2 - ADDS_WIDTH / 2, ly1, lx2, ly2))
      return false;
   if(!m_btDeleteLayer.Text("DELETE"))
      return false;
   m_btDeleteLayer.Locking(false);
   if(!Add(m_btDeleteLayer))
      return false;
```

Let us move on to the third and final block of describing the complete architecture of the model being created. Here you can find all the methods used above.

After creating all the elements, we exit the method with 'true'. The complete code of all methods and classes is available in the attachment below.

This concludes the arrangement of the elements of our interface. It can now be added to the Expert Advisor. But in this form, it will be just a beautiful picture on the symbol chart. Next, we need to implement the necessary functionality in the form.

#### 2.3 Implementing the tool functionality

We continue to work on creating our tool and the next step is to provide the interface with the necessary functionality. Before proceeding, let's get back to the desired algorithm for our tool.

1. First, we need to open the file with the saved trained model. To do this, the user clicks on the object to select a file. This opens a dialog box in which the user selects an existing file with the given extension.
2. After the user selects a file, the tool should load the model from the specified file and display information about the loaded model (type and number of neuron layers, number of neurons in each layer).
3. Together with the output of information about the default loaded model, all its neural layers are set to be copied to the new model. Information about them is also copied to the description block of the created model.
4. The user should be able to manually change the number of copied neural layers. Simultaneously with the change in the number of copied neural layers, changes must be made to the architecture of the created model. This will be reflected in the block describing the architecture of the created model.
5. After selecting the number of copied neural layers, the user can manually specify the type and architecture of the new neural layer and add it to the created model by pressing the "ADD LAYER" button.
6. If some neural layer was added to the model by mistake, the user can select such a neural layer in the block describing the model architecture and delete it by pressing the "DELETE" button. Please note that only added neural layers can be deleted. To remove the layers of the donor model, you should use the tool to change the number of copied neural layers.
7. After creating the architecture of the created neural network, the user presses the "SAVE MODEL" button. This opens a dialog box in which the user should select an existing file or specify the name of a new one.

It seems to me a logical scenario of working with the tool. However, some efforts are needed to implement it. First, we need the functionality of obtaining information about the saved model. Previously, we did not provide the user with information about the loaded model. To implement this functionality, we will need to make changes to the neural network class. But since this functionality does not affect the operation of the model itself, we will add it to the new CNetModify class, which will be a direct successor to the previously created CNet neural network model class.

We will not create any new objects in the new class. Therefore, the class constructor and destructor will remain empty. The LayersTotal method returns the number of neural layers in the model. There is nothing complicated in its algorithm, since it simply returns the size of the array. Its full code is available in the attachment.

```
class CNetModify :  public CNet
  {
public:
                     CNetModify(void) {};
                    ~CNetModify(void) {};
   //---
   uint              LayersTotal(void);
   CArrayObj*        GetLayersDiscriptions(void);
  };
```

Let us dwell a little on the GetLayersDiscriptions method for obtaining information about the neural networks used. As a result of executing this method, we should receive a dynamic array with the neural network architecture description, similar to the model description passed in the parameters of the model constructor method. The complexity of organizing this process is connected to the fact that we have not previously created methods for obtaining hyperparameters of neural layers. Therefore, we need to add the corresponding method to the neural layer classes. To begin with, we will add the GetLayerInfo method to the CNeuronBaseOCL neural layer base class.

The new method does not contain parameters and, after execution, will return the CLayerDescription neural layer description object. In the method body, we will first create an instance of the neural layer description object. Then fill it with the hyperparameters of the current neural layer. After that, exit the method and return the created object pointer to the calling program.

```
CLayerDescription* CNeuronBaseOCL::GetLayerInfo(void)
  {
   CLayerDescription* result = new CLayerDescription();
   if(!result)
      return result;
//---
   result.type = Type();
   result.count = Output.Total();
   result.optimization = optimization;
   result.activation = activation;
   result.batch = (int)(optimization == LS ? iBatch : 1);
   result.layers = 1;
//---
   return result;
  }
```

By adding a method to the neural layer base class, we have added a method to all its descendants. So, all neural layers have got this method. Now we can get similar information from any neural layer. If this data is enough for you, then you can finish working with the neural layer and move on model information collecting method.

But if you need specific information for each neural layer, you will need to override this method in all neural layers. Below is an example of method overriding in the subsampling layer, which allows getting data on the analyzed window size and its movement step. In the method body, first call the parent class method to get the underlying hyperparameters. And then supplement the resulting neural layer description object with specific parameters. After that exit the method by returning a pointer to the neural layer description object to the calling program.

```
CLayerDescription* CNeuronProofOCL::GetLayerInfo(void)
  {
   CLayerDescription *result = CNeuronBaseOCL::GetLayerInfo();
   if(!result)
      return result;
   result.window = (int)iWindow;
   result.step = (int)iStep;
//---
   return result;
  }
```

Similar methods for all the previously discussed types of neural layers are available in the attachment below.

Now we can obtain information about the hyperparameters of each neural layer. This information can be combined into a common structure. Let's get back to our CNetModify::GetLayersDiscriptions method and create a dynamic array in it to store pointers to neural layer description objects.

Next, we will create a loop through all the neural layers. In the loop body, we will request from each neural layer an architecture description object by calling the above created method. The obtained objects will be added to the dynamic array.

After executing all iterations of the loop, we will have a dynamic array with the description of the full loaded model architecture. Return it to the caller program after method completion.

```
CArrayObj* CNetModify::GetLayersDiscriptions(void)
  {
   CArrayObj* result = new CArrayObj();
   for(uint i = 0; i < LayersTotal(); i++)
     {
      CLayer* layer = layers.At(i);
      if(!layer)
         break;
      CNeuronBaseOCL* neuron = layer.At(0);
      if(!neuron)
         break;
      if(!result.Add(neuron.GetLayerInfo()))
         break;
     }
//---
   return result;
  }
```

At this stage, we have implemented the possibility of obtaining a description of the architecture of a previously created model. Now, we can move on to implementing a method for loading a pre-trained model from a user-specified file. To implement this functionality, let us create the CNetCreatorPanel::LoadModel method. The method will receive in parameters the name of the file to load the model.

In the method body, we first load the model from the specified file. Notice that we don't check the value of the parameter before calling the model's Load method. This is because all controls are implemented in the load method. We only check the operation result. In case of a model loading error, output the error information the loaded model description block.

```
bool CNetCreatorPanel::LoadModel(string file_name)
  {
   float error, undefine, forecast;
   datetime time;
   ResetLastError();
   if(!m_Model.Load(file_name, error, undefine, forecast, time, false))
     {
      m_lstPTModel.ItemsClear();
      m_lstPTModel.ItemAdd("Error of load model", 0);
      m_lstPTModel.ItemAdd(file_name, 1);
      int err = GetLastError();
      if(err == 0)
         m_lstPTModel.ItemAdd("The file is damaged");
      else
         m_lstPTModel.ItemAdd(StringFormat("error id: %d", GetLastError()), 2);
      m_edPTModel.Text("Select file");
      return false;
     }
```

After successfully loading the model, display the name of the loaded file and the number of neural layers in the corresponding elements of the interface.

Delete the description of the previously loaded model, if any. Then call the method for collecting information about the architecture of the loaded model.

```
   m_edPTModel.Text(file_name);
   m_edPTModelLayers.Text((string)m_Model.LayersTotal());
   if(!!m_arPTModelDescription)
      delete m_arPTModelDescription;
   m_arPTModelDescription = m_Model.GetLayersDiscriptions();
```

After receiving information about the loaded model, create a loop, in the body of which output the received information in the corresponding block of the interface.

```
   m_lstPTModel.ItemsClear();
   int total = m_arPTModelDescription.Total();
   for(int i = 0; i < total; i++)
     {
      CLayerDescription* temp = m_arPTModelDescription.At(i);
      if(!temp)
         return false;
      //---
      string item = StringFormat("%s (units %d)", LayerTypeToString(temp.type), temp.count);
      if(!m_lstPTModel.AddItem(item, i))
         return false;
     }
```

At the end of the method, change the range of values for the allowed number of copied neural layers to the total size of the loaded model. Instruct the tool to copy the entire loaded model. Then exit the method.

```
   m_spPTModelLayers.MaxValue(total);
   m_spPTModelLayers.Value(total);
//---
   return true;
  }
```

As you can see, the above method receives the name of the file to load data from the calling program, via the parameters. We need to enable the user to select the model file.

Let's create another OpenPreTrainedModel method. In the body of this method, we only call the standard FileSelectDialog function, which already implements the interface of the file dialog box. At function call, specify the required file extensions and the FSD\_FILE\_MUST\_EXIST flag, which indicates that only an existing file can be specified.

With certain flags, this function allows selecting multiple files. Therefore, as a result of execution, FileSelectDialog returns the number of selected files. The names of the file are contained in the array, a pointer to which the function receives in parameters.

Thus, when the user selects a file, its name is passed in the parameters to the above method. Otherwise, a message is generated prompting that the user should select a file to load data.

```
bool CNetCreatorPanel::OpenPreTrainedModel(void)
  {
   string filenames[];
   if(FileSelectDialog("Select a file to load data", NULL,
                       "Neuron Net (*.nnw)|*.nnw|All files (*.*)|*.*",
                       FSD_FILE_MUST_EXIST, filenames, NULL) > 0)
     {
      if(!LoadModel(filenames[0]))
         return false;
     }
   else
      m_edPTModel.Text("Files not selected");
//---
   return true;
  }
```

We gradually move forward and have already created a visualization of the interface. We have also created a chain of methods for selecting a file and loading a pre-trained model. But so far, these 2 program blocks are not combined into a single organic program. The data loading method displays information about the loaded data model on the panel. But for now, it's a one-way road. We need to specify the way back, at which the program will receive information about the user's actions and the user reaction to information.

To do this, use the event handler. In CAppDialog child classes, this mechanism is implemented through macro substitutions. For this purpose, a block of macros is created in the program code, which begins with the EVENT\_MAP\_BEGIN macro and ends with the EVENT\_MAP\_END macro. Between them are a number of macros corresponding to various events. In our case, we will use the ON\_EVENT macro, which implies event processing by a numeric identifier. To handle the mouse click event on the file name object, we specify in the macro body the ON\_CLICK event, the m\_edPTModel object pointer, and the name of the method to be called when the OpenPreTrainedModel event occurs. Thus, when the mouse button is pressed on the m\_edPTModel object, which corresponds to file name entering box, the program will call the OpenPreTrainedModel method and thereby start the chain of pre-trained model loading methods.

```
EVENT_MAP_BEGIN(CNetCreatorPanel)
ON_EVENT(ON_CLICK, m_edPTModel, OpenPreTrainedModel)
ON_EVENT(ON_CLICK, m_btAddLayer, OnClickAddButton)
ON_EVENT(ON_CLICK, m_btDeleteLayer, OnClickDeleteButton)
ON_EVENT(ON_CLICK, m_btSave, OnClickSaveButton)
ON_EVENT(ON_CHANGE, m_spPTModelLayers, ChangeNumberOfLayers)
ON_EVENT(ON_CHANGE, m_lstPTModel, OnChangeListPTModel)
EVENT_MAP_END(CAppDialog)
```

Similarly, let us describe other events and methods called by them:

- OnClickAddButton — "ADD LAYER" button click event
- OnClickDeleteButton — method handling the DELETE button click event
- OnClickSaveButton — method handling the SAVE MODEL button click
- ChangeNumberOfLayers — method handling the event of changing the number of neural layers to copy
- OnChangeListPTModel — method handling mouse click on a neural layer in the model architecture description link.

The full code of all these methods is available in the attachment. Let us consider the method that solves the new model, since its implementation is rather complicated and requires the creation of additional methods in the CNetModify neural network model class.

The algorithm of this method can be conditionally divided into 3 blocks:

- copying neural layers from a pre-trained model
- adding new neural layers to the model
- saving the model to a file

At the moment, only the last point has been implemented in our neural network class. We have no methods for copying neural layers from another model or for adding new neural layers to an existing model.

Let's go point by point. First we will create a mechanism for copying neural layers. We know that, depending on the architecture of the neural layer, it can contain a different number of objects. However, we need a universal algorithm allowing the copying of all types of neural layers with different parameter optimization methods. Copying of the trained model involves transferring not only the architecture, but also all weights. Now here is the question: Why do we have to copy all elements of each neural layer? Why can't we simply copy the pointer to the necessary object of the neural layer? By using pointers, we can access the same object from different parts of the program code. So, we will use this property. Let's create two methods. One will return a pointer to the neural layer object by its number in the model structure. And the second one will add a pointer to the neural layer object to the model architecture.

```
CLayer* CNetModify::GetLayer(uint layer)
  {
   if(!layers || LayersTotal() <= layer)
      return NULL;
//---
   return layers.At(layer);
  }

bool CNetModify::AddLayer(CLayer *new_layer)
  {
   if(!new_layer)
      return false;
   if(!layers)
     {
      layers = new CArrayLayer();
      if(!layers)
         return false;
     }
//---
   return layers.Add(new_layer);
  }
```

Since we copy a block of successive neural layers, then by transferring pointers to a new model while preserving the sequence, we save all the relationships between such neural layers.

This was the first point. Let us go on. The constructor of our model can create a new model according to the architecture description. When adding neural layers to the model, we created a similar description of neural layers. It would seem, we can simply add new layers, which the model already knows how to do. But the difficulty is in the lack of a bridge between the copied neural layers and the newly created ones.

According to the architecture of our neural layers, the weights of one neural layer are directly related to the elements of another neural layer. Therefore, to maintain the model functioning in the feed forward and backward modes, we need to build this connection. If you look at the initialization method of the CNeuronBaseOCL neural layer base class, you can notice among its parameters the number of neurons in the subsequent neural layer. This parameter determines the size of the weight matrix being created and associated buffers used in parameter optimization.

First we add to the class the CNeuronBaseOCL method that will adjust the weight matrix according to the specified number of neurons in the subsequent layerCNeuronBaseOCL::numOutputs.

In the parameters of the method, we will pass the number of neurons in the subsequent layer and the parameter optimization method.

In the method body, we check the number of elements in the subsequent neural layer received in the parameters and, if necessary, create a weight matrix of the appropriate sizes. Fill it with random weights, since it refers to the newly added neural layer. For the filled matrix, create a buffer in the OpenCL context and pass the matrix contents into it.

It is necessary to pass data to the OpenCL context because our class method will try to load the data from the context before saving the data to the file. In case of an error, it will abort the model saving with a negative result. Of course, we could make changes to the methods of our neural layer classes. But I think such labor costs exceed the costs of transferring information to the OpenCL context and back.

```
bool CNeuronBaseOCL::numOutputs(const uint outputs, ENUM_OPTIMIZATION optimization_type)
  {
   if(outputs > 0)
     {
      if(CheckPointer(Weights) == POINTER_INVALID)
        {
         Weights = new CBufferFloat();
         if(CheckPointer(Weights) == POINTER_INVALID)
            return false;
        }
      Weights.BufferFree();
      Weights.Clear();
      int count = (int)((Output.Total() + 1) * outputs);
      if(!Weights.Reserve(count))
         return false;
      float k = (float)(1 / sqrt(Output.Total() + 1));
      for(int i = 0; i < count; i++)
        {
         if(!Weights.Add((2 * GenerateWeight()*k - k)*WeightsMultiplier))
            return false;
        }
      if(!Weights.BufferCreate(OpenCL))
         return false;
```

After creating the weight matrix, let us create the data buffers used in the weight optimization process.

If there is no need for a matrix of weights and relevant buffers, then remove them as unnecessary. Then exit the method.

The full code of the method is available in the attachment below.

Now, let us get back to the CNetModify class to create a method for adding neural layers according to the given AddLayers description. In method parameters, pass a pointer to the dynamic array with a description of the architecture of neural layers being added. Immediately, in the method body, check the received data. The received pointer must be valid and must contain a description of at least one neural layer.

```
bool CNetModify::AddLayers(CArrayObj *new_layers)
  {
   if(!new_layers || new_layers.Total() <= 0)
      return false;
//---
   if(!layers || LayersTotal() <= 0)
     {
      Create(new_layers);
      return true;
     }
```

Next, check the number of neural layers that exist in the model. If there are none, simply call the constructor of the parent class. It will create a new model with the given architecture.

If we are to add neural layers to an existing model, then we first declare local variables.

```
   CLayerDescription *desc = NULL, *next = NULL;
   CLayer *temp;
   int outputs;
```

Then do a little preparatory work and call the above created method for joining two neural layers.

```
   int shift = (int)LayersTotal() - 1;
   CLayer* last_layer = layers.At(shift);
   if(!last_layer)
      return false;
//---
   CNeuronBaseOCL* neuron = last_layer.At(0);
   if(!neuron)
      return false;
//---
   desc = neuron.GetLayerInfo();
   next = new_layers.At(0);
   outputs = (next == NULL || (next.type != defNeuron && next.type != defNeuronBaseOCL) ? 0 : next.count);
   if(!neuron.numOutputs(outputs, next.optimization))
      return false;
   delete desc;
```

Further, similarly to the constructor of the parent class, loop through the dynamic array of the model architecture description and sequentially add all the neural layers. The code of this block completely repeats the code of the parent class constructor. So, I will not repeat it in this article. The complete code of all methods and classes is available in the attachment below.

Let us get back to the CNetCreatorPanel class of the tool and create a method for handling the model save button press event, which will combine the above methods for creating a new model into a single sequence.

At the beginning of the OnClickSaveButton method, we will prompt the user to specify a file to save the model. To do this, we will use the already familiar FileSelectDialog function. This time we will change the flag to indicate that a file is being created for writing. Also, specify the default file name.

```
bool CNetCreatorPanel::OnClickSaveButton(void)
  {
   string filenames[];
   if(FileSelectDialog("Select files to save", NULL,
                       "Neuron Net (*.nnw)|*.nnw|All files (*.*)|*.*",
                       FSD_WRITE_FILE, filenames, "NewModel.nnw") <= 0)
     {
      Print("File not selected");
      return false;
     }
```

Next, create a new instance of the neural network class and check the result of the operation.

```
   string file_name = filenames[0];
   if(StringLen(file_name) - StringLen(EXTENSION) > StringFind(file_name, EXTENSION))
      file_name += EXTENSION;
   CNetModify* new_model = new CNetModify();
   if(!new_model)
      return false;
```

After successfully creating a new model, implement a loop to copy the required number of neural layers. For all copied neural layers, the learning flag should be switched to false. Thus, we disable the process of updating the weights of these layers in the process of subsequent training. Later, we can programmatically change this flag for all neural layers of the model by literally calling a single method.

```
   int total = m_spPTModelLayers.Value();
   bool result = true;
   for(int i = 0; i < total && result; i++)
     {
      CLayer* temp = m_Model.GetLayer((uint)i);
      if(!temp)
        {
         result = false;
         break;
        }
      CNeuronBaseOCL* neuron = temp.At(0);
      neuron.TrainMode(false);
      if(!new_model.AddLayer(temp))
         result = false;
     }
```

After completing the iterations of copying neural layers, call the above method to add neural layers, which completes the creation of a new model.

```
   new_model.SetOpenCL(m_Model.GetOpenCL());
   if(result && m_arAddLayers.Total() > 0)
      if(!new_model.AddLayers(GetPointer(m_arAddLayers)))
         result = false;
```

After that, we just need to save the created model.

```
   if(result && !new_model.Save(file_name, 1.0e37f, 100, 0, 0, false))
      result = false;
//---
   if(!!new_model)
      delete new_model;
   LoadModel(m_edPTModel.Text());
//---
   return result;
  }
```

After saving the model, we can delete it, since the training will be carried out in another program.

Note that when deleting the model, the copied neural layers will also be deleted. This is because we did not copy data into the new model, but only passed pointers. So, if you want to create another model based on the one already used, then you will need to reload it. To avoid the unnecessary routine, let us call the method for reloading the model. And only after that exit the method.

This concludes the work with the class code. Coming next is testing.

### 3\. Testing

To test the created tool, let us create the NetCreator.mq5 Expert Advisor. The EA code is quite simple and contains only the connection of the above created CNetCreatorPanel class. Actually, the integration of the class into the EA is performed at 3 points. Initialization and launch of the model in the OnInit function. Destroying the class in the OnDeinit function. Passing events to the class in the OnChartEvent method. The code for all integration points is given below.

```
#include "NetCreatorPanel.mqh"
CNetCreatorPanel Panel;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   if(!Panel.Create(0, "NetCreator", 0, 50, 50))
      return INIT_FAILED;
   if(!Panel.Run())
      return INIT_FAILED;
//---
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
//---
   Panel.Destroy(reason);
  }

void OnChartEvent(const int id,
                  const long &lparam,
                  const double &dparam,
                  const string &sparam)
  {
//---
   if(id == CHARTEVENT_OBJECT_CLICK)
      Sleep(0);
   Panel.ChartEvent(id, lparam, dparam, sparam);
  }
```

Practical testing has confirmed our expectation regarding the transferring neural layers from one model to another with the possibility of adding new layers. In addition, the tool allows you to create a completely new model. Thus, you can divert from the description of the created model in the program code.

### Conclusion

In this article, we have created a tool that enables the transfer of part of the neural layers from one model to another. It also enables the addition of an arbitrary number of new layers of arbitrary architecture. I invite everyone to experiment with their previously trained models and see how changing the architecture can affect the productivity of the model.

You can try to combine different architectures in one model and conduct a number of other experiments to change the architecture of the model. At the same time, if you keep the architectures of the result and source data layers, then you can try to "put" a completely new model architecture into an already existing Expert Advisor. Then train the model and compare the influence of the architecture and the error of the model.

### List of references

1. [Neural networks made easy (Part 20): Autoencoders](https://www.mql5.com/en/articles/11172)
2. [Neural networks made easy (Part 21): Variational autoencoders (VAE)](https://www.mql5.com/en/articles/11206)
3. [Neural networks made easy (Part 22): Unsupervised learning of recurrent models](https://www.mql5.com/en/articles/11245)

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | NetCreator.mq5 | EA | Model building tool |
| 2 | NetCreatotPanel.mqh | Class library | Class library for creating the tool |
| 3 | NeuroNet.mqh | Class library | A library of classes for creating a neural network |
| 4 | NeuroNet.cl | Code Base | OpenCL program code library |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/11273](https://www.mql5.com/ru/articles/11273)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/11273.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/11273/mql5.zip "Download MQL5.zip")(71.47 KB)

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

**[Go to discussion](https://www.mql5.com/en/forum/434710)**

![Population optimization algorithms](https://c.mql5.com/2/48/logo.png)[Population optimization algorithms](https://www.mql5.com/en/articles/8122)

This is an introductory article on optimization algorithm (OA) classification. The article attempts to create a test stand (a set of functions), which is to be used for comparing OAs and, perhaps, identifying the most universal algorithm out of all widely known ones.

![DoEasy. Controls (Part 14): New algorithm for naming graphical elements. Continuing work on the TabControl WinForms object](https://c.mql5.com/2/48/MQL5-avatar-doeasy-library-2__2.png)[DoEasy. Controls (Part 14): New algorithm for naming graphical elements. Continuing work on the TabControl WinForms object](https://www.mql5.com/en/articles/11288)

In this article, I will create a new algorithm for naming all graphical elements meant for building custom graphics, as well as continue developing the TabControl WinForms object.

![Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)](https://c.mql5.com/2/48/development.png)[Developing a trading Expert Advisor from scratch (Part 24): Providing system robustness (I)](https://www.mql5.com/en/articles/10593)

In this article, we will make the system more reliable to ensure a robust and secure use. One of the ways to achieve the desired robustness is to try to re-use the code as much as possible so that it is constantly tested in different cases. But this is only one of the ways. Another one is to use OOP.

![Learn how to design a trading system by Alligator](https://c.mql5.com/2/49/trading-system-by-Alligator.png)[Learn how to design a trading system by Alligator](https://www.mql5.com/en/articles/11549)

In this article, we'll complete our series about how to design a trading system based on the most popular technical indicator. We'll learn how to create a trading system based on the Alligator indicator.

[![](https://www.mql5.com/ff/si/m0dtjf9x3brdz07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dtop.experts%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=widauvjabtsckwovwaperzkotrcrttvb&s=25ef75d39331f608a319410bf27ff02c1bd7986622ecc1eec8968a650f044731&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=lvsrznwflroikswbywoteztitapzbaxq&ssn=1769157937820545564&ssn_dr=0&ssn_sr=0&fv_date=1769157937&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F11273&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Neural%20networks%20made%20easy%20(Part%2023)%3A%20Building%20a%20tool%20for%20Transfer%20Learning%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176915793733651588&fz_uniq=5062695633971750704&sv=2552)

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