---
title: Introduction to MQL5 (Part 5): A Beginner's Guide to Array Functions in MQL5
url: https://www.mql5.com/en/articles/14306
categories: Trading, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:02:17.776621
---

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/14306&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068962747170881399)

MetaTrader 5 / Trading


### Introduction

Part 5 of our series will introduce you to the fascinating world of MQL5, designed especially for complete novices looking for a gentle introduction to the intricacies of array functions. This section aims to dismantle the misconceptions that are frequently associated with array functions, guaranteeing that each line of code is not only understood but comprehended thoroughly. Regardless of prior coding experience, I sincerely believe that everyone should have the opportunity to learn about the MQL5 language, which is why I will always be committed to creating an inclusive environment.

In this article, simplicity and clarity are the primary themes. I want to serve as a conduit for people who are curious about coding and the uncharted territory of it. Though they can be confusing at first, I intend to walk you through each array function one line at a time so that you have a fun and educational learning experience. Together, we will solve the puzzles surrounding array functions and equip you with the knowledge necessary to successfully negotiate the complex world of algorithmic trading. This isn't just an article — it's an invitation to go on an amazing journey of coding transformation.

But things don't stop here. Beyond the code, we hope to create a community where both novice and seasoned programmers can congregate to exchange ideas, pose queries, and promote teamwork. This is an invitation to go on a life-changing coding adventure, not just an article. Greetings from Part 5, where knowledge and accessibility collide and everyone who codes is appreciated. Have fun with coding!

In this article, we will cover the following array functions:

- ArrayBsearch

- ArrayResize

- ArrayCopy

- ArrayCompare

- ArrayFree
- ArraySetAsSeries
- ArrayGetAsSeries

- ArrayIsSeries

- ArrayInitialize

- ArrayFill

- ArrayIsDynamic

- ArrayMaximum

- ArrayMinimum

I'd like to share a video that summarizes the lessons we learned in [Part 4](https://www.mql5.com/en/articles/14232) before we get into Part 5. This is a summary to make sure everyone is aware of the situation. Let's keep making MQL5 arrays easier to understand for total novices while building a supportive and knowledge-sharing community. Embark with me on this coding journey!

### 1\. ArrayBsearch

You can use the "ArrayBsearch()" function for arrays that are arranged in ascending order. This suggests sorting the values in the array from smallest to largest in ascending order. The function employs a binary search technique, which produces results reliably for sorted arrays but may not work well for unsorted or randomly ordered arrays. Therefore, to perform effective and accurate searches, you must ensure your array is correctly sorted before using "ArrayBsearch()".

**Analogy**

Assume you have a set of numbers that are sorted from smallest to largest in a specific order. Imagine you are trying to search through this sorted list for a specific number, say 30. Rather than manually going through each number, the "ArrayBsearch()" function functions as a smart advisor. It informs you that 30 is located at position 2 (Index 2) in the list and swiftly directs you to the appropriate location. It's like having a helpful friend who methodically expedites your search!

**Syntax:**

```
int ArrayBsearch(array[],value);
```

**Explanation:**

- **“int":** This is the data type that the function returns. In this case, it's an integer, which represents the index of the found or suggested position of the value in the array.
- **“ArrayBsearch”:** This is the name of the function
- **“array\[\]”:** Array to be searched.
- **“value”:** This is the value to be searched for in the array.

**Example:**

```
void OnStart()
  {

// Declare an array of sorted numbers
   double sortedArray[] = {10, 20, 30, 40, 50};

// Value to search for
   double searchValue = 30;

// Call ArrayBsearch function
   int resultIndex = ArrayBsearch(sortedArray, searchValue);

// Print out the index of 30 in the array
   Print("Found the resultIndex at index ", resultIndex); // The output will be index 2

  }
```

**Explanation:**

**“double sortedArray\[\] = {10, 20, 30, 40, 50};”:**

- This line declares an array named “sortedArray” containing sorted numbers {10, 20, 30, 40, 50}.

**“double searchValue = 30;”:**

- This line sets the “searchValue” to 30, the number we want to find in the array.

**“int resultIndex = ArrayBsearch(sortedArray, searchValue);”:**

- This line calls the “ArrayBsearch()” function, passing the “sortedArray” and “searchValue” as arguments. It returns the index where the “searchValue” is found or the suggested insertion point if the value is not present.

**“Print("Found the resultIndex at index ", resultIndex);”:**

- This line prints the result of the search. If “searchValue” is found, it prints the index; otherwise, it prints the suggested insertion point.

In this example, “sortedArray” is the array where we're searching, and “searchValue” is the value we want to find in the array. The function returns the index where the value is found and prints the result.

### 2. **ArrayResize**

The MQL5 function "ArrayResize()" lets you modify a dynamic array's size while the program is running. Dynamic arrays allow for size adjustments during program execution, in contrast to static arrays, whose size is predetermined. In other words, "ArrayResize()" is a tool that, when your program is running, allows you to resize or expand a dynamic array according to your needs at that precise moment. It increases the flexibility to handle data during runtime more effectively.

Programming's static arrays have a fixed size that's set at the program compilation stage. The number of elements stays constant because the size is fixed and cannot be changed during runtime. The memory allotted for these arrays is determined by the size at which they are declared. An array with five elements, for example, would always have room for five elements.

**Example:**

```
// Static array declaration
int staticArray[5] = {1, 2, 3, 4, 5};
```

Conversely, dynamic arrays offer flexibility since they let the size be adjusted or determined while the program is running. These arrays are declared without a size at first, and in MQL5, functions like "ArrayResize()" can be used to change the memory allocation. When a data structure's size needs to be flexible to accommodate different numbers of elements as needed during the program's execution, dynamic arrays are especially helpful.

**Example:**

```
// Dynamic array declaration
int dynamicArray[];
```

**Analogy**

Let's say you have a magical backpack (array) that has a ton of toys (elements) in it. When embarking on an adventure, you can choose the number of toys a static backpack can accommodate, and it will remain that way the entire way. You're in trouble if you want to carry more toys than it can accommodate.

A dynamic backpack can be thought of as something special that can grow to fit additional toys or toys you wish to share with friends. To change the size of your backpack and carry as many toys as you need for your magical adventure, use "ArrayResize()" as you would a spell.

You can change the size of the dynamic backpack while you're out and about, so you're not limited to it. This adaptability helps to ensure that your magical journey is always full of surprises and excitement, whether you find new toys or decide to share them with others! It is analogous to saying to the array, "Hey, get ready for more elements!" or "All right, let's make some room." This dynamic adjustment that takes place while the program is running offers versatility and flexibility, which makes it an invaluable tool for arrays whose initial sizes are unknown.

**Syntax**:

```
ArrayResize
(
    array[],          // Reference to the array to be resized
    new_size,         // New size for the array
    reserve_size = 0  // Optional space reserved for future elements
);
```

**Parameters:**

- **“Array\[\]”:** This is your toy box (array) that you want to resize.
- **“new\_size”:** This is the number of toys (elements) you want your box to hold now. If you had 5 toys, and you wanted space for 10, “new\_size” would be 10.
- **“reserve\_size = 0”:** Sometimes, you might want to make room for even more toys in the future without resizing again. The “reserve\_size” is like saying, "Hey, be ready for more toys!"

**Example:**

```
void OnStart()
  {

// Dynamic array declaration
   int dynamicArray[];

// Resizing the dynamic array to have 5 elements
   ArrayResize(dynamicArray, 5);

// Assigning values to dynamic array elements
   dynamicArray[0] = 10;
   dynamicArray[1] = 20;
   dynamicArray[2] = 30;
   dynamicArray[3] = 40;
   dynamicArray[4] = 50;

// Accessing elements in a dynamic array
   Print("Element at index 2: ", dynamicArray[2]); // Output: 30

// Resizing the dynamic array to have 8 elements
   ArrayResize(dynamicArray, 8);

// Assigning values to the additional elements
   dynamicArray[5] = 60;
   dynamicArray[6] = 70;
   dynamicArray[7] = 80;

// Accessing elements after resizing
   Print("Element at index 6: ", dynamicArray[6]); // Output: 70

  }
```

**Explanation:**

**Dynamic Array Declaration:**

```
int dynamicArray[];
```

- Here, we declare a dynamic array named “dynamicArray()” without specifying its initial size.

**Resizing the Dynamic Array to 5 Elements:**

```
ArrayResize(dynamicArray, 5);
```

- The “ArrayResize()” function is used to set the size of the dynamic array to 5 elements.

**Assigning Values to Dynamic Array Elements:**

```
dynamicArray[0] = 10;
dynamicArray[1] = 20;
dynamicArray[2] = 30;
dynamicArray[3] = 40;
dynamicArray[4] = 50;
```

- Values are assigned to individual elements of the dynamic array.

**Accessing Elements in a Dynamic Array:**

```
Print("Element at index 2: ", dynamicArray[2]); // Output: 30
```

- The “Print” function is used to display the value at index 2 of the dynamic array. In this case, it will print “30”.

**Resizing the Dynamic Array to 8 Elements:**

```
ArrayResize(dynamicArray, 8);
```

- The dynamic array is resized again to have 8 elements, and it retains the values from the previous resizing.

**Assigning Values to Additional Elements:**

```
dynamicArray[5] = 60;
dynamicArray[6] = 70;
dynamicArray[7] = 80;
```

- Additional values are assigned to the newly added elements after resizing.

**Accessing Elements after Resizing:**

```
Print("Element at index 6: ", dynamicArray[6]); // Output: 70
```

- The Print function is used to display the value at index 6 of the dynamic array after the second resizing. In this case, it will print 70

### 3. **ArrayCopy**

In MQL5, the function "ArrayCopy()" is used to duplicate elements between arrays. It enables you to replicate a specific range-defined portion of an array into another array in a selective manner. This function makes it easier to manage and arrange data inside arrays, which makes it easier to extract and move particular elements between arrays.

**Analogy**

Consider that you have two lists of items and that you wish to copy certain items exactly from the first list to the second. This is where MQL5's "ArrayCopy()" function is useful. It functions as a copy assistant, letting you select particular items from an array and neatly copy them into another list.

Here's a more concrete example: imagine you have an array with five different item prices, and you want to create a second array that contains the prices of just three particular items. You can neatly extract and duplicate just those three prices into a new array while maintaining the original array by using the "ArrayCopy()" function. It's similar to having a useful tool that makes copying and choosing items from one collection to another easier, increasing the efficiency and organization of your array manipulation tasks.

**Syntax:**

```
ArrayCopy(
          dst_array[],         // The destination array to receive copied elements
          src_array[],         // The source array from which elements will be copied
          dst_start=0,         // The index in the destination array to start writing from
          src_start=0,         // The index in the source array from which to start copying
          count                // The number of elements to copy; default is to copy the entire array
);
```

This powerful command empowers you to skillfully merge arrays with precision and control. In this enchanting process, “dst\_array” serves as the destination where elements will be copied, and “src\_array” acts as the source from which elements are drawn. Additional parameters such as “dst\_start”, “src\_start”, and “count” provide the flexibility to finely adjust the merging operation. Think of it as crafting a command that orchestrates the flawless fusion of arrays in the captivating domain of MQL5 programming!

**Example:**

```
void OnStart()
  {

// Declare two dynamic arrays
   int sourceArray[];
   int destinationArray[];

// Resizing the dynamic arrays to have 5 elements each
   ArrayResize(sourceArray,5);
   ArrayResize(destinationArray,5);

// Assigning values to dynamic array elements
   sourceArray[0] = 1;
   sourceArray[1] = 2;
   sourceArray[2] = 3;
   sourceArray[3] = 4;
   sourceArray[4] = 5;

   destinationArray[0] = 10;
   destinationArray[1] = 20;
   destinationArray[2] = 30;
   destinationArray[3] = 40;
   destinationArray[4] = 50;

// Copy elements from sourceArray to destinationArray starting from index 1
   ArrayCopy(destinationArray, sourceArray, 5, 0, WHOLE_ARRAY);

// Print the value of the element at index 7 in destinationArray
   Comment("Value at index 7 in destinationArray: ", destinationArray[7]);

  }
```

**Explanation:**

**Declaration of Arrays:**

```
int sourceArray[];
int destinationArray[];
```

- Here, we declare two dynamic arrays named “sourceArray” and “destinationArray”.


**Resizing Arrays:**

```
ArrayResize(sourceArray, 5);
ArrayResize(destinationArray, 5);
```

- “The ArrayResize()” function is used to set the size of the dynamic arrays. In this case, both arrays are resized to have 5 elements each.


**Assigning Values:**

```
sourceArray[0] = 1;
sourceArray[1] = 2;
sourceArray[2] = 3;
sourceArray[3] = 4;
sourceArray[4] = 5;

destinationArray[0] = 10;
destinationArray[1] = 20;
destinationArray[2] = 30;
destinationArray[3] = 40;
destinationArray[4] = 50;
```

- Values are assigned to individual elements of the “sourceArray” and “destinationArray”.


**Array Copy:**

```
ArrayCopy(destinationArray, sourceArray, 5, 0, WHOLE_ARRAY);
```

- The ArrayCopy() function is employed to copy elements from “sourceArray” to “destinationArray”. It specifies copying 5 elements starting from index 0.


**Print Value:**

```
Comment("Value at index 7 in destinationArray: ", destinationArray[7]);
```

- A comment is printed, displaying the value at index 7 in “destinationArray”.


The overall purpose of the code is to demonstrate the “ArrayCopy()” function by copying elements from “sourceArray” to “destinationArray” starting from specific indices. The last line prints the value of an element in “destinationArray” to confirm the successful copy.

### 4. **ArrayCompare**

The “ArrayCompare()” function in MQL5 serves as a tool for comparing two arrays and evaluating their elements systematically. It initiates the comparison from the beginning (index 0) of both arrays, checking if the elements at corresponding indices are equal. If all elements match, the arrays are considered equal. However, if a discrepancy arises at any index, the function assesses which array holds the numerically greater element, providing a basis for determining their relationship. This function is particularly useful for gauging the similarity or dissimilarity between arrays in terms of their contents.

**Analogy**

Picture a scenario where you have two lists of numbers: List A and List B. "ArrayCompare()" functions as a sort of specialized investigator, analyzing these lists and informing us about their relationships. Starting with the numbers at the beginning of both lists, the investigator compares them. When it detects a discrepancy in the numbers, it determines which list is "greater" or "lesser" right away. It determines that the lists are "equal" if it can review both lists and finds nothing unusual.

Now, the detective has a unique way of reporting its findings:

- If List A is considered less than List B, it reports -1.
- If both lists are considered equal, the report is 0.
- If List A is considered greater than List B, the report is 1.
- If there's any confusion or problem during the investigation, it reports -2.

So, “ArrayCompare()” helps us understand the relationship between two lists of numbers, just like a detective figuring out who's who in a case.

**Syntax:**

```
int ArrayCompare(const void& array1[], const void& array2[], int start1 = 0, int start2 = 0, int count = WHOLE_ARRAY);
```

**Parameters:**

- array1\[\]: First array.
-  array2\[\]: Second array.
- start1: The initial element's index in the first array from which the comparison starts. The default start index is 0.
- start2: The initial element's index in the second array from which the comparison starts. The default start index is 0.
- count: The number of elements to be compared. All elements of both arrays participate in comparison by default (count = WHOLE\_ARRAY).

**Example:**

```
void OnStart()
  {

// Declare two arrays
   int ListA[] = {1, 2, 3, 4, 5};
   int ListB[] = {1, 2, 3, 4, 6};
// Use ArrayCompare to compare the arrays
   int result = ArrayCompare(ListA, ListB, 0, 0, WHOLE_ARRAY);
// Print the result
   if(result == -1)
      {
      Print("ListA is less than ListB");
      }
   else if(result == 0)
      {
      Print("ListA is equal to ListB");
      }
   else if(result == 1)
      {
      Print("ListA is greater than ListB");
      }
    else if(result == -2)
      {
       Print("Error: Incompatible arrays or invalid parameters");
      }

  }
```

**Explanation:**

**“int ListA\[\] = {1, 2, 3, 4, 5};”:**

- Declares an integer array named “ListA” and initializes it with values 1, 2, 3, 4, and 5.

**“int ListB\[\] = {1, 2, 3, 4, 6};”:**

- Declares an integer array named “ListB” and initializes it with values 1, 2, 3, 4, and 6.

**“int result = ArrayCompare(ListA, ListB, 0, 0, WHOLE\_ARRAY);”:**

- Uses the “ArrayCompare()” function to compare the arrays “ListA” and “ListB”. The comparison starts from index 0 of both arrays, and it compares the whole arrays.

**The conditional statements (“if”, “else if”) check the value of the result variable and print messages based on the comparison result:**

- If “result” is “-1”, it means “ListA” is considered less than “ListB”.
- If “result” is “0”, it means “ListA” is equal to “ListB”.
- If “result” is “1”, it means “ListA” is considered greater than “ListB”.
- If “result” is “-2”, it indicates an error due to incompatible arrays or invalid parameters.

Given the arrays:

```
int ListA[] = {1, 2, 3, 4, 5};
int ListB[] = {1, 2, 3, 4, 6};
```

The result of ArrayCompare(ListA, ListB, 0, 0, WHOLE\_ARRAY)” will be -1.

**Explanation:**

- The comparison starts at the first element (index 0) of both arrays.
- Elements at indices 0 to 3 are the same in both arrays.
- At index 4, ListA has 5, while ListB has 6.
- Since 5 < 6, ListA is considered less than ListB.

Therefore, the result will be -1. Feel free to modify the values in ListA and ListB to see how the comparison result changes!

### 5. **ArrayFree**

In MQL5, calling "ArrayFree()" is akin to pressing the reset button for your dynamic array. Consider your array as a container for different items. It is similar to emptying the container and preparing it to hold new items when you use "ArrayFree()." It's a means of making room for new data by getting rid of outdated information. To put it another way, think of it as clearing the slate for whatever comes next. By using this function, you can be sure that your array is empty and prepared for new MQL5 programming experiences.

**Analogy**

Imagine you have a magical bag — your array. Sometimes, you want to use it for different things, like collecting toys. But before getting new toys, you need to make sure the bag is empty. That's what “ArrayFree()” does — it waves a wand and clears your bag so you can put in new toys or numbers. It's like saying, "Okay, bag, get ready for more fun stuff!" This way, you're all set for new adventures with your magical bag in the world of MQL5.

**Syntax:**

```
ArrayFree(array[] // dynamic array to be freed);
```

**Example:**

```
void OnStart()
  {

// Declare a dynamic array
   int dynamicArray[];
// Resize the dynamic array and assign values
   ArrayResize(dynamicArray, 5);
   dynamicArray[0] = 10;
   dynamicArray[1] = 20;
   dynamicArray[2] = 30;
   dynamicArray[3] = 40;
   dynamicArray[4] = 50;

// Print elements before freeing the array
   Print("Index 0 before freeing: ", dynamicArray[0]); // Output will be 10

// Free the dynamic array using ArrayFree
   ArrayFree(dynamicArray);

// Attempting to access elements after freeing (should result in an error)
//   Print("Index 0 after freeing: ", dynamicArray[0]);

// Reassign new values to the array
   ArrayResize(dynamicArray, 3);
   dynamicArray[0] = 100;
   dynamicArray[1] = 200;
   dynamicArray[2] = 300;

// Print elements after reassigning values
   Print("Index 0 after reassigning: ", dynamicArray[0]); // Output will be 100

  }
```

**Explanation:**

**Declare a dynamic array:**

```
int dynamicArray[];
```

- Initializes an empty dynamic array.


**Resize and assign values:**

```
ArrayResize(dynamicArray, 5);
dynamicArray[0] = 10;
dynamicArray[1] = 20;
dynamicArray[2] = 30;
dynamicArray[3] = 40;
dynamicArray[4] = 50;
```

- Resizes the dynamic array to have 5 elements and assigns values to each element.


**Print elements before freeing:**

```
Print("Elements before freeing: ", dynamicArray[0]); // Output will be 10
```

- Prints the value at the first index of the array, which is 10.


**Free the dynamic array:**

```
ArrayFree(dynamicArray);
```

- Releases the memory occupied by the dynamic array.


**Attempt to access elements after freeing:**

```
// Print("Elements after freeing: ", dynamicArray[0]);
```

- This line is commented out to avoid runtime errors since the array has been freed.


**Assign new values:**

```
ArrayResize(dynamicArray, 3);
dynamicArray[0] = 100;
dynamicArray[1] = 200;
dynamicArray[2] = 300;
```

- Resizes the array to have 3 elements and assigns new values.


**Print elements after reassigning values:**

```
Print("Elements after reassigning: ", dynamicArray[0]); // Output will be 100
```

- Prints the value at the first index of the array after reassigning, which is 100.


In this example, after freeing the dynamic array using “ArrayFree()”, we resize it again to have 3 elements and assign new values to those elements. This demonstrates how you can reuse a dynamic array after freeing it.

There's more magic to be discovered as we work through the complexities of MQL5 array functions. Stay tuned for in-depth explorations of more features that will improve your proficiency with code. The path is far from over, regardless of your level of experience as a developer. As we delve deeper into the marvels in the next sections, exciting discoveries are in store. Let's continue this coding journey together as you maintain your curiosity!

### 6. **ArraySetAsSeries**

In MQL5 programming, “ArraySetAsSeries()” is a function that allows you to modify the indexing direction of an array. By using this function, you can set the array to be accessed from the end to the beginning, altering the default forward direction. This is particularly useful when dealing with financial data or other arrays where accessing elements in reverse chronological order is beneficial.

_Note: It's important to note that this enchantment specifically works on dynamic arrays, the ones that can gracefully adjust their size during runtime._

**Analogy**

Imagine you have a stack of enchanted storybooks neatly arranged on a shelf. Each book is like a special number, waiting for you to explore its exciting tale. Normally, you read the stories in the order they appear on the shelf, starting from the first book and moving toward the last.

Imagine you want to embark on a quest to discover the latest story you added to your collection without taking all the books off the shelf. That's where the enchantment of “ArraySetAsSeries()” comes in! When you cast this spell on your bookshelf (array), it's like saying, "Let's rearrange the stories so that the newest one you added magically appears first." This is especially helpful when your stories (numbers) change over time, like recording how many new books you collect each day. With "ArraySetAsSeries()," you get to open the latest storybook first and journey backward through your magical library to see how your collection has grown. It's like having a reverse-reading spell for your extraordinary literary adventures!

**Syntax:**

```
ArraySetAsSeries(
   array[],    // array to be set as series
   bool   flag // true denotes reverse order of indexing
);
```

**Parameters:**

- **“array\[\]”:** This is the array that you want to enchant with time-series properties. It's like selecting the magical artifact that you want to imbue with special powers.
- **“bool flag”:** This is a boolean value. When set to “true”, it activates the mystical reversal of indexing, turning the array into a time-series wonder where the last element becomes the first. If set to “false”, the array behaves in the regular, non-magical way.

**Example:**

```
void OnStart()
  {

// Declare a dynamic array
   int magicalArray[];

// Assign values to the array
   ArrayResize(magicalArray, 5);
   magicalArray[0] = 10;
   magicalArray[1] = 20;
   magicalArray[2] = 30;
   magicalArray[3] = 40;
   magicalArray[4] = 50;

// Print elements before setting as series
   Print("Elements before setting as series:");
   Print("Index 0: ", magicalArray[0]);
   Print("Index 1: ", magicalArray[1]);
   Print("Index 2: ", magicalArray[2]);
   Print("Index 3: ", magicalArray[3]);
   Print("Index 4: ", magicalArray[4]);

// Set the array as a series
   ArraySetAsSeries(magicalArray, true);

// Print elements after setting as series
   Print("Elements after setting as series:");
   Print("Index 0: ", magicalArray[0]);
   Print("Index 1: ", magicalArray[1]);
   Print("Index 2: ", magicalArray[2]);
   Print("Index 3: ", magicalArray[3]);
   Print("Index 4: ", magicalArray[4]);

  }
```

**Explanation:**

**Dynamic Array Declaration:**

```
double magicalArray[];
```

- Declares a dynamic array named “magicalArray” without specifying its size.


**Assign Values to the Array:**

```
ArrayResize(magicalArray, 5);
magicalArray[0] = 10;
magicalArray[1] = 20;
magicalArray[2] = 30;
magicalArray[3] = 40;
magicalArray[4] = 50;
```

- We resize the array to have 5 elements.

- We then assign specific values to each element of the array.


**Print Elements before Setting as Series:**

```
Print("Elements before setting as series:");
Print("Index 0: ", magicalArray[0]); // output will be 10
Print("Index 1: ", magicalArray[1]); // output will be 20
Print("Index 2: ", magicalArray[2]); // output will be 30
Print("Index 3: ", magicalArray[3]); // output will be 40
Print("Index 4: ", magicalArray[4]); // output will be 50
```

- This section prints the values of each element of the array before setting it as a series.


**Set the Array as a Series:**

```
ArraySetAsSeries(magicalArray, true);
```

- We use “ArraySetAsSeries()” to set the array as a series. The second parameter “true” indicates reverse order of indexing.


**Print Elements after Setting as Series:**

```
Print("Elements after setting as series:");
Print("Index 0: ", magicalArray[0]); // output will be 50
Print("Index 1: ", magicalArray[1]); // output will be 40
Print("Index 2: ", magicalArray[2]); // output will be 30
Print("Index 3: ", magicalArray[3]); // output will be 20
Print("Index 4: ", magicalArray[4]); // output will be 10
```

- Finally, we print the values of each element after setting the array as a series. The order of printing reflects the reversed indexing due to setting it as a series.


In summary, the code demonstrates how to assign values to a dynamic array, print its elements before and after setting it as a series, and observe the change in indexing order.

In the enchanting journey of MQL5 programming, we've explored the magical function “ArraySetAsSeries()”. It's like waving a wand to reverse the order of an array's time-traveling capabilities! Remember that practice makes perfect as we close out this chapter. Try new things, read slowly, and feel free to ask questions. You can get assistance from the community on your magical coding adventures. To more coding, cheers!"

### 7. **ArrayGetAsSeries**

The “ArrayGetAsSeries()” function in MQL5 is used to determine if an array has the AS\_SERIES flag set. This flag affects the order in which array elements are accessed. If the function returns true, it indicates that elements are accessed in reverse order; otherwise, if it returns false, the array maintains its default order. This function is handy when dealing with arrays where the sequence of data access is crucial, and it provides a way to check and adapt the data access pattern based on the array's configuration.

**Analogy**

Imagine you have a magical list of numbers, and sometimes this list likes to play a special game called "Time Travel." When you ask this magical tool, "ArrayGetAsSeries()" it tells you if your list is playing the game or not. If it says "true," it means the list is playing, and you read the numbers backward, like counting down. If it says "false," the list is just normal, and you read the numbers from the start to the end, like counting up. So, it helps you understand the rules of your magical list!

**Syntax:**

```
bool ArrayGetAsSeries(
array[]    // // The array that is being examined for its time series configuration.
);
```

**Example:**

```
void OnStart()
  {

// Declare two dynamic arrays
   int timeSeriesArray[];
   int regularArray[];
// Resize the arrays to have 5 elements
   ArrayResize(timeSeriesArray, 5);
   ArrayResize(regularArray, 5);
// Assign values to the arrays
   timeSeriesArray[0] = 1;
   timeSeriesArray[1] = 2;
   timeSeriesArray[2] = 3;
   timeSeriesArray[3] = 4;
   timeSeriesArray[4] = 5;

   regularArray[0] = 5;
   regularArray[1] = 4;
   regularArray[2] = 3;
   regularArray[3] = 2;
   regularArray[4] = 1;
// Set the time series flag for the first array
   ArraySetAsSeries(timeSeriesArray, true);
// Check if the dynamic arrays follow the time series convention using if statements
   if(ArrayGetAsSeries(timeSeriesArray))
     {
      Print("timeSeriesArray is a time series. Elements are accessed from end to beginning.");
     }
   else
     {
      Print("timeSeriesArray maintains its original order. Elements are accessed from beginning to end.");
     }

   if(ArrayGetAsSeries(regularArray))
     {
      Print("regularArray is a time series. Elements are accessed from end to beginning.");
     }
   else
     {
      Print("regularArray maintains its original order. Elements are accessed from beginning to end.");
     }

  }
```

**Explanation:**

```
// Declare two dynamic arrays
int timeSeriesArray[];
int regularArray[];
```

- These lines declare two dynamic arrays named “timeSeriesArray” and “regularArray”. Dynamic arrays in MQL5 can change in size during runtime.


```
// Resize the arrays to have 5 elements
ArrayResize(timeSeriesArray, 5);
ArrayResize(regularArray, 5);
```

- These lines use the “ArrayResize()” function to set the size of both arrays to 5 elements. This step ensures that the arrays have enough space to store elements.


```
// Assign values to the arrays
timeSeriesArray[0] = 1;
timeSeriesArray[1] = 2;
timeSeriesArray[2] = 3;
timeSeriesArray[3] = 4;
timeSeriesArray[4] = 5;

regularArray[0] = 5;
regularArray[1] = 4;
regularArray[2] = 3;
regularArray[3] = 2;
regularArray[4] = 1;
```

- These lines assign specific values to the elements of both arrays. “timeSeriesArray” is assigned values in ascending order, while “regularArray” is assigned values in descending order.


```
// Set the time series flag for the first array
ArraySetAsSeries(timeSeriesArray, true);
```

- This line uses the “ArraySetAsSeries()” function to set the time series flag for “timeSeriesArray” to “true”. This means that elements in “timeSeriesArray” will be accessed from end to beginning.


```
// Check if the dynamic arrays follow the time series convention using if statements
    if(ArrayGetAsSeries(timeSeriesArray))
    {
        Print("timeSeriesArray is a time series. Elements are accessed from end to beginning.");
    }
    else
    {
        Print("timeSeriesArray maintains its original order. Elements are accessed from beginning to end.");
    }

    if(ArrayGetAsSeries(regularArray))
    {
        Print("regularArray is a time series. Elements are accessed from end to beginning.");
    }
    else
    {
        Print("regularArray maintains its original order. Elements are accessed from beginning to end.");
    }
```

- The provided code snippet checks whether the dynamic arrays, “timeSeriesArray” and “regularArray”, adhere to the time series convention using conditional statements. It utilizes the “ArrayGetAsSeries()” function to determine if the time series flag is set for each array. The first “if” statement checks “timeSeriesArray”, and if it is identified as a time series, a corresponding message is printed indicating that its elements are accessed from end to beginning. If not, the “else” block prints a message stating that “timeSeriesArray” maintains its original order, and elements are accessed from beginning to end. The process is repeated for “regularArray”. This conditional check is crucial for understanding how elements within these dynamic arrays are indexed, providing valuable insights into the direction of array access.


Understanding the intricacies of “ArrayGetAsSeries()” is a valuable skill in MQL5. Whether you're navigating time series data or working with arrays in their original order, these functions empower you in your algorithmic trading journey. As a beginner, you should ask questions, and with that, we can build the community together. Happy coding!

### 8. **ArrayIsSeries**

When determining whether an array in MQL5 represents a timeseries, the "ArrayIsSeries()" function is essential. An array that contains time-related data is called a timeseries in financial programming, and it is frequently used to store price values like open, high, low, and close prices. When a timeseries is detected, the function analyzes the provided array and returns "true"; otherwise, it returns "false." This determination is essential when working with financial data on a chart, where understanding the temporal nature of the data is crucial.

When creating custom indicators in MQL5, especially in the context of technical analysis, it becomes essential to differentiate between regular arrays and timeseries arrays. The “ArrayIsSeries()” function simplifies this process, allowing developers to tailor their code based on whether the array contains time-dependent information. This function contributes to the efficiency and accuracy of algorithmic trading strategies, technical analysis tools, and other financial applications developed using the MQL5 language.

**Difference between ArrayGetAsSeries and ArrayIsSeries**

Both “ArrayGetAsSeries()” and “ArrayIsSeries()” are functions that pertain to array behavior, but they serve distinct purposes. “ArrayGetAsSeries()” is employed to check whether the indexing of an array is set to retrieve elements from the back to the front, commonly referred to as reverse order. This function is valuable when manipulating arrays, allowing developers to ascertain whether data is accessed in a chronological or reversed manner. It returns “true” if the array is set as a series (accessed in reverse order) and “false” otherwise.

On the other hand, “ArrayIsSeries()” is focused on identifying whether an array is a timeseries. Timeseries arrays are prevalent in financial programming, representing data such as open, high, low, and close prices over time. Unlike “ArrayGetAsSeries()”, “ArrayIsSeries()” doesn't concern itself with the direction of array indexing. Instead, it checks if the array contains time-related information. If the array is a timeseries, it returns “true”; otherwise, it returns “false”. These functions complement each other in providing a comprehensive toolkit for handling array behavior, offering flexibility when dealing with various types of financial data in algorithmic trading systems and technical analysis tools.

**Analogy**

Imagine you have a list of things, like the prices of your favorite toys, every day. Now, if we want to know if this list is special and related to time, just like a story, we can use the magic spell called “ArrayIsSeries()”. This spell checks if our list has a time-traveling touch, making it a "timeseries." It doesn't care if the list reads backward or forward; it's more interested in knowing if it's like a time-traveling adventure.

So, if the spell says "true," it means our list is like a time-traveling tale, maybe showing the prices of toys over days. But if it says "false," our list might just be a regular collection of numbers without any time-related magic. It's like asking, "Is this list a special time-traveling story?" And the spell gives us a simple answer — yes or no!

**Syntax:**

```
bool ArrayIsSeries(
array[] //the array you want to check if it's a timeseries.
)
```

**Example:**

```
void OnStart()
  {

// Declare an array
   double priceSeries[];
// Resize the array and assign values (considering it as a time series)
   ArrayResize(priceSeries, 5);
   priceSeries[0] = 1.1;
   priceSeries[1] = 1.2;
   priceSeries[2] = 1.3;
   priceSeries[3] = 1.4;
   priceSeries[4] = 1.5;
// Check if the array is a time series
   bool isSeries = ArrayIsSeries(priceSeries);
// Print the result
   if(isSeries)
     {
      Print("This array is a time series!");
     }
   else
     {
      Print("This array is not a time series.");
     }

  }
```

**Explanation:**

```
// Declare an array
double priceSeries[];
```

- This line declares an empty dynamic array named “priceSeries” to store double values.


```
// Resize the array and assign values (considering it as a time series)
ArrayResize(priceSeries, 5);
priceSeries[0] = 1.1;
priceSeries[1] = 1.2;
priceSeries[2] = 1.3;
priceSeries[3] = 1.4;
priceSeries[4] = 1.5;
```

- Here, the array is resized to have 5 elements, and specific values are assigned to each element. These values represent a hypothetical time series.


```
// Check if the array is a time series
bool isSeries = ArrayIsSeries(priceSeries);
```

- This line uses the “ArrayIsSeries()” function to check whether the array “priceSeries” is considered a time series. The result (“true” or “false”) is stored in the boolean variable “isSeries”.


```
// Print the result
if (isSeries) {
    Print("This array is a time series!");
} else {
    Print("This array is not a time series.");
}
```

- Finally, the code prints a message indicating whether the array is considered a time series based on the result obtained from “ArrayIsSeries()”. If it is a time series, it prints one message; otherwise, it prints another message.


The output in this case is 'This array is not a time series.' Why? Because our array does not represent a time series even after we assign values to it, I realize that at first, especially for newcomers, it might seem a little confusing. But for now, we will keep it simple. Inquiries are welcome as we investigate more and learn together.

### **9\. ArrayInitialize**

“ArrayInitialize()” is a function in MQL5 that sets the initial values of all elements in a numeric array to a specified preset value. Instead of manually assigning the same value to each element one by one, “ArrayInitialize()” streamlines the process by applying the chosen value to all elements at once. This function is useful for preparing an array with a consistent starting point, especially when dealing with numeric data where uniform initialization is required. Keep in mind that it only sets the initial values and does not affect any reserve elements or future expansions made using “ArrayResize()”.

**Analogy**

Imagine you have a set of magical containers called arrays, and each container has some special spaces inside to hold values. Now, when you want to start with a specific value in all these spaces, you use a special command called “ArrayInitialize()”. This command magically sets the initial value you want in all those spaces at once, saving you the effort of doing it for each space individually.

However, here's the interesting part: if later on, you decide to make these containers larger and add more spaces to them using another magical command (ArrayResize), the new spaces will be there, but they won't have the same magical values as the original ones. You'll have to choose the values to enter each one separately because they will differ slightly. It's similar to expanding your castle's interior space — you don't always decorate the new rooms with the same furnishings as the old ones.

**Syntax:**

```
int ArrayInitialize(
   array[],   // initialized array
   value       // value that will be set
);
```

**Example:**

```
void OnStart()
  {

// Declare a dynamic array
   int myArray[];

// Resize the array to have an initial size (let's use 3 elements)
   ArrayResize(myArray, 3);

// Assign values to all elements before initialization
   myArray[0] = 10;
   myArray[1] = 20;
   myArray[2] = 30;

// Assign values to all elements before initialization
   myArray[0] = 10;
   myArray[1] = 20;
   myArray[2] = 30;

// Initialize the array with a value (let's use 0.0)
   ArrayInitialize(myArray, 0);

// Print the values of all elements after initialization
   Print("Values after initialization:");
   Print("myArray[0] = ", myArray[0]); // outpot wil be 0
   Print("myArray[1] = ", myArray[1]); // outpot wil be 0
   Print("myArray[2] = ", myArray[2]); // outpot wil be 0
// Resize the array to have 5 elements
   ArrayResize(myArray, 5);

// Assign values to the additional elements after resizing
   myArray[3] = 40;
   myArray[4] = 50;

// Print the values of all elements after resizing
   Print("Values after resizing:");
   Print("myArray[3] = ", myArray[3]); // outpot wil be 40
   Print("myArray[4] = ", myArray[4]); // outpot wil be 50

  }
```

**Explanation:**

```
// Declare a dynamic array
   int myArray[];
```

- Here, we declare a dynamic integer array named “myArray”. It doesn't have a predefined size.


```
// Resize the array to have an initial size (let's use 3 elements)
   ArrayResize(myArray, 3);
```

- We resize “myArray” to have an initial size of 3 elements. This means we allocate memory for three integers in the array.


```
// Assign values to all elements before initialization
   myArray[0] = 10;
   myArray[1] = 20;
   myArray[2] = 30;
```

- Before initializing the array, we manually assign values to its elements. In this case, we set “myArray\[0\]” to 10, “myArray\[1\]” to 20, and “myArray\[2\]” to 30.


```
// Initialize the array with a value (let's use 0.0)
   ArrayInitialize(myArray, 0);
```

- Now, we use the “ArrayInitialize()” function to set all elements of “myArray” to the specified value, which is 0 in this case.


```
// Print the values of all elements after initialization
Print("Values after initialization:");
Print("myArray[0] = ", myArray[0]); // Output will be 0
Print("myArray[1] = ", myArray[1]); // Output will be 0
Print("myArray[2] = ", myArray[2]); // Output will be 0
```

- We print the values of all elements in “myArray” after the initialization. As expected, all elements are now set to 0.


```
// Resize the array to have 5 elements
ArrayResize(myArray, 5);
```

- Next, we resize “myArray” to have a total of 5 elements. This means the array can now accommodate two more elements.


```
// Assign values to the additional elements after resizing
myArray[3] = 40;
myArray[4] = 50;
```

- After resizing, we assign values to the additional elements (“myArray\[3\]” and “myArray\[4\]”).


```
// Print the values of all elements after resizing
Print("Values after resizing:");
Print("myArray[3] = ", myArray[3]); // Output will be 40
Print("myArray[4] = ", myArray[4]); // Output will be 50
```

- Finally, we print the values of all elements in “myArray” after resizing, including the newly added elements.


A powerful tool that allows programmers to set the value of each element in an array to a specified value is the "ArrayInitialize()" function in MQL5. This ensures a consistent starting point for every array element and provides clarity and control over the array's initial state. Remember that the function initializes each element to the same specified number. This may seem simple, but it's an important step in setting up arrays for different kinds of applications. In later articles, we will delve deeper into the realm of algorithmic trading, which will highlight the importance of "ArrayInitialize()." Remain focused and enjoy your coding!

### 10. **ArrayFill**

“ArrayFill()” is a function in MQL5 that plays a vital role in simplifying array manipulation tasks. This function allows developers to efficiently fill a range of array elements with a specified value, eliminating the need for manual iteration and assignment. Instead of writing multiple lines of code to individually set each element, “ArrayFill()” provides a concise and effective solution. This capability enhances code readability and reduces the chances of errors, especially when dealing with large arrays or repetitive assignments. The function's ability to quickly populate array elements with a common value streamlines the coding process, making it a valuable tool for handling various scenarios where bulk initialization is required.

**Analogy**

Let's say you have a box with several slots, and you want to use some of those slots for the same toy without having to go through each one individually. Like a magic spell, "ArrayFill()" allows you to select a toy and instruct it to "Fill these slots with this toy." So, you can place every toy at once rather than placing each one individually! It's similar to telling someone who has a bunch of toy cars, "Fill the first five slots with red cars and the next five with blue ones." This time-saving magic trick helps you maintain order in your toy box!

**Syntax:**

```
ArrayFill(
    array[], // array to be filled
    start,   // Starting slot (index) for filling
    count,   // Number of slots to fill
    value    // The value to fill the slots with
);
```

**Parameters**

- **“array\[\]”:** This is your array of shelves.
- **“start”:** This is like specifying the first shelf where you want to start placing items. You provide the index or position.
- **“count”:** It's similar to saying, "I want to place this item on the next X shelves." You determine the number of shelves to fill.
- **“value”:** This is the item you want to place on the shelves. It can be any item – a number, a color, or anything that fits on the shelves.

**Example:**

```
void OnStart()
  {

// Declare an array of shelves
   int roomShelves[];

// Set the size of the array (number of shelves)
   ArrayResize(roomShelves, 10);

// Fill the first 5 shelves with books (value 42)
   ArrayFill(roomShelves, 0, 5, 42);

// Fill the next 5 shelves with toys (value 99)
   ArrayFill(roomShelves, 5, 5, 99);

// Display the contents of the shelves after filling
   Print("Contents of the shelves after filling:");
   Print("Shelf 0: ", roomShelves[0]); // output will be 42
   Print("Shelf 1: ", roomShelves[1]); // output will be 42
   Print("Shelf 2: ", roomShelves[2]); // output will be 42
   Print("Shelf 3: ", roomShelves[3]); // output will be 42
   Print("Shelf 4: ", roomShelves[4]); // output will be 42
   Print("Shelf 5: ", roomShelves[5]); // output will be 99
   Print("Shelf 6: ", roomShelves[6]); // output will be 99
   Print("Shelf 7: ", roomShelves[7]); // output will be 99
   Print("Shelf 8: ", roomShelves[8]); // output will be 99
   Print("Shelf 9: ", roomShelves[9]); // output will be 99

  }
```

**Explanation:**

**“int roomShelves\[\];”:**

- Declares an integer array named “roomShelves” to represent the shelves in a room.

**“ArrayResize(roomShelves, 10);”:**

- Resizes the “roomShelves” array to have 10 elements, representing 10 shelves in the room.

**“ArrayFill(roomShelves, 0, 5, 42);”:**

- Fills the first 5 shelves (indices 0 to 4) with the value 42, representing books on those shelves.

**“ArrayFill(roomShelves, 5, 5, 99);”:**

- Fills the next 5 shelves (indices 5 to 9) with the value 99, representing toys on those shelves.

**“Print("Contents of the shelves after filling:");”:**

- Prints a message indicating that the following lines will display the contents of the shelves.

**“Print("Shelf 0: ", roomShelves\[0\]);” to “Print("Shelf 9: ", roomShelves\[9\]);”:**

- Prints the contents of each shelf, displaying the index of the shelf and its corresponding value

This MQL5 code illustrates how to fill a dynamic array called "roomShelves" using the "ArrayFill()" function. Initially, the array was resized to have ten shelves. The next step involves using "ArrayFill()" to fill the first five shelves with 42 and the next five shelves with 99. The values assigned to each element in the array following the filling process are finally revealed when the contents of each shelf are printed. The code demonstrates how "ArrayFill()" offers a flexible method to set predefined values inside an array structure by effectively initializing designated segments of a dynamic array with predefined values.

_Note: “ArrayFill_() _” and “ArrayInitialize_() _” are both array manipulation functions in MQL5, but they serve distinct purposes. “ArrayFill_() _” is designed for filling a specific range of elements within an array with a given value. It allows for efficient bulk assignment to a subset of the array, making it useful for modifying or initializing portions of the array selectively. In contrast, “ArrayInitialize_() _” is a more general function that uniformly sets the value of all elements in the entire array. It ensures a consistent starting state for the entire array, providing a quick way to initialize all elements to the same value. So, while “ArrayFill_() _” is specialized for targeted assignments, “ArrayInitialize_() _” is a broader tool for uniform initialization across the entire array._

### 11. **ArrayIsDynamic**

To ascertain whether an array is dynamic or static, one useful tool in MQL5 is the "ArrayIsDynamic()" function. A static array's size is fixed at compile time, whereas the size of a dynamic array can be altered during runtime. "ArrayIsDynamic()" determines whether an array is dynamic or static and returns a simple "true" or "false" response depending on the nature of the array. With the use of this data, the program can modify its behavior according to the flexibility of the array, determining whether its size can be changed while the program is running.

**Analogy**

An array is similar to a magical box in the world of programming that can hold several objects. Some boxes now have a special magic that makes it possible for them to resize to fit more items or to take up less space when necessary. These are referred to as dynamic boxes. Conversely, some boxes are set in size and will not budge under any circumstances. These are referred to as static boxes.

Now, to determine whether a box is dynamic or static, you can use the wizard-like "ArrayIsDynamic()" function on it. The results of this spell indicate whether the box is static — its size remains constant—or dynamic—it can change its size. Programmers need to know this stuff because it makes sense when deciding how to manipulate the box in their magical code. If it is static, they must take care not to go over its set size; if it is dynamic, they can make it grow or shrink as needed.

**Syntax:**

```
bool ArrayIsDynamic(array[]   // array to be checked);
```

**Example:**

```
void OnStart()
  {

// Declare a static array
   int staticArray[5];

// Declare a dynamic array
   int dynamicArray[];

// Check if the static array is dynamic
   bool isStaticArrayDynamic = ArrayIsDynamic(staticArray);

   if(isStaticArrayDynamic)
     {
      Print("The staticArray is dynamic.");  // This message won't be printed.
     }
   else
     {
      Print("The staticArray is static, meaning its size is fixed.");
     }

// Check if the dynamic array is dynamic
   bool isDynamicArrayDynamic = ArrayIsDynamic(dynamicArray);

   if(isDynamicArrayDynamic)
     {
      Print("The dynamicArray is dynamic, meaning its size can be changed.");
     }
   else
     {
      Print("The dynamicArray is static.");  // This message won't be printed.
     }

  }
```

**Explanation:**

**“int staticArray\[5\];”:**

- This line declares an array named “staticArray” with a fixed size of 5 elements.

**“int dynamicArray\[\];”:**

- This line declares a dynamic array named “dynamicArray” without specifying a fixed size.

**“bool isStaticArrayDynamic = ArrayIsDynamic(staticArray);”:**

- This line uses the “ArrayIsDynamic()” function to check if “staticArray” is dynamic and assigns the result to “isStaticArrayDynamic”.

**“bool isStaticArrayDynamic = ArrayIsDynamic(staticArray);”:**

- This line uses the “ArrayIsDynamic()” function to check if “staticArray” is dynamic and assigns the result to “isStaticArrayDynamic”.

**Print the result for the static array:**

- The subsequent “if-else” block prints a message indicating whether “staticArray” is dynamic or static based on the result obtained in the previous step.

**“bool isDynamicArrayDynamic = ArrayIsDynamic(dynamicArray);”:**

- This line uses the “ArrayIsDynamic()” function to check if “dynamicArray” is dynamic and assigns the result to “isDynamicArrayDynamic”.

**Print the result for the dynamic array:**

- The subsequent “if-else” block prints a message indicating whether “dynamicArray” is dynamic or static based on the result obtained in the previous step.

The code demonstrates the use of  “ArrayIsDynamic()” to determine whether an array is dynamic or static.

### 12. **ArrayMaximum**

The “ArrayMaximum()” function in MQL5 is a powerful tool designed to identify the index of the maximum element within a numeric array. This function proves particularly useful in scenarios where determining the highest value is crucial for decision-making. By efficiently searching through the array, the function returns the index of the maximum element, taking into account the array serial. In situations where the array represents financial or technical data, finding the maximum value is a fundamental step in extracting meaningful insights or making informed trading decisions.

**Analogy**

Imagine you have a list of numbers, like the scores of different games you played. The “ArrayMaximum()” function is like a little helper that looks through your list and tells you which game you did the best in. So, if you want to know which game had the highest score, this helper checks each score, and when it finds the highest one, it points to that game and says, "This one is your best game!" It's like having a friend quickly find the game where you did the most awesome job, without you having to go through the whole list yourself. In computer programs, this helper is handy for finding the biggest number in a bunch of numbers.

**Syntax:**

```
int ArrayMaximum(
    array[],       // Array for search
    start,          // Index to start checking with
    count = WHOLE_ARRAY    // Number of checked elements (default: search in the entire array)
);
```

**Parameters:**

- **“array\[\]”:** This is the array for which you want to find the maximum value.
- **“start":** This parameter allows you to specify the index in the array from where you want to start searching for the maximum value.
- **“count = WHOLE\_ARRAY”:** It represents the number of elements to consider in the search. The default value “WHOLE\_ARRAY” means that the function will search in the entire array.

Now, when you call “ArrayMaximum(array, start, count)”, the function will find the largest element in the specified range of the array and return its index. If no maximum is found, it returns -1.

**Example:**

```
void OnStart()
  {

// Declare an array with integer values
   int myArray[] = {42, 18, 56, 31, 75, 23};

// Find the maximum value and its index
   int maxIndex = ArrayMaximum(myArray);

// Check if a maximum was found
   if(maxIndex != -1)
     {
      Print("The maximum value in the array is: ", myArray[maxIndex]);
      Print("Index of the maximum value: ", maxIndex);
     }
   else
     {
      Print("No maximum value found in the array.");
     }

  }
```

**Explanation:**

**“int myArray\[\] = {42, 18, 56, 31, 75, 23};”:**

- This line declares an integer array named “myArray” and initializes it with six integer values.

**“int maxIndex = ArrayMaximum(myArray);”:**

- The “ArrayMaximum()” function is called on “myArray” to find the index of the maximum value in the array. The result is stored in the variable “maxIndex”.

**“if (maxIndex != -1) {“:**

- This conditional statement checks whether a maximum value was found. If “maxIndex” is not equal to “-1”, it means a maximum value exists in the array.

**“Print("The maximum value in the array is: ", myArray\[maxIndex\]);”:**

- If a maximum value is found, this line prints the maximum value using the index obtained from “maxIndex”.

**“Print("Index of the maximum value: ", maxIndex);”:**

- This line prints the index of the maximum value.

**“} else { Print("No maximum value found in the array."); }”:**

- When maxIndex is -1, which occurs when no maximum value is found, this block is executed and a message stating that no maximum value was found is printed.

To determine the maximum value and its index in an array, utilize "ArrayMaximum()" as this code illustrates.

### 13. **ArrayMinimum**

A useful function in MQL5 called "ArrayMinimum()" lets you determine the index of the smallest (minimum) element in a numeric array's first dimension. This function works with arrays of different sizes and is very flexible. It ensures accurate results by accounting for the serial order of the array.

"ArrayMinimum()" is primarily used to find the index of the smallest element in the array. It gives the index of the lowest value while taking the array's element order into account. The function returns -1 if there is no minimum value found by the search. When you need to determine the exact location of the smallest element within an array, this feature can come in handy.

**Syntax:**

```
int ArrayMinimum(array[],start,count = WHOLE_ARRAY);
```

**Example:**

```
void OnStart()
  {

// Declare an integer array
   int myArray[] = {10, 5, 8, 3, 12};

// Find the index of the minimum element in the entire array
   int minIndex = ArrayMinimum(myArray, 0, WHOLE_ARRAY);

// Print the result
   Print("Index of the minimum element: ", minIndex);

  }
```

Just as in "ArrayMaximum()" this code will find the index of the minimum element in the entire "myArray" and print the result.

### **Conclusion**

In this article, we discussed the world of array functions in MQL5, uncovering their functionalities and applications. From searching and copying to resizing and handling time series arrays, we explored a variety of functions, including ArrayBsearch, ArrayCopy, ArrayCompare, ArrayResize, ArrayFree, ArraySetAsSeries, ArrayGetAsSeries, ArrayInitialize, ArrayFill, ArrayIsDynamic, ArrayIsSeries, ArrayMaximum, and ArrayMinimum. Each function plays a crucial role in manipulating arrays and enhancing the capabilities of trading algorithms.

As we conclude this article, it's important to note that I intentionally did not cover all array functions to ensure a focused and digestible learning experience. The upcoming article will expand our other array functions like ArrayPrint, ArrayRange, ArrayInsert, ArrayRemove, ArrayReverse, ArraySize, ArraySort, ArraySwap, ArrayToFP16,  ArrayToFP8,  ArrayFromFP16, and ArrayFromFP8. This gradual approach aims to facilitate a smoother learning curve. So, buckle up for the next installment, where we'll continue our exploration of MQL5 array functions!

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Introduction to MQL5 (Part 36): Mastering API and WebRequest Function in MQL5 (X)](https://www.mql5.com/en/articles/20938)
- [Introduction to MQL5 (Part 35): Mastering API and WebRequest Function in MQL5 (IX)](https://www.mql5.com/en/articles/20859)
- [Introduction to MQL5 (Part 34): Mastering API and WebRequest Function in MQL5 (VIII)](https://www.mql5.com/en/articles/20802)
- [Introduction to MQL5 (Part 33): Mastering API and WebRequest Function in MQL5 (VII)](https://www.mql5.com/en/articles/20700)
- [Introduction to MQL5 (Part 32): Mastering API and WebRequest Function in MQL5 (VI)](https://www.mql5.com/en/articles/20591)
- [Introduction to MQL5 (Part 31): Mastering API and WebRequest Function in MQL5 (V)](https://www.mql5.com/en/articles/20546)
- [Introduction to MQL5 (Part 30): Mastering API and WebRequest Function in MQL5 (IV)](https://www.mql5.com/en/articles/20425)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/463992)**
(4)


![Oluwatosin Mary Babalola](https://c.mql5.com/avatar/2024/2/65cfcb6a-f1c4.jpg)

**[Oluwatosin Mary Babalola](https://www.mql5.com/en/users/excel_om)**
\|
15 Mar 2024 at 22:03

**MetaQuotes:**

Check out the new article: [Introduction to MQL5 (Part 5): A Beginner's Guide to Array Functions in MQL5](https://www.mql5.com/en/articles/14306).

Author: [Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913 "13467913")

Thank you for sharing your knowledge on the difference between ArrayGetAsSeries and ArrayIsSeries which I have been trying to figure out for a while now. I like your approach on explaining complex topic in a beginner friendly way backed up with analogy… I’m also a bit confused about the difference between ArrayCopy and ArrayInsert. I’d appreciate if you can include that in your next article.


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
15 Mar 2024 at 22:26

**Oluwatosin Mary Babalola [#](https://www.mql5.com/en/forum/463992#comment_52735156):**

Thank you for sharing your knowledge on the difference between ArrayGetAsSeries and ArrayIsSeries which I have been trying to figure out for a while now. I like your approach on explaining complex topic in a beginner friendly way backed up with analogy… I’m also a bit confused about the difference between ArrayCopy and ArrayInsert. I’d appreciate if you can include that in your next article.

Hello Oluwatosin, your request has been noted


![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
17 Mar 2024 at 21:02

**Oluwatosin Mary Babalola [#](https://www.mql5.com/en/forum/463992#comment_52735156):**

Thank you for sharing your knowledge on the difference between ArrayGetAsSeries and ArrayIsSeries which I have been trying to figure out for a while now. I like your approach on explaining complex topic in a beginner friendly way backed up with analogy… I’m also a bit confused about the difference between ArrayCopy and ArrayInsert. I’d appreciate if you can include that in your next article.

Do you know about the MQL5 programming book? Specifically, it covers [array "seriesness" (direction) functions](https://www.mql5.com/en/book/common/arrays/arrays_as_series), as well as [copies and inserts](https://www.mql5.com/en/book/common/arrays/arrays_edit) \- here is an excerpt:

Unlike theArrayInsertfunction, theArrayCopyfunction does not shift the existing elements of the receiving array but writes new elements to the specified positions over the old ones.

Both sections contain example programs.

![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
26 Mar 2024 at 12:41

**Oluwatosin Mary Babalola [#](https://www.mql5.com/en/forum/463992#comment_52735156):**

Thank you for sharing your knowledge on the difference between ArrayGetAsSeries and ArrayIsSeries which I have been trying to figure out for a while now. I like your approach on explaining complex topic in a beginner friendly way backed up with analogy… I’m also a bit confused about the difference between ArrayCopy and ArrayInsert. I’d appreciate if you can include that in your next article.

**Difference between ArrayInsert and ArrayCopy:**

The main difference between "ArrayInsert()" and "ArrayCopy()" is how they handle elements that already exist. "ArrayCopy()" may modify the original array by substituting elements from another array for those at a given position. On the other hand, "ArrayInsert()" preserves the array's structure and sequence by moving the current elements to make room for the new ones. Essentially, "ArrayInsert()" provides a versatile method for manipulating arrays in MQL5, akin to adding a new element to a sequence without causing any other pieces to move. Comprehending this distinction enables you to precisely manipulate array operations in your programming pursuits.

Note that for static arrays, if the number of elements to be inserted equals or exceeds the array size, "ArrayInsert()" will not add elements from the source array to the destination array. Under such circumstances, inserting can only take place if it starts at index 0 of the destination array. In these cases, the destination array is effectively completely replaced by the source array.

**Analogy**

Imagine you have two sets of building blocks (arrays), each with its unique arrangement. Now, let's say you want to combine these sets without messing up the existing structures. "ArrayInsert()" is like a magic tool that lets you smoothly insert new blocks from one set into a specific spot in the other set, expanding the overall collection.

Now, comparing "ArrayInsert()" with "ArrayCopy()": When you use "ArrayCopy()," it's a bit like rearranging the original set by replacing some blocks with new ones from another set. On the flip side, "ArrayInsert()" is more delicate. It ensures the existing order stays intact by shifting blocks around to make room for the newcomers. It's like having a meticulous assistant who knows exactly where to put each block, maintaining the set's original design.

For static sets (arrays), there's an important rule. If the number of new blocks is too much for the set, "ArrayInsert()" won't force them in. However, starting the insertion process from the very beginning of the set (index 0) can effectively replace the entire set with the new blocks.

![Advanced Variables and Data Types in MQL5](https://c.mql5.com/2/73/Advanced_Variables_and_Data_Types_in_MQL5___LOGO.png)[Advanced Variables and Data Types in MQL5](https://www.mql5.com/en/articles/14186)

Variables and data types are very important topics not only in MQL5 programming but also in any programming language. MQL5 variables and data types can be categorized as simple and advanced ones. In this article, we will identify and learn about advanced ones because we already mentioned simple ones in a previous article.

![Quantization in machine learning (Part 2): Data preprocessing, table selection, training CatBoost models](https://c.mql5.com/2/59/Quantization_in_Machine_Learning_Logo_2___Logo.png)[Quantization in machine learning (Part 2): Data preprocessing, table selection, training CatBoost models](https://www.mql5.com/en/articles/13648)

The article considers the practical application of quantization in the construction of tree models. The methods for selecting quantum tables and data preprocessing are considered. No complex mathematical equations are used.

![Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://c.mql5.com/2/60/Neural_networks_are_easy_wPart_636_Logo.png)[Neural networks made easy (Part 63): Unsupervised Pretraining for Decision Transformer (PDT)](https://www.mql5.com/en/articles/13712)

We continue to discuss the family of Decision Transformer methods. From previous article, we have already noticed that training the transformer underlying the architecture of these methods is a rather complex task and requires a large labeled dataset for training. In this article we will look at an algorithm for using unlabeled trajectories for preliminary model training.

![Developing a Replay System (Part 32): Order System (I)](https://c.mql5.com/2/59/sistema_de_Replay_32_logo_.png)[Developing a Replay System (Part 32): Order System (I)](https://www.mql5.com/en/articles/11393)

Of all the things that we have developed so far, this system, as you will probably notice and eventually agree, is the most complex. Now we need to do something very simple: make our system simulate the operation of a trading server. This need to accurately implement the way the trading server operates seems like a no-brainer. At least in words. But we need to do this so that the everything is seamless and transparent for the user of the replay/simulation system.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=fihquofvempnrcwgnzysuwjekdoiytkv&ssn=1769180536665050748&ssn_dr=0&ssn_sr=0&fv_date=1769180536&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F14306&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Introduction%20to%20MQL5%20(Part%205)%3A%20A%20Beginner%27s%20Guide%20to%20Array%20Functions%20in%20MQL5%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176918053641141167&fz_uniq=5068962747170881399&sv=2552)

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