---
title: Introduction to MQL5 (Part 6): A Beginner's Guide to Array Functions in MQL5 (II)
url: https://www.mql5.com/en/articles/14407
categories: Trading, Expert Advisors, Machine Learning
relevance_score: 3
scraped_at: 2026-01-23T18:01:56.816424
---

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/14407&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068952220206038882)

MetaTrader 5 / Trading


### Introduction

Welcome to Part Six of our MQL5 journey! A fascinating continuation of our series awaits you as we look into the specifics of MQL5 programming, equipping you with the knowledge and skills needed to successfully navigate the dynamic world of automated trading. This chapter will take us further into the topic of array functions. We established the foundation in Part 5 that came before by introducing some array functions.

Now, we'll explore the remaining array functions in Part 6, which will guarantee that you have a thorough understanding of these useful tools. Our objective is still to cover the basic ideas required for automating trading strategies, regardless of your experience level as a developer or level of familiarity with algorithmic trading. Our goal in delving into the nuances of these functions is to promote a comprehensive comprehension so that each reader can competently traverse the ever-changing terrain of MQL5 programming.

In this article, we will cover the following array functions:

- ArrayPrint

- ArrayInsert

- ArraySize

- ArrayRange

- ArrarRemove

- ArraySwap

- ArrayReverse

- ArraySort

### 1\. ArrayPrint

In MQL5, you can print the elements of an array using the predefined function "ArrayPrint()". This function is frequently used for debugging, since it offers a rapid and practical way to view the values kept in an array while an algorithm or script runs. To assist traders and developers in tracking and validating the data at various points in their code, the function outputs the array elements to the console or journal.

**Analogy**

Assume you have a dedicated bookshelf where you store your books. You may occasionally forget which books are on which shelf. Now consider "ArrayPrint()" as the secret phrase to use to view every book title on your shelf without physically visiting each one. Say "ArrayPrint()" to see a tidy list of all the book titles on your unique shelf whenever you're inquiring about what books you own. It's similar to quickly scanning your bookshelf to ensure all your favorite titles are there!

However, this tool is not limited to standard printing; consider "ArrayPrint()" as the magic command you give your intelligent bookshelf when you want to be able to see the titles as well as organize them specifically. You can use this command to specify how much information you want for each book, such as the author and publication date, or if you would rather only see the titles. Even the order in which the titles appear is customizable. "ArrayPrint()" is capable of much more, which we'll discuss later. It's not limited to just displaying the titles of your books. Await the magic with anticipation!

**Syntax:**

**```**
**ArrayPrint(array[], digit , Separator, Start, Count, Flags);**
**```**

****Parameters:****

- **array\[\]:** This is the array to be printed. It can be an array of different data types or an array from a simple structure.

- **Digits:** The number of decimal places to display for each number in the array is set by this parameter.

- **Separator:** This parameter specifies the space that should be between each element of the array when printed.

- **Start:** It specifies the index of the element the printing should start from.

- **Count:** This specifies the number of elements to print.

- **Flags:** it is used to modify the output. This is optional because it is enabled by default. ARRAYPRINT\_HEADER (this flag prints headers for the structure array), ARRAYPRINT\_INDEX (it prints the index on the left side.), and ARRAYPRINT\_LIMIT (it prints only the first 100 and the last 100 array elements). ARRAYPRINT\_ALIGN (this flag enables the alignment of the printed values) and ARRAYPRINT\_DATE (It prints the date in the day, month, and year).


**Example:**

```
void OnStart()
  {

// Define an array of doubles
   double ThisArray[] = { 1.46647, 2.76628, 3.83367, 4.36636, 5.9393};

// Print the entire array using ArrayPrint with 2 decimal places and a single spacing between array elements
   ArrayPrint(Array,2," ",0,WHOLE_ARRAY);

  }
```

**Explanation:**

In this code snippet, we work with an array of double values with a given name. The array is defined with five elements, each containing a decimal number. The subsequent line uses the “ArrayPrint()” function to display the entire content of the array.

Let’s break down the parameters used in the "ArrayPrint()" function:

- **"** **ThisArray":** This is the array we want to print.

- **"2":** Specifies the number of decimal places to display for each element in the array.

- **" ":** Sets a single space as the separator between array elements.

- **"0":** specify that the printing starts from the beginning of the array.

- **“WHOLE\_ARRAY”:** Specifies that the entire array should be printed.


**Output:**

![Figure 1. Code Output in MetaTrader5](https://c.mql5.com/2/72/Figure_1._ArrayPrint.png)

That straightforwardly illustrates the double-valued "ArrayPrint()" function in action. Let's now take a closer look at some more examples that use structures. As we explore the possibilities of using "ArrayPrint()" to organize and display structured data, get ready for a little more intricacy.

**Example:**

```
void OnStart()
  {

// Define a structure for storing information about students
   struct StudentsDetails
     {
      string         name;
      int            age;
      string         address;
      datetime       time; // Add a time field
     };

// Create an array of Students structures
   StudentsDetails Students[3];

// Fill in details for each person
   Students[0].name = "Abioye";
   Students[0].age = 25;
   Students[0].address = "123 MQL5 St";
   Students[0].time = TimeCurrent();

   Students[1].name = "Israel";
   Students[1].age = 26;
   Students[1].address = "456 MQL4 St";
   Students[1].time = TimeCurrent();

   Students[2].name = "Pelumi";
   Students[2].age = 27;
   Students[2].address = "789 MetaQuotes St";
   Students[2].time = TimeCurrent();

// Print the details of each person using ArrayPrint
   ArrayPrint(Students, 0, " | ", 0, WHOLE_ARRAY);

  }
```

**Explanation:**

```
struct StudentsDetails
{
   string         name;
   int            age;
   string         address;
   datetime       time; // Add a time field
};
```

- To store student data, a structure called “StudentsDetails” is defined.

- The name, age, address, and the time and date of the current day are all added as members of the structure.


```
StudentsDetails Students[3];
```

- An array named Students of type “StudentsDetails” is created with a size of 3, allowing storage for three students.


```
Students[0].name = "Abioye";
Students[0].age = 25;
Students[0].address = "123 MQL5 St";
Students[0].time = TimeCurrent();

Students[1].name = "Israel";
Students[1].age = 26;
Students[1].address = "456 MQL4 St";
Students[1].time = TimeCurrent();

Students[2].name = "Pelumi";
Students[2].age = 27;
Students[2].address = "789 MetaQuotes St";
Students[2].time = TimeCurrent();
```

- Each student's details are filled in. For instance, the name, age, address, and time fields are assigned values, and “Students\[0\]” represents the first student.


```
ArrayPrint(Students, 0, " | ", 0, WHOLE_ARRAY);
```

- To show all of the student's information in the array, the “ArrayPrint()” function is used. The array is printed, with the field separator set to " \| ".


**Output:**

![Figure 2. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_2.png)

We utilize the “ArrayPrint()” function to display the student information after entering all the necessary details for each student. The image above shows the default output, which shows how the details are displayed without any extra formatting flags. Using the designated separator "\|," each student's name, age, address, and current time and date are presented cleanly. This is the first representation; we'll look at how to add more formatting options to further customize the output.

It's crucial to know that adding a specific formatting flag to an “ArrayPrint()” function tells the computer to apply that formatting flag and ignore the others. To illustrate how the “ARRAYPRINT\_HEADER” flag affects the result, we'll include it in the example.

**Example:**

```
ArrayPrint(Students, 0, " | ", 0, WHOLE_ARRAY,ARRAYPRINT_HEADER);
```

**Output:**

![Figure 3. Code Output in MetaTrader5](https://c.mql5.com/2/72/Figure_3.png)

To make each field easier to identify, the flag instructs the function to include headers (\[name\] \[age\] \[address\] \[time\]) for the structure array. “ARRAYPRINT\_INDEX” is among the other flags that are purposefully left out in this instance to highlight how each flag functions on its own.

In the comparison images, observe that the second output differs from the first, as we've introduced the “ARRAYPRINT\_HEADER” flag. This flag instructs the “ArrayPrint” function to include headers for each field, providing clear labels for the information displayed. Notably, the indexing information is absent in the second output. This emphasizes the point that each formatting flag operates independently, and including a specific flag modifies the output accordingly. To demonstrate the versatility of adjusting the output to your preferences, we'll also experiment with various flag combinations.

It is important to note that when working with time data, the “ArrayPrint” function provides even more versatility. Flags like “ARRAYPRINT\_MINUTES” and “ARRAYPRINT\_SECONDS” can be used to adjust the time format. These flags give you the ability to adjust the level of detail in the time information that is shown, giving you a customized display according to your tastes. If you opt for the “ARRAYPRINT\_MINUTES” flag, the function will output only the current hour and minutes, omitting the date and seconds. On the other hand, using the “ARRAYPRINT\_SECONDS” flag refines the output further, displaying only the hour, minutes, and seconds. These flags provide a granular level of control over the time representation, ensuring that your output precisely matches your requirements without including unnecessary details.

**Example:**

```
ArrayPrint(Students, 0, " | ", 0, WHOLE_ARRAY,ARRAYPRINT_MINUTES);
```

**Output:**

![Figure 4. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_4.png)

These flags are not mutually exclusive. You can combine multiple flags to tailor the output even more precisely. For instance, if you apply both the ARRAYPRINT\_HEADER and ARRAYPRINT\_MINUTES flags together, the function will include column headers and present the time in a format that shows only the current hour and minutes.

**Example:**

```
ArrayPrint(Students,0," | ",0,WHOLE_ARRAY,ARRAYPRINT_HEADER | ARRAYPRINT_MINUTES);
```

**Output:**

![Figure 5. Code Output in MetaTrader5](https://c.mql5.com/2/72/Figure_5.png)

This showcases how the flags work seamlessly together to provide a customized and informative output.

### **2\. ArrayInsert**

A useful method for inserting elements from one array into another is to use the "ArrayInsert()" function. Arranging elements from the source array at a designated location, enables you to increase the size of the destination array. Imagine it as integrating a new piece into an existing puzzle without causing any disruptions to the puzzle's overall design.

**Difference between ArrayInsert and ArrayCopy:**

The main difference between "ArrayInsert()" and "ArrayCopy()" is how they handle elements that already exist. "ArrayCopy()" may modify the original array by substituting elements from another array for those at a given position. On the other hand, "ArrayInsert()" preserves the array's structure and sequence by moving the current elements to make room for the new ones. Essentially, "ArrayInsert()" provides a versatile method for manipulating arrays in MQL5, akin to adding a new element to a sequence without causing any other pieces to move. Comprehending this distinction enables you to precisely manipulate array operations in your programming pursuits.

Note that for static arrays, if the number of elements to be inserted equals or exceeds the array size, "ArrayInsert()" will not add elements from the source array to the destination array. Under such circumstances, inserting can only take place if it starts at index 0 of the destination array. In these cases, the destination array is effectively completely replaced by the source array.

**Analogy**

Imagine you have two sets of building blocks (arrays), each with its own unique arrangement. Now, let's say you want to combine these sets without messing up the existing structures. "ArrayInsert()" is like a magic tool that lets you smoothly insert new blocks from one set into a specific spot in the other set, expanding the overall collection.

Now, comparing "ArrayInsert()" with "ArrayCopy()": When you use "ArrayCopy()," it's a bit like rearranging the original set by replacing some blocks with new ones from another set. On the flip side, "ArrayInsert()" is more delicate. It ensures the existing order stays intact by shifting blocks around to make room for the newcomers. It's like having a meticulous assistant who knows exactly where to put each block, maintaining the set's original design.

For static sets (arrays), there's an important rule. If the number of new blocks is too much for the set to handle, "ArrayInsert()" won't force them in. However, if you start the insertion process from the very beginning of the set (index 0), it can effectively replace the entire set with the new blocks. Understanding these concepts helps you become a master builder in the world of MQL5 programming!

**Syntax:**

```
ArrayInsert(DestinationArray[],SourceArray[],DestinationIndexStart,SourceIndexStart,count);
```

**Parameters:**

- **DestinationArray\[\]:** The array that will receive elements from the source array and be inserted into it.

- **SourceArray\[\]:** The array to be inserted into the destination array is called the source array.

- **DestinationIndexStart:** The index where insertion starts in the destination array.

- **SourceIndexStart:** The index within the source array that will be used to copy elements for insertion.

- **Count:** The number of elements that should be inserted from the source array into the destination array.


**Example:**

**```**
**void OnStart()**
**{**

**// Declare two dynamic arrays**
**int SourceArray[];**
**int DestinationArray[];**

**// Resizing the dynamic arrays to have 5 elements each**
**ArrayResize(SourceArray, 5);**
**ArrayResize(DestinationArray, 5);**

**// Assigning values to dynamic array elements**
**SourceArray[0] = 1;**
**SourceArray[1] = 3;**
**SourceArray[2] = 5;**
**SourceArray[3] = 7;**
**SourceArray[4] = 9;**

**// Assigning different values to DestinationArray**
**DestinationArray[0] = 15;**
**DestinationArray[1] = 20;**
**DestinationArray[2] = 25;**
**DestinationArray[3] = 30;**
**DestinationArray[4] = 35;**

**// Print the elements of SourceArray before ArrayInsert/ArrayCopy**
**Print("Elements of SourceArray before ArrayInsert/ArrayCopy: ");**
**ArrayPrint(SourceArray, 2, " ", 0, WHOLE_ARRAY);**

**// Print the elements of DestinationArray before ArrayInsert/ArrayCopy**
**Print("Elements of DestinationArray before ArrayInsert/ArrayCopy: ");**
**ArrayPrint(DestinationArray, 2, " ", 0, WHOLE_ARRAY);**

**// Using ArrayInsert to insert SourceArray into DestinationArray at index 2**
**ArrayInsert(DestinationArray, SourceArray, 2, 0, WHOLE_ARRAY);**

**// Print the modified DestinationArray after ArrayInsert**
**Print("Elements of DestinationArray after using ArrayInsert: ");**
**ArrayPrint(DestinationArray, 2, " ", 0, WHOLE_ARRAY);**

**// Reset DestinationArray to demonstrate ArrayCopy**
**ArrayFree(DestinationArray);**
**ArrayResize(DestinationArray, 5);**

**DestinationArray[0] = 15;**
**DestinationArray[1] = 20;**
**DestinationArray[2] = 25;**
**DestinationArray[3] = 30;**
**DestinationArray[4] = 35;**

**// Using ArrayCopy to copy elements from SourceArray to DestinationArray**
**ArrayCopy(DestinationArray, SourceArray, 2, 0, WHOLE_ARRAY);**

**// Print the modified DestinationArray after ArrayCopy**
**Print("Elements of DestinationArray after using ArrayCopy: ");**
**ArrayPrint(DestinationArray, 2, " ", 0, WHOLE_ARRAY);**

**}**
**```**

**Explanation:**

```
int SourceArray[];
int DestinationArray[];
```

- “SourceArray” and "DestinationArray," two dynamic arrays, are declared here. Integer values will be kept in these arrays.


```
ArrayResize(SourceArray, 5);
ArrayResize(DestinationArray, 5);
```

- The dynamic arrays are resized to contain five elements apiece by these lines. For this, the “ArrayResize()” function is employed.


```
SourceArray[0] = 1;
SourceArray[1] = 3;
SourceArray[2] = 5;
SourceArray[3] = 7;
SourceArray[4] = 9;
```

- The “SourceArray” elements are given values.


```
DestinationArray[0] = 15;
DestinationArray[1] = 20;
DestinationArray[2] = 25;
DestinationArray[3] = 30;
DestinationArray[4] = 35;
```

- The “DestinationArray” elements are given values.


```
Print("Elements of SourceArray before ArrayInsert/ArrayCopy: ");
ArrayPrint(SourceArray, 2, " ", 0, WHOLE_ARRAY);
```

- This line uses the “ArrayPrint()” function to print the elements of “SourceArray” after printing a message to the console. A space is used as a separator, and two decimal places are displayed in the format.


```
Print("Elements of DestinationArray before ArrayInsert/ArrayCopy: ");
ArrayPrint(DestinationArray, 2, " ", 0, WHOLE_ARRAY);
```

- Similar to the previous line, this prints a message and then the elements of “DestinationArray”.


```
ArrayInsert(DestinationArray, SourceArray, 2, 0, WHOLE_ARRAY);
```

- This line inserts the elements of “SourceArray” into “DestinationArray” beginning at index 2 using the “ArrayInsert()” function.


```
Print("Elements of DestinationArray after using ArrayInsert: ");
ArrayPrint(DestinationArray, 2, " ", 0, WHOLE_ARRAY);
```

- After the “ArrayInsert()” operation, this prints a message followed by the modified elements of “DestinationArray”.


```
ArrayFree(DestinationArray);
ArrayResize(DestinationArray, 5);
```

- These lines resize “DestinationArray” to contain five elements once more after freeing up memory.


```
DestinationArray[0] = 15;
DestinationArray[1] = 20;
DestinationArray[2] = 25;
DestinationArray[3] = 30;
DestinationArray[4] = 35;
```

- The “DestinationArray” elements are given values again.


```
ArrayCopy(DestinationArray, SourceArray, 2, 0, WHOLE_ARRAY);
```

- This line uses the ArrayCopy function to copy elements from SourceArray into DestinationArray starting from index 2.


```
Print("Elements of DestinationArray after using ArrayCopy: ");
ArrayPrint(DestinationArray, 2, " ", 0, WHOLE_ARRAY);
```

- This prints a message and then the modified elements of “DestinationArray” after the “ArrayCopy()” operation.

**Output:**

![Figure 6. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_6.png)

The objective of this code sample is to demonstrate the differences between MQL5's “ArrayInsert()” and “ArrayCopy()” functions. While manipulating array elements is a common use for both functions, their functions are distinct. Two dynamic arrays—“SourceArray” and “DestinationArray”—are used in this example. Before performing any operations, the script displays the elements contained in these arrays. Then, to insert elements from —“SourceArray” into designated locations within “DestinationArray()”, ArrayInsert is utilized. After that, the arrays are reset, and elements from —“SourceArray” are copied into “DestinationArray()” using “ArrayCopy()”. Their actions are where they diverge most: When inserting elements at a specific position into the destination array, “ArrayInsert()” moves the existing elements to make room for the new elements. It is useful for putting elements at the desired index. With “ArrayCopy()”, elements from the source array are copied and substituted for any existing elements in the destination array. Efficient in copying elements between arrays without affecting the values that are already set.

### **3\. ArraySize**

The MQL5 function “ArraySize()” is intended to ascertain how many elements are contained in a one-dimensional array. Returning an integer that represents the total count of elements within the specified array, makes the process of determining the size of an array simpler.

**Analogy**

Assume that you have a bookshelf full of different books, each of which is a representation of an element in an array. Like a librarian, the “ArraySize()” function tells you the precise number of books on your shelf. Similarly, you can manage and arrange your data more effectively by using “ArraySize(),” which, when applied to an array, tells you the total number of elements it contains. Programmers can use it as a useful tool to comprehend the size of their arrays and make sure they have the appropriate number of "books" for their coding endeavors.

**Syntax:**

```
ArraySize( array[]);
```

**Parameter:**

- **array\[\]:** The array for which you wish to find the size is indicated by this parameter.


**Example:**

```
void OnStart()
  {

// Declare an array
   int array[5];

// Get the size of the array using ArraySize
   int arraySize = ArraySize(array);

// Print the array size
   Print("The size of array is: ", arraySize); // Output will be 5

  }
```

**Explanation:**

**“int array\[5\];”**

- An integer array with the name “array” and a size of “5” is declared in this line.


**“int arraySize = ArraySize(array);”**

- This line creates a new integer variable called “arraySize” and sets its value to the result of “ArraySize” (array) using the assignment operator “=”. Since the size of an array is always an integer, the “int” type is utilized. The function provided by MQL5 to determine the size of an array is called “ArraySize” (uppercase), and the variable we've declared to store the result is called “arraySize” (lowercase). The case sensitivity of programming languages must be noted. Uppercase ArraySize denotes the built-in array, while lowercase arraySize denotes our particular variable.


**“Print("The size of array is: ", arraySize);”:**

- This line uses the “Print” function to print a message to the console. It shows the array's size, which is derived from the “arraySize” variable.


You must take your time and learn the intricacies of every function we come across as we explore deeper into the amazing world of MQL5 programming. It's similar to learning how to use tools in a craftsman's workshop to comprehend functions like ArraySize, ArrayInsert, and ArrayPrint; each has a specific use. Take your time to learn and understand the subtleties; don't rush the process. The more complex ideas we'll cover in later articles will have these functions as their foundation.

### **4\. ArrayRange**

The “ArrayRange()” function in MQL5 programming is essential for figuring out the number of elements in a specified dimension of a multidimensional array. It is a useful tool for developers working with complex arrays, giving them accurate information about how many elements are in a given tier or dimension in a multidimensional array. Without having to deal with the complexities of figuring out how many elements there are overall across all dimensions, this function's concentration on a specific dimension offers in-depth insights.

**Difference between ArrayRange and ArraySize**

Let's now distinguish between “ArraySize()” and “ArrayRange()”. Although they both provide insights into array dimensions, the functions’ respective scopes are different. A one-dimensional array's entire element count can be found using “ArraySize()”.

One-dimensional and multidimensional arrays differ from each other in terms of how they arrange data and are structured. A simple list with elements arranged in a single line is analogous to a one-dimensional array. Elements can be accessed by referring to where they are in this linear structure.

Nevertheless, multidimensional arrays add more levels of structure. With their components arranged in rows and columns, they resemble matrices or tables. The row and column indices must be specified to access elements in a multidimensional array, offering a more structured method of data organization and retrieval. One-dimensional arrays are essentially simple, linear sequences, whereas multidimensional arrays add complexity by arranging elements more like a grid.

**Analogy**

Imagine you have a vast bookshelf representing a multidimensional array, where each shelf has a different dimension. The “ArrayRange()” function in MQL5 is like a magical magnifying glass that lets you focus on a specific shelf, revealing the exact number of books (elements) on that shelf. This tool is incredibly handy when dealing with a complex library of information.

Let's now contrast “ArraySize()” and "ArrayRange()." Should the books be organized linearly, akin to a one-dimensional array, then “ArraySize()” denotes the total number of books on the entire bookshelf. Alternatively, you can use “ArrayRange()” to magnify a particular section of the bookcase to get a precise count of the number of books that are there.

**Syntax:**

```
ArrayRange(array[], dimensionIndex);
```

**Parameters:**

- **array\[\]:** The array whose range you wish to verify.
- **dimensionIndex:** The dimension index, starting at 0, for which the range needs to be ascertained.

**Example:**

```
void OnStart()
  {

// Declare a three-dimensional array
   double my3DArray[][2][4];

// Get the range of the first dimension (index 0)
   int dimension1Index = ArrayRange(my3DArray, 0);

// Get the range of the second dimension (index 1)
   int dimension2Index = ArrayRange(my3DArray, 1);

// Get the range of the third dimension (index 2)
   int dimension3Index = ArrayRange(my3DArray, 2);

   Print("Number of elements in dimension 1: ", dimension1Index);
   Print("Number of elements in dimension 2: ", dimension2Index);
   Print("Number of elements in dimension 3: ", dimension3Index);

  }
```

**Explanation:**

```
double my3DArray[][2][4];
```

- A three-dimensional array called "my3DArray" is declared in this line.


```
int dimension1Index = ArrayRange(my3DArray, 0);
```

- In this case, the range (number of elements) in the my3DArray's first dimension (index 0) is ascertained using the “ArrayRange()” function. The variable “dimension1Index” holds the result.


```
int dimension2Index = ArrayRange(my3DArray, 1);
```

- Similarly, this line obtains and stores in the variable “dimension2Index” the range of the second dimension (index 1) of the “my3DArray”.


```
int dimension3Index = ArrayRange(my3DArray, 2);
```

- This line assigns the value of the variable “dimension3Index” to the range of the third dimension (index 2) of the “my3DArray”.


```
Print("Number of elements in dimension 1: ", dimension1Index);
Print("Number of elements in dimension 2: ", dimension2Index);
Print("Number of elements in dimension 3: ", dimension3Index);
```

- Lastly, we show the results and the number of elements in each dimension using the Print function. The first, second, and third dimensions of the three-dimensional array's sizes are included in the printed information.


**Output:**

**![Figure 7. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_7.png)**

### **5\. ArrayRemove**

“ArrayRemove()” function is an effective tool that lets programmers remove particular elements from an array. The array's size and structure are automatically adjusted to accommodate the deletion, ensuring a seamless removal process. Developers have flexibility when manipulating arrays because they can specify the starting index and the number of elements they wish to remove. When working with arrays that must be dynamically modified in response to shifting program conditions, this function is especially helpful.

But when it comes to "ArrayRemove()," its behavior varies depending on whether static or dynamic arrays are used. For dynamic arrays, the function ensures a streamlined removal process by effectively removing the specified element or elements and smoothly adjusting the array size. On the other hand, “ArrayRemove()” eliminates the specified elements and keeps the original array size when working with static arrays. But to overcome the fixed nature of static arrays, the function duplicates the elements that come after the end of the array to fill the empty spaces. This method allows for the removal of elements while keeping the size fixed, giving “ArrayRemove()” a more nuanced understanding of various array scenarios. As we work through this section, more examples and insights will help us gain a deeper understanding of “ArrayRemove()” and how it works in different array scenarios.

**Analogy**

Consider our array as a bookshelf, with individual books representing bits of information. Now, MQL5 gives us a unique tool similar to a bookshelf organizer called “ArrayRemove()”. We can remove particular books from our bookshelf and neatly arrange the remaining books to fill in any gaps by using this organizer.

Imagine you have a dynamic bookshelf where you can easily add or remove books. In this case, the organizer smoothly adjusts the shelf after removing books with no issue. However, if your bookshelf is more like a fixed-size display, where you can't change its size (a static array), When working with static arrays, “ArrayRemove()” cannot change the size of the shelf, so it cleverly duplicates the book at the end of the array to fill the empty slot. It's like making a copy of the last books on the shelf and placing it in the gap left by the removed book. This way, the fixed-size bookshelf maintains its completeness, and no space is wasted.

So, if you remove a book from the middle of your bookshelf, “ArrayRemove()” ensures that the end of the shelf is copied to fill the gap, preserving the array's structure. This can be particularly handy when you have a specific number of slots (elements) to maintain, providing a method to tidy up your bookshelf without changing its size.

**Syntax:**

```
ArrayRemove(array[],start_index,count);
```

**Parameters:**

- **array\[\]:** This is the array that will have its elements eliminated. The storage area, or bookshelf, is where you want to make changes.

- **start\_index:** It indicates the starting point of the removal within the array. To eliminate books beginning on the third shelf, for example, you would set the index to 3.

- **count:** The number of elements to be removed from the array. If you want to remove three books, you'd set the count to 3.


**Example:**

**```**
**void OnStart()**
**{**

**// Declare fixed-size array**
**int fixedSizeArray[5] = {11, 13, 17, 21, 42};**

**// Declare dynamic array**
**int dynamicArray[];**
**ArrayResize(dynamicArray, 5);**
**dynamicArray[0] = 11;**
**dynamicArray[1] = 13;**
**dynamicArray[2] = 17;**
**dynamicArray[3] = 21;**
**dynamicArray[4] = 42;**

**// Print initial arrays**
**Print("Initial fixedSizeArray: ");**
**ArrayPrint(fixedSizeArray, 0, " ", 0, WHOLE_ARRAY);**

**Print("Initial dynamicArray: ");**
**ArrayPrint(dynamicArray, 0, " ", 0, WHOLE_ARRAY);**

**// Remove two elements at index 2 from both arrays**
**ArrayRemove(fixedSizeArray, 2, 2);**
**ArrayRemove(dynamicArray, 2, 2);**

**// Print arrays after removal**
**Print("After removing 3 elements at index 2 - fixedSizeArray: ");**
**ArrayPrint(fixedSizeArray, 0, " ", 0, WHOLE_ARRAY);**

**Print("After removing 3 elements at index 2 - dynamicArray: ");**
**ArrayPrint(dynamicArray, 0, " ", 0, WHOLE_ARRAY);**

**}**
**```**

**Explanation:**

```
// Declare fixed-size array
   int fixedSizeArray[5] = {11, 13, 17, 21, 42};

// Declare dynamic array
   int dynamicArray[];
   ArrayResize(dynamicArray, 5);
   dynamicArray[0] = 11;
   dynamicArray[1] = 13;
   dynamicArray[2] = 17;
   dynamicArray[3] = 21;
   dynamicArray[4] = 42;
```

- Declares that an integer static array with a fixed size of 5 is called "fixedSizeArray." uses the values 11, 13, 17, 21, and 42 to initialize the array.

- Declares an integer dynamic array named “dynamicArray” without specifying an initial size. Resizes “dynamicArray” to have a size of 5 using “ArrayResize”.


```
Print("Initial fixedSizeArray: ");
ArrayPrint(fixedSizeArray, 0, " ", 0, WHOLE_ARRAY);

Print("Initial dynamicArray: ");
ArrayPrint(dynamicArray, 0, " ", 0, WHOLE_ARRAY);
```

- Print the initial elements of “fixedSizeArray” and “dynamicArray”using “ArrayPrint”.


```
ArrayRemove(fixedSizeArray, 2, 2);
ArrayRemove(dynamicArray, 2, 2);
```

- Using "ArrayRemove," two elements are removed from "fixedSizeArray" and "dynamicArray," beginning at index 2.


```
Print("After removing 3 elements at index 2 - fixedSizeArray: ");
ArrayPrint(fixedSizeArray, 0, " ", 0, WHOLE_ARRAY);

Print("After removing 3 elements at index 2 - dynamicArray: ");
ArrayPrint(dynamicArray, 0, " ", 0, WHOLE_ARRAY);
```

- After the removal process, use "ArrayPrint" to print the elements of "fixedSizeArray" and "dynamicArray."


**Output:**

**![Figure 8. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_8.png)**

The output of the given code is shown in the image above, which shows how the “ArrayRemove()” function behaves with both static and dynamic arrays. When it comes to the dynamic array, the procedure is very simple: it just involves deleting the elements that are specified at the designated index. To fill the empty spaces created by the removal, the function for the static array duplicates elements that appear after the end of the array. This subtle behavior demonstrates the way that “ArrayRemove()” adjusts to various array types.

The concepts of “ArrayRemove()” will become clearer as we work through these articles and get into real-world examples. Please don't hesitate to ask more questions; together, and we'll continue to explore and understand these concepts together.

### **6\. ArraySwap**

The purpose of the “ArraySwap()” function in MQL5 programming is to switch the entire contents of two dynamic arrays. All elements between two arrays can be exchanged more easily with the help of this function. It offers a simple way to switch the whole dataset between arrays, which expedites the MQL5 process of rearranging array contents.

**Analogy**

Assume you have two bookcases filled with books. The "ArraySwap()" function allows you to swap out every book on one shelf for every other book, much like a sorcerer's spell. To make all the books from “Shelf A” move to “Shelf B” and all the books from “Shelf B” move to “Shelf A,” you can use the “ArraySwap()” spell if you have “Shelf A” stocked with some books and “Shelf B” stocked with other books. It's an easy way to switch over every book on two shelves without having to worry about any specific books.

**Syntax:**

```
ArraySwap(dynamic_array1, dynamic_array2);
```

**Parameters:**

Assume you have two bookcases filled with books. The "ArraySwap()" function allows you to swap out every book on one shelf for every other book, much like a sorcerer's spell. To make all the books from “Shelf A” move to “Shelf B” and all the books from “Shelf B” move to “Shelf A,” you can use the “ArraySwap()” spell if you have “Shelf A” stocked with some books and “Shelf B” stocked with other books. It's an easy way to switch over every book on two shelves without having to worry about any specific books.

**Example:**

```
void OnStart()
  {

// Declare dynamic arrays
   int dynamic_array1[];
   int dynamic_array2[];

// Resize dynamic arrays to have 5 elements each
   ArrayResize(dynamic_array1, 5);
   ArrayResize(dynamic_array2, 5);

// Assign values to dynamic arrays
   dynamic_array1[0] = 1;
   dynamic_array1[1] = 3;
   dynamic_array1[2] = 5;
   dynamic_array1[3] = 7;
   dynamic_array1[4] = 9;

   dynamic_array2[0] = 11;
   dynamic_array2[1] = 13;
   dynamic_array2[2] = 15;
   dynamic_array2[3] = 17;
   dynamic_array2[4] = 19;

// Print initial dynamic arrays
   Print("Initial dynamic_array1: ");
   ArrayPrint(dynamic_array1, 0, " ", 0, WHOLE_ARRAY);

   Print("Initial dynamic_array2: ");
   ArrayPrint(dynamic_array2, 0, " ", 0, WHOLE_ARRAY);

// Swap the contents of dynamic_array1 and dynamic_array2
   ArraySwap(dynamic_array1, dynamic_array2);

// Print dynamic arrays after swapping
   Print("After swapping - dynamic_array1: ");
   ArrayPrint(dynamic_array1, 0, " ", 0, WHOLE_ARRAY);

   Print("After swapping - dynamic_array2: ");
   ArrayPrint(dynamic_array2, 0, " ", 0, WHOLE_ARRAY);
  }
```

**Explanation:**

```
// Declare dynamic arrays
   int dynamic_array1[];
   int dynamic_array2[];

// Resize dynamic arrays to have 5 elements each
   ArrayResize(dynamic_array1, 5);
   ArrayResize(dynamic_array2, 5);
```

- The dynamic integer arrays "dynamic\_array1" and "dynamic\_array2" are declared in these lines. Each dynamic array's size is set to 5 using the “ArrayResize” function.


```
// Assign values to dynamic arrays
   dynamic_array1[0] = 1;
   dynamic_array1[1] = 3;
   dynamic_array1[2] = 5;
   dynamic_array1[3] = 7;
   dynamic_array1[4] = 9;

   dynamic_array2[0] = 11;
   dynamic_array2[1] = 13;
   dynamic_array2[2] = 15;
   dynamic_array2[3] = 17;
   dynamic_array2[4] = 19;
```

- These lines give each element in “dynamic\_array1” and “dynamic\_array2” a specific value.


```
// Print initial dynamic arrays
   Print("Initial dynamic_array1: ");
   ArrayPrint(dynamic_array1, 0, " ", 0, WHOLE_ARRAY);

   Print("Initial dynamic_array2: ");
   ArrayPrint(dynamic_array2, 0, " ", 0, WHOLE_ARRAY);
```

- These lines print the initial values of “dynamic\_array1” and “dynamic\_array2” to the console.


```
// Swap the contents of dynamic_array1 and dynamic_array2
   ArraySwap(dynamic_array1, dynamic_array2);
```

- To replace all of the contents of “dynamic\_array1” with “dynamic\_array2”, utilize the “ArraySwap()”  function.


```
// Print dynamic arrays after swapping
   Print("After swapping - dynamic_array1: ");
   ArrayPrint(dynamic_array1, 0, " ", 0, WHOLE_ARRAY);

   Print("After swapping - dynamic_array2: ");
   ArrayPrint(dynamic_array2, 0, " ", 0, WHOLE_ARRAY);
```

- Following the swapping process, these lines print the updated values of “dynamic\_array1” and “dynamic\_array2” to the console.


**Output:**

![Figure 9. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_9.png)

### **7\. ArrayReverse**

An array's elements can be rearranged, or flipped in order, by using the "ArrayReverse()" function. With the help of this feature, developers can quickly flip the elements in the sequence, making the final element the first and vice versa. This operation offers a flexible and effective way to change the arrangement of elements within arrays that contain a variety of data types. When reversing an element's order is required in programming, "ArrayReverse()" makes the process easier and results in more concise, readable code.

**Analogy**

Imagine a shelf containing a row of books, each with a number on it. It's as if those books were magically rearranged with the "ArrayReverse()" function, with the far right book now at the far left, and vice versa. It's a simple method to change the books on your shelf's order. Not only that, but it works similarly in programming, where you can use "ArrayReverse()" to flip the order of items in a list or array, starting with the last and working your way up to the first. It's like casting a spell backward on your array, effortlessly flipping it around.

**Syntax:**

```
ArrayReverse(array[], start, count);
```

**Parameters:**

- **array\[\]:** The array you want to reverse is represented by this parameter.
- **start:** The index within the array you want the reversing to start.
- **count:** The number of elements that should be reversed in the array.

**Examples:**

```
void OnStart()
  {

// Declare and initialize a 10-element array
   int array[10] = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19};

// Print the original array
   Print("Original Array: ");
   ArrayPrint(array, 0, " ", 0, WHOLE_ARRAY);

// Reverse the array starting from index 4
   ArrayReverse(array, 4, WHOLE_ARRAY);

// Print the array after reversal
   Print("Array after reversal from index 4: ");
   ArrayPrint(array, 0, " ", 0, WHOLE_ARRAY);

  }
```

This code initializes the array with the specified elements, prints the original array, reverses it starting from index 4, and then prints the result.

**Output:**

**![Figure 10. Code Output in MeterTrader5](https://c.mql5.com/2/72/figure_10.png)**

### **8\. ArraySort**

For sorting an array's elements in ascending order, the “ArraySort()” function is a useful tool. Through the use of this function, you can quickly arrange the elements of an array sequentially, starting with the smallest value and working their way up to the largest. This feature comes in especially handy when working with arrays that hold numerical values.

**Analogy**

Imagine that you wish to neatly arrange a collection of mixed-up numbers from the smallest to the largest. You can quickly sort these numbers into the proper order with the help of the “ArraySort()” function, which works like a magic spell. Thus, you can easily identify the smallest and largest numbers in your list by using a single, straightforward command to sort your numbers nicely. Because of the “ArraySort()” function's magic, you can see the numbers in an understandable and structured manner!

**Syntax:**

ArraySort(array\[\]); // array\[\] is the array you want to sort in ascending order

**Example:**

```
void OnStart()
  {

// Declare an array of numbers
   double array[5] = {9.5, 2.1, 7.8, 1.3, 5.6};

// Print the array before sorting
   Print("Array before sorting: ");
   ArrayPrint(array, 1, " ", 0, WHOLE_ARRAY);

// Use ArraySort to arrange the array in ascending order
   ArraySort(array);

// Print the array after sorting
   Print("Array after sorting: ");
   ArrayPrint(array, 1, " ", 0, WHOLE_ARRAY);

  }
```

**Outpt:**

![Figure 11. Code Output in MetaTrader5](https://c.mql5.com/2/72/figure_11.png)

### **Conclusion**

We have now covered many of the key ideas for array management in MQL5, including ArrayPrint, ArrayInsert, ArraySize, ArrayRange, ArrarRemove, ArraySwap, ArrayReverse, and ArraySort. The objective is to cover a broad range of essential ideas that serve as the foundation for trading strategy automation as we move through this series. It is critical to understand these array functions, particularly when working with historical data to create expert advisors. Regardless of experience level, I promise to ensure every reader masters these fundamental concepts, laying the groundwork for a rewarding journey into MQL5 programming and algorithmic trading.

As we conclude this article, I urge everyone to approach each idea with patience and curiosity because these basic building blocks will be important in future articles and will make developing reliable automated trading systems both rewarding and approachable. Your understanding is my top priority, so please don't hesitate to reach out and ask questions if you need assistance or clarification on any part of this article. We'll go into a thorough video session to cover all aspects of the array functions discussed in Parts 5 and 6 in the next article. To dispel any remaining doubts, this video offers a visual manual to improve your comprehension of these crucial MQL5 programming ideas.

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
**[Go to discussion](https://www.mql5.com/en/forum/464815)**
(2)


![Oscar Hayman](https://c.mql5.com/avatar/2020/5/5EC7925F-D6FE.jpg)

**[Oscar Hayman](https://www.mql5.com/en/users/oscarhayman)**
\|
27 May 2024 at 10:35

Please check the #5 ArrayRemove function example for \`static array\`. The function "count" is 2 and in the explanation you show as 3 elements removed. There seems to be a mistake.


![Israel Pelumi Abioye](https://c.mql5.com/avatar/2023/11/6554a830-8858.png)

**[Israel Pelumi Abioye](https://www.mql5.com/en/users/13467913)**
\|
27 May 2024 at 12:30

**Oscar Hayman [#](https://www.mql5.com/en/forum/464815#comment_53493490):**

Please check the #5 ArrayRemove function example for \`static array\`. The function "count" is 2 and in the explanation you show as 3 elements removed. There seems to be a mistake.

Hello Oscar, will look into that. Thank you.


![Neural networks made easy (Part 66): Exploration problems in offline learning](https://c.mql5.com/2/61/Neural_networks_are_easy_Part_66_LOGO.png)[Neural networks made easy (Part 66): Exploration problems in offline learning](https://www.mql5.com/en/articles/13819)

Models are trained offline using data from a prepared training dataset. While providing certain advantages, its negative side is that information about the environment is greatly compressed to the size of the training dataset. Which, in turn, limits the possibilities of exploration. In this article, we will consider a method that enables the filling of a training dataset with the most diverse data possible.

![The Group Method of Data Handling: Implementing the Multilayered Iterative Algorithm in MQL5](https://c.mql5.com/2/74/The_Group_Method_of_Data_Handling_Implementing_the_Multilayered_Iterative_Algorithm_in_MQL5___LOGO.png)[The Group Method of Data Handling: Implementing the Multilayered Iterative Algorithm in MQL5](https://www.mql5.com/en/articles/14454)

In this article we describe the implementation of the Multilayered Iterative Algorithm of the Group Method of Data Handling in MQL5.

![Build Self Optmising Expert Advisors in MQL5](https://c.mql5.com/2/74/Build_Self_Optmising_Expert_Advisors_in_MQL5__LOGO.png)[Build Self Optmising Expert Advisors in MQL5](https://www.mql5.com/en/articles/14630)

Build expert advisors that look forward and adjust themselves to any market.

![Gain An Edge Over Any Market](https://c.mql5.com/2/74/Gain_An_Edge_Over_Any_Market___LOGO.png)[Gain An Edge Over Any Market](https://www.mql5.com/en/articles/14441)

Learn how you can get ahead of any market you wish to trade, regardless of your current level of skill.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14407&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068952220206038882)

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