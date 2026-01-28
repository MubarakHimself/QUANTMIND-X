---
title: MQL Parsing by Means of MQL
url: https://www.mql5.com/en/articles/5638
categories: Integration
relevance_score: 3
scraped_at: 2026-01-23T21:15:30.192071
---

[![](https://www.mql5.com/ff/sh/x8fwvn495ta7y774z2/01.png)Does your broker offer sponsored hosting for trading?Now it's even easier to get MetaTrader VPS for free – contact your broker for details](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=xscnzeyhifcgygpwvysykhqydcmmbgpp&s=f87b748147e376d34c8f0fdb9737b1766f20cc2174769a0e6b9975b5c2e8ddae&uid=&ref=https://www.mql5.com/en/articles/5638&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5071697348553288895)

MetaTrader 5 / Integration


### Introduction

Programming is essentially formalizing and automating some processes using general-purpose or special-purpose languages. The [MetaTrader](https://www.metatrader5.com/ "https://www.metatrader5.com/") trading platform allows applying programming to solving a wide variety of the trader's problems, using embedded MQL language. Normally, the coding process is based on analyzing and processing application data according to the rules specified in source codes. However, a need sometimes arises for analyzing and processing the source codes themselves. Here are some examples.

One of the most consistent and popular tasks is contextual and semantic search in source code base. Surely, you can search for strings in a source code as in a normal text; however, the semantics of the sought for is getting lost. After all, in case of source codes, it is desirable to distinguish among the specifics of using the substring in each specific case. If a programmer wants to find where a specific variable is used, such as "notification," then a simple search by its name can return much more than necessary, where the string occurs in other values, such as a method name or a literal, or in comments.

A more complicated and sought for, as a rule, at larger projects, is the task of identifying code structure, dependencies, and class hierarchy. It is closely connected to meta-programming that allows performing code refactoring/improving and code generation. Recall that [MetaEditor](https://www.metatrader5.com/en/metaeditor/help "https://www.metatrader5.com/en/metaeditor/help") provides some opportunities of generating codes, particularly, creating the source codes of Expert Advisors using [Wizard](https://www.metatrader5.com/en/metaeditor/help/mql5_wizard "https://www.metatrader5.com/en/metaeditor/help/mql5_wizard") or forming a header file by source code. However, the potentials of this technology are much more powerful.

Code structure analysis allows computing various [quality metrics](https://en.wikipedia.org/wiki/Software_metric "https://en.wikipedia.org/wiki/Software_metric") and statistics, as well as finding the typical sources of runtime errors that cannot be detected by the compiler. In fact, compiler itself is, of course, the first tool of analyzing a source code and returns warnings of many types; however, checking for all potential errors is not usually built into it — this task is too large, so it is normally assigned to separate programs.

Moreover, parsing source codes is used for styling (formatting) and obfuscation (scrambling).

Many tools implementing the above problems are available to industrial programming languages. In case of MQL, the choice is limited. We can try to arrange analyzing the MQL with available means by tweaking that places MQL on the same level with C++. It works quite easily with some tools, such as [Doxygen](https://en.wikipedia.org/wiki/Doxygen "https://en.wikipedia.org/wiki/Doxygen"), but requires a deeper adaptation for more powerful means, such as [lint](https://en.wikipedia.org/wiki/Lint_(software) "https://en.wikipedia.org/wiki/Lint_(software)"), since MQL still is not C++.

It should be noted that this article only deals with [static code analysis](https://en.wikipedia.org/wiki/Static_program_analysis "https://en.wikipedia.org/wiki/Static_program_analysis"), while there are dynamic analyzers allowing you to keep 'on-the-go' track of memory operating errors, workflow locks, correctness of the values of variables, and much more in a virtual environment.

Various approaches can be used to perform the static analysis of source code. In case of simple problems, such as search for input variables of an MQL program, it is sufficient to use the regular expression library. In more general terms, analysis must be based on a parser considering the MQL grammar. It is this approach that we are going to consider in this article. We will also try to apply it practically.

In other words, we are going to write an MQL parser in MQL and obtain the metadata of source codes. This will both allow us solve the above problems and offer some other fantastic challenges hereafter. Thus, having a completely correct parser, we could rest on it to develop an MQL interpreter or convert MQL automatically into other trading languages, and vice versa (the so-called [transpiling](https://en.wikipedia.org/wiki/Source-to-source_compiler "https://en.wikipedia.org/wiki/Source-to-source_compiler")). However, I used the term of 'fantastic' for a reason. Although all these techniques are widely used in other areas already, we have to gain insight into the fundamentals first to approach them on the MetaTrader platform.

### Technology Review

There are many different parsers. We are not going to get into technical details — you can find introductory information on [Wikipedia](https://en.wikipedia.org/wiki/Parsing "https://en.wikipedia.org/wiki/Parsing"), and a huge amount of resources have been developed for advanced studies.

We should just note that the parser functions based on the so-called language-describing grammar. One of the most common forms of describing the grammar is that of [Backus-Naur](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form "https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form") (BNF). There are numerous BNF modifications, but we will not get into too many details and consider only the basic points.

In BNF, all the language structures are defined by the so-called non-terminals, while indivisible entities are terminals. Here, 'terminal' means the final point in parsing a text, i.e., a token containing a fragment of source code 'as is' and interpreted as a whole. This may be, for instance, the character of comma, a bracket, or a single word. We define the list of terminals, i.e., the alphabet, by ourselves within the grammar. Based on some rules, all other components of the program are made of terminals.

For example, we can specify that the program consists of operators in the simplified BNF-notation, as follows:

```
program ::= operator more_operators
more_operators ::= program | {empty}
```

Here, it is said that non-terminal 'program' can consist of one or more operators, the subsequent operators having been described using a recursive link to 'program'. Character '\|' (without quotation marks in BNF) means logical OR — selecting one of the options. To complete the recursion, special terminal {empty} is used in the above entry. It can be represented as an empty string or an option to skip the rule.

Character 'operator' is also a non-terminal and requires 'expanding' via other non-terminals and terminals, for example, like this:

```
operator ::= name '=' expression ';'
```

This entry defines that each operator shall start with the variable name, then the '=' sign, an expression, and the operator ends with character ';'. Characters '=' and ';' are terminals. The name consists of letters:

```
name ::= letter more_letters
more_letters ::= name | {empty}
letter ::= [A-Z]
```

Here, any character, from 'A' through 'Z' may be used as a letter (their set is denoted with square brackets).

Let the expression be formed of operands and arithmetics (operations):

```
expression ::= operand more_operands
more_operands ::= operation expression | {empty}
```

In the simplest case, an expression consists of only one operand. However, there can be more of them (more\_operands), then they are appended to it via the operation character as a subexpression. Let operands be able to be references to variables or numbers, while operations can be '+' or '-':

```
operand ::= name | number
number ::= digit more_digits
more_digits ::= number | {empty}
digit ::= [0-9]
operation ::= '+' | '-'
```

Thus, we have described the grammar of a simple language, in which calculations can be made with numbers and variables, such as:

```
A = 10;
X = A - 5;
```

When we start parsing the text, then we, in fact, check which rules work and which of them fail. Those having worked should sooner or later produce 'production' — find a terminal that coincides with the content in the current position in the text, cursor being moved to the next position. This process is repeated until the entire text is associated to non-terminals fragment by fragment, forming a sequence permitted by the grammar.

In the example above, the parser having character 'A' at its input will start viewing the rules as outlined below:

```
program
  operator
    name
      letter
        'A'
```

and find the first match. The cursor will move to the next character '='. Since 'letter' is one letter, the parser will return into rule 'name'. Since 'name' can only consist of letters, the option more\_letters does not function (it is selected to be equal to {empty}), and the parser returns into the rule 'operator', where it is the terminal '=' that follows the name. This will be the second match. Then, expanding the rule 'expression', the parser will find the operand — integer 10 (as a sequence of two digits), and, finally, semicolon will complete the parsing of the first string. Following its outcomes, we will, in fact, know the variable name, the expression content, namely the fact that it consists of one number, and its value. The second string is parsed in a similar manner.

It is important to note that the grammar of one and the same language can be recorded in different ways, the latter ones formally matching each other. However, in some cases, following the rules literally may lead to certain problems. For example, description of a number could be represented as follows:

```
number ::= number digit | {empty}
```

This entry format is named [left recursion](https://en.wikipedia.org/wiki/Left_recursion "https://en.wikipedia.org/wiki/Left_recursion"): Non-terminal 'number' is both in the left and in the right part determining the rules of its 'production', it being the very first, left, in the string (hence the name of 'left recursion'). This is the simplest, explicit left recursion. However, it can be implicit, if the non-terminal is expanded after some intermediate rules into a string, that starts with that non-terminal.

Left recursion often occurs in formal BNF-notations of the grammars of programming languages. However, some types of parsers, depending on their implementations, may be stuck in a loop with similar rules. Indeed, if we consider the rule as guidelines to action (parsing algorithm), then this entry will recursively enter the 'number' again and again, without reading any new terminals from the input stream, which, in theory, should happen when expanding the non-terminal 'digit'.

Since we are trying to create the MQL grammar not from scratch, but using, where possible, the BNF-notations of C++ grammar, attention will have to be paid to left recursions, and the rules shall be re-written in an alternative manner. At the same time, we will also have to implement protection against infinite looping — as we will see further, the grammars of the C++-type or MQL-type languages are so ramified that it seems to to be impossible to check them for correctness manually.

Here, it is pertinent to note that writing parsers is a real science and it is recommended to start mastering this domain gradually, on a 'simple-to-complex' basis. The simplest one is the [recursive descent parser](https://en.wikipedia.org/wiki/Recursive_descent_parser "https://en.wikipedia.org/wiki/Recursive_descent_parser"). It considers the input text as a whole corresponding to the starting non-terminal of the grammar. In the example above, that was non-terminal 'program'. Following each suitable rule, the parser checks the sequence of input characters for matching the terminals and moves along the text when finding the matches. If the parser finds a mismatch at any moment of parsing, it rolls back to the rules, in which an alternative had been specified, and, thus, checks all possible language structures. This algorithm completely repeats the operations that we performed purely theoretically in the example above.

The 'roll-back' operation is called 'backtracking' and can provide negative effect upon the speed of response. Thus, in the worst scenario, a classical descent parser generates an exponentially growing number of options when viewing the text. There are various options to solving this problem, such as a predicting parser that needs no backtracking. Its operation time is linear.

However, this is only possible for grammars, for which the rule of 'production' can be unambiguously selected by a pre-defined number of the subsequent characters k. Such more advanced parsers are based in their operation on special transition tables that are computed in advance based on all the rules of the grammar. They include, but are not limited to, [LL-parsers](https://en.wikipedia.org/wiki/LL_parser "https://en.wikipedia.org/wiki/LL_parser") and [LR-parsers](https://en.wikipedia.org/wiki/LR_parser "https://en.wikipedia.org/wiki/LR_parser").

LL stands for Left-to-right, Leftmost derivation. It means that the text is viewed from left to right, and the rules are viewed from left to right as well, which is equivalent to a top-down conclusion (from general to specific), and, in this sense, LL is a relative of our descent parser.

LR stands for Left-to-right, Rightmost derivation. It means that the text is viewed from left to right, as before, but the rules are viewed from right to left, which is equivalent to bottom-up forming the language structures, i.e., from single characters to larger and larger non-terminals. Besides, LR has fewer problems with the left recursion.

In the names of parsers LL(k) and LR(k), the number of characters k is usually specified as lookahead, up to which they view the text forward. In most cases, selecting k = 1 is sufficient. This sufficiency is sketchy, though. The matter is that many modern programming languages, including C++ and MQL, are not languages having a [context-free grammar](https://en.wikipedia.org/wiki/Context-free_grammar "https://en.wikipedia.org/wiki/Context-free_grammar"). In other words, the same fragments of the text can be interpreted differently, depending on the context. In such cases, to make a decision regarding the message of what is written, one or even any quantity of characters is usually not enough, since you have to tie the parser in with other tools, such as preprocessor or symbol table (the list of the identifiers already recognized and their meanings).

For the C++ language, there is a prototypical case of ambiguity (it suites for MQL, as well). What does the expression below mean?

```
x * y;
```

This may be the product of variables x and y; however, it may be the description of variable y as the x type pointer. Don't let it embarrass you that the product of multiplication, if it is multiplication, is not saved anywhere, since the operation of multiplication can be overloaded and have side effects.

Another problem, from which the most C++ compilers suffered in the past, was ambiguity in interpreting two consecutive characters '>'. The matter is that, upon introducing the templates, the structures of the following type started to appear in source codes:

```
vector<pair<int,int>> v;
```

while the sequence '>>' was initially defined as a shift operator. For a while, until a finer process was introduced for such specific cases, we had to write similar expressions with spaces:

```
vector<pair<int,int> > v;
```

In our parser, we will also have to circumvent this problem.

Generally, even this brief introduction makes it clear that the description and implementation of advanced parsers would require more efforts, in terms of both outlining scope and the time it takes for mastering them. So, in this article, we will confine ourselves to the simplest parser — the recursive descent one.

### Planning

Thus, the task of the parser is to read the text fed to its input, break it into a stream of indivisible fragments (tokens), and then compare them to the permitted language structures described using the MQL grammar in the BNF-notation or in one close to it.

For a start, we will need a class that reads files — we will name it FileReader. Since the MQL project can consist of several files included from the main one using directive #include, it may be necessary to have many FileReader instances, so we will factor one more class, FileReaderController, in our developments.

Broadly speaking, the text from files to be processed represent a standard string. However, we will need to transfer it among different classes, while MQL, unfortunately, does not allow for string pointers (I remember about the references, but they cannot be used in declaring class members, while the only alternative — passing the reference to all the methods via inputs is difficult to work with). Therefore, we will create a separate class, Source, representing a string wrapper. It will execute another important function.

The matter is that, resulting from connecting the 'includes' (and, therefore, recursively reading the header files from dependencies), we will obtain a consolidated text from all files at the controller output. To detect errors, we will have to use the shift in the consolidated source code to obtain the name and string of the original file, from which the text has been taken. This 'map' of source codes locations in files will also be supported and stored by class Source.

Here, a pertinent question arises: Was it impossible not to combine source codes, but process each file individually, instead? Yes, it would probably be more correct. However, in that case, it would require creating a parser instance for each file and then somehow cross-linking the syntax trees generated by the parser at the output. Therefore, I decided to combine the source codes and feed them to a single parser. You can experiment with the alternative approach, if you want.

For FileReaderController to be able to find the #include directives, it is necessary not to just read the text from files, but also perform preview in searching for those directives. Thus, a kind of preprocessor is required. In MQL, it does other good jobs. Particularly, it allows identifying macros and then replacing them with actual expressions (moreover, it considers the potential recursion of calling a macro from a macro). But it would be better not to spread ourselves too thin in our first MQL parsing project. Therefore, we will not process macros in our preprocessor — that would require not just additionally describing the grammar of macros, but also interpreting them on-the-go to have correct expressions to be substituted in the source code. Do you remember what we said about the interpreter in introduction? Now, it would be useful here, and it will become clear later why it is important. This is area of your independent experiments number 2.

The preprocessor will be implemented in class Preprocessor. At its level, a rather controversial process takes place. While reading the files and searching for #include directives in them, parsing and moving within the text are performed at the lowest level — character by character. However, the preprocessor 'transparently' passes everything that is not a directive through itself, and operates with the largest blocks at the output — whole files or file fragments among directives. And then parsing will continue at an intermediate level, to describe which, we will have to introduce a couple of terms.

First of all, it is lexical unit — an abstract minimal unit of lexical analysis, a non-zero length substring. Another term is often used in conjunction with it — token that is another analysis unit, which is not abstract, but concrete. Both represent a fragment of text, such as an individual character, word, or even a block of comments. The fine difference between them is that we mark fragments with a meaning at the token level. For example, if word 'int' appears in the text, it is a lexical unit for MQL, which we will denote with token INT - an element in enumeration of all permissible tokens in the MQL language. In other words, the set of lexical units shall mean a dictionary of strings corresponding with token types.

One of the pros of tokens is that they allow dividing the text into fragments larger than characters. As a result, the text is parsed in two passes: First, high-level tokens are formed from the stream of letters, and then, based on them, language structures are parsed. This allows considerably simplifying the language grammar and the parser operation.

Special class, Scanner, will highlight tokens in the text. It can be considered as a low-level parser with the pre-defined and hard-wired grammar processing the text by letters. The exact types of tokens that will be needed are considered below. If somebody sets out to experiment number 1 (loading each file to a dedicated parser), then it will be there where we will be able to combine the preprocessor with the scanner and, as soon as the token "#include <something>" is found, create a new FileReader, a new scanner, and a new parser, and transfer control to them.

All key (reserved) words, as well as the symbols of punctuation and operations, will be the tokens of MQL. The full list of MQL keywords is attached in file reserved.txt and included in the source code of the scanner.

Identifiers, numbers, strings, literals, and other constants, such as dates, will also be independent tokens.

When parsing a text into tokens, all spaces, line breaks, and tabulations are suppressed (ignored). The only exception in form of special processing should be done for line breaks, since counting them will allow pointing out a line containing an error, if any.

Thus, having fed the scanner input with a consolidated text, we will obtain a list of tokens at the output. It is this list of tokens that will be processed by the parser that we implement in class Parser.

To interpret tokens by the MQL rules, it is necessary to pass grammar to the parser in the BNF-notation. To describe the grammar, let us try to repeat the approach used by parser [boost::spirit](https://en.wikipedia.org/wiki/Spirit_Parser_Framework "https://en.wikipedia.org/wiki/Spirit_Parser_Framework") in a simplified form. Essentially, the grammar rules shall be described using the expressions of the MQL expressions due to overloading some operators.

For this purpose, let us introduce the hierarchy of classes Terminal, NonTerminal, and their derivatives. Terminal will be the base class that, by default, corresponds with a unit token. As it was said in the theoretical part, terminal is a finite indivisible element of parsing the rules: If a character is found in the current position of the text, that coincides with the terminal token, it means that the character corresponds with the grammar. We can read it and move on.

We will use NonTerminal for complex structures, in which terminals and other non-terminals can be used in various combinations. This can be shown by an example.

Suppose we have to describe a simple grammar to calculate expressions, in which only integers and operations 'plus' ('+') and 'multiply' ('\*') can be utilized. For the sake of simplicity, we will confine ourselves to a scenario where there are only two operands, such as 10+1 or 5\*6.

Based on this task, it is necessary, first of all, to identify the terminal corresponding with an integer. It is this terminal that will be compared to any valid operand in the expression. Considering that each time the scanner finds an integer in the text, it produces the relevant token CONST\_INTEGER, let us define the Terminal class object referring to that token. In a pseudocode, this will be:

```
Terminal value = CONST_INTEGER;
```

This entry means that we have created the 'value' object of class Terminal attached to token 'integer'.

Symbols of operations are also terminals with the relevant tokens, PLUS and STAR, generated by the scanner for single characters '+' and '\*':

```
Terminal plus = PLUS;
Terminal star = STAR;
```

To be able to use any of them in the expression, let us introduce a non-terminal combining two operations by OR:

```
NonTerminal operation = plus | star;
```

This is where overloading the operators comes into play: In class Terminal (and all its descendants), operator\| must create references from the parent object (in this case, 'operation') to the descendant ones ('plus' and 'star') and mark the type of operation with them — logical OR.

When the parser starts checking non-terminal 'operation' for matching with the text under cursor, it will delegate further checking ("depth") to object 'operation', and that object will call in the loop checking for descendant elements, 'plus' and 'star' (up to the first match, since it is OR). Since they are terminals, they will just return their tokens to the parser, and the latter one will find out whether the character in the text matches one of the operations.

The expression can consist of several values and operations among them. Therefore, the expression is a non-terminal, too, and it has to be 'expanded' via terminals and non-terminals.

```
NonTerminal expression = value + operation + value;
```

Here, we overload operator+ that means that operands must follow each other in the specified order. Again, the implementation of the function means that the parent non-terminal, 'expression', must save the references to descendant objects 'value', 'operation', and another 'value', operation type being logical AND. Indeed, in this case, the rule shall only be followed, if all the components are available.

Checking the text with the parser for corresponding with the correct expression will first call for a check in the 'expression' array of references and then in objects 'value' and 'operation' (the latter will recursively refer to 'plus' and 'minus'), and finally to 'value' again. At any stage, if checking goes down to the terminal level, the value of the token is compared to the current character in the text and, if they match, the cursor moves to the next token; if not, then an alternative should be searched for. For example, in this case, if checking for operation 'plus' is unsuccessful, checking will continue for 'star'. If all alternatives are exhausted and no match has been found, it means that the grammar rules are violated.

Operators '\|' and '+' are not all the operators that we are going to overload in our classes. We will provide a complete description of them in the implementation section.

Declaring the objects of class Terminal and its derivatives containing the references to other, increasingly smaller things forms the abstract syntax tree (AST) of the pre-defined grammar. It is abstract, because it is not associated with the specific tokens from the input text, i.e., theoretically, the grammar describes an infinite set of valid strings - MQL codes, in our case.

Thus, we have generally considered the main classes of the project. To facilitate envisioning the entire picture, let us summarize them in the UML diagram of classes.

![UML-Diagram of MQL Parsing Classes](https://c.mql5.com/2/35/parser.png)

**UML Diagram of MQL Parsing Classes**

Some classes, such as TreeNode, have not been considered yet. The parser will use its objects while parsing the input text to save all the matches 'terminal=token' found. As a result, we will obtain the so-called concrete syntax tree (CST) at the output, in which all tokens are someway included hierarchically in the terminals and non-terminals of the grammar.

In principle, creating the tree will be optional, since it may turn out to be too large for the real source codes. Instead of obtaining the parsing outputs as a tree, we will provide the callback interface — Callback. Having created our object implementing this interface, we will pass it to the parser and be able to receive notifications about each 'production' produced, i.e., each grammar rule that has worked. Thus, we will be able to analyze syntax and semantics 'on-the-go', without waiting for a full tree.

Classes of non-terminals having the prefix of 'Hidden' will be used to automatically implicitly create the intermediate groups of grammar objects, which we are going to describe in more details in the next section.

### Implementation

**Reading Files**

_Source_

Class Source is, first of all, a storage of the string containing the text to be processed. Basically, it looks as follows:

```
#define SOURCE_LENGTH 100000

class Source
{
  private:
    string source;

  public:
    Source(const uint length = SOURCE_LENGTH)
    {
      StringInit(source, length);
    }

    Source *operator+=(const string &x)
    {
      source += x;
      return &this;
    }

    Source *operator+=(const ushort x)
    {
      source += ShortToString(x);
      return &this;
    }

    ushort operator[](uint i) const
    {
      return source[i];
    }

    string get(uint start = 0, uint length = -1) const
    {
      return StringSubstr(source, start, length);
    }

    uint length() const
    {
      return StringLen(source);
    }
};
```

The class has variable 'source' for the text and overridden operators for the most frequent operations with strings. So far, let us leave the second role of this class behind the scenes, that consists in maintaining the list of files, from which the stored string is assembled. Having such a 'wrapping' for the input text, we can try to fill it from one file. Class FileReader is responsible for this task.

_FileReader_

Before we start programming, the method of opening and reading the file should be defined. Since we are processing a text, it is logical to select the FILE\_TXT mode. This would release us from the manual control over the line break characters that, on top of everything else, can be differently coded in different editors (normally, it is the pair of symbols, CR LF; however, in MQL source codes publicly available, you can see alternatives, such as CR only or LF only). It should be reminded that files are read string by string when in the text mode.

Another problem to think of is supporting texts in different encodings. Since we are going to read several different files, a part of which may be saved as single-byte strings (ANSI), while another part as wider double-byte ones (UNICODE), it is better to let the system select the right modes on-the-go, i.e., from file to file. Moreover, files can also be saved in the UTF-8 encoding.

MQL turned out to be able to automatically read various text files in a correct encoding, if the following inputs are set for function FileOpen:

```
FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
```

We will then use this combination, having added to it, by default, flags FILE\_SHARE\_READ \| FILE\_SHARE\_WRITE.

In class FileReader, we will provide members for storing the file name ('filename'), open file descriptor ('handle'), and the current text line ('line').

```
class FileReader
{
  protected:
    const string filename;
    int handle;
    string line;
```

In addition, we will track the current line number and the cursor position in the line (column).

```
    int linenumber;
    int cursor;
```

We will save the read lines in an instance of object Source.

```
    Source *text;
```

We will pass the file name, flags, and the ready Source object for receiving the data to the constructor.

```
  public:
    FileReader(const string _filename, Source *container = NULL, const int flags = FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE, const uint codepage = CP_UTF8): filename(_filename)
    {
      handle = FileOpen(filename, flags, 0, codepage);
      if(handle == INVALID_HANDLE)
      {
        Print("FileOpen failed ", _filename, " ", GetLastError());
      }
      line = NULL;
      cursor = 0;
      linenumber = 0;
      text = container;
    }

    string pathname() const
    {
      return filename;
    }
```

Let us check whether the file has been opened successfully and see about closing the descriptor in the destructor.

```
    bool isOK()
    {
      return (handle > 0);
    }

    ~FileReader()
    {
      FileClose(handle);
    }
```

Character-wise reading the data from the file will be ensured by method getChar.

```
    ushort getChar(const bool autonextline = true)
    {
      if(cursor >= StringLen(line))
      {
        if(autonextline)
        {
          if(!scanLine()) return 0;
          cursor = 0;
        }
        else
        {
          return 0;
        }
      }
      return StringGetCharacter(line, cursor++);
    }
```

When the string containing text 'line' is empty or read through the end, this method tries to read the next string using method scanLine. If the 'line' string contains some unprocessed characters, getChar simply returns the character under the cursor and then moves the cursor to the next position.

Method scanLine is defined in an obvious way:

```
    bool scanLine()
    {
      if(!FileIsEnding(handle))
      {
        line = FileReadString(handle);
        linenumber++;
        cursor = 0;
        if(text != NULL)
        {
          text += line;
          text += '\n';
        }
        return true;
      }

      return false;
    }
```

Note that, since the file is opened in the text mode, we do not get line feeds returned; however, we need them to count the lines and as the final signs of some language structures, such as single-line comments. So we are adding symbol '\\n'.

Along with reading the data from the file as such, class FileReader must enable comparing the input data under cursor to lexical units. For that purpose, let us add the following methods.

```
    bool probe(const string lexeme) const
    {
      return StringFind(line, lexeme, cursor) == cursor;
    }

    bool match(const string lexeme) const
    {
      ushort c = StringGetCharacter(line, cursor + StringLen(lexeme));
      return probe(lexeme) && (c == ' ' || c == '\t' || c == 0);
    }

    bool consume(const string lexeme)
    {
      if(match(lexeme))
      {
        advance(StringLen(lexeme));
        return true;
      }
      return false;
    }

    void advance(const int next)
    {
      cursor += next;
      if(cursor > StringLen(line))
      {
        error(StringFormat("line is out of bounds [%d+%d]", cursor, next));
      }
    }
```

Method 'probe' compares the text to the lexical unit passed. Method 'match' does practically the same, but additionally checks that the lexical unit is mentioned as a single word, that is, it must be followed by a separator, such as space, tabulation, or end of line. Method 'consume' "gobbles" the lexical unit/word passed, i.e., it verifies that the input text matches the predefined text, and, in case of success, moves the cursor over the end of the lexical unit. In case of failure, the cursor is not moved, while the method returns 'false'. Method 'advance' simply moves the cursor forward by a predefined number of characters.

Finally, let us consider a small method returning the sign of the file end.

```
    bool isEOF()
    {
      return FileIsEnding(handle) && cursor >= StringLen(line);
    }
```

There are other helper methods for reading the fields in this class; you can find them in the source codes attached.

Objects of class FileReader must be created somewhere. Let us delegate that to class FileReaderController.

_FileReaderController_

In class FileReaderController, a stack of included files ('includes'), a map of files already included ('files'), a pointer to the current file being processed ('current'), and the input text ('source') must be maintained.

```
class FileReaderController
{
  protected:
    Stack<FileReader *> includes;
    Map<string, FileReader *> files;
    FileReader *current;
    const int flags;
    const uint codepage;

    ushort lastChar;
    Source *source;
```

Lists, stacks, arrays, such as BaseArray, and maps ('Map') occurring in source codes are included from supporting header files that will not be described here, since I have already used them in my previous articles. However, the complete archive is attached hereto, of course.

Controller creates an empty 'source' object in its constructor:

```
  public:
    FileReaderController(const int _flags = FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE, const uint _codepage = CP_UTF8, const uint _length = SOURCE_LENGTH): flags(_flags), codepage(_codepage)
    {
      current = NULL;
      lastChar = 0;
      source = new Source(_length);
    }
```

The 'source', as well as the subordinate objects of FileReader are released from the map in the destructor:

```
#define CLEAR(P) if(CheckPointer(P) == POINTER_DYNAMIC) delete P;

    ~FileReaderController()
    {
      for(int i = 0; i < files.getSize(); i++)
      {
        CLEAR(files[i]);
      }
      delete source;
    }
```

To include one or another file into processing, including the very first project file with extension mq5, let us provide for method 'include'.

```
    bool include(const string _filename)
    {
      Print((current != NULL ? "Including " : "Processing "), _filename);

      if(files.containsKey(_filename)) return true;

      if(current != NULL)
      {
        includes.push(current);
      }

      current = new FileReader(_filename, source, flags, codepage);
      source.mark(source.length(), current.pathname());

      files.put(_filename, current);

      return current.isOK();
    }
```

It checks map 'files' for whether the predefined file has already been processed, and immediately returns 'true', if the file is available. Otherwise, the process continues. If this is the very first file, we will create object FileReader, make it 'current', and save in map 'files'. If this is not the first file, i.e., some other file is being processed at this moment, then we should save it in stack 'includes'. As soon as the included file is completely processed, we will come back to processing the current file, starting from the point at which the other file was included.

One line in this method 'include' will not be compiled yet:

```
      source.mark(source.length(), current.pathname());
```

The class 'source' does not contain method 'mark' yet. As it should be clear from the context, at this point we switch from one file to another one, and, therefore, we should mark somewhere the source and its shift in the combined text. This is what the method 'mark' will do. At any moment, the current length of the input text is the point where the data of the new file will be added. Let us come back to class Source and add the map of files:

```
class Source
{
  private:
    Map<uint,string> files;

  public:
    void mark(const uint offset, const string file)
    {
      files.put(offset, file);
    }
```

The main task of reading the characters from the file in class FileReaderController is executed by method getChar that delegates a part of the work to the current FileReader object.

```
    ushort getChar(const bool autonextline = true)
    {
      if(current == NULL) return 0;

      if(!current.isEOF())
      {
        lastChar = current.getChar(autonextline);
        return lastChar;
      }
      else
      {
        while(includes.size() > 0)
        {
          current = includes.pop();
          source.mark(source.length(), current.pathname());
          if(!current.isEOF())
          {
            lastChar = current.getChar();
            return lastChar;
          }
        }
      }
      return 0;
    }
```

If there is a current file and it has not been read until the end, we will call its method getChar and return the character obtained. If the current file has been read until the end, then we will check whether there are directives for including any other files in the stack 'includes'. If there are files, we extract the upper one, make it 'current', and continue reading the characters from it. Additionally, we should remember noting in the object 'source' that the data source was switched to the older file.

Class FileReaderController can also return the sign of finishing the reading.

```
    bool isAtEnd()
    {
      return current == NULL || (current.isEOF() && includes.size() == 0);
    }
```

Among other things, let us provide a couple of methods to get the current file and text.

```
    const Source *text() const
    {
      return source;
    }

    FileReader *reader()
    {
      return current;
    }
```

Now everything is ready for pre-processing the files.

_Preprocessor_

The preprocessor will control the only instance of class FileReaderController (controller), as well as decide whether it is necessary to load header files (flag 'loadIncludes'):

```
class Preprocessor
{
  protected:
    FileReaderController *controller;
    const string includes;
    bool loadIncludes;
```

The matter is that we may want to process some files without dependencies — for example, for the purpose of debugging or reducing work time. We will save the default folder for header files in the string variable 'includes'.

The constructor receives all these values and the name of the initial file (and path thereto) from the user, creates a controller, and calls the method 'include' for the file.

```
  public:
    Preprocessor(const string _filename, const string _includes, const bool _loadIncludes = false, const int _flags = FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE, const uint _codepage = CP_UTF8, const uint _length = SOURCE_LENGTH): includes(_includes)
    {
      controller = new FileReaderController(_flags, _codepage, _length);
      controller.include(_filename);
      loadIncludes = _loadIncludes;
    }
```

Now, let us write method 'run' to be called by the client directly to start processing of one or more files.

```
    bool run()
    {
      while(!controller.isAtEnd())
      {
        if(!scanLexeme()) return false;
      }
      return true;
    }
```

We will read lexical units until the controller bumps into the end of data.

Here comes method 'scanLexeme':

```
    bool scanLexeme()
    {
      ushort c = controller.getChar();

      switch(c)
      {
        case '#':
          if(controller.reader().consume("include"))
          {
            if(!include())
            {
              controller.reader().error("bad include");
              return false;
            }
          }
          break;
          ...
      }
      return true; // symbol consumed
    }
```

If the program sees the character '#', then it tries to "absorb" the next word 'include'. If it is not there, then it was a single character '#' to be just skipped (getChar moves the cursor one position further). If word 'include' is found, the directive must be processed, which is done by the method 'include'.

```
    bool include()
    {
      ushort c = skipWhitespace();

      if(c == '"' || c == '<')
      {
        ushort q = c;
        if(q == '<') q = '>';

        int start = controller.reader().column();

        do
        {
          c = controller.getChar();
        }
        while(c != q && c != 0);

        if(c == q)
        {
          if(loadIncludes)
          {
            Print(controller.reader().source());

            int stop = controller.reader().column();

            string name = StringSubstr(controller.reader().source(), start, stop - start - 1);
            string path = "";

            if(q == '"')
            {
              path = controller.reader().pathname();
              StringReplace(path, "\\", "/");
              string parts[];
              int n = StringSplit(path, '/', parts);
              if(n > 0)
              {
                ArrayResize(parts, n - 1);
              }
              else
              {
                Print("Path is empty: ", path);
                return false;
              }

              int upfolder = 0;
              while(StringFind(name, "../") == 0)
              {
                name = StringSubstr(name, 3);
                upfolder++;
              }

              if(upfolder > 0 && upfolder < ArraySize(parts))
              {
                ArrayResize(parts, ArraySize(parts) - upfolder);
              }

              path = StringImplodeExt(parts, CharToString('/')) + "/";
            }
            else // '<' '>'
            {
              path = includes; // folder;
            }

            return controller.include(path + name);
          }
          else
          {
            return true;
          }
        }
        else
        {
          Print("Incomplete include");
        }
      }
      return false;
    }
```

This method uses 'skipWhitespace' (not covered herein) to skip all spaces after the word 'include', finds an opening quotation mark or character '<', then scans the text through the closing quotation mark or closing character '>', and finally extracts the string containing the path and the name of the header file. Then the options are processed to load files from the same folder or from the standard folder of header files. As a result, a new path and a new name for loading are formed, following which the controller is assigned to process the file.

Along with processing the directive #include, we must skip comment blocks and strings in order not to interpret them as instructions, if lexical unit '#include' were to be inside. Therefore, let us add the relevant options into the operator 'switch' in the method 'scanLexeme'.

```
        case '/':
          if(controller.reader().probe("*"))
          {
            controller.reader().advance(1);
            if(!blockcomment())
            {
              controller.reader().error("bad block comment");
              return false;
            }
          }
          else
          if(controller.reader().probe("/"))
          {
            controller.reader().advance(1);
            linecomment();
          }
          break;
        case '"':
          if(!literal())
          {
            controller.reader().error("unterminated string");
            return false;
          }
          break;
```

This is, for example, how comment blocks are skipped:

```
    bool blockcomment()
    {
      ushort c = 0, c_;

      do
      {
        c_ = c;
        c = controller.getChar();
        if(c == '/' && c_ == '*') return true;
      }
      while(!controller.reader().isEOF());

      return false;
    }
```

Other helper methods are implemented similarly.

Thus, having class 'Preprocessor' and other classes, theoretically, we can already load the text from working files, something like this.

```
#property script_show_inputs

input string SourceFile = "filename.txt";
input string IncludesFolder = "";
input bool LoadIncludes = false;

void OnStart()
{
  Preprocessor loader(SourceFile, IncludesFolder, LoadIncludes);

  if(!loader.run())
  {
    Print("Loader failed");
    return;
  }

  // output entire data as it is assembled from one or many files
  int handle = FileOpen("dump.txt", FILE_WRITE | FILE_TXT | FILE_ANSI, 0, CP_UTF8);
  FileWriteString(handle, loader.text().get());
  FileClose(handle);
}
```

Why "theoretically"? The matter is that MetaTrader allows working only with "sandbox" files, i.e., the directory MQL5/Files. However, our goal is to process source codes contained in folders MQL5/Include, MQL5/Scripts, MQL5/Experts, and MQL5/Indicators.

To circumvent this restriction, let us use the feature of Windows to assign [symbolic links](https://en.wikipedia.org/wiki/Symbolic_link "https://en.wikipedia.org/wiki/Symbolic_link") to file system objects. In our case, the so-called 'junctions' are best stuited for forwarding the access to folders on the local computer. They are created using command:

```
mklink /J new_name existing_target
```

Parameter new\_name is the name of the new virtual 'folder' that will point at a real folder existing\_target.

To create junctions to the specified folders containing source codes, let us open the folder MQL5/Files, create subfolder Sources in it, and go to that subfolder. Then we will copy the file makelink.bat attached hereto. This command script actually contains one string:

```
mklink /J %1 "..\..\%1\"
```

It takes one input %1 — the name of the application folder from among those inside MQL5, such as 'Include'. Relative path '..\\..\\' suggests that the command file is in the above MQL5/Files/Sources folder, and then the target folder (existing\_target) will be formed as MQL5/%1. For example, if, being in folder Sources, we execute command

```
makelink Include
```

then in the folder Sources, a virtual folder Include will appear, getting into which, we will get into MQL5/Include. Similarly, we can create "twins" for folders Experts, Scripts, etc. In picture below, the Explorer is shown, with the opened MQL5/Include/Expert folder with standard header files available in folder MQL5/Files/Sources.

![Windows Symbolic Links for Folders of the MQL5 source Codes](https://c.mql5.com/2/35/mklinks.png)

**Windows Symbolic Links for Folders of the MQL5 source Codes**

If necessary, we can delete symbolic links as normal files (but, of course, we should first make sure that we are deleting the folder having a small arrow in its bottom-left corner, not the original folder).

We could create a junction directly on the root working folder of MQL5, but I prefer and recommend to open access occasionally: All MQL programs will be able to use the link to read your source codes, including logins, passwords, and top-secret trading systems, provided that they are stored there.

Upon creating the links, the parameter 'IncludesFolder' of the above script will really work: The value of Sources/Include/ points at real folder MQL5/Include. In parameter 'SourceFile', we can exemplify the analysis by the source code of a script, such as Sources/Scripts/test.mq5.

**Tokenization**

Token types that must be distinguished in MQL are combined in enumeration 'TokenType' in the homonym header file (attached). We will not set it out in this article. Let us just note that there are single-character tokens among them, such as various brackets and braces ('(', '\[', or '{'), signs of equality '=', plus '+', or minus '-', as well as two-character ones, such as '==', '!=', etc. Besides, separate tokens will be numbers, strings, dates (i.e., constants of the supported types), all the words reserved in MQL, such as operators, types, this, modifiers like 'input', 'const', etc., as well as identifiers (other words). Additionally, there is token EOF to denote the end of input data.\
\
_Token_\
\
When viewing a text, the scanner will identify the type of each subsequent token by a special algorithm (to be considered below) and create an object of class Token. This is a very simple class.\
\
```\
class Token\
{\
  private:\
    TokenType type;\
    int line;\
    int offset;\
    int length;\
\
  public:\
    Token(const TokenType _type, const int _line, const int _offset, const int _length = 0)\
    {\
      type = _type;\
      line = _line;\
      offset = _offset;\
      length = _length;\
    }\
\
    TokenType getType() const\
    {\
      return type;\
    }\
\
    int getLine() const\
    {\
      return line;\
    }\
    ...\
\
    string content(const Source *source) const\
    {\
      return source.get(offset, length);\
    }\
};\
```\
\
The object stores the type of the token, its shift within the text, and its length. If we need a string value of the token, we pass to method 'content' a pointer to string 'source' and cut out the relevant fragment from it.\
\
Now it is time to refer to the scanner that is also called "tokenizer".\
\
_Scanner (tokenizer)_\
\
In class Scanner, we will describe a static array with MQL keywords:\
\
```\
class Scanner\
{\
  private:\
    static string reserved[];\
```\
\
and then initialize it in the source code by including a text file:\
\
```\
static string Scanner::reserved[] =\
{\
#include "reserved.txt"\
};\
```\
\
Let us add to this array a static map of correspondence between the string representation and the type of each token.\
\
```\
    static Map<string, TokenType> keywords;\
```\
\
Let us fill out the map in the constructor (see below).\
\
In the scanner, we will also need a pointer to inputs, the resulting list of tokens, and multiple counters.\
\
```\
    const Source *source; // wrapped string\
    List<Token *> *tokens;\
    int start;\
    int current;\
    int line;\
```\
\
Variable 'start' will always point at the beginning of the next token to be processed. Variable 'current' is the cursor of moving within the text. It will always 'run' forward from 'start' as the current characters are being checked for corresponding to a token and, as soon as a match is found, the substring from 'start' to 'current' will fall into the new token. Variable 'line' is the number of the current line in the total text.\
\
Constructor of class Scanner:\
\
```\
  public:\
    Scanner(const Source *_source): line(0), current(0)\
    {\
      tokens = new List<Token *>();\
      if(keywords.getSize() == 0)\
      {\
        for(int i = 0; i < ArraySize(reserved); i++)\
        {\
          keywords.put(reserved[i], TokenType(BREAK + i));\
        }\
      }\
      source = _source;\
    }\
```\
\
Here BREAK is the token type identifier for the first reserved word in alphabetical order. The order of strings in file reserved.txt and identifiers in enumeration TokenType must match. For example, element BREAK in the enumeration is apparently corresponding with 'break'.\
\
Method 'scanTokens' takes the central place in the class.\
\
```\
    List<Token *> *scanTokens()\
    {\
      while(!isAtEnd())\
      {\
        // We are at the beginning of the next lexeme\
        start = current;\
        scanToken();\
      }\
\
      start = current;\
      addToken(EOF);\
      return tokens;\
    }\
```\
\
More and more new tokens are generated in its cycle. Methods 'isAtEnd' and 'addToken' are simple:\
\
```\
    bool isAtEnd() const\
    {\
      return (uint)current >= source.length();\
    }\
\
    void addToken(TokenType type)\
    {\
      tokens.add(new Token(type, line, start, current - start));\
    }\
```\
\
All the hard work is done by method 'scanToken'. However, before presenting it, we should learn about some simple helper methods — they are similar to those we have already seen in class 'Preprocessor', this is why their purpose seems not to need any explanations.\
\
```\
    bool match(ushort expected)\
    {\
      if(isAtEnd()) return false;\
      if(source[current] != expected) return false;\
\
      current++;\
      return true;\
    }\
\
    ushort previous() const\
    {\
      if(current > 0) return source[current - 1];\
      return 0;\
    }\
\
    ushort peek() const\
    {\
      if(isAtEnd()) return '\0';\
      return source[current];\
    }\
\
    ushort peekNext() const\
    {\
      if((uint)(current + 1) >= source.length()) return '\0';\
      return source[current + 1];\
    }\
\
    ushort advance()\
    {\
      current++;\
      return source[current - 1];\
    }\
```\
\
Now, back to method 'scanToken'.\
\
```\
    void scanToken()\
    {\
      ushort c = advance();\
      switch(c)\
      {\
        case '(': addToken(LEFT_PAREN); break;\
        case ')': addToken(RIGHT_PAREN); break;\
        ...\
```\
\
It reads the next character and, depending on its code, creates a token. We will not provide all one-character tokens here, since they are created similarly.\
\
If a token suggests to be a two-character one, the processing complexifies:\
\
```\
        case '-': addToken(match('-') ? DEC : (match('=') ? MINUS_EQUAL : MINUS)); break;\
        case '+': addToken(match('+') ? INC : (match('=') ? PLUS_EQUAL : PLUS)); break;\
        ...\
```\
\
Forming tokens for lexemes '--', '-=', '++', and '+=' is shown below.\
\
The current version of the scanner skips comments:\
\
```\
        case '/':\
          if(match('/'))\
          {\
            // A comment goes until the end of the line\
            while(peek() != '\n' && !isAtEnd()) advance();\
          }\
```\
\
You can save them in special tokens, if you wish.\
\
Block structures, such as strings, literals, and preprocessor directives are processed in the allocated helper methods, we will not consider them in details:\
\
```\
        case '"': _string(); break;\
        case '\'': literal(); break;\
        case '#': preprocessor(); break;\
```\
\
This is an example of how a string is scanned:\
\
```\
    void _string()\
    {\
      while(!(peek() == '"' && previous() != '\\') && !isAtEnd())\
      {\
        if(peek() == '\n')\
        {\
          line++;\
        }\
        advance();\
      }\
\
      if(isAtEnd())\
      {\
        error("Unterminated string");\
        return;\
      }\
\
      advance(); // The closing "\
\
      addToken(CONST_STRING);\
    }\
```\
\
If no token type has triggered, then default testing is performed, where numbers, identifiers, and key words are checked.\
\
```\
        default:\
\
          if(isDigit(c))\
          {\
            number();\
          }\
          else if(isAlpha(c))\
          {\
            identifier();\
          }\
          else\
          {\
            error("Unexpected character `" + ShortToString(c) + "` 0x" + StringFormat("%X", c) + " @ " + (string)current + ":" + source.get(MathMax(current - 10, 0), 20));\
          }\
          break;\
```\
\
implementations of isDigit and isAlpha are obvious. Here, only the method 'identifier' is shown.\
\
```\
    void identifier()\
    {\
      while(isAlphaNumeric(peek())) advance();\
\
      // See if the identifier is a reserved word\
      string text = source.get(start, current - start);\
\
      TokenType type = keywords.get(text);\
      if(type == null) type = IDENTIFIER;\
\
      addToken(type);\
    }\
```\
\
Full implementations of all methods can be found in the source codes attached. Not to reinvent the wheel, I took a part of the code from book [Crafting Interpreters](https://www.mql5.com/go?link=http://craftinginterpreters.com/ "http://craftinginterpreters.com/") having made some corrections, of course.\
\
Basically, that's the whole scanner. In case of no errors, method 'scanTokens' will return to the user a list of tokens, which can be passed to the parser. However, the parser must have a grammar to refer to when parsing the list of tokens. Therefore, before proceeding to the parser, we must consider the grammar description. We form it from the objects of class Terminal and its derivatives.\
\
**Grammar Description**\
\
Let us first imagine that we have to describe not the MQL grammar, but that of a certain simple language for calculating arithmetic expressions, i.e., a calculator. This is the permissible calculation formula:\
\
(10 + 1) \* 2\
\
Let us only permit integers and operations '+', '-', '\*', and '/', without prioritizing: We will use '(' and ')' for priorities.\
\
The entry point of the grammar must be a non-terminal describing the entire expression. Suppose it will be sufficient for that to write:\
\
```\
NonTerminal expression;\
```\
\
The expression consists of operands, i.e., integer values, and operator symbols. All the above are terminals, i.e., they can be created based on tokens supported by the scanner.\
\
Suppose, we describe them as follows:\
\
```\
Terminal plus(PLUS), star(STAR), minus(MINUS), slash(SLASH);\
Terminal value(CONST_INTEGER);\
```\
\
As we can see, the constructor of terminals must allow for passing the token type as a parameter.\
\
The simplest possible expression is just a number. It would be logical to denote that as follows:\
\
```\
expression = value;\
```\
\
This is restarting the assignment operator. In it, we will have to save the link to object 'value' (let's name it 'eq' from 'equivalence') in a variable inside the 'expression'. As soon as the parser is assigned to check the 'expression' for matching with the entered string, it delegates checking to the non-terminal. The latter one will "see" the link to 'value' and ask the parser to check the 'value', and checking will finally reach the terminal where just matching the tokens will take place, i.e., the token in the terminal and that in the input stream.\
\
However, the expression may additionally have an operation and the second operand; therefore, it is necessary to extend the rule 'expression'. For this purpose, we will preliminarily describe the new non-terminal:\
\
```\
NonTerminal operation;\
operation = (plus | star | minus | slash) + value;\
```\
\
Here, many interesting things take place behind the scenes. Operator '\|' must be overloaded in the class to ensure logically grouping the elements by OR. However, the operator is called for a terminal, i.e., a simple character, while we need a group of elements. Therefore, the first element of the group, for which the execution environment will call an operator ('plus', in this case), must check whether it is a member of a group and, if there is still no group, create it dynamically as an object of class HiddenNonTerminalOR. Then the implementation of the overloaded operator must add 'this' and the adjacent right terminal 'star' (passed to the operator function as an argument) to the newly created group. The operator returns a link to this group for the subsequent (chained) operators '\|' to be now called for HiddenNonTerminalOR.\
\
To maintain the array containing the group members, we will certainly provide array 'next' in the class. Its name means the next detail level of grammar elements. For each element that we add to this array of child nodes, we should set a backlink to the parent node. We will name it 'parent'. Availability of a non-zero 'parent' will exactly mean the membership in the group. Resulting from the execution of the code within brackets, we will obtain HiddenNonTerminalOR with an array containing all 4 operation symbols.\
\
Then overloaded operator '+' comes into play. It must work similarly to operator '\|', that is, to create an implicit group of elements, too; but this time that of class HiddenNonTerminalAND; and they must be checked by the rule of logical AND at the parsing stage.\
\
Please note that we have a dependency hierarchy of terminals and non-terminals formed — in this case, object HiddenNonTerminalAND will contain two child elements: The newly created group HiddenNonTerminalOR and 'value'. HiddenNonTerminalAND is, in turn, dependent on non-terminal 'operation'.\
\
Priority of operations '\|' and '+' is that, in the absence of brackets, AND is processed first, and OR is processed after that. That's exactly why we had to put in brackets all the versions of characters in 'operation'.\
\
Having the description of non-terminal 'operation', we can correct the grammar of the expression:\
\
```\
expression = value + operation;\
```\
\
They allegedly describe the expressions represented as A @ B, where A and B are integers, while @ is an action. But there is a sticking point here.\
\
We already have two rules involving object 'value'. This means that the link to a parent set in the first rule will be re-written in the second one. For that not to happen, not objects must be inserted in the rules, but their copies.\
\
For this purpose, we will provide overloading for two operators: '~' and '^'. The first of them, the unary one, is placed before the operand. In the object that has received the call of the relevant operator function, we will dynamically create a proxy object and return it to the calling code. The secondary operator is binary. Along with the object, we will pass to it the current string number in the grammar source code, i.e., constant \_\_LINE\_\_ predefined by the MQL compiler. Thus, we will have the ability to distinguish the object instances defined implicitly by the numbers of lines where they are created. This will help debug complex grammars. Put it another way, operators '~' and '^' perform the same work, but the first one is in the release mode, while the second one in the debugging mode.\
\
Proxy instance represents an object of class HiddenNonTerminal, in which the above variable 'eq' refers to the original object.\
\
Thus, we are going to re-write the grammar of expressions, considering creating proxy objects.\
\
```\
operation = (plus | star | minus | slash) + ~value;\
expression = ~value + operation;\
```\
\
Since 'operation' is used only once, we needn't make a copy for it. Each logical reference increases the recursion by one when expanding the expression. However, to avoid errors in large grammars, we recommend making references everywhere. If even a non-terminal is being used only once now, it may later occur in another part of the grammar. We will provide the source code with checking for the parent node override to display an error message.\
\
Now, our grammar can process '10+1'. But it has lost the ability to read a separate number. Indeed, non-terminal 'operation' must be optional. For this purpose, let us implement an overloaded operator '\*'. If a grammar element is multiplied by 0, then we can omit it when performing the checks, since its absence does not lead to an error.\
\
```\
expression = ~value + operation*0;\
```\
\
Overloading the multiplication operator will allow us to implement another important thing — repeating the element as many times as we wish. In this case, we will multiply the element by 1. In the terminal class, this property, i.e., multiplicity or optionality, is stored in variable 'mult'. Cases where an element both is optional and can repeat many times are easily implemented by two links: The first one must be optional (optional\*0), and the other one multiple (optional = element\*1).\
\
The current grammar of the calculator has one more weakness. It is not suited for long expressions with several operations, such as 1+2+3+4+5. To correct this, we should change non-terminal 'operation'.\
\
```\
operation = (plus | star | minus | slash) + ~expression;\
```\
\
We will replace 'value' with the 'expression' itself, having allowed therewith the cyclic parsing of more and more new endings of expressions.\
\
The finishing touch is supporting expressions enclosed by brackets. It is not hard to guess that they play the same role as unit value ('value'). Therefore, let us redefine it as an alternative from two options: An integer and a subexpression in brackets. The entire grammar will appear as follows:\
\
```\
NonTerminal expression;\
NonTerminal value;\
NonTerminal operation;\
\
Terminal number(CONST_INTEGER);\
Terminal left(LEFT_PAREN);\
Terminal right(RIGHT_PAREN);\
Terminal plus(PLUS), star(STAR), minus(MINUS), slash(SLASH);\
\
value = number | left + expression^__LINE__ + right;\
operation = (plus | star | minus | slash) + expression^__LINE__;\
expression = value + operation*0;\
```\
\
Let us take a closer look at how the above classes are arranged internally.\
\
_Terminal_\
\
In class Terminal, let us describe the fields for the token type ('me'), multiplicity properties ('mult'), optional name ('name', for identifying the non-terminals in the logs), links to the production ('eq'), to the parent ('parent'), and subordinate elements (array 'next').\
\
```\
class Terminal\
{\
  protected:\
    TokenType me;\
    int mult;\
    string name;\
    Terminal *eq;\
    BaseArray<Terminal *> next;\
    Terminal *parent;\
```\
\
The fields are filled out in constructors and in setter methods and read using getter methods that we do not discuss here for brevity sake.\
\
We will overload operators according to the following principle:\
\
```\
    virtual Terminal *operator|(Terminal &t)\
    {\
      Terminal *p = &t;\
      if(dynamic_cast<HiddenNonTerminalOR *>(p.parent) != NULL)\
      {\
        p = p.parent;\
      }\
\
      if(dynamic_cast<HiddenNonTerminalOR *>(parent) != NULL)\
      {\
        parent.next << p;\
        p.setParent(parent);\
      }\
      else\
      {\
        if(parent != NULL)\
        {\
          Print("Bad OR parent: ", parent.toString(), " in ", toString());\
\
          ... error\
        }\
        else\
        {\
          parent = new HiddenNonTerminalOR("hiddenOR");\
\
          p.setParent(parent);\
          parent.next << &this;\
          parent.next << p;\
        }\
      }\
      return parent;\
    }\
```\
\
Here, grouping by OR is shown. Everything is similar for AND.\
\
Setting the feature of multiplicity is in operator '\*':\
\
```\
    virtual Terminal *operator*(const int times)\
    {\
      mult = times;\
      return &this;\
    }\
```\
\
In destructor, we will take care of correctly deleting the instances created.\
\
```\
    ~Terminal()\
    {\
      Terminal *p = dynamic_cast<HiddenNonTerminal *>(parent);\
      while(CheckPointer(p) != POINTER_INVALID)\
      {\
        Terminal *d = p;\
        if(CheckPointer(p.parent) == POINTER_DYNAMIC)\
        {\
          p = dynamic_cast<HiddenNonTerminal *>(p.parent);\
        }\
        else\
        {\
          p = NULL;\
        }\
        CLEAR(d);\
      }\
    }\
```\
\
Finally, the main method of class Terminal, responsible for parsing.\
\
```\
    virtual bool parse(Parser *parser)\
    {\
      Token *token = parser.getToken();\
\
      bool eqResult = true;\
```\
\
Here, we have received the reference to the parser and read the current token from it (parser class will be considered below).\
\
```\
      if(token.getType() == EOF && mult == 0) return true;\
```\
\
If the token is EOF and the current element is optional, it means that a correct end of text has been found.\
\
Then we will check whether there is a reference from overloaded operator '=' to the original instance of the element, if we are in the copy. If there is a reference, we will send it into method 'match' of the parser for checking.\
\
```\
      if(eq != NULL) // redirect\
      {\
        eqResult = parser.match(eq);\
\
        bool lastResult = eqResult;\
\
        // if multiple tokens expected and while next tokens are successfully consumed\
        while(eqResult && eq.mult == 1 && parser.getToken() != token && parser.getToken().getType() != EOF)\
        {\
          token = parser.getToken();\
          eqResult = parser.match(eq);\
        }\
\
        eqResult = lastResult || (mult == 0);\
\
        return eqResult; // redirect was fulfilled\
      }\
```\
\
Moreover, here the situation is processed where an element may repeat ('mult' = 1): The parser is called again and again, while method 'match' is returning success.\
\
Success mark — 'true' or 'false' — is returned from method 'parse' both in this branch and in other situations, such as for a terminal:\
\
```\
      if(token.getType() == me) // token match\
      {\
        parser.advance(parent);\
        return true;\
      }\
```\
\
For terminal, we just compare its token 'me' to the current token in inputs and, if there is a match, assign the parser to move the cursor to the next input token, using method 'advance'. In the same method, the parser notifies the client program that result has been produced in non-terminal 'parent'.\
\
For a group of elements, everything is a bit more complicated. Let us consider logical AND; the version for OR will be similar. Using virtual method hasAnd (in class Terminal, it returns 'false', while it is overridden in descendants), we find out whether the array with subordinate elements has been filled for checking by AND.\
\
```\
      else\
      if(hasAnd()) // check AND-ed conditions\
      {\
        parser.pushState();\
        for(int i = 0; i < getNext().size(); i++)\
        {\
          if(!parser.match(getNext()[i]))\
          {\
            if(mult == 0)\
            {\
              parser.popState();\
              return true;\
            }\
            else\
            {\
              parser.popState();\
              return false;\
            }\
          }\
        }\
\
        parser.commitState(parent);\
        return true;\
      }\
```\
\
Since this non-terminal will be considered correct if all its components match with the grammar, we will call method 'match' of the parser for all of them, in the cycle. If at least one negative result happens, the entire checking will fail. However, there is an exception: If non-terminal is optional, the grammar rules will still be followed, even if 'false' is returned from method 'match'.\
\
Note that we save the current state (pushState) before the cycle in the parser, restore it (popState) at early exit, and, in case of the checking completed fully successfully, confirm the new state (commitState). This is necessary to delay the notifications for the client code on the new 'production' until the entire grammar rule works completely. The word "state" shall simply mean the current cursor position in the stream of input tokens.\
\
If neither token nor the groups of subordinate elements have triggered within method 'parse', all that remains is just to check the optionality of the current object:\
\
```\
      else\
      if(mult == 0) // last chance\
      {\
        // parser.advance(); - don't consume token and proceed to next result\
        return true;\
      }\
```\
\
Otherwise, we "fall" into the method end that signifies an error, i.e., the text does not correspond with the grammar.\
\
```\
      if(dynamic_cast<HiddenNonTerminal *>(&this) == NULL)\
      {\
        parser.trackError(&this);\
      }\
\
      return false;\
    }\
```\
\
Now let's describe classes derived from class Terminal.\
\
_Non-terminals, hidden and explicit_\
\
The main task of class HiddenNonTerminal is to create dynamic instances of the objects and collect garbage.\
\
```\
class HiddenNonTerminal: public Terminal\
{\
  private:\
    static List<Terminal *> dynamic; // garbage collector\
\
  public:\
    HiddenNonTerminal(const string id = NULL): Terminal(id)\
    {\
    }\
\
    HiddenNonTerminal(HiddenNonTerminal &ref)\
    {\
      eq = &ref;\
    }\
\
    virtual HiddenNonTerminal *operator~()\
    {\
      HiddenNonTerminal *p = new HiddenNonTerminal(this);\
      dynamic.add(p);\
      return p;\
    }\
    ...\
};\
```\
\
Class HiddenNonTerminalOR ensures overloading operator '\|' (simpler than that in class Terminal, because HiddenNonTerminalOR itself is a "container", i.e., the owner of a group of the subordinate grammar elements).\
\
```\
class HiddenNonTerminalOR: public HiddenNonTerminal\
{\
  public:\
    virtual Terminal *operator|(Terminal &t) override\
    {\
      Terminal *p = &t;\
      next << p;\
      p.setParent(&this);\
      return &this;\
    }\
    ...\
};\
```\
\
Class HiddenNonTerminalAND has been implemented in a similar way.\
\
Class NonTerminal ensures overloading operator '=' ("production" in the rules).\
\
```\
class NonTerminal: public HiddenNonTerminal\
{\
  public:\
    NonTerminal(const string id = NULL): HiddenNonTerminal(id)\
    {\
    }\
\
    virtual Terminal *operator=(Terminal &t)\
    {\
      Terminal *p = &t;\
      while(p.getParent() != NULL)\
      {\
        p = p.getParent();\
        if(p == &t)\
        {\
          Print("Cyclic dependency in assignment: ", toString(), " <<== ", t.toString());\
          p = &t;\
          break;\
        }\
      }\
\
      if(dynamic_cast<HiddenNonTerminal *>(p) != NULL)\
      {\
        eq = p;\
      }\
      else\
      {\
        eq = &t;\
      }\
      eq.setParent(this);\
      return &this;\
    }\
};\
```\
\
Finally, there is class Rule — the descendant of NonTerminal. However, its entire role is to mark some rules as primary (if they generate object Rule) or secondary (if they result in NonTerminal) when describing the grammar.\
\
For the convenience of describing the non-terminals, the following macros have been created:\
\
```\
// debug\
#define R(Y) (Y^__LINE__)\
\
// release\
#define R(Y) (~Y)\
\
#define _DECLARE(Cls) NonTerminal Cls(#Cls); Cls\
#define DECLARE(Cls) Rule Cls(#Cls); Cls\
#define _FORWARD(Cls) NonTerminal Cls(#Cls);\
#define FORWARD(Cls) Rule Cls(#Cls);\
```\
\
The line, a unique name, is specified as an argument of macros. Forward declaration will be required where non-terminals refer to each other — we have seen that in the grammar of the calculator.\
\
Moreover, for generating terminals with tokens, special class Keywords is implemented that supports collecting the garbage.\
\
```\
class Keywords\
{\
  private:\
    static List<Terminal *> keywords;\
\
  public:\
    static Terminal *get(const TokenType t, const string value = NULL)\
    {\
      Terminal *p = new Terminal(t, value);\
      keywords.add(p);\
      return p;\
    }\
};\
```\
\
To use it in describing the grammar, the following macros have been created:\
\
```\
#define T(X) Keywords::get(X)\
#define TC(X,Y) Keywords::get(X,Y)\
```\
\
Let us see how the calculator grammar considered above is described using the implemented program interfaces.\
\
```\
  FORWARD(expression);\
  _DECLARE(value) = T(CONST_INTEGER) | T(LEFT_PAREN) + R(expression) + T(RIGHT_PAREN);\
  _DECLARE(operation) = (T(PLUS) | T(STAR) | T(MINUS) | T(SLASH)) + R(expression);\
  expression = R(value) + R(operation)*0;\
```\
\
Finally, we are ready to study class Parser.\
\
_Parser_\
\
Class Parser has members to store the input list of tokens ('tokens'), the current position in it ('cursor'), the furthest known position ('maxcursor', it is used in error diagnostics), a stack of positions before calling the nested groups of elements ('states', for rolling back, remembering the 'backtracking'), and a link to the input text ('source', for printing logs and for other purposes).\
\
```\
class Parser\
{\
  private:\
    BaseArray<Token *> *tokens; // input stream\
    int cursor;                 // current token\
    int maxcursor;\
    BaseArray<int> states;\
    const Source *source;\
```\
\
Besides, the parser tracks the entire hierarchy of calls by the grammar elements, using 'stack'. Class TreeNode used in this template is a simple container for a pair (terminal, token), and its source code can be found in the archive attached. Errors are accumulated for diagnostics in another stack — 'errors'.\
\
```\
    // current stack, how the grammar unwinds\
    Stack<TreeNode *> stack;\
\
    // holds current stack on every problematic point\
    Stack<Stack<TreeNode *> *> errors;\
```\
\
The parser constructor receives the list of tokens, source text, and the optional flag of enabling forming a syntax tree during parsing.\
\
```\
  public:\
    Parser(BaseArray<Token *> *_tokens, const Source *text, const bool _buildTree = false)\
```\
\
If the tree mode is enabled, all successful "productions" that have got onto the stack as the objects of TreeNode are beaded on the tree root — variable 'tree':\
\
```\
    TreeNode *tree;   // concrete syntax tree (optional result)\
```\
\
for this purpose, class TreeNode supports an array of child nodes. After the parser has finished its work, the tree, provided that it has been enabled, can be obtained using the following method:\
\
```\
    const TreeNode *getCST() const\
    {\
      return tree;\
    }\
```\
\
the main method of the parser, 'match', in its simplified form, appears as follows.\
\
```\
    bool match(Terminal *p)\
    {\
      TreeNode *node = new TreeNode(p, getToken());\
      stack.push(node);\
      int previous = cursor;\
      bool result = p.parse(&this);\
      stack.pop();\
\
      if(result) // success\
      {\
        if(stack.size() > 0) // there is a holder to bind to\
        {\
          if(cursor > previous) // token was consumed\
          {\
            stack.top().bind(node);\
          }\
          else\
          {\
            delete node;\
          }\
        }\
      }\
      else\
      {\
        delete node;\
      }\
\
      return result;\
    }\
```\
\
Methods 'advance' and 'commitState' that we saw as we were getting acquainted with the classes of terminals, are implemented like this (some details are skipped).\
\
```\
    void advance(const Terminal *p)\
    {\
      production(p, cursor, tokens[cursor], stack.size());\
      if(cursor < tokens.size() - 1) cursor++;\
\
      if(cursor > maxcursor)\
      {\
        maxcursor = cursor;\
        errors.clear();\
      }\
    }\
\
    void commitState(const Terminal *p)\
    {\
      int x = states.pop();\
      for(int i = x; i < cursor; i++)\
      {\
        production(p, i, tokens[i], stack.size());\
      }\
    }\
```\
\
'advance' moves the cursor along the list of tokens. If the position has exceeded the maximum one, we can remove the errors accumulated, since they are recorded at each unsuccessful check.\
\
Method 'production' uses a callback interface to notify the user of the parser about the 'production' — we will use it further in tests.\
\
```\
    void production(const Terminal *p, const int index, const Token *t, const int level)\
    {\
      if(callback) callback.produce(&this, p, index, t, source, level);\
    }\
```\
\
The interface is defined as:\
\
```\
interface Callback\
{\
  void produce(const Parser *parser, const Terminal *, const int index, const Token *, const Source *context, const int level);\
  void backtrack(const int index);\
};\
```\
\
Object implementing this interface on the client side can be connected to the parser using method setCallback — then it will be called at each 'production'. Alternatively, such object can be individually connected to any terminal due to overloading operator \[Callback \*\]. It is useful for debugging to place break points at the specific grammar points.\
\
Let us try the parser in practice.\
\
### Practice, Part 1: Calculator\
\
We have already had the calculator grammar. Let us create a debugging script for it. We will also supplement it afterwards for tests with the MQL grammar.\
\
```\
#property script_show_inputs\
\
enum TEST_GRAMMAR {Expression, MQL};\
\
input TEST_GRAMMAR TestMode = Expression;;\
input string SourceFile = "Sources/calc.txt";;\
input string IncludesFolder = "Sources/Include/";;\
input bool LoadIncludes = false;\
input bool PrintCST = false;\
\
#include <mql5/scanner.mqh>\
#include <mql5/prsr.mqh>\
\
void OnStart()\
{\
  Preprocessor loader(SourceFile, IncludesFolder, LoadIncludes);\
  if(!loader.run())\
  {\
    Print("Loader failed");\
    return;\
  }\
\
  Scanner scanner(loader.text());\
  List<Token *> *tokens = scanner.scanTokens();\
\
  if(!scanner.isSuccess())\
  {\
    Print("Tokenizer failed");\
    delete tokens;\
    return;\
  }\
\
  Parser parser(tokens, loader.text(), PrintCST);\
\
  if(TestMode == Expression)\
  {\
    testExpressionGrammar(&parser);\
  }\
  else\
  {\
    //...\
  }\
\
  delete tokens;\
}\
\
void testExpressionGrammar(Parser *p)\
{\
  _FORWARD(expression);\
  _DECLARE(value) = T(CONST_INTEGER) | T(LEFT_PAREN) + R(expression) + T(RIGHT_PAREN);\
  _DECLARE(operation) = (T(PLUS) | T(STAR) | T(MINUS) | T(SLASH)) + R(expression);\
  expression = R(value) + R(operation)*0;\
\
  while(p.match(&expression) && !p.isAtEnd())\
  {\
    Print("", "Unexpected end");\
    break;\
  }\
\
  if(p.isAtEnd())\
  {\
    Print("Success");\
  }\
  else\
  {\
    p.printState();\
  }\
\
  if(PrintCST)\
  {\
    Print("Concrete Syntax Tree:");\
    TreePrinter printer(p);\
    printer.printTree();\
  }\
\
  Comment("");\
}\
```\
\
Intent of the script is to read the passed file in the preprocessor, transform it into a stream of tokens using the scanner, and check with the parser for the specified grammar. Checking is performed by calling method 'match', into which the root grammar rule, 'expression', is passed.\
\
As an option (PrintCST), we can display the syntax tree of the processed expression in the log, using utility class TreePrinter.\
\
Attention! For real programs, the tree will be very large. This option is only recommended when debugging the small fragments of grammars or where the entire grammar is not large, as in case of our calculator.\
\
If we run a test script for the file with expression "(10+1)\*2", we will obtain the following tree (remember to select TestMode=Expression and PrintCST= true):\
\
```\
Concrete Syntax Tree:\
|  |  |Terminal LEFT_PAREN @ (\
|  |   |  | |Terminal CONST_INTEGER @ 10\
|  |   |  |NonTerminal value\
|  |   |  |  |Terminal PLUS @ +\
|  |   |  |  |  | |Terminal CONST_INTEGER @ 1\
|  |   |  |  |  |NonTerminal value\
|  |   |  |  |NonTerminal expression\
|  |   |  |NonTerminal operation\
|  |   |NonTerminal expression\
|  |  |Terminal RIGHT_PAREN @ )\
|  |NonTerminal value\
|  |  |Terminal STAR @ *\
|  |  |  | |Terminal CONST_INTEGER @ 2\
|  |  |  |NonTerminal value\
|  |  |NonTerminal expression\
|  |NonTerminal operation\
|NonTerminal expression\
```\
\
Vertical lines denote the levels of processing the non-terminals that have been explicitly described in the grammar, i.e., the named ones. Spaces correspond with the levels, where the implicitly created non-terminals of classes HiddenXYZ were "expanded" — all such nodes are not displayed in the log by default; but in class TreePrinter, there is an option to enable them.\
\
Note that option PrintCST functions based on a special structure with metadata — a tree of TreeNode objects. Our parser can optionally produce it upon analysis as a response to calling method getCST. Recall that including the tree arrangement mode is set by the third parameter of the parser constructor.\
\
You can experiment with other expressions, including those incorrect to make sure that error processing is present. For example, if we impair the expression having made it '10+', we will obtain the notification:\
\
```\
Failed\
First 2 tokens read out of 3\
Source: EOF (file:Sources/Include/Layouts/calc.txt; line:1; offset:4) ``\
Expected:\
CONST_INTEGER in expression;operation;expression;value;\
LEFT_PAREN in expression;operation;expression;value;\
```\
\
Well, all classes work. Now we can move to the main practical part — MQL parsing.\
\
### Practice, Part 2: MQL Grammar\
\
On the engineering side, everything is ready for writing the MQL grammar. However, it is much more complicated than a small calculator. Creating it from scratch would be a Herculean task. To solve the problem, let us use the fact that MQL is a semblance of C++.\
\
For C++, many ready-made descriptions of grammar are publicly accessible. One of them is attached hereto as file cppgrmr.htm. It would also be challenging to transfer it completely to our grammar. First, many structures are not supported in MQL anyway. Second, there is often the left recursion in the notation, because of which the rules have to be changed. Finally, third, it is desirable to limit the size of grammar, since it may provide a negative affect on the processing rate: It would be reasonable to leave some rarely used features to be added optionally by those who would really need them.\
\
The sequence of mentioning the alternatives of OR matters, since the first triggered version intercepts the subsequent checks. If versions may, in some conditions, partly coincide due to skipping some optional elements, then we will have to either rearrange them or specify longer and more specific structures first and then the shorter and more general ones.\
\
Let us demonstrate how some notations from the htm file are transformed into the grammar of our rules and terminals.\
\
In C++ grammar:\
\
```\
assignment-expression:\
  conditional-expression\
  unary-expression assignment-operator assignment-expression\
\
assignment-operator: one of\
  = *= /= %= += –= >= <= &= ^= |=\
```\
\
In MQL grammar:\
\
```\
_FORWARD(assignment_expression);\
_FORWARD(unary_expression);\
\
...\
\
assignment_expression =\
    R(unary_expression) + R(assignment_operator) + R(assignment_expression)\
  | R(conditional_expression);\
\
_DECLARE(assignment_operator) =\
    T(EQUAL) | T(STAR_EQUAL) | T(SLASH_EQUAL) | T(DIV_EQUAL)\
  | T(PLUS_EQUAL) | T(MINUS_EQUAL) | T(GREATER_EQUAL) | T(LESS_EQUAL)\
  | T(BIT_AND_EQUAL) | T(BIT_XOR_EQUAL) | T(BIT_OR_EQUAL);\
```\
\
In C++ grammar:\
\
```\
unary-expression:\
  postfix-expression\
  ++ unary-expression\
  –– unary-expression\
  unary-operator cast-expression\
  sizeof unary-expression\
  sizeof ( type-name )\
  allocation-expression\
  deallocation-expression\
```\
\
In MQL grammar:\
\
```\
unary_expression =\
    R(postfix_expression)\
  | T(INC) + R(unary_expression) | T(DEC) + R(unary_expression)\
  | R(unary_operator) + R(cast_expression)\
  | T(SIZEOF) + T(LEFT_PAREN) + R(type) + T(RIGHT_PAREN)\
  | R(allocation_expression) | R(deallocation_expression);\
```\
\
In C++ grammar:\
\
```\
statement:\
  labeled-statement\
  expression-statement\
  compound-statement\
  selection-statement\
  iteration-statement\
  jump-statement\
  declaration-statement\
  asm-statement\
  try-except-statement\
  try-finally-statement\
```\
\
In MQL grammar:\
\
```\
statement =\
    R(expression_statement) | R(codeblock) | R(selection_statement)\
  | R(labeled_statement) | R(iteration_statement) | R(jump_statement);\
```\
\
There is a rule for declaration\_statement in MQL grammar, too. But it is transferred. Many rules were recorded in a way simpler than in C++. In principle, this grammar is a living organism, or, in English parlance, "work in progress." Errors are much likely to occur when interpreting specific structures in source codes.\
\
For MQL grammar, the entry point is rule 'program', consisting of 1 or more 'elements':\
\
```\
  _DECLARE(element) =\
     R(class_decl)\
   | R(declaration_statement) | R(function) | R(sharp) | R(macro);\
\
  _DECLARE(program) = R(element)*1;\
```\
\
In our test script, the presented MQL grammar is described in function testMQLgrammar:\
\
```\
void testMQLgrammar(Parser *p)\
{\
  // all grammar rules go first\
  // ...\
  _DECLARE(program) = R(element)*1;\
```\
\
And it is there where parsing is launched (similarly to the calculator):\
\
```\
  while(p.match(&program) && !p.isAtEnd())\
  ...\
```\
\
If an error occurs, the problematic element should be localized by logs, and the specific grammar rule must be debugged on a separate input fragment of the text (it is recommended to use a fragment containing 5-6 tokens at most). In other words, method 'match' of the parser should be called for a specific rule, and a file with a separate language structure should be fed to the input. To output the traces of the parser into the log, it is necessary to uncomment the directive in the script:\
\
```\
//#define PRINTX Print\
```\
\
Attention! The amount of information to be displayed is very large.\
\
Before debugging, it is recommended to place different elements of the rule in different lines, since this will mark the anonymous instances of the objects with the unique numbers of source lines.\
\
Yet, the parser was not created to check the text for its compliance with MQL syntax, but to extract semantic data. Let us try to do that.\
\
### Practice, Part 3: Listing the Methods of Classes and Hierarchy of Classes\
\
As the first application task based on MQL parsing, let us list all methods of classes. For this purpose, let us define a class that implements interface Callback and record the relevant "productions."\
\
In principle, it would be more logical to parse based on a syntax tree. However, that would overload the memory for storing the tree and a separate algorithm to iterate over that tree. However, in fact, the parser itself already iterates over it in the same sequence while parsing the text (since it is this sequence, in which the tree would be built if the relevant mode had been enabled). Therefore, it is easier to parse on-the-go.\
\
MQL grammar has the following rule:\
\
```\
  _DECLARE(method) = R(template_decl)*0 + R(method_specifiers)*0 + R(type) + R(name_with_arg_list) + R(modifiers)*0;\
```\
\
It consists of many other non-terminals that, in turn, are revealed through other non-terminals, so the syntax tree of the method is highly branched. In the production processor, we will intercept all fragments relating to non-terminal 'method' and put them into one common string. At the moment where the next production turns out to be for another non-terminal, it means that the method description is over, and the result can be displayed in the log.\
\
```\
class MyCallback: public Callback\
{\
    virtual void produce(const Parser *parser, const Terminal *p, const int index, const Token *t, const Source *context, const int level) override\
    {\
      static string method = "";\
\
      // collect all tokens from `method` nonterminal\
      if(p.getName() == "method")\
      {\
        method += t.content(context) + " ";\
      }\
      // as soon as other [non]terminal is detected and string is filled, signature is ready\
      else if(method != "")\
      {\
        Print(method);\
        method = "";\
      }\
    }\
```\
\
To connect our processor to the parser, let us extend the test script in the following manner (in OnStart):\
\
```\
  MyCallback myc;\
  Parser parser(tokens, loader.text(), PrintCST);\
  parser.setCallback(&myc);\
```\
\
In addition to the list of methods, let us collect information about the declaration of classes — it will be particularly required to identify the context, in which the methods are defined, but we can also be able to build the derivation hierarchy.\
\
To store metadata regarding a random class, let us prepare class named 'Class' ;-).\
\
```\
  class Class\
  {\
    private:\
      BaseArray<Class *> subclasses;\
      Class *superclass;\
      string name;\
\
    public:\
      Class(const string n): name(n), superclass(NULL)\
      {\
      }\
\
      ~Class()\
      {\
        subclasses.clear();\
      }\
\
      void addSubclass(Class *derived)\
      {\
        derived.superclass = &this;\
        subclasses.add(derived);\
      }\
\
      bool hasParent() const\
      {\
        return superclass != NULL;\
      }\
\
      Class *operator[](int i) const\
      {\
        return subclasses[i];\
      }\
\
      int size() const\
      {\
        return subclasses.size();\
      }\
      ...\
   };\
```\
\
It has an array of subclasses and a link to a superclass. Method addSubclass is responsible for filling those interrelated fields. We will put the instances of the Class objects into a map with the string key represented as the class name:\
\
```\
  Map<string,Class *> map;\
```\
\
The map is in the same object of MyCallback.\
\
Now we can expand method 'produce' from interface Callback. To collect tokens relating to the declaration of class, we will have to intercept a bit more rules, since we need not just the complete declaration, but that with the specific properties highlighted: Name of the new class, types of its template, if any, the name of the base class, and the types of its template, if any.\
\
Let us add the relevant variables to collect data (Attention! Classes in MQL can be nested classes, although not frequently, but we do not consider that for simplicity! At the same time, our MQL grammar does support that).\
\
```\
      static string templName = "";\
      static string templBaseName = "";\
      static string className = "";\
      static string baseName = "";\
```\
\
In the context of non-terminal 'template\_decl', identifiers are template types:\
\
```\
      if(p.getName() == "template_decl" && t.getType() == IDENTIFIER)\
      {\
        if(templName != "") templName += ",";\
        templName += t.content(context);\
      }\
```\
\
The relevant grammar rules for 'template\_decl', as well as for the objects used below, can be studied in the source codes attached.\
\
In the context of non-terminal 'class\_name', identifier is the class name; if templName has not been an empty string by that time, then these are template types that should be added into the description:\
\
```\
      if(p.getName() == "class_name" && t.getType() == IDENTIFIER)\
      {\
        className = t.content(context);\
        if(templName != "")\
        {\
          className += "<" + templName + ">";\
          templName = "";\
        }\
      }\
```\
\
The first identifier in the context of 'derived\_clause', if any, is the name of the base class.\
\
```\
      if(p.getName() == "derived_clause" && t.getType() == IDENTIFIER)\
      {\
        if(baseName == "") baseName = t.content(context);\
        else\
        {\
          if(templBaseName != "") templBaseName += ",";\
          templBaseName += t.content(context);\
        }\
      }\
```\
\
All subsequent identifiers are the template types of the base class.\
\
As soon as the class declaration is completed, the rule 'class\_decl' triggers in the grammar. By that moment, all data has already been collected and can be added to the map of classes.\
\
```\
      if(p.getName() == "class_decl") // finalization\
      {\
        if(className != "")\
        {\
          if(map[className] == NULL)\
          {\
            map.put(className, new Class(className));\
          }\
          else\
          {\
            // Class already defined, maybe forward declaration\
          }\
        }\
\
        if(baseName != "")\
        {\
          if(templBaseName != "")\
          {\
            baseName += "<" + templBaseName + ">";\
          }\
          Class *base = map[baseName];\
          if(base == NULL)\
          {\
            // Unknown class, maybe not included, but strange\
            base = new Class(baseName);\
            map.put(baseName, base);\
          }\
\
          if(map[className] == NULL)\
          {\
            Print("Error: base name `", baseName, "` resolved before declaration of the class: ", className);\
          }\
          else\
          {\
            base.addSubclass(map[className]);\
          }\
\
          baseName = "";\
        }\
        className = "";\
        templName = "";\
        templBaseName = "";\
      }\
```\
\
In the end, we clear all strings and wait for the next declarations to appear.\
\
Upon the successful parsing of a program text, it remains to display the hierarchy of classes in any convenient way. In the test script, class MyCallback provides function 'print' to display in the log. It, in turn, uses method 'print' in the objects of class 'Class'. We are going to leave these helper algorithms for independent reading and, besides, as one more small programming problem for those wishing to try their strengths (such competitions often appear spontaneously on the forum of mql5.com). The existing implementation is purely pragmatic and does not pretend to be optimal. It simply ensures the goal: Displaying the tree-structured hierarchy of the Class-type objects in the log as starkly as possible. However, this can be done in a more efficient manner.\
\
Let us check how the test script works to analyze some MQL projects. Hereinafter, let us set parameter TestMode = MQL.\
\
For example, for the standard Expert Advisor 'MACD Sample.mq5', having set SourceFile = Sources/Experts/Examples/MACD/MACD Sample.mq5 and LoadIncludes = true, that is, with all the dependencies, we will obtain the following result (the list of methods is shortened largely):\
\
```\
Processing Sources/Experts/Examples/MACD/MACD Sample.mq5\
Scanning...\
#include <Trade\Trade.mqh>\
Including Sources/Include/Trade\Trade.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
#include "StdLibErr.mqh"\
Including Sources/Include/StdLibErr.mqh\
#include "OrderInfo.mqh"\
Including Sources/Include/Trade/OrderInfo.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
#include "SymbolInfo.mqh"\
Including Sources/Include/Trade/SymbolInfo.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
#include "PositionInfo.mqh"\
Including Sources/Include/Trade/PositionInfo.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
#include "SymbolInfo.mqh"\
Including Sources/Include/Trade/SymbolInfo.mqh\
#include <Trade\PositionInfo.mqh>\
Including Sources/Include/Trade\PositionInfo.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
#include "SymbolInfo.mqh"\
Including Sources/Include/Trade/SymbolInfo.mqh\
Files processed: 8\
Source length: 175860\
File map:\
Sources/Experts/Examples/MACD/MACD Sample.mq5 0\
Sources/Include/Trade\Trade.mqh 900\
Sources/Include/Object.mqh 1277\
Sources/Include/StdLibErr.mqh 1657\
Sources/Include/Object.mqh 2330\
Sources/Include/Trade\Trade.mqh 3953\
Sources/Include/Trade/OrderInfo.mqh 4004\
Sources/Include/Trade/SymbolInfo.mqh 4407\
Sources/Include/Trade/OrderInfo.mqh 38837\
Sources/Include/Trade\Trade.mqh 59925\
Sources/Include/Trade/PositionInfo.mqh 59985\
Sources/Include/Trade\Trade.mqh 75648\
Sources/Experts/Examples/MACD/MACD Sample.mq5 143025\
Sources/Include/Trade\PositionInfo.mqh 143091\
Sources/Experts/Examples/MACD/MACD Sample.mq5 158754\
Lines: 4170\
Tokens: 18005\
Defining grammar...\
Parsing...\
CObject :: CObject * Prev ( void ) const\
CObject :: void Prev ( CObject * node )\
CObject :: CObject * Next ( void ) const\
CObject :: void Next ( CObject * node )\
CObject :: virtual bool Save ( const int file_handle )\
CObject :: virtual bool Load ( const int file_handle )\
CObject :: virtual int Type ( void ) const\
CObject :: virtual int Compare ( const CObject * node , const int mode = 0 ) const\
CSymbolInfo :: string Name ( void ) const\
CSymbolInfo :: bool Name ( const string name )\
CSymbolInfo :: bool Refresh ( void )\
CSymbolInfo :: bool RefreshRates ( void )\
\
...\
\
CSampleExpert :: bool Init ( void )\
CSampleExpert :: void Deinit ( void )\
CSampleExpert :: bool Processing ( void )\
CSampleExpert :: bool InitCheckParameters ( const int digits_adjust )\
CSampleExpert :: bool InitIndicators ( void )\
CSampleExpert :: bool LongClosed ( void )\
CSampleExpert :: bool ShortClosed ( void )\
CSampleExpert :: bool LongModified ( void )\
CSampleExpert :: bool ShortModified ( void )\
CSampleExpert :: bool LongOpened ( void )\
CSampleExpert :: bool ShortOpened ( void )\
Success\
Class hierarchy:\
\
CObject\
  ^\
  +--CSymbolInfo\
  +--COrderInfo\
  +--CPositionInfo\
  +--CTrade\
  +--CPositionInfo\
\
CSampleExpert\
```\
\
Now, let us try a third-party project. I took EA 'SlidingPuzzle2' from [this article](https://www.mql5.com/en/articles/1998). I placed it at: "SourceFile = Sources/Experts/Examples/Layouts/SlidingPuzzle2.mq5". Having included all header files (LoadIncludes = true), we will obtain the following result (shortened):\
\
```\
Processing Sources/Experts/Examples/Layouts/SlidingPuzzle2.mq5\
Scanning...\
#include "SlidingPuzzle2.mqh"\
Including Sources/Experts/Examples/Layouts/SlidingPuzzle2.mqh\
#include <Layouts\GridTk.mqh>\
Including Sources/Include/Layouts\GridTk.mqh\
#include "Grid.mqh"\
Including Sources/Include/Layouts/Grid.mqh\
#include "Box.mqh"\
Including Sources/Include/Layouts/Box.mqh\
#include <Controls\WndClient.mqh>\
Including Sources/Include/Controls\WndClient.mqh\
#include "WndContainer.mqh"\
Including Sources/Include/Controls/WndContainer.mqh\
#include "Wnd.mqh"\
Including Sources/Include/Controls/Wnd.mqh\
#include "Rect.mqh"\
Including Sources/Include/Controls/Rect.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
#include "StdLibErr.mqh"\
Including Sources/Include/StdLibErr.mqh\
#include "Scrolls.mqh"\
Including Sources/Include/Controls/Scrolls.mqh\
#include "WndContainer.mqh"\
Including Sources/Include/Controls/WndContainer.mqh\
#include "Panel.mqh"\
Including Sources/Include/Controls/Panel.mqh\
#include "WndObj.mqh"\
Including Sources/Include/Controls/WndObj.mqh\
#include "Wnd.mqh"\
Including Sources/Include/Controls/Wnd.mqh\
#include <Controls\Edit.mqh>\
Including Sources/Include/Controls\Edit.mqh\
#include "WndObj.mqh"\
Including Sources/Include/Controls/WndObj.mqh\
#include <ChartObjects\ChartObjectsTxtControls.mqh>\
Including Sources/Include/ChartObjects\ChartObjectsTxtControls.mqh\
#include "ChartObject.mqh"\
Including Sources/Include/ChartObjects/ChartObject.mqh\
#include <Object.mqh>\
Including Sources/Include/Object.mqh\
Files processed: 17\
Source length: 243134\
File map:\
Sources/Experts/Examples/Layouts/SlidingPuzzle2.mq5 0\
Sources/Experts/Examples/Layouts/SlidingPuzzle2.mqh 493\
Sources/Include/Layouts\GridTk.mqh 957\
Sources/Include/Layouts/Grid.mqh 1430\
Sources/Include/Layouts/Box.mqh 1900\
Sources/Include/Controls\WndClient.mqh 2377\
Sources/Include/Controls/WndContainer.mqh 2760\
Sources/Include/Controls/Wnd.mqh 3134\
Sources/Include/Controls/Rect.mqh 3509\
Sources/Include/Controls/Wnd.mqh 14312\
Sources/Include/Object.mqh 14357\
Sources/Include/StdLibErr.mqh 14737\
Sources/Include/Object.mqh 15410\
Sources/Include/Controls/Wnd.mqh 17033\
Sources/Include/Controls/WndContainer.mqh 46214\
Sources/Include/Controls\WndClient.mqh 61689\
Sources/Include/Controls/Scrolls.mqh 61733\
Sources/Include/Controls/Panel.mqh 62137\
Sources/Include/Controls/WndObj.mqh 62514\
Sources/Include/Controls/Panel.mqh 72881\
Sources/Include/Controls/Scrolls.mqh 78251\
Sources/Include/Controls\WndClient.mqh 103907\
Sources/Include/Layouts/Box.mqh 115349\
Sources/Include/Layouts/Grid.mqh 126741\
Sources/Include/Layouts\GridTk.mqh 131057\
Sources/Experts/Examples/Layouts/SlidingPuzzle2.mqh 136066\
Sources/Include/Controls\Edit.mqh 136126\
Sources/Include/ChartObjects\ChartObjectsTxtControls.mqh 136555\
Sources/Include/ChartObjects/ChartObject.mqh 137079\
Sources/Include/ChartObjects\ChartObjectsTxtControls.mqh 177423\
Sources/Include/Controls\Edit.mqh 213551\
Sources/Experts/Examples/Layouts/SlidingPuzzle2.mqh 221772\
Sources/Experts/Examples/Layouts/SlidingPuzzle2.mq5 241539\
Lines: 6102\
Tokens: 27248\
Defining grammar...\
Parsing...\
CRect :: CPoint LeftTop ( void ) const\
CRect :: void LeftTop ( const int x , const int y )\
CRect :: void LeftTop ( const CPoint & point )\
\
...\
\
CSlidingPuzzleDialog :: virtual bool Create ( const long chart , const string name , const int subwin , const int x1 , const int y1 , const int x2 , const int y2 )\
CSlidingPuzzleDialog :: virtual bool OnEvent ( const int id , const long & lparam , const double & dparam , const string & sparam )\
CSlidingPuzzleDialog :: void Difficulty ( int d )\
CSlidingPuzzleDialog :: virtual bool CreateMain ( const long chart , const string name , const int subwin )\
CSlidingPuzzleDialog :: virtual bool CreateButton ( const int button_id , const long chart , const string name , const int subwin )\
CSlidingPuzzleDialog :: virtual bool CreateButtonNew ( const long chart , const string name , const int subwin )\
CSlidingPuzzleDialog :: virtual bool CreateLabel ( const long chart , const string name , const int subwin )\
CSlidingPuzzleDialog :: virtual bool IsMovable ( CButton * button )\
CSlidingPuzzleDialog :: virtual bool HasNorth ( CButton * button , int id , bool shuffle = false )\
CSlidingPuzzleDialog :: virtual bool HasSouth ( CButton * button , int id , bool shuffle = false )\
CSlidingPuzzleDialog :: virtual bool HasEast ( CButton * button , int id , bool shuffle = false )\
CSlidingPuzzleDialog :: virtual bool HasWest ( CButton * button , int id , bool shuffle = false )\
CSlidingPuzzleDialog :: virtual void Swap ( CButton * source )\
Success\
Class hierarchy:\
\
CPoint\
\
CSize\
\
CRect\
\
CObject\
  ^\
  +--CWnd\
  |    ^\
  |    +--CDragWnd\
  |    +--CWndContainer\
  |    |    ^\
  |    |    +--CScroll\
  |    |    |    ^\
  |    |    |    +--CScrollV\
  |    |    |    +--CScrollH\
  |    |    +--CWndClient\
  |    |         ^\
  |    |         +--CBox\
  |    |              ^\
  |    |              +--CGrid\
  |    |                   ^\
  |    |                   +--CGridTk\
  |    +--CWndObj\
  |         ^\
  |         +--CPanel\
  |         +--CEdit\
  +--CGridConstraints\
  +--CChartObject\
       ^\
       +--CChartObjectText\
            ^\
            +--CChartObjectLabel\
                 ^\
                 +--CChartObjectEdit\
                 |    ^\
                 |    +--CChartObjectButton\
                 +--CChartObjectRectLabel\
\
CAppDialog\
  ^\
  +--CSlidingPuzzleDialog\
```\
\
Here, hierarchy of classes is more interesting.\
\
Although I have tested the parser on different projects, it will more than likely "stub toe" on certain programs. One of the problems not yet solved in it is related to processing macros. As it has already been said above, correct analysis suggests a dynamic interpretation and expanding of macros with substituting the results into the source code before the parsing starts.\
\
In the current MQL grammar, we have tried to define calling the macros as a less strict call of functions. However, it does not always work by far.\
\
For example, in library [TypeToBytes](https://www.mql5.com/en/code/16280), the parameters of macros are used to generate metatypes. Here is one of the cases:\
\
```\
#define _C(A, B) CASTING<A>::Casting(B)\
```\
\
Further, this macro is used as follows:\
\
```\
Res = _C(STRUCT_TYPE<T1>, Tmp);\
```\
\
If we try to run the parser on this code, it will be unable to "digest" STRUCT\_TYPE<T1>, since in the reality this parameter represents a templated type, while the parser implies a value or, more broadly speaking, an expression (and particularly interprets characters '<' and '>' as comparators). Now, a similar problem will be created by calls of macros, after which there is no semicolon. However, we can circumvent it having inserted ';' into the source code being processed.\
\
Those who want it can perform experiment number 3 (the first two ones were mentioned in the beginning of this article), which would consist in searching for an iteration algorithm to modify the current grammar with such macros rules that would allow you to successfully parse such complicated cases.\
\
### Conclusions\
\
We have considered the generally and technologically simplest way of data parsing, including analyzing the source codes written in MQL. For this purpose, the grammar of the MQL language has been presented, as well as the implementation of standard tools, i.e., parser and scanner. Using them to obtain the source code structure allows calculating their statistics, identifying the quality indicators, showing dependencies, and automatically changing the formats.\
\
At the same time, the implementation presented here requires some improvements to reach 100% compatibility with complex MQL projects, particularly in terms of supporting expansion of the macros.\
\
In case of deeper preparation, such as saving information on the entities found in the table of names, this approach would also allow performing code generation, controlling typical errors, and performing other, more complicated tasks.\
\
Translated from Russian by MetaQuotes Ltd.\
\
Original article: [https://www.mql5.com/ru/articles/5638](https://www.mql5.com/ru/articles/5638)\
\
**Attached files** \|\
\
\
[Download ZIP](https://www.mql5.com/en/articles/download/5638.zip "Download all attachments in the single ZIP archive")\
\
[MQL5PRSR.zip](https://www.mql5.com/en/articles/download/5638/mql5prsr.zip "Download MQL5PRSR.zip")(34.09 KB)\
\
**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.\
\
This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.\
\
#### Other articles by this author\
\
- [Backpropagation Neural Networks using MQL5 Matrices](https://www.mql5.com/en/articles/12187)\
- [Parallel Particle Swarm Optimization](https://www.mql5.com/en/articles/8321)\
- [Custom symbols: Practical basics](https://www.mql5.com/en/articles/8226)\
- [Calculating mathematical expressions (Part 2). Pratt and shunting yard parsers](https://www.mql5.com/en/articles/8028)\
- [Calculating mathematical expressions (Part 1). Recursive descent parsers](https://www.mql5.com/en/articles/8027)\
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs (Part 3). Form Designer](https://www.mql5.com/en/articles/7795)\
- [MQL as a Markup Tool for the Graphical Interface of MQL Programs. Part 2](https://www.mql5.com/en/articles/7739)\
\
**Last comments \|**\
**[Go to discussion](https://www.mql5.com/en/forum/308538)**\
(14)\
\
\
![David_NZ](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[David\_NZ](https://www.mql5.com/en/users/david_nz)**\
\|\
4 Feb 2021 at 04:26\
\
Hi, I am a relative newbie here.\
\
Having read the article assumed the RESULT would be a search tool to look up code generated by the wizard.\
\
Downloaded the zip file. Installed the script mql.mq5 compiled it without errors.\
\
Running mql.ex5 produces no result on screen.\
\
I am in learn mode. Want to be able to use the wizard to use code out of the codebase then analyze and modify to get a usable EA.\
\
Any help would be appreciated.\
\
Thanks\
\
![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)\
\
**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**\
\|\
4 Feb 2021 at 11:54\
\
**David\_NZ:**\
\
Hi, I am a relative newbie here.\
\
Having read the article assumed the RESULT would be a search tool to look up code generated by the wizard.\
\
Downloaded the zip file. Installed the script mql.mq5 compiled it without errors.\
\
Running mql.ex5 produces no result on screen.\
\
I am in learn mode. Want to be able to use the wizard to use code out of the codebase then analyze and modify to get a usable EA.\
\
Any help would be appreciated.\
\
Thanks\
\
This is not "a search tool to look up code generated by the wizard". I wrote the article in Russian, and did not proofread its English translation made by MQ (they do not practice that), but I hope the article contains all details and is clear enough on what the presented scripts can do. Specifically MQL analysis is demonstrated by extraction of class hierarchy and methods from sources. I do not understand which wizard you mean.\
\
You should explain exactly what you did (including some preparations on the system level required to run the script), and what you get. Which result on screen do you expect? The script outputs results in the log.\
\
Since you have the [source code](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "MQL5 Algo Forge: Cloud Workspace for Algorithmic Trading Development"), you can debug the script and find out what happens line by line.\
\
![David_NZ](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[David\_NZ](https://www.mql5.com/en/users/david_nz)**\
\|\
5 Feb 2021 at 02:41\
\
Hi Stan,\
\
Thanks for taking the time to reply.\
\
Was not aware of where the Print commands in a script writes to a log file.\
\
Yes there is a log file produced when I run the script.\
\
I am constantly frustrated in looking at code created by (Wizard in MetaEditor new doc)\
\
Calls to functions not in the [opened file](https://www.mql5.com/en/docs/files/fileopen "MQL5 documentation: FileOpen function") #includes then down more #include levels.\
\
Was hoping your script would have a search box and then tell me which \*.mqh file the function was hidden in.\
\
As this script is not a search facility I will not take up more of your time.\
\
Thanks\
\
![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)\
\
**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**\
\|\
5 Feb 2021 at 14:31\
\
**David\_NZ:**\
\
Hi Stan,\
\
Thanks for taking the time to reply.\
\
Was not aware of where the Print commands in a script writes to a log file.\
\
Yes there is a log file produced when I run the script.\
\
I am constantly frustrated in looking at code created by (Wizard in MetaEditor new doc)\
\
Calls to functions not in the [opened file](https://www.mql5.com/en/docs/files/fileopen "MQL5 documentation: FileOpen function") #includes then down more #include levels.\
\
Was hoping your script would have a search box and then tell me which \*.mqh file the function was hidden in.\
\
As this script is not a search facility I will not take up more of your time.\
\
Thanks\
\
The script does parse the source code and follow all nested includes, but it does not provide a ready-made solution to show you which header file contains specific function - for this purpose the script should be customized for your needs.\
\
You still did not make clear what is your requirement/use-case.\
\
If you have just cretaed a new program by MQL wizard and want to drill down all its sources including the dependencies (header files), then you can easily do it right from MetaEditor: right click any identifier by mouse to open the context menu, then click "Go to definition" (Alt+G) - this will open required file and show you where the function or variable comes from.\
\
![David_NZ](https://c.mql5.com/avatar/avatar_na2.png)\
\
**[David\_NZ](https://www.mql5.com/en/users/david_nz)**\
\|\
8 Feb 2021 at 06:03\
\
Hi Stan, Your last paragraph was the solution I needed.\
\
Thanks.\
\
![Studying candlestick analysis techniques (Part II): Auto search for new patterns](https://c.mql5.com/2/35/Pattern_I__3.png)[Studying candlestick analysis techniques (Part II): Auto search for new patterns](https://www.mql5.com/en/articles/5630)\
\
In the previous article, we analyzed 14 patterns selected from a large variety of existing candlestick formations. It is impossible to analyze all the patterns one by one, therefore another solution was found. The new system searches and tests new candlestick patterns based on known candlestick types.\
\
![The power of ZigZag (part II). Examples of receiving, processing and displaying data](https://c.mql5.com/2/35/MQL5-avatar-zigzag_head__1.png)[The power of ZigZag (part II). Examples of receiving, processing and displaying data](https://www.mql5.com/en/articles/5544)\
\
In the first part of the article, I have described a modified ZigZag indicator and a class for receiving data of that type of indicators. Here, I will show how to develop indicators based on these tools and write an EA for tests that features making deals according to signals formed by ZigZag indicator. As an addition, the article will introduce a new version of the EasyAndFast library for developing graphical user interfaces.\
\
![MetaTrader 5 and Python integration: receiving and sending data](https://c.mql5.com/2/35/mt5-3002__1.png)[MetaTrader 5 and Python integration: receiving and sending data](https://www.mql5.com/en/articles/5691)\
\
Comprehensive data processing requires extensive tools and is often beyond the sandbox of one single application. Specialized programming languages are used for processing and analyzing data, statistics and machine learning. One of the leading programming languages for data processing is Python. The article provides a description of how to connect MetaTrader 5 and Python using sockets, as well as how to receive quotes via the terminal API.\
\
![Studying candlestick analysis techniques (part I): Checking existing patterns](https://c.mql5.com/2/35/Pattern_I__2.png)[Studying candlestick analysis techniques (part I): Checking existing patterns](https://www.mql5.com/en/articles/5576)\
\
In this article, we will consider popular candlestick patterns and will try to find out if they are still relevant and effective in today's markets. Candlestick analysis appeared more than 20 years ago and has since become quite popular. Many traders consider Japanese candlesticks the most convenient and easily understandable asset price visualization form.\
\
[Running robots on virtual hosting is easyFollow our step-by-step MetaTrader VPS guide for beginnersRead![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/01.png)![](https://www.mql5.com/ff/sh/au4fqg4kms7s9mq1z2/02.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=uzpprdshbcrtxvjxpmescehprypbymxc&s=516438f25b531570d9b7d49dcfb29c82fa1021f5ede6571df8026dbfbafcd13f&uid=&ref=https://www.mql5.com/en/articles/5638&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5071697348553288895)\
\
This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).\
\
![close](https://c.mql5.com/i/close.png)\
\
![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)\
\
You are missing trading opportunities:\
\
- Free trading apps\
- Over 8,000 signals for copying\
- Economic news for exploring financial markets\
\
RegistrationLog in\
\
latin characters without spaces\
\
a password will be sent to this email\
\
An error occurred\
\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)\
\
You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)\
\
If you do not have an account, please [register](https://www.mql5.com/en/auth_register)\
\
Allow the use of cookies to log in to the MQL5.com website.\
\
Please enable the necessary setting in your browser, otherwise you will not be able to log in.\
\
[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)\
\
- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)