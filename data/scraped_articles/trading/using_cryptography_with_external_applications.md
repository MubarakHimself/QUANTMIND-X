---
title: Using cryptography with external applications
url: https://www.mql5.com/en/articles/8093
categories: Trading, Trading Systems, Integration
relevance_score: 0
scraped_at: 2026-01-24T13:34:09.117895
---

[We've created a channel for MQL5 developersFollow MQL5.community on social media and be the first to receive important updatesLearn more![](https://www.mql5.com/ff/sh/a83xrgctr82w45z9z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=pwgdbvtemvkwsfltqonysypfvtrtufji&s=e99a66a1660cd810b1edbac65597df695e2c2220d1e937834f402f9aeabd4289&uid=&ref=https://www.mql5.com/en/articles/8093&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082959126465614367)

MetaTrader 5 / Trading


### Introduction

Cryptography is rarely used in MQL programs. There are not so many opportunities in everyday trading to use cryptography. An exception would be a paranoid signal copier wishing to protect the sent data from listening, and, perhaps, that's all. When data does not leave the terminal, it is very difficult to imagine why one would need to encrypt/decrypt it. Furthermore, this may indicate low competence of the developer, who creates an extra terminal load.

Maybe there is no need to use cryptography in trading? Actually, there is. For example, consider licensing. There can be a small company or even a separate developer whose products are popular. Licensing issues are relevant in this case, and therefore license encryption/description is required.

It is possible to specify in the license user data and an editable list of products. An indicator or an Expert Advisor starts operation, checks the availability of a license and its expiration for the given product. A program sends a request to the server, updates the license if necessary or receives a new one. This can be not the most efficient and safest route, but we will use it in this article for demonstration purposes. Obviously, in this case, the license will be read/written by different software tools — a terminal, a remote server, control modules and logging modules. All they can be written by different people, at different times and in different languages.

The purpose of this article is to study the encryption/decryption modes, in which an object encrypted by a program in C# or C++ can be decrypted by the MetaTrader terminal and vice versa.

The article is intended both for intermediate-skilled programmers and for beginners.

### Setting a Task

This has already been mentioned in the introduction. We will try to simulate a solution to a real problem, requiring the creation, encryption and decryption of a license for several products — indicators and Expert Advisors. It is not important for us which program encrypts and which one decrypts the license. For example, a license can be initially created on the developer's computer then it is corrected in the sales department and decrypted on the trader's computer. The process must be free from errors associated with poorly configured algorithms.

Along with the solution of the major task, we will consider the complex problem of licensing. This will not be a ready to use license, but one of the possible variants, which should be further edited and developed.

### Source Data

Let us refer to the terminal documentation to obtain the source data for operation. There are two standard functions responsible for encryption/decryption procedures:

```
int  CryptEncode(
   ENUM_CRYPT_METHOD   method,        // conversion method
   const uchar&        data[],        // source array
   const uchar&        key[],         // encryption key
   uchar&              result[]       // destination array
   );

int  CryptDecode(
   ENUM_CRYPT_METHOD   method,        // conversion method
   const uchar&        data[],        // source array
   const uchar&        key[],         // encryption key
   uchar&              result[]       // destination array
   );
```

The specific method used for encryption/decryption is determined by the [**method**](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants#enum_crypt_method) argument. In this case, we are interested in three values which the **method** argument can have: **CRYPT\_AES128, CRYPT\_AES256,** **CRYPT\_DES**. These three values represent symmetric encryption algorithms with different key lengths.

In this article, we will only use one of them, **CRYPT\_AES128**. This is a symmetric block cipher algorithm with a 128-bit (16 byte) key. The usage of other algorithms is similar.

The AES algorithm (not only the selected one, but also those having another key length) have some important settings which are not provided in the above functions. These include the encryption mode and **padding**. We will not go deep into details regarding these terms. So, the terminal uses the **Electronic Codebook (ECB)** encryption mode and **Padding** equal to zero. I would like to thank fellow traders, as this was explained in the [MQL5 forum](https://www.mql5.com/en/forum/214528#comment_6670526). Now our task will be much easier to solve.

### Developing an Operation Object

Since we consider encryption/decryption as applied to licensing, our operation object is the license. This should be some kind of construction containing information about various products to which the license applies. The following data is required here:

1. License expiration for this product.
2. Product name.

Let us create an appropriate structure with the accompanying simplest methods:

```
#define PRODMAXLENGTH 255

struct ea_user
  {
   ea_user() {expired = -1;}
   datetime expired;                //License expiration (-1 - unlimited)
   int      namelength;             //Product name length
   char    uname[PRODMAXLENGTH];    //Product name
   void SetEAname(string name)
     {
      namelength = StringToCharArray(name, uname);
     }
   string GetEAname()
     {
      return CharArrayToString(uname, 0, namelength);
     }
   bool IsExpired()
     {
      if (expired == -1)
         return false; // NOT expired
      return expired <= TimeLocal();
     }
  };//struct ea_user
```

Here some explanations:

- The product name is stored as a fixed-length array.

- The maximum product length name is limited to **PRODMAXLENGTH**.
- This strategy can be easily packed into a byte array — this is what we are going to do before encrypting the entire object.

However, this structure alone is not enough. Obviously, the license must contain user details. The information could be included in the already described structure, but this would be inefficient as the user may have multiple product licenses. A more reasonable solution is to create a separate structure for the user and to add to it the required number of product licenses. Thus, a user will have one license containing permissions and restrictions for all licensed products.

Information that can be contained in the structure describing the user:

1. The unique user ID. The name could also be saved, but it seems undesirable to send personal data every time, even in an encrypted form.
2. Information about the user's accounts on which the products can be used.
3. User's license expiration date. This field can limit the usage of all existing products, even unlimited ones, to the time of service of the user as such.
4. The number of licensed products in the user terminal:

```
#define COUNTACC 5

struct user_lic {
   user_lic() {
      uid       = -1;
      log_count =  0;
      ea_count  =  0;
      expired   = -1;
      ArrayFill(logins, 0, COUNTACC, 0);
   }
   long uid;                       //User ID
   datetime expired;               //End of user service (-1 - unlimited)
   int  log_count;                 //The number of the user's accounts
   long logins[COUNTACC];          //User's accounts
   int  ea_count;                  //The number of licensed products
   bool AddLogin(long lg){
      if (log_count >= COUNTACC) return false;
      logins[log_count++] = lg;
      return true;
   }
   long GetLogin(int num) {
      if (num >= log_count) return -1;
      return logins[num];
   }
   bool IsExpired() {
      if (expired == -1) return false; // NOT expired
      return expired <= TimeLocal();
   }
};//struct user_lic
```

Here are some clarifications:

- User accounts are stored in a fixed-length array. They are represented by account numbers. Although, if necessary, the information can be easily supplemented with server or broker name and the number of activations for a particular account.

As for now, this structure contains sufficient information about the user and the relevant products. Each of them is a type, an instance of which can be processed by the **StructToCharArray** function.

Now, we need to serialize the structure data into a byte array, which can further be encrypted. This will be implemented as follows:

- Create and initialize an instance of the user\_lic structure.

- Serialize it in a byte array.
- Create and initialize one or more instance of the ea\_user structure.
- Serialize them in the same byte array, increasing its size and adjusting the ea\_count field.

Create a class to perform these operations:

```
class CLic {

public:

   static int iSizeEauser;
   static int iSizeUserlic;

   CLic() {}
  ~CLic() {}

   int SetUser(const user_lic& header){
      Reset();
      if (!StructToCharArray(header, dest) ) return 0;
      return ArraySize(dest);
   }//int SetUser(user_lic& header)

   int AddEA(const ea_user& ea) {
      int c = ArraySize(dest);
      if (c == 0) return 0;
      uchar tmp[];
      if (!StructToCharArray(ea, tmp) ) return 0;
      ArrayInsert(dest, tmp, c);
      return ArraySize(dest);
   }//int AddEA(ea_user& ea)

   bool GetUser(user_lic& header) const {
      if (ArraySize(dest) < iSizeUserlic) return false;
      return CharArrayToStruct(header, dest);
   }//bool GetUser(user_lic& header)

   //num - 0 based
   bool GetEA(int num, ea_user& ea) const {
      int index = iSizeUserlic + num * iSizeEauser;
      if (ArraySize(dest) < index + iSizeEauser) return false;
      return CharArrayToStruct(ea, dest, index);
   }//bool GetEA(int num, ea_user& ea)

   int Encode(ENUM_CRYPT_METHOD method, string key, uchar&  buffer[]) const {
      if (ArraySize(dest) < iSizeUserlic) return 0;
      if(!IsKeyCorrect(method, key) ) return 0;
      uchar k[];
      StringToCharArray(key, k);
      return CryptEncode(method, dest, k, buffer);
   }

   int Decode(ENUM_CRYPT_METHOD method, string key, uchar&  buffer[]) {
      Reset();
      if(!IsKeyCorrect(method, key) ) return 0;
      uchar k[];
      StringToCharArray(key, k);
      return CryptDecode(method, buffer, k, dest);
   }

protected:
   void Reset() {ArrayResize(dest, 0);}

   bool IsKeyCorrect(ENUM_CRYPT_METHOD method, string key) const {
      int len = StringLen(key);
      switch (method) {
         case CRYPT_AES128:
            if (len == 16) return true;
            break;
         case CRYPT_AES256:
            if (len == 32) return true;
            break;
         case CRYPT_DES:
            if (len == 7) return true;
            break;
      }
#ifdef __DEBUG_USERMQH__
   Print("Key length is incorrect: ",len);
#endif
      return false;
   }//bool IsKeyCorrect(ENUM_CRYPT_METHOD method, string key)

private:
   uchar dest[];
};//class CLic

   static int CLic::iSizeEauser  = sizeof(ea_user);
   static int CLic::iSizeUserlic = sizeof(user_lic);
```

Two functions have been added to the class, enabling encryption and decryption, and a protected function for checking the key length. You can see from its code, that, for example, the key length for the **CRYPT\_AES128** method must be equal to 16 bytes. Actually, it must not be less than 16 bytes. Probably, further it is somehow normalized, which is hidden from the developer. We will not rely on this and will strictly set the required key length.

Finally, it is already possible to encrypt the resulting byte array and save it to a binary file. This file should be stored in the **File** folder of the terminal, according to the general rules. If necessary, it can be read and decrypted:

```
bool CreateLic(ENUM_CRYPT_METHOD method, string key, CLic& li, string licname) {
   uchar cd[];
   if (li.Encode(method, key, cd) == 0) return false;
   int h = FileOpen(licname, FILE_WRITE | FILE_BIN);
   if (h == INVALID_HANDLE) {
#ifdef __DEBUG_USERMQH__
      Print("File create failed: ",licname);
#endif
      return false;
   }
   FileWriteArray(h, cd);
   FileClose(h);
#ifdef __DEBUG_USERMQH__
   li.SaveArray();
#endif
   return true;
}// bool CreateLic(ENUM_CRYPT_METHOD method, string key, const CLic& li, string licname)

bool ReadLic(ENUM_CRYPT_METHOD method, string key, CLic& li, string licname) {
   int h = FileOpen(licname, FILE_READ | FILE_BIN);
   if (h == INVALID_HANDLE) {
#ifdef __DEBUG_USERMQH__
      Print("File open failed: ",licname);
#endif
      return false;
   }
   uchar cd[];
   FileReadArray(h,cd);
   if (ArraySize(cd) < CLic::iSizeUserlic) {
#ifdef __DEBUG_USERMQH__
      Print("File too small: ",licname);
#endif
      return false;
   }
   li.Decode(method, key, cd);
   FileClose(h);
   return true;
}// bool ReadLic(ENUM_CRYPT_METHOD method, string key, CLic& li, string licname)
```

Both functions are clear and do not require additional explanation. The attached CryptoMQL.zip array contains two scripts and a library file, which implements encryption/decryption, as well as the encrypted license file lic.txt.

### The C\# Project

Let us create a simple C# project to simulate the process of decryption and editing by another program. Use Visual Studio 2017 and create a console application for the .NET Framework platform. Check connection of System.Security and the System.Security.Cryptography space.

The following issue arises in the code: MQL and C# have different time formats. This issue has already been addressed and solved in [this article](https://www.mql5.com/en/articles/6549). The author has done a great job, and we can use his **MtConverter** class in our project.

Create two classes, **EaUser and UserLic**, with the fields similar to ea\_user and user\_lic structures. The purpose is to decrypt the license created by the terminal (the lic.txt file), to parse the received data, to modify the objects and to re-encrypt, creating a new file. This task must be easy to implement, if you set encryption/decryption modes carefully. Here is how the appropriate piece of code looks like:

```
            using (Aes aesAlg = Aes.Create())
            {
                aesAlg.Key = Key;
                aesAlg.IV = IV;
                aesAlg.Mode = CipherMode.ECB;
                aesAlg.Padding = PaddingMode.Zeros;
                ..................................
```

Pay attention to the last two lines, where the encryption mode (ECB) and padding are set. We use the available information about the settings for these modes. The first line in the block concerning the key installation should be clear. It uses the same key, which is used for encryption in the terminal, but this time it is converted into a byte array:

```
            string skey = "qwertyuiopasdfgh";
            byte[] Key  = Encoding.ASCII.GetBytes(s: skey);
```

Pay attention to the line where the **"IV"** parameter is set. This is the so-called "initialization vector", i.e. a random number that participates in all encryption modes except ECB mode. Therefore, we simply create an array of the desired length at this point:

```
byte[] iv   = new byte[16];
```

In addition, note that the situation with the key in C# differs from that in MQL. If the key length (in this case the "qwertyuiopasdfgh" line) is greater than 16, an exception will be thrown. That is why it was a good decision to strictly control the key length in the MQL code.

The rest is pretty simple. Read the binary file -> Decrypt the stream -> Fill the created UserLic class instance using **BinaryReader**. Probably, a similar result could be achieved by making the corresponding classes serializable. You can test this possibility by yourself.

Let us modify any field, in our case we will change the user ID. Then, encrypt the data the same way and create a new file "lic\_C#.txt". The above operations are performed by two statical function in the project, **EncryptToFile\_Aes** and **DecryptFromFile\_Aes**. For debugging purposes, I have added two similar functions, which operate not with the files, but with byte arrays: **EncryptToArray\_Aes** and **DecryptFromArray\_Aes**.

The CryptoC#.zip project with all the required files is attached below.

Anyone might have noticed the following project flaws:

- It does not have the required checks of called functions arguments.
- There is no exception handling.
- Single-stream mode of operation.

I have not implemented the above features, because the article does not aim at creating a fully-featured application. If we implement all the required parts, the extra code part would be too big and would distract attention from the basic problem.

### The C++ Project

The next project is created in C++. Let us create a console application in the Visual Studio 2017 environment. We do not have any support for encryption/decryption out of the box. Therefore, we have to connect the well-known OpenSSL library by downloading and installing the OpenSSL installation package. As a result, we can use all the Open SSL libraries and includes, which should be connected to the created projects. For details about how to connect libraries to a project, please read [this article](https://www.mql5.com/en/articles/7144). Unfortunately, the OpenSSL documentation is far from complete, however there is nothing better to use.

Once the libraries are connected, proceed to writing the code. The first thing to do is to describe again the already known two structures:

```
constexpr size_t PRODMAXLENGTH = 255;

#pragma pack(1)
typedef struct EA_USER {
        EA_USER();
        EA_USER(std::string name);
        EA_USER(EA_USER& eauser);
        std::time_t  expired;
        long   namelength;
        char eaname[PRODMAXLENGTH];
        std::string GetName();
        void SetName(std::string newName);
        std::string GetTimeExpired();
        std::string ToString();
        size_t ToArray(byte* pbyte);
        constexpr size_t GetSize() noexcept;
        friend std::ostream& operator<< (std::ostream& out, EA_USER& eauser);
        friend std::istream& operator>> (std::istream& in, EA_USER& eauser);
}EAUSER, *PEAUSER;
#pragma pack()

constexpr size_t COUNTACC = 5;

#pragma pack(1)
typedef struct USER_LIC
{
        using PEAUNIC = std::unique_ptr<EAUSER>;

        USER_LIC();
        USER_LIC(USER_LIC&& ul);
        USER_LIC(const byte* pbyte);
        int64_t uid;
        std::time_t expired;
        long log_count;
        int64_t logins[COUNTACC];
        long ea_count;
        std::vector<PEAUNIC> pEa;

        std::string GetTimeExpired();
        std::string ToString();
        size_t ToArray(byte* pbyte);
        void AddEA(EA_USER eau);
        bool AddAcc(long newAcc);
        size_t GetSize();

        friend std::ostream& operator<< (std::ostream& out, USER_LIC& ul);
        friend std::istream& operator>> (std::istream& in, USER_LIC& ul);

	USER_LIC& operator = (const USER_LIC&) = delete;
	USER_LIC(const USER_LIC&) = delete;
} USERLIC, *PUSERLIC;
#pragma pack()
```

The code is a bit more complex than in the C#. There are differences in certain field types. For example, in this project the field with the array of accounts has the **int64\_t** array type, and the MQL include file has the **long** type. This is connected with the size of the appropriate types. If you do not control such features, this can cause hard-to-catch errors. Some parts are easier: here we do not need to convert time.

Also, in this project we can face the incorrect key length problem. To solve this problem, include the following function in the project:

```
std::string AES_NormalizeKey(const void *const apBuffer, size_t aSize)
```

This function will "trim" the **appBuffer** array to the required **aSize** length. Also, let us write the following auxiliary function:

```
void handleErrors(void)
{
        ERR_print_errors_fp(stderr);
}
```

This function will provide explanations of error codes from the OpenSSL library. The following two functions implement the main operations:

```
int aes_decrypt(const byte* key, const byte* iv, const byte* pCtext, byte* pRtext, int iTextsz)
int aes_encrypt(const byte* key, const byte* iv, const byte* pPtext, byte* pCtext, int iTextsz)
```

The implementation of methods is provided in the attached files. I will only mention some of the essential points:

- No initialization vector is used here. We create an array of the desired size and pass it at the call point.
- The library does not provide any benefits regarding padding. Set this mode by calling:



```
EVP_CIPHER_CTX_set_padding(ctx.get(), 0);
```

Make sure to pass "zero", not like this:

```
EVP_CIPHER_CTX_set_padding(ctx.get(), EVP_PADDING_ZERO);
```

as may seem appropriate here. There are even more issues connected with the addition. The fact is that if the padding value is zero (as in our project), then the developer must take care to ensure that the length of the encrypted object is a multiple of **BLOCK\_SIZE = AES\_BLOCK\_SIZE**, namely of 16 bytes. That is why, before calling aes\_encrypt(......), it is necessary to provide the appropriate alignment of the array to be encrypted.


Perform the same sequence of actions, as we did in the previous project:

- Decrypt the resulting file, edit it and encrypt it again. In this case, add information about one more user account to the license.
- Now, we receive one more encrypted file, lic\_С++.txt . The file size is different this time. This is the clock size (16 byres), which was added during the alignment.

All the source files of the project are available in the CryptoС++.zip archive attached below.

### Final Checks and Results

Now, we move on to the final operation step. Move the recently encrypted lic\_С++.txt file to the File folder of MetaTrader data directory and decrypt it using the previously written script decryptuser.mq5 . We get the expected result: the file has been successfully decrypted, despite the change in length.

So, what do we get as a result? Most importantly, we have determined the encryption/decryption parameters, which allow transferring encrypted files from one program to another. Obviously, we can assume later that if encryption/decryption fails, the issue can be caused by errors in application programs.

### Hashes

Most of you may know that cryptography is not limited to encryption/decryption alone. Let us consider a cryptographic primitive - hashing. This process implies converting of some arbitrary array to a fixed-length array. Such an array is called a hash, and the conversion function is called a hash function. Two initial arrays that differ from each other in at least one bit, will produce completely different hashes, which can be used for identification and comparison.

Here is an example. The user registers on the site and enters his identification data. The data is saved in a very secret database. Now, the same user tries to log into the site by entering the login and password on the main page. What should the website do? It can simply retrieve the user's password from the database and compare it with the entered one. But this is not safe. The safe way is to compare the hash of the stored password and the hash of the entered password. Even if the stored hash is stolen, the password itself will remain safe. By comparing the hashes, we can determine whether the entered password is correct.

Hashing is a one-way process. Using the available hash, it is not possible to obtain the data array for which the hash is received. So, hashes are very important for cryptography. Let us consider hash calculation in different environments.

Our purpose is the same: to find out how to make sure that the hash for the same initial data would be the same when calculating in the terminal and other third-party programs.

In MQL, hash is calculated using the same library function that we used earlier: **CryptEncode**. The **method** function argument should be set to a value for calculating the hash. Let us use the **CRYPT\_HASH\_SHA256** value. Documentation provides other values and other hash types, so you can read further on this topic. Use the line of the already existing password: **"qwertyuiopasdfgh"** as the source array. Calculate its hash and write the hash to a file. The resulting code is very simple; therefore, we simply include it to the attached script file **decryptuser.mq5**, without creating separate classes and functions:

```
string k = "qwertyuiopasdfgh";

uchar key[], result[], enc[];
      StringToCharArray(k, enc);
      int sha = CryptEncode(CRYPT_HASH_SHA256,enc,key,result);
      string sha256;
      for(int i = 0; i < sha; i++) sha256 += StringFormat("%X ",result[i]);
      Print("SHA256 len: ",sha," Value:  ",sha256);
      int h = FileOpen("sha256.bin", FILE_WRITE | FILE_BIN);
      if (h == INVALID_HANDLE) {
         Print("File create failed: sha256.bin");
      }else {
         FileWriteArray(h, result);
         FileClose(h);
      }
```

The **key** array which was earlier used for encryption is not used here. Write the resulting hash to the **result** array, output to the window and write to the sha256.bin file. The length of the resulting hash is fixed at 32 bytes. You can change the size of the source array, let us say make it one character long, but the hash size will still be 32 bytes.

Repeat the same calculation by adding the required functionality to C# and C++ projects. The changes are minimal and are very simple. We use the same source array from the password string. Add similar code lines. Calculate and... You will get discouragingly different results! Well, they are "not entirely different". The hashes calculated by the MQL script and the C++ project are the same. But the C# project gives a different result. Let us try to use another string, consisting of one character "a". Again, calculation in the C# project will produce a different hash.

the problem is connected with the **StringToCharArray** function call, which converts a string to an array. If you look at the resulting array after the **StringToCharArray** call, you will see that the array has doubled in size. For example, after calling a function with the string "a", the resulting array will have two elements. The second element will be "0". Call of **Encoding.ASCII.GetBytes** in the C# avoids this. In this case, "0" will not be included into the array.

Now, we can add to the C# project a block of code that appends "0" to the byte array. After that we can use this byte array to calculate the hash. Now we obtain the expected result. All three projects calculate the same hash for the same input data. The resulting hashes are available in files **sha256.bin, sha256\_c#.bin, sha256\_С++.bin**, which are located in the CryptoMQL.zip archive attached below.

Please note that the above example concerns text data. Obviously, when it comes to an initially binary array, there is no need to call **StringToCharArray** and **Encoding.ASCII.GetBytes**. And there will be no issue with an extra 0. So, another possible option is to remove 0 from an MQL project instead of adding it in C#.

Nevertheless, we have solved the initial problem - we have found out under which conditions a certain object hash will be identical, even when calculated in different environments. We have also achieved the goal indicated at the beginning of the article. We have determined which encryption/decryption modes should be used to ensure the compatibility of results in different environments.

### Conclusion

Although encryption/decryption operations are not frequently used in algorithmic trading in the MetaTrader 5 terminal, this task may be helpful when such a need arises.

What is beyond this article? Creation of archives - such an option is available for the CrypetEncode function. **Base64** encoding standard, which is also available. I think, there is no need to consider these modes. True, passwords can be set when creating archives, but:

- This possibility is not mentioned in the documentation.
- The creation of archives, even those protected by a password, has nothing to do with cryptography.

Another option is Base64 encoding. There are some misleading references to this standard in connection with cryptography. This standard must not be used for encryption/decryption! If you wish, you can find out more about this standard and its usage in practice.

The object of our work is the license. For the purpose of this article, I have selected the way which might be helpful in understanding encryption/decryption tasks. So, I used byte arrays. They are encrypted, written to a file, decrypted and so on. In a real situation, this would be extremely inconvenient and can cause errors. When packing the original structures into an array and unpacking them, a one-bit error would damage the entire license. Moreover, such a situation is quite possible, given the differences in the sizes of different types, as shown above. Therefore, another possible format to store the license is the text. These are **xml** and **json**. A good solution to consider is the usage of the json format, as we may use excellent existing parsers for MQL, C# and C++.

### Programs used in the article:

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | CryptoMQL.zip | Archive | Archive with encryption/decryption scripts. |
| 2 | CryptoC#.zip | Archive | C# encryption/decryption project. |
| 3 | CryptoС++.zip | Archive | C++ encryption/decryption project. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/8093](https://www.mql5.com/ru/articles/8093)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/8093.zip "Download all attachments in the single ZIP archive")

[CryptoMQL.zip](https://www.mql5.com/en/articles/download/8093/cryptomql.zip "Download CryptoMQL.zip")(4.43 KB)

[CryptoCc.zip](https://www.mql5.com/en/articles/download/8093/cryptocc.zip "Download CryptoCc.zip")(7.06 KB)

[CryptoCl4.zip](https://www.mql5.com/en/articles/download/8093/cryptocl4.zip "Download CryptoCl4.zip")(2456.49 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Parsing HTML with curl](https://www.mql5.com/en/articles/7144)
- [Arranging a mailing campaign by means of Google services](https://www.mql5.com/en/articles/6975)
- [A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/355799)**
(2)


![Antonio valeriano costa](https://c.mql5.com/avatar/avatar_na2.png)

**[Antonio valeriano costa](https://www.mql5.com/en/users/valerianocosta)**
\|
4 Apr 2023 at 20:51

Congratulations on this great article.

![majidpiryonesi](https://c.mql5.com/avatar/avatar_na2.png)

**[majidpiryonesi](https://www.mql5.com/en/users/majidpiryonesi)**
\|
6 Apr 2023 at 10:54

Hi everyone.

I am newbie in forex

and I want to trade via QT Application.

please help me how to connect my App and my account?

I attached a screenshot

how find this parameters?in my forex account?

??????? how can find 5 parameters in my account?

![Gradient Boosting (CatBoost) in the development of trading systems. A naive approach](https://c.mql5.com/2/41/yandex_catboost.png)[Gradient Boosting (CatBoost) in the development of trading systems. A naive approach](https://www.mql5.com/en/articles/8642)

Training the CatBoost classifier in Python and exporting the model to mql5, as well as parsing the model parameters and a custom strategy tester. The Python language and the MetaTrader 5 library are used for preparing the data and for training the model.

![Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in a subwindow](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library.png)[Timeseries in DoEasy library (part 48): Multi-period multi-symbol indicators on one buffer in a subwindow](https://www.mql5.com/en/articles/8257)

The article considers an example of creating multi-symbol multi-period standard indicators using a single indicator buffer for construction and working in the indicator subwindow. I am going to prepare the library classes for working with standard indicators working in the program main window and having more than one buffer for displaying their data.

![Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://c.mql5.com/2/40/MQL5-avatar-doeasy-library__1.png)[Timeseries in DoEasy library (part 49): Multi-period multi-symbol multi-buffer standard indicators](https://www.mql5.com/en/articles/8292)

In the current article, I will improve the library classes to implement the ability to develop multi-symbol multi-period standard indicators requiring several indicator buffers to display their data.

![A system of voice notifications for trade events and signals](https://c.mql5.com/2/39/logo.png)[A system of voice notifications for trade events and signals](https://www.mql5.com/en/articles/8111)

Nowadays, voice assistants play a prominent role in human life, as we often use navigators, voice search and translators. In this article, I will try to develop a simple and user friendly system of voice notifications for various trade events, market states or signals generated by trading signals.

[![](https://www.mql5.com/ff/sh/wm94j0jmkwd29943z2/ddfa713cb3cdd580c3e81e0e13b5b1b8.jpg)\\
Revised MetaTrader 5 Web Terminal\\
\\
Trade with no restrictions from any mobile device, OS and web browser\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=fkjlpstbxdmrrwpblfatcsdjyrxbizyj&s=f462f051eb7aaec36d6b31792d312d60d3f5a50c83b12d0d66e85d5d61bd941b&uid=&ref=https://www.mql5.com/en/articles/8093&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082959126465614367)

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