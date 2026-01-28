---
title: Creating a Trading Administrator Panel in MQL5 (Part VII): Trusted User, Recovery and Cryptography
url: https://www.mql5.com/en/articles/16339
categories: Trading, Integration
relevance_score: 6
scraped_at: 2026-01-22T17:59:37.189286
---

[Launching MetaTrader VPS for the first time?Read our comprehensive, step-by-step instructions![](https://www.mql5.com/ff/sh/0xb0c8bjq5sadh89z2/01.png)Read](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/articles/13586&a=gxygkojxdwrcfbbgfrchvjgelflsnelu&s=49eab2fb45d89f59a191e88145774dcd7f9533039acb10dd9c28061b04fa92fe&uid=&ref=https://www.mql5.com/en/articles/16339&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049542047519583575)

MetaTrader 5 / Examples


### Contents:

- [Introduction](https://www.mql5.com/en/articles/16339#para1)
- [Implementation Of Trusted User in Admin Panel](https://www.mql5.com/en/articles/16339#para2)
- [Cryptography and Examples of application in Admin Panel](https://www.mql5.com/en/articles/16339#para3)
- [Testing](https://www.mql5.com/en/articles/16339#para4)
- [Conclusion](https://www.mql5.com/en/articles/16339#para5)

### Introduction

The [MQL5 Market](https://www.mql5.com/en/market) exemplifies the importance of device identity in ensuring the security of distributed products. Our current exploration is motivated by the challenges we've encountered while working with our secure Admin Panel. Since implementing enhanced security measures, the development and testing process has faced delays due to frequent login prompts and authentication requests triggered by each file compilation or feature test on the terminal. This additional layer of security, while necessary, has introduced friction in our workflow.

The warning shown below appears on the first page of the [Market](https://www.mql5.com/en/market) when publishing a new product. It highlights how MQL5 prioritizes and enforces robust security measures to protect both developers and users.

Every Program sold at the Market is encrypted with a special key to protect the Product from illegal use. The encryption key is unique for every Buyer, and it is bound to his computer, so that all the Products in the Market have automatic copy protection.

Purchased Product can be activated at least five times. This ensures the balance of interests between Buyers and Sellers. The number of available activations is set by the Seller.

Many applications and websites implement second-layer protection selectively, activating it only when suspicious activity is detected, such as anonymous IP usage, login attempts from new devices, or multiple failed login attempts. This approach minimizes interruptions while maintaining security.

In our case, delayed testing during development is caused by repeatedly entering passwords and checking the Telegram app for generated 6-digit codes. The frequent prompts can become tedious, particularly when triggered by terminal activity changes. Below are some notable activities that lead to device reinitialization and subsequent password requests:

- Pair change
- Time frame switching
- Terminal reboot etc.

In certain scenarios, our programs reinitialize repeatedly due to varying activities—a process that is unavoidable for technical or operational reasons. The user validation algorithm is embedded at the start of the initialization function, making it impossible for the program to proceed without passing this step. However, we can introduce a bypass mechanism within the initialization function to optimize the process. This bypass algorithm monitors the number of login attempts, allowing for a more seamless experience during valid sessions.

If the number of failed password attempts exceeds the set limit, it signals suspicious behavior or indicates that the original user may have lost or forgotten their password. On the other hand, if the correct password is entered within three attempts, the system intelligently bypasses 2FA, saving significant time by avoiding repeated Telegram-based authentication. This approach enhances both security and efficiency, particularly during intensive development and testing phases.

The image below highlights the issue of multiple verification messages sent to Telegram during app development, showcasing the practical need for this refined algorithm.

![Too many 2FA prompts during development and testing](https://c.mql5.com/2/130/Telegram_7QVSUnU5tX.gif)

Admin Pane: Personal Telegram 2FA code delivery bot

Previously, the repeated prompts for both password and 2FA authentication during every program initialization made the process extremely time-consuming. To save time during development, I added a line of code to print the 6-digit verification code sent to Telegram directly in the terminal’s Expert log. This allowed me to quickly retrieve the code from the log during app testing without needing to switch to the Telegram app, streamlining the process significantly.

The code snippet in question is part of the function responsible for sending Telegram messages. Within it, I included a feature to log the message to the journal for convenience. However, for this development phase, I plan to remove this feature to enhance security in future versions. Leaving sensitive information, such as verification codes, exposed in the journal could pose a security risk if unauthorized access occurs. This step reflects a balance between optimizing the development process and adhering to best practices for securing sensitive data.

The code snippet below shows the line that prints the message sent.

```
//+------------------------------------------------------------------+
//| Send the message to Telegram                                     |
//+------------------------------------------------------------------+
bool SendMessageToTelegram(string message, string chatId, string botToken)
{
    string url = "https://api.telegram.org/bot" + botToken + "/sendMessage";
    string jsonMessage = "{\"chat_id\":\"" + chatId + "\", \"text\":\"" + message + "\"}";

    char postData[];
    ArrayResize(postData, StringToCharArray(jsonMessage, postData) - 1);

    int timeout = 5000;
    char result[];
    string responseHeaders;
    int responseCode = WebRequest("POST", url, "Content-Type: application/json\r\n", timeout, postData, result, responseHeaders);

    if (responseCode == 200)
    {
        Print("Message sent successfully: ", message); //This line prints the message in journal,thus a security leak that I used to bypass telegram during app tests.
        return true;
    }
    else
    {
        Print("Failed to send message. HTTP code: ", responseCode, " Error code: ", GetLastError());
        Print("Response: ", CharArrayToString(result));
        return false;
    }
}
```

Highlighted below is a message sent to Telegram, including the verification code. This approach was a convenient way to bypass checking Telegram for the code during development. However, leaving it in place poses a significant security risk, as unauthorized individuals could potentially retrieve the code directly from the terminal without needing access to Telegram. The text is normally printed on the terminal's Experts tab:

```
2024.11.22 08:44:47.980 Admin Panel V1.22 (Volatility 75 (1s) Index,M1) Message sent successfully: A login attempt was made on the Admin Panel. Please use this code to verify your identity: 028901
2024.11.22 08:44:48.040 Admin Panel V1.22 (Volatility 75 (1s) Index,M1) Password authentication successful. A 2FA code has been sent to your Telegram.
```

**Discussion Goal:**

The goal of this discussion is to integrate a trusted user recognition system that streamlines login and 2FA processes. Users who successfully enter the correct password within three attempts can bypass 2FA, while exceeding the limit triggers 2FA, requiring a code sent via Telegram and the hardcoded password for recovery. This system balances convenience with security by limiting retry attempts to mitigate brute force attacks while ensuring a smooth authentication process. Trust is session-specific, requiring revalidation with each new login. The next section will provide further insights into measures for addressing brute force attacks.

**What is brute force attack?**

A brute force attack is a method used by attackers to gain unauthorized access to accounts, systems, or encrypted data by systematically trying every possible combination of passwords, encryption keys, or credentials until the correct one is found. This approach relies on computational power and persistence rather than exploiting vulnerabilities in the system itself.

Key Features of a Brute Force Attack:

- Trial and Error: The attacker repeatedly attempts different combinations of characters until access is gained.
- Automation: Specialized tools or scripts are often used to automate the process, allowing thousands or even millions of attempts in a short period.
- Time-Intensive: The time required depends on the complexity of the target password or key. Stronger passwords with more characters and varied combinations take significantly longer to crack.

Types of Brute Force Attacks:

1. Simple Brute Force: Trying all possible combinations of characters.
2. Dictionary Attack: Using a list of commonly used passwords (e.g., "123456," "password," or "qwerty") to attempt access.
3. Credential Stuffing: Using leaked username-password pairs from other breaches to gain access to accounts on different platforms.

Prevention:

- Limiting login attempts.
- Enforcing strong passwords (long, with a mix of uppercase, lowercase, numbers, and symbols).
- Implementing two-factor authentication (2FA) to add another layer of security.

The current security for Panel Access is vulnerable to the mentioned attack, and while limiting login attempts is a crucial step, we also aim to avoid frequent authentication through Telegram to save time. Sending the verification code serves to protect against unauthorized access and alerts the real admin to any potential malicious activity on the Panel, providing a safeguard if they are unaware of any security breaches.

### Implementation Of Trusted User in Admin Panel

To implement the new feature in the Admin Panel, we modified only a few sections of our source code related to security and the initialization of the EA program. This approach ensures that the rest of the program remains unchanged, minimizing the risk of errors when managing a large code. The focus here is on the login handler function, and we have introduced new variables, which are declared in the next code snippet.

```
int failedAttempts = 0; // Counter for failed login attempts
bool isTrustedUser = false; // Flag for trusted users
```

The global variables _failedAttempts_ and _isTrustedUser_ are critical for implementing the new features. _failedAttempts_ monitors the number of incorrect password entries and helps determine when 2FA should be enforced. The _isTrustedUser_ flag recognizes users who successfully log in within the allowed attempts, skipping the 2FA process for these users. By resetting these variables appropriately in the _OnLoginButtonClick_ function, the system maintains dynamic control over the authentication flow, ensuring flexibility and security.

**Enhancing Login Function for the new features**

The _OnLoginButtonClick_ function integrates the new features by tracking login attempts with the _failedAttempts_ counter and identifying trusted users with the _isTrustedUser_ flag. When a correct password is entered, it checks if the user qualifies as trusted by having fewer than three failed attempts. Trusted users skip the 2FA step and proceed directly to the Admin Home Panel. If the failed attempts exceed the limit, the function generates a 2FA code, sends it to the user via Telegram along with the hardcoded password for recovery, and shows the 2FA authentication prompt. This ensures a balance between user convenience for trusted users and enforced security after failed attempts.

```
//+------------------------------------------------------------------+
//| Handle login button click                                        |
//+------------------------------------------------------------------+

 void OnLoginButtonClick()
{
    string enteredPassword = passwordInputBox.Text();

    if (enteredPassword == Password)
    {
        failedAttempts = 0; // Reset attempts on successful login
        isTrustedUser = true;

        if (failedAttempts <= 3) // Skip 2FA for trusted users
        {
            authentication.Destroy();
            adminHomePanel.Show();
            Print("Login successful. 2FA skipped for trusted user.");
        }
        else
        {

        }
    }
    else
    {
        failedAttempts++;
        feedbackLabel.Text("Wrong password. Try again.");
        passwordInputBox.Text("");

        if (failedAttempts >= 3)
        {

            Print("Too many failed attempts. 2FA will be required.");

            twoFACode = GenerateRandom6DigitCode();
            SendMessageToTelegram(
                "A login attempt was made on the Admin Panel.\n" +
                "Use this code to verify your identity: " + twoFACode + "\n" +
                "Reminder: Your admin password is: " + Password,
                Hardcoded2FAChatId, Hardcoded2FABotToken
            );
            authentication.Destroy();

            ShowTwoFactorAuthPrompt();
            Print("Password authentication successful. A 2FA code has been sent to your Telegram.");
            failedAttempts = 0; // Reset attempts after requiring 2FA
        }
    }
}
```

**Five instances during program initialization**

The authentication process, including the new features for skipping 2FA, takes effect during initialization because the _ShowAuthenticationPrompt()_ function displays the login interface where the _OnLoginButtonClick_ handler is linked to the " _Login_" button. Here's how it works step by step:

1. Triggering the Authentication Prompt: In the _OnInit()_ function, _ShowAuthenticationPrompt()_ is called first. This function creates and displays the authentication dialog containing the password input field and the _"Login"_ button. The program halts further execution until the user interacts with the dialog.
2. Handling Login Attempts: When the user clicks the _"Login"_ button, the _OnLoginButtonClick_ function is executed. This function checks the entered password, updates the _failedAttempts_ counter, and determines whether to grant access directly or enforce 2FA based on the number of failed attempts.
3. Proceeding After Successful Authentication: If the login is successful and the user qualifies as a trusted user (fewer than three failed attempts), the authentication dialog is destroyed, and the Admin Home Panel is displayed immediately. This bypasses 2FA for trusted users.
4. Requiring 2FA When Necessary: If the user exceeds the allowed failed attempts, the program enforces 2FA. A 6-digit code is generated and sent via Telegram, and the 2FA authentication dialog is displayed through the _ShowTwoFactorAuthPrompt()_ function. Remember the function for generating our digit code highlighted in red.

```
//+------------------------------------------------------------------+
//| Generate a random 6-digit code for 2FA                           |
//+------------------------------------------------------------------+
string GenerateRandom6DigitCode()
{
    int code = MathRand() % 1000000; // Generate a 6-digit number
    return StringFormat("%06d", code); // Ensure leading zeros
}
```

    5\.  Further Initialization: Once the user is authenticated (either via a trusted login or successful 2FA), the rest of the panels (Admin Home Panel, Communications Panel, Trade Management Panel) are initialized in the background but remain hidden until explicitly shown during navigation.

Here is the initialization function code snippet:

```
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    if (!ShowAuthenticationPrompt())
    {
        Print("Authorization failed. Exiting...");
        return INIT_FAILED;
    }

    if (!adminHomePanel.Create(ChartID(), "Admin Home Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Admin Home Panel");
        return INIT_FAILED;
    }

    if (!CreateAdminHomeControls())
    {
        Print("Home panel control creation failed");
        return INIT_FAILED;
    }

    if (!communicationsPanel.Create(ChartID(), "Communications Panel", 0, 30, 30, 500, 500))
    {
        Print("Failed to create Communications panel dialog");
        return INIT_FAILED;
    }

    if (!tradeManagementPanel.Create(ChartID(), "Trade Management Panel", 0,260, 30, 1040, 170))
    {
        Print("Failed to create Trade Management panel dialog");
        return INIT_FAILED;
    }

    if (!CreateControls())
    {
        Print("Control creation failed");
        return INIT_FAILED;
    }

    if (!CreateTradeManagementControls())
    {
        Print("Trade management control creation failed");
        return INIT_FAILED;
    }

    adminHomePanel.Hide(); // Hide home panel by default on initialization
    communicationsPanel.Hide(); // Hide the Communications Panel
    tradeManagementPanel.Hide(); // Hide the Trade Management Panel
    return INIT_SUCCEEDED;
}
```

### Cryptography and Examples of application in Admin Panel

[Cryptography](https://www.mql5.com/en/book/advanced/crypt) in MQL5 involves using algorithms and methods to secure sensitive data such as passwords, messages, or authentication codes.

Below are four methods for applying the concept in Admin Panel.

1\. Hashing Passwords

- The _HashPassword_ function uses the _[CryptEncode](https://www.mql5.com/en/docs/common/cryptencode)_ function with the [CRYPT\_HASH\_SHA256](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) algorithm to hash the input password. The hashed result is converted into a hexadecimal string using _CharArrayToHex_ for storage or comparison. In the _VerifyPassword_ function, the entered password is hashed and compared to the stored hash, ensuring no plaintext passwords are stored or processed.
- Storing hashed passwords ensures that even if the stored data is accessed, the actual passwords remain secure. Hashing is one-way, meaning the original password cannot be derived from the hash, adding an essential layer of security to authentication.

```
// Example for storing our hard-coded password

string HashPassword(string password)
{
    uchar hash[];
    CryptEncode(CRYPT_HASH_SHA256, password, hash);
    return CharArrayToHex(hash);
}

// Usage
string PasswordHash = HashPassword("2024"); // Store this instead of plaintext
bool VerifyPassword(string enteredPassword)
{
    return HashPassword(enteredPassword) == PasswordHash;
}
```

2\. Encrypting Sensitive Data

- The _EncryptData_ function uses the [AES-256](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants) algorithm (CRYPT\_AES256) to encrypt sensitive information, like 2FA codes, using a secure encryption key. The _DecryptData_ function reverses this process, decrypting the data back into its original form. The key is critical in this process, as it must match during encryption and decryption.

- Encryption protects sensitive data during storage or transmission. For example, if the 2FA code is intercepted during communication, the encrypted version ensures that unauthorized users cannot interpret it without the correct key.

```
//Example for encryption of our 2FA verification code
string EncryptData(string data, string key)
{
    uchar encrypted[];
    CryptEncode(CRYPT_AES256, data, encrypted, StringToCharArray(key));
    return CharArrayToHex(encrypted);
}

string DecryptData(string encryptedData, string key)
{
    uchar decrypted[];
    uchar input[];
    StringToCharArray(encryptedData, input);
    CryptDecode(CRYPT_AES256, input, decrypted, StringToCharArray(key));
    return CharArrayToString(decrypted);
}

// Usage
string key = "StrongEncryptionKey123"; // Use a secure key
string encrypted2FA = EncryptData(twoFACode, key);
Print("Encrypted 2FA code: ", encrypted2FA);
```

3\. Secure Random Number Generation for 2FA

- The _GenerateSecureRandom6DigitCode_ function uses the [_CryptEncode_](https://www.mql5.com/en/docs/common/cryptencode) function with the _[CRYPT\_HASH\_SHA256](https://www.mql5.com/en/docs/constants/namedconstants/otherconstants)_ algorithm to generate a cryptographically secure random sequence. The result is then converted into a 6-digit number using modular arithmetic and formatting

- Standard random functions like _MathRand_ are pseudo-random and predictable. Using cryptographic randomness ensures that 2FA codes are secure and resistant to prediction or brute-force attacks, enhancing the security of the authentication process.

```
// Example for generating a secure verification code
string GenerateSecureRandom6DigitCode()
{
    uchar randomBytes[3];
    CryptEncode(CRYPT_HASH_SHA256, MathRand(), randomBytes); // Use CryptEncode for randomness
    int randomValue = (randomBytes[0] << 16) | (randomBytes[1] << 8) | randomBytes[2];
    randomValue = MathAbs(randomValue % 1000000); // Ensure 6 digits
    return StringFormat("%06d", randomValue);
}
```

4\. Secure Communications with Telegram

- The _SendEncryptedMessageToTelegram_ function encrypts the message using the _EncryptData_ function before transmitting it to the Telegram server via the _SendMessageToTelegram_ function. The encrypted message can only be decrypted by someone with the correct decryption key.

- Encrypting communication ensures the confidentiality of sensitive information, such as 2FA codes, even if the transmission is intercepted. This is especially important when using third-party communication platforms, where data might not be fully secure.

```
//Example of the securely sending to Telegram

bool SendEncryptedMessageToTelegram(string message, string chatId, string botToken, string key)
{
    string encryptedMessage = EncryptData(message, key);
    return SendMessageToTelegram(encryptedMessage, chatId, botToken);
}

// Usage
string key = "StrongEncryptionKey123";
SendEncryptedMessageToTelegram("Your 2FA code is: " + twoFACode, Hardcoded2FAChatId, Hardcoded2FABotToken, key);
```

### Testing

At this stage, we will present the results of our security enhancements to the Admin Panel. The update simplifies the login process for trusted users, allowing them to quickly access the panel, while untrusted users are required to undergo secondary verification to confirm their authenticity. If a user forgets their password, they can recover it through the authentication process.

In the Expert Log below, we show the failed login attempts, followed by an image illustrating the login flow

```
2024.11.22 03:53:59.675 Admin Panel V1.23 (Volatility 75 (1s) Index,M2) Too many failed attempts. 2FA will be required.
2024.11.22 03:54:00.643 Admin Panel V1.23 (Volatility 75 (1s) Index,M2) Message sent successfully: Check your telegram for verification code and Password
2024.11.22 03:54:00.646 Admin Panel V1.23 (Volatility 75 (1s) Index,M2) Password authentication successful. A 2FA code has been sent to your Telegram.
2024.11.22 03:54:22.946 Admin Panel V1.23 (Volatility 75 (1s) Index,M2) 2FA authentication successful. Access granted to Admin Home Panel.
```

![Telegram message sent for verification](https://c.mql5.com/2/130/telegra_m_msg.PNG)

A Telegram message sent for verification

![Failed Admin Panel Login Attempts](https://c.mql5.com/2/130/terminal64_2QW1E83rgp.gif)

Admin Panel: Failed Login Attempts

For trusted users, the login process is straightforward. They can bypass the secondary verification step, streamlining their access to the Admin Panel. Below is the Expert Log showing the successful login for trusted users, followed by an image that illustrates the simple process for those who enter the correct password.

```
2024.11.22 03:57:41.563 Admin Panel V1.23 (Volatility 75 (1s) Index,M2) Login successful. 2FA skipped for trusted user.
```

![Trusted user easy login wth password](https://c.mql5.com/2/130/terminal64_xaKaXprx4r.gif)

Admin Panel: Trusted User Successful login with password

### Conclusion

In this discussion of enhancing the Admin Panel, we made significant strides in both functionality and security. The introduction of a trusted user feature allows for a smoother and more efficient login experience for known users by limiting login attempts to three and bypassing 2FA for successful authentication within this threshold. This approach balances security and usability, reducing friction for legitimate users while maintaining strict access control measures for untrusted attempts.

We also explored the potential of cryptography to strengthen the panel's security framework. By employing password hashing, encryption for sensitive data, and secure random number generation for 2FA codes, we ensured the integrity and confidentiality of critical information. Hashing protects stored passwords from unauthorized access, while encryption safeguards sensitive data during transmission or storage. Additionally, cryptographically secure randomness ensures the unpredictability of generated codes, adding an extra layer of defense against brute-force attacks.

These advancements help us ensure a user-friendly, and secure administrative tool for managing trading communications and operations. By addressing both functionality and security, we set the stage for future enhancements that can further streamline workflows while adhering to the highest standards of data protection.

[Back to Contents](https://www.mql5.com/en/articles/16339#para0)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/16339.zip "Download all attachments in the single ZIP archive")

[Admin\_Panel\_V1.23.mq5](https://www.mql5.com/en/articles/download/16339/admin_panel_v1.23.mq5 "Download Admin_Panel_V1.23.mq5")(75.34 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [From Novice to Expert: Higher Probability Signals](https://www.mql5.com/en/articles/20658)
- [From Novice to Expert: Navigating Market Irregularities](https://www.mql5.com/en/articles/20645)
- [From Novice to Expert: Automating Trade Discipline with an MQL5 Risk Enforcement EA](https://www.mql5.com/en/articles/20587)
- [From Novice to Expert: Trading the RSI with Market Structure Awareness](https://www.mql5.com/en/articles/20554)
- [From Novice to Expert: Developing a Geographic Market Awareness with MQL5 Visualization](https://www.mql5.com/en/articles/20417)
- [The MQL5 Standard Library Explorer (Part 5): Multiple Signal Expert](https://www.mql5.com/en/articles/20289)
- [The MQL5 Standard Library Explorer (Part 4): Custom Signal Library](https://www.mql5.com/en/articles/20266)

**[Go to discussion](https://www.mql5.com/en/forum/477236)**

![Mastering Log Records (Part 1): Fundamental Concepts and First Steps in MQL5](https://c.mql5.com/2/102/logify60x60.png)[Mastering Log Records (Part 1): Fundamental Concepts and First Steps in MQL5](https://www.mql5.com/en/articles/16447)

Welcome to the beginning of another journey! This article opens a special series where we will create, step by step, a library for log manipulation, tailored for those who develop in the MQL5 language.

![MQL5 Wizard Techniques you should know (Part 49): Reinforcement Learning with Proximal Policy Optimization](https://c.mql5.com/2/103/MQL5_Wizard_Techniques_you_should_know_Part_49___LOGO.png)[MQL5 Wizard Techniques you should know (Part 49): Reinforcement Learning with Proximal Policy Optimization](https://www.mql5.com/en/articles/16448)

Proximal Policy Optimization is another algorithm in reinforcement learning that updates the policy, often in network form, in very small incremental steps to ensure the model stability. We examine how this could be of use, as we have with previous articles, in a wizard assembled Expert Advisor.

![Neural Networks Made Easy (Part 94): Optimizing the Input Sequence](https://c.mql5.com/2/80/Neural_networks_are_easy_Part_94____LOGO.png)[Neural Networks Made Easy (Part 94): Optimizing the Input Sequence](https://www.mql5.com/en/articles/15074)

When working with time series, we always use the source data in their historical sequence. But is this the best option? There is an opinion that changing the sequence of the input data will improve the efficiency of the trained models. In this article I invite you to get acquainted with one of the methods for optimizing the input sequence.

![Trading Insights Through Volume: Moving Beyond OHLC Charts](https://c.mql5.com/2/102/Trading_Insights_Through_Volume_Moving_Beyond_OHLC_Charts___LOGO.png)[Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)

Algorithmic trading system that combines volume analysis with machine learning techniques, specifically LSTM neural networks. Unlike traditional trading approaches that primarily focus on price movements, this system emphasizes volume patterns and their derivatives to predict market movements. The methodology incorporates three main components: volume derivatives analysis (first and second derivatives), LSTM predictions for volume patterns, and traditional technical indicators.

[What's wrong with regular VPS?Here are the 8 most common problems that algorithmic traders may encounterRead![](https://www.mql5.com/ff/sh/hzatb686qjqxwtr4z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/450486&a=drhremihlwuaqyvgpzfddbtmgciejpba&s=c37d25bcceb93ed153b814e6ba4d4839461a9b2d68dd82b95b142be06d310f3f&uid=&ref=https://www.mql5.com/en/articles/16339&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5049542047519583575)

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