---
title: Getting Started with MQL5 Algo Forge
url: https://www.mql5.com/en/articles/18518
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:18:21.359762
---

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/18518&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068075050445239594)

MetaTrader 5 / Examples


The new [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") is more than just a list of your projects – it's a full-fledged social network for developers. You can easily track changes, maintain project history, connect with like-minded professionals, and discover new ideas. Here, you can follow interesting authors, form teams, and collaborate on algorithmic trading projects.

MQL5 Algo Forge is built on Git, the modern version control system. It equips every developer with a powerful toolset for tracking project history, branching, experimenting, and working in teams. But how does it all work? In this article, we'll explain how to get started with [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/").

![](https://c.mql5.com/2/152/Algo-Forge-sheme-01.png)

### What Is a Repository in MQL5 Algo Forge?

Developing software is usually a lengthy process that requires time and debugging. Code must not only be written and maintained but also stored securely. Modern trading algorithms go far beyond simple moving average crossovers. They are built on mathematical libraries, neural networks, and machine learning. This means developers need a convenient way to save changes quickly and access up-to-date code from anywhere.

[MQL5 Storage](https://www.metatrader5.com/en/metaeditor/help/mql5storage "https://www.metatrader5.com/en/metaeditor/help/mql5storage") is the built-in version control system integrated into MetaEditor. It previously used Subversion 1.7, a centralized version control system in which all history was stored on MetaQuotes servers. Without an internet connection, you couldn't commit changes, roll back to previous versions, or view history. Today, it has been fully replaced by a more powerful and flexible solution – MQL5 Algo Forge.

Now, MQL5 Algo Forge works differently:

|  | MQL5 Algo Forge (Git-based) | MQL5 Storage (legacy) |
| --- | --- | --- |
| History storage | Local and cloud | MetaQuotes cloud only |
| Offline work | Full (commits, rollbacks, diff) | Limited or unavailable |
| Operation speed | Instant, local | Dependent on network/server |

Algo Forge gives you freedom. You can work offline on a train, create experimental branches, save intermediate results, and later merge them into the main branch.

A project repository in Algo Forge is not just a cloud folder. It is a structured Git repository that exists on your local drive and synchronizes with the [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") cloud server. It consists of several layers:

#### Working Directory in MetaEditor

| ![](https://c.mql5.com/2/152/01-icon.png) | > Your .MQ5, .MQH, .SET, and other files that you edit in MetaEditor. This is where you write Expert Advisor code, connect indicators, and test strategies. Until you explicitly add changes, Algo Forge does not track them. |
| --- | --- |

**Staging Area (Index)**

| ![](https://c.mql5.com/2/152/02-icon.png) | > Before saving changes to the project history, you **prepare** them. Using the 'Git Add' command, you select the files to include in the next commit. This allows you to group changes **logically** – for example, saving adjustments to trading logic separately from updates to the interface or configuration. |

**Local Repository**

| ![](https://c.mql5.com/2/152/03-icon.png) | > Once you commit changes with 'Git Commit', Algo Forge stores a snapshot of your files in the local repository on your computer. At any time, you can review previous versions, roll back to an earlier state, or analyze which modifications affected your Expert Advisor's performance. |

#### Remote Repository in [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/")

| ![](https://c.mql5.com/2/152/04-icon.png) | > When ready, you can push commits to the [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") server using 'Git Push'. This creates a secure backup and makes your changes available to team members. Even if you work solo, this is useful: your code is safely stored in the cloud, and your project history remains intact. |

### How to Connect to MQL5 Algo Forge

To start working with MQL5 Algo Forge, all you need is an [MQL5 account](https://www.mql5.com/en/articles/24). Simply log in with your credentials at [https://forge.mql5.io/](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/").

![](https://c.mql5.com/2/165/forge_login.png)

Once logged in, you can explore all of its features. Public projects are available in the [Explore](https://www.mql5.com/go?link=https://forge.mql5.io/explore/repos "https://forge.mql5.io/explore/repos") section, where you can browse projects, study their code, share your own work, and collaborate with others.

![](https://c.mql5.com/2/151/forge_explore__4.png)

However, most of your development work will take place in the MetaEditor environment. To connect MetaEditor to MQL5 Algo Forge, just log in with your MQL5 account credentials under the [Community](https://www.metatrader5.com/en/metaeditor/help/beginning/settings#mql5community "https://www.metatrader5.com/en/metaeditor/help/beginning/settings#mql5community") tab.

![](https://c.mql5.com/2/151/metaeditor_community__1.png)

### What Is Git and How It Works in MQL5 Algo Forge

Git is a version control system that records how your files looked at different points in time. It helps prevent the loss of important changes, makes it easy to roll back, supports branching for experiments, and enables multiple developers to collaborate on the same project.

In MetaEditor, you don't work directly with Git commands – everything is handled through a user-friendly interface. Under the hood, however, Git is executing a few essential operations. Let's break them down and see what happens when you use the corresponding commands in MetaEditor.

Let's create a new project "Base\_EA" in the [public projects](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#shared "https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects#shared") folder.

![](https://c.mql5.com/2/151/create_project.png)

Right-click on the project file to see available Git commands.

![](https://c.mql5.com/2/151/git_menu.png)

**1\. Git Add File/Folder** – Staging Changes

When you edit files, Git doesn't automatically track them. To tell the system that you want to save these changes, you add files to the **index**. This can be done via 'Git Add File/Folder' or when selecting files for a commit.

> Git takes a snapshot of the file's current state and prepares it for saving – like placing documents into a "To be signed" folder.

**2\. Git Commit** – Saving a Project Snapshot

Clicking 'Git Commit' in MetaEditor **captures the current state** of your project and saves it to the version history.

Here's what happens:

- Git compares your changes with previous files.
- It stores only the differences (to save space).
- Changes are written into a hidden .git folder.
- The commit gets a unique SHA-1 identifier.

A commit is essentially a checkpoint, showing how the project looked at that moment. You can always return to any such point later and continue from there.

**3\. Git Push** – Sending to the MQL5 Algo Forge Cloud

When you select 'Git Push;, your local commits are uploaded to the MQL5 Algo Forge server.

> It's like uploading a new version of your project to the cloud but with a complete change history.

For convenience, MetaEditor automatically performs 'Git Push' right after 'Git Commit'. This ensures that your current version is always synced to the cloud. If pushing changes to the server fails for any reason, you can run 'Git Push' manually.

**4\. Git Pull** – Retrieving the Latest Version

When collaborating, someone else may update the project before you. Or you might commit changes from one computer and continue work on another (or on the same computer but within a different terminal). To fetch these updates, use 'Git Pull', which retrieves all changes from the Forge.

> Git will download the new commits from the cloud and merge them with your local version.

**5\. Git Branch** – Branching and Experimenting Safely

Sometimes you my want to test ideas without affecting the main project. This can be done through **branches**. For example, you could create a branch to test a different indicator or add a filter to your strategy. Each branch has its own name, and you can switch between them freely.

> A branch is simply a parallel line of development. You can experiment as much as you like, then either discard it or merge it into the main branch.

**6\. Git Difference** – Reviewing File Changes

This command shows exactly what changed in a file (line by line) before committing. It opens a comparison panel highlighting added, deleted, or modified lines.

> ![](https://c.mql5.com/2/151/git_diff.png)

**7\. Git Log** – Viewing Project History

The 'Git Log' command lists all commits with their date, author, and commit message. This gives you a clear timeline of project development and lets you track who changed what and when.

> ![](https://c.mql5.com/2/151/git_log.png)

**8. Git Revert** — Undoing a Commit

'Git Revert' creates a new commit that cancels the effect of a previous one without erasing it from history. This is the safest way to undo changes. But it only works cleanly on the latest commit.

This command can be especially useful if you notice that a commit has broken the Expert Advisor – you can quickly undo it without affecting other commits. However, if the reverted commit affects the same code as later commits, a merge conflict may occur, which must be resolved manually.

### Safe Practices for Working with Projects

One of the most common issues in version control is conflicts between local and remote versions. To avoid them, follow simple rules: **Always pull first**. When you open a project to continue working, run 'Git Pull' to ensure you have the latest version.

![](https://c.mql5.com/2/152/Algo-Forge-sheme-02__2.png)

Another rule is **Always commit + push last**. When you finish working, run 'Git Commit' (with the automatic push) to send your updates to the cloud. Following these habits will keep your workflow smooth and conflict-free, while ensuring your code is always safe.

### MQL5 Algo Forge – Everything You Need for Reliable Project Management

You don't need to type Git commands manually. Everything is integrated into MetaEditor:

- **Adding files, committing, and branching** – via the project context menu.
- **Push and Pull** – just two buttons for syncing with the cloud.
- **Branch and Revert** – simple commands for development control.
- **Change history** – available as a clear, built-in log.

These 8 commands are all you need to use MQL5 Algo Forge confidently. They provide structured development, safe experimentation, teamwork support, and complete protection against data loss.

### Get Started Today

MQL5 Algo Forge is more than just storage – it's a complete project management system for algorithmic traders. It lets you track every change, experiment without risk, collaborate in teams, and maintain stable, reliable code.

If Git once felt too complex, MQL5 Algo Forge makes it simple. MetaEditor integrates the essential commands, so you can focus on writing code, saving progress, and staying ready for any rollback.

[Welcome to MQL5 Algo Forge!](https://www.mql5.com/go?link=https://forge.mql5.io/ "Welcome to MQL5 Algo Forge!")

### Useful links:

- [MetaEditor Manual](https://www.metatrader5.com/en/metaeditor/help/mql5storage "https://www.metatrader5.com/en/metaeditor/help")

- [Git Documentation](https://www.mql5.com/go?link=https://git-scm.com/doc "https://git-scm.com/doc")
- [Git Cheat Sheet (Official)](https://www.mql5.com/go?link=https://education.github.com/git-cheat-sheet-education.pdf "https://education.github.com/git-cheat-sheet-education.pdf")
- [Use of .gitignore](https://www.mql5.com/go?link=https://git-scm.com/docs/gitignore "https://git-scm.com/docs/gitignore")

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/18518](https://www.mql5.com/ru/articles/18518)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

#### Other articles by this author

- [Installing MetaTrader 5 and Other MetaQuotes Apps on HarmonyOS NEXT](https://www.mql5.com/en/articles/18612)
- [MetaTrader 5 on macOS](https://www.mql5.com/en/articles/619)
- [How to earn money by fulfilling traders' orders in the Freelance service](https://www.mql5.com/en/articles/1019)
- [MetaTrader 4 on macOS](https://www.mql5.com/en/articles/1356)
- [Working with ONNX models in float16 and float8 formats](https://www.mql5.com/en/articles/14330)
- [Regression models of the Scikit-learn Library and their export to ONNX](https://www.mql5.com/en/articles/13538)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/494099)**
(20)


![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
5 Sep 2025 at 12:33

**Fernando Carreiro [#](https://www.mql5.com/ru/forum/494065/page2#comment_57965598):**

_As_ a test, I added a description of the _Heikin Ashi_ publication as a README file in Markdown format and committed it to the repository.

Please check if you have received notification of this change and if you can update the fork.

Saw this in the repository web interface:

![](https://c.mql5.com/3/473/4979927684518.png)

I will try to update the fork later

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
5 Sep 2025 at 12:49

**Fernando Carreiro [#](https://www.mql5.com/ru/forum/494065/page2#comment_57965598):**

_As_ a test, I added a description of the _Heikin Ashi_ publication as a README file in Markdown format and committed it to the repository.

Please check if you have received notification of this change and if you can update the fork.

First up, my local fork clone doesn't have the latest commit yet:

![](https://c.mql5.com/3/473/5865087903590.png)

Connecting the original repository, according to the Git documentation:

[![](https://c.mql5.com/3/473/6123151674940__1.png)](https://c.mql5.com/3/473/6123151674940.png "https://c.mql5.com/3/473/6123151674940.png")

I go to the web interface of the fork and see this:

![](https://c.mql5.com/3/473/6543144990025.png)

I click the "Sync" button and then do a Pull in MetaEditor:

![](https://c.mql5.com/3/473/5614065099091.png)

As you can see, all your commits were safely in the fork and then after Pull in the clone of the fork on my local computer.

On [this](https://www.mql5.com/go?link=https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork") documentation page there are other ways to synchronise using console commands, but I haven't tested them, as all commits are already synchronised.

I'll experiment more later to see how the Commit and Push command in MetaEditor will behave for the fork. I wonder if it will try to send edits to the original repository as well.

![Fernando Carreiro](https://c.mql5.com/avatar/2025/9/68d40cf8-38fb.png)

**[Fernando Carreiro](https://www.mql5.com/en/users/fmic)**
\|
5 Sep 2025 at 13:00

**[@Yuriy Bykov](https://www.mql5.com/en/users/antekov) [#](https://www.mql5.com/en/forum/494099/page2#comment_57965867):**

First up, my local fork clone doesn't have the latest commit yet:

Connecting the original repository, according to the Git documentation:

I go to the web interface of the fork and see this:

I click the "Sync" button and then do a Pull in MetaEditor:

As you can see, all your commits were safely in the fork and then after Pull in the clone of the fork on my local computer.

On [this](https://www.mql5.com/go?link=https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork") page of the documentation there are other ways to synchronise using console commands, but I haven't tested them, because all commits are already synchronised.

This is all fine, but you have proven my point that a "Clone" and a "Fork" are not the same, and the method that MetaQuotes has adopted requires extra intervention outside of MetaEditor just to be able to synchronise the project.

Not to mention that it requires extra storage space on the AlgoForge servers, for "forks", while a "clone" requires no extra storage nor extra steps.

I consider the MetaQuotes implementations too "flawed" for effective use and will continue to use an external Git client, or using VSCode (which works just fine with AlgoForge without issues).

![Vladislav Boyko](https://c.mql5.com/avatar/2025/12/692e1587-6181.png)

**[Vladislav Boyko](https://www.mql5.com/en/users/boyvlad)**
\|
5 Sep 2025 at 13:22

**Fernando Carreiro [#](https://www.mql5.com/en/forum/494099/page2#comment_57966403):**

I consider the MetaQuotes implementations too "flawed" for effective use and will continue to use an external Git client, or using VSCode (which works just fine with AlgoForge without issues).

We are glad to welcome you to our community of external Git clients users!😁

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
5 Sep 2025 at 14:25

**Fernando Carreiro [#](https://www.mql5.com/ru/forum/494065/page2#comment_57966404):**

I find the MetaQuotes implementation too "flawed" to use effectively and will continue to use an external Git client or VSCode (which works fine with AlgoForge with no issues).

Unfortunately, this is indeed the case for now. I too prefer to use an external client for now. But if you compare what has been added to MetaEditor in the last 5 months, it's a noticeable progress. It's just that before there were no tools for working with the new repository at all, and now there is at least such a reduced version.

![Trend criteria in trading](https://c.mql5.com/2/106/Trend_Criteria_in_Trading_LOGO.png)[Trend criteria in trading](https://www.mql5.com/en/articles/16678)

Trends are an important part of many trading strategies. In this article, we will look at some of the tools used to identify trends and their characteristics. Understanding and correctly interpreting trends can significantly improve trading efficiency and minimize risks.

![Automating Trading Strategies in MQL5 (Part 28): Creating a Price Action Bat Harmonic Pattern with Visual Feedback](https://c.mql5.com/2/165/19105-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 28): Creating a Price Action Bat Harmonic Pattern with Visual Feedback](https://www.mql5.com/en/articles/19105)

In this article, we develop a Bat Pattern system in MQL5 that identifies bullish and bearish Bat harmonic patterns using pivot points and Fibonacci ratios, triggering trades with precise entry, stop loss, and take-profit levels, enhanced with visual feedback through chart objects

![Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://c.mql5.com/2/165/19285-simplifying-databases-in-mql5-logo__2.png)[Simplifying Databases in MQL5 (Part 1): Introduction to Databases and SQL](https://www.mql5.com/en/articles/19285)

We explore how to manipulate databases in MQL5 using the language's native functions. We cover everything from table creation, insertion, updating, and deletion to data import and export, all with sample code. The content serves as a solid foundation for understanding the internal mechanics of data access, paving the way for the discussion of ORM, where we'll build one in MQL5.

![Analyzing binary code of prices on the exchange (Part II): Converting to BIP39 and writing GPT model](https://c.mql5.com/2/118/Analyzing_the_Binary_Code_of_Stock_Exchange_Prices_Part_II___LOGO.png)[Analyzing binary code of prices on the exchange (Part II): Converting to BIP39 and writing GPT model](https://www.mql5.com/en/articles/17110)

Continuing tries to decipher price movements... What about linguistic analysis of the "market dictionary" that we get by converting the binary price code to BIP39? In this article, we will delve into an innovative approach to exchange data analysis and consider how modern natural language processing techniques can be applied to the market language.

[![](https://www.mql5.com/ff/sh/0wxx5f0vuwq7xh89z2/01.png)VPS for 24/7 tradingContact your broker and find out how to get a free hosting subscriptionLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=nhetzvgituppcfrhndpblbihmzziogdh&s=d00c975c8bda3d8c1b29f042ad33ac81952ccea2f130a8f1ffa9015bab8ade87&uid=&ref=https://www.mql5.com/en/articles/18518&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068075050445239594)

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