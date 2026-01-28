---
title: Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects
url: https://www.mql5.com/en/articles/19436
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:17:47.363144
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=crddatindijxbtdamgbmdhyavbwhdtqu&ssn=1769177863520377090&ssn_dr=1&ssn_sr=0&fv_date=1769177863&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19436&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Moving%20to%20MQL5%20Algo%20Forge%20(Part%203)%3A%20Using%20External%20Repositories%20in%20Your%20Own%20Projects%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917786400879298&fz_uniq=5068061684507014407&sv=2552)

MetaTrader 5 / Examples


### Introduction

In the [second part](https://www.mql5.com/en/articles/17698) of our transition to [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/"), we focused on solving one of the important challenges –working with multiple repositories. Using the combination of the [Adwizard](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/Adwizard "https://forge.mql5.io/antekov/Adwizard") library project and the [Simple Candles](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles "https://forge.mql5.io/antekov/SimpleCandles") Expert Advisor, we encountered and successfully resolved issues mostly related to file inclusion paths and branch merging. We also tried to use MetaEditor tools (where possible) throughout the entire workflow, from creating a separate branch for fixes to merging it via a Pull Request. However, where MetaEditor functionality was not enough, we switched to the MQL5 Algo Forge web interface, an external Git client in Visual Studio Code, or Git console commands. This clearly demonstrated how even in individual development, you can apply Git best practices to maintain order and a clear history of changes within your project.

But that was only one side: using the storage as a "closed" ecosystem where the developer owns all the repositories used. The logical next step, and one of the main reasons for moving to Git, is the ability to fully leverage public repositories from other community members. This is where the true potential of distributed development reveals itself: the ability to easily connect and update third-party code, contribute to its improvement, and assemble complex projects from ready-made, well-tested components.

In this article, we finally turn to this promising, yet more complex, task: how to practically connect and use libraries from third-party repositories within MQL5 Algo Forge. And not "someday in the future" but right now, without waiting for further development of MetaEditor's repository tools.

### Mapping out the Path

In this article, we will continue working with our Simple Candles project repository, which will serve as an excellent testing ground for experimentation. The existing trading strategy already includes a custom volatility calculation functionally similar to the standard Average True Range (ATR) indicator. However, instead of relying solely on own implementation, we will explore how to improve the code by including specialized, ready-to-use solutions from the community.

To do this, we will turn to the publicly available [SmartATR](https://www.mql5.com/go?link=https://forge.mql5.io/steverosenstock/SmartATR "https://forge.mql5.io/steverosenstock/SmartATR") repository, assuming it contains a more advanced and optimized version of the indicator. Our long-term practical goal is to modify the EA so that it can choose between continuing to use the internal calculation or switching to the external SmartATR library algorithm. However, in this article, we will not focus on building a fully functional EA but instead examine the key aspects of working with external repositories.

To achieve this, we need to do the following. Download the SmartATR library code to our local machine and set it up for inclusion in our project. We will cover how to add an external repository into your working environment so that it can later be easily updated when new versions are released. After that, we will apply modifications both to the Simple Candles project and (as it turns out to be necessary) to the SmartATR library code itself. Ideally, we could avoid the last step, but since our case requires it, we will use this as a practical example of how to introduce changes to someone else's repository. Finally, we will verify the integration by testing whether the SmartATR library can be successfully included and compiled as part of our project.

This approach will allow us to carefully walk through the entire process of integrating external code. The experience will be universal: once we have successfully added one library, we will be able to include any other public repository from MQL5 Algo Forge into our projects using the same approach.

### Obtaining External Code

At first glance, this shouldn't pose a problem. Any Git repository can be cloned to a local computer using the standard console command:

git clone ...

However, we agreed to follow a specific order: first try working through MetaEditor interface, then the MQL5 Algo Forge web interface, and only if those approaches fail, resort to external tools such as Visual Studio Code or Git console commands.

So the first question is: how do we view someone else's repository in MetaEditor to select it for cloning? The partial answer could be found in the [help](https://www.mql5.com/go?link=https://forge.mql5.io/help/en/guide "https://forge.mql5.io/help/en/guide") documentation, but very few users would know where to look right away. We ourselves only came across this page later. Before that, we noticed that the Shared Projects folder in MetaEditor only showed our own repositories. To investigate further, we tried the context menu options available for this folder in MetaEditor's Navigator.

![](https://c.mql5.com/2/168/299237195735.png)

The New Project option is not the right one here, as it only creates a new repository owned by us. Refresh doesn't t add any external repositories either. The 'Show All Files' option behaves oddly: after running it, duplicate names appeared for our repositories that hadn't yet been cloned locally. Fortunately, pressing Refresh removes these duplicate names. Our last hope was the 'Show All Public Projects' option, it also produced no visible changes.

Unfortunately, this means that for now we cannot rely solely on MetaEditor to clone external repositories. Let's look at a couple of alternative approaches to achieve our goal.

### Approach One: Direct Cloning

Let's start with an experiment. If we create an empty folder with any arbitrary name (e.g., _TestRepo_) inside Shared Projects, it becomes visible in MetaEditor. From there, we can even execute the Clone command from its context menu. However, judging by the logs, MetaEditor then attempts to clone a repository with the same name ( _TestRepo_) from our personal storage – a repository which, of course, doesn't exist:

![](https://c.mql5.com/2/168/3447693584872.png)

This confirms that the method won't work for cloning someone else's repository. Let's instead try cloning the SmartATR repository directly into Shared Projects using the 'git clone ...' console command and see what happens.

![](https://c.mql5.com/2/168/4542162208156.png)

After cloning, a new _SmartATR_ folder appears in Shared Projects, and it is displayed in MetaEditor's Navigator. More importantly, we can not only view this repository but also work with it as a repository: perform Pull and view the change history (Log) directly from MetaEditor.

![](https://c.mql5.com/2/168/4156381571887.png)

Thus, what MetaEditor currently lacks is a context menu option such as 'Clone from...', which would allow the user to specify the URL of a repository from the storage, or alternatively, open a dialog to search and select from all public repositories in MQL5 Algo Forge (similar to the [Explore](https://www.mql5.com/go?link=https://forge.mql5.io/explore/repos "https://forge.mql5.io/explore/repos") section in the web interface). Another possible improvement could be to display not only personal repositories under Shared Projects but also public repositories that the user has starred in the web interface ( [Starred Repositories](https://www.mql5.com/go?link=https://forge.mql5.io/antekov?tab=stars "https://forge.mql5.io/antekov?tab=stars")), with the ability to toggle their visibility. But let's not speculate too far about what changes might eventually be introduced in MetaEditor.

For now, returning to our successfully cloned SmartATR repository, we can say the immediate goal has been achieved. The project's source code is now available locally, which means we can use it in our own projects. However, there's a caveat. We can only proceed with the direct usage if SmartATR code requires no modifications, meaning we can use it "out of the box", updating only when new versions are released. Let's see of we can do this.

### Checking Functionality

Within the SmartATR project, we received a [file](https://www.mql5.com/go?link=https://forge.mql5.io/steverosenstock/SmartATR/src/branch/main/SmartATR.mq5 "https://forge.mql5.io/steverosenstock/SmartATR/src/branch/main/SmartATR.mq5") with the source code for a MetaTrader 5 indicator, which (according to the description) calculates the Average True Range (ATR) using a more advanced approach. Let's try compiling it… and we immediately encounter an error.

![](https://c.mql5.com/2/169/5483193296254.png)

Regardless of how serious the error is, the important point is this: we cannot use the project without making changes. At this stage, we must decide whether to apply fixes only for our local use or whether to share them and contribute to the original repository. Other developers might also encounter the same issue when trying to use code from this project. So, the second option is more preferable, as it aligns with the philosophy of open-source development.

For now, however, let's assume that we will not be publishing fixes at the moment. First of all, we need to resolve the errors; only then will we have something meaningful to publish. In this case, if we only plan to make local changes to the SmartATR project, we can simply create a new local branch.

Let's try to do this. The original SmartATR repository contains only a _main_ branch, so we'll create a new _develop_ branch via the context menu of the project folder in MetaEditor. The branch appears in the list of branches shown in MetaEditor. After pressing Push, the logs confirm that the operation was successful. At this point, we might expect that the new branch has been created in the original repository. But checking the MQL5 Algo Forge web interface shows otherwise: nothing has changed.

Next, let's try editing the code and committing changes from MetaEditor. We add comments before each line that caused an error, noting the need for fixes, and commit these changes. The MetaEditor logs indicate that both the commit and the push were successful.

![](https://c.mql5.com/2/168/687987947882.png)

Yet once again, by checking the original repository in the MQL5 Algo Forge web interface, we see that nothing has changed. This is at the very least unusual. Let's check the project in Visual Studio Code. and try to understand what's happening. We open the folder with our SmartATR project clone and see the following:

![](https://c.mql5.com/2/168/5431673854632.png)

The latest commit exists, but VS Code suggests we publish the _develop_ branch. This means that the branch does not yet exist in the remote repository, nor does our commit. We try to publish the branch but get an error:

![](https://c.mql5.com/2/168/2313020453146.png)

Check the logs to find out the reason:

![](https://c.mql5.com/2/169/6292185480171.png)

Our user account does not have write permissions for the original repository. This makes sense. Otherwise, the project could easily devolve into chaos with uncontrolled edits from anyone. This means that we can only make modifications in our local copy. However, these changes cannot be synchronized with the remote and will exist only in our local clone. This is far from ideal. Beyond collaboration options, external repositories perform a very important role – they serve as an store project backups. Giving up that safety net would be unwise.

It's also worth noting that when working exclusively in MetaEditor, there was no indication that something was wrong. According to the MetaEditor logs, everything appeared fine: no errors, and all changes were "successfully" pushed… to a non-existent repository. Hopefully, this issue will be corrected in future builds.

### Approach Two: Cloning a Fork

Let's now try a different route. Here too, we'll need to step outside the current capabilities of MetaEditor – this time we will additionally use the MQL5 Algo Forge web interface. For developers who find command-line Git operations challenging, this provides a compromise. In the web interface of MQL5 Algo Forge, we can fork the desired original repository.

Fork is a fundamental concept in version control systems and collaborative development platforms, including MQL5 Algo Forge. It means the process of creating a complete and independent copy of the original repository within the platform.

When a user forks someone else's repository, the platform creates an exact copy under the user's account. This copy inherits all change history, branches, and files from the source project at the time of forking, but from that moment onward it becomes an autonomous repository. The new owner can freely modify it without affecting the original.

Thus, a fork enables any user to build upon an existing project and develop it along their own path, effectively creating a new branch of evolution for the code. This concept enables the creation of derivative projects and alternative implementations within the open-source ecosystem.

Forks are also the primary means of contributing changes to projects where the user does not have direct write access. The standard workflow is as follows: create a fork, implement and test the required changes in it, and then notify the maintainer of the original repository about the proposed improvements via a Pull Request, which we already covered in Part 2. This is the basis of the decentralized collaborative development model.

Despite their independence, forks maintain a technical link to the source repository. This makes it possible to track changes in the original and synchronize them into your fork, merging new commits from the upstream project into your own.

It's important to distinguish between a fork and a simple clone. Cloning refers to creating a local copy of a repository on a specific computer, while a fork is a complete copy on the platform itself, establishing a new remote repository under another user's ownership.

Therefore, once we fork a repository, it becomes our own repository. It also becomes visible in the Shared Projects list in MetaEditor and available for cloning directly through MetaEditor.

### Testing Work with a Fork

Thanks to the kind assistance of [Fernando Carreiro](https://www.mql5.com/en/users/fmic), we were able to test this mechanism in practice. We forked his repository [FMIC](https://www.mql5.com/go?link=https://forge.mql5.io/fmic/FMIC "https://forge.mql5.io/fmic/FMIC"), and at the same time added the original repository to our Watch and Starred lists in the MQL5 Algo Forge web interface.

![](https://c.mql5.com/2/169/455643250829.png)

As expected, the fork appeared in the list of repositories displayed under Shared Projects in MetaEditor:

![](https://c.mql5.com/2/169/2206397848340.png)

This allowed us to successfully clone our newly created [fork](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/FMIC "https://forge.mql5.io/antekov/FMIC") of the [FMIC](https://www.mql5.com/go?link=https://forge.mql5.io/fmic/FMIC "https://forge.mql5.io/fmic/FMIC") repository to the local computer.

Next, we asked Fernando to commit some changes so that we could test how updates would be reflected in our fork. He added a sample README.md file describing the _Heikin Ashi_ publication and committed it to the repository.

Afterwards, in the web interface we indeed saw a notification about the new changes:

![](https://c.mql5.com/2/167/4979927684518.png)

However, these notifications did not yet affect either our fork stored on MQL5 Algo Forge or the local clone on our computer. Let's try to pull Fernando's changes into our repositories. First, we check that the latest changes are indeed missing in our local clone:

![](https://c.mql5.com/2/167/5865087903590.png)

The last commit in our local history was dated August 27, 2025, whereas Fernando's changes were made later.

Now, if we visit our fork in the web interface, we see a message that our _main_ branch is three commits behind the original repository:

![](https://c.mql5.com/2/167/6543144990025.png)

We also see a Sync button, which should synchronize our _main_ folder with the upstream branch. We then check the history of commits and see three new commits dated September 5, 2025, which were absent before:

![](https://c.mql5.com/2/167/5614065099091.png)

In other words, all commits made in the original repository successfully propagated first into our fork on MQL5 Algo Forge, and then into our local clone of that fork.

For those who wish to explore this mechanism in more detail, we recommend consulting the following GitHub documentation sections:

- [Fork a repository](https://www.mql5.com/go?link=https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#configuring-git-to-sync-your-fork-with-the-upstream-repository")
- [Configuring a remote repository](https://www.mql5.com/go?link=https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork")
- [Synchronize](https://www.mql5.com/go?link=https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork")

While this documentation isn't written specifically for MQL5 Algo Forge, much of the web interface behaves similarly, and console Git commands are universally applicable regardless of the hosting platform. Provided, of course, that the platform is based on Git.

For example, following the [upstream configuration](https://www.mql5.com/go?link=https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork "https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork") guidelines, we can set up synchronization so that every Pull/Push operation also updates our fork clone against the original repository:

![](https://c.mql5.com/2/167/6123151674940.png)

However, when working exclusively through MetaEditor and the MQL5 Algo Forge web interface, this extra configuration step is not strictly necessary.

### Forking SmartATR

Let's now return to the repository we originally planned to use. We'll repeat the same steps – creating a fork via the MQL5 Algo Forge web interface and cloning it locally – for the SmartATR repository.

We begin by searching for the original repository in the [Explore](https://www.mql5.com/go?link=https://forge.mql5.io/explore "https://forge.mql5.io/explore") section by entering its name:

![](https://c.mql5.com/2/169/887573748352.png)

Since the repository already has several forks created by other users, the search results also display those forks. To fork the true original, we scroll further down the results and open the page for [steverosenstock/SmartATR](https://www.mql5.com/go?link=https://forge.mql5.io/steverosenstock/SmartATR "https://forge.mql5.io/steverosenstock/SmartATR").

There we click the Fork button:

![](https://c.mql5.com/2/169/5591117392956.png)

After clicking, we are redirected to the fork creation settings page. Here, we can rename the forked repository (as it will appear in our list of repositories), specify which branches from the original should be included, and edit the repository description if desired:

![](https://c.mql5.com/2/169/6497296548080.png)

By default, the fork is created as an exact copy of the original repository. That works perfectly for us, so we simply click "Fork repository".

The fork is successfully created:

![](https://c.mql5.com/2/169/721878904103.png)

Next, we clone this repository to our local computer. Before doing so, we delete the previously cloned original SmartATR folder from the local computer. If MetaEditor was already open, we need to refresh the folder list by selecting _Refresh_ from the _Shared Projects_ context menu. After that, the _SmartATR_ folder appears, and from its context menu we select 'Git Clone':

![](https://c.mql5.com/2/169/1927892487177.png)

The SmartATR project is successfully cloned:

![](https://c.mql5.com/2/169/668841846475.png)

We are now ready to begin making changes.

### Making Changes

Since our goal is to introduce fixes that either resolve or at least neutralize a specific error, we begin by creating a new branch whose name clearly reflects this purpose; for example, fixes/news-impact:

![](https://c.mql5.com/2/169/3000321059806.png)

![](https://c.mql5.com/2/169/1416115595937.png)

We then switch to this branch in the project's context menu by selecting "Git Branches → fixes-news-impact".

![](https://c.mql5.com/2/169/2980043640440.png)

Note that although we originally included a slash ("/") in the branch name, the branch was actually created with that character automatically replaced by a hyphen ("-"). This is a limitation imposed by MetaEditor, which only permits Latin letters and hyphens in branch names. Technically, Git itself allows slashes, and through the web interface we can freely create branches containing them.

Let's test the importance of this restriction. We'll create another branch directly in the MQL5 Algo Forge web interface, this time explicitly including a slash in its name: fixes/cast-warning. From the [Branches](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SmartATR/branches "https://forge.mql5.io/antekov/SmartATR/branches") page, we select New branch, using the _main_ branch as the base:

![](https://c.mql5.com/2/169/4041067775361.png)

The branch is successfully created:

![](https://c.mql5.com/2/169/679274932159.png)

However, when attempting to run a Pull in MetaEditor, we are presented with an error message:

![](https://c.mql5.com/2/169/4539721824261.png)

Even so, the new branch with the slash in its name does appear in the branch list inside MetaEditor, and switching to it works without further issues:

![](https://c.mql5.com/2/169/744979030739.png)

After making note of this peculiarity, we switch back to the branch _fixes-news-impact_ and introduce the temporary fix that removes the cause of the compilation error:

![](https://c.mql5.com/2/169/4007654239939.png)

Once the indicator compiles without errors, we commit our changes through the context menu option 'Git Commit':

![](https://c.mql5.com/2/169/3365587193702.png)

In the commit dialog, we check the list of modified files. The check is simple in this case, as we only changed one file. It is strongly recommended to add a descriptive comment that explains the nature of the fix. After confirming all is correct, we press OK.

![](https://c.mql5.com/2/169/6028555719179.png)

Our changes are now committed and pushed to our fork of the SmartATR repository in MQL5 Algo Forge. At this stage, the corrected version of the indicator can already be used locally, with a safe copy also stored in the repository. Optionally, we could submit a Pull Request to the original project author by pressing 'New pull request' in the repository's web interface:

![](https://c.mql5.com/2/169/3624350909238.png)

However, it's too early, since our modification simply disables part of the functionality rather than improving the code. For now, we do not create a pull request.

The SmartATR indicator is ready to be integrated into our Simple Candles project.

### Integrating the Indicator

Following best practices, we create a new branch in the Simple Candles project repository – _article-19436-forge3_ – based on the _develop_ branch. To vary our approach, we create this branch using the MQL5 Algo Forge web interface.

![](https://c.mql5.com/2/170/5078034278030.png)

To make the branch appear locally, we run 'Git Pull' in MetaEditor and then switch to the new branch _article-19436-forge3_.

Since our intention is to apply the indicator within the trading strategy, we add it directly to the strategy class implementation in _SimpleCandlesStrategy.mqh_. Specifically, we introduce a class field to store the indicator handle:

```
//+------------------------------------------------------------------+
//| Trading strategy using unidirectional candlesticks               |
//+------------------------------------------------------------------+
class CSimpleCandlesStrategy : public CVirtualStrategy {
protected:
   //...

   int               m_iATRHandle;        // SmartATR indicator handle

   //...
};
```

Next, we call _iCustom()_ in the class constructor, passing the required symbol, timeframe, path to the indicator file, and its parameters:

```
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CSimpleCandlesStrategy::CSimpleCandlesStrategy(string p_params) {
// Read the parameters from the initialization string
   // ...

   if(IsValid()) {
      // Load the SmartATR indicator
      m_iATRHandle = iCustom(
                        m_symbol, m_timeframe,
                        "Shared Projects/SmartATR/SmartATR.ex5",
                        // Indicator parameters
                        m_periodATR,   // Initial ATR period (used for first calculation, adaptively changes)
                        false,         // Enable adaptive period (dynamic lookback)
                        7,             // Minimum ATR period (adaptive mode)
                        28,            // Maximum ATR period (adaptive mode)
                        false,         // Weight True Range by volume
                        false,         // Weight True Range by economic news events (MT5 Calendar)
                        2.0,           // Multiplier: alert if ATR exceeds this factor of average
                        false          // Enable pop-up & sound alerts on high volatility
                     );

      // ...
   }
}
```

Note the path specified for the indicator. It starts with _Shared Projects_, then the project folder name _SmartATR_, followed by the indicator filename _SmartATR.ex5_. Including the _.ex5_ extension is optional, but keeping it helps avoid confusion.

There's one important nuance which should be taken into account when working in the _Shared Projects_ folder. This refers to both your own and forked projects. All compiled executables are not placed directly in the repository folder. This is because the Shared Projects folder is located in the terminal data root folder: _MQL5/Shared Projects_. On the one hand, this is good since the version control system will not try to suggest indexing executable files. On the other hand, it can be a bit confusing at first: Where do we find the compiled Expert Advisor and indicator files?

They are actually created in their respective standard folders, such as _MQL5/Experts_ for EAs or _MQL5/Indicators_ for indicators. Within those, a _Shared Projects_ subdirectory is automatically created. So, the compiled files are created right in those subfolders. This means that compiling a file from _MQL5/Shared Projects/SmartATR.mq5_ will produce the executable at _MQL5/Indicators/Shared Projects/SmartATR/SmartATR.ex5._

Accordingly, the _iCustom()_ call must reference the indicator path relative to _MQL5/Indicators._

Finally, we compile the advisor file _SimpleCandles.mq5_ and run it in the strategy tester. The logs show the following:

![](https://c.mql5.com/2/170/4866540050288.png)

Thus, the SmartATR indicator has been successfully loaded, initialized, and is ready for use. At this point, we are only demonstrating its integration. We might add actual use within the strategy logic later. We commit these changes and push them to the MQL5 Algo Forge repository.

### **Conclusion**

This article demonstrates how adopting MQL5 Algo Forge enables a fundamentally more flexible workflow for developers. Previously, we only examined self-contained repositories, but here we have successfully integrated an external library from a third-party repository into our project.

The key moment was the correct workflow based on forking – creating a personal copy of an external repository that allows full modification while staying synchronized with the upstream project. The successful integration of SmartATR into Simple Candles supports this approach – from locating and forking the repository, to modifying and applying its code in a live trading strategy.

Importantly, this process was achieved entirely with current MetaEditor capabilities, without waiting for future updates. The limitations of MetaEditor (such as lack of direct access to third-party repositories and restricted branch naming) are easily overcome by complementing it with the MQL5 Algo Forge web interface and, if necessary, standard Git console commands. In short, the system is already viable for practical use, and the remaining interface shortcomings are inconveniences rather than blockers.

However, let's not stop there. We will continue using repositories to separate projects and our sharing experiences gained along the way.

Thank you for your attention! See you soon!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/19436](https://www.mql5.com/ru/articles/19436)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**[Go to discussion](https://www.mql5.com/en/forum/496061)**

![The MQL5 Standard Library Explorer (Part 1): Introduction with CTrade, CiMA, and CiATR](https://c.mql5.com/2/171/19341-the-mql5-standard-library-explorer-logo.png)[The MQL5 Standard Library Explorer (Part 1): Introduction with CTrade, CiMA, and CiATR](https://www.mql5.com/en/articles/19341)

The MQL5 Standard Library plays a vital role in developing trading algorithms for MetaTrader 5. In this discussion series, our goal is to master its application to simplify the creation of efficient trading tools for MetaTrader 5. These tools include custom Expert Advisors, indicators, and other utilities. We begin today by developing a trend-following Expert Advisor using the CTrade, CiMA, and CiATR classes. This is an especially important topic for everyone—whether you are a beginner or an experienced developer. Join this discussion to discover more.

![Overcoming The Limitation of Machine Learning (Part 4): Overcoming Irreducible Error Using Multiple Forecast Horizons](https://c.mql5.com/2/171/19383-overcoming-the-limitation-of-logo__1.png)[Overcoming The Limitation of Machine Learning (Part 4): Overcoming Irreducible Error Using Multiple Forecast Horizons](https://www.mql5.com/en/articles/19383)

Machine learning is often viewed through statistical or linear algebraic lenses, but this article emphasizes a geometric perspective of model predictions. It demonstrates that models do not truly approximate the target but rather map it onto a new coordinate system, creating an inherent misalignment that results in irreducible error. The article proposes that multi-step predictions, comparing the model’s forecasts across different horizons, offer a more effective approach than direct comparisons with the target. By applying this method to a trading model, the article demonstrates significant improvements in profitability and accuracy without changing the underlying model.

![Automating Trading Strategies in MQL5 (Part 33): Creating a Price Action Shark Harmonic Pattern System](https://c.mql5.com/2/171/19479-automating-trading-strategies-logo.png)[Automating Trading Strategies in MQL5 (Part 33): Creating a Price Action Shark Harmonic Pattern System](https://www.mql5.com/en/articles/19479)

In this article, we develop a Shark pattern system in MQL5 that identifies bullish and bearish Shark harmonic patterns using pivot points and Fibonacci ratios, executing trades with customizable entry, stop-loss, and take-profit levels based on user-selected options. We enhance trader insight with visual feedback through chart objects like triangles, trendlines, and labels to clearly display the X-A-B-C-D pattern structure

![Automating The Market Sentiment Indicator](https://c.mql5.com/2/171/19609-automating-the-market-sentiment-logo__1.png)[Automating The Market Sentiment Indicator](https://www.mql5.com/en/articles/19609)

In this article, we automate a custom market sentiment indicator that classifies market conditions into bullish, bearish, risk-on, risk-off, and neutral. The Expert Advisor delivers real-time insights into prevailing sentiment while streamlining the analysis process for current market trends or direction.

[Need a reliable hosting solution for your robots?Contact your broker and find out about available Sponsored MetaTrader VPS offeringsLearn more![](https://www.mql5.com/ff/sh/0pw0dk81s56qy774z2/01.png)](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/449311&a=vljwvezfjkfbvviocwskggexlvgykvob&s=70cf8e354b9a125332533ffb65d7365abe8dde5b5c1ede9caac479a9e9df4f25&uid=&ref=https://www.mql5.com/en/articles/19436&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5068061684507014407)

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