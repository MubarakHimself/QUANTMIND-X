---
title: Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories
url: https://www.mql5.com/en/articles/17698
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:18:10.976695
---

[![](https://www.mql5.com/ff/si/6pp0j40fqxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=luckhiizjxvmvgigcufevttapwwrwbld&s=08cd1d929f27358481aded3c1c5f4e75a9bd5f52c477127afef2a5c532aec5c5&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=fafvqxjbexysmgifnktzldbcpyjkuzbk&ssn=1769177887409642236&ssn_dr=0&ssn_sr=0&fv_date=1769177887&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17698&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Moving%20to%20MQL5%20Algo%20Forge%20(Part%202)%3A%20Working%20with%20Multiple%20Repositories%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917788795961438&fz_uniq=5068071077600490783&sv=2552)

MetaTrader 5 / Integration


### Introduction

In the first [article](https://www.mql5.com/en/articles/17646), we began transitioning from the built-in SVN-based MQL5 Storage in MetaEditor to a more flexible and modern solution based on the Git version control system: [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/"). The main reason for this step was the need to fully leverage repository branches while working on multiple projects or on different functionalities within a single project.

The transition started with the creation of a new repository in MQL5 Algo Forge and the setup of a local development environment using Visual Studio Code, along with the necessary MQL5 and Git extensions and supporting tools. We then added a .gitignore file to the repository to exclude standard and temporary files from version control. All existing projects were uploaded into a dedicated _archive_ branch, designated as an archival storage of all previously written code. The _main_ branch was left empty and prepared for organizing new project branches. In this way, we laid the foundation for distributing different project codes across separate branches of the repository.

However, since the publication of the first article, MetaEditor has significantly expanded its support for the new repository system. These changes encourage us to reconsider the previously outlined approach. Therefore, in this article, we will deviate slightly from the original plan and explore how to create a public project that integrates other public projects as components. The project we will focus on is the development of a multi-currency Expert Advisor. Several [articles](https://www.mql5.com/ru/blogs/post/756958) have already been published that describe the approaches to the code development and modifications for this project. Now, we will take full advantage of Git version control to organize and streamline the development process.

### Mapping out the path

It's hard to believe, but at the time of the previous article, MetaEditor did not yet include the "Git: menu or file context menu commands for working with MQL5 Algo Forge repositories. As a result, considerable effort was required to configure workflows using external tools like Visual Studio Code. Since we didn't know how MetaEditor would eventually implement repository support, we had to use the tools available at that time.

Since then, new MetaEditor releases have introduced built-in support for MQL5 Algo Forge. MetaQuotes has also published a new article, " [Getting Started with MQL5 Algo Forge](https://www.mql5.com/en/articles/18518)", which explains the basics and demonstrates key features. However, the most important development, in our view, is the implementation of Shared Projects in MetaEditor.

Why is this significant? Previously, we knew that the MQL5 folder would act as a repository hosted on the Git servers of MQL5 Algo Forge. Apparently, this repository would have the fixed name _mql5_. This meant that each user would have a repository named _mql5_ in Algo Forge. This repository would then be cloned into the _MQL5_ folder after installing a new terminal, logging into the Community, and connecting the repository. At the same time, MQL5 Algo Forge has always allowed the creation of additional repositories. More precisely, not additional, but separate ones, not related to the _mql5_ repository. Naturally, this raised a question: how would MetaEditor handle these other repositories?

Would users be able to select which repository to use in each installation of the terminal? Or not? Would only the mql5 repository be supported, with users forced to separate their work into branches for different projects? We initially prepared for this worst-case scenario. Managing multiple projects across branches in a single repository is not particularly convenient. Fortunately, our concerns proved unfounded.

MetaQuotes introduced a more elegant solution that effectively solves two problems at once. On one hand, we have the main repository named _mql5_. This works well for those already accustomed to MQL5 Storage. They can now continue using version control without worrying about which version control system lies underneath.

On the other hand, all other user repositories are now available as folders inside Shared Projects. This provides a new standard root folder, alongside existing (ones such as _MQL5_, _MQL5/Experts_, _MQL5/Include_, etc.), intended for storing code from the user's additional repositories.

Let's consider an example. Suppose we have two separate repositories in MQL5 Algo Forge, neither of which is the default mql5. The first repository ( [Adwizard](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/Adwizard.git "https://forge.mql5.io/antekov/Adwizard.git")) contains only library code, i.e. only _\*.mqh_ include files, without any _\*.mq5_ files that could be compiled into an Expert Advisor or indicator. The second repository ( [Simple Candles](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles.git "https://forge.mql5.io/antekov/SimpleCandles.git")) contains _\*.mq5_ files that use the include files from the first repository. For simplicity, we'll refer to the first repository as the library repository and the second as the project repository.

Our goal is to determine how to use code from the library repository while developing within the project repository. This scenario could become increasingly common if, for instance, code shared in the mql5.com Code Base is also mirrored by its authors in MQL5 Algo Forge as public repositories. In such cases, linking one or more repositories as external libraries to a project could be handled in the same way we are about to explore in this article.

### Getting Started

Let's first look at the situation from the perspective of the developer who owns both repositories. This means we can freely make changes to the code in either repository without waiting for a review and approval via the Pull Request mechanism. We begin by creating a clean terminal folder and copying two files from any previously installed copy of MetaTrader 5:

![Forge MQL5](https://c.mql5.com/2/169/014.png)

To avoid searching for the terminal's working folder deep within the file system, we recommend running the terminal in Portable mode. In Windows, one way to do this is to create a shortcut to the terminal's executable file and add the /portable flag to the target field in the shortcut properties.

After launching the terminal, open a new demo account just in case, update to the latest version, sign in to the Community, and then launch MetaEditor by pressing F4. Connect MQL5 Algo Forge, if it hasn't already connected automatically.

In the Navigator, we now see the 'Shared Projects' folder, which lists the repositories we previously created through the web interface. However, if we open this folder in Explorer, it is still empty. This means that the actual repository files have not yet been downloaded to our computer.

![Portable](https://c.mql5.com/2/169/017__1.png)

To clone them, right-click each of the needed repositories and select "Git Clone" from the context menu. The log confirms successful cloning of both Adwizard and Simple Candles. The folders with the cloned repositories now appear in Explorer:

![](https://c.mql5.com/2/166/2067792536991.png)

At this point, the code for both projects is available locally and ready to use.

### First Problem

Let's open _SimpleCandles.mq5_ and try compiling it:

![](https://c.mql5.com/2/167/347763823853.png)

As expected, compilation errors occur. Let's try to understand their reasons. These are not critical, since we know the code compiled successfully before. The only thing that has changed is the relative placement of the library and project files. The first two fundamental errors come from the fact that the compiler cannot find the library files where it expects them. Back in [Part 28](https://www.mql5.com/en/articles/17608#para5), we agreed to use the following folder structure to store the library and projects parts:

![](https://c.mql5.com/2/166/6376036121922.png)

That is, we store the library repository inside a subfolder of the project repository. This gave us a predictable structure for locating library files. This time, however, we'll change this. Instead of using a mandatory _Include_ subfolder inside the project, we will use the _MQL5/Shared Projects_ folder. Ideally, this folder will remain stable and continue serving the same purpose in future MetaEditor versions.

To fix the issue, we'll update the #include directives in two files. But before making any code changes, let's follow good development practice and create a separate branch for this isolated task. Once the fixes are tested, we can merge the branch back into the main development branch.

Let's see what branches we already have in the project repository. This can be done in several ways:

- Via the repository folder's context menu in MetaEditor:

![](https://c.mql5.com/2/167/1714533181010.png)

- Through the web interface on the repository's [branches](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles/branches "https://forge.mql5.io/antekov/SimpleCandles/branches") page:

![](https://c.mql5.com/2/166/1109011303079.png)

- Using an external Git tool such as Visual Studio Code. Next to the repository name we see _main_, which is the name of the current branch. A click on it shows a list of available branches (and menu items for creating new branches):

![](https://c.mql5.com/2/166/2306114561845.png)

![](https://c.mql5.com/2/166/867224697517.png)

Currently, the repository has four branches:

- **main** — the primary branch. It is created with the repository. In the simplest case, all work can be done here without creating any additional branches. In more complex cases, this branch is used to store the states of the files that provide stable versions of the code. Any ongoing changes that have not yet been completed and tested are made in other branches.
- **develop** — the development branch. In simple cases, it can be used as the only branch to add changes and implement new features. This option is sufficient if new features are implemented sequentially. That is, we do not start implementing new functionality until the project is in a fully functional and stable state after we added the previous features. Before starting work on a new feature, branches are merged: edits made in the _develop_ branch are merged into _main_. If multiple features are being developed simultaneously, it becomes inconvenient to work in one development branch. In such cases, additional feature branches can be created.
- The examples or such branches are **article-17608-close-manager** and **article-17607**. The former is a feature branch for position-closing logic based on profit/loss thresholds. This branch is already merged into _develop_ and _develop_ is then merged into _main_. The second feature branch is used for enhancements to automated optimization. It is still in progress, not yet merged into _develop_ or _main_.

It's important to stress that Git does not enforce any specific branch usage rules. So, we can choose the option that is most convenient for us. There are certain workflows that some developers have found useful and shared with others. This is how "Best Practices" appear. You're free to adopt or adapt whichever branching model suits your project. As an example, take a look at one of the proposed branching principles described in this [article](https://www.mql5.com/go?link=https://nvie.com/posts/a-successful-git-branching-model/ "https://nvie.com/posts/a-successful-git-branching-model/").

Now let's get back to our repository.

One detail that may raise questions is the prefix _origin/_ (or _refs/remotes/origin/_ as shown in MetaEditor). This prefix simply indicates that the branch exists in the remote repository, not just locally. Typically, we keep local and remote branches in sync. In MetaEditor, running a Commit command automatically triggers a Push command, and thus the commit is sent to the remote repository.

If commits are made outside MetaEditor, it's possible to commit locally without pushing. In such cases, the local and remote branches with the same name may be different. The _origin/_ prefix helps distinguish between them. With this prefix, we mean a branch in a remote repository; otherwise, it's a local one.

### Creating a New Branch

Since the planned edits only ensure the code compiles correctly after the placement of its library part changed, we'll base a new branch generated from _develop_. We first switch to _origin/develop_, after which it appears in the list as a local _develop_ branch.

![](https://c.mql5.com/2/167/3452336236645.png)

Then we create a new branch (execute the New command) and enter the desired name. Following our convention, article-related branches begin with article- plus the article's unique identifier. This can optionally be followed by a short suffix describing the topic of the article. That's why the new branch is named "article-17698-forge2".

We could create a branch in other ways too: using the web interface, the Visual Studio Code interface, or the command line interface. From the command line in the repository's root folder, we can run:

git checkout -b article-17698-forge2 develop

This tells Git to switch (checkout) to a new branch (-b) called _article-17698-forge2_, based on the _develop_ branch.

If a branch is created outside the web interface, it will exist only on our local machine until the first push to the remote repository. The opposite is also true: if a branch is created in the remote repository through the web interface, it will not appear on our local machine until the first pull from the remote repository.

You can push changes like this:

![](https://c.mql5.com/2/167/3550352901607.png)

or like this:

![](https://c.mql5.com/2/167/4732069076380.png)

The console command for the Push operation, when it includes the creation of a new branch, must contain additional parameters confirming that we truly want the branch to be created in the remote repository:

git push --set-upstream origin article-17698-forge2

After this, the branch exists both in the local copy of the repository and in the remote repository hosted in MQL5 Algo Forge. At this point, we can start making edits without fear of breaking the functionality of other branches.

### Making Changes

The required modifications are very simple. In the file _SimpleCandles.mq5_, we update the line that includes a file from the _Adwizard_ library. Since the root folders of the _Simple Candles_ and _Adwizard_ repositories are now located at the same level inside the _Shared Projects_ folder, the path to _Expert.mqh_ must first move one level up (../) before descending into the subfolders of the library repository:

```
#include "Include/Adwizard/Experts/Expert.mqh"
#include "../Adwizard/Experts/Expert.mqh"
```

A similar adjustment is needed in the file _Strategies/SimpleCandlesStrategy.mqh_:

```
#include "../Include/Adwizard/Virtual/VirtualStrategy.mqh"
#include "../../Adwizard/Virtual/VirtualStrategy.mqh"
```

After these changes, _SimpleCandles.mq5_ compiles successfully again. We can now commit the changes to the repository:

![](https://c.mql5.com/2/167/2088166288605.png)

As mentioned earlier, when committing through the MetaEditor interface, the Push command is executed automatically, sending the changes to the remote repository in MQL5 Algo Forge.

When working with console commands, we can achieve the same result as follows. If, in addition to editing existing files, we also created new ones, we first need to add them to the repository index:

git add .

This command adds all new files found in the repository folder. To add only specific files, replace the dot (.) with their names. After that, we commit the changes with a specified comment and push them to the remote repository:

git commit -m "Fix relative paths to include files from Adwizard"

git push

At this point, the changes in the _article-17698-forge2_ branch are complete. It can be merged into the _develop_ branch and then closed.

### Second Problem

Here we run into an unpleasant surprise. MetaEditor currently lacks tools for merging branches. In other words, we can now create new branches, but we cannot transfer changes from one branch to another! Hopefully, this functionality will be added in the near future. For now, we must once again turn to alternative tools to perform repository operations.

There are two main ways we can merge branches. The first is to use the merge interface in Visual Studio Code or standard console commands. For our case, the following commands can be used:

git checkout develop

git pull

git merge --no-ff article-17698-forge2

First, we switch to the _develop branch_. Then, as a precaution, we update it (in case changes were made that have not yet reached our local machine). Finally, the last command performs the actual merge. Merge conflicts are possible, but in our scenario, they are unlikely, since we are still considering the case of a single developer working on the project Even when working from multiple locations, as long as we regularly update the repositories to the latest state, conflicts should not arise.

However, let us not dwell on the nuances of this method. Instead, we will take a closer look at the second approach. Here we use the MQL5 Algo Forge web interface.

### Using Pull Requests for Merging

Like other Git-based platforms (such as GitHub, GitLab, or Bitbucket), MQL5 Algo Forge also includes a mechanism called a Pull Request (PR).

A Pull Request allows a developer to propose changes made in a branch to be merged into a repository. In other words, creating a PR is a way of notifying the repository owner and other contributors: _"I’ve completed work in my branch, please review and merge these changes into the main branch (main/master/develop)."_

A PR is not a feature of Git itself but an additional layer built on top of it. It organizes the process of code review and discussion before changes are merged into the main branch.

Pull Requests also solve several other critical tasks in modern development: continuous integration (CI) with automated tests, quality control by other developers, and documentation of changes in the form of PR comments explaining why certain modifications were made. However, these practices are most relevant for team-based projects, while MQL5 projects are usually individual ones. Anyway, the workflow may become more important as collaborative projects emerge in the future.

That said, we have already replicated the start of a typical workflow for adding new functionality or fixes using PRs:

1. **Pull the latest changes**. Before starting work, we updated the local _develop_ branch.

2. **Create a new branch for the task**. From the updated _develop_ branch, we created a branch with a clear name _article-17698-forge2_.

3. **Make changes in the new branch**. We modified and tested several files, then committed the changes.


The next steps are as follows.

4. Create a **Pull Request**. In the MQL5 Algo Forge web interface, navigate to the Pull Requests tab and click the large red 'New pull request' button.


**![](https://c.mql5.com/2/167/162327984614.png)**

This opens the branch selection page. At this stage, the PR has not yet been created; first, we must define where the changes will be merged. Once the branches are selected, we can review the list of changes. Then, click 'New pull request' again.

![](https://c.mql5.com/2/167/6457360194609.png)

A new page opens where we can provide a detailed description of the changes. Here, we can also assign reviewers. By default, the request is directed to ourselves, which is exactly what we need in this case.

![](https://c.mql5.com/2/167/1834186545882.png)

5. **Review and discussion**. Since we are working alone, we can skip this step. Normally, this stage involves:

   - reviewers examining the code and leaving comments (general or tied to specific lines),

   - the PR author responding to comments and making corrections in the same branch,

   - and all new pushes automatically being added to the existing PR.

6. **Merging**. After reviewers approve (if any) and CI passes (if configured), the PR can be merged. Typically, there are several merge options:


   - **Merge commit:** Creates a separate merge commit, preserving the branch history.

   - **Squash and merge:** Combines all PR commits into a single commit added to the target branch. Useful to avoid cluttering the history with minor commits like "fixed typo".

   - **Rebase and merge:** Reapplies PR commits on top of the target branch (in our case it's _develop_). This produces a clean, linear history.

For our purposes, we'll choose the first option, since we want to preserve the full commit history. So, click 'Create merge commit'.

![](https://c.mql5.com/2/167/899572206155.png)

Now comes the final page of Pull Request operations. Here we check the option 'Delete branch ...' to close the temporary development branch. The commit history will still reflect that the branch existed. But keeping it open serves no purpose since we've achieved our goal. For future changes that solve other tasks, we will create new branches. This way, the repository's branch list always provides a clear snapshot of currently ongoing parallel work.

Leave the rest of the settings as they are and click 'Create merge commit'.

![](https://c.mql5.com/2/167/764065837304.png)

The process is now complete: the _article-17698-forge2_ branch has been merged into _develop_ and deleted:

![](https://c.mql5.com/2/167/3520817170770.png)

In general, using Pull Requests even in your own repository is a good and recommended practice, even for solo projects. Before merging, you can visually review all changes, often catching things missed during commits: unnecessary comments, stray files, or non-optimal edits. In essence, this is a form of self-discipline. Moreover, adopting this workflow builds good habits (branching, code review, etc.). So if you later join a development team, this process will already be familiar to you.

So yes - submitting PRs to yourself is not only possible but encouraged for any serious project. It improves code quality and enforces discipline. Of course, for quick fixes consisting of one or two commits, nothing prevents you from merging directly with git merge. But for larger changes, a PR is the better approach.

### Conclusion

Overall, the workflow with personal repositories is now well established. We have covered the path from cloning repositories to refining a process where changes leave the code functional and ready for further development. The project repository can now meaningfully use code from another repository (or several) as libraries.

However, we have only considered one scenario: when the same user owns both the project and library repositories. The other scenario is when the project owner wants to use someone else's repositories as libraries. This case is not as simple. And yet, this kind of workflow, with active reuse of community code and collaboration, was one of the stated goals behind moving to the new repository system. Nevertheless, the foundation has been laid.

We will stop here for now. See you in the next part!

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17698](https://www.mql5.com/ru/articles/17698)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/495205)**
(21)


![Vladislav Boyko](https://c.mql5.com/avatar/2025/12/692e1587-6181.png)

**[Vladislav Boyko](https://www.mql5.com/en/users/boyvlad)**
\|
10 Sep 2025 at 00:04

**Vladislav Boyko [#](https://www.mql5.com/ru/forum/495034/page2#comment_57998024):**

The Cyrillic alphabet was hardly there - I gave up using it even in comments a long time ago.

Apparently, I was mistaken. I just added this to the .mq5 file with UTF-8 encoding:

```
// Cyrillic
```

and after saving the file encoding changed to "UTF-16 LE BOM".

* * *

It seems to be MetaEditor's fault. I added Cyrillic characters and saved the file using Notepad++ and the encoding remained UTF-8.

![](https://c.mql5.com/3/474/5772329205076.png)

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
10 Sep 2025 at 00:12

**Vladislav Boyko [#](https://www.mql5.com/ru/forum/495034/page2#comment_57997808):**

I also find it strange to argue for the need to be able to remove branches from the local repository. Given that Git's branching model is its killer feature and Git encourages frequent creation, deletion and merging of branches.

So I'm also in favour of deleting branches after they merge with the main branches. It's just the first time I've heard that after deletion, they create a branch for the new fiche with the same name, not a new one.

What is the purpose of watching diff?

Yes, it is a very necessary thing. I actively use it too, but only in VS Code. And there, strangely enough, there are no crashes there, though I look through files with "bad" encoding.

I sometimes check files with [that script](https://www.mql5.com/ru/forum/495034#comment_57996395) before  commit. And, as it turned out, not for nothing - there were cases when normal files suddenly changed encoding. Perhaps after inserting something from the clipboard, I don't know for sure.

I've never encountered such a thing. It's pretty unexpected, too. Maybe the breakage of normal files was due to simultaneous use of different ME builds to work with the same files? I don't know...

I looked at the commit history, that files added two months ago do indeed already have UTF-8 encoding, and files added three months ago are still UTF-16 LE. Apparently there was a switch to UTF-8 encoding somewhere around that time.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
10 Sep 2025 at 00:17

**Vladislav Boyko [#](https://www.mql5.com/ru/forum/495034/page2#comment_57998355):**

I guess I was wrong. I just added this to the .mq5 file with UTF-8 encoding:

and after saving the file encoding changed to "UTF-16 LE BOM".

* * *

It seems to be MetaEditor's fault. I added Cyrillic characters and saved the file using Notepad++ and the encoding remained UTF-8.

I confirm, after adding Russian letters and saving the file the encoding changes from UTF-8 to UTF-16 LE. If all Russian letters are removed and saved, it still remains UTF-16 LE.

![Vladislav Boyko](https://c.mql5.com/avatar/2025/12/692e1587-6181.png)

**[Vladislav Boyko](https://www.mql5.com/en/users/boyvlad)**
\|
10 Sep 2025 at 00:32

**Vladislav Boyko [#](https://www.mql5.com/ru/forum/495034/page2#comment_57998355):**

It seems to be MetaEditor's fault.

Here is a proof that you can make UTF-8, Cyrillic and Git compatible:

[https://forge.mql5.io/junk/utf8-cyrillic-test/commit/e87d37b02e88d44305dea0f7f6630c6173471aea](https://www.mql5.com/go?link=https://forge.mql5.io/junk/utf8-cyrillic-test/commit/e87d37b02e88d44305dea0f7f6630c6173471aea "https://forge.mql5.io/junk/utf8-cyrillic-test/commit/e87d37b02e88d44305dea0f7f6630c6173471aea")

All you need to do is to ask MetaEditor not to change the file encoding.

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
10 Sep 2025 at 18:48

**Vladislav Boyko [#](https://www.mql5.com/ru/forum/495034/page2#comment_57998355):**

I guess I was wrong. I just added this to the .mq5 file with UTF-8 encoding:

and after saving the file encoding changed to "UTF-16 LE BOM".

* * *

It seems to be MetaEditor's fault. I added Cyrillic characters and saved the file using Notepad++ and the encoding remained UTF-8.

Most likely, UTF-8 was without BOM, ME doesn't like it. At least it used to leave files in UTF-8 only if BOM was present. Other editors are smarter and work without BOM.

![Automating Trading Strategies in MQL5 (Part 31): Creating a Price Action 3 Drives Harmonic Pattern System](https://c.mql5.com/2/169/19449-automating-trading-strategies-logo__2.png)[Automating Trading Strategies in MQL5 (Part 31): Creating a Price Action 3 Drives Harmonic Pattern System](https://www.mql5.com/en/articles/19449)

In this article, we develop a 3 Drives Pattern system in MQL5 that identifies bullish and bearish 3 Drives harmonic patterns using pivot points and Fibonacci ratios, executing trades with customizable entry, stop loss, and take-profit levels based on user-selected options. We enhance trader insight with visual feedback through chart objects.

![Self Optimizing Expert Advisors in MQL5 (Part 14): Viewing Data Transformations as Tuning Parameters of Our Feedback Controller](https://c.mql5.com/2/168/19382-self-optimizing-expert-advisors-logo.png)[Self Optimizing Expert Advisors in MQL5 (Part 14): Viewing Data Transformations as Tuning Parameters of Our Feedback Controller](https://www.mql5.com/en/articles/19382)

Preprocessing is a powerful yet quickly overlooked tuning parameter. It lives in the shadows of its bigger brothers: optimizers and shiny model architectures. Small percentage improvements here can have disproportionately large, compounding effects on profitability and risk. Too often, this largely unexplored science is boiled down to a simple routine, seen only as a means to an end, when in reality it is where signal can be directly amplified, or just as easily destroyed.

![Dynamic mode decomposition applied to univariate time series in MQL5](https://c.mql5.com/2/169/19188-dynamic-mode-decomposition-logo.png)[Dynamic mode decomposition applied to univariate time series in MQL5](https://www.mql5.com/en/articles/19188)

Dynamic mode decomposition (DMD) is a technique usually applied to high-dimensional datasets. In this article, we demonstrate the application of DMD on univariate time series, showing its ability to characterize a series as well as make forecasts. In doing so, we will investigate MQL5's built-in implementation of dynamic mode decomposition, paying particular attention to the new matrix method, DynamicModeDecomposition().

![Developing a Custom Market Sentiment Indicator](https://c.mql5.com/2/168/19422-developing-a-custom-market-logo.png)[Developing a Custom Market Sentiment Indicator](https://www.mql5.com/en/articles/19422)

In this article we are developing a custom market sentiment indicator to classify conditions into bullish, bearish, risk-on, risk-off, or neutral. Using multi-timeframe, the indicator can provide traders with a clearer perspective of overall market bias and short-term confirmations.

[![](https://www.mql5.com/ff/si/x6w0dk14xy0tf97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F586%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dhow.test.expert%26utm_content%3Dbuy.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=repptjucjbnrxhoeoqbekpbncvsnhylz&s=3da978a0c510a6306b46ee79cdf8418a5c0da5e081f296e18b262b00031a2310&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nprirkgoptaiaiyujgvrpmqxtuxgfcby&ssn=1769177887409642236&ssn_dr=0&ssn_sr=0&fv_date=1769177887&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17698&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Moving%20to%20MQL5%20Algo%20Forge%20(Part%202)%3A%20Working%20with%20Multiple%20Repositories%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917788795981852&fz_uniq=5068071077600490783&sv=2552)

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