---
title: Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases
url: https://www.mql5.com/en/articles/19623
categories: Integration
relevance_score: 9
scraped_at: 2026-01-22T17:40:22.097333
---

[![](https://www.mql5.com/ff/sh/6xjc81sb5f2g45z9z2/01.png)Follow MQL5.community on social mediaWe publish the best technical materials from experts – free from advertising and irrelevant contentLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/455636%3Futm_source=www.mql5.com%26utm_medium=display%26utm_content=follow.channel%26utm_campaign=AAA380.mql5.socials&a=yexgeaiatphxecqagtoxizolvboismyb&s=4e531fd1f983c26570e2dac7588b735354f2f9e0aea561427c030e4a1d2f060b&uid=&ref=https://www.mql5.com/en/articles/19623&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5049293424747718868)

MetaTrader 5 / Examples


### Introduction

Our transition to MQL5 Algo Forge continues. We have set up our workflow with personal repositories and have turned to one of the main reasons for this move – the ability to easily use community-contributed code. In [Part 3](https://www.mql5.com/en/articles/19436), we explored how to add a public library from another repository to our own project.

The experiment with connecting the SmartATR library to the SimpleCandles Expert Advisor clearly demonstrated that simple cloning is not always convenient, especially when the code requires modifications. We, instead, followed the proper workflow: we created a fork, which became our personal copy of someone else's repository for fixing bugs and making modifications, while preserving the option to later propose these changes to the author via a Pull Request.

Despite certain limitations we encountered within the MetaEditor interface, combining it with the MQL5 Algo Forge web interface allowed us to successfully complete the entire chain of actions, from cloning to committing edits and finally linking the project with an external library. Thus, we solved a specific task and examined a universal template for integrating any third-party components.

In today's article, we will take a closer look at the stage of publishing the edits made in the repository, a certain set of changes that form a complete solution, whether it's adding new functionality to a project or fixing a discovered issue. This is the process of committing or releasing a new product version. We will see how to organize this process and what capabilities MQL5 Algo Forge provides for it.

### Finding a Branch

In previous parts, we recommended using a separate repository branch for making a set of edits that address a specific task. However, after completing work on such a branch, it's best to merge it into another (main) branch and then delete it. Otherwise, the repository can quickly turn into a dense thicket where even the owner could easily get lost. Therefore, obsolete branches should be removed. But sometimes, you may need to revert the code to the state it was in just before a certain branch was deleted. How to do this?

First, let's clarify that a branch is simply a sequence of commits arranged chronologically. Technically, a branch is a pointer to a commit considered the latest in a chain of consecutive commits. Therefore, deleting a branch does not delete the commits themselves. At worst, they might be reassigned to another branch or even merged into a single summary commit; but in any case, they continue to exist in the repository (with rare exceptions). Thus, returning to the state "before deleting a branch" essentially means reverting to one of the commits that exist in some branch. The question then becomes: how do we find that commit?

Let's look at the state of the SimpleCandles repository after the changes mentioned in Part 3 were made:

![](https://c.mql5.com/2/170/2313440626744.png)

We can see the history of commits and a colored visualization of the relationships between branches on the left. Each commit is identified by its hash (or more precisely, part of it), i.e. a large unique number that distinguishes it from all others. To shorten its representation, the hash is displayed in hexadecimal form (for example, _b04ebd1257_).

Such a commit tree can be viewed for any repository on a dedicated [page](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles/graph "https://forge.mql5.io/antekov/SimpleCandles/graph") of the MQL5 Algo Forge web interface. The screenshot shown was taken some time ago, so visiting this page now will show a slightly different picture: new commits will have appeared in the tree, and the interweaving of branches will have changed due to additional merge commits.

We can also see branch names next to some commits. These are displayed for the most recent commits in each branch. In the provided screenshot, we can count six different branches: _main_, _develop_, _convert-to-utf8_, _article-17608-close-manager_, _article-17607_, and _article-19436-forge3_. The last one is the branch used for changes made while writing Part 3. But when working on Part 2, we also created a separate branch for the planned changes. It was named _article-17698-forge2_, and it has since been deleted, which is why no commit now carries this branch name. So, where can we find it?

If we look at the full commit message for _58beccf233_, it mentions this branch name and indicates that it was merged into _develop_.

![](https://c.mql5.com/2/170/1789851317979.png)

So, we have found the desired commit, but locating it this way is not convenient. Moreover, if we had merged branches manually using console commands like _'_ git merge' instead of via a Pull Request, we could have written any arbitrary comment for the merge commit. In that case, finding the right commit would have been even harder, since the branch name might not have been included in the message at all.

Now that we've found the desired commit, we can switch to it, restoring our local repository to the state it was in right after that commit. To do this, we can use the commit hash in the 'git checkout' command. However, there are some nuances here. If we try to switch to this commit in MetaEditor by selecting it from the history opened via the project's context menu option "Git Log":

![](https://c.mql5.com/2/172/5708360104884.png)

... we'll encounter an error message:

![](https://c.mql5.com/2/172/1430948056159.png)

Perhaps, there's a reason for that. Let's take a closer look at what's going on. We'll start by introducing the new concepts of "tag" and "HEAD pointer".

### Tags

A tag in the Git version control system is an additional name assigned to a specific commit. You can also think of a tag as a pointer or reference to a particular version of the code in the repository, since it directly points to a specific commit. Using a tag allows you to return at any time to the exact state of the code that corresponds to the tagged commit. Tags are helpful for marking important milestones in a project development, such as version releases, completion stages, or stable builds. In the MQL5 Algo Forge web interface, you can view all tags of a selected repository on a separate [page](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/Adwizard/tags "https://forge.mql5.io/antekov/Adwizard/tags").

There are two types of tags in Git: lightweight and annotated. Lightweight tags contain only a name, while annotated tags can include additional information such as the author, date, comments, and even a signature. In most cases, lightweight tags are used.

To create a tag via the web interface, you can open the page of any commit (for example, [this one](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/Adwizard/commit/2ca48e5ccdbe988ecd2e6fb812dd192df0bd9126 "https://forge.mql5.io/antekov/Adwizard/commit/2ca48e5ccdbe988ecd2e6fb812dd192df0bd9126")), click the 'Operations' button, and select 'Create tag'.

![](https://c.mql5.com/2/172/5525564759803.png)

However, we'll return to tag creation a bit later.

To create a tag via Git console commands, you use the 'git tag' command. To create a lightweight tag, simply specify the name of the tag:

git tag <tag-name>

\# Example

git tag v1.0

To create an annotated tag, you'll need to add some extra parameters:

git tag -a <tag-name> -m "Tag description"

\# Example:

git tag -a v1.0 -m "Version 1.0 release"

In addition to marking versions of code intended for publication or deployment (releases), tags can also be used to signal CI/CD pipelines to trigger predefined actions when a commit with a certain tag appears, or to mark significant development milestones, such as the completion of major features or the fixing of critical bugs, even if they don't represent a new version release.

### The HEAD pointer

Having discussed tags, it's worth mentioning another important concept – the HEAD pointer. Its behavior is similar to a tag with a fixed name HEAD, which automatically moves to the latest commit in the currently checked-out branch. HEAD is often referred to as the "marker of the current branch" or "pointer to the active branch". It essentially answers the question: "Where are we in the repository right now?" However, it is not technically a tag.

Physically, this pointer is stored in the .git/HEAD file in the repository. The contents of HEAD may contain either a symbolic reference (a tag or branch name) or a commit hash. When switching between branches, the HEAD pointer automatically updates to point to the latest commit in the current branch. When a new commit is added, Git not only creates the commit object but also moves the HEAD pointer to it.

Thus, the name HEAD can be used in Git console commands instead of the hash of the latest commit or the current branch name. Using the special symbols ~ and ^, you can reference commits located before the latest one. For example, HEAD~2 refers to the commit two steps before the most recent commit. We won't delve into these details right now.

For further discussion, we should also mention the two possible states a repository can be in. The normal state, called 'attached HEAD', means that new commits will appear ahead of the latest commit in the current branch. In this state, all edits are added to the branch sequentially and without conflicts.

The other state, known as 'detached HEAD', occurs when the HEAD pointer refers to a commit that is not the latest in any branch. This can happen, for example, when:

- switching the repository to a specific past commit (e.g., using 'git checkout <commit-hash>'),
- switching by tag name (e.g., 'git checkout tags/<tag-name>'),
- switching to a branch that still exists in the remote repository but has been removed from the local repository (e.g., 'git checkout origin/<branch-name>').

This state should be avoided whenever possible, as any changes in this state not associated with any branch may be lost when switching to another branch. However, if you're not going to make changes in this state, it's ok to have it.

### No Tags So Far

Let's now return to our attempt to switch our local repository to a specific commit that once was the latest in the deleted branch _article-17698-forge2._

Switching a repository to a specific past commit isn't something developers typically do in everyday Git workflows. Under normal circumstances, you'll rarely need to perform such an operation. However, if you do choose to do it, the repository will enter what's known as the "detached HEAD" state. In this case, that commit belongs to the _develop_ branch, which already has newer commits following it, so it's no longer the latest one in the branch.

Still, if we use Git's command-line interface to perform this switch, the operation will complete successfully. Though Git will clearly warn us about being in a "detached HEAD: state:

![](https://c.mql5.com/2/172/1372782474508.png)

Attentive readers may notice that in the last screenshot we switched to a commit with the hash _58beccf233_, but Git reports that the HEAD pointer is now at _58beccf_. Where did the last three digits go? Nothing's wrong. They haven't disappeared. Git simply allows the use of shortened commit hashes as long as they remain unique within the repository. Depending on the interface, you might see hashes shortened to anywhere between 4 and 10 characters.

If you ever need to see the full commit hash, you can do so by running the 'git log' command. Each full commit hash contains 40 digits.

![](https://c.mql5.com/2/172/3607780728145.png)

Because each hash is generated randomly and uniquely, even the first few digits are almost guaranteed to be distinct within a repository. That's why providing only a short prefix of the hash is usually enough for Git to recognize exactly which commit you're referring to in your commands.

### Using UTF-8 Encoding

Here's another interesting aspect. In earlier versions, MetaEditor used the UTF-16LE encoding to save source code files. However, for some reason, files saved in this encoding were treated by Git as binary rather than text files As a result, it was impossible to view the exact lines of code that had been modified in a commit (even though this worked fine in Visual Studio Code). The only information displayed was the file sizes before and after the changes within the commit.

Here' what it looked like in the MQL5 Algo Forge web interface:

![](https://c.mql5.com/2/171/6054836407030.png)

Now, new files created in MetaEditor are saved using UTF-8 encoding, and even the use of national alphabet characters no longer triggers an automatic switch to UTF-16LE. Therefore, it makes sense to convert older files, carried over into the new repository from earlier projects, into UTF-8. After performing such a conversion, starting from the next commit, you'll be able to see exactly which lines and characters were changed. For example, in the MQL5 Algo Forge web interface, it might look like this:

![](https://c.mql5.com/2/171/3117899481444.png)

But that was a short digression. Let's return to the discussion of how to publish a new version of code in the repository.

### Back to the Main Task

So, among the branches in our repository, let's pay attention to these two: _article-17608-close-manager_ and _article-17607_. The changes made in these branches have not yet been merged into _develop_, since the tasks associated with them are still in progress. These branches will continue to develop, so it's too early to merge them into _develop_. We'll continue work on one of them ( _article-17607_), bring it to a logical point of completion, and then merge it with _develop_. The resulting state of the code will be tagged with a version number.

To do this, we first need to prepare the selected branch for further edits, because while it existed, other branches also introduced changes. Those changes have already been merged into _develop_. Therefore, we must ensure that these updates from _develop_ are also incorporated into our chosen branch.

There are several ways to merge changes from _develop_ into _article-17607_. For example, we could create a Pull Request via the web interface and repeat the merging process described in the previous part. However, that approach is best used when you want to merge new, untested code into a branch containing stable, tested code. In our case, the situation is the opposite: we want to bring stable, verified updates from develop into a branch that still contains new, unchecked code. Therefore, it's perfectly fine to perform the merge using Git console commands. We'll use the console and monitor the process in Visual Studio Code.

First, let's check the current state of the repository. In the version control panel, we can see the commit history with branch names. The current branch is _article-19436-forge3_, where the latest changes were made. On the right side of the terminal, the output of the 'git status' command is shown:

![](https://c.mql5.com/2/170/4223123915642.png)

The command confirms that our repository is currently on the _article-19436-forge3_ branch and that its state is synchronized with the corresponding branch in the remote repository.

Next, we switch to the _article-17607_ branch using the command 'git checkout article-17607':

![](https://c.mql5.com/2/170/5406945010391.png)

Then merge it with _develop_ using 'git merge develop':

![](https://c.mql5.com/2/170/6050128217997.png)

Since the external changes affected parts of the code we didn't modify while working in _article-17607_, no conflicts arose during the merge. As a result, Git created a new merge commit.

Now we run 'git push' to send the updated information to the remote repository:

![](https://c.mql5.com/2/170/1688555844181.png)

If we check the MQL5 Algo Forge repository, we'll see that our merge steps have successfully been reflected in the remote repository:

![](https://c.mql5.com/2/170/6175618866452.png)

The last commit shown in the screenshot is the merge commit between _develop_ and _article-17607_.

Also note the free end of the _article-19436-forge3_ branch, which is not yet connected to any other branch. The changes from this branch haven't been merged into _develop_ yet, as the work there is still ongoing. We'll leave it as is for now. When the time comes, we'll return to it.

This completes the preparation for continuing development in _article-17607_, and we can now proceed with the coding work itself. The solution for the task associated with this branch is described in another [article](https://www.mql5.com/en/articles/17607). So, I won't repeat it here. Instead, let's move on to describing how to finalize and record the achieved state of the code after completing the task.

### Performing the Merge

Before publishing a particular state of the code, we first need to merge it into the main branch. Our primary branch is _main_. All updates from the _develop_ branch will eventually flow into main. Changes from individual task branches are merged into develop. For now, we're not ready to merge new code into _main_, so we'll limit ourselves to merging updates into _develop_. For demonstrating this mechanism, the specific choice of which branch serves as the main one isn't particularly important.

Let's look at the state of the SimpleCandles repository after finishing work on the selected task:

![](https://c.mql5.com/2/172/5672431479552.png)

As shown, the latest commit was made in the article-17607 branch. Using the MQL5 Algo Forge web interface, we create a Pull Request to merge this branch into develop, as described earlier.

![](https://c.mql5.com/2/172/2132438864223.png)

Let's verify that everything went as expected. We open the commit history page again with the branch tree view:

![](https://c.mql5.com/2/172/4519275737371.png)

We can see that the commit with hash _432d8a0fd7_ is no longer marked as the latest in _article-17607_. Instead, a new commit with hash _001d35b4a7_ appears as the latest in _develop_. Since this commit records the merging of two branches, we'll refer to it as the merge commit.

Next, open the [merge commit](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles/commit/001d35b4a768735e183b68e0c90bfc850b819665 "https://forge.mql5.io/antekov/SimpleCandles/commit/001d35b4a768735e183b68e0c90bfc850b819665") page and create a new tag. Earlier in the article, we showed where to do this; and now it's time to actually do it:

![](https://c.mql5.com/2/172/826710367414.png)

In the pop-up window, enter the tag name "v0.1", since this is still far from the final version. We don't yet know how many more additions will be made to the project, but hopefully quite a few. Therefore, such a small version number serves as a reminder to ourselves that there's still plenty of work ahead. Incidentally, it doesn't look like the web interface currently supports creating annotated tags.

The tag has now been successfully created, and you can see the result on the following page:

![](https://c.mql5.com/2/172/2816554921811.png)

or on the repository's dedicated [tags](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles/tags "https://forge.mql5.io/antekov/SimpleCandles/tags") page.

![](https://c.mql5.com/2/172/2005698736821.png)

If we update our local repository using 'git pull', the newly created tag will appear there as well. However, since MetaEditor currently doesn't display repository tags, let's check how they look in Visual Studio Code. If you hover the mouse over the desired commit in the commit tree, a color label with the related tag name appears in the tooltip:

![](https://c.mql5.com/2/172/4124947236042.png)

Now that the tag is created, we can either stop here and use its name in a 'git checkout' command to switch to that exact code state or go further and create a release based on it.

### Creating a Release

A release is a mechanism for marking and distributing specific versions of software, regardless of the programming language used. Commits and branches represent the development workflow, while releases are its official outcomes, i.e. the versions we want to publish. The main purposes of using releases are as follows:

- _Versioning_. We mark particular states of the code in the repository code as stable, meaning they are free of critical errors (at least apparent ones) and have verified functionality. Other users can use on these specific versions.

- _Distributing binaries_. Releases can include compiled or packaged files (such as .ex5, .dll, or .zip), so that users don't have to compile the project themselves.

- _User communication_. A release should include a description, typically listing changes, new features, fixed bugs, and other relevant information about that version. The main goal of this description is to help users decide whether they should update to it.

It's worth mentioning that CI/CD systems, including MQL5 Algo Forge, can automatically create releases when a branch is merged into the _main_ branch, perform automatic builds, and publish binary files. However, such automation requires prior setup and the configuration of event-handling scripts. These are more advanced (though not essential) capabilities, so we'll leave them aside for now.

A release can be created based on an existing tag, or a new tag can be generated during the release creation process. Since we already have a tag, we'll create a new release using it. To do this, go to the repository tags page and click 'New release' next to the desired tag:

![](https://c.mql5.com/2/173/3678018893836.png)

In the release creation form, you can specify several basic properties:

- Release name, target branch, and tag (either an existing one or a new one to be created),
- Release notes, i.e. a summary of what's new, what's been fixed, and what known issues have been addressed,
- Attached files, for example, compiled programs, documentation, or links to external resources.

Here's what it looks like in the MQL5 Algo Forge web interface:

![](https://c.mql5.com/2/172/6175426468136.png)

You can save a release as a draft and update its details later, or publish it right away. Even if you publish it now, you can make edits later: for instance, you can still adjust the release description afterward. Once published, the release will appear on the repository's [Releases](https://www.mql5.com/go?link=https://forge.mql5.io/antekov/SimpleCandles/releases "https://forge.mql5.io/antekov/SimpleCandles/releases") page, visible to other users:

![](https://c.mql5.com/2/172/2272779380697.png)

That's it! The new version is now live and ready for use. A little later, we updated the release name (which doesn't have to match the tag name) and added a link to the above-mentioned article describing the implemented solution.

### Conclusion

Let's pause for a moment and reflect on the progress we've made. We didn't just explore the technical aspects of version control. We completed a full transformation, moving from scattered edits to a structured, coherent workflow for managing code. The most important milestone we've reached is the final step: releasing completed work as official product versions for end users. Our current repository might not yet represent a fully mature project, but we've laid all the groundwork to reach that level.

This approach fundamentally changes how we perceive the project. What once was a loose collection of source files is now an organized system with a clear history of changes and well-defined checkpoints, allowing us to revert to any stable state at any time. This benefits everyone: both developers and users of the finished solutions.

By mastering these tools, we've elevated our work with the MQL5 Algo Forge repository to a new level, opening the door to more complex and large-scale projects in the future.

Thank you for your attention! See you next time!


Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/19623](https://www.mql5.com/ru/articles/19623)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository](https://www.mql5.com/en/articles/17646)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**[Go to discussion](https://www.mql5.com/en/forum/497266)**

![Overcoming The Limitation of Machine Learning (Part 5): A Quick Recap of Time Series Cross Validation](https://c.mql5.com/2/174/19775-overcoming-the-limitation-of-logo__1.png)[Overcoming The Limitation of Machine Learning (Part 5): A Quick Recap of Time Series Cross Validation](https://www.mql5.com/en/articles/19775)

In this series of articles, we look at the challenges faced by algorithmic traders when deploying machine-learning-powered trading strategies. Some challenges within our community remain unseen because they demand deeper technical understanding. Today’s discussion acts as a springboard toward examining the blind spots of cross-validation in machine learning. Although often treated as routine, this step can easily produce misleading or suboptimal results if handled carelessly. This article briefly revisits the essentials of time series cross-validation to prepare us for more in-depth insight into its hidden blind spots.

![Introduction to MQL5 (Part 22): Building an Expert Advisor for the 5-0 Harmonic Pattern](https://c.mql5.com/2/174/19856-introduction-to-mql5-part-22-logo__1.png)[Introduction to MQL5 (Part 22): Building an Expert Advisor for the 5-0 Harmonic Pattern](https://www.mql5.com/en/articles/19856)

This article explains how to detect and trade the 5-0 harmonic pattern in MQL5, validate it using Fibonacci levels, and display it on the chart.

![Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://c.mql5.com/2/174/18361-bivariate-copulae-in-mql5-part-logo.png)[Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://www.mql5.com/en/articles/18361)

This is the first part of an article series presenting the implementation of bivariate copulae in MQL5. This article presents code implementing Gaussian and Student's t-copulae. It also delves into the fundamentals of statistical copulae and related topics. The code is based on the Arbitragelab Python package by Hudson and Thames.

![From Novice to Expert: Market Periods Synchronizer](https://c.mql5.com/2/174/19841-from-novice-to-expert-market-logo.png)[From Novice to Expert: Market Periods Synchronizer](https://www.mql5.com/en/articles/19841)

In this discussion, we introduce a Higher-to-Lower Timeframe Synchronizer tool designed to solve the problem of analyzing market patterns that span across higher timeframe periods. The built-in period markers in MetaTrader 5 are often limited, rigid, and not easily customizable for non-standard timeframes. Our solution leverages the MQL5 language to develop an indicator that provides a dynamic and visual way to align higher timeframe structures within lower timeframe charts. This tool can be highly valuable for detailed market analysis. To learn more about its features and implementation, I invite you to join the discussion.

[![](https://www.mql5.com/ff/si/w766tj9vyj3g607n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Fmarket%2Fmt5%2Fexpert%3FHasRent%3Don%26utm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Drent.expert%26utm_content%3Drent.expert%26utm_campaign%3D0622.MQL5.com.Internal&a=sorsafcerhkgwrjzwwrpvelbicxjwzon&s=ae91b1eae8acb61167455495742e6cc8eb55ccedb33fd953f8256b68cbe9c3b4&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=iuwsktuvhdfaxpjrbiremhswowkrpnaf&ssn=1769092820951386531&ssn_dr=1&ssn_sr=0&fv_date=1769092820&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F19623&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Moving%20to%20MQL5%20Algo%20Forge%20(Part%204)%3A%20Working%20with%20Versions%20and%20Releases%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909282100163159&fz_uniq=5049293424747718868&sv=2552)

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