---
title: Moving to MQL5 Algo Forge (Part 1): Creating the Main Repository
url: https://www.mql5.com/en/articles/17646
categories: Integration
relevance_score: 9
scraped_at: 2026-01-22T17:40:32.337406
---

[![](https://www.mql5.com/ff/si/3fgkjn78mkxpxwmxc2.gif)](https://www.mql5.com/ff/go?link=https%3A%2F%2Ftrade.metatrader5.com%2Fterminal%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.800.80%26utm_term%3Dtrade.in.browser%26utm_content%3Dmt5.web.platform%26utm_campaign%3Den.0009.desktop.default&a=ocndbzpeklfncxysjbwfhhbalbrsdbtv&s=a4309643278437a00bdd33c5809fc6b4b4032749c00fccd07b3b84e7b8b45126&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=bfogggabsofabcpxuzmgaibarmaxasdrj&uid=cyjfbyxzpliwrxsdswhbzknyinvglwaz&ssn=1769092830234918256&ssn_dr=1&ssn_sr=0&fv_date=1769092830&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17646&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Moving%20to%20MQL5%20Algo%20Forge%20(Part%201)%3A%20Creating%20the%20Main%20Repository%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909283103058932&fz_uniq=5049295237223917790&sv=2552)

MetaTrader 5 / Examples


### Contents

- [Introduction](https://www.mql5.com/en/articles/17646/#para1)
- [Tools Currently Available](https://www.mql5.com/en/articles/17646#para2)
- [Starting the Transition](https://www.mql5.com/en/articles/17646#para3)
- [Repository Management Tool](https://www.mql5.com/en/articles/17646#para4)
- [Creating the Main Repository](https://www.mql5.com/en/articles/17646#para5)
- [Cloning the Repository](https://www.mql5.com/en/articles/17646#para6)
- [Configuring Ignored Files](https://www.mql5.com/en/articles/17646#para7)
- [Committing Changes](https://www.mql5.com/en/articles/17646#para8)
- [Creating an Archive Branch](https://www.mql5.com/en/articles/17646#para9)
- [Preparing to Create Project Branches](https://www.mql5.com/en/articles/17646#para10)
- [Conclusion](https://www.mql5.com/en/articles/17646#para11)

### Introduction

When working on a medium- to large-sized project, the need inevitably arises to track changes in the code, group them by meaning, and be able to roll back to previous versions. Generally, this is handled by version control systems such as Git or Subversion (SVN).

Most development environments provide built-in tools for working with repositories of some version control system. MetaEditor is no exception - it supports its own repository MQL Storage based on SVN. As the [documentation](https://www.metatrader5.com/en/metaeditor/help/mql5storage "https://www.metatrader5.com/en/metaeditor/help/mql5storage") states:

MQL5 Storage is a personal online repository for MQL4/MQL5 source codes. It is integrated into MetaEditor: you can save and receive data from the storage directly in the editor.

The storage features the version control system. This means that you can always find out when and how the files were changed, as well as cancel any changes and go back to the previous version.

With MQL5 Storage, your source codes always remain secure. The data is stored on a protected server, and you are the only one having access to it. If your hard drive fails, all previously saved codes can be easily restored.

The storage allows you to easily share and synchronize chart [templates and profiles](https://www.metatrader5.com/en/terminal/help/charts_advanced/templates_profiles "https://www.metatrader5.com/en/terminal/help/charts_advanced/templates_profiles"), sets of parameters for testing MQL5 programs and sets of trading instruments between different platforms.

MQL5 Storage allows remote development of MQL5 applications in teams using [shared projects](https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects "https://www.metatrader5.com/en/metaeditor/help/mql5storage/projects").

However, the Git version control system has become far more popular - and rightfully so, in our opinion. Perhaps for this reason, back in mid 2024, MetaQuotes revealed its plans to adopt Git as the built-in version control system in MetaEditor, along with the launch of MQL5 [Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/"), an in-house alternative to GitHub.

At the time this article is being written, the new repository is already available for use, but MetaEditor integration has not yet been completed. Thus, while MetaEditor remains the main development environment, developers are still limited to MQL Storage based on SVN.

In our work on various projects, we actively used the existing version control system. However, when writing the article series " [Developing a Multi-Currency Expert Advisor](https://www.mql5.com/ru/blogs/post/756958)", the lack of support for parallel code development in branches and their subsequent merging became especially noticeable. While SVN itself supports branching, MetaEditor does not provide an interface for it. External SVN clients [could be used](https://www.metatrader5.com/en/metaeditor/help/mql5storage/mql5storage_svn_client "https://www.metatrader5.com/en/metaeditor/help/mql5storage/mql5storage_svn_client"), but that would require restructuring the familiar workflow.

For this reason, the announcement of [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") was warmly received. We hoped MetaEditor would finally support branching. But seven months later, those expectations remain unfulfilled. Therefore, let us explore how we can improve the development process with the tools currently available.

For better understanding, at least basic knowledge of version control systems is required. If necessary, we recommend reviewing materials on the [MQL5 website](https://www.mql5.com/en/search#!keyword=GIT&module=mql5_module_articles) or elsewhere, such as [Getting started with Git](https://www.mql5.com/go?link=https://docs.github.com/en/get-started/learning-to-code/getting-started-with-git "https://docs.github.com/en/get-started/learning-to-code/getting-started-with-git").

### Tools Currently Available

Soon after MetaQuotes released the news about the launch of [MQL5 Algo Forge](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/"), a repository named _mql5_ was created in it, which contained all the files from our current MQL Storage. However, they all had been added in a single commit under the user [super.admin](https://www.mql5.com/go?link=https://forge.mql5.io/super.admin "https://forge.mql5.io/super.admin"), meaning no commit history was preserved. This was expected. Migrating full histories between different version control systems is nearly impossible.

Later, some interface elements in MetaEditor began showing a new repository name. For example, the title of the version history dialog was changed to "MQL5 Algo Forge":

![](https://c.mql5.com/2/128/4244733982049.png)

Yet the essence has not changed: all modifications still go into MQL Storage, not into Algo Forge. Thus, the files copied to Algo Forge seven months ago have not been updated since.

What has changed, however, is that we can now work with multiple repositories. Previously, there was only one repository. We had to create different projects in separate folders within the MQL5 data folder, which was always the repository root. This meant that when we created a new terminal instance for a new project and activated storage, we downloaded all projects from storage, even the unrelated ones. As their number grew, this became inconvenient.

Simply deleting irrelevant projects from the data folder was not an option, since every commit would then require manually excluding those deleted items from being sent to storage.

With a proper Git repository, we now have several options for structuring storage and managing different projects:

- Each project exists in its own repository.
- Each project exists in a separate branch of a single repository.
- A hybrid model: some projects are located in separate repositories, while others coexist as branches within one repository.

Each approach has pros and cons. For instance, if multiple projects share the same library, it is inconvenient to keep the library in a separate repository. Instead, it is better to maintain it in a dedicated branch, merging changes into project branches as needed. On the other hand, for self-contained projects, a separate repository is preferable as this avoids storing unrelated code from other projects.

### Starting the Transition

When making changes, it is wise to preserve what already exists. Using the automatically created _mql5_ repository may not be the best idea, since MetaQuotes may perform additional actions there under super.admin. Therefore, first of all, we will create a new repository to store all our current projects. For this, we will adopt the model of storing different projects in different branches. To implement this separation, we define the following conventions:

- The _main_ branch will either remain empty or contain only a minimal set of common code used by all projects.
- A separate _archive_ branch will store all code available at the time of migration. From here, we can copy code into individual project branches.
- Other branches will be related to individual projects and will be named accordingly.
- A project may have multiple active branches, depending on the chosen branching strategy (for example, the approach described in " [A Successful Git Branching Model](https://www.mql5.com/go?link=https://nvie.com/posts/a-successful-git-branching-model/ "https://nvie.com/posts/a-successful-git-branching-model/")").

Let us assume we have a _MetaTrader5_ folder with an installed terminal and a connected MQL Storage. That is, the terminal's _MetaTrader5/MQL5_ data folder contains, along with standard files, the code of our projects.

![](https://c.mql5.com/2/128/5275759089546.png)

Create a new folder named _MetaTrader5.forge_ and copy two executable files there:

![](https://c.mql5.com/2/128/2349477028919.png)

Launch the MetaTrader terminal from this folder in portable mode. On our system, it started in this mode upon a double click. Otherwise, you may need to explicitly specify the /portable key when running it from the command line or create a shortcut, adding this key to the application start command. There is no need to open a trading demo account or log in to MQL5.community at this stage.

An initial folder structure is created in the new folder, including an empty _MQL5_ data folder. It doesn't contain our files yet.

![](https://c.mql5.com/2/128/3939695019255.png)

Launch MetaEditor from the terminal by pressing F4.

![](https://c.mql5.com/2/128/2386667541547.png)

If you right-click a folder name, the context menu offers the option to activate MQL5 Algo Forge storage (although, in reality, this will activate the MQL Storage). Do not activate it yet, as we intend to migrate to the new repository type.

Then close both MetaTrader and MetaEditor, since we will not need them for a while and will need to perform some actions directly in the terminal folder.

### Repository Management Tool

The next step is to choose a tool for working with the future repository. Later, this role may be taken over by MetaEditor itself, but for now we have to use something else. You can use any tool for working with a Git repository, for example, [Visual Studio Code](https://www.mql5.com/go?link=https://code.visualstudio.com/download "https://code.visualstudio.com/download") (VSCode) in combination with Git. The Windows version of Git can be downloaded from [gitforwindows.org](https://www.mql5.com/go?link=https://gitforwindows.org/ "https://gitforwindows.org/").

So, install Git and VSCode (or make sure they are already installed). In VSCode, install the MQL Tools extension for working with MQL5 files:

![](https://c.mql5.com/2/128/5806821556017.png)

After installing the extension, specify the path to the MetaEditor executable in the 'Metaeditor5 Dir' parameter in settings. Since there is no need to work with MQL5 source files located outside the working folder of a terminal instance, you can follow the recommendation and provide the path relative to the currently opened folder in VSCode:

![](https://c.mql5.com/2/128/5030266235006.png)

Further down in the extension settings, we strongly recommend enabling the "Portable MT5" option.

For syntax highlighting, you will need to install the C/C++ extension for Visual Studio Code.

![](https://c.mql5.com/2/128/2420365446786.png)

Unfortunately, while MQL5 is very similar to C++, it includes certain language constructs that are not used in standard C++. As a result, the extension may occasionally display syntax errors that are irrelevant in the context of MQL5.

Now, open the _MetaTrader5.forge/MQL5_ data folder in VSCode:

![](https://c.mql5.com/2/128/3620089208193.png)

Try opening any Expert Advisor file:

![](https://c.mql5.com/2/128/5276053139695.png)

Syntax highlighting works correctly, and the top-right corner of the editor window now displays additional buttons for MQL5 syntax checking and compilation via MetaEditor. However, all #include directives generate error messages. This happens because MQL5 is not C++, and the location of standard library include files differs. To fix this, follow the suggested resolution: in the settings for the C/C++ extension for VSCode, add the path to _MetaTrader5.forge/MQL5/Include_.

![](https://c.mql5.com/2/128/2511646435813.png)

Once this is done, the error messages disappear, and the Expert Advisor compiles successfully:

![](https://c.mql5.com/2/128/1169291573467.png)

At this point, we will temporarily set VSCode aside, as the main player now takes the stage: MQL5 Algo Forge.

### Creating the Main Repository

Go to [https://forge.mql5.io/](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") and either register a new account or log in using your existing MQL5.community credentials:

![](https://c.mql5.com/2/128/3368656135366.png)

From the top-right menu, select "New repository".

![](https://c.mql5.com/2/128/4978558654991.png)

Choose a name for the repository (for example, _mql5-main_). When cloning the repository locally, you can specify any name for the root folder, so the repository name itself is not critical. During initialization, add both _.gitignore_ and _README.md_ files.

![](https://c.mql5.com/2/128/4169449485726.png)

The repository is now created. The first commit is made automatically:

![](https://c.mql5.com/2/128/286894430371.png)

For the next steps, copy the repository URL. In our case, it is: https://forge.mql5.io/antekov/mql5-main.git. We can now return from the browser to VSCode, MetaEditor, and the MetaTrader 5 terminal.

### Cloning the Repository

To clone the repository locally, we need an empty _MQL5_ folder in the terminal directory. Currently, it is already filled with files, so we must proceed as follows:

- Close VSCode, MetaEditor, and MetaTrader 5.
- Rename the existing _MQL5_ folder (for example, to _MQL6_).

Now there is no _MQL5_ folder in the _MetaTrader5.forge_ directory:

![](https://c.mql5.com/2/128/746091394490.png)

Open this folder in VSCode and launch the VSCode terminal by pressing \[Ctrl + \`\].

![](https://c.mql5.com/2/128/2799470886542.png)

Copy the repository URL and execute the cloning command, specifying the local repository's root folder name after the URL (it should match MQL5):

```
git clone https://forge.mql5.io/antekov/mql5-main.git MQL5
```

If the repository is private and this is your first time cloning it, you will be prompted to enter your credentials. As a result, a new _MQL5_ subfolder will appear in the terminal directory, containing _.gitignore_ and _README.md_.

![](https://c.mql5.com/2/128/1940227299513.png)

Now move all files and folders from _MetaTrader5.forge/MQL6_ into _MetaTrader5.forge/MQL5_, then delete the old _MetaTrader5.forge/MQL6_ folder.

![](https://c.mql5.com/2/128/3786369253834.png)

Open the _MetaTrader5.forge/MQL5_ folder in VSCode. On the left-hand Source Control panel, you will see that the repository folder has a large number of new files (in our case, 581):

![](https://c.mql5.com/2/128/3152996616160.png)

But most of these files should not be located in our repository since they belong to the standard MetaTrader 5 installation. In new versions, the contents and structure of these standard libraries and example projects may change. We cannot modify them without risking conflicts during the next update of the MetaTrader terminal or when switching to a new working folder. Therefore, there is no reason to include them in our repository.

### Configuring Ignored Files

This is precisely where the _.gitignore_ file becomes useful. Here we can specify which files and directories Git should ignore. This way, instead of hundreds of irrelevant changes, we will only see modifications in our own files. Since we have not yet added any custom files to the repository, all currently listed files should be ignored.

So, open _.gitignore_ and replace its default content with something like this:

```
# ---> MQL5
# VSCode Preferences
.vscode/*

# Executables
*.ex5

# MQL5 Standard Files
/Experts/Advisors/
/Experts/Examples/
/Experts/Free Robots/
/Experts/Market/
/Files/
/Images/
/Include/Arrays/
/Include/Canvas/
/Include/ChartObjects/
/Include/Charts/
/Include/Controls/
/Include/Expert/
/Include/Files/
/Include/Generic/
/Include/Graphics/
/Include/Indicators/
/Include/Math/
/Include/OpenCL/
/Include/Strings/
/Include/Tools/
/Include/Trade/
/Include/WinAPI/
/Include/MovingAverages.mqh
/Include/Object.mqh
/Include/StdLibErr.mqh
/Include/VirtualKeys.mqh
/Indicators/Examples/
/Indicators/Free Indicators/
/Libraries/
/Logs/
/Profiles/
/Scripts/Examples/
/Scripts/UnitTests/
/Services/
/Shared Projects/
/experts.dat
/mql5.*
```

This way you instruct the version control system to exclude VSCode settings files, compiled Expert Advisor and indicator files (the repository usually only stores source code), and standard MetaTrader 5 files located in the listed folders. Only your own source code will be tracked.

### Committing Changes

After saving _.gitignore_, VSCode shows just a single modified file - the _.gitignore_ itself. All other files in the _MQL5_ folder, which is now the root folder of our _mql5-main_ repository, although physically present, are now ignored:

![](https://c.mql5.com/2/129/151321173178.png)

Perform a commit in the local repository by adding a message such as: "Add standard files to .gitignore", and click "Commit".

![](https://c.mql5.com/2/129/5077669354578.png)

At this point, the change exists only in the local repository. To push it to the remote repository, run the "Push" command. This can be done in different ways, for example, by clicking "Sync Changes" in VSCode, selecting "Push" from the overflow menu (...) next to CHANGES, or running the 'git push' command manually.

However, before pushing the last commit upstream, check the commit history in the GRAPH. You should see two commits: "Initial commit" and "Add standard files to .gitignore". The branch name is shown in color on the right, next to commits. The first commit is labeled as _origin/main_, and the second just _main_. Actually it's the same _main_ branch. Here, _origin_ is an alias of the remote repository, so the _origin/_ prefix means that this commit is the last one in the _main_ branch in the upstream repository. The second commit appears without this prefix, so it exists and is the last one only in the local repository.

Press "Sync Changes":

![](https://c.mql5.com/2/129/1898693225719.png)

The changes now have been successfully pushed to the remote repository, and the purple "origin" label has moved to the second commit. You can confirm this by viewing the repository in the web interface:

![](https://c.mql5.com/2/129/6472628422103.png)

At this stage, we have prepared a nearly empty repository for our work. Its only branch named _main_ contains two files that we have added. Any other files present in the data folder of this terminal instance are ignored by the version control system.

### Creating an Archive Branch

As mentioned earlier, the first step is to place all our existing files into one branch so they fall under version control and can be pushed to the remote repository. This ensures we can later retrieve them on any other computer if needed.

As any other operations in the version control system, branches can be created in different ways. From VSCode, open the repository menu (...). Then select "Checkout to…"

![](https://c.mql5.com/2/129/4962495943233.png)

Next, in the list of actions, select "Create new branch…".

![](https://c.mql5.com/2/129/51798523726.png)

In the web interface, from the remote repository go to the branches tab and click a new branch creating button (highlighted with a green rectangle below):

![](https://c.mql5.com/2/129/4756943274246.png)

There is one difference between these two methods. The first one creates a new branch on the local computer, and, until you push changes, the remote repository knows nothing about this branch. The second method creates a new branch in the remote repository, so the local repository does not know about the existence of this branch until you retrieve changes with the pull command. Either way, you just need to sync by running the pull or push commands).

The third way is to use the command line. To create a new branch named "archive" in your local repository, you can run, for example, the following command from the repository folder:

```
git checkout -b archive
```

If you do this in the VSCode integrated terminal, you will see the following:

![](https://c.mql5.com/2/129/1639869804942.png)

This terminal informs you that the repository has been switched to a newly created "archive" branch. In the commit list, the branch name _main_ changed to _archive_. The changelog will prompt you to push a new branch to the remote repository. However, we will first add our files to this branch.

We had an initial folder with the MetaTrader terminal, which contained the _MQL5_ folder with all our files. The repository was created inside another terminal folder. Now, copy all files from the old original _MetaTrader5/MQL5_ folder to the new repository's folder:

![](https://c.mql5.com/2/129/2690373388558.png)

The version control system immediately detected new files in the repository folder, and certain actions became available for them. You need to add all new files to the index (to enable version control). This can be done either through the VSCode interface by selecting "+" (Stage All Changes) in the change list header, or by running the command

git add .

![](https://c.mql5.com/2/129/2462960423014.png)

Now each file in the list of changes is labeled with A, meaning it's been indexed, instead of U. Next, commit and push changes to the remote repository:

![](https://c.mql5.com/2/129/6174214249662.png)

As we can see, the last commit in the _main_ branch is now the second commit, while the third commit has been made in the _archive_ branch. Let's verify that the new branch has been pushed to the remote repository:

![](https://c.mql5.com/2/129/140457061414.png)

The latest commit is visible in the remote repository and belongs to the new _archive_ branch. Click on it to see the specific changes included in this commit.

![](https://c.mql5.com/2/129/249696992441.png)

At this point, all files have been successfully added to the archive branch. Now, let's switch back to the _main_ branch in our local repository.

### Preparing to Create Project Branches

To switch to the main branch, execute the following command in the console

git checkout main

The local repository should return to the state it was in before we copied all the files. However, inspecting the contents of the _MQL5_ folder after switching, you may notice that many of our Expert Advisor folders remain.

![](https://c.mql5.com/2/129/3186184835157.png)

These folders still contain compiled .ex5 files. This happens because, having excluded them from version control, Git does not remove these files when switching from the archive branch back to main. Only the source files that were staged and committed to the repository have been removed from the folder.

This is not very convenient, so we need to remove the compiled files and any empty directories that remain after deleting them. Doing this manually would be time-consuming, especially since we might need to repeat it in the future. Therefore, it's more practical to write a simple script to handle this task automatically.

During the script development, it became clear that deleting files alone is insufficient to restore the root folder to its original state after switching branches. It is also necessary to remove directories that have become empty after .ex5 files are deleted. Some folders may be intentionally empty and should not be removed. These will be added to an exclusion list. This list will include all folders previously listed in _.gitignore_ as well as the _.git_ folder containing version control metadata.

Here is an example of such a script:

```
import os

def delete_ex5_files_and_empty_dirs(path, excluded=['.git', '.vscode']):
    # Exceptions
    excluded = {os.path.join(path, dir) for dir in excluded}

    # Check all folders and files in the directory tree
    for root, dirs, files in os.walk(path, topdown=False):
        is_excluded = False
        for ex in excluded:
            if root.startswith(ex):
                is_excluded = True
                break
        if is_excluded:
            continue

        # Delete all files with extension .ex5
        for file in files:
            if file.endswith('.ex5'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f'File removed: {file_path}')

        # Delete all folders that have become empty after deleting files
        for dir in dirs:
            dir_path = os.path.join(root, dir)

            # IF the directory is empty after deleting files
            if dir_path not in excluded and not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    print(f'Empty folder removed: {dir_path}')
                except OSError:
                    pass  # If error occurred, ignore

excluded = [\
    '.git',\
    '.vscode',\
    'Experts\\Advisors',\
    'Experts\\Examples',\
    'Experts\\Free Robots',\
    'Experts\\Market'\
    'Files',\
    'Images',\
    'Include\\Arrays',\
    'Include\\Canvas',\
    'Include\\ChartObjects',\
    'Include\\Charts',\
    'Include\\Controls',\
    'Include\\Expert',\
    'Include\\Files',\
    'Include\\Generic',\
    'Include\\Graphics',\
    'Include\\Indicators',\
    'Include\\Math',\
    'Include\\OpenCL',\
    'Include\\Strings',\
    'Include\\Tools',\
    'Include\\Trade',\
    'Include\\WinAPI',\
    'Indicators\\Examples',\
    'Indicators\\Free Indicators',\
    'Libraries',\
    'Logs',\
    'Presets',\
    'Profiles',\
    'Scripts\\Examples',\
    'Scripts\\UnitTests',\
    'Services',\
    'Shared Projects',\
]

if __name__ == '__main__':
    current_dir = os.getcwd()  # Current working directory
    delete_ex5_files_and_empty_dirs(current_dir, excluded)
```

Save this script as _clean.py_ in the repository root and add it to version control in the main branch. From now on, after switching from archive back to main, simply run this script to automatically clean up compiled files and empty folders.

### Conclusion

Here we conclude our initial experiments with the new repository system. All our files have been successfully transferred into it. We have laid the groundwork for creating new project branches for new projects. Since the code is safely stored in the archive branch, we can gradually migrate projects into separate branches as needed.

An interesting next step will be to try creating a public repository for the source code of our article series " [Developing a Multi-Currency Expert Advisor](https://www.mql5.com/ru/blogs/post/756958)". It is still unclear how to structure the code across the different parts of the series, but we will address this in the near future.

Thank you for your attention! See you soon!

### Archive contents

| # | Name | Version | Description |
| --- | --- | --- | --- |
|  | **MQL5** |  | **Repository root folder (Terminal Data Folder)** |
| --- | --- | --- | --- |
| 1 | .gitignore | 1.00 | File listing folders and files to be ignored by the Git version control system |
| --- | --- | --- | --- |
| 2 | clean.py | 1.00 | Script to delete compiled files and empty directories when switching to the repository's main branch |
| --- | --- | --- | --- |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/17646](https://www.mql5.com/ru/articles/17646)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17646.zip "Download all attachments in the single ZIP archive")

[MQL5.zip](https://www.mql5.com/en/articles/download/17646/MQL5.zip "Download MQL5.zip")(1.66 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Moving to MQL5 Algo Forge (Part 4): Working with Versions and Releases](https://www.mql5.com/en/articles/19623)
- [Moving to MQL5 Algo Forge (Part 3): Using External Repositories in Your Own Projects](https://www.mql5.com/en/articles/19436)
- [Moving to MQL5 Algo Forge (Part 2): Working with Multiple Repositories](https://www.mql5.com/en/articles/17698)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (II)](https://www.mql5.com/en/articles/17328)
- [Developing a multi-currency Expert Advisor (Part 24): Adding a new strategy (I)](https://www.mql5.com/en/articles/17277)
- [Developing a multi-currency Expert Advisor (Part 23): Putting in order the conveyor of automatic project optimization stages (II)](https://www.mql5.com/en/articles/16913)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/495124)**
(7)


![Maxim Kuznetsov](https://c.mql5.com/avatar/2016/1/56935A91-AF51.png)

**[Maxim Kuznetsov](https://www.mql5.com/en/users/nektomk)**
\|
7 Apr 2025 at 15:52

**Yuriy Bykov projects from this repository.**
**To be honest, this scenario has really not been considered. I don't know whether banning a user on the forum now restricts access to the current MQL Storage repository, and whether this will also restrict access to the new repository. If so, this risk factor is certainly worth considering.**

It is difficult to check this - so the [risk assessment](https://www.mql5.com/en/articles/3650 "Article: Assessing Risk in a Sequence of Transactions with a Single Asset ") is theoretical ;-) but there is a risk as such

MQLStorage requires login to the community. The technical possibility of login is in the hands of admins. In theory, if you violate the rules severely (or someone will think that seriously) can get a hard ban. With a temporary ban krode as only "defeat in rights", that is simply components of the site and individual services are banned.

But there are also virtuals, servers, data centres, networks that have earned ban-po-ip . MQLStorage is most likely unavailable from there. You can get it without personal efforts and even just by dynamic ip :-)

To minimise such risks - keep full backups and an independent mirror of the repository. That's another pleasure...

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
9 Sep 2025 at 09:00

**Maxim Kuznetsov projects:-)**

Firstly, [https://forge.mql5.io/](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") has two authorisation options. You can create an account completely independent from MQL5.com

Secondly, a ban on the forum means only a ban on posting and has no effect on other services.

And thirdly, what do bans have to do with it? Get involved in the development of robots, not in the forums.

![](https://c.mql5.com/3/474/683046782747.png)

[https://c.mql5.com/3/474/683046782747.png](https://c.mql5.com/3/474/683046782747.png "https://c.mql5.com/3/474/683046782747.png")

![Stanislav Korotky](https://c.mql5.com/avatar/2010/10/4CA7CFA0-1F0C.jpg)

**[Stanislav Korotky](https://www.mql5.com/en/users/marketeer)**
\|
9 Sep 2025 at 14:09

**Rashid Umarov [#](https://www.mql5.com/ru/forum/484371#comment_57991835):**

Firstly, [https://forge.mql5.io/](https://www.mql5.com/go?link=https://forge.mql5.io/ "https://forge.mql5.io/") has two authorisation options. You can create an account completely independent from MQL5.com

But how to access ME projects if there is no dependence on mql5.com? It seems to be obligatory to log in to the community there.

![Rashid Umarov](https://c.mql5.com/avatar/2012/5/4FC60566-2EEC.jpg)

**[Rashid Umarov](https://www.mql5.com/en/users/rosh)**
\|
9 Sep 2025 at 14:13

**Stanislav Korotky [#](https://www.mql5.com/ru/forum/484371#comment_57994138):**

And then how to access projects from ME, if there is no dependence on mql5.com? It seems to be necessary to log in to the community there.

Oh, right. The account will be created in MQL5.com anyway.

![Yuriy Bykov](https://c.mql5.com/avatar/avatar_na2.png)

**[Yuriy Bykov](https://www.mql5.com/en/users/antekov)**
\|
9 Sep 2025 at 14:27

**Stanislav Korotky [#](https://www.mql5.com/ru/forum/484371#comment_57994138):**

And then how to access projects from ME, if there is no dependence on mql5.com? It seems to be necessary to log in to the community there.

You don't have to log in to the community yet. If you clone a repository from any repository, such as Algo Forge or GitHub, into a folder inside the MQL5 data folder, it will be visible just as a folder with files. This is enough for editing, launching and debugging, but all operations with the repository will have to be performed using third-party tools. I used this option for some time, while ME could not work with Algo Forge yet. But in general it is easier with mql5.com account.

![Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG](https://c.mql5.com/2/168/16340-elevate-your-trading-with-smart-logo.png)[Elevate Your Trading With Smart Money Concepts (SMC): OB, BOS, and FVG](https://www.mql5.com/en/articles/16340)

Elevate your trading with Smart Money Concepts (SMC) by combining Order Blocks (OB), Break of Structure (BOS), and Fair Value Gaps (FVG) into one powerful EA. Choose automatic strategy execution or focus on any individual SMC concept for flexible and precise trading.

![Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback](https://c.mql5.com/2/168/19442-automating-trading-strategies-logo__2.png)[Automating Trading Strategies in MQL5 (Part 30): Creating a Price Action AB-CD Harmonic Pattern with Visual Feedback](https://www.mql5.com/en/articles/19442)

In this article, we develop an AB=CD Pattern EA in MQL5 that identifies bullish and bearish AB=CD harmonic patterns using pivot points and Fibonacci ratios, executing trades with precise entry, stop loss, and take-profit levels. We enhance trader insight with visual feedback through chart objects.

![Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://c.mql5.com/2/168/19428-statistical-arbitrage-through-logo.png)[Statistical Arbitrage Through Cointegrated Stocks (Part 4): Real-time Model Updating](https://www.mql5.com/en/articles/19428)

This article describes a simple but comprehensive statistical arbitrage pipeline for trading a basket of cointegrated stocks. It includes a fully functional Python script for data download and storage; correlation, cointegration, and stationarity tests, along with a sample Metatrader 5 Service implementation for database updating, and the respective Expert Advisor. Some design choices are documented here for reference and for helping in the experiment replication.

![From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading](https://c.mql5.com/2/168/19299-from-novice-to-expert-animated-logo.png)[From Novice to Expert: Animated News Headline Using MQL5 (X)—Multiple Symbol Chart View for News Trading](https://www.mql5.com/en/articles/19299)

Today we will develop a multi-chart view system using chart objects. The goal is to enhance news trading by applying MQL5 algorithms that help reduce trader reaction time during periods of high volatility, such as major news releases. In this case, we provide traders with an integrated way to monitor multiple major symbols within a single all-in-one news trading tool. Our work is continuously advancing with the News Headline EA, which now features a growing set of functions that add real value both for traders using fully automated systems and for those who prefer manual trading assisted by algorithms. Explore more knowledge, insights, and practical ideas by clicking through and joining this discussion.

[![](https://www.mql5.com/ff/si/s2n3m9ymjh52n07n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F523%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dchoose.signals%26utm_content%3Dsubscribe.signal%26utm_campaign%3D0622.MQL5.com.Internal&a=fyznzyduwsltgnhlftytumasbfgbwlqw&s=91bc0eca8f132d3df7d14cdb1baebac753aef179403d60dc83856af55a4d6769&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=dypnenydubiobgqvadkjqakuxhkdyffm&ssn=1769092830234918256&ssn_dr=1&ssn_sr=0&fv_date=1769092830&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17646&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Moving%20to%20MQL5%20Algo%20Forge%20(Part%201)%3A%20Creating%20the%20Main%20Repository%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176909283102955542&fz_uniq=5049295237223917790&sv=2552)

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