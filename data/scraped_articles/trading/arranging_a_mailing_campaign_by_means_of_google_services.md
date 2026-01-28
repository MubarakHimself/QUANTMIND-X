---
title: Arranging a mailing campaign by means of Google services
url: https://www.mql5.com/en/articles/6975
categories: Trading
relevance_score: 0
scraped_at: 2026-01-24T13:34:48.933031
---

[![](https://www.mql5.com/ff/sh/jup0jccfs9655z9z2/01.png)Learn to create your own robotsRead our book "MQL5 Programming for Traders"Begin](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/book%3Futm_source=www.mql5.com%26utm_medium=display%26utm_term=read.algobook%26utm_content=visit.page%26utm_campaign=algobook.promo.04.2024&a=rsxjstxkzbrlgjjrxaglpezpvrjflnvw&s=7224440013c3dbc50ba9cc078cd015fabca36df446b8e75028d6b30234663872&uid=&ref=https://www.mql5.com/en/articles/6975&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082967561781383751)

MetaTrader 5 / Trading


### Introduction

A trader may want to arrange a mailing campaign to maintain business relationships with other traders, subscribers, clients or friends.
Besides, there may be a necessity to send screenshots, logs or reports. These may not be the most frequently arising tasks but having such a
feature is clearly an advantage. It would be definitely difficult or even outright impossible to use convenient MQL tools here. At the end of
the article, we will get back to the issue of using exclusively MQL tools to solve this task. Until then, we will use the combination of MQL and
C#. This will allow us to write the necessary code relatively easily and connect it to the terminal. Besides, it will also set a very
interesting challenge related to this connection.

The article is intended for beginner and mid-level developers who want to deepen their knowledge of writing libraries and integrating them
with the terminal, as well as become more familiar with Google services.

### Setting a task

Now let's define more precisely what we are going to do. There is an updatable list of contacts allowing users to send emails with attachments
once or repeatedly to any contact from the list. Things to consider:

- Certain contacts from the list may not have an address or it may be incorrect. Also, there may be multiple addresses.

- The list can be changed — contacts can be added or deleted.
- Contacts can be duplicated.
- They may also be excluded from mailing campaigns while remaining in the list. In other words, the contact's activity should be
adjustable.
- Moreover, the list will most certainly contain contacts unrelated to the task in question.


Implementing the list management is the most evident task. What options do we have here?

1. An HDD database or a CSV file is inconvenient and not reliable enough. It is not always available, and an additional software may be
    required to manage such a storage.
2. A database of a special website featuring a Joomla-type CMS. This is a good working solution. Data is protected and accessible from
    anywhere. Besides, emails can be easily sent from the website. However, there is also a significant drawback. A special add-on is
    required to interact with such a website. Such an add-on may be quite large and riddled with security holes. In other words, a reliable
    infrastructure is a must here.



3. Using the ready-made Google services. There you can securely store and manage contacts, as well as access them from different devices. In
    particular, you can form various lists (groups) and send emails. This is all that we need for comfortable work. So let's stick to this
    option.

Interacting with Google is heavily documented, for example [here](https://www.mql5.com/en/articles/3331). To
start working with Google, register an account and create a list of contacts there. The list should contain contacts we are to send emails to.
In the contacts, create a group with a certain name, for example "Forex", and add selected contacts to it. Each contact is capable of saving
multiple data to be available later. Unfortunately, if a user still needs an additional data field, it cannot be created. This should not
cause inconvenience since there are a lot of data fields available. I will show how to use them later on.

Now it is time to move on to the main tasks.

### Preparations on Google side

Suppose that we already have a Google account. Resume the project development using the Google "develop console". [Here](https://www.mql5.com/en/articles/3331)
you can find out in details how to use the console and develop a project. Of course, the article the link above leads to describes another project.
Our project needs a name. Let it be "

**WorkWithPeople**". We will need other services. At this stage, enable the following ones:

- People API
- Gmail API

The first one provides access to the list of contacts (in fact, it provides access to other things as well but we only need the list). There is
another service for accessing the list of contacts —

**Contacts API**, but at present it is not recommended for use, so we do not pay attention to it.

As the name suggests, the second service provides access to mail.

Enable the services and get the keys granting the application access to them. There is no need to write them down or remember. Download the
attached file in json format containing all the necessary data for accessing Google resources, including these keys. Save the file on your
disk, perhaps giving it a more meaningful name. In my case, it is called

**"WorkWithPeople\_gmail.json"**. This completes direct work with Google. We have created the account, the contact list and the
project, as well as got the access file.

Now let's move on to working with VS 2017.

### Project and packages

Open VS 2017 and create a standard **Class Library (.NET Framework)** project. Name it in any memorizable way (in my case,
it coincides with the Google project name "

**WorkWithPeople**", although this is not obligatory). Install additional packages using **NuGet** right away:

- Google.Apis
- Google.Apis.People.v1
- Google.Apis.PeopleService.v1
- Google.Apis.Gmail.v1
- MimeKit

During the installation, NuGet offers to install related packages. Agree to do that. In our case, the project receives the Google packages for
working with contacts and managing emails. Now we are ready to develop the code.

### Accessing a contact

Let's start with the auxiliary class. If we consider the amount of data a certain Google contact contains, it becomes obvious that the main part
of it is not needed for our task. We need a contact name and an address to send an email. In fact, we need data from yet another field, but more on
that later.

The appropriate class may look as follows:

```
namespace WorkWithPeople
{
    internal sealed class OneContact
    {
        public OneContact(string n, string e)
        {
            this.Name  = n;
            this.Email = e;
        }
        public string Name  { get; set; }
        public string Email { get; set; }
    }
}
```

There are two "string"-type properties storing the contact name and address, as well as a simple constructor featuring two parameters to
initialize them. No additional checks are implemented. They are to be performed elsewhere.

A list of simple elements is created when reading the contact list. This allows conducting a mailing campaign based on data of this newly
built list. If you want to update the list, remove all list elements and repeat the operation of reading and selecting data from the Google
account.

There is yet another auxiliary class. The contact list may contain an invalid email address or there may be no addresses at all. Before sending an
email, we need to ensure that the address is present and correct. Let's develop the new auxiliary class to achieve that:

```
namespace WorkWithPeople
{
    internal static class ValidEmail
    {
        public stati cbool IsValidEmail(this string source) => !string.IsNullOrEmpty(source) && new System.ComponentModel.DataAnnotations.EmailAddressAttribute().IsValid(source);
    }
}
```

To perform a check, we use available tools, although we can use regular expressions as well. For convenience of further use, we develop the
code as an extension method. As it is not difficult to guess, the method returns

**true** if a string containing a mailing address passes the check and **false** otherwise. Now it is time to move on to
the main part of the code.

### Access and working with services

We have already created the project, got the keys and downloaded the JSON file for authorizing the application. So let's create a new class **ContactsPeople**
and add the appropriate assemblies to the file:

```
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Net.Mail;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Google.Apis.Auth.OAuth2;
using Google.Apis.People.v1;
using Google.Apis.Services;
using Google.Apis.Util.Store;
using Google.Apis.Http;
using Google.Apis.PeopleService.v1;
using Google.Apis.PeopleService.v1.Data;
using Google.Apis.Gmail.v1;
using Google.Apis.Gmail.v1.Data;

namespace WorkWithPeople
{
    internal sealed class ContactsPeople
    {
       public static string Applicationname { get; } = "WorkWithPeople";

.....
```

Add the static property containing the Google project name. This static property is made read-only.

Add closed fields and enumeration to the class:

```
        private enum             PersonStatus
        {
            Active,
            Passive
        };
        private string           _groupsresourcename;
        private List<OneContact> _list = new List<OneContact>();
        private UserCredential   _credential;
        private PeopleService    _pservice;
        private GmailService     _gservice;
```

The enumeration is used to mark a contact as "active" (receives emails) and "passive" (does not receive emails). Other closed fields:

- \_groupsresourcename. Google resource name corresponding to the group created in the "contacts". (In our case, the selected group name was "Forex").

- \_list. The list of contacts a mailing campaign is to apply to.
- \_credential. Application "powers".
- \_pservice, \_gservice. Services for working with contacts and mail.

Let's write the code of the main working function:

```
        publicint WorkWithGoogle(string credentialfile,
                                   string user,
                                   string filedatastore,
                                   string groupname,
                                   string subject,
                                   string body,
                                   bool   isHtml,
                                   List<string> attach = null)
        {
          ...
```

Its arguments are:

- credentialfile. Name and path of accessing the JSON file containing all the data for accessing the services. It was previously downloaded from the
Google account.



- user. Google account name — XXXXX@gmail.com address.

- filedatastore. Name of an auxiliary folder — storage on a user's PC (may be arbitrary). The folder is created within AppData (%APPDATA%) and
contains the file with additional access data.



- groupname. Name of a contact group for a mailing campaign we created. In our case, it is "Forex".
- subject, body, isHtml. Email subject and text and whether it is written in html format.
- attach. List of attached files.

Returned value — number of sent emails. Start writing the function code:

```
            if (!File.Exists(credentialfile))
                throw (new FileNotFoundException("Not found: " + credentialfile));
            using (var stream = new FileStream(credentialfile, FileMode.Open, FileAccess.Read))
            {
                if (_credential == null) {
                    _credential = GoogleWebAuthorizationBroker.AuthorizeAsync(
                        GoogleClientSecrets.Load(stream).Secrets,
                        new[]
                        {
                            GmailService.Scope.GmailSend,
                            PeopleService.Scope.ContactsReadonly
                        },
                        user,
                        CancellationToken.None,
                        new FileDataStore(filedatastore)).Result;
                        CreateServicies();
                }
                else if (_credential.Token.IsExpired(Google.Apis.Util.SystemClock.Default)) {
                    bool refreshResult = _credential.RefreshTokenAsync(CancellationToken.None).Result;
                    _list.Clear();
                    if (!refreshResult) return 0;
                    CreateServicies();
                }

            }// using (var stream = new FileStream(credentialfile, FileMode.Open, FileAccess.Read))
```

Pay attention to the array of strings defining the access to which service should be requested:

- GmailService.Scope.GmailSend. This is an access to sending emails.
- PeopleService.Scope.ContactsReadonly. Access to contacts in read-only mode.

Besides, note calling **GoogleWebAuthorizationBroker.AuthorizeAsync**. Its name suggests that the call is to be performed
asynchronously.

Note that if a previously received token is overdue, the code updates it and removes all objects from the previously formed \_list.

The auxiliary CreateServicies() function creates and initializes the necessary objects:

```
        private void         CreateServicies()
        {
            _pservice = new PeopleService(new BaseClientService.Initializer()
            {
                HttpClientInitializer = _credential,
                ApplicationName = Applicationname
            });
            _gservice = new GmailService(new BaseClientService.Initializer()
            {
                HttpClientInitializer = _credential,
                ApplicationName = Applicationname
            });
        }
```

As we can see, we get access to the necessary services after executing the code segments displayed above:

\- Using the JSON data file, we first request the "powers" and save them in the \_credential field. Then we call the service constructors
passing the "power" and project name fields to them as the initializing list.

It is time to obtain the contact list of a group selected for a mailing campaign:

```
            try {
                  if (_list.Count == 0)
                    GetPeople(_pservice, null, groupname);
            }
            catch (Exception ex) {
                ex.Data.Add("call GetPeople: ", ex.Message);
                throw;
            }
#if DEBUG
            int i = 1;
            foreach (var nm in _list) {
                Console.WriteLine("{0} {1} {2}", i++, nm.Name, nm.Email);
            }
#endif
            if (_list.Count == 0) {
                Console.WriteLine("Sorry, List is empty...");
                return 0;
            }
```

The **GetPeople(...)** function (described later) is to fill in the \_list storing the contacts. This function serves as an
exception source, therefore its block is wrapped in the

**try** block. No exception types are detected in connected assemblies, therefore the **catch** block is written in
the most general form. In other words, we do not have to include all possible occurrences here in order not to lose valuable data for
debugging. Therefore, add data you consider necessary to the exception and re-activate it.

Keep in mind that **\_list** is updated only when it is empty, i.e. when it receives a new token or updates the old one.

The next block is executed only for the debugging version. The entire formed list is simply displayed in the console.

The final block is quite obvious one. If the list remains empty, the further work has no point and is stopped accompanied by the appropriate
message.

The function ends with the code block forming an outgoing email and conducting a mailing campaign:

```
            using (MailMessage mail = new MailMessage
            {
                Subject = subject,
                Body = body,
                IsBodyHtml = isHtml
            })  // MailMessage mail = new MailMessage
            {
                if (attach != null)
                {
                    foreach (var path in attach)
                        mail.Attachments.Add(new Attachment(path));
                } //  if (attach != null)

                foreach (var nm in _list)
                    mail.To.Add(new MailAddress(nm.Email, nm.Name));
                try
                {
                    SendOneEmail(_gservice, mail);
                }
                catch (Exception ex)
                {
                    ex.Data.Add("call SendOneEmail: ", ex.Message);
                    throw;
                }
            }// using (MailMessage mail = new MailMessage
```

An instance of the **MailMessage** library class is created here. This is followed by its subsequent initialization and filling
in the fields. The list of attachments is added if present. Finally, the mailing list obtained during the previous stage is formed.

Mailing is performed by the **SendOneEmail(...)** function to be described later. Just like the **GetPeople(...)**
function, it may also become an exception source. Therefore, its call is also wrapped in the **try** block, and
handling in

**catch** is made similarly.

At this point, the work of the **WorkWithGoogle(...)** main function is considered complete, and it returns the **\_list.Count**
value assuming that email messages were sent to each contact from the list.

### Filling in the contact list

After getting access, **\_list** is ready to be filled. This is done by the function:

```
        private void         GetPeople(PeopleService service, string pageToken, string groupName)
        {
           ...
```

Its arguments are:

- service. A link to a previously created access class to Google contacts.
- pageToken. There may be multiple contacts. This argument tells the developer that the list of contacts takes up several pages.
- groupName. Name of a contact group we are interested in.

First time, the function is called with pageToken = NULL. If a request to Google subsequently returns the token with the value different from
NULL, the function is called recursively.

```
            if (string.IsNullOrEmpty(_groupsresourcename))
            {
                ContactGroupsResource groupsResource = new ContactGroupsResource(service);
                ContactGroupsResource.ListRequest listRequest = groupsResource.List();
                ListContactGroupsResponse response = listRequest.Execute();
                _groupsresourcename = (from gr in response.ContactGroups
                                       where string.Equals(groupName.ToUpperInvariant(), gr.FormattedName.ToUpperInvariant())
                                       select gr.ResourceName).Single();
                if (string.IsNullOrEmpty(_groupsresourcename))
                    throw (new MissingFieldException($"Can't find GroupName: {groupName}"));
            }// if (string.IsNullOrEmpty(_groupsresourcename))
```

We need to find out a resource name by a group name. To achieve this, request the list of all resources and find out the necessary one in a simple
lambda expression. Note that there should be only one resource with the required name. If no resource is found during the work, the exception
is enabled.

```
Google.Apis.PeopleService.v1.PeopleResource.ConnectionsResource.ListRequest peopleRequest =
                new Google.Apis.PeopleService.v1.PeopleResource.ConnectionsResource.ListRequest(service, "people/me")
                {
                    PersonFields = "names,emailAddresses,memberships,biographies"
                };
            if (pageToken != null) {
                peopleRequest.PageToken = pageToken;
            }
```

Let's construct the request to Google to obtain the necessary list. To do this, specify the fields from the Google contact data we are interested
in:

- names, emailAddresses. For creating the instance of the **OneContact** class.
- memberships. To check if a contact belongs to our group.
- biographies. This field is selected for managing the contact activity, although it was designed to store a contact's biography. In order for a
contact to be recognized as active and send emails to its address, it is necessary that the field starts with one. In any other case, the
contact is considered passive and ignored even if it is located in the necessary group. It is not necessary to use this particular field
for that. In our case, it is presumably selected due to its relatively infrequent use. This is very convenient for a user managing a
mailing campaign, as it allows "enabling/disabling" certain contacts.




Finally, make a request:

```
            var request = peopleRequest.Execute();
            var list1 = from person in request.Connections
                     where person.Biographies != null
                     from mem in person.Memberships
                     where string.Equals(_groupsresourcename, mem.ContactGroupMembership.ContactGroupResourceName) &&
                           PersonActive(person.Biographies.FirstOrDefault()?.Value) == PersonStatus.Active
                     let name = person.Names.First().DisplayName
                     orderby name
                     let email = person.EmailAddresses?.FirstOrDefault(p => p.Value.IsValidEmail())?.Value
                     where !string.IsNullOrEmpty(email)
                     select new OneContact(name, email);
            _list.AddRange(list1);
            if (request.NextPageToken != null) {
                GetPeople(service, request.NextPageToken, groupName);
            }
        }//void GetPeople(PeopleService service, string pageToken, string groupName)
```

Make a request and sort the necessary data in lambda expression. It looks rather bulky but is in fact quite simple. A contact should have a
non-zero biography, be in the correct group, be an active contact and have a correct address. Let's show here the function defining the
"active/passive" status of a single contact by the "biographies" field contents:

```
        private PersonStatus PersonActive(string value)
        {
            try {
                switch (Int32.Parse(value))
                {
                    case 1:
                        return PersonStatus.Active;
                    default:
                        return PersonStatus.Passive;
                }
            }
            catch (FormatException)   { return PersonStatus.Passive; }
            catch (OverflowException) { return PersonStatus.Passive; }
        }//PersonStatus PersonActive(string value)
```

This is the only function in the project that does not seek to re-enable the exceptions trying to handle some of them on the spot instead.

We are almost done! Add the obtained list to **\_list**. If not all contacts are read, call the function recursively with a new
token value.

### Sending emails

This is performed by the following auxiliary function:

```
        private void SendOneEmail(GmailService service, MailMessage mail)
        {
            MimeKit.MimeMessage mimeMessage = MimeKit.MimeMessage.CreateFromMailMessage(mail);
            var encodedText = Base64UrlEncode(mimeMessage.ToString());
            var message = new Message { Raw = encodedText };

            var request = service.Users.Messages.Send(message, "me").Execute();
        }//  bool SendOneEmail(GmailService service, MailMessage mail)
```

Its calling is described above. The objective of this simple function is to prepare emails for sending and perform a mailing campaign.
Besides, the function features all "heavy" preparatory operations. Unfortunately, Google does not accept data in the form of the

**MailMessage** class. Therefore, prepare data in an acceptable form and code it. The **MimeKit** assembly includes
the tools that perform coding. However, I believe that it is much easier to use a simple function available to us. I will not show it here due to
its simplicity. Note the specialized

**userId** of **string** type in the **service.Users.Messages.Send** call. It is equal to the special **"me"**
value allowing Google to access your account to obtain sender data.

This concludes the analysis of the **ContactsPeople** class. The remaining functions are auxiliary, so there is no point in
dwelling on them.

### **Terminal connector**

The only remaining issue is connecting the (unfinished) assembly to the terminal. At first glance, the task is simple. Define several static
methods, compile the project and copy it to the terminal's Libraries folder. Call the static methods of the assembly from the MQL code. But
what exactly should we copy? There is an assembly in the form of a dll library. There are also about a dozen assemblies downloaded by NuGet we
actively use in our work. There is a JSON file storing data for accessing Google. Let's try to copy the entire set to the Libraries folder.
Create a primitive MQL script (there is no point in attaching its code here) and try calling a static method from the assembly. Exception!
Google.Apis.dll is not found. This is a very unpleasant surprise, which means that the CLR cannot find the desired assembly, although it is
located in the same folder as our main assembly. Why is this happening? It is not worth examining the situation here in detail. All interested
in details may find them in the famous book by Richter (in the section about searching for private assemblies).

There are already many examples of fully functional .Net applications that work with MetaTrader. Such issues occurred there as well. How were
they solved?

[Here](https://www.mql5.com/en/articles/3331) the issue was solved by creating a channel between a .Net application and an MQL
program, while

[here](https://www.mql5.com/en/articles/5563) an event-based model was used. I can suggest a similar approach involving
passing the required data from an MQL program to a .Net application using the command line.

But it is worthwhile to consider more "elegant", simple and universal solution. I mean managing the assembly download using the **AppDomain.AssemblyResolve**
event. This event occurs when the execution requirement cannot bind an assembly by name. In this case, the event
handler can load and return the assembly from another folder (having an address the handler knows). Therefore, a rather beautiful solution
suggests itself here:

1. Create a folder having a different name in the "Libraries" folder (in my case, it is "WorkWithPeople").

2. The assembly whose methods are to be imported to a file with MQL is copied to the "Libraries" folder.
3. All other project assemblies, including the JSON file containing data on accessing the Google services, are copied to the
    "WorkWithPeople" folder.
4. Let our main assembly in the Libraries folder know the address where it should look for other assemblies — the full path to the
    "WorkWithPeople" folder.

As a result, we get a workable solution without cluttering up the "Libraries" folder. It only remains to implement the decisions in the code.

### Control class

Let's create a static class

```
    public static class Run
    {

        static Run() {
            AppDomain.CurrentDomain.AssemblyResolve += ResolveAssembly;
        }// Run()
```

and add the event handler to it so that it appears in the handler chain as soon as possible. Let's define the handler itself:

```
        static Assembly ResolveAssembly(object sender, ResolveEventArgs args) {
            String dllName = new AssemblyName(args.Name).Name + ".dll";
            return Assembly.LoadFile(Path.Combine(_path, dllName) );
        }// static Assembly ResolveAssembly(object sender, ResolveEventArgs args)
```

Now whenever an assembly is detected, this handler is called. Its objective is to download and return the assembly combining the path from the **\_path**
variable (defined during the initiation) and the calculated name. Now the exception appears only if the handler is unable to find the assembly.

The initialization function looks as follows:

```
public static void Initialize(string Path, string GoogleGroup, string AdminEmail, string Storage)
        {
            if (string.IsNullOrEmpty(Path) ||
                string.IsNullOrEmpty(GoogleGroup) ||
                string.IsNullOrEmpty(AdminEmail) ||
                string.IsNullOrEmpty(Storage)) throw (new MissingFieldException("Initialize: bad parameters"));
            _group = GoogleGroup;
            _user = AdminEmail;
            _storage = Storage;
            _path = Path;
        }//  Initialize(string Path, string GoogleGroup, string AdminEmail, string Storage)
```

This function should be called the very first BEFORE an attempt to send emails is made. Its arguments are:

- Path. The path where the handler looks for assemblies and where the file with data for accessing Google is located.
- GoogleGroup. Name of a group in the contacts used for mailing.
- AdminEmail. Account name/mail address (XXX@google.com), on behalf of which a mailing is performed.
- Storage. A name of an auxiliary file where some additional data is stored.

All described arguments should not be empty strings, otherwise an exception is activated.

Create a list and a simple adding function for included files:

```
public static void AddAttachment (string attach) { _attachList.Add(attach);}
```

The function features no error-checking tools since it is to deal with screenshots and other files preliminarily created in the MetaTrader
environment. It is assumed that this is done by the control tool working in the terminal.

Let's create an object for mailing right away

```
static ContactsPeople _cContactsPeople = new ContactsPeople();
```

and execute it by calling the function:

```
public static int DoWork(string subject, string body, bool isHtml = false) {
            if (string.IsNullOrEmpty(body))
                throw (new MissingFieldException("Email body null or empty"));
            int res = 0;
            if (_attachList.Count > 0) {
                res = _cContactsPeople.WorkWithGoogle(Path.Combine(_path, "WorkWithPeople_gmail.json"),
                    _user,
                    _storage,
                    _group,
                    subject,
                    body,
                    isHtml,
                    _attachList);
                _attachList.Clear();
            } else {
                res = _cContactsPeople.WorkWithGoogle(Path.Combine(_path, "WorkWithPeople_gmail.json"),
                    _user,
                    _storage,
                    _group,
                    subject,
                    body,
                    isHtml);
            }// if (_attachList.Count > 0) ... else ...
            return res;
        }// static int DoWork(string subject, string body, bool isHtml = false)
```

The inputs are as follows:

- subject. Email subject.
- body. Email text.
- isHtml. Whether an email has html format.

There are two options of calling **\_cContactsPeople.WorkWithGoogle** depending on whether an email features attachments.
The first argument of the call is particularly interesting:

```
Path.Combine(_path, "WorkWithPeople_gmail.json")
```

This is a full path to the file containing data for accessing Google services.

The **DoWork(...)** function returns the number of sent emails.

The entire project for VS++ 2017, except for the data file for accessing Google, is located in the attached **google.zip**
archive.

### Preparations on MetaTrader side

The assembly code is ready. Let's move on to the terminal and create a simple script there. It can be written like this (part of the code at the
beginning is skipped):

```
#import "WorkWithPeople.dll"

void OnStart()
  {
   string scr = "scr.gif";
   string fl = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\";
   ChartScreenShot(0, scr, 800, 600);
   Run::Initialize("e:\\Forex\\RoboForex MT5 Demo\\MQL5\\Libraries\\WorkWithPeople\\" ,"Forex" ,"ХХХХХХ@gmail.com" ,"WorkWithPeople" );
   Run::AddAttachment(fl + scr);
   int res = Run::DoWork("some subj" ,
                         "Very big body" ,
                          false );
   Print("result: ", res);
  }
```

The code is quite obvious. Import the assembly. The first thing we do is initialize it, add the previously made screenshot and perform the
mailing. The complete code can be found in the attached file

**google\_test1.mq5**.

Another example is an indicator working on M5 and sending an email with a screenshot every time a new candle is detected:

```
#import "WorkWithPeople.dll"

input string scr="scr.gif";

string fp;

int OnInit()
  {
   fp=TerminalInfoString(TERMINAL_DATA_PATH)+"\\MQL5\\Files\\";
   Run::Initialize("e:\\Forex\\RoboForex MT5 Demo\\MQL5\\Libraries\\WorkWithPeople\\","Forex","0ndrei1960@gmail.com","WorkWithPeople");

   return(INIT_SUCCEEDED);
  }

int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   if(IsNewCandle())
     {
      ChartScreenShot(0,scr,800,600);
      Run::AddAttachment(fp+scr);
      string body="Time: "+TimeToString(TimeLocal());
      int res=Run::DoWork("some subj",body,false);
      Print(body);
     }
   return(rates_total);
  }
```

The complete code of the indicator can be found in the attached file **google\_test2.mq5**. It is very simple, so no
further comments are required for it.

### Conclusion

Let's have a look at the results. We analyzed using Google contacts for interacting with partners, as well as the method of integrating
assemblies with the terminal allowing users to avoid cluttering folders with unnecessary files. The assembly code efficiency is worth
mentioning as well. We have not focused enough attention on this issue here but it is possible to offer a set of activities to address it:

- Divide the objectives of authorizing in Google and sending emails. Engage in authorization in a separate thread by a timer.
- Try using a thread pool for sending emails.
- Use asynchronous tools for "heavy" coding of email attachments.

This does not mean you should use all these methods but their use may increase performance and allow applying the resulting assembly both with
MetaTrader and independently as a part of a separate process.

In conclusion, let's get back to the issue of using MQL tools for solving this task. Is it possible? According to the Google documentation,
the answer is yes. It is possible to achieve the same results using

**GET/POST** requests, and the appropriate examples are available. Therefore, it is possible to use the regular **WebRequest**.
The feasibility of this method is still a matter of argument. Due to a very large number of requests, it would be quite difficult to write,
debug and maintain such a code.

### Programs used in the article

| # | Name | Type | Description |
| --- | --- | --- | --- |
| 1 | google\_test1.mq5 | Script | The script making a screenshot and sending it to several addresses. |
| 2 | google\_test1.mq5 | Indicator | The sample indicator sending an email at each new candle |
| 3 | google.zip | Archive | The assembly and test console application project. |

Translated from Russian by MetaQuotes Ltd.

Original article: [https://www.mql5.com/ru/articles/6975](https://www.mql5.com/ru/articles/6975)

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/6975.zip "Download all attachments in the single ZIP archive")

[google\_test1.mq5](https://www.mql5.com/en/articles/download/6975/google_test1.mq5 "Download google_test1.mq5")(0.95 KB)

[google\_test2.mq5](https://www.mql5.com/en/articles/download/6975/google_test2.mq5 "Download google_test2.mq5")(2.76 KB)

[google.zip](https://www.mql5.com/en/articles/download/6975/google.zip "Download google.zip")(12.71 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [MVC design pattern and its application (Part 2): Diagram of interaction between the three components](https://www.mql5.com/en/articles/10249)
- [MVC design pattern and its possible application](https://www.mql5.com/en/articles/9168)
- [Using cryptography with external applications](https://www.mql5.com/en/articles/8093)
- [Building an Expert Advisor using separate modules](https://www.mql5.com/en/articles/7318)
- [Parsing HTML with curl](https://www.mql5.com/en/articles/7144)
- [A DLL for MQL5 in 10 Minutes (Part II): Creating with Visual Studio 2017](https://www.mql5.com/en/articles/5798)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/319410)**
(2)


![RowdyCoder](https://c.mql5.com/avatar/2018/7/5B57B996-3746.jpg)

**[RowdyCoder](https://www.mql5.com/en/users/focusedbit)**
\|
1 Dec 2019 at 14:26

This looks like something I'd definitely be interested in. I haven't implemented it yet but I read through it all, very sound. I'm a software engineer by trade using C#.

However, I just started learning the MQL API. It didn't dawn on my until your article that I could/should write my external libs and in C#.

Thanks for this.


![Andrei Novichkov](https://c.mql5.com/avatar/2016/11/58342001-4AC3.png)

**[Andrei Novichkov](https://www.mql5.com/en/users/andreifx60)**
\|
1 Dec 2019 at 15:05

**focusedbit:**

I'm glad you enjoyed my article.

![Extract profit down to the last pip](https://c.mql5.com/2/36/MQL5-avatar-profit_digging__1.png)[Extract profit down to the last pip](https://www.mql5.com/en/articles/7113)

The article describes an attempt to combine theory with practice in the algorithmic trading field. Most of discussions concerning the creation of Trading Systems is connected with the use of historic bars and various indicators applied thereon. This is the most well covered field and thus we will not consider it. Bars represent a very artificial entity; therefore we will work with something closer to proto-data, namely the price ticks.

![Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://c.mql5.com/2/36/fix_open_200.png)[Developing a cross-platform Expert Advisor to set StopLoss and TakeProfit based on risk settings](https://www.mql5.com/en/articles/6986)

In this article, we will create an Expert Advisor for automated entry lot calculation based on risk values. Also the Expert Advisor will be able to automatically place Take Profit with the select ratio to Stop Loss. That is, it can calculate Take Profit based on any selected ratio, such as 3 to 1, 4 to 1 or any other selected value.

![Library for easy and quick development of MetaTrader programs (part X): Compatibility with MQL4 - Events of opening a position and activating pending orders](https://c.mql5.com/2/36/MQL5-avatar-doeasy__5.png)[Library for easy and quick development of MetaTrader programs (part X): Compatibility with MQL4 - Events of opening a position and activating pending orders](https://www.mql5.com/en/articles/6767)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the ninth part, we started improving the library classes for working with MQL4. Here we will continue improving the library to ensure its full compatibility with MQL4.

![Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data](https://c.mql5.com/2/36/MQL5-avatar-doeasy__4.png)[Library for easy and quick development of MetaTrader programs (part IX): Compatibility with MQL4 - Preparing data](https://www.mql5.com/en/articles/6651)

In the previous articles, we started creating a large cross-platform library simplifying the development of programs for MetaTrader 5 and MetaTrader 4 platforms. In the eighth part, we implemented the class for tracking order and position modification events. Here, we will improve the library by making it fully compatible with MQL4.

[![](https://www.mql5.com/ff/sh/rvgkjnsrvj1mzh89z2/01.png)Best VPS for tradersTwo-click launch from MetaTrader, minimum ping to broker, 15 USD/monthLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=wpjhvzsogglsviotmypjoyhhtuxlrzhi&s=aa6c5782a1658c2f617954d478dea9989a27ae26ecabc09d0ab1204277fdf8e3&uid=&ref=https://www.mql5.com/en/articles/6975&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082967561781383751)

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