---
title: Websockets for MetaTrader 5: Asynchronous client connections with the Windows API
url: https://www.mql5.com/en/articles/17877
categories: Integration
relevance_score: 6
scraped_at: 2026-01-23T17:18:50.751205
---

[![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/01.png)![](https://www.mql5.com/ff/sh/ub4fqgrk4rkv8gz9z2/02.png)Explore your trading for freeUpdated statistics in MetaTrader 5 will help you to thoroughly evaluate results and reduce risksLearn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/forum/454106&a=bkbqgaxtrafeuegfvjisjjwjohagrvnr&s=25c5856d7857fc6b6db7cffb15ae4ce40fd19d1ab594d8a900ad65673d9ffa0e&uid=&ref=https://www.mql5.com/en/articles/17877&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5068086814360663368)

MetaTrader 5 / Examples


### Introduction

The article, ["WebSockets for MetaTrader 5: Using the Windows API"](https://www.mql5.com/en/articles/10275 "/en/articles/10275"), illustrated the utilization of the Windows API for the implementation of a websocket client within MetaTrader 5 applications. The implementation presented there was constrained by its synchronous operational mode.

In this article, we revisit the application of the Windows API to construct a websocket client for MetaTrader 5 programs, with the objective of achieving asynchronous client functionality. A practical methodology for realizing this objective involves the creation of a custom dynamically linked library (DLL) that exports functions suitable for integration with MetaTrader 5 applications.

Accordingly, this article will discuss the development process of the DLL and subsequently present a demonstration of its application through an MetaTrader 5 program example.

### WinHTTP asynchronous mode

The prerequisites for asynchronous operation within the WinHTTP library, as delineated in its [documentation](https://www.mql5.com/go?link=https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpopen "https://learn.microsoft.com/en-us/windows/win32/api/winhttp/nf-winhttp-winhttpopen"), are twofold. Firstly, during the invocation of the WinHTTPOpen function, the session handle must be configured with either the WINHTTP\_FLAG\_ASYNC or WINHTTP\_FLAG\_SECURE\_DEFAULTS flag.

```
// Set hSession
hSession = WinHttpOpen(L"MyApp", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, WINHTTP_FLAG_ASYNC);
if (hSession == NULL)
        ErrorCode = ERROR_INVALID_HANDLE;
// Return error code
return ErrorCode;
```

After the establishment of a valid session handle, users are required to register a callback function to receive notifications pertaining to various events associated with specific WinHTTP function calls. This status callback function is informed of the progress of asynchronous operations via notification flags.

Registration of the callback function is accomplished through the WinHttpSetStatusCallback function, which also permits the specification of notification flags that the callback will manage. Users can elect to subscribe to a comprehensive set of notifications or a more limited subset. Furthermore, distinct callback functions can be designated for session, request, and websocket handles, respectively.

It is important to note that the registration of a callback function immediately following session handle creation is not mandatory; the WinHttpSetStatusCallback function can be invoked for any valid HINTERNET handle at any stage before or during websocket connection initialization.

```
if (!WinHttpSetOption(hWebSocket, WINHTTP_OPTION_CONTEXT_VALUE, (LPVOID)this, sizeof(this)))
{
        // Handle error
        ErrorCode = GetLastError();
        return ErrorCode;
}

if (WinHttpSetStatusCallback(hWebSocket, WebSocketCallback, WINHTTP_CALLBACK_FLAG_ALL_COMPLETIONS, 0) == WINHTTP_INVALID_STATUS_CALLBACK)
{
        ErrorCode = GetLastError();
        return ErrorCode;
}
```

The signature of the user-defined callback function incorporates a parameter that points to a user-defined data structure, commonly referred to as a context value. This mechanism facilitates the transfer of data from the callback function.

The context value must be specified before callback function registration through a call to the WinHttpSetOption function with the WINHTTP\_OPTION\_CONTEXT\_VALUE option flag. It is pertinent to acknowledge that empirical testing encountered difficulties in reliably retrieving a registered context value via this method.

While the possibility of implementation error cannot be entirely discounted, consistent failure necessitated the adoption of a global variable as an alternative, a detail that will be dealt with in the subsequent discussion of DLL implementation.

Finally, a crucial consideration regarding the user-defined callback function is the requirement for thread safety. However, given that the present work involves the creation of a DLL for use within the MetaTrader 5 environment, this constraint can be relaxed. This is due to the inherent single-threaded nature of MetaTrader 5 programs in general, and while DLL code executes within the thread pool of the loading process, on MetaTrader 5 programs, only a single thread is active.

```
void WebSocketCallback(HINTERNET hInternet, DWORD_PTR dwContext, DWORD dwInternetStatus, LPVOID lpvStatusInformation, DWORD dwStatusInformationLength)
```

### Implementing the DLL

The DLL is constructed within the Visual Studio environment using the C++ programming language. This process necessitates the installation of the "C++ Desktop development" workload in Visual Studio, alongside either the Windows 10 or Windows 11 Software Development Kit (SDK). The Windows SDK is a prerequisite, as it furnishes the WinHTTP library file (.lib), to which the DLL will be linked during compilation. The resultant DLL comprises at least three fundamental components.

The first is a class that encapsulates the essential WinHTTP websocket client functionality. The second is a singular callback function that operates alongside a global variable, facilitating the manipulation of a websocket connection both within the callback function's scope and externally. The third component consists of a set of simplified function wrappers that will be exposed by the DLL for utilization in MetaTrader 5 programs. The implementation commences with the code defined in the asyncwebsocketclient.h header file.

This header file begins by declaring the WebSocketClient class, where each instance represents an individual client connection.

```
// WebSocket client class
class WebSocketClient {
private:
        // Application session handle to use with this connection
        HINTERNET hSession;
        // Windows connect handle
        HINTERNET hConnect;
        // The initial HTTP request handle to start the WebSocket handshake
        HINTERNET hRequest;
        // Windows WebSocket handle
        HINTERNET hWebSocket;
        //initialization flag
        DWORD initialized;
        //sent bytes
        DWORD bytesTX;
        //last error code
        DWORD ErrorCode;
        //last completed websocket operation as indicated by callback function
        DWORD completed_websocket_operation;
        //internal queue of frames sent from a server
        std::queue<Frame>* frames;
        //client state;
        ENUM_WEBSOCKET_STATE status;
        //sets an hSession handle
        DWORD Initialize(VOID);
        // reset state of object
        /*
         reset_error: boolean flag indicating whether to rest the
         internal error buffers.
        */
        VOID  Reset(bool reset_error = true);

public:
        //constructor(s)
        WebSocketClient(VOID);
        WebSocketClient(const WebSocketClient&) = delete;
        WebSocketClient(WebSocketClient&&) = delete;
        WebSocketClient& operator=(const WebSocketClient&) = delete;
        WebSocketClient& operator=(WebSocketClient&&) = delete;
        //destructor
        ~WebSocketClient(VOID);
        //received bytes;
        DWORD bytesRX;
        // receive buffer
        std::vector<BYTE> rxBuffer;
        // received frame type;
        WINHTTP_WEB_SOCKET_BUFFER_TYPE rxBufferType;
        // Get the winhttp websocket handle
        /*
        return: returns the hWebSocket handle which is used to
        identify a websocket connection instance
        */
        HINTERNET WebSocketHandle(VOID);

        // Connect to a server
        /*
           hsession: HINTERNET session handle
           host: is the url
           port: prefered port number to use
           secure: 0 is false, non-zero is true
           return: DWORD error code, 0 indicates success
           and non-zero for failure
        */
        DWORD Connect(const WCHAR* host, const INTERNET_PORT port, const DWORD secure);

        // Send data to the WebSocket server
        /*
           bufferType: WINHTTP_WEB_SOCKET_BUFFER_TYPE enumeration of the frame type
           pBuffer: pointer to the data to be sent
           dwLength: size of pBuffer data
           return: DWORD error code, 0 indicates success
           and non-zero for failure
        */
        DWORD Send(WINHTTP_WEB_SOCKET_BUFFER_TYPE bufferType, void* pBuffer, DWORD dwLength);

        // Close the connection to the server
        /*
           status: WINHTTP_WEB_SOCKET_CLOSE_STATUS enumeration of the close notification to be sent
           reason: character string of extra data sent with the close notification
           return: DWORD error code, 0 indicates success
           and non-zero for failure
        */
        DWORD Close(WINHTTP_WEB_SOCKET_CLOSE_STATUS status, CHAR* reason = NULL);

        // Retrieve the close status sent by a server
        /*
           pusStatus: pointer to a close status code that will be filled upon return.
           pvReason: pointer to a buffer that will receive a close reason
           dwReasonLength: The length of the pvReason buffer,
           pdwReasonLengthConsumed:The number of bytes consumed. If pvReason is NULL and dwReasonLength is 0,
       pdwReasonLengthConsumed will contain the size of the buffer that needs to be allocated
       by the calling application.
           return: DWORD error code, 0 indicates success
           and non-zero for failure
        */
        DWORD QueryCloseStatus(USHORT* pusStatus, PVOID pvReason, DWORD dwReasonLength, DWORD* pdwReasonLengthConsumed);

        // read from the server
        /*
           bufferType: WINHTTP_WEB_SOCKET_BUFFER_TYPE enumeration of the frame type
           pBuffer: pointer to the data to be sent
           pLength: size of pBuffer
           bytesRead: pointer to number bytes read from the server
           pBufferType: pointer to type of frame sent from the server
           return: DWORD error code, 0 indicates success
           and non-zero for failure
        */
        DWORD Receive(PVOID pBuffer, DWORD pLength, DWORD* bytesRead, WINHTTP_WEB_SOCKET_BUFFER_TYPE* pBufferType);

        // Check client state
        /*
           return: ENUM_WEBSOCKET_STATE enumeration
        */
        ENUM_WEBSOCKET_STATE Status(VOID);

        // get frames cached in the internal queue
        /*
           pBuffer: User supplied container to which data is written to
           pLength: size of pBuffer
           pBufferType: WINHTTP_WEB_SOCKET_BUFFER_TYPE enumeration of frame type
        */
        VOID Read(BYTE* pBuffer, DWORD pLength, WINHTTP_WEB_SOCKET_BUFFER_TYPE* pBufferType);

        // get bytes received
        /*
           return: Size of most recently cached frame sent from a server
        */
        DWORD ReadAvailable(VOID);

        // get the last error
        /*
           return: returns the last error code
        */
        DWORD LastError(VOID);

        // activate callback function
        /*
           return: DWORD error code, 0 indicates success
           and non-zero for failure
        */
        DWORD EnableCallBack(VOID);
        // set error
        /*
           message: Error description to be captured
           errorcode: new user defined error code
        */
        VOID SetError(const DWORD errorcode);

        // get the last completed operation
        /*
          returns: DWORD constant of last websocket operation
        */
        DWORD LastOperation(VOID);

        //deinitialize the session handle and free up resources
        VOID Free(VOID);

        //the following methods define handlers meant to be triggered by the callback function//

        // on error
        /*
           result: pointer to WINHTTP_ASYNC_RESULT structure that
           encapsulates the specific event that triggered the error
        */
        VOID OnError(const WINHTTP_ASYNC_RESULT* result);

        // read completion handler
        /*
           read: Number of bytes of data successfully read from the server
           buffertype: type of frame read-in.
           Called when successfull read is completed
        */
        VOID OnReadComplete(const DWORD read, const WINHTTP_WEB_SOCKET_BUFFER_TYPE buffertype);

        // websocket close handler
        /*
           Handles the a successfull close request
        */
        VOID OnClose(VOID);

        // Send operation handler
        /*
           sent: the number of bytes successfully sent to the server if any
           This is a handler for an asynchronous send that interacts with the
           callback function
        */
        VOID OnSendComplete(const DWORD sent);

        //set the last completed websocket operation
        /*
          operation : constant defining the operation flagged as completed by callback function
        */
        VOID OnCallBack(const DWORD operation);
};
```

Alongside the class, the structure, Frame is defined to represent a message frame.

```
struct Frame
{
        std::vector<BYTE>frame_buffer;
        WINHTTP_WEB_SOCKET_BUFFER_TYPE frame_type;
        DWORD frame_size;
};
```

Furthermore, the enumeration ENUM\_WEBSOCKET\_STATE is declared to describe the various states of a websocket connection.

```
// client state
enum ENUM_WEBSOCKET_STATE
{
        CLOSED = 0,
        CLOSING = 1,
        CONNECTING = 2,
        CONNECTED = 3,
        SENDING = 4,
        POLLING = 5
};
```

Next, asyncwebsocketclient.h declares a global variable named clients. This variable is a container, specifically a map, designed to store active websocket connections. The global scope of this map container ensures its accessibility to any callback function defined within the library.

```
// container for  websocket objects accessible to callback function
extern std::map<HINTERNET, std::shared_ptr<WebSocketClient>>clients;
```

The asyncwebsocketclient.h file concludes by defining a set of functions qualified by WEBSOCK\_API. This qualifier serves to mark these functions for export by the DLL. These functions constitute the aforementioned function wrappers and represent the interface through which developers will interact with the DLL within their MetaTrader 5 applications.

```
// deinitializes a session handle
/*
   websocket_handle: HINTERNET the websocket handle to close
*/
VOID WEBSOCK_API client_reset(HINTERNET websocket_handle);

//creates a client connection to a server
/*
           url: the url of the server
           port: port
           secure: use secure connection(non-zero) or not (zero)
           websocket_handle: in-out,HINTERNET non NULL session handle
           return: returns DWORD, zero if successful or non-zero on failure

*/

DWORD  WEBSOCK_API client_connect(const WCHAR* url, INTERNET_PORT port, DWORD secure, HINTERNET* websocket_handle);

//destroys a client connection to a server
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
*/

void WEBSOCK_API client_disconnect(HINTERNET websocket_handle);

//writes data to a server (non blocking)
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
           bufferType: WINHTTP_WEB_SOCKET_BUFFER_TYPE enumeration of the frame type
           message: pointer to the data to be sent
           length: size of pBuffer data
           return: DWORD error code, 0 indicates success
           and non-zero for failure
*/

DWORD WEBSOCK_API client_send(HINTERNET websocket_handle, WINHTTP_WEB_SOCKET_BUFFER_TYPE buffertype, BYTE* message, DWORD length);

//reads data sent from a server cached internally
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
           out: User supplied container to which data is written to
           out_size: size of out buffer
           buffertype: WINHTTP_WEB_SOCKET_BUFFER_TYPE enumeration of frame type
           return: DWORD error code, 0 indicates success
           and non-zero for failure
*/

DWORD WEBSOCK_API client_read(HINTERNET websocket_handle, BYTE* out, DWORD out_size, WINHTTP_WEB_SOCKET_BUFFER_TYPE* buffertype);

//listens for a response from a server (non blocking)
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
           return: DWORD error code, 0 indicates success
           and non-zero for failure
*/

DWORD WEBSOCK_API client_poll(HINTERNET websocket_handle);

//gets the last generated error
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
           lasterror: container that will hold the error description
           length: the size of lasterror container
           lasterrornum: reference to which the last error code is written
           return: DWORD error code, 0 indicates success
           and non-zero for failure
*/

DWORD WEBSOCK_API  client_lasterror(HINTERNET websocket_handle);

//checks whether there is any data cached internally
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
           return: returns the size of the last received frame in bytes;
*/

DWORD WEBSOCK_API client_readable(HINTERNET websocket_handle);

//return the state of a websocket connection
/*
           websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
           return: ENUM_WEBSOCKET_STATE enumeration of the state of a client
*/

ENUM_WEBSOCKET_STATE WEBSOCK_API client_status(HINTERNET websocket_handle);

//return the last websocket operation
/*
     websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
     return : DWORD constant corresponding to a unique callback status value as defined in API
*/
DWORD WEBSOCK_API client_lastcallback_notification(HINTERNET websocket_handle);

//return the websocket handle
/*
     websocket_handle: a valid (non NULL) websocket handle created by calling client_connect()
     return : HINTERNET returns the websocket handle for a client connection
*/
HINTERNET WEBSOCK_API client_websocket_handle(HINTERNET websocket_handle);
```

Examination of the WebSocketCallback() function definition reveals the utilization of the global clients variable for managing notifications. The current implementation is configured to handle what is referred to as completion notifications, in the WinHTTP documentation. These notifications are triggered upon the successful completion of any asynchronous operation. For instance, WINHTTP\_CALLBACK\_STATUS\_WRITE\_COMPLETE signals the completion of a send operation, while WINHTTP\_CALLBACK\_STATUS\_READ\_COMPLETE indicates the completion of a read operation.

```
void WebSocketCallback(HINTERNET hInternet, DWORD_PTR dwContext, DWORD dwInternetStatus, LPVOID lpvStatusInformation, DWORD dwStatusInformationLength)
{
        if (WinHttpWebSocketClient::clients.find(hInternet)!= WinHttpWebSocketClient::clients.end())
        {
                WinHttpWebSocketClient::clients[hInternet]->OnCallBack(dwInternetStatus);
                switch (dwInternetStatus)
                {
                case WINHTTP_CALLBACK_STATUS_CLOSE_COMPLETE:
                        WinHttpWebSocketClient::clients[hInternet]->OnClose();
                        break;

                case WINHTTP_CALLBACK_STATUS_WRITE_COMPLETE:
                        WinHttpWebSocketClient::clients[hInternet]->OnSendComplete(((WINHTTP_WEB_SOCKET_STATUS*)lpvStatusInformation)->dwBytesTransferred);
                        break;

                case WINHTTP_CALLBACK_STATUS_READ_COMPLETE:
                        WinHttpWebSocketClient::clients[hInternet]->OnReadComplete(((WINHTTP_WEB_SOCKET_STATUS*)lpvStatusInformation)->dwBytesTransferred, ((WINHTTP_WEB_SOCKET_STATUS*)lpvStatusInformation)->eBufferType);
                        break;

                case WINHTTP_CALLBACK_STATUS_REQUEST_ERROR:
                        WinHttpWebSocketClient::clients[hInternet]->OnError((WINHTTP_ASYNC_RESULT*)lpvStatusInformation);
                        break;

                default:
                        break;
                }
        }

}
```

The hInternet argument of the callback function serves as the HINTERNET handle upon which the callback was initially registered. This handle is employed to index a member within the global map container, clients, yielding a pointer to a WebSocketClient instance. The specific callback notification is conveyed through the dwInternetStatus argument. Notably, the data represented by the lpvStatusInformation and dwStatusInformationLength arguments varies according to the value of dwInternetStatus.

For instance, when dwInternetStatus assumes the values WINHTTP\_CALLBACK\_STATUS\_READ\_COMPLETE or WINHTTP\_CALLBACK\_STATUS\_WRITE\_COMPLETE, the lpvStatusInformation parameter contains a pointer to a WINHTTP\_WEB\_SOCKET\_STATUS structure, with dwStatusInformationLength indicating the size of the referenced data.

Our implementation selectively processes a subset of notifications provided by this callback function, each resulting in a modification of the state of the corresponding WebSocketClient instance. Specifically, the OnCallBack() method captures the status codes associated with these notifications. This information kept in the WebSocketClient instance where it can be exposed to users through a wrapper function.

```
VOID WebSocketClient::OnCallBack(const DWORD operation)
{
        completed_websocket_operation = operation;
}
```

The OnReadComplete() method within the WebSocketClient class is responsible for transferring raw data into a queued buffer of frames, from which users can subsequently query the availability of data for retrieval.

```
VOID WebSocketClient::OnReadComplete(const DWORD read, const WINHTTP_WEB_SOCKET_BUFFER_TYPE buffertype)
{
        bytesRX = read;
        rxBufferType = buffertype;
        status = ENUM_WEBSOCKET_STATE::CONNECTED;
        Frame frame;
        frame.frame_buffer.insert(frame.frame_buffer.begin(), rxBuffer.data(), rxBuffer.data() + read);
        frame.frame_type = buffertype;
        frame.frame_size = read;
        frames->push(frame);
}
```

The OnSendComplete() method updates internal fields that flag a successful send operation, which also triggers a change in the state of the websocket client.

```
VOID WebSocketClient::OnSendComplete(const DWORD sent)
{
        bytesTX = sent;
        status = ENUM_WEBSOCKET_STATE::CONNECTED;
        return;
}
```

Finally, the OnError() method captures any error-related information provided via the lpvStatusInformation argument.

```
VOID WebSocketClient::OnError(const WINHTTP_ASYNC_RESULT* result)
{
        SetError(result->dwError);
        Reset(false);
}
```

Establishing a connection to a server is accomplished via the client\_connect() function. This function is invoked with the server address (as a string), the port number (as an integer), a boolean value specifying whether the connection should be secure, and a pointer to an HINTERNET value. The function returns a DWORD value as well setting the HINTERNET argument. In the event of any error during the connection process, the function will set the HINTERNET argument to NULL and return a non-zero error code.

Internally, the client\_connect() function initializes an instance of the WebSocketClient class and utilizes it to establish the connection. Upon successful connection establishment, a pointer to this WebSocketClient instance is stored in the global clients container, with the instance's websocket handle serving as the unique key. This websocket handle is employed to uniquely identify a specific websocket connection, both within the DLL's internal operations and externally by the calling MetaTrader 5 program.

```
DWORD  WEBSOCK_API client_connect( const WCHAR* url, INTERNET_PORT port, DWORD secure, HINTERNET* websocketp_handle)
{
        DWORD errorCode = 0;
        auto client = std::make_shared<WebSocketClient>();

        if (client->Connect(url, port, secure) != NO_ERROR)
                errorCode = client->LastError();
        else
        {
                HINTERNET handle = client->WebSocketHandle();
                if (client->EnableCallBack())
                {
                        errorCode = client->LastError();
                        client->Close(WINHTTP_WEB_SOCKET_CLOSE_STATUS::WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS);
                        client->Free();
                        handle = NULL;
                }
                else
                {
                        clients[handle] = client;
                        *websocketp_handle = handle;
                }
        }
        return errorCode;
}
```

The initialization and establishment of websocket connections are managed by the Connect() method of the WebSocketClient class. The connection process is initially performed synchronously. Thereafter, the callback function is registered on the websocket handle through a call to the EnableCallBack() method, enabling asynchronous event notifications.

```
DWORD WebSocketClient::Connect(const WCHAR* host, const INTERNET_PORT port, const DWORD secure)
  {

   if((status != ENUM_WEBSOCKET_STATE::CLOSED))
     {
      ErrorCode = Close(WINHTTP_WEB_SOCKET_CLOSE_STATUS::WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS,NULL);
      return WEBSOCKET_ERROR_CLOSING_ACTIVE_CONNECTION;
     }

   status = ENUM_WEBSOCKET_STATE::CONNECTING;

// Return 0 for success
   if(hSession == NULL)
     {
      ErrorCode = Initialize();
      if(ErrorCode)
        {
         Reset(false);
         return ErrorCode;
        }
     }

// Cracked URL variable pointers
   URL_COMPONENTS UrlComponents;

// Create cracked URL buffer variables
   std::unique_ptr <WCHAR> scheme(new WCHAR[0x20]);
   std::unique_ptr <WCHAR> hostName(new WCHAR[0x100]);
   std::unique_ptr <WCHAR> urlPath(new WCHAR[0x1000]);

   DWORD dwFlags = 0;
   if(secure)
      dwFlags |= WINHTTP_FLAG_SECURE;

   if(scheme == NULL || hostName == NULL || urlPath == NULL)
     {
      ErrorCode = ERROR_NOT_ENOUGH_MEMORY;
      Reset();
      return ErrorCode;
     }

// Clear error's
   ErrorCode = 0;

// Setup UrlComponents structure
   memset(&UrlComponents, 0, sizeof(URL_COMPONENTS));
   UrlComponents.dwStructSize = sizeof(URL_COMPONENTS);
   UrlComponents.dwSchemeLength = -1;
   UrlComponents.dwHostNameLength = -1;
   UrlComponents.dwUserNameLength = -1;
   UrlComponents.dwPasswordLength = -1;
   UrlComponents.dwUrlPathLength = -1;
   UrlComponents.dwExtraInfoLength = -1;

// Get the individual parts of the url
   if(!WinHttpCrackUrl(host, NULL, 0, &UrlComponents))
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset();
      return ErrorCode;
     }

// Copy cracked URL hostName & UrlPath to buffers so they are separated
   if(wcsncpy_s(scheme.get(), 0x20, UrlComponents.lpszScheme, UrlComponents.dwSchemeLength) != 0 ||
      wcsncpy_s(hostName.get(), 0x100, UrlComponents.lpszHostName, UrlComponents.dwHostNameLength) != 0 ||
      wcsncpy_s(urlPath.get(), 0x1000, UrlComponents.lpszUrlPath, UrlComponents.dwUrlPathLength) != 0)
     {
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

   if(port == 0)
     {
      if((_wcsicmp(scheme.get(), L"wss") == 0) || (_wcsicmp(scheme.get(), L"https") == 0))
        {
         UrlComponents.nPort = INTERNET_DEFAULT_HTTPS_PORT;
        }
      else
         if((_wcsicmp(scheme.get(), L"ws") == 0) || (_wcsicmp(scheme.get(), L"http")) == 0)
           {
            UrlComponents.nPort = INTERNET_DEFAULT_HTTP_PORT;
           }
         else
           {
            ErrorCode = ERROR_INVALID_PARAMETER;
            Reset(false);
            return ErrorCode;
           }
     }
   else
      UrlComponents.nPort = port;

// Call the WinHttp Connect method
   hConnect = WinHttpConnect(hSession, hostName.get(), UrlComponents.nPort, 0);
   if(!hConnect)
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

// Create a HTTP request
   hRequest = WinHttpOpenRequest(hConnect, L"GET", urlPath.get(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, dwFlags);
   if(!hRequest)
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

// Set option for client certificate

   if(!WinHttpSetOption(hRequest, WINHTTP_OPTION_CLIENT_CERT_CONTEXT, WINHTTP_NO_CLIENT_CERT_CONTEXT, 0))
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

// Add WebSocket upgrade to our HTTP request
#pragma prefast(suppress:6387, "WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET does not take any arguments.")
   if(!WinHttpSetOption(hRequest, WINHTTP_OPTION_UPGRADE_TO_WEB_SOCKET, 0, 0))
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

// Send the WebSocket upgrade request.
   if(!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, 0, 0, 0, 0))
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

// Receive response from the server
   if(!WinHttpReceiveResponse(hRequest, 0))
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

// Finally complete the upgrade
   hWebSocket = WinHttpWebSocketCompleteUpgrade(hRequest, NULL);
   if(hWebSocket == 0)
     {
      // Handle error
      ErrorCode = GetLastError();
      Reset(false);
      return ErrorCode;
     }

   status = ENUM_WEBSOCKET_STATE::CONNECTED;

// Return should be zero
   return ErrorCode;
  }

DWORD WebSocketClient::EnableCallBack(VOID)
  {
   if(!WinHttpSetOption(hWebSocket, WINHTTP_OPTION_CONTEXT_VALUE, (LPVOID)this, sizeof(this)))
     {
      // Handle error
      ErrorCode = GetLastError();
      return ErrorCode;
     }

   if(WinHttpSetStatusCallback(hWebSocket, WebSocketCallback, WINHTTP_CALLBACK_FLAG_ALL_COMPLETIONS, 0) == WINHTTP_INVALID_STATUS_CALLBACK)
     {
      ErrorCode = GetLastError();
      return ErrorCode;
     }

   return ErrorCode;
  }
```

Upon establishing a connection to a server and obtaining a valid handle, users can initiate communication with the remote endpoint. Transmitting data to the server is performed through a call to the client\_send() function. This function requires the following parameters: a valid websocket handle, a WINHTTP\_WEB\_SOCKET\_BUFFER\_TYPE enumeration value specifying the type of websocket frame to be transmitted, a BYTE array containing the data payload, and an ulong argument indicating the size of the data array. The function returns a zero value if no immediate errors are encountered; otherwise, it returns a specific error code.

Internally, the WinHttpWebSocketSend() function is invoked asynchronously. Consequently, the return value of client\_send() represents an intermediate status, signifying the absence of preliminary errors during the setup of the send operation. The outcome of the actual data transmission is not returned synchronously. Instead, the result is communicated asynchronously via a notification accessible through the registered callback function. In the context of a successful send operation, a WINHTTP\_CALLBACK\_STATUS\_WRITE\_COMPLETE notification is anticipated. Conversely, if an error occurs during any operation (send or receive), a WINHTTP\_CALLBACK\_STATUS\_REQUEST\_ERROR notification is typically propagated to the callback function.

```
DWORD WEBSOCK_API client_send(HINTERNET websocket_handle, WINHTTP_WEB_SOCKET_BUFFER_TYPE buffertype, BYTE* message, DWORD length)
  {
   DWORD out = 0;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = WEBSOCKET_ERROR_INVALID_HANDLE;
   else
      out = clients[websocket_handle]->Send(buffertype, message, length);

   return out;
  }
```

Notifications received by the internal callback function can be retrieved using the client\_lastcallback\_notification() function. This function returns the most recent notification received by the callback for a specific connection, identified by the websocket handle provided as its sole argument. The subsequent code snippet illustrates a potential approach to handling these notifications within an MetaTrader 5 program. The symbolic constants corresponding to these notifications are defined in the asyncwinhttp.mqh file, which are derived from the original winhttp.h header file.

```
DWORD WEBSOCK_API client_lastcallback_notification(HINTERNET websocket_handle)
  {
   DWORD out = 0;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = WEBSOCKET_ERROR_INVALID_HANDLE;
   else
      out = clients[websocket_handle]->LastOperation();

   return out;
  }
```

Receiving data transmitted by the server initially requires placing the client in what is referred to as a polling state by invoking the client\_poll() function. This action internally calls the WinHttpWebSocketReceive() function within the WebSocketClient class. Similar to the send operation, the WinHTTP function is invoked asynchronously, resulting in the immediate return of an intermediate status.

The WebSocketClient class incorporates internal buffers to accommodate the raw data upon its arrival. Once a read operation is successfully completed, this data is enqueued within an internal data structure. This process is managed by the OnReadComplete() method of the WebSocketClient class. Upon completion of a read operation, the state of the websocket connection transitions, and it ceases to actively "listen" for incoming messages.

This implies that an asynchronous read request is not continuous and does not represent persistent polling. To retrieve subsequent messages from the server, the client\_poll() function must be invoked again. Essentially, calling client\_poll() places the websocket client in a temporary, non-blocking polling state, capturing data when it becomes available and subsequently triggering the WINHTTP\_CALLBACK\_STATUS\_READ\_COMPLETE notification.

The current state of the websocket client can be queried by calling the client\_status() function, which returns a value of the ENUM\_WEBSOCKET\_STATE enumeration type.

```
DWORD WEBSOCK_API client_poll(HINTERNET websocket_handle)
  {
   DWORD out = 0;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = WEBSOCKET_ERROR_INVALID_HANDLE;
   else
      out = clients[websocket_handle]->Receive(clients[websocket_handle]->rxBuffer.data(), (DWORD)clients[websocket_handle]->rxBuffer.size(), &clients[websocket_handle]->bytesRX, &clients[websocket_handle]->rxBufferType);

   return out;
  }

ENUM_WEBSOCKET_STATE WEBSOCK_API client_status(HINTERNET websocket_handle)
  {
   ENUM_WEBSOCKET_STATE out = {};
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = {};
   else
      out = clients[websocket_handle]->Status();

   return out;
  }
```

The retrieval of raw data received from the server is facilitated by the client\_read() function, which accepts the following arguments:

- a valid HINTERNET websocket handle,
- a reference to a pre-allocated BYTE array, an ulong value specifying the size of the aforementioned array,
- a reference to a WINHTTP\_WEB\_SOCKET\_BUFFER\_TYPE value.

The received data is written to the provided BYTE array, and the type of the websocket frame is copied to the buffertype argument. Critically, this function operates by reading from an internal queue of received frames, rather than directly interacting with the network socket. Consequently, client\_read() is a synchronous operation, independent of the asynchronous mechanisms of the WinHTTP library. A non-zero return value indicates a failure to copy data from the internal queue. Upon successful retrieval of a frame using this function, the frame is removed (dequeued) from the internal queue. The client\_readable() function can be invoked to determine the size of the data frame currently at the front of the queue of frames received from the server.

```
DWORD WEBSOCK_API client_read(HINTERNET websocket_handle, BYTE* out, DWORD out_size, WINHTTP_WEB_SOCKET_BUFFER_TYPE* buffertype)
  {
   DWORD rout = 0;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      rout = WEBSOCKET_ERROR_INVALID_HANDLE;
   else
      clients[websocket_handle]->Read(out, out_size, buffertype);

   return rout;
  }

DWORD WEBSOCK_API client_readable(HINTERNET websocket_handle)
  {
   DWORD out = 0;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = 0;
   else
      out = clients[websocket_handle]->ReadAvailable();

   return out;
  }
```

Error codes can be obtained by calling the client\_lasterror() function. The function returns a DWORD value of the last error encountered. Users can also obtain the current websocket handle value with client\_websocket\_handle(). This could be useful when trying to ascertain if a handle has been closed or not.

```
DWORD WEBSOCK_API  client_lasterror(HINTERNET websocket_handle)
  {
   DWORD out = 0;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = WEBSOCKET_ERROR_INVALID_HANDLE;
   else
      out = clients[websocket_handle]->LastError();

   return out;
  }

HINTERNET WEBSOCK_API client_websocket_handle(HINTERNET websocket_handle)
  {
   HINTERNET out = NULL;
   if(websocket_handle == NULL || clients.find(websocket_handle) == clients.end())
      out = NULL;
   else
      out = clients[websocket_handle]->WebSocketHandle();
   return out;
  }
```

Graceful termination of a connection to a server is initiated by invoking the client\_disconnect() function. This function does not return a value. However, it immediately transitions the state of the websocket client to a closing state. If a corresponding close frame is subsequently received from the server, the WINHTTP\_CALLBACK\_STATUS\_CLOSE\_COMPLETE notification will be triggered, thereby altering the websocket state to closed.

```
void WEBSOCK_API client_disconnect(HINTERNET websocket_handle)
  {

   if(clients.find(websocket_handle) != clients.end())
     {
      if(clients[websocket_handle]->WebSocketHandle() != NULL)
        {
         clients[websocket_handle]->Close(WINHTTP_WEB_SOCKET_CLOSE_STATUS::WINHTTP_WEB_SOCKET_SUCCESS_CLOSE_STATUS);
        }
     }

   return;
  }
```

The final function exported by the DLL is client\_reset(). Ideally, this function should be called after disconnecting from a server. Its purpose is to deallocate the internal memory buffers associated with the closed connection. While not strictly mandatory, invoking this function can be beneficial for reclaiming memory resources that may be required elsewhere during program execution. Calling client\_reset() effectively invalidates all data associated with the specified websocket handle, including error codes, error messages, and any unread data remaining in the internal queue of frames.

```
VOID WEBSOCK_API client_reset(HINTERNET websocket_handle)
  {
   if(clients.find(websocket_handle) != clients.end())
     {
      clients[websocket_handle]->Free();
      clients.erase(websocket_handle);
     }
  }
```

### Redefining the CWebsocket class

Before examining an MetaTrader 5 application that utilizes the functions detailed in the preceding section, a redefinition of the CWebsocket class, previously discussed in the aforementioned article, will be undertaken. This redefinition will repurpose the class to leverage the newly developed asynchronous websocket client. The source code for this adaptation is located in the asyncwebsocket.mqh file.

The ENUM\_WEBSOCKET\_STATE enumeration has been expanded to incorporate additional states that reflect the asynchronous nature of the client. The POLLING state is engaged when an asynchronous read operation is initiated. Upon the underlying socket receiving data and making it available for retrieval, the callback function signals the completion of the asynchronous read operation, and the state of the websocket client transitions to its default state: CONNECTED. Similarly, an asynchronous send operation transitions the state to SENDING. The outcome of this operation is communicated asynchronously via the callback function, whereby a successful transmission results in a return to the default CONNECTED state.

```
//+------------------------------------------------------------------+
//|   client state enumeration                                       |
//+------------------------------------------------------------------+

// client state
enum ENUM_WEBSOCKET_STATE
  {
   CLOSED = 0,
   CLOSING = 1,
   CONNECTING = 2,
   CONNECTED = 3,
   SENDING = 4,
   POLLING = 5
  };
```

Several new methods have been integrated into the CWebsocket class to accommodate its enhanced capabilities. The remaining methods retain their original signatures, with only their internal implementations modified to incorporate the new DLL dependency. These methods are as follows:

Connect(): This method serves as the initial point of interaction for establishing a connection to a server. It accepts the following parameters:

-  \_serveraddress: The complete address of the server (string data type).
-  \_port: The server's port number (ushort data type).
-  \_secure: A boolean value indicating whether a secure connection should be established (boolean data type).

The implementation of this method has been significantly simplified, as the majority of the connection establishment logic is now handled by the underlying DLL.

```
//+------------------------------------------------------------------------------------------------------+
//|Connect method used to set server parameters and establish client connection                          |
//+------------------------------------------------------------------------------------------------------+
bool CWebsocket::Connect(const string _serveraddress,const INTERNET_PORT port=443, bool secure = true)
  {
   if(initialized)
     {
      if(StringCompare(_serveraddress,serveraddress,false))
         Close();
      else
         return(true);
     }

   serveraddress = _serveraddress;

   int dot=StringFind(serveraddress,".");

   int ss=(dot>0)?StringFind(serveraddress,"/",dot):-1;

   serverPath=(ss>0)?StringSubstr(serveraddress,ss+1):"/";

   int sss=StringFind(serveraddress,"://");

   if(sss<0)
      sss=-3;

   serverName=StringSubstr(serveraddress,sss+3,ss-(sss+3));
   serverPort=port;

   DWORD connect_error = client_connect(serveraddress,port,ulong(secure),hWebSocket);

   if(hWebSocket<=0)
     {
      Print(__FUNCTION__," Connection to ", serveraddress, " failed. \n", GetErrorDescription(connect_error));
      return(false);
     }
   else
      initialized = true;

   return(true);
  }
```

If the Connect() method returns a boolean true value, indicating a successful connection, data transmission can commence via the WebSocket client. Two methods are provided for this purpose:

-  SendString(): This method accepts a string as input.
-  Send(): This method accepts an unsigned character array as its sole parameter.

Both methods return a boolean true upon successful initiation of the send operation and internally invoke the private method clientsend(), which manages all send operations for the class.

```
//+------------------------------------------------------------------+
//|public method for sending raw string messages                     |
//+------------------------------------------------------------------+
bool CWebsocket::SendString(const string msg)
  {
   if(!initialized || hWebSocket == NULL)
     {
      Print(__FUNCTION__, " No websocket connection ");
      return(false);
     }

   if(StringLen(msg)<=0)
     {
      Print(__FUNCTION__, " Message buffer is empty ");
      return(false);
     }

   BYTE msg_array[];

   StringToCharArray(msg,msg_array,0,WHOLE_ARRAY);

   ArrayRemove(msg_array,ArraySize(msg_array)-1,1);

   DWORD len=(ArraySize(msg_array));

   return(clientsend(msg_array,WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE));
  }

//+------------------------------------------------------------------+
//|Public method for sending data prepackaged in an array            |
//+------------------------------------------------------------------+
bool CWebsocket::Send(BYTE &buffer[])
  {
   if(!initialized || hWebSocket == NULL)
     {
      Print(__FUNCTION__, " No websocket connection ");
      return(false);
     }

   return(clientsend(buffer,WINHTTP_WEB_SOCKET_BINARY_MESSAGE_BUFFER_TYPE));
  }
```

Invoking the Poll() method initiates an asynchronous read operation at the lower level, transitioning the websocket client into the POLLING state. This state signifies that the client is awaiting a response from the server.

```
//+------------------------------------------------------------------+
//|  asynchronous read operation (polls for server response)         |
//+------------------------------------------------------------------+
ulong CWebsocket::Poll(void)
  {
   if(hWebSocket!=NULL)
      return client_poll(hWebSocket);
   else
      return WEBSOCKET_ERROR_INVALID_HANDLE;
  }
```

To ascertain whether data has been received and successfully read by the client, two options are available to the user:

- CallBackResult(): This method checks the last notification received from the callback function. A successful read operation should result in a read complete notification.
- ReadAvailable(): This method returns the size (in bytes) of the data currently available for retrieval in the internal buffer.

```
//+------------------------------------------------------------------+
//| call back notification                                           |
//+------------------------------------------------------------------+
ulong CWebsocket::CallBackResult(void)
  {
   if(hWebSocket!=NULL)
      return client_lastcallback_notification(hWebSocket);
   else
      return WINHTTP_CALLBACK_STATUS_DEFAULT;
  }
//+------------------------------------------------------------------+
//|  check if any data has read from the server                      |
//+------------------------------------------------------------------+
ulong CWebsocket::ReadAvailable(void)
  {
   if(hWebSocket!=NULL)
      return(client_readable(hWebSocket));
   else
      return 0;
  }
```

The raw data transmitted by the server can then be accessed using either the Read() or ReadString() methods. Both methods return the size of the data received. ReadString() requires a string variable passed by reference, into which the received data will be written, whereas Read() writes the data to an unsigned character array.

```
//+------------------------------------------------------------------+
//|public method for reading data sent from the server               |
//+------------------------------------------------------------------+
ulong CWebsocket::Read(BYTE &buffer[],WINHTTP_WEB_SOCKET_BUFFER_TYPE &buffertype)
  {
   if(!initialized || hWebSocket == NULL)
     {
      Print(__FUNCTION__, " No websocket connection ");
      return(false);
     }

   ulong bytes_read_from_socket=0;

   clientread(buffer,buffertype,bytes_read_from_socket);

   return(bytes_read_from_socket);

  }
//+------------------------------------------------------------------+
//|public method for reading data sent from the server               |
//+------------------------------------------------------------------+
ulong CWebsocket::ReadString(string &_response)
  {
   if(!initialized || hWebSocket == NULL)
     {
      Print(__FUNCTION__, " No websocket connection ");
      return(false);
     }

   ulong bytes_read_from_socket=0;

   ZeroMemory(rxbuffer);

   WINHTTP_WEB_SOCKET_BUFFER_TYPE rbuffertype;

   clientread(rxbuffer,rbuffertype,bytes_read_from_socket);

   _response=(bytes_read_from_socket)?CharArrayToString(rxbuffer):"";

   return(bytes_read_from_socket);
  }
```

When the WebSocket client is no longer required, the connection to the server can be terminated using either the Close() method. The Abort() method differs from Close() in that it forces closure of a websocket connection by closing the underlying handles, which also resets the values of internal class properties to their default states. The method can be explicitly called to perform resource cleanup.

Finally, the WebSocketHandle() method returns the underlying HINTERNET websocket handle.

```
//+------------------------------------------------------------------+
//| Closes a websocket client connection                             |
//+------------------------------------------------------------------+
void CWebsocket::Close(void)
  {
   if(!initialized || hWebSocket == NULL)
      return;
   else
      client_disconnect(hWebSocket);
  }
//+--------------------------------------------------------------------------+
//|method for abandoning a client connection. All previous server connection |
//|   parameters are reset to their default state                            |
//+--------------------------------------------------------------------------+
void CWebsocket::Abort(void)
  {
   client_reset(hWebSocket);
   reset();
  }
//+------------------------------------------------------------------+
//|   websocket handle                                               |
//+------------------------------------------------------------------+
HINTERNET CWebsocket::WebSocketHandle(void)
  {
   if(hWebSocket!=NULL)
      return client_websocket_handle(hWebSocket);
   else
      return NULL;
  }
```

### Using the DLL

This section presents an illustrative program designed to demonstrate the utilization of the asyncwinhttpwebsockets.dll. The program incorporates a graphical user interface (GUI) that establishes a connection to the websocket echo service hosted at https://echo.websocket.org, a resource specifically provisioned for testing websocket client implementations. Construction of this application necessitates the freely available Easy And Fast GUI MQL5 library. The program is implemented as an Expert Advisor (EA) within the MetaTrader 5 environment. The user interface features two buttons, enabling the establishment and termination of a connection to the designated server. Additionally, a text input field is provided to allow users to enter messages for transmission to the server.

The operations performed by the websocket client are recorded and displayed, with each log entry timestamped to indicate the time of occurrence. The source code for this program is provided below.

```
//+------------------------------------------------------------------+
//|                                                         Echo.mq5 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#include <EasyAndFastGUI\WndCreate.mqh>
#include <asyncwebsocket.mqh>
//+------------------------------------------------------------------+
//| Gui application class                                            |
//+------------------------------------------------------------------+
class CApp:public CWndCreate
  {
protected:

   CWindow           m_window;                   //main window

   CTextEdit         m_rx;                       //text input to specify messages to be sent

   CTable            m_tx;                       //text box displaying received messages from the server

   CButton           m_connect;                  //connect button

   CButton           m_disconnect;               //disconnect button

   CTimeCounter      m_timer_counter;                //On timer objects

   CWebsocket*        m_websocket;               //websocket connection
public:
                     CApp(void);                 //constructor
                    ~CApp(void);                 //destructor

   void              OnInitEvent(void);
   void              OnDeinitEvent(const int reason);

   virtual void      OnEvent(const int id, const long &lparam, const double &dparam,const string &sparam);
   void              OnTimerEvent(void);
   bool              CreateGUI(void);

protected:

private:
   uint              m_row_index;
   void              EditTable(const string newtext);
  };
//+------------------------------------------------------------------+
//|  constructor                                                     |
//+------------------------------------------------------------------+
CApp::CApp(void)
  {
   m_row_index = 0;
   m_timer_counter.SetParameters(10,50);
   m_websocket = new CWebsocket();
  }
//+------------------------------------------------------------------+
//|   destructor                                                     |
//+------------------------------------------------------------------+
CApp::~CApp(void)
  {
   if(CheckPointer(m_websocket) == POINTER_DYNAMIC)
      delete m_websocket;
  }
//+------------------------------------------------------------------+
//| On initialization                                                |
//+------------------------------------------------------------------+
void CApp::OnInitEvent(void)
  {
  }
//+------------------------------------------------------------------+
//| On DeInitilization                                               |
//+------------------------------------------------------------------+
void CApp::OnDeinitEvent(const int reason)
  {
   CWndEvents::Destroy();
  }
//+------------------------------------------------------------------+
//| on timer event                                                   |
//+------------------------------------------------------------------+
void CApp::OnTimerEvent(void)
  {
   CWndEvents::OnTimerEvent();

   if(m_timer_counter.CheckTimeCounter())
     {

      ENUM_WEBSOCKET_STATE client_state = m_websocket.ClientState();
      ulong operation = m_websocket.CallBackResult();

      switch((int)operation)
        {
         case WINHTTP_CALLBACK_STATUS_CLOSE_COMPLETE:
            if(client_state == CLOSED)
              {
               EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Disconnected]");
               m_websocket.Abort();
              }
            break;
         case WINHTTP_CALLBACK_STATUS_WRITE_COMPLETE:
            if(client_state!=POLLING)
              {
               m_websocket.Poll();
               EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Send Complete]");
              }
            break;

         case WINHTTP_CALLBACK_STATUS_READ_COMPLETE:
            if(m_websocket.ReadAvailable())
              {
               string response;
               m_websocket.ReadString(response);
               EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Received]-> "+response);
              }
            break;
         case WINHTTP_CALLBACK_STATUS_REQUEST_ERROR:
            EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Error]-> "+m_websocket.LastErrorMessage());
            m_websocket.Abort();
            break;
         default:
            break;
        }
     }
  }
//+------------------------------------------------------------------+
//| create the gui                                                   |
//+------------------------------------------------------------------+
bool CApp::CreateGUI(void)
  {
//---check the websocket object
   if(CheckPointer(m_websocket) == POINTER_INVALID)
     {
      Print(__FUNCTION__," Failed to create websocket client object ", GetLastError());
      return false;
     }
//---initialize window creation
   if(!CWndCreate::CreateWindow(m_window,"Connect to https://echo.websocket.org Echo Server ",1,1,750,300,true,false,true,false))
      return(false);
//---
   if(!CWndCreate::CreateTextEdit(m_rx,"",m_window,0,false,0,25,750,750,"Click connect button below, input your message here, then press enter key to send"))
      return(false);
//---
   if(!CWndCreate::CreateButton(m_connect,"Connect",m_window,0,5,50,240,false,false,clrNONE,clrNONE,clrNONE,clrNONE,clrNONE))
      return(false);
//---create text edit for width in frequency units
   if(!CWndCreate::CreateButton(m_disconnect,"Disonnect",m_window,0,500,50,240,false,false,clrNONE,clrNONE,clrNONE,clrNONE,clrNONE))
      return(false);
//---create text edit for amount of padding
   string tableheader[1] = {"Client Operations Log"};
   if(!CWndCreate::CreateTable(m_tx,m_window,0,1,10,tableheader,5,75,0,0,true,true,5))
      return(false);
//---
   m_tx.TextAlign(0,ALIGN_LEFT);
   m_tx.ShowTooltip(false);
   m_tx.DataType(0,TYPE_STRING);
   m_tx.IsDropdown(false);
   m_tx.SelectableRow(false);
   int cwidth[1] = {740};
   m_tx.ColumnsWidth(cwidth);
//---init events
   CWndEvents::CompletedGUI();
//---
   return(true);
  }
//+------------------------------------------------------------------+
//| edit the table                                                   |
//+------------------------------------------------------------------+
void CApp::EditTable(const string newtext)
  {
   if(newtext==NULL)
      return;

   if((m_row_index+1)==m_tx.RowsTotal())
     {
      m_tx.AddRow(m_row_index+1,true);
      m_tx.Update();
     }

   m_tx.SetValue(0,m_row_index++,newtext,0,true);
   m_tx.Update(true);
  }
//+------------------------------------------------------------------+
//| Event handler                                                    |
//+------------------------------------------------------------------+
void CApp::OnEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
   if(id==CHARTEVENT_CUSTOM+ON_END_EDIT)
     {
      if(lparam==m_rx.Id())
        {
         if(m_websocket.ClientState() == CONNECTED)
           {
            string textinput = m_rx.GetValue();
            if(StringLen(textinput)>0)
              {
               EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Sending]-> "+textinput);
               m_websocket.SendString(textinput);
              }
           }
        }
      return;
     }
   else
      if(id == CHARTEVENT_CUSTOM+ON_CLICK_BUTTON)
        {
         if(lparam==m_connect.Id())
           {
            if(m_websocket.ClientState() != CONNECTED)
              {
               EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Connecting]");
               if(m_websocket.Connect("https://echo.websocket.org/"))
                 {
                  EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Connected]");
                  m_websocket.Poll();
                 }
               else
                  EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[FailedToConnect]");
              }
            return;
           }
         if(lparam==m_disconnect.Id())
           {
            if(m_websocket.ClientState() != CLOSED)
              {
               EditTable("["+TimeToString(TimeCurrent(),TIME_MINUTES|TIME_SECONDS)+"]"+"[Disconnecting]");
               m_websocket.Close();
              }
           }
         return;
        }
  }
//+------------------------------------------------------------------+
CApp app;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   ulong tick_counter=::GetTickCount();
//---
   app.OnInitEvent();
//---
   if(!app.CreateGUI())
     {
      ::Print(__FUNCTION__," > error");
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {

   app.OnDeinitEvent(reason);
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(void)
  {
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
   app.OnTimerEvent();
  }
//+------------------------------------------------------------------+
//| Trade function                                                   |
//+------------------------------------------------------------------+
void OnTrade(void)
  {
  }
//+------------------------------------------------------------------+
//| ChartEvent function                                              |
//+------------------------------------------------------------------+
void OnChartEvent(const int    id,
                  const long   &lparam,
                  const double &dparam,
                  const string &sparam)
  {
   app.ChartEvent(id,lparam,dparam,sparam);
  }
//+------------------------------------------------------------------+
```

The graphic below shows how the program works.

![Echo EA Demonstration](https://c.mql5.com/2/136/EchoDemo__1.gif)

### Conclusion

This article detailed the development of a WebSocket client for MetaTrader 5 utilizing the WinHTTP library in an asynchronous operational mode. A dedicated class was constructed to encapsulate this functionality, and its implementation was demonstrated within an Expert Advisor (EA) designed to interact with the echo server hosted at echo.websocket.org. The complete source code, including that of the dynamic link library, is provided in the supplementary materials. Specifically, the C++ source files, along with a CMakeLists.txt build configuration file, are located within the designated C++ directory. Additionally, the MQL5 directory in the supplementary materials contains a pre-compiled asyncwinhttpwebsockets.dll file for immediate deployment.

For users wishing to build the client library from the provided source code, the CMake build system is [required](https://www.mql5.com/go?link=https://cmake.org/download/ "https://cmake.org/download/"). If CMake is installed, the graphical user interface (cmake-gui) can be invoked. The user must then specify the source code directory, which corresponds to the location of the CMakeLists.txt file (i.e., MqlWebsocketsClientDLL\\Source\\C++), and a separate build directory, which can be created at any desired location.

Thereafter, clicking the "Configure" button will initiate the configuration process. A dialogue window will prompt the user to "Specify the generator for this project," where the appropriate version of Visual Studio installed on the system should be selected. Under the "Optional platform for generator" setting, users can specify "Win32" to compile a 32-bit version of the DLL; otherwise, leaving this field blank will result in a default 64-bit compilation. Upon clicking "Finish," CMake will process the initial configuration.

An error notification will then appear, indicating the necessity to configure specific entries within the CMakeLists.txt file. To address this, the user should locate the entry labeled "ADDITIONAL\_LIBRARY\_DEPENDENCIES," click in the adjacent field, and navigate to the directory containing the winhttp.lib file.

Following this, the user should locate the entry labeled "OUTPUT\_DIRECTORY\_Xxx\_RELEASE" (where "Xxx" denotes the architecture, X64 or X86) and set the corresponding path to the "Libraries" folder of a MetaTrader installation.

After configuring these options, clicking "Configure" again should complete the configuration process without further error notifications. The build file can then be generated by clicking "Generate." Successful generation will activate the "Open Project" button, which, when clicked, will open the generated Visual Studio project file.

To build the DLL, the user should select "Build" then "Build Solution" within Visual Studio. The resulting DLL will be available within a few seconds.

| FIle or Folder name | Description |
| --- | --- |
| MqlWebsocketsClientDLL\\Source\\C++ | Folder contains the full source code files for asycnwinhttpwebsockets.dll |
| MqlWebsocketsClientDLL\\Source\\MQL5\\Include\\asyncwinhttp.mqh | Contains the import directive, that lists all functions exposed by the asycnwinhttpwebsockets.dll |
| MqlWebsocketsClientDLL\\Source\\MQL5\\Include\\asyncwebsocket.mqh | Contains the definition of the CWebsocket class that wraps the functionality provided by the underlying DLL functions. |
| MqlWebsocketsClientDLL\\Source\\MQL5\\Experts\\Echo.mq5 | The code file for an EA that demonstraites the application of the DLL |
| MqlWebsocketsClientDLL\\Source\\MQL5\\Experts\\Echo.ex5 | The compiled EA that demonstraites the application of the DLL |
| MqlWebsocketsClientDLL\\Source\\MQL5\\Libraries\\asycnwinhttpwebsockets.dll | The compiled DLL providing asynchronous Winhttp websocket functionality |

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/17877.zip "Download all attachments in the single ZIP archive")

[asyncwinhttp.mqh](https://www.mql5.com/en/articles/download/17877/asyncwinhttp.mqh "Download asyncwinhttp.mqh")(13.42 KB)

[asyncwebsocket.mqh](https://www.mql5.com/en/articles/download/17877/asyncwebsocket.mqh "Download asyncwebsocket.mqh")(19.09 KB)

[Echo.mq5](https://www.mql5.com/en/articles/download/17877/echo.mq5 "Download Echo.mq5")(10.84 KB)

[Echo.ex5](https://www.mql5.com/en/articles/download/17877/echo.ex5 "Download Echo.ex5")(306.48 KB)

[MqlWebsocketsClientDLL.zip](https://www.mql5.com/en/articles/download/17877/mqlwebsocketsclientdll.zip "Download MqlWebsocketsClientDLL.zip")(339.6 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Building Volatility models in MQL5 (Part I): The Initial Implementation](https://www.mql5.com/en/articles/20589)
- [Bivariate Copulae in MQL5 (Part 2): Implementing Archimedean copulae in MQL5](https://www.mql5.com/en/articles/19931)
- [Bivariate Copulae in MQL5 (Part 1): Implementing Gaussian and Student's t-Copulae for Dependency Modeling](https://www.mql5.com/en/articles/18361)
- [Dynamic mode decomposition applied to univariate time series in MQL5](https://www.mql5.com/en/articles/19188)
- [Singular Spectrum Analysis in MQL5](https://www.mql5.com/en/articles/18777)
- [Resampling techniques for prediction and classification assessment in MQL5](https://www.mql5.com/en/articles/17446)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/485546)**
(9)


![Shephard Mukachi](https://c.mql5.com/avatar/2017/5/5920C545-5DFD.jpg)

**[Shephard Mukachi](https://www.mql5.com/en/users/mukachi)**
\|
2 May 2025 at 21:17

**Ryan L Johnson [#](https://www.mql5.com/en/forum/485546#comment_56596595):**

It might help to see some example code at:

The same basic premise applies to an EA.

That is an excellent idea.

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
15 May 2025 at 23:47

I thank both of you for your response to my question.  I must have missed the overloaded function definitions and only read about the first.  Do you possibly know if Terminal is smart enough to parallel process the iCustom calls to maximize processor utilization's as I plan to vary the symbol parameter for each of the 28 pairs and plan to have multiple iCustom calls like the Brooky Trend Strength.

Also can either of you tell me where I can post comments on bugs in MQ5 and also where suggestions for the Mq administrators.  I have found a few, most recently the **Bars** difference between the terminal and the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ").  Also, I have a 3 screem setup with the main display on the far left.  Trying to move a panel. such as the Navigator or Market panels, from the left to the right is very tedious.  The drag mouse pointer is on the left most screen but the dragging panel is in the middle.  I think either the Terminal or Windows is going crazy when the mouse moves one pixel and then switches displays to move the panel one pixel and back again

![Silk Road Trading LLC](https://c.mql5.com/avatar/2025/5/68239006-fc9d.png)

**[Ryan L Johnson](https://www.mql5.com/en/users/rjo)**
\|
16 May 2025 at 00:13

**CapeCoddah [#](https://www.mql5.com/en/forum/485546#comment_56713270):**

Do you possibly know if Terminal is smart enough to parallel process the iCustom calls to maximize processor utilization's as I plan to vary the symbol parameter for each of the 28 pairs and plan to have multiple iCustom calls like the Brooky Trend Strength.

Not a problem. You just need separate instances of indicator handles and CopyBuffer()'s. Even though all indicators run in the same thread, you can run 100 or so indicator instances.

**CapeCoddah [#](https://www.mql5.com/en/forum/485546#comment_56713270):**

Also can either of you tell me where I can post comments on bugs in MQ5 and also where suggestions for the Mq administrators.  I have found a few, most recently the **Bars** difference between the terminal and the [strategy tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ").

Bars() will hiccup where there is missing price data--where rates\_total will not. If I remember correctly what I read in the past, Bars() can be fixed by referencing time stamps. Might be worth a search.

**CapeCoddah [#](https://www.mql5.com/en/forum/485546#comment_56713270):**

I have a 3 screem setup with the main display on the far left.  Trying to move a panel. such as the Navigator or Market panels, from the left to the right is very tedious.  The drag mouse pointer is on the left most screen but the dragging panel is in the middle.  I think either the Terminal or Windows is going crazy when the mouse moves one pixel and then switches displays to move the panel one pixel and back again

I really don't know on this one. I have 3 computers, each having its own monitor and terminal. I do know that Windows generally has multiple monitor display settings, including picture-in-picture maybe as a workaround.

Can someone else with real multiple monitors on a single machine chime in here, please?

![CapeCoddah](https://c.mql5.com/avatar/avatar_na2.png)

**[CapeCoddah](https://www.mql5.com/en/users/capecoddah)**
\|
16 May 2025 at 09:42

Great Information!!!

Thanks Ryan, your comment regarding bars vs rates\_total is appropriate.  My problem is that the two are identical in Terminal but in the [STrategy Tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ") Visualize, Bars is one greater which led to my bobo by not reading the documentation to the end.  I am going to take your input and use it for iCustom.  I presume that there must be a separate iCustom address for each combination of Symbol and Time specifications.

Also, is there any way for an EA to display Text on the screen in the Strategy Tester?  In Mq4 it did it automatically but not now.  I use a lot of class objects to display information and putting a second copy in the template slows the Strategy Tester even more.

On the 3 panel display, I think the problem is the terminal does not properly update the monitor location when the mouse moves from screen 2 to screen 1.

I have 2 mini pcs that each support 3 monitors so I have the 3 screens attached to both minis and use HDMI1 for one pc and HDMI2 for the other.  Works great with 43" Fire Tvs although you must make sure the remotes are properly configured to control just one monitor (call amazon support).  The only drawback is the on off button shuts down all monitors and sometimes I need to pull the plug to synchronize power.

CapeCoddah

![Silk Road Trading LLC](https://c.mql5.com/avatar/2025/5/68239006-fc9d.png)

**[Ryan L Johnson](https://www.mql5.com/en/users/rjo)**
\|
16 May 2025 at 19:58

**CapeCoddah [#](https://www.mql5.com/en/forum/485546#comment_56715852):**

My problem is that the two are identical in Terminal but in the [STrategy Tester](https://www.mql5.com/en/articles/239 "Article: The Fundamentals of Testing in MetaTrader 5 ") Visualize, Bars is one greater which led to my bobo by not reading the documentation to the end.  I am going to take your input and use it for iCustom.  I presume that there must be a separate iCustom address for each combination of Symbol and Time specifications.

1. A single indicator file in a single directory can be reused by multiple instances of iCustom().
2. A single indicator handle can be reused by multiple instances of CopyBuffer().
3. I now understand why you're using Bars(), as rates\_total alone is limited to a single timeframe. Presumably, you're using Bars() in a separate loop for each timeframe.

**CapeCoddah [#](https://www.mql5.com/en/forum/485546#comment_56715852):**

Also, is there any way for an EA to display Text on the screen in the Strategy Tester?  In Mq4 it did it automatically but not now.  I use a lot of class objects to display information and putting a second copy in the template slows the Strategy Tester even more.

Not that I'm aware of. You're already using the only method that I know from the Testing Visualization MT5 Help page.

**CapeCoddah [#](https://www.mql5.com/en/forum/485546#comment_56715852):**

On the 3 panel display, I think the problem is the terminal does not properly update the monitor location when the mouse moves from screen 2 to screen 1.

Unfortunately, there's no way for me to test this with my own setup. Are you stretching a single MT5 terminal screen across all monitors? I have seen others fix issues that way.


![Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://c.mql5.com/2/137/Building_a_Custom_Market_Regime_Detection_System_in_MQL5_Part_1.png)[Building a Custom Market Regime Detection System in MQL5 (Part 2): Expert Advisor](https://www.mql5.com/en/articles/17781)

This article details building an adaptive Expert Advisor (MarketRegimeEA) using the regime detector from Part 1. It automatically switches trading strategies and risk parameters for trending, ranging, or volatile markets. Practical optimization, transition handling, and a multi-timeframe indicator are included.

![Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization](https://c.mql5.com/2/137/logo-17865.png)[Automating Trading Strategies in MQL5 (Part 15): Price Action Harmonic Cypher Pattern with Visualization](https://www.mql5.com/en/articles/17865)

In this article, we explore the automation of the Cypher harmonic pattern in MQL5, detailing its detection and visualization on MetaTrader 5 charts. We implement an Expert Advisor that identifies swing points, validates Fibonacci-based patterns, and executes trades with clear graphical annotations. The article concludes with guidance on backtesting and optimizing the program for effective trading.

![From Basic to Intermediate: FOR Statement](https://c.mql5.com/2/94/Do_b4sico_ao_intermediqrio_Comando_FOR___LOGO.png)[From Basic to Intermediate: FOR Statement](https://www.mql5.com/en/articles/15406)

In this article, we will look at the most basic concepts of the FOR statement. It is very important to understand everything that will be shown here. Unlike the other statements we've talked about so far, the FOR statement has some quirks that quickly make it very complex. So don't let stuff like this accumulate. Start studying and practicing as soon as possible.

![Atmosphere Clouds Model Optimization (ACMO): Practice](https://c.mql5.com/2/95/Atmosphere_Clouds_Model_Optimization__LOGO___1.png)[Atmosphere Clouds Model Optimization (ACMO): Practice](https://www.mql5.com/en/articles/15921)

In this article, we will continue diving into the implementation of the ACMO (Atmospheric Cloud Model Optimization) algorithm. In particular, we will discuss two key aspects: the movement of clouds into low-pressure regions and the rain simulation, including the initialization of droplets and their distribution among clouds. We will also look at other methods that play an important role in managing the state of clouds and ensuring their interaction with the environment.

[![](https://www.mql5.com/ff/si/5k7a2kbftss6k97n82.png)](https://www.mql5.com/ff/go?link=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F1171%3Futm_source%3Dwww.mql5.com%26utm_medium%3Ddisplay.footer%26utm_term%3Dbest.vps%26utm_content%3Drent.vps%26utm_campaign%3D0622.MQL5.com.Internal&a=nwegcasiojnqcoyrdlgofmjtfardztwf&s=d64d6f3c87f2458cba81f6d7b6694dd9e89dd354d4abc1d0584e405285806c9f&v=1&host=https%3A%2F%2Fwww.mql5.com%2Fff%2F&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&uid=nbbziygmkzjqrhbwbpnabymgmyqpukwi&ssn=1769177929908903507&ssn_dr=0&ssn_sr=0&fv_date=1769177929&ref=https%3A%2F%2Fwww.mql5.com%2Fen%2Farticles%2F17877&back_ref=https%3A%2F%2Fwww.google.com%2F&title=Websockets%20for%20MetaTrader%205%3A%20Asynchronous%20client%20connections%20with%20the%20Windows%20API%20-%20MQL5%20Articles&scr_res=1920x1080&ac=176917792938529437&fz_uniq=5068086814360663368&sv=2552)

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