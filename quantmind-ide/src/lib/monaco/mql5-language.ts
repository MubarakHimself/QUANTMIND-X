/**
 * MQL5 Language Definition for Monaco Editor
 *
 * Provides syntax highlighting, tokenization, and language configuration
 * for MQL5 (MetaQuotes Language 5) used in MetaTrader 5.
 */

import type * as Monaco from 'monaco-editor';

// MQL5 Keywords
const mql5Keywords = [
  // Control flow
  'if', 'else', 'switch', 'case', 'default', 'for', 'while', 'do', 'break', 'continue', 'return',
  // Declarations
  'class', 'struct', 'enum', 'interface', 'template', 'typedef',
  // Access modifiers
  'public', 'private', 'protected', 'virtual', 'override', 'const', 'static', 'extern', 'input',
  // Types
  'void', 'bool', 'char', 'uchar', 'short', 'ushort', 'int', 'uint', 'long', 'ulong', 'float', 'double',
  'string', 'datetime', 'color', 'array', 'object',
  // Other
  'new', 'delete', 'this', 'operator', 'sizeof', 'typename', 'import', 'export', 'property', 'script',
  'strict', 'pack', 'alignas', 'alignof', 'constexpr', 'noexcept'
];

// MQL5 Built-in Functions
const mql5Builtins = [
  // Trading functions
  'OrderSend', 'OrderSendAsync', 'OrderCalcMargin', 'OrderCalcProfit', 'OrderCheck',
  'PositionOpen', 'PositionClose', 'PositionModify', 'PositionGetDouble', 'PositionGetInteger',
  'PositionGetString', 'PositionGetSymbol', 'PositionGetTicket', 'PositionsTotal', 'PositionSelect',
  'PositionGetMagic', 'PositionGetIdentifier',
  // Order functions
  'OrderGetDouble', 'OrderGetInteger', 'OrderGetString', 'OrderGetTicket', 'OrdersTotal', 'OrderSelect',
  // Symbol functions
  'SymbolInfoDouble', 'SymbolInfoInteger', 'SymbolInfoString', 'SymbolInfoTick', 'SymbolInfoSessionQuote',
  'SymbolInfoSessionTrade', 'SymbolName', 'SymbolSelect', 'SymbolsTotal', 'Symbol', 'SymbolExist',
  // Account functions
  'AccountInfoDouble', 'AccountInfoInteger', 'AccountInfoString', 'AccountBalance', 'AccountCredit',
  'AccountEquity', 'AccountFreeMargin', 'AccountMargin', 'AccountProfit', 'AccountLeverage',
  'AccountCurrency', 'AccountName', 'AccountNumber', 'AccountServer', 'AccountCompany',
  // Market Info
  'MarketInfo', 'iOpen', 'iClose', 'iHigh', 'iLow', 'iVolume', 'iTime', 'iHighest', 'iLowest',
  'iBarShift', 'iBars', 'CopyOpen', 'CopyClose', 'CopyHigh', 'CopyLow', 'CopyVolume', 'CopyTime',
  'CopyRates', 'CopyBuffer', 'CopyTicks', 'CopyTicksRange',
  // Indicator functions
  'iAC', 'iAD', 'iADX', 'iAlligator', 'iATR', 'iBearsPower', 'iBands', 'iBandsOnArray',
  'iBullsPower', 'iCCI', 'iCCIOnArray', 'iDeMarker', 'iEnvelopes', 'iEnvelopesOnArray',
  'iForce', 'iFractals', 'iGator', 'iIchimoku', 'iMA', 'iMAOnArray', 'iMACD', 'iMFI',
  'iMomentum', 'iMomentumOnArray', 'iOBV', 'iOsMA', 'iRSI', 'iRSIOnArray', 'iRVI', 'iSAR',
  'iStdDev', 'iStdDevOnArray', 'iStochastic', 'iWPR', 'iCustom', 'IndicatorCreate',
  // String functions
  'StringConcatenate', 'StringFind', 'StringGetCharacter', 'StringGetDouble', 'StringGetInteger',
  'StringInit', 'StringLen', 'StringReplace', 'StringSetCharacter', 'StringSetDouble', 'StringSetInteger',
  'StringSubstr', 'StringToLower', 'StringToUpper', 'StringTrimLeft', 'StringTrimRight', 'StringSplit',
  'StringToCharArray', 'StringToColor', 'StringToDouble', 'StringToInteger', 'StringToShortArray', 'StringToTime',
  'StringFormat', 'StringCompare', 'StringFill', 'StringAdd', 'StringBufferLen',
  // Array functions
  'ArrayBSearch', 'ArrayCopy', 'ArrayCompare', 'ArrayFill', 'ArrayFree', 'ArrayGetAsSeries',
  'ArrayInitialize', 'ArrayIsDynamic', 'ArrayIsSeries', 'ArrayMaximum', 'ArrayMinimum', 'ArrayRange',
  'ArrayResize', 'ArrayReverse', 'ArraySetAsSeries', 'ArraySize', 'ArraySort', 'ArrayCopyRates',
  'ArrayCopySeries', 'ArrayDimension', 'ArrayType',
  // Math functions
  'MathAbs', 'MathArccos', 'MathArcsin', 'MathArctan', 'MathArctan2', 'MathCeil', 'MathCos',
  'MathExp', 'MathFloor', 'MathLog', 'MathLog10', 'MathMax', 'MathMin', 'MathMod', 'MathPow',
  'MathRound', 'MathSin', 'MathSqrt', 'MathTan', 'MathExpm1', 'MathLog1p', 'MathIsValidNumber',
  'MathRandom', 'MathRand', 'MathSrand', 'MathSwap',
  // Time functions
  'TimeCurrent', 'TimeTradeServer', 'TimeLocal', 'TimeToStruct', 'StructToTime', 'TimeToString',
  'TimeToDateTime', 'Day', 'DayOfWeek', 'DayOfYear', 'Hour', 'Minute', 'Month', 'Seconds', 'Year',
  'TimeDayOfWeek', 'TimeDayOfYear', 'TimeHour', 'TimeMinute', 'TimeMonth', 'TimeSeconds', 'TimeYear',
  // File functions
  'FileOpen', 'FileClose', 'FileDelete', 'FileFlush', 'FileIsEnding', 'FileIsExist', 'FileIsLineEnding',
  'FileReadArray', 'FileReadBool', 'FileReadDatetime', 'FileReadDouble', 'FileReadFloat', 'FileReadInteger',
  'FileReadLong', 'FileReadNumber', 'FileReadString', 'FileReadStruct', 'FileSeek', 'FileSize', 'FileTell',
  'FileWrite', 'FileWriteArray', 'FileWriteDouble', 'FileWriteFloat', 'FileWriteInteger', 'FileWriteLong',
  'FileWriteString', 'FileWriteStruct', 'FileCopy', 'FileMove', 'FolderCreate', 'FolderDelete', 'FolderClean',
  // Object functions
  'ObjectCreate', 'ObjectName', 'ObjectDelete', 'ObjectsDeleteAll', 'ObjectFind', 'ObjectGetDouble',
  'ObjectGetFloat', 'ObjectGetInteger', 'ObjectGetString', 'ObjectMove', 'ObjectsTotal', 'ObjectSelect',
  'ObjectSetDouble', 'ObjectSetFloat', 'ObjectSetInteger', 'ObjectSetString', 'ObjectGetTimeByValue',
  'ObjectGetValueByTime', 'ObjectGetTime', 'ObjectShift', 'ObjectGetType', 'ObjectSetString',
  // Print and Alert
  'Print', 'Alert', 'Comment', 'MessageBox', 'PlaySound', 'SendMail', 'SendNotification',
  // Terminal
  'TerminalInfoDouble', 'TerminalInfoInteger', 'TerminalInfoString', 'TerminalCompany', 'TerminalName',
  'TerminalPath', 'UninitializeReason', 'IsStopped', 'OnStart', 'OnTick', 'OnTimer', 'OnTrade', 'OnTradeTransaction',
  'OnChartEvent', 'OnCalculate', 'OnTester', 'OnTesterInit', 'OnTesterPass', 'OnTesterDeInit', 'OnBookEvent',
  'OnChartEvent', 'OnInit', 'OnDeinit'
];

// MQL5 Constants
const mql5Constants = [
  // Order types
  'ORDER_TYPE_BUY', 'ORDER_TYPE_SELL', 'ORDER_TYPE_BUY_LIMIT', 'ORDER_TYPE_SELL_LIMIT',
  'ORDER_TYPE_BUY_STOP', 'ORDER_TYPE_SELL_STOP', 'ORDER_TYPE_BUY_STOP_LIMIT', 'ORDER_TYPE_SELL_STOP_LIMIT',
  // Order states
  'ORDER_STATE_STARTED', 'ORDER_STATE_PLACED', 'ORDER_STATE_CANCELED', 'ORDER_STATE_PARTIAL',
  'ORDER_STATE_FILLED', 'ORDER_STATE_REJECTED', 'ORDER_STATE_EXPIRED',
  // Trade actions
  'TRADE_ACTION_DEAL', 'TRADE_ACTION_PENDING', 'TRADE_ACTION_SLTP', 'TRADE_ACTION_MODIFY',
  'TRADE_ACTION_REMOVE', 'TRADE_ACTION_REQUEST',
  // Positions
  'POSITION_TYPE_BUY', 'POSITION_TYPE_SELL',
  // Timeframes
  'PERIOD_M1', 'PERIOD_M2', 'PERIOD_M3', 'PERIOD_M4', 'PERIOD_M5', 'PERIOD_M6', 'PERIOD_M10',
  'PERIOD_M12', 'PERIOD_M15', 'PERIOD_M20', 'PERIOD_M30', 'PERIOD_H1', 'PERIOD_H2', 'PERIOD_H3',
  'PERIOD_H4', 'PERIOD_H6', 'PERIOD_H8', 'PERIOD_H12', 'PERIOD_D1', 'PERIOD_W1', 'PERIOD_MN1',
  // Return codes
  'TRADE_RETCODE_DONE', 'TRADE_RETCODE_DONE_PARTIAL', 'TRADE_RETCODE_REQUOTE', 'TRADE_RETCODE_REJECT',
  'TRADE_RETCODE_CANCEL', 'TRADE_RETCODE_PLACED', 'TRADE_RETCODE_ERROR',
  // Colors
  'clrNONE', 'clrBlack', 'clrWhite', 'clrRed', 'clrGreen', 'clrBlue', 'clrYellow', 'clrOrange',
  'clrGray', 'clrSilver', 'clrAqua', 'clrMagenta', 'clrLime', 'clrMaroon', 'clrNavy', 'clrPurple',
  // Special values
  'EMPTY_VALUE', 'EMPTY', 'NULL', 'WHOLE_ARRAY', 'INVALID_HANDLE', 'DBL_MAX', 'DBL_MIN', 'INT_MAX', 'INT_MIN',
  // Chart properties
  'CHART_IS_OBJECT', 'CHART_BRING_TO_TOP', 'CHART_MOUSE_SCROLL', 'CHART_EVENT_MOUSE_WHEEL',
  'CHART_EVENT_MOUSE_MOVE', 'CHART_FIRST_VISIBLE_BAR', 'CHART_WIDTH_IN_BARS', 'CHART_WIDTH_IN_PIXELS',
  'CHART_HEIGHT_IN_PIXELS', 'CHART_COLOR_BACKGROUND', 'CHART_COLOR_FOREGROUND', 'CHART_COLOR_GRID',
  'CHART_COLOR_VOLUME', 'CHART_COLOR_CHART_UP', 'CHART_COLOR_CHART_DOWN', 'CHART_COLOR_CHART_LINE',
  'CHART_COLOR_CANDLE_BULL', 'CHART_COLOR_CANDLE_BEAR', 'CHART_COLOR_BID', 'CHART_COLOR_ASK', 'CHART_COLOR_LAST'
];

// MQL5 Operators
const mql5Operators = [
  '+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=',
  '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '>>>', '?:', '=',
  '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='
];

/**
 * Register MQL5 language with Monaco
 */
export function registerMQL5Language(monaco: typeof Monaco): void {
  // Register the language
  monaco.languages.register({ id: 'mql5' });

  // Set token provider for syntax highlighting
  monaco.languages.setMonarchTokensProvider('mql5', {
    keywords: mql5Keywords,
    builtins: mql5Builtins,
    constants: mql5Constants,
    operators: mql5Operators,

    symbols: /[=><!~?:&|+\-*\/\^%]+/,

    escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

    tokenizer: {
      root: [
        // Preprocessor directives
        [/#\w+/, 'keyword.directive'],
        [/#property/, 'keyword.directive'],
        [/#include/, 'keyword.directive'],
        [/#import/, 'keyword.directive'],
        [/#define/, 'keyword.directive'],
        [/#undef/, 'keyword.directive'],
        [/#ifdef/, 'keyword.directive'],
        [/#ifndef/, 'keyword.directive'],
        [/#endif/, 'keyword.directive'],
        [/#else/, 'keyword.directive'],

        // Comments
        ['\\/\\/.*$', 'comment'],
        ['\\/\\*', 'comment', '@comment'],

        // Strings
        [/"([^"\\]|\\.)*$/, 'string.invalid'],
        [/"/, 'string', '@string'],

        // Characters
        [/'([^'\\]|\\.)'/, 'string'],
        [/'/, 'string.invalid'],

        // Numbers
        [/\d[\d_]*\.[\d_]+([eE][\-+]?\d+)?[fFdD]?/, 'number.float'],
        [/0[xX][0-9a-fA-F_]+[lL]?/, 'number.hex'],
        [/0[bB][01_]+[lL]?/, 'number.binary'],
        [/\d[\d_]*[lL]?/, 'number'],

        // Identifiers and keywords
        [/[a-zA-Z_]\w*/, {
          cases: {
            '@keywords': 'keyword',
            '@builtins': 'type.identifier',
            '@constants': 'constant',
            '@default': 'identifier'
          }
        }],

        // Operators
        [/[{}()\[\]]/, '@brackets'],
        [/[<>](?!@symbols)/, '@brackets'],
        [/@symbols/, {
          cases: {
            '@operators': 'operator',
            '@default': ''
          }
        }],

        // Delimiter
        [/[;,.]/, 'delimiter'],
      ],

      comment: [
        [/[^\/*]+/, 'comment'],
        [/\/\*/, 'comment', '@push'],
        ['\\*/', 'comment', '@pop'],
        [/[\/*]/, 'comment']
      ],

      string: [
        [/[^\\"]+/, 'string'],
        [/@escapes/, 'string.escape'],
        [/\\./, 'string.escape.invalid'],
        [/"/, 'string', '@pop']
      ],
    }
  });

  // Set language configuration
  monaco.languages.setLanguageConfiguration('mql5', {
    comments: {
      lineComment: '//',
      blockComment: ['/*', '*/']
    },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')']
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" },
      { open: '/*', close: ' */', notIn: ['string'] }
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
      { open: "'", close: "'" }
    ],
    folding: {
      markers: {
        start: /^\s*#region\b/,
        end: /^\s*#endregion\b/
      }
    },
    indentationRules: {
      increaseIndentPattern: /^.*\{[^}\"']*$/,
      decreaseIndentPattern: /^(.*\*\/)?\s*\}.*$/
    }
  });
}

/**
 * Get MQL5 theme colors for syntax highlighting
 */
export function getMQL5ThemeColors(theme: 'dark' | 'light'): Monaco.editor.IStandaloneThemeData {
  if (theme === 'dark') {
    return {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'keyword', foreground: '569cd6', fontStyle: 'bold' },
        { token: 'keyword.directive', foreground: 'c586c0' },
        { token: 'type.identifier', foreground: 'dcdcaa' },
        { token: 'constant', foreground: '4fc1ff' },
        { token: 'comment', foreground: '6a9955', fontStyle: 'italic' },
        { token: 'string', foreground: 'ce9178' },
        { token: 'number', foreground: 'b5cea8' },
        { token: 'number.float', foreground: 'b5cea8' },
        { token: 'number.hex', foreground: '5bb498' },
        { token: 'operator', foreground: 'd4d4d4' },
        { token: 'identifier', foreground: '9cdcfe' },
        { token: 'delimiter', foreground: 'd4d4d4' },
      ],
      colors: {
        'editor.background': '#1e1e1e',
        'editor.foreground': '#d4d4d4',
        'editorLineNumber.foreground': '#858585',
        'editor.selectionBackground': '#264f78',
        'editor.lineHighlightBackground': '#2a2d2e'
      }
    };
  } else {
    return {
      base: 'vs',
      inherit: true,
      rules: [
        { token: 'keyword', foreground: '0000ff', fontStyle: 'bold' },
        { token: 'keyword.directive', foreground: '800080' },
        { token: 'type.identifier', foreground: '795e26' },
        { token: 'constant', foreground: '0070c1' },
        { token: 'comment', foreground: '008000', fontStyle: 'italic' },
        { token: 'string', foreground: 'a31515' },
        { token: 'number', foreground: '098658' },
        { token: 'operator', foreground: '000000' },
        { token: 'identifier', foreground: '001080' },
      ],
      colors: {
        'editor.background': '#ffffff',
        'editor.foreground': '#000000',
        'editorLineNumber.foreground': '#999999',
        'editor.selectionBackground': '#add6ff',
        'editor.lineHighlightBackground': '#f5f5f5'
      }
    };
  }
}

export { mql5Keywords, mql5Builtins, mql5Constants };
