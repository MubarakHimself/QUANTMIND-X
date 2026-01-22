# QuantMindX Article Collector - Chrome Extension

A Chrome/Brave browser extension for collecting MQL5 articles for the QuantMindX knowledge base.

## Features

- **Right-click to save**: Right-click any link or page → "Save to QuantMindX"
- **Article management**: View all saved articles in the extension popup
- **Export to JSON**: Export collected articles for processing
- **Duplicate detection**: Won't save the same URL twice
- **Article count**: See how many articles you've collected

## Installation Instructions (Brave Browser)

1. **Open the Extensions page**:
   - Click the menu (☰) in the top-right
   - Go to: **More tools** → **Extensions**
   - Or navigate to: `brave://extensions/`

2. **Enable Developer Mode**:
   - Toggle the **Developer mode** switch in the top-right corner

3. **Load the extension**:
   - Click **Load unpacked**
   - Navigate to: `/home/mubarkahimself/Desktop/QUANTMINDX/quantmindx-extension`
   - Click **Select Folder**

4. **Pin the extension** (optional but recommended):
   - Click the puzzle icon (🧩) in the toolbar
   - Find "QuantMindX Article Collector"
   - Click the pin icon to keep it visible

## How to Use

### Saving Articles

**Method 1: Right-click on a link**
1. Navigate to an MQL5 article page
2. Find a link to another article
3. Right-click the link
4. Select **"Save to QuantMindX"**
5. You'll see a notification confirming the save

**Method 2: Right-click on the page**
1. Navigate to an MQL5 article page
2. Right-click anywhere on the page
3. Select **"Save to QuantMindX"**
4. The current page will be saved

### Viewing Saved Articles

1. Click the extension icon in your toolbar
2. See stats: Total saved & Uncategorized count
3. Scroll through the list of saved articles

### Exporting Articles

1. Click the extension icon
2. Click **"Export JSON"**
3. Choose where to save the file
4. The file will be named: `quantmindx_articles_YYYY-MM-DD.json`

### Clearing All Articles

1. Click the extension icon
2. Click **"Clear All"**
3. Confirm the action (this cannot be undone!)

## Exported JSON Format

```json
[
  {
    "url": "https://www.mql5.com/en/articles/12345",
    "title": "Article Title Here",
    "timestamp": "2026-01-22T13:45:00.000Z",
    "category": "uncategorized"
  }
]
```

## Next Steps

After exporting the JSON:
1. The AI will review and categorize the articles
2. Articles will be scraped and converted to Markdown
3. Content will be stored in the QuantMindX knowledge base

## Troubleshooting

**Extension doesn't appear after installation:**
- Make sure Developer Mode is enabled
- Try reloading the extension (click the refresh icon)

**Right-click menu doesn't show "Save to QuantMindX":**
- Reload the extension
- Right-click on an actual link, not just anywhere

**Export doesn't work:**
- Make sure you have articles saved
- Check that Brave has permission to download files

## Files

- `manifest.json` - Extension configuration
- `background.js` - Context menu and storage logic
- `popup.html` - Extension popup UI
- `popup.js` - Popup interaction logic
- `icon*.png` - Extension icons
