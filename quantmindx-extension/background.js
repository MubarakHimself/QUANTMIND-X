// Create context menu on install
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "saveToQuantMindX",
        title: "Save to QuantMindX",
        contexts: ["link", "page"]
    });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "saveToQuantMindX") {
        const url = info.linkUrl || info.pageUrl;
        const title = tab.title || "Untitled";

        saveArticle(url, title);
    }
});

// Save article to chrome.storage
function saveArticle(url, title) {
    chrome.storage.local.get(['articles'], (result) => {
        const articles = result.articles || [];

        // Check if URL already exists
        const exists = articles.some(article => article.url === url);

        if (!exists) {
            articles.push({
                url: url,
                title: title,
                timestamp: new Date().toISOString(),
                category: "uncategorized"
            });

            chrome.storage.local.set({ articles: articles }, () => {
                console.log('Article saved:', url);

                // Show notification
                chrome.notifications.create({
                    type: 'basic',
                    iconUrl: 'icon48.png',
                    title: 'QuantMindX',
                    message: 'Article saved successfully!',
                    priority: 1
                });
            });
        } else {
            chrome.notifications.create({
                type: 'basic',
                iconUrl: 'icon48.png',
                title: 'QuantMindX',
                message: 'Article already saved',
                priority: 1
            });
        }
    });
}
