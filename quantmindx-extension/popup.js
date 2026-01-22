// Load and display articles when popup opens
document.addEventListener('DOMContentLoaded', () => {
    loadArticles();

    document.getElementById('exportBtn').addEventListener('click', exportArticles);
    document.getElementById('clearBtn').addEventListener('click', clearArticles);
});

function loadArticles() {
    chrome.storage.local.get(['articles'], (result) => {
        const articles = result.articles || [];

        // Update stats
        document.getElementById('totalCount').textContent = articles.length;
        const uncategorized = articles.filter(a => a.category === 'uncategorized').length;
        document.getElementById('uncategorizedCount').textContent = uncategorized;

        // Display articles
        const articlesList = document.getElementById('articlesList');

        if (articles.length === 0) {
            articlesList.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📄</div>
          <div>No articles saved yet</div>
          <div style="font-size: 12px; margin-top: 10px;">
            Right-click on any MQL5 article and select "Save to QuantMindX"
          </div>
        </div>
      `;
            return;
        }

        articlesList.innerHTML = articles.map(article => `
      <div class="article-item">
        <div class="article-title" title="${article.title}">${article.title}</div>
        <div class="article-url" title="${article.url}">${article.url}</div>
        <div class="article-meta">
          Saved: ${new Date(article.timestamp).toLocaleDateString()} | 
          Category: ${article.category}
        </div>
      </div>
    `).join('');
    });
}

function exportArticles() {
    chrome.storage.local.get(['articles'], (result) => {
        const articles = result.articles || [];

        if (articles.length === 0) {
            alert('No articles to export');
            return;
        }

        // Create JSON blob
        const dataStr = JSON.stringify(articles, null, 2);
        const blob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);

        // Download file
        const timestamp = new Date().toISOString().split('T')[0];
        chrome.downloads.download({
            url: url,
            filename: `quantmindx_articles_${timestamp}.json`,
            saveAs: true
        });
    });
}

function clearArticles() {
    if (confirm('Are you sure you want to clear all saved articles? This cannot be undone.')) {
        chrome.storage.local.set({ articles: [] }, () => {
            loadArticles();
        });
    }
}
