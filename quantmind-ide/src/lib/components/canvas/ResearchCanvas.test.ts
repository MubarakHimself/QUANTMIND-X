import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'ResearchCanvas.svelte'), 'utf-8');
const srcNoComments = src
  .replace(/<!--[\s\S]*?-->/g, '')
  .replace(/\/\*[\s\S]*?\*\//g, '');

describe('ResearchCanvas.svelte — canonical research surface', () => {
  it('starts on logs now that books live in Shared Assets and Development', () => {
    expect(src).toContain("let activeTab = $state<ResearchTab>('logs')");
  });

  it('keeps only logs, personal, news, and dept tasks tabs', () => {
    expect(src).toContain("id: 'logs'");
    expect(src).toContain("id: 'personal'");
    expect(src).toContain("id: 'news'");
    expect(src).toContain("id: 'dept-tasks'");
    expect(src).not.toContain("id: 'books'");
    expect(src).not.toContain("label: 'Books'");
    expect(src).not.toContain("id: 'articles'");
    expect(src).not.toContain("label: 'Articles'");
  });

  it('loads logs on mount and does not eagerly load duplicate article/book surfaces', () => {
    expect(src).toContain("await loadTab('logs');");
    expect(src).not.toContain("await loadTab('articles');");
    expect(src).not.toContain("await loadTab('books');");
    expect(src).not.toContain("apiFetch<ArticleItem[]>('/knowledge/articles')");
    expect(src).not.toContain("apiFetch<BookItem[]>('/knowledge/books')");
  });

  it('renders the inline research department kanban on the dept tasks tab', () => {
    expect(src).toContain("{:else if activeTab === 'dept-tasks'}");
    expect(src).toContain('<DepartmentKanban department="research" />');
  });

  it('retains the data-dept marker on the root element', () => {
    expect(src).toContain('data-dept="research"');
  });

  it('does not keep article grouping helpers after moving articles to Shared Assets', () => {
    expect(srcNoComments).not.toContain('getArticleCategory');
    expect(srcNoComments).not.toContain('formatArticleCategory');
    expect(srcNoComments).not.toContain('articleGroups');
    expect(srcNoComments).not.toContain('article-groups');
  });
});
