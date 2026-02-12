<script lang="ts">
  import { ChevronLeft, ChevronRight } from 'lucide-svelte';
  
  export let currentPage = 1;
  export let totalPages = 1;
  export let pageSize = 20;
  export let totalItems = 0;
  export let onPageChange: ((page: number) => void) | undefined = undefined;
  
  $: showingStart = totalItems === 0 ? 0 : (currentPage - 1) * pageSize + 1;
  $: showingEnd = Math.min(currentPage * pageSize, totalItems);
  $: totalPages = Math.ceil(totalItems / pageSize) || 1;
  
  function goToPage(page: number) {
    if (page >= 1 && page <= totalPages && page !== currentPage) {
      currentPage = page;
      onPageChange?.(page);
    }
  }
  
  function goToFirst() {
    goToPage(1);
  }
  
  function goToLast() {
    goToPage(totalPages);
  }
  
  function goToPrev() {
    goToPage(currentPage - 1);
  }
  
  function goToNext() {
    goToPage(currentPage + 1);
  }
</script>

<nav class="pagination" aria-label="Pagination navigation">
  <div class="pagination-info" aria-live="polite">
    <span class="showing">{showingStart}-{showingEnd}</span>
    <span class="separator">of</span>
    <span class="total">{totalItems}</span>
  </div>
  
  <div class="pagination-controls" role="navigation" aria-label="Page navigation">
    <button 
      class="page-btn"
      on:click={goToFirst}
      disabled={currentPage === 1}
      aria-label="Go to first page"
      title="First page"
    >
      <ChevronLeft size={14} />
      <ChevronLeft size={14} class="double" />
    </button>
    
    <button 
      class="page-btn"
      on:click={goToPrev}
      disabled={currentPage === 1}
      aria-label="Go to previous page"
      title="Previous page"
    >
      <ChevronLeft size={14} />
    </button>
    
    <div class="page-numbers" aria-label={`Page ${currentPage} of ${totalPages}`}>
      <span class="current-page">{currentPage}</span>
      <span class="page-separator">/</span>
      <span class="total-pages">{totalPages}</span>
    </div>
    
    <button 
      class="page-btn"
      on:click={goToNext}
      disabled={currentPage === totalPages}
      aria-label="Go to next page"
      title="Next page"
    >
      <ChevronRight size={14} />
    </button>
    
    <button 
      class="page-btn"
      on:click={goToLast}
      disabled={currentPage === totalPages}
      aria-label="Go to last page"
      title="Last page"
    >
      <ChevronRight size={14} />
      <ChevronRight size={14} class="double" />
    </button>
  </div>
  
  <div class="page-size-selector">
    <label for="page-size" class="sr-only">Items per page</label>
    <select 
      id="page-size"
      bind:value={pageSize}
      aria-label="Items per page"
    >
      <option value={10}>10</option>
      <option value={20}>20</option>
      <option value={50}>50</option>
      <option value={100}>100</option>
    </select>
  </div>
</nav>

<style>
  .pagination {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-top: 1px solid var(--border-subtle);
    font-size: 12px;
  }
  
  .pagination-info {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text-muted);
  }
  
  .showing, .total {
    color: var(--text-primary);
  }
  
  .pagination-controls {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .page-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  
  .page-btn:hover:not(:disabled) {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border-color: var(--border-color);
  }
  
  .page-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .page-btn .double {
    margin-left: -8px;
  }
  
  .page-numbers {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 0 8px;
    color: var(--text-secondary);
  }
  
  .current-page {
    color: var(--text-primary);
    font-weight: 600;
  }
  
  .page-size-selector {
    display: flex;
    align-items: center;
  }
  
  .page-size-selector select {
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }
  
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
