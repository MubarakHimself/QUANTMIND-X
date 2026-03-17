<script>
  // Demo component - simplified for Svelte 5 migration
  // The @skeletonlabs/skeleton-svelte package was not installed
  // Using basic HTML/Tailwind instead
</script>

<div class="p-8 max-w-4xl mx-auto">
  <h1 class="text-2xl font-bold mb-6 text-white">Skeleton UI Demo</h1>

  <div class="space-y-8">
    <!-- Loading Placeholders with Tailwind -->
    <section class="bg-surface-100 p-6 rounded-xl">
      <h2 class="text-lg font-semibold mb-4">Loading Placeholders</h2>

      <div class="space-y-4">
        <!-- Text placeholder -->
        <div class="space-y-2">
          <div class="h-4 w-3/4 bg-surface-400 animate-pulse rounded"></div>
          <div class="h-4 w-1/2 bg-surface-400 animate-pulse rounded"></div>
          <div class="h-4 w-5/6 bg-surface-400 animate-pulse rounded"></div>
        </div>

        <!-- Avatar placeholder -->
        <div class="flex items-center gap-4">
          <div class="w-12 h-12 bg-surface-400 animate-pulse rounded-full"></div>
          <div class="flex-1 space-y-2">
            <div class="h-4 w-1/3 bg-surface-400 animate-pulse rounded"></div>
            <div class="h-3 w-1/4 bg-surface-400 animate-pulse rounded"></div>
          </div>
        </div>

        <!-- Card placeholder -->
        <div class="border border-surface-300 rounded-lg p-4 space-y-3">
          <div class="h-6 w-2/3 bg-surface-400 animate-pulse rounded"></div>
          <div class="h-4 w-full bg-surface-400 animate-pulse rounded"></div>
          <div class="h-4 w-4/5 bg-surface-400 animate-pulse rounded"></div>
          <div class="flex gap-2 mt-4">
            <div class="h-8 w-20 bg-surface-400 animate-pulse rounded"></div>
            <div class="h-8 w-20 bg-surface-400 animate-pulse rounded"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Progress Component - using native HTML -->
    <section class="bg-surface-100 p-6 rounded-xl">
      <h2 class="text-lg font-semibold mb-4">Progress Component</h2>
      <div class="space-y-4">
        <div>
          <p class="text-sm mb-2">Loading data...</p>
          <progress value="65" max="100" class="h-2 w-full"></progress>
        </div>
        <div>
          <p class="text-sm mb-2">Processing...</p>
          <progress value="0" max="100" class="h-2 w-full indeterminate"></progress>
        </div>
      </div>
    </section>

    <!-- Theme Colors -->
    <section class="bg-surface-100 p-6 rounded-xl">
      <h2 class="text-lg font-semibold mb-4">Theme Colors (Iceberg)</h2>
      <div class="flex flex-wrap gap-4">
        <div class="w-16 h-16 bg-primary-500 rounded-lg flex items-center justify-center text-xs">Primary</div>
        <div class="w-16 h-16 bg-secondary-500 rounded-lg flex items-center justify-center text-xs">Secondary</div>
        <div class="w-16 h-16 bg-success-500 rounded-lg flex items-center justify-center text-xs">Success</div>
        <div class="w-16 h-16 bg-warning-500 rounded-lg flex items-center justify-center text-xs">Warning</div>
        <div class="w-16 h-16 bg-error-500 rounded-lg flex items-center justify-center text-xs">Error</div>
      </div>
    </section>

    <!-- Utility Buttons -->
    <section class="bg-surface-100 p-6 rounded-xl">
      <h2 class="text-lg font-semibold mb-4">Button Styles</h2>
      <div class="flex flex-wrap gap-4">
        <button class="px-4 py-2 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 transition-colors">
          Filled
        </button>
        <button class="px-4 py-2 border border-blue-500 text-blue-500 rounded-lg font-medium hover:bg-blue-500/10 transition-colors">
          Outline
        </button>
        <button class="px-4 py-2 text-blue-500 rounded-lg font-medium hover:bg-blue-500/10 transition-colors">
          Ghost
        </button>
      </div>
    </section>
  </div>
</div>
