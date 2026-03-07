<script lang="ts">
  import { Database } from "lucide-svelte";
</script>

<div class="database-page">
  <h2><Database size={20} /> Database Visualization</h2>
  <div class="db-stats">
    <div class="db-stat">
      <span class="label">Total Records</span><span class="value"
        >12,450</span
      >
    </div>
    <div class="db-stat">
      <span class="label">Strategies</span><span class="value">8</span>
    </div>
    <div class="db-stat">
      <span class="label">Trades</span><span class="value">1,234</span>
    </div>
    <div class="db-stat">
      <span class="label">Last Sync</span><span class="value">2 min ago</span>
    </div>
  </div>
  <div class="db-tables">
    <div class="table-card">
      <h4>strategies</h4>
      <table>
        <thead
          ><tr
            ><th>id</th><th>name</th><th>status</th><th>created</th
            ></tr
          ></thead
        >
        <tbody>
          <tr
            ><td>1</td><td>ICT Scalper v2</td><td>primal</td><td
              >2025-01-15</td
            ></tr
          >
          <tr
            ><td>2</td><td>SMC Reversal</td><td>ready</td><td
              >2025-02-01</td
            ></tr
          >
        </tbody>
      </table>
    </div>
    <div class="table-card">
      <h4>trades</h4>
      <table>
        <thead
          ><tr><th>id</th><th>symbol</th><th>type</th><th>pnl</th></tr
        ></thead
      >
        <tbody>
          <tr
            ><td>1024</td><td>EURUSD</td><td>BUY</td><td
              class="positive">+45.20</td
            ></tr
          >
          <tr
            ><td>1023</td><td>GBPUSD</td><td>SELL</td><td
              class="negative">-12.50</td
            ></tr
          >
        </tbody>
      </table>
    </div>
  </div>
</div>

<style>
  .database-page {
    padding: 20px;
  }

  .database-page h2 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 20px 0;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .db-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
  }

  .db-stat {
    display: flex;
    flex-direction: column;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .db-stat .label {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }

  .db-stat .value {
    font-size: 24px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .db-tables {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
  }

  .table-card {
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .table-card h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .table-card table {
    width: 100%;
    border-collapse: collapse;
  }

  .table-card th,
  .table-card td {
    padding: 8px 12px;
    text-align: left;
    font-size: 12px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .table-card th {
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-tertiary);
  }

  .table-card td {
    color: var(--text-primary);
  }

  .table-card td.positive {
    color: #10b981;
  }

  .table-card td.negative {
    color: #ef4444;
  }
</style>
