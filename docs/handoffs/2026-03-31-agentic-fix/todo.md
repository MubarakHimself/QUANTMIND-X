# QuantMindX Agentic/UI TODO Tracker (Production, No Mock Data)

Last updated: 2026-04-05

## Active Correction Track (contract-first)

- [x] P0: Canonicalize active agent identities (`floor_manager`, department heads) and keep legacy aliases compatibility-only.
- [x] P0: Manifest-first chat-context normalization in chat endpoints (strip heavy payloads).
- [x] P1: Workspace Resource Contract backend (manifest/search/read) for natural resource discovery across canvases.
- [x] P1: Session contract normalization (`interactive_session`, `workflow_session`) at session creation/update boundaries.
- [x] P1: Standardized mutating tool return envelope with `ui_projection_event` (task/mail/workflow/resource updates).
- [x] P1: Replace monolithic department/floor-manager prompt defaults with slim base prompts plus dynamic skill/MCP/memory/compaction/session contracts.
- [x] P2: Frontend attachment/natural-search wiring to consume backend resource-contract endpoints.
- [x] P2: AgentPanel right-rail resize behavior with per-canvas width persistence and keyboard accessibility.
- [x] P2: AgentPanel visual step/status colors decoupled from hardcoded blue; now uses theme-correlated accent variables.
- [x] P2: AgentPanel internal scroll/overflow hardening for long attachment/session/message lists.
- [x] P2: Structured streaming status events parity across department and floor-manager chat paths.
- [x] P2: AgentPanel mail list/detail workspace now marks messages as read on open via live backend read-state endpoint.
- [x] P2: AgentPanel tool/thinking stream rows now render as compact collapsible events instead of permanently expanded broad lines.
- [ ] P2: Full Chrome navigation verification across all canvases for session isolation, attachments, live update projections, and mail send/receive.
  - `Research -> send_mail -> Portfolio MAIL inbox` is now verified live.
  - `Development -> send_mail -> Research MAIL inbox` is now re-verified live from an actual department-agent chat response, including target inbox detail rendering.
  - `Portfolio -> MAIL` detail workspace and session-history rendering were re-verified live on 2026-04-04.
  - `Shared Assets -> Strategies` is now verified live with canonical WF1 roots and JSON strategy-tree detail rendering.
  - browser-side verification now confirms `GET /api/video-ingest/jobs` returns durable WF1/quarantine context after backend restart.
  - `FlowForge` is now verified live after backend restart with durable WF1 manifest cards (`PENDING 5`, `RUNNING 2`) and without dead pending placeholder cards.
  - `Research -> Dept Tasks`, `Development -> Dept Tasks`, and `FlowForge -> Dept Tasks` are now verified live against canonical workflow state.
  - `Trading -> Dept Tasks` is now also verified live against canonical workflow state.
  - FlowForge now has real pause/resume/retry controls and department kanban decomposition.
  - Persisted/restarted workflow runs are now operator-actionable again after coordinator hydration; live pause was re-verified against the API after restart.
  - `FlowForge -> pause/resume` is now also verified live end to end:
    - `RUNNING -> PENDING_REVIEW`
    - linked `Development -> Dept Tasks` task `IN_PROGRESS -> BLOCKED`
    - `resume` returns both surfaces to active state
  - Cold-start task projection is now hardened:
    - `FlowForge -> Dept Tasks` now shows live `IN PROGRESS` workflow work after backend restart instead of downgrading everything to `TODO`.
    - `GET /api/tasks/research` now includes hydrated running workflow work instead of only manifest-pending rows.
  - `Research -> Attach canvas context` is now re-verified live after the frontend search-scope fix and no longer leaks `Development` resources like `Active EAs ACTIVE-TAB`.
 - Remaining Chrome verification work is broader cross-canvas live tool-event rendering during active agent responses plus any remaining attach-picker cleanup/curation.
  - `Live Trading` right-rail chat is now verified live end to end:
    - session creation works
    - streaming status rows render
    - session history updates after send
  - `Development` attached-chat runtime is now verified live end to end for the AlgoForge reference-book path:
    - `Development -> Reference Books -> MQL5 Reference` attaches correctly
    - one compact evolving stream row renders during generation
    - final assistant response lands in the panel
    - exact attached-resource grounding is now re-verified after backend restart:
      - prompt: `State the exact attachment label you received, and nothing else.`
      - response: `MQL5 Reference`
  - `Shared Assets -> Docs -> Books -> MQL5 Reference` detail view is now verified live
  - `Workshop -> Attach canvas context` is now hierarchy-based and live-verified after hard reload:
    - `Shared Assets -> Docs / Strategies`
    - now also supports nested Shared Assets drill-down:
      - `Shared Assets -> Docs -> Articles -> category -> resource`
  - Shared Assets attach-menu rehydration is now hardened:
    - department/workshop attach menus no longer get stuck showing only the last lazy-loaded Shared Assets category
    - `Development -> Shared Assets context` now shows both `Docs` and `Strategies`
    - `Development -> Shared Assets -> Docs` now shows `Books` and `Articles` again
  - Shared Assets strategy hierarchy is now verified live end to end:
    - `Strategies -> Scalping -> Single Videos -> me_at_the_zoo_smoke`
  - `Live Trading` is now corrected back to dashboard-only:
    - no right-rail agent panel
    - trading-agent interaction remains in the `Trading` canvas only
  - `Shared Assets -> Docs` now bootstraps through the lightweight counts contract and lazy category loading instead of trying to preload the whole corpus on mount.
  - still worth extending the same live verification path to more canvases beyond Shared Assets when time allows
  - Shared Assets strategy detail now exposes explicit WF1 stage groups in the live API/UI:
    - `Research`
    - `Development`
    - `Variants`
    - `Compilation`
    - `Reports`
  - still needs a live Chrome pass on the detail panel itself to confirm the new stage summary is readable/useful in the operator UI, not just correct by API/tests
- [x] P2: Align stream termination + provider fallback behavior with Claude Agent SDK event/env contract (`message_stop`, `ANTHROPIC_BASE_URL`).

## Deployment Readiness Track

- [ ] Remove/contain remaining runtime reliance on deprecated legacy agent config files (keep explicit compat routes only).
- [x] Ensure active node/session/model-config runtime stays provider-neutral and resolves defaults from configured runtime edges instead of hardcoded vendor fallbacks.
- [x] Remove synthetic provider/model fallback rows from active settings UI surfaces that should reflect only live runtime state.
- [ ] Validate zero mock-data fallbacks in active runtime UI paths (tasks/mail/assets/chat/session history).
  - Shared Assets `strategies` now resolves real canonical WF1 roots instead of internal manifest/request file leakage.
  - Development no longer duplicates FlowForge with a `Pipeline` tab; department execution stays in `Dept Tasks`.
  - Department task boards are now backed by canonical workflow state instead of mail-consumer-only empty boards.
  - Remaining no-mock sweep is concentrated in other legacy/non-primary surfaces and any MT5-adjacent fallback views.
- [ ] Rebuild/fix the active department tool inventory so the canonical tool set is real and loadable in production (current ToolRegistry still logs missing tool modules/classes).
  - `strategy_extraction` is now rebuilt and live in the canonical registry.
  - Remaining degraded inventory is mostly the deliberately gated/unconfigured surfaces such as `broker_tools`, plus any other legacy permissions that still point at non-production adapters.
- [ ] Audit remaining non-active MT5 simulated fallback helpers (`src/risk/integrations/mt5/*`) and either gate them behind explicit test-only config or remove them from production paths.
- [ ] Re-run backend/frontend focused regression tests and capture commands + outcomes in handoff progress.
- [ ] Re-run the next Chrome verification pass specifically for:
  - FlowForge pause/resume/retry card controls
  - broader department kanban card movement/writeback across more than FlowForge/Research after workflow state changes
  - cross-canvas workflow writeback after manual operator actions
  - note: pause/resume writeback is now confirmed for the live WF1 research path (`RUNNING -> PENDING_REVIEW/BLOCKED -> RUNNING/IN_PROGRESS`)
- [x] Fix Risk physics UI regression so Linux/non-MT5 hosts render an honest host-unavailable state instead of a broken failure banner.
- [x] Remove or replace Trading canvas placeholders (`TRADING JOURNAL — coming soon —`, `RISK PHYSICS — routed from Risk —`) with production-grade live surfaces.
- [x] Harden `/api/paper-trading/active` and related paper-trading API imports so non-MT5 hosts degrade safely instead of returning a 500.

## Immediate Next Slice (Now)

1. Finish the remaining cross-canvas Chrome verification beyond the now-passing mail send/receive path:
   - session isolation
   - attachment behavior
   - live tool-event rendering
   - writeback projection checks
   - `Research -> Logs` and `Development` reference-book attachments are now verified live
   - Research no longer owns a local `Books` surface; canonical books now live in `Shared Assets` and are consumed from `Development`
   - Development attach picker now no longer leaks placeholder WF1 roots from Shared Assets
   - Research attach picker now no longer leaks Development resources from backend search over-return
   - Development attach picker now stays curated on empty open instead of dumping the global backend file list
   - Shared Assets grid now has a real `/api/assets/counts` backend contract again and settles to live counts
   - Shared Assets bootstrap now uses the counts contract only; category payloads remain lazy-loaded
  - stream-event compaction is now fixed in the active agent panel:
    - repeated `status` / `tool` events update one evolving collapsible row instead of stacking multiple `started/streaming/completed` rows
  - session history contract is now cleaned up for department panels:
    - empty draft sessions can be excluded at the API level
    - badges now show `Current` / `N msgs` instead of misleading `ACTIVE`
    - new sessions auto-title from the first user message
    - historical placeholder `Chat ...` rows now derive usable preview titles from the first user message in the history list
   - department `New session` regression is now fixed live after the `exclude_empty` cleanup:
    - `Research -> New session` no longer collapses back to the empty-state panel
    - `Development -> New session` is re-verified live with successful `MQL5 Reference` attachment grounding
  - Workshop recent history is now cleaned up to the same non-empty-session standard:
    - stale empty `Chat ...` rows are gone
    - a fresh `New Chat` draft is still preserved locally before first send
   - still needs the broader parity pass for stream/tool-event/writeback behavior
   - Shared Assets is now corrected back to an asset-library-only canvas:
    - the stale `DEPT TASKS` tile / `dept-kanban` branch is removed
   - Risk is now corrected to the canonical tab set:
    - `Backtest` is removed
    - active tabs are `Physics`, `Compliance`, `Calendar`, `Dept Tasks`, `DPR`
   - `Shared Assets -> Docs -> Articles` article-sync operator surface is now verified live:
    - `Sync Now`
    - scraper selector including `Firecrawl`
    - `Set API Key`
    - category folders with real counts
   - Shared Assets strategy drill-down now has meaningful live path labels:
    - breadcrumb/title update to `Scalping`, `Single Videos`, etc. instead of repeating `Strategies`
   - session rename/delete is now verified live on a disposable Portfolio session:
    - row creation
    - rename save
    - delete removal
2. Finish the active video-ingest runtime modernization before any paid OpenRouter test:
  - [x] live-click the Workshop `video-ingest` tab and verify the new persisted recent-jobs surface renders from backend state, not local optimistic rows
  - [x] verify Shared Assets detail renders the blocking-error banner for quarantined WF1 roots (e.g. `me_at_the_zoo_smoke`)
  - [x] verify Workshop recent-job cards can jump directly into the canonical Shared Assets strategy root via `Open Strategy`
   - evolve the now-live model-budget-aware chunk sizing from static budgets to measured/runtime-aware budgets if needed
   - evaluate whether the next step after captions-first should be provider-native `video_url` or staying on `frames + captions` for v1
   - run a real single-video and playlist smoke pass against the canonical WF1 tree once provider/API budget is available
   - confirm end-to-end `video_ingest -> research` stage advancement through the canonical workflow surface, not just manifest writeback
   - confirm persisted `source/audio/`, `source/captions/`, `source/timelines/`, and `source/chunks/` all render and remain retrievable in Shared Assets after a real completed job
3. Explore and track workflow runtime readiness explicitly:
   - AlgoForge / video-ingest path
   - OpenRouter-backed video processing path
   - Kanban/workflow live projection
   - shared-assets artifact browsing / resource projection
   - operator workflow controls (`pause/resume/retry`) via live Chrome verification
4. Continue rebuilding the remaining active department tool inventory against the new manifest-first contract and Claude-style tool usage model.
5. Extend the new prompt contract into workflow/session compaction playbooks and long-running harness behavior.
6. Continue no-mock-data sweeps and deployment hardening.
   - Redis mail is now configuration-driven and no longer the default startup path.
   - Optional Finnhub/GitHub startup services are now gated behind real configuration.
7. Remove/contain remaining runtime reliance on deprecated legacy agent config files (compat-only surface).
8. Continue MT5 production hardening beyond this slice: verify UI surfaces that expose MT5 status/bridge setup and eliminate any remaining fake terminal/account states.
9. Continue provider-neutral cleanup in remaining legacy/non-active surfaces without disturbing the active department-based runtime.
10. Continue replacing old `copilot` / `analyst` / `quantcode` labels and config dependencies in active runtime surfaces with canonical department-based identities, leaving compat routes only where required.
11. Continue cross-canvas UI cleanup after the Agents/settings stabilization:
    the next live UX targets remain the right-rail chat/session surfaces, mail viewport use, and canvas tiling density.
12. Deepen workflow harness behavior after the new controls landed:
    - pause/resume semantics for long-running hidden workflow sessions
    - retry policy/backoff metadata surfaced in FlowForge/department kanban
    - explicit operator-visible step status for blocked vs paused vs approval-gated workflows
13. Continue FlowForge production cleanup after the header compaction:
    - keep the compact title-only header
    - consider whether the `Dept Tasks` tab should show a live count badge later without reintroducing a large summary tile
    - re-run the full Chrome operator path via button clicks now that persisted workflow hydration is live after restart
- Next WF1 slice:
  - bind real OpenRouter provider selection from Settings into a paid single-video smoke test once the user provides the API key for this pass
  - verify playlist ingest end-to-end against the canonical shared-assets artifact tree
  - continue WF1 state projection from source artifacts into Research and Development handoff surfaces
  - make the next Research/Development-facing WF1 surfaces visible, not just attachable, using canonical strategy roots instead of demo cards
  - decide whether to surface current video-ingest provider/model summary directly in more than FlowForge before the OpenRouter test pass
  - keep the new `blocking_error` / `blocking_error_detail` split and extend it to any other workflow/runtime surfaces that still expose raw provider stderr directly
  - extend the new operator-friendly error summarization beyond Workshop recent jobs to any remaining workflow/status cards that still show raw provider traces
  - clean up obvious dev/test strategy roots from the local runtime dataset before deployment packaging so Shared Assets does not ship with throwaway `test/probe/playlist_batch` artifacts, even though the live workflow read model now filters them out
  - once the repo-book bootstrap slice is verified, consider surfacing those same canonical developer-guide PDFs more directly in Shared Assets detail/navigation, not just through the Research/Development knowledge views
 - if needed later, extend the nested attach-browser model beyond Shared Assets docs if more canvases need deeper folder trees
  - if needed later, refine Shared Assets `Docs` curation so bootstrap PDFs can be surfaced more prominently than the long scraped-article list
  - still worth a later UX pass on Shared Assets `Docs` ordering if we want `Books` to remain visually dominant over the much larger article corpus
  - if needed later, extend the new menu-resource preload fallback beyond `shared-assets` to other canvases whose resource lists currently depend on prior client-side navigation/state hydration
  - later UX cleanup:
    - consider surfacing `MQL5 Book` / `MQL5 Reference` above the longer docs/article corpus inside `Shared Assets -> Docs`
    - if the settings workflow still feels dense after the new wider modal, consider a second pass on the Agents settings pane:
      - collapsible non-prompt sections
      - an even more dominant prompt editor mode
      - tighter MCP/skills summary blocks

- Remaining live issues observed in Chrome:
  - Shared Assets grid initial-count UX is fixed: tiles now show `Loading...` until live counts are ready.
  - WebSocket handshake path regression (`/ws/main` 403) is fixed; active runtime now connects on `/ws`.
  - Portfolio API path regressions are fixed:
    - `/api/brokers/accounts` no longer false-404s from route precedence.
    - Portfolio canvas now uses `/api/portfolio/brokers` and `/api/portfolio/routing-matrix` with live `200` responses.
  - Remaining portfolio test debt:
    - `src/lib/components/canvas/PortfolioCanvas.test.ts` is currently out-of-date against the modern PortfolioCanvas implementation and needs contract refresh before it can gate CI meaningfully.
  - Remaining Workshop test debt:
    - `src/lib/components/canvas/WorkshopCanvas.test.ts` is broadly stale against the modern Workshop runtime and still needs a focused contract refresh before it can be trusted as a real gate.
  - Remaining Shared Assets curation work:
    - `Research -> Articles` has now been removed so Shared Assets remains the canonical categorized article surface
    - consider surfacing the two bootstrap MQL/MT5 PDFs above the long article corpus in Shared Assets docs
    - refine the `Strategies` folder tree further so it reflects the intended family/variant/report structure more explicitly in Shared Assets detail/navigation
  - later, make deeper strategy-root detail more explicit for workflow artifacts:
      - source output
      - TRD/SDD artifacts
      - EA/variant folders
      - report folders
    - stage-group projection is now landed; the remaining work is deeper drill-down and better workflow-stage navigation inside the detail view

## Next deployment/setup checkpoint

- Before any Contabo/SSH work:
  - [ ] finish the next autonomous runtime slice first
  - [ ] clean repo/worktree state enough for a safe commit
  - [x] decide what to exclude from Git packaging for now:
    - scraped-article corpus
    - large bootstrap PDF docs
    - tracked payload audit captured:
      - `data/scraped_articles` is currently about `104M`
      - `mql5.pdf` is about `32M`
      - `mql5book.pdf` is about `14M`
    - `.gitignore` now reflects this source-code-only packaging rule
    - Git index now stages removals for those payloads while leaving local files intact
  - [ ] commit
  - [ ] push
- Then use the existing deployment docs for Contabo access:
  - [ ] re-open [Contabo_Deployment_Guide.md](/home/mubarkahimself/Desktop/QUANTMINDX/docs/Contabo_Deployment_Guide.md)
  - [ ] use the recorded local key path:
    - `~/.ssh/id_ed25519`
  - [ ] connect to:
    - `root@155.133.27.86`
  - [ ] remove old QuantMindX deployment state from the server before pulling fresh code
  - [ ] pull the pushed repo state
  - [ ] migrate shared assets / PDFs / scraped articles into server-managed storage instead of keeping them as long-term repo payload
