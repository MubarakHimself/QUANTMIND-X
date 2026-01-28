# Technical Requirements Document (TRD): QuantCode Agent v1

## 1. Objective
The **QuantCode Agent** is an autonomous "Quantitative Engineering Factory." Its mission is to implement production-grade MQL5 Expert Advisors and Python Router logic using a high-precision, iterative loop that leverages real-time compiler feedback and a reusable component library.

---

## 2. Core Operational Modes (@Mods)

1.  **@General:** Standard trading logic focused on standalone retail strategies.
2.  **@PropFirm:** Specialized logic with strict equity-clamp protection, trailing drawdown awareness, and adherence to specific prop firm T&Cs (ingested from the Analyst).

---

## 3. The "Meta-Compiler" Feedback Loop

QuantCode MUST verify all MQL5 code before completion:
1.  **Node: Implementation:** Writes `.mq5`/`.mqh` files.
2.  **Node: Syntax Guard (LSP-Sim):** An LLM-based node checks for common MQL5 errors (missing `OnInit` returns, uninitialized handles).
3.  **Node: Meta-Compiler Link:**
    *   Invokes `metaeditor64.exe /compile:file.mq5 /log:file.log`.
    *   Parses the `.log` for errors and warnings.
4.  **Loop: Healing:** If errors exist, QuantCode re-reads the code + log and iteratively fixes it until a "Success" status is achieved.

---

## 4. Reusable Asset Hub & Library Management

To handle high volumes (18+ bots), QuantCode manages a centralized library:
*   **Directory:** `MQL5/Include/QuantMindX/`
*   **Components:** Reusable classes for Sockets, Risk Management, and Indicator Wrappers.
*   **Private Knowledge Base:** QuantCode maintains its own `quantcode_kb`, indexing all successful helper functions it creates for future reuse.

---

## 5. Multi-Bot Orchestration (The Factory)

*   **Queuing System:** Supports a list of 18+ TRDs.
*   **Manager-Worker Pattern:** The Lead QuantCode agent can spawn "Sub-agent Sessions" (parallel LangGraph threads) to implement multiple bots simultaneously.
*   **Bot Lifecycle:**
    *   Initial code pushed to `/workspace/development`.
    *   On user promotion, registered in the Python **Strategy Router**.

---

## 6. Technical Integration

### Global Variables & Components
*   **Variables:** Supports a shared configuration schema (JSON/YML) for bot-specific parameters.
*   **Inbuilt Indicators:** Prioritizes native MQL5 `iIndicator` handles for performance; falls back to Python custom indicators via the **Socket Bridge** for complexity.

### Cloud Compatibility
*   **Agnostic Design:** All CLI tools and file paths must support **Wine** execution paths for future migration to Linux-based cloud environments.
