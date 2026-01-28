# Spec Requirements: Risk Management System

## Initial Description
A comprehensive risk management system for QuantMindX that integrates econophysics-based market analysis with position sizing logic. The system will include physics-based market regime detection (Lyapunov chaos analysis, Ising model phase transition analysis), Monte Carlo validation, and a risk governor for calculating optimal position sizes based on Kelly criterion and stop loss parameters.

## Requirements Discussion

### First Round Questions

**Q1:** I assume we need to create the directory structure src/risk/ with physics/ and sizing/ subdirectories, plus a main governor.py file. Is that correct?
**Answer:** Yes, src/risk/ does not exist and must be created. Proposed structure: src/risk/physics/ (for Lyapunov, Ising, RMT), src/risk/sizing/ (for Kelly logic), src/risk/governor.py (Main entry point).

**Q2:** I'm thinking we should refactor the existing physics modules rather than just copying them. Should we convert Ising Pipeline.py to class IsingRegimeSensor and Lyapunov Pipeline.py to class ChaosSensor?
**Answer:** Refactor and Integrate. Do NOT just copy the script files. Convert: Ising Pipeline.py → class IsingRegimeSensor, Lyapunov Pipeline.py → class ChaosSensor. This ensures they are importable Python modules, not standalone scripts.

**Q3:** For the Monte Carlo implementation, should we adapt the mathematical core from quant-traderr-lab/Monte Carlo/Monte Carlo Pipeline.py but wrap it in a clean class MonteCarloValidator?
**Answer:** Adapt the Logic. Use the mathematical core from quant-traderr-lab/Monte Carlo/Monte Carlo Pipeline.py, but wrap it in a clean class MonteCarloValidator. The existing logic (GBM simulation, confident intervals) is correct; just improve the code quality.

**Q4:** Should we include the RiskGovernor integration for lot calculation that takes risk_percentage (from Kelly) + stop_loss_pips (from Strategy) and outputs lots using the formula: Lots = (Balance * Risk%) / (SL_Pips * Pip_Value)?
**Answer:** Include it. The system must be end-to-end executable. The RiskGovernor should take the risk_percentage (from Kelly) + stop_loss_pips (from Strategy) and output lots. Formula: Lots = (Balance * Risk%) / (SL_Pips * Pip_Value).

**Q5:** Should we use a constants/config class approach instead of hardcoding values? Create src/risk/config.py with default thresholds?
**Answer:** Use Constants/Config Class. No UI needed yet. Create src/risk/config.py with default thresholds (e.g., LYAPUNOV_CHAOS_THRESHOLD = 0.1). Do not hardcode magic numbers deep in the logic; pull them to the config file.

### Existing Code to Reference

**Similar Features Identified:**
- Feature: Ising Model - Path: `/home/mubarkahimself/Desktop/QUANTMINDX/quant-traderr-lab/Ising Model/Ising Pipeline.py`
- Feature: Lyapunov Exponent - Path: `/home/mubarkahimself/Desktop/QUANTMINDX/quant-traderr-lab/Lyapunov Exponent/Lyapunov Pipeline.py`
- Feature: Monte Carlo - Path: `/home/mubarkahimself/Desktop/QUANTMINDX/quant-traderr-lab/Monte Carlo/Monte Carlo Pipeline.py`
- Components to potentially reuse: Configuration pattern (CONFIG dictionary), logging utilities, mathematical core logic
- Backend logic to reference: MT5 AccountManager in `mcp-metatrader5-server/src/mcp_mt5/`

### Follow-up Questions

**Follow-up 1:** Integration with Existing System: The RiskGovernor needs to integrate with the MT5 AccountManager. Should I assume the system will have access to the current account balance through the MCP server's AccountManager, or do we need to implement a separate balance query mechanism?
**Answer:** [Pending user response]

**Follow-up 2:** Output Format: The RiskGovernor should output calculated lot sizes. Should I assume the output format should be a simple numerical value (float) that can be directly used by trading logic, or do we need a more structured output format with additional metadata?
**Answer:** [Pending user response]

**Follow-up 3:** Error Handling: The existing physics modules have basic error handling. Should we maintain the same level of error handling (warnings and basic logging) or implement more robust exception handling for the risk management system?
**Answer:** [Pending user response]

**Follow-up 4:** Testing Approach: Since this is a backend logic engine, should I assume we'll need unit tests for each component (physics sensors, Monte Carlo validator, risk governor) and integration tests for the end-to-end flow?
**Answer:** [Pending user response]

**Follow-up 5:** Configuration Defaults: You mentioned using a config file with default thresholds. Should I include the specific threshold values from the original scripts (e.g., `CONFIG['TARGET_SENTIMENT'] = 0.75` from Ising model) as the defaults, or should we use more conservative values?
**Answer:** [Pending user response]

## Visual Assets

### Files Provided:
No visual assets provided.

### Visual Insights:
No visual assets analyzed.

## Requirements Summary

### Functional Requirements
- Physics-based market regime detection using Ising model and Lyapunov exponent analysis
- Monte Carlo simulation for portfolio validation and risk assessment
- Risk governor for calculating optimal position sizes based on Kelly criterion
- Configuration system with default thresholds and constants
- Integration with MT5 AccountManager for balance queries
- End-to-end executable risk management pipeline

### Reusability Opportunities
- Configuration pattern from existing scripts (CONFIG dictionary)
- Mathematical core logic from Ising, Lyapunov, and Monte Carlo modules
- Logging utilities and error handling patterns
- Vectorized operations from Monte Carlo engine
- Account balance query from MT5 AccountManager

### Scope Boundaries

**In Scope:**
- Creation of src/risk/ directory structure
- Refactoring of physics modules into class-based components
- Implementation of Monte Carlo validator
- Development of risk governor with lot calculation logic
- Configuration management system
- Integration with existing QuantMindX system

**Out of Scope:**
- UI development for risk management interface
- Real-time market data streaming
- Advanced visualization components
- Machine learning model training
- Database persistence for risk metrics
- Advanced analytics dashboard

### Technical Considerations
- Integration points: MT5 AccountManager for balance queries
- Technology constraints: Python 3.x, numpy, pandas, scikit-learn dependencies
- Code patterns to follow: Class-based architecture, CONFIG dictionary pattern, vectorized operations
- Performance considerations: Optimized lattice calculations for Ising model
- Error handling: Maintain consistent logging pattern from existing modules