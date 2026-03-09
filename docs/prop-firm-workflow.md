# Prop Firm Workflow Documentation

## Overview
This document describes the complete prop firm integration workflow for QUANTMINDX.

## Two-Book Architecture

### Personal Book
- Trader's own capital accounts
- Standard risk parameters (5% max drawdown)
- Standard loss threshold: 5 consecutive losses

### Prop Firm Book
- Funded prop firm accounts (FundedNext, TrueForex, etc.)
- Tighter risk parameters (3% max drawdown)
- Tighter loss threshold: 3 consecutive losses

## Supported Prop Firms

| Firm | Max Drawdown | Daily Loss | Challenge Period |
|------|-------------|------------|-----------------|
| FundedNext | 4% | 4% | 30 days |
| TrueForex | 5% | 5% | 60 days |
| MyCryptoBuddy | 4% | 4% | 45 days |

## Prop Firm Tags

EAs can be tagged with prop firm metadata:
- `@fundednext` - FundedNext account
- `@challenge_active` - Active challenge
- `@funded` - Funded account

## New Tools

### 1. Gemini CLI Research Tool
Research using Gemini CLI for market analysis.

### 2. Prop Firm Research Tool
Analyzes prop firm rules and requirements.

### 3. Task List Tool with Hooks
Task management with pre/post hooks.

### 4. Shared Assets Tool
Share reports and assets across the system.

## UI Components

### LiveTradingView Book Filter
Filter accounts by Personal/Prop Firm with color-coded badges.

### PropFirmDashboard
Overview of active challenges, funded accounts, and performance metrics.

## API Endpoints

- `GET /api/propfirm/challenges` - Active challenges
- `GET /api/propfirm/accounts` - Funded accounts
- `POST /api/propfirm/emergency-kill` - Emergency kill all prop firm trades
