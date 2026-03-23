# P0-4: SAST/DAST Security Scans — Implementation Plan

**Created:** 2026-03-22
**Status:** In Progress
**P0 Blocker:** #4 - Unknown security posture

---

## Phase 1: SAST (Static Application Security Testing)

### 1.1 Integrate SonarCloud Scanner into GitHub Actions
- **Tool:** SonarCloud (free for open source)
- **Action:** `SonarSource/sonarcloud-github-action@v2`
- **Configuration:**
  - Python ruleset (`python`)
  - TypeScript/Svelte ruleset (`javascript`, `typescript`)
  - Security rules enabled (`sonar.security.languages=python,javascript,typescript`)
- **Quality Gate:** 0 critical/high vulnerabilities
- **File:** `.github/workflows/sonarcloud.yml` (new)
- **Triggers:** Every PR and main branch push

### 1.2 Configure SonarCloud Project Properties
- **File:** `sonar-project.properties` (root)
- **Settings:**
  - Python source: `src/`
  - TypeScript/Svelte source: `quantmind-ide/src/`
  - Test paths: `tests/`, `quantmind-ide/src/**/*.test.ts`
  - Exclusion patterns for generated code

### 1.3 SonarCloud Token Management
- Store `SONAR_TOKEN` in GitHub Secrets
- Reference via `${{ secrets.SONAR_TOKEN }}`

---

## Phase 2: DAST (Dynamic Application Security Testing)

### 2.1 Integrate OWASP ZAP into GitHub Actions
- **Tool:** OWASP ZAP Baseline Scan
- **Action:** `owasp/zap-baseline@v2`
- **Configuration:**
  - Target URL: Configurable via env var (default: staging URL)
  - Scan mode: Baseline (passive scan)
  - Authenticated scan: Optional (requires test credentials)
- **File:** `.github/workflows/zap-dast.yml` (new)
- **Triggers:** Every PR and main branch push

### 2.2 ZAP Wrapper Script for Local Dev
- **File:** `scripts/zap_scan_wrapper.sh`
- **Purpose:** Local DAST scanning for developers
- **Usage:** `./scripts/zap_scan_wrapper.sh http://localhost:8000`

### 2.3 ZAP Configuration
- **File:** `.zap/config.xml`
- **Rules:** Exclude false positives, set alert thresholds

---

## Phase 3: Dependency Auditing

### 3.1 npm Audit (Frontend)
- **Add to existing workflow:** `multi-platform-build.yml`
- **Command:** `npm audit --audit-level=critical`
- **Fail condition:** Any critical vulnerabilities
- **Already partially configured in CI (via `npm ci`)**

### 3.2 pip-audit (Backend Python)
- **Tool:** `pip-audit` (Python vulnerability scanner)
- **Action:** `pypa/pip-audit` in GitHub Actions
- **File:** `.github/workflows/dependency-audit.yml` (new)
- **Triggers:** Every PR and main branch push
- **Fail condition:** Any critical vulnerabilities

### 3.3 npm Audit (if not already in CI)
- **Add to:** `multi-platform-build.yml` or separate workflow
- **Command:** `npm audit --audit-level=critical`

---

## Phase 4: Reporting & Remediation

### 4.1 Security Scan Reports Directory
- **Location:** `_bmad-output/test-artifacts/security-scan-reports/`
- **Contents:**
  - `sast-report-{date}.json` — SonarCloud scan results
  - `dast-report-{date}.json` — ZAP scan results
  - `dep-audit-report-{date}.json` — Dependency audit results
  - `vulnerability-tracker.md` — Issue tracking template

### 4.2 Risk Waiver Template
- **File:** `_bmad-output/test-artifacts/security-scan-reports/RISK_WAIVER_TEMPLATE.md`
- **Contents:**
  - Vulnerability ID
  - Description
  - Severity
  - Risk acceptance rationale
  - compensating controls
  - Approval signatures
  - Expiration date

### 4.3 GitHub Issues Integration
- Create GitHub Issues for critical/high vulnerabilities
- Link to scan reports
- Assign to appropriate team

---

## Implementation Checklist

- [ ] Create `.github/workflows/sonarcloud.yml` (SAST)
- [ ] Create `sonar-project.properties`
- [ ] Create `.github/workflows/zap-dast.yml` (DAST)
- [ ] Create `scripts/zap_scan_wrapper.sh`
- [ ] Create `.zap/config.xml`
- [ ] Create `.github/workflows/dependency-audit.yml`
- [ ] Add npm audit to `multi-platform-build.yml` (frontend)
- [ ] Add pip-audit step to CI
- [ ] Create `_bmad-output/test-artifacts/security-scan-reports/RISK_WAIVER_TEMPLATE.md`
- [ ] Create `_bmad-output/test-artifacts/security-scan-reports/vulnerability-tracker.md`
- [ ] Add SONAR_TOKEN to GitHub Secrets (requires manual setup)

---

## Acceptance Criteria

1. SAST scan (SonarCloud) — 0 critical/high vulnerabilities
2. DAST scan (OWASP ZAP) — 0 critical/high vulnerabilities
3. Dependency audit (npm audit / pip-audit) — 0 critical vulnerabilities
4. Report generated and filed at `_bmad-output/test-artifacts/security-scan-reports/`
5. Findings remediated or accepted with risk waiver

---

## Files to Create/Modify

### New Files:
1. `.github/workflows/sonarcloud.yml`
2. `.github/workflows/zap-dast.yml`
3. `.github/workflows/dependency-audit.yml`
4. `sonar-project.properties`
5. `scripts/zap_scan_wrapper.sh`
6. `.zap/config.xml`
7. `_bmad-output/test-artifacts/security-scan-reports/RISK_WAIVER_TEMPLATE.md`
8. `_bmad-output/test-artifacts/security-scan-reports/vulnerability-tracker.md`

### Files to Modify:
1. `.github/workflows/multi-platform-build.yml` — add npm audit

---

## Notes

- CodeQL is already configured in `.github/workflows/codeql.yml` but lacks deep SAST coverage for Python/TypeScript vulnerabilities
- SonarCloud provides superior SAST coverage including security hotspots
- ZAP DAST requires a running server — consider running against staging environment
- All secrets use GitHub Secrets — no hardcoded credentials
