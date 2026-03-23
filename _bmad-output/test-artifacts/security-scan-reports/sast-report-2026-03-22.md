# SAST/DAST Security Scan Report

**Scan Date:** 2026-03-22
**Report ID:** SECURITY-BASELINE-2026-03-22
**Status:** INITIAL BASELINE

---

## Executive Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| SAST (SonarCloud) | 0 | 0 | 0 | 0 |
| DAST (ZAP) | N/A | N/A | N/A | N/A |
| npm audit | 0 | 1 | 1 | 0 |
| pip-audit | 0 | 4 | 0 | 0 |
| **TOTAL** | **0** | **5** | **1** | **0** |

**Acceptance Criteria:** 0 Critical/High vulnerabilities
**Current Status:** 5 High vulnerabilities require remediation

---

## Phase 1: SAST (Static Application Security Testing)

### SonarCloud SAST
- **Status:** Not yet run (requires SONAR_TOKEN in GitHub Secrets)
- **Configuration:** `.github/workflows/sonarcloud.yml`
- **Quality Gate:** 0 Critical/High vulnerabilities

### CodeQL (Existing)
- **Status:** Already configured in `.github/workflows/codeql.yml`
- **Languages:** Python, JavaScript/TypeScript
- **Schedule:** Weekly + on PR/push

---

## Phase 2: DAST (Dynamic Application Security Testing)

### OWASP ZAP
- **Status:** Workflow configured but not yet run
- **Configuration:** `.github/workflows/zap-dast.yml`
- **Requirements:** Running server target URL
- **Note:** Requires `ZAP_TARGET_URL` variable set (defaults to localhost:8000)

---

## Phase 3: Dependency Auditing

### Frontend (npm audit)

**Run Date:** 2026-03-22

| Package | Severity | Vulnerability | Fix Available |
|---------|----------|---------------|---------------|
| rollup (4.0.0-4.58.0) | HIGH | Arbitrary File Write via Path Traversal | Yes |
| @sveltejs/kit (2.49.0-2.53.2) | MODERATE | CPU/Memory exhaustion | Yes |

**Recommendation:** Run `npm audit fix` to remediate

### Backend (pip-audit)

**Run Date:** 2026-03-22

| Package | Severity | Vulnerability | Fix Available |
|---------|----------|---------------|---------------|
| langgraph (1.0.9) | HIGH | CVE-2026-28277 - Arbitrary code execution via msgpack | Yes (1.0.10) |
| pip | HIGH | CVE-2025-8869 - tar extraction path traversal | Yes (25.3) |
| pip | HIGH | CVE-2026-1703 - wheel extraction path traversal | Yes (26.0) |
| yt-dlp | HIGH | CVE-2026-26331 - Command injection via --netrc-cmd | Yes (2026.2.21) |

**Note:** pip vulnerability requires Python version with PEP 706 support for full mitigation.

---

## Vulnerability Details

### 1. HIGH: Rollup - Arbitrary File Write via Path Traversal
- **Package:** rollup
- **Affected Versions:** 4.0.0 - 4.58.0
- **Fix:** Update to latest version via `npm audit fix`
- **Impact:** Arbitrary file write during build process
- **Remediation:** `npm audit fix`

### 2. MODERATE: SvelteKit - CPU/Memory Exhaustion
- **Package:** @sveltejs/kit
- **Affected Versions:** 2.49.0 - 2.53.2
- **Fix:** Update to latest version via `npm audit fix`
- **Impact:** Denial of service via malicious form data
- **Remediation:** `npm audit fix`

### 3. HIGH: LangGraph - Arbitrary Code Execution (CVE-2026-28277)
- **Package:** langgraph
- **Affected Version:** 1.0.9
- **Fix Version:** 1.0.10
- **Description:** Unsafe msgpack deserialization in checkpoint loaders
- **Impact:** Arbitrary code execution if attacker can modify checkpoint data
- **Prerequisite:** Attacker needs write access to checkpoint store
- **Remediation:** `pip install langgraph==1.0.10` or set `LANGGRAPH_STRICT_MSGPACK=true`

### 4. HIGH: pip - Tar Extraction Path Traversal (CVE-2025-8869)
- **Package:** pip
- **Affected Version:** < 25.3 (in fallback mode)
- **Fix:** Upgrade pip or use Python with PEP 706 (>=3.9.17, >=3.10.12, >=3.11.4, >=3.12)
- **Impact:** File extraction outside target directory
- **Remediation:** `pip install --upgrade pip`

### 5. HIGH: pip - Wheel Extraction Path Traversal (CVE-2026-1703)
- **Package:** pip
- **Affected Version:** < 26.0
- **Fix:** Upgrade pip
- **Impact:** Malicious wheel files extracted outside installation directory
- **Remediation:** `pip install --upgrade pip`

### 6. HIGH: yt-dlp - Command Injection (CVE-2026-26331)
- **Package:** yt-dlp
- **Affected Version:** < 2026.2.21
- **Fix Version:** 2026.2.21
- **Description:** Arbitrary command injection via `--netrc-cmd` option
- **Impact:** Only exploitable if `--netrc-cmd` is used
- **Remediation:** `pip install --upgrade yt-dlp`

---

## Recommended Actions

### Immediate (P0)
1. Run `npm audit fix` in `quantmind-ide/` to fix frontend vulnerabilities
2. Run `pip install --upgrade pip langgraph==1.0.10 yt-dlp` to fix backend vulnerabilities
3. Set `LANGGRAPH_STRICT_MSGPACK=true` environment variable for defense-in-depth

### Short-term (P1)
1. Configure SonarCloud token in GitHub Secrets
2. Run full SAST/DAST scans
3. File risk waivers for accepted findings

### Long-term (P2)
1. Set up automated weekly vulnerability scanning
2. Implement dependency update automation (Dependabot)
3. Create security review process for new dependencies

---

## Next Scan Date

Scheduled: After remediation of above vulnerabilities

---

**Report Generated:** 2026-03-22
**Next Review:** Upon remediation completion
