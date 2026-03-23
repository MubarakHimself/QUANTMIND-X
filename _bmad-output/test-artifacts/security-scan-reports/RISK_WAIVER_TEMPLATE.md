# Security Risk Waiver Template

## Vulnerability Risk Acceptance Form

**Date:** YYYY-MM-DD
**Waiver ID:** WAIVER-YYYY-###
**Status:** PENDING | APPROVED | REJECTED | EXPIRED

---

## Section 1: Vulnerability Details

| Field | Value |
|-------|-------|
| **Vulnerability ID** | (e.g., CVE-XXXX-XXXX, SonarQube ID, ZAP Alert ID) |
| **Tool Detected By** | [ ] SonarCloud/SAST  [ ] ZAP/DAST  [ ] npm audit  [ ] pip-audit  [ ] Other |
| **Severity** | [ ] Critical  [ ] High  [ ] Medium  [ ] Low |
| **CVSS Score** | (if applicable) |
| **Affected Component** | (package name, file path, or API endpoint) |
| **Vulnerable Version(s)** | |
| **Fixed Version(s)** | (if known) |

---

## Section 2: Vulnerability Description

```
Paste or summarize the vulnerability description here.
Include:
- What the vulnerability is
- How it could be exploited
- Potential impact if exploited
```

---

## Section 3: Risk Assessment

### Likelihood of Exploitation
- [ ] Low - Requires specific conditions/credentials
- [ ] Medium - Exploitable under normal conditions
- [ ] High - Easily exploitable

### Impact if Exploited
- [ ] Low - Minimal data/system impact
- [ ] Medium - Significant data/system impact
- [ ] High - Critical data/system impact, regulatory implications

### Overall Risk Rating
- [ ] Low  [ ] Medium  [ ] High  [ ] Critical

---

## Section 4: Risk Acceptance Rationale

### Business Justification
```
Explain why this vulnerability cannot or should not be remediated:
- Compatibility issues
- Breaking changes
- Resource constraints
- Business critical dependency
- False positive
```

### Compensating Controls
```
List controls in place to mitigate the risk:
- Network segmentation
- WAF rules
- Monitoring/alerting
- Manual review processes
- Rate limiting
```

### Alternative Solutions Considered
```
List alternative approaches considered but rejected:
1.
2.
3.
```

---

## Section 5: Approval

### Requestor
| | |
|---|---|
| **Name** | |
| **Role** | |
| **Date** | |

### Security Review
| | |
|---|---|
| **Reviewer** | |
| **Role** | Security Team |
| **Date** | |
| **Recommendation** | [ ] Approve  [ ] Reject  [ ] Needs More Info |

### Final Approval
| | |
|---|---|
| **Approver** | |
| **Role** | (Security Lead / CTO / CISO) |
| **Date** | |
| **Signature** | |

---

## Section 6: Conditions & Expiration

### Waiver Conditions
- [ ] Vulnerability must be re-evaluated on or before expiration date
- [ ] Compensating controls must remain in place
- [ ] New instances of similar vulnerability require new waiver

### Expiration Date
**Waiver Valid Until:** YYYY-MM-DD

---

## Section 7: Tracking

| Date | Action | By |
|------|--------|-----|
| YYYY-MM-DD | Waiver created | |
| YYYY-MM-DD | Security review | |
| YYYY-MM-DD | Final approval | |
| YYYY-MM-DD | Re-evaluation | |
| YYYY-MM-DD | Waiver expired/renewed | |

---

## Section 8: Related Documents

- Scan Report: `_bmad-output/test-artifacts/security-scan-reports/`
- GitHub Issue: #
- Remediation Notes:

---

**Template Version:** 1.0
**Last Updated:** 2026-03-22
