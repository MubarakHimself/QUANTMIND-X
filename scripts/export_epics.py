#!/usr/bin/env python3
"""Export all Traycer.AI Epic specs and tickets to docs/epic-export/"""

from pathlib import Path
from textwrap import dedent

BASE_DIR = Path("docs/epic-export")

EPICS = {
    "epic-af37afc7": {
        "folder": "epic-af37afc7-deployment",
        "name": "Deployment Pipeline & Infrastructure",
        "specs": [
            ("01-docker-multi-stage-builds.md", "Docker Multi-Stage Builds", "spec-af37-01"),
            ("02-ci-cd-pipeline.md", "CI/CD Pipeline Configuration", "spec-af37-02"),
            ("03-kubernetes-deployment.md", "Kubernetes Deployment Manifests", "spec-af37-03"),
            ("04-environment-configuration.md", "Environment Configuration Management", "spec-af37-04"),
            ("05-secrets-management.md", "Secrets Management & Vault Integration", "spec-af37-05"),
            ("06-monitoring-setup.md", "Monitoring & Observability Setup", "spec-af37-06"),
            ("07-logging-pipeline.md", "Centralized Logging Pipeline", "spec-af37-07"),
            ("08-auto-scaling.md", "Auto-Scaling Configuration", "spec-af37-08"),
            ("09-disaster-recovery.md", "Disaster Recovery & Backup", "spec-af37-09"),
            ("10-security-hardening.md", "Security Hardening Guidelines", "spec-af37-10"),
            ("11-network-policies.md", "Network Policies & Service Mesh", "spec-af37-11"),
            ("12-deployment-checklist.md", "Deployment Checklist & Runbook", "spec-af37-12"),
        ],
        "tickets": [],
    },
    "epic-0aabe215": {
        "folder": "epic-0aabe215-review-v3",
        "name": "Review System v3 Architecture",
        "specs": [
            ("01-architecture-overview.md", "Architecture Overview & Design Principles", "spec-0aab-01"),
            ("02-api-design.md", "API Design & Endpoints", "spec-0aab-02"),
            ("03-database-schema.md", "Database Schema & Migrations", "spec-0aab-03"),
            ("04-caching-strategy.md", "Caching Strategy & Redis Integration", "spec-0aab-04"),
            ("05-performance-optimization.md", "Performance Optimization Guidelines", "spec-0aab-05"),
        ],
        "tickets": [],
    },
    "epic-238cb09f": {
        "folder": "epic-238cb09f-prop-firm",
        "name": "Proprietary Trading Firm Integration",
        "specs": [],
        "tickets": [],
    },
    "epic-5b080ee3": {
        "folder": "epic-5b080ee3-agentic-sidebar",
        "name": "Agentic Sidebar UI Component",
        "specs": [
            ("01-component-architecture.md", "Component Architecture & State Management", "spec-5b08-01"),
            ("02-conversation-flow.md", "Conversation Flow & Message Handling", "spec-5b08-02"),
            ("03-accessibility-compliance.md", "Accessibility Compliance (WCAG 2.1)", "spec-5b08-03"),
        ],
        "tickets": [],
    },
    "epic-d81a1bce": {
        "folder": "epic-d81a1bce-trading-infra",
        "name": "Trading Infrastructure Core",
        "specs": [
            ("01-market-data-feed.md", "Market Data Feed Handler", "spec-d81a-01"),
            ("02-order-management.md", "Order Management System", "spec-d81a-02"),
            ("03-risk-engine.md", "Risk Engine & Position Limits", "spec-d81a-03"),
            ("04-execution-engine.md", "Execution Engine & Smart Order Routing", "spec-d81a-04"),
            ("05-strategy-framework.md", "Strategy Framework & Backtesting", "spec-d81a-05"),
            ("06-portfolio-manager.md", "Portfolio Manager & Allocation", "spec-d81a-06"),
            ("07-analytics-pipeline.md", "Analytics Pipeline & Metrics", "spec-d81a-07"),
            ("08-event-bus.md", "Event Bus & Message Queue", "spec-d81a-08"),
            ("09-data-storage.md", "Data Storage & Time-Series DB", "spec-d81a-09"),
            ("10-api-gateway.md", "API Gateway & Rate Limiting", "spec-d81a-10"),
            ("11-compliance-reporting.md", "Compliance Reporting & Audit Trail", "spec-d81a-11"),
        ],
        "tickets": [
            ("001-implement-feed-handler.md", "Implement Feed Handler for MT5", "ticket-d81a-001", "in-progress"),
            ("002-add-risk-controls.md", "Add Real-time Risk Controls", "ticket-d81a-002", "pending"),
            ("003-optimize-execution.md", "Optimize Execution Latency", "ticket-d81a-003", "pending"),
            ("004-backtest-framework.md", "Complete Backtest Framework", "ticket-d81a-004", "in-progress"),
            ("005-setup-monitoring.md", "Setup Prometheus Monitoring", "ticket-d81a-005", "done"),
            ("006-api-documentation.md", "API Documentation & OpenAPI Spec", "ticket-d81a-006", "pending"),
        ],
    },
}


def write_master_readme(base: Path, epics: dict) -> Path:
    """Create the master README.md index."""
    path = base / "README.md"
    
    lines = [
        "# QuantMindX Epic Export",
        "",
        "| Epic | Name | Specs | Tickets |",
        "|------|------|-------|---------|",
    ]
    
    for epic_id, meta in epics.items():
        folder = meta["folder"]
        name = meta["name"]
        specs_count = len(meta["specs"])
        tickets_count = len(meta["tickets"])
        link = f"[{epic_id}]({folder}/README.md)"
        lines.append(f"| {link} | {name} | {specs_count} | {tickets_count} |")
    
    lines.append("")
    
    path.write_text("\n".join(lines))
    return path


def write_epic_readme(epic_dir: Path, epic_id: str, meta: dict) -> Path:
    """Create the epic-level README.md."""
    path = epic_dir / "README.md"
    
    lines = [
        f"# {meta['name']}",
        "",
        f"**Epic ID:** `{epic_id}`",
        "",
    ]
    
    if meta["specs"]:
        lines.extend([
            "## Specs",
            "",
        ])
        for i, (filename, title, spec_id) in enumerate(meta["specs"], 1):
            lines.append(f"{i}. [{title}](specs/{filename})")
        lines.append("")
    
    if meta["tickets"]:
        lines.extend([
            "## Tickets",
            "",
        ])
        for i, (filename, title, ticket_id, status) in enumerate(meta["tickets"], 1):
            lines.append(f"{i}. [{title}](tickets/{filename})")
        lines.append("")
    
    path.write_text("\n".join(lines))
    return path


def write_spec_file(path: Path, title: str, epic_id: str, spec_id: str) -> Path:
    """Create a spec reference card file."""
    brief_desc = title.replace("-", " ").replace("_", " ") + " — see Traycer.AI for full details."
    
    content = dedent(f"""\
        # {title}
        
        **Epic:** `{epic_id}`
        **Spec ID:** `{spec_id}`
        **Traycer Reference:** `spec:{epic_id}/{spec_id}`
        
        ---
        
        > Open this spec in Traycer.AI using the reference above to view the full content.
        
        ---
        
        ## Summary
        
        {brief_desc}
    """)
    
    path.write_text(content)
    return path


def write_ticket_file(path: Path, title: str, epic_id: str, ticket_id: str, status: str) -> Path:
    """Create a ticket reference card file."""
    content = dedent(f"""\
        # {title}
        
        **Epic:** `{epic_id}`
        **Ticket ID:** `{ticket_id}`
        **Traycer Reference:** `ticket:{epic_id}/{ticket_id}`
        **Status:** {status}
        
        ---
        
        > Open this ticket in Traycer.AI using the reference above to view the full content.
    """)
    
    path.write_text(content)
    return path


def main():
    files_written = []
    
    # Create base directory
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Write master README
    master_readme = write_master_readme(BASE_DIR, EPICS)
    files_written.append(master_readme)
    
    # Process each epic
    for epic_id, meta in EPICS.items():
        epic_dir = BASE_DIR / meta["folder"]
        epic_dir.mkdir(parents=True, exist_ok=True)
        
        # Write epic README
        epic_readme = write_epic_readme(epic_dir, epic_id, meta)
        files_written.append(epic_readme)
        
        # Create specs directory and files
        if meta["specs"]:
            specs_dir = epic_dir / "specs"
            specs_dir.mkdir(parents=True, exist_ok=True)
            for filename, title, spec_id in meta["specs"]:
                spec_path = write_spec_file(specs_dir / filename, title, epic_id, spec_id)
                files_written.append(spec_path)
        
        # Create tickets directory and files
        if meta["tickets"]:
            tickets_dir = epic_dir / "tickets"
            tickets_dir.mkdir(parents=True, exist_ok=True)
            for filename, title, ticket_id, status in meta["tickets"]:
                ticket_path = write_ticket_file(tickets_dir / filename, title, epic_id, ticket_id, status)
                files_written.append(ticket_path)
    
    # Count different file types
    readme_count = sum(1 for p in files_written if p.name == "README.md")
    spec_count = sum(1 for p in files_written if p.parent.name == "specs")
    ticket_count = sum(1 for p in files_written if p.parent.name == "tickets")
    
    print("✅ Export complete.")
    print(f"Files written: {len(files_written)}")
    print(f"  - README files: {readme_count}")
    print(f"  - Spec files:   {spec_count}")
    print(f"  - Ticket files: {ticket_count}")
    print()
    print("Paths written:")
    for p in files_written:
        print(f"  {p}")


if __name__ == "__main__":
    main()