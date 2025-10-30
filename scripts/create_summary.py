#!/usr/bin/env python3
"""
GitHub Actions Job Summary Creator

Creates a formatted summary for GitHub Actions UI
100% LOCAL - NO EXTERNAL APIS
"""
import json
from pathlib import Path


def create_summary():
    """Create GitHub Actions job summary from analysis results"""
    results_file = Path('analysis_results.json')
    
    if not results_file.exists():
        print("⚠️ Analysis results not found")
        print("\n### ⚠️ Analysis Incomplete\n")
        print("The analysis did not complete successfully. Check the logs for errors.")
        return
    
    try:
        with open(results_file, 'r') as f:
            r = json.load(f)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        print("\n### ❌ Error Loading Results\n")
        print(f"Failed to load analysis results: {e}")
        return
    
    # Extract data
    stats = r.get('stats', {})
    metadata = r.get('analysis_metadata', {})
    errors = r.get('errors', [])
    warnings = r.get('warnings', [])
    security = r.get('security_issues', [])
    complexity = r.get('complexity_metrics', {})
    coverage = r.get('test_coverage', {})
    deps = r.get('dependency_health', {})
    healing = r.get('self_healing_actions', [])
    
    # Determine status
    if len(security) > 5:
        status = '🔴 **Critical**'
    elif len(errors) > 0:
        status = '🔴 **Errors Found**'
    elif len(warnings) > 5:
        status = '🟡 **Warnings**'
    else:
        status = '🟢 **Healthy**'
    
    # Create summary
    print(f"""
### ✅ Analysis Complete

**Repository Status:** {status}

#### 📈 Statistics
- **Files Scanned:** {metadata.get('files_scanned', 0)}
- **Files Skipped:** {metadata.get('files_skipped', 0)}
- **Total Lines of Code:** {stats.get('total_lines', 0):,}
- **Repository Size:** {stats.get('repository_size_mb', 0)} MB

#### 📊 File Breakdown
- Python: {stats.get('python_files', 0)}
- JavaScript: {stats.get('javascript_files', 0)}
- TypeScript: {stats.get('typescript_files', 0)}
- Markdown: {stats.get('markdown_files', 0)}
- JSON: {stats.get('json_files', 0)}
- YAML: {stats.get('yaml_files', 0)}

#### 🎯 Analysis Results
- **Errors:** {len(errors)}
- **Warnings:** {len(warnings)}
- **Security Issues:** {len(security)}

#### 📊 Code Quality Metrics
""")
    
    if complexity:
        avg_complexity = complexity.get('average_complexity', 'N/A')
        avg_maintain = complexity.get('average_maintainability', 'N/A')
        high_complex = len(complexity.get('high_complexity_files', []))
        low_maintain = len(complexity.get('low_maintainability_files', []))
        
        print(f"""- **Avg Cyclomatic Complexity:** {avg_complexity}
- **Avg Maintainability Index:** {avg_maintain}
- **High Complexity Files:** {high_complex}
- **Low Maintainability Files:** {low_maintain}""")
    else:
        print("- *No complexity metrics available (radon not installed)*")
    
    print("\n#### 🧪 Test Coverage")
    if coverage.get('has_tests'):
        cov_pct = coverage.get('coverage_percent', 0)
        covered = coverage.get('covered_lines', 0)
        total = coverage.get('total_lines', 0)
        print(f"""- **Coverage:** {cov_pct:.1f}%
- **Covered Lines:** {covered:,}
- **Total Lines:** {total:,}
- **Files with Low Coverage (<50%):** {len(coverage.get('missing_coverage_files', []))}""")
    else:
        print("- **Status:** No tests detected")
    
    print("\n#### 🔒 Security")
    if len(security) > 0:
        sec_by_type = {}
        for issue in security:
            sec_type = issue.get('type', 'unknown')
            sec_by_type[sec_type] = sec_by_type.get(sec_type, 0) + 1
        
        print(f"- **Total Issues:** {len(security)}")
        for sec_type, count in sec_by_type.items():
            print(f"- {sec_type.title()}: {count}")
    else:
        print("- **Status:** ✅ No security issues found")
    
    print("\n#### 📦 Dependencies")
    py_deps = deps.get('python', {})
    node_deps = deps.get('node', {})
    
    if py_deps.get('total', 0) > 0 or node_deps.get('total', 0) > 0:
        if py_deps.get('total', 0) > 0:
            print(f"- **Python:** {py_deps['total']} packages ({len(py_deps.get('outdated', []))} outdated)")
        if node_deps.get('total', 0) > 0:
            print(f"- **Node.js:** {node_deps['total']} packages ({len(node_deps.get('outdated', []))} outdated)")
    else:
        print("- *No dependency files found*")
    
    print("\n#### 🔧 Self-Healing")
    if healing:
        print(f"- **Actions Taken:** {len(healing)}")
        for action in healing:
            print(f"  - {action}")
    else:
        print("- **Actions Taken:** None needed")
    
    print("\n#### 🛠️ Tools Status")
    tools = r.get('tools_available', {})
    installed = [tool for tool, available in tools.items() if available]
    failed = [tool for tool, available in tools.items() if not available]
    
    if installed:
        print(f"- **Installed:** {', '.join(installed)}")
    if failed:
        print(f"- **Not Available:** {', '.join(failed)}")
    
    print(f"""
---

📝 **Full results available in:** [README.md](../blob/main/README.md)  
📋 **Changelog:** [CHANGELOG.md](../blob/main/CHANGELOG.md)  
📊 **Raw data:** Download artifacts from this workflow run
""")


if __name__ == "__main__":
    create_summary()
