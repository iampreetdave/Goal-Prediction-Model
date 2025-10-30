#!/usr/bin/env python3
"""
Advanced README Updater

Updates README.md with comprehensive analysis results
100% LOCAL - NO EXTERNAL APIS
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import re
import subprocess


class AdvancedReadmeUpdater:
    """Handles README creation and updates with full analysis data"""
    
    # HTML comment markers for identifying analysis section
    ANALYSIS_MARKER_START = "<!-- AUTO-ANALYSIS-START -->"
    ANALYSIS_MARKER_END = "<!-- AUTO-ANALYSIS-END -->"
    
    def __init__(self, results_file: str = 'analysis_results.json'):
        """Initialize updater with results file path"""
        self.results_file = Path(results_file)
        self.readme_path = Path('README.md')
        self.results = self.load_results()
    
    def load_results(self) -> Dict[str, Any]:
        """Load analysis results from JSON file"""
        try:
            if not self.results_file.exists():
                print("⚠️  No results file found")
                return self.empty_results()
            
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading results: {e}")
            return self.empty_results()
    
    def empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'timestamp': datetime.now().isoformat(),
            'errors': [], 'warnings': [], 'stats': {},
            'complexity_metrics': {}, 'test_coverage': {},
            'security_issues': [], 'dependency_health': {},
            'file_insights': [], 'recent_changes': [],
            'self_healing_actions': [],
            'historical_trends': [],
            'analysis_metadata': {'first_run': True}
        }
    
    def detect_project_name(self) -> str:
        """Detect project name from git remote or directory"""
        # Try git remote URL
        try:
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                match = re.search(r'/([^/]+?)(?:\.git)?$', url)
                if match:
                    return match.group(1).replace('-', ' ').replace('_', ' ').title()
        except:
            pass
        
        # Fallback to directory name
        try:
            return Path.cwd().name.replace('-', ' ').replace('_', ' ').title()
        except:
            return "Project Repository"
    
    def create_default_readme(self) -> str:
        """Create a default README if none exists"""
        project_name = self.detect_project_name()
        
        readme_lines = [
            f"# {project_name}",
            "",
            f"Welcome to {project_name}! This README was automatically generated.",
            "",
            "## About",
            "",
            "Add your project description here.",
            "",
            "## Getting Started",
            "",
            "Add installation and usage instructions here.",
            "",
            "## Features",
            "",
            "- Feature 1",
            "- Feature 2",
            "- Feature 3",
            ""
        ]
        
        return '\n'.join(readme_lines)
    
    def read_existing_readme(self) -> str:
        """Read existing README or create new one"""
        if self.readme_path.exists():
            try:
                with open(self.readme_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️  Error reading README: {e}")
                return self.create_default_readme()
        else:
            print("ℹ️  Creating new README.md")
            return self.create_default_readme()
    
    def remove_old_analysis_section(self, content: str) -> str:
        """Remove old analysis section using HTML markers"""
        pattern = f"{re.escape(self.ANALYSIS_MARKER_START)}.*?{re.escape(self.ANALYSIS_MARKER_END)}"
        content = re.sub(pattern, '', content, flags=re.DOTALL)
        return content.rstrip() + "\n\n"
    
    def generate_trends_chart(self) -> str:
        """Generate ASCII trend visualization"""
        trends = self.results.get('historical_trends', [])
        
        if len(trends) < 2:
            return ""
        
        chart_lines = ["\n### 📈 Trends (Last 30 Days)\n"]
        
        # Error trend
        error_counts = [t['error_count'] for t in trends[-7:]]
        if error_counts:
            chart_lines.append("**Errors:** ")
            chart_lines.append(''.join(['█' if e > 0 else '▁' for e in error_counts]))
            chart_lines.append(f" (Current: {error_counts[-1]})\n")
        
        # Coverage trend
        coverage = [t.get('test_coverage', 0) for t in trends[-7:]]
        if any(coverage):
            chart_lines.append(f"**Test Coverage:** {coverage[-1]:.1f}% ")
            
            if len(coverage) > 1:
                change = coverage[-1] - coverage[-2]
                if change > 0:
                    chart_lines.append(f"📈 +{change:.1f}%")
                elif change < 0:
                    chart_lines.append(f"📉 {change:.1f}%")
            chart_lines.append("\n")
        
        # Lines of code trend
        loc = [t.get('total_lines', 0) for t in trends[-7:]]
        if any(loc):
            chart_lines.append(f"**Lines of Code:** {loc[-1]:,} ")
            
            if len(loc) > 1 and loc[-2] > 0:
                change_pct = ((loc[-1] - loc[-2]) / loc[-2]) * 100
                if abs(change_pct) > 0.1:
                    chart_lines.append(f"({'📈' if change_pct > 0 else '📉'} {change_pct:+.1f}%)")
            chart_lines.append("\n")
        
        return ''.join(chart_lines)
    
    def generate_analysis_section(self) -> str:
        """Generate the comprehensive analysis section for README"""
        
        # Extract data from results
        stats = self.results.get('stats', {})
        errors = self.results.get('errors', [])
        warnings = self.results.get('warnings', [])
        security = self.results.get('security_issues', [])
        complexity = self.results.get('complexity_metrics', {})
        coverage = self.results.get('test_coverage', {})
        deps = self.results.get('dependency_health', {})
        changes = self.results.get('recent_changes', [])
        insights = self.results.get('file_insights', [])
        healing = self.results.get('self_healing_actions', [])
        
        error_count = len(errors)
        warning_count = len(warnings)
        security_count = len(security)
        
        # Format timestamp
        try:
            timestamp = datetime.fromisoformat(self.results['timestamp'])
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M UTC')
        except:
            timestamp_str = "Unknown"
        
        # Determine overall status
        if security_count > 5:
            status_emoji, status_text, status_color = '🔴', f'{security_count} Security Issues', '🔴'
        elif error_count > 0:
            status_emoji, status_text, status_color = '❌', f'{error_count} Errors', '🔴'
        elif warning_count > 5:
            status_emoji, status_text, status_color = '⚠️', f'{warning_count} Warnings', '🟡'
        else:
            status_emoji, status_text, status_color = '✅', 'Healthy', '🟢'
        
        # Start building the section
        lines = [
            self.ANALYSIS_MARKER_START,
            "",
            "---",
            "",
            "## 📊 Automated Repository Analysis",
            "",
            "<div align=\"center\">",
            "",
            f"{status_color} **Status:** {status_emoji} **{status_text}**  ",
            f"🕒 **Last Analysis:** {timestamp_str}",
            "",
            "</div>",
            ""
        ]
        
        # Self-healing notice
        if healing:
            lines.append(f"> 🔧 **Self-Healing Actions Taken:** {', '.join(healing)}")
            lines.append("")
        
        # Main statistics table
        lines.extend([
            "### 📈 Project Statistics",
            "",
            "| Metric | Value | Status |",
            "|--------|-------|--------|"
        ])
        
        if stats:
            lines.append(f"| **Total Files** | {stats.get('total_files', 0)} | - |")
            lines.append(f"| **Lines of Code** | {stats.get('total_lines', 0):,} | - |")
            lines.append(f"| **Repository Size** | {stats.get('repository_size_mb', 0)} MB | - |")
            
            # Test coverage
            if coverage.get('has_tests'):
                cov_pct = coverage.get('coverage_percent', 0)
                cov_status = '🟢' if cov_pct >= 80 else '🟡' if cov_pct >= 60 else '🔴'
                lines.append(f"| **Test Coverage** | {cov_pct:.1f}% | {cov_status} |")
            else:
                lines.append("| **Test Coverage** | No tests | 🔴 |")
            
            # Maintainability index
            if complexity.get('average_maintainability'):
                mi = complexity['average_maintainability']
                mi_status = '🟢' if mi >= 20 else '🟡' if mi >= 10 else '🔴'
                lines.append(f"| **Maintainability Index** | {mi:.1f} | {mi_status} |")
            
            # Cyclomatic complexity
            if complexity.get('average_complexity'):
                cc = complexity['average_complexity']
                cc_status = '🟢' if cc <= 5 else '🟡' if cc <= 10 else '🔴'
                lines.append(f"| **Avg Cyclomatic Complexity** | {cc:.1f} | {cc_status} |")
            
            lines.append(f"| **Security Issues** | {security_count} | {'🔴' if security_count > 0 else '🟢'} |")
        
        lines.append("")
        
        # Add trends
        trends_chart = self.generate_trends_chart()
        if trends_chart:
            lines.append(trends_chart)
        
        # Code quality concerns
        if complexity:
            high_complexity = complexity.get('high_complexity_files', [])
            low_maintain = complexity.get('low_maintainability_files', [])
            
            if high_complexity or low_maintain:
                lines.extend([
                    "",
                    "### ⚠️ Code Quality Concerns",
                    ""
                ])
                
                if high_complexity:
                    lines.extend([
                        "<details>",
                        "<summary>High Complexity Files</summary>",
                        ""
                    ])
                    for item in high_complexity[:5]:
                        lines.append(f"- `{item['file']}` (Complexity: {item['complexity']})")
                    lines.extend(["", "</details>", ""])
                
                if low_maintain:
                    lines.extend([
                        "<details>",
                        "<summary>Low Maintainability Files</summary>",
                        ""
                    ])
                    for item in low_maintain[:5]:
                        lines.append(f"- `{item['file']}` (Score: {item['score']}, Rank: {item['rank']})")
                    lines.extend(["", "</details>", ""])
        
        # Security issues
        if security_count > 0:
            lines.extend([
                "",
                f"### 🔒 Security Vulnerabilities ({security_count})",
                "",
                "<details>",
                "<summary>Click to expand security issues</summary>",
                ""
            ])
            
            for issue in security[:10]:
                if issue.get('type') in ['python', 'node']:
                    lines.append(
                        f"- **{issue.get('package', 'Unknown')}** "
                        f"({issue.get('version', 'unknown')}) - "
                        f"{issue.get('vulnerability', 'Security issue')} "
                        f"[{issue.get('severity', 'unknown').upper()}]"
                    )
                elif issue.get('type') == 'code':
                    lines.append(
                        f"- `{issue.get('file', 'Unknown')}:{issue.get('line', '?')}` - "
                        f"{issue.get('issue', 'Security issue')} "
                        f"[{issue.get('severity', 'unknown').upper()}]"
                    )
            
            if security_count > 10:
                lines.append(f"\n*...and {security_count - 10} more*")
            
            lines.extend(["", "</details>", ""])
        
        # Dependency health
        if deps.get('python', {}).get('outdated') or deps.get('node', {}).get('outdated'):
            lines.extend([
                "",
                "### 📦 Dependency Health",
                ""
            ])
            
            py_deps = deps.get('python', {})
            if py_deps.get('outdated'):
                lines.append(f"**Python:** {len(py_deps['outdated'])} outdated packages")
                lines.extend([
                    "<details>",
                    "<summary>View outdated Python packages</summary>",
                    ""
                ])
                for pkg in py_deps['outdated'][:10]:
                    lines.append(f"- `{pkg['name']}`: {pkg['current']} → {pkg['latest']}")
                lines.extend(["", "</details>", ""])
            
            node_deps = deps.get('node', {})
            if node_deps.get('outdated'):
                lines.append(f"**Node.js:** {len(node_deps['outdated'])} outdated packages")
                lines.extend([
                    "<details>",
                    "<summary>View outdated Node packages</summary>",
                    ""
                ])
                for pkg in node_deps['outdated'][:10]:
                    lines.append(f"- `{pkg['name']}`: {pkg['current']} → {pkg['latest']}")
                lines.extend(["", "</details>", ""])
        
        # File insights
        if insights:
            lines.extend([
                "",
                "### 📁 File Insights",
                "",
                "<details>",
                "<summary>Largest Files</summary>",
                ""
            ])
            for insight in insights[:5]:
                lines.append(f"- `{insight['file']}` ({insight['size']})")
            lines.extend(["", "</details>", ""])
        
        # Recent changes
        if changes:
            lines.extend([
                "",
                "### 📝 Recent Changes",
                ""
            ])
            for change in changes[:5]:
                lines.append(
                    f"- [`{change['hash']}`](../../commit/{change['hash']}) "
                    f"{change['message']} "
                    f"*({change['time']} by {change['author']})*"
                )
            
            if Path('CHANGELOG.md').exists():
                lines.extend(["", "[View Full Changelog →](CHANGELOG.md)"])
        
        # Footer
        lines.extend([
            "",
            "---",
            "",
            "<div align=\"center\">",
            "  <sub>🤖 Automated analysis powered by <a href=\".github/workflows/daily-analysis.yml\">GitHub Actions</a></sub><br>",
            "  <sub>Includes complexity metrics, security scanning, test coverage, and trend analysis</sub>",
            "</div>",
            "",
            self.ANALYSIS_MARKER_END
        ])
        
        return '\n'.join(lines)
    
    def update_readme(self) -> bool:
        """Main function to update README with analysis results"""
        try:
            print("\n📝 Updating README.md...")
            
            # Read or create README
            content = self.read_existing_readme()
            
            # Remove old analysis section
            content = self.remove_old_analysis_section(content)
            
            # Generate new analysis section
            analysis_section = self.generate_analysis_section()
            
            # Combine content
            new_content = content + analysis_section
            
            # Write updated README
            with open(self.readme_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ README.md updated successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error updating README: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    updater = AdvancedReadmeUpdater()
    success = updater.update_readme()
    exit(0 if success else 1)
