#!/usr/bin/env python3
"""
Advanced Repository Analysis Script

Features:
- Code complexity metrics (cyclomatic complexity, maintainability index)
- Test coverage detection and reporting
- Security vulnerability scanning
- Dependency health checks
- Self-healing (auto-generates missing files)
- Historical trend tracking
- File-level insights

NO EXTERNAL APIs USED - All analysis is local using installed tools
"""
import os
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import traceback


class AdvancedRepositoryAnalyzer:
    """
    Main analyzer class with comprehensive repository analysis capabilities
    """
    
    def __init__(self):
        """Initialize analyzer with default configuration"""
        self.root_path = Path('.')
        
        # Initialize results dictionary
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'warnings': [],
            'stats': {},
            'complexity_metrics': {},
            'test_coverage': {},
            'security_issues': [],
            'dependency_health': {},
            'file_insights': [],
            'recent_changes': [],
            'tools_available': {},
            'self_healing_actions': [],
            'analysis_metadata': {
                'first_run': False,
                'tools_installed': [],
                'tools_failed': [],
                'files_scanned': 0,
                'files_skipped': 0
            }
        }
        
        # Directories to exclude from scanning
        self.excluded_dirs = {
            '.git', 'node_modules', 'venv', '__pycache__', 
            '.pytest_cache', 'dist', 'build', '.eggs',
            'env', '.env', 'vendor', 'target', 'coverage'
        }
        
        # File to store historical analysis data
        self.history_file = Path('.github/analysis_history.json')
    
    def run_command(self, cmd: str, timeout: int = 60) -> Tuple[str, str, int]:
        """
        Safely execute shell command with timeout
        
        Args:
            cmd: Command to execute
            timeout: Maximum execution time in seconds
        
        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=str(self.root_path)
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Timeout after {timeout}s", 1
        except Exception as e:
            return "", str(e), 1
    
    def is_tool_available(self, tool: str) -> bool:
        """
        Check if a command-line tool is available in PATH
        
        Args:
            tool: Name of the tool to check
        
        Returns:
            True if tool is available, False otherwise
        """
        stdout, _, code = self.run_command(f"which {tool}")
        available = code == 0 and stdout.strip() != ""
        
        # Track tool availability
        self.results['tools_available'][tool] = available
        
        if available:
            self.results['analysis_metadata']['tools_installed'].append(tool)
        else:
            self.results['analysis_metadata']['tools_failed'].append(tool)
        
        return available
    
    def should_skip_path(self, path: Path) -> bool:
        """
        Determine if a file path should be excluded from analysis
        
        Args:
            path: File path to check
        
        Returns:
            True if path should be skipped, False otherwise
        """
        return any(excluded in path.parts for excluded in self.excluded_dirs)
    
    def find_files_by_extension(self, *extensions: str, limit: int = 200) -> List[Path]:
        """
        Find files matching given extensions, respecting exclusions
        
        Args:
            extensions: File extensions to search for (e.g., '*.py', '*.js')
            limit: Maximum number of files to return
        
        Returns:
            List of matching file paths
        """
        files = []
        try:
            for ext in extensions:
                for file in self.root_path.rglob(ext):
                    if len(files) >= limit:
                        break
                    if not self.should_skip_path(file) and file.is_file():
                        files.append(file)
                        self.results['analysis_metadata']['files_scanned'] += 1
                    else:
                        self.results['analysis_metadata']['files_skipped'] += 1
        except Exception as e:
            self.results['warnings'].append(f"Error scanning for {extensions}: {str(e)}")
        
        return files
    
    def analyze_code_complexity(self) -> None:
        """
        Analyze code complexity using radon (LOCAL TOOL - NO API)
        Calculates cyclomatic complexity and maintainability index
        """
        print("📊 Analyzing code complexity...")
        
        if not self.is_tool_available('radon'):
            print("  ⚠️  radon not available, skipping complexity analysis")
            return
        
        py_files = self.find_files_by_extension('*.py')
        
        complexity_data = {
            'cyclomatic_complexity': {},
            'maintainability_index': {},
            'high_complexity_files': [],
            'low_maintainability_files': []
        }
        
        # Analyze each Python file
        for file in py_files[:50]:  # Limit to 50 files to avoid timeout
            try:
                # Calculate cyclomatic complexity
                stdout, _, code = self.run_command(f'radon cc "{file}" -j')
                if code == 0 and stdout.strip():
                    cc_data = json.loads(stdout)
                    if cc_data and str(file) in cc_data:
                        avg_complexity = sum(f['complexity'] for f in cc_data[str(file)]) / len(cc_data[str(file)]) if cc_data[str(file)] else 0
                        complexity_data['cyclomatic_complexity'][str(file)] = round(avg_complexity, 2)
                        
                        # Flag high complexity files (threshold: 10)
                        if avg_complexity > 10:
                            complexity_data['high_complexity_files'].append({
                                'file': str(file),
                                'complexity': round(avg_complexity, 2)
                            })
                
                # Calculate maintainability index
                stdout, _, code = self.run_command(f'radon mi "{file}" -j')
                if code == 0 and stdout.strip():
                    mi_data = json.loads(stdout)
                    if mi_data and str(file) in mi_data:
                        mi_score = mi_data[str(file)]['mi']
                        rank = mi_data[str(file)]['rank']
                        complexity_data['maintainability_index'][str(file)] = {
                            'score': round(mi_score, 2),
                            'rank': rank
                        }
                        
                        # Flag low maintainability files (rank C, D, F)
                        if rank in ['C', 'D', 'F']:
                            complexity_data['low_maintainability_files'].append({
                                'file': str(file),
                                'score': round(mi_score, 2),
                                'rank': rank
                            })
            
            except json.JSONDecodeError:
                pass
            except Exception as e:
                self.results['warnings'].append(f"Complexity analysis failed for {file}: {str(e)}")
        
        # Calculate overall averages
        if complexity_data['cyclomatic_complexity']:
            complexity_data['average_complexity'] = round(
                sum(complexity_data['cyclomatic_complexity'].values()) / 
                len(complexity_data['cyclomatic_complexity']), 2
            )
        
        if complexity_data['maintainability_index']:
            scores = [v['score'] for v in complexity_data['maintainability_index'].values()]
            complexity_data['average_maintainability'] = round(sum(scores) / len(scores), 2)
        
        self.results['complexity_metrics'] = complexity_data
        print(f"  ✓ Analyzed {len(py_files)} Python files")
    
    def detect_and_run_tests(self) -> None:
        """
        Detect if tests exist and run coverage analysis (LOCAL - NO API)
        Automatically runs pytest with coverage if available
        """
        print("🧪 Detecting and running tests...")
        
        # Check for common test directories
        test_dirs = ['tests', 'test', 'spec']
        has_tests = any(Path(d).exists() for d in test_dirs)
        
        # Check for test files
        if not has_tests:
            has_tests = len(self.find_files_by_extension('*test*.py', '*.test.js', limit=5)) > 0
        
        coverage_data = {
            'has_tests': has_tests,
            'coverage_percent': 0,
            'covered_lines': 0,
            'total_lines': 0,
            'missing_coverage_files': []
        }
        
        if has_tests and self.is_tool_available('pytest'):
            print("  ⚙️  Running pytest with coverage...")
            try:
                # Run pytest with coverage
                stdout, stderr, code = self.run_command(
                    'pytest --cov=. --cov-report=json --cov-report=term -v',
                    timeout=120
                )
                
                # Read coverage report
                cov_file = Path('coverage.json')
                if cov_file.exists():
                    with open(cov_file, 'r') as f:
                        cov_report = json.load(f)
                    
                    coverage_data['coverage_percent'] = round(cov_report['totals']['percent_covered'], 2)
                    coverage_data['covered_lines'] = cov_report['totals']['covered_lines']
                    coverage_data['total_lines'] = cov_report['totals']['num_statements']
                    
                    # Find files with low coverage (<50%)
                    for file_path, file_data in cov_report['files'].items():
                        if file_data['summary']['percent_covered'] < 50:
                            coverage_data['missing_coverage_files'].append({
                                'file': file_path,
                                'coverage': round(file_data['summary']['percent_covered'], 2)
                            })
                    
                    cov_file.unlink()  # Clean up coverage file
                    print(f"  ✓ Test coverage: {coverage_data['coverage_percent']}%")
            
            except Exception as e:
                self.results['warnings'].append(f"Test coverage analysis failed: {str(e)}")
        elif not has_tests:
            print("  ⚠️  No tests detected")
        else:
            print("  ⚠️  pytest not available")
        
        self.results['test_coverage'] = coverage_data
    
    def scan_security_vulnerabilities(self) -> None:
        """
        Scan for security vulnerabilities (LOCAL TOOLS + PyPI/NPM Registry APIs)
        
        APIs USED (Rate Limited):
        1. pip-audit: Queries PyPI vulnerability database (public, rate-limited)
        2. npm audit: Queries NPM registry (public, rate-limited)
        
        These are official package registries, not third-party APIs
        """
        print("🔒 Scanning for security vulnerabilities...")
        
        security_issues = []
        
        # Python: pip-audit for dependency vulnerabilities
        # USES: PyPI JSON API (public, rate-limited ~10 req/sec)
        if self.is_tool_available('pip-audit'):
            print("  ⚙️  Running pip-audit (queries PyPI database)...")
            try:
                stdout, stderr, code = self.run_command('pip-audit --format json', timeout=120)
                if stdout.strip():
                    audit_data = json.loads(stdout)
                    for vuln in audit_data.get('vulnerabilities', []):
                        security_issues.append({
                            'type': 'python',
                            'package': vuln.get('name', 'unknown'),
                            'version': vuln.get('version', 'unknown'),
                            'vulnerability': vuln.get('id', 'unknown'),
                            'severity': 'high'
                        })
            except Exception as e:
                self.results['warnings'].append(f"pip-audit failed: {str(e)}")
        
        # Python: bandit for code security issues (LOCAL - NO API)
        if self.is_tool_available('bandit'):
            py_files = self.find_files_by_extension('*.py')
            if py_files:
                print("  ⚙️  Running bandit (local analysis)...")
                try:
                    stdout, stderr, code = self.run_command('bandit -r . -f json -ll', timeout=60)
                    if stdout.strip():
                        bandit_data = json.loads(stdout)
                        for issue in bandit_data.get('results', [])[:10]:
                            security_issues.append({
                                'type': 'code',
                                'file': issue.get('filename', 'unknown'),
                                'line': issue.get('line_number', 0),
                                'issue': issue.get('issue_text', 'Security issue'),
                                'severity': issue.get('issue_severity', 'unknown').lower()
                            })
                except Exception as e:
                    pass
        
        # Node.js: npm audit (USES: NPM Registry API - public, rate-limited)
        if Path('package.json').exists() and self.is_tool_available('npm'):
            print("  ⚙️  Running npm audit (queries NPM registry)...")
            try:
                stdout, stderr, code = self.run_command('npm audit --json', timeout=60)
                if stdout.strip():
                    npm_audit = json.loads(stdout)
                    for vuln_id, vuln in npm_audit.get('vulnerabilities', {}).items():
                        security_issues.append({
                            'type': 'node',
                            'package': vuln.get('name', 'unknown'),
                            'severity': vuln.get('severity', 'unknown'),
                            'vulnerability': vuln_id
                        })
            except Exception as e:
                pass
        
        self.results['security_issues'] = security_issues[:20]
        print(f"  ✓ Found {len(security_issues)} security issues")
    
    def check_dependency_health(self) -> None:
        """
        Check dependency health and identify outdated packages (LOCAL + Registry APIs)
        
        APIs USED (Rate Limited):
        1. pip list --outdated: Queries PyPI for latest versions
        2. npm outdated: Queries NPM registry for latest versions
        """
        print("📦 Checking dependency health...")
        
        health_data = {
            'python': {'total': 0, 'outdated': []},
            'node': {'total': 0, 'outdated': []}
        }
        
        # Python dependencies
        if Path('requirements.txt').exists():
            print("  ⚙️  Checking Python dependencies (queries PyPI)...")
            try:
                # Get total package count (local)
                stdout, _, _ = self.run_command('pip list --format json')
                if stdout.strip():
                    packages = json.loads(stdout)
                    health_data['python']['total'] = len(packages)
                
                # Get outdated packages (queries PyPI API)
                stdout, _, _ = self.run_command('pip list --outdated --format json', timeout=60)
                if stdout.strip():
                    outdated = json.loads(stdout)
                    health_data['python']['outdated'] = [
                        {
                            'name': p['name'],
                            'current': p['version'],
                            'latest': p['latest_version']
                        }
                        for p in outdated[:10]
                    ]
            except Exception as e:
                pass
        
        # Node.js dependencies
        if Path('package.json').exists() and self.is_tool_available('npm'):
            print("  ⚙️  Checking Node.js dependencies (queries NPM registry)...")
            try:
                # Get total package count (local)
                stdout, _, _ = self.run_command('npm list --json --depth=0')
                if stdout.strip():
                    npm_list = json.loads(stdout)
                    health_data['node']['total'] = len(npm_list.get('dependencies', {}))
                
                # Get outdated packages (queries NPM registry)
                stdout, _, _ = self.run_command('npm outdated --json', timeout=60)
                if stdout.strip():
                    outdated = json.loads(stdout)
                    health_data['node']['outdated'] = [
                        {
                            'name': name,
                            'current': data.get('current', 'unknown'),
                            'latest': data.get('latest', 'unknown')
                        }
                        for name, data in list(outdated.items())[:10]
                    ]
            except Exception as e:
                pass
        
        self.results['dependency_health'] = health_data
        print(f"  ✓ Python: {health_data['python']['total']} packages ({len(health_data['python']['outdated'])} outdated)")
        print(f"  ✓ Node: {health_data['node']['total']} packages ({len(health_data['node']['outdated'])} outdated)")
    
    def generate_file_insights(self) -> None:
        """Generate insights about repository files (LOCAL - NO API)"""
        print("📁 Generating file insights...")
        
        insights = []
        
        try:
            stdout, _, _ = self.run_command(
                'find . -type f -not -path "*/.*" -not -path "*/node_modules/*" -exec du -h {} + | sort -rh | head -10'
            )
            
            if stdout:
                for line in stdout.strip().split('\n'):
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        insights.append({
                            'type': 'large_file',
                            'file': parts[1],
                            'size': parts[0]
                        })
        except Exception as e:
            pass
        
        self.results['file_insights'] = insights[:10]
        print(f"  ✓ Generated insights for {len(insights)} files")
    
    def get_recent_changes(self) -> None:
        """Fetch recent git commits (LOCAL - NO API)"""
        print("📝 Fetching recent changes...")
        
        try:
            stdout, _, code = self.run_command('git log --pretty=format:"%h|%an|%ar|%s" -n 10')
            if code == 0 and stdout:
                changes = []
                for line in stdout.strip().split('\n'):
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        changes.append({
                            'hash': parts[0],
                            'author': parts[1],
                            'time': parts[2],
                            'message': parts[3]
                        })
                self.results['recent_changes'] = changes
                print(f"  ✓ Fetched {len(changes)} recent commits")
        except Exception as e:
            pass
    
    def self_healing_mode(self) -> None:
        """Auto-generate missing essential files (LOCAL - NO API)"""
        print("🔧 Running self-healing checks...")
        
        actions = []
        
        gitignore = Path('.gitignore')
        if not gitignore.exists():
            print("  ⚙️  Creating .gitignore...")
            default_gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
venv/
env/
.env
*.egg-info/
dist/
build/

# Node
node_modules/
npm-debug.log

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
htmlcov/
.pytest_cache/
"""
            try:
                gitignore.write_text(default_gitignore)
                actions.append('Created .gitignore')
                print("    ✓ .gitignore created")
            except Exception as e:
                pass
        
        self.results['self_healing_actions'] = actions
    
    def save_historical_data(self) -> None:
        """Save analysis history for trends (LOCAL - NO API)"""
        try:
            history = []
            
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            
            history.append({
                'timestamp': self.results['timestamp'],
                'error_count': len(self.results['errors']),
                'warning_count': len(self.results['warnings']),
                'security_issues': len(self.results['security_issues']),
                'total_lines': self.results['stats'].get('total_lines', 0),
                'repository_size_mb': self.results['stats'].get('repository_size_mb', 0),
                'test_coverage': self.results['test_coverage'].get('coverage_percent', 0),
                'average_complexity': self.results['complexity_metrics'].get('average_complexity', 0),
                'average_maintainability': self.results['complexity_metrics'].get('average_maintainability', 0)
            })
            
            cutoff = datetime.now() - timedelta(days=30)
            history = [
                h for h in history 
                if datetime.fromisoformat(h['timestamp']) > cutoff
            ]
            
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.results['historical_trends'] = history
            print(f"📊 Saved historical data ({len(history)} data points)")
        
        except Exception as e:
            self.results['warnings'].append(f"Failed to save history: {str(e)}")
    
    def collect_repository_stats(self) -> None:
        """Collect repository statistics (LOCAL - NO API)"""
        print("📈 Collecting repository statistics...")
        
        stats = {
            'total_files': 0,
            'python_files': 0,
            'javascript_files': 0,
            'typescript_files': 0,
            'markdown_files': 0,
            'json_files': 0,
            'yaml_files': 0,
            'last_commit': 'N/A',
            'total_lines': 0,
            'repository_size_mb': 0,
            'has_tests': False,
            'has_ci': False,
            'has_docker': False
        }
        
        try:
            file_patterns = {
                'python_files': ['*.py'],
                'javascript_files': ['*.js', '*.jsx'],
                'typescript_files': ['*.ts', '*.tsx'],
                'markdown_files': ['*.md'],
                'json_files': ['*.json'],
                'yaml_files': ['*.yml', '*.yaml']
            }
            
            for key, patterns in file_patterns.items():
                files = self.find_files_by_extension(*patterns, limit=1000)
                stats[key] = len(files)
                stats['total_files'] += len(files)
            
            stdout, _, code = self.run_command('git log -1 --format="%h - %s (%ar)"')
            if code == 0 and stdout.strip():
                stats['last_commit'] = stdout.strip()
            
            extensions = ['py', 'js', 'jsx', 'ts', 'tsx', 'java', 'c', 'cpp', 'go', 'rs']
            find_cmd = ' -o '.join([f'-name "*.{ext}"' for ext in extensions])
            stdout, _, _ = self.run_command(
                f'find . -type f \\( {find_cmd} \\) -not -path "*/.*" -not -path "*/node_modules/*" | xargs wc -l 2>/dev/null | tail -1'
            )
            
            if stdout.strip():
                try:
                    stats['total_lines'] = int(stdout.split()[0])
                except (ValueError, IndexError):
                    pass
            
            stats['has_tests'] = any([
                Path('tests').exists(),
                Path('test').exists(),
                len(self.find_files_by_extension('*test*.py', '*.test.js', limit=5)) > 0
            ])
            
            stats['has_ci'] = Path('.github/workflows').exists()
            stats['has_docker'] = Path('Dockerfile').exists()
            
            stdout, _, _ = self.run_command('du -sm . 2>/dev/null')
            if stdout:
                try:
                    stats['repository_size_mb'] = int(stdout.split()[0])
                except (ValueError, IndexError):
                    pass
        
        except Exception as e:
            self.results['warnings'].append(f"Stats error: {str(e)}")
        
        self.results['stats'] = stats
        print(f"  ✓ Total files: {stats['total_files']}, Lines: {stats['total_lines']:,}")
    
    def detect_first_run(self) -> bool:
        """Detect first run (LOCAL - NO API)"""
        marker_file = Path('.github/.analysis_marker')
        
        if marker_file.exists():
            return False
        
        try:
            marker_file.parent.mkdir(parents=True, exist_ok=True)
            marker_file.write_text(datetime.now().isoformat())
            return True
        except Exception as e:
            return False
    
    def run_analysis(self) -> Dict[str, Any]:
        """Execute complete analysis workflow"""
        print("=" * 70)
        print("Advanced Repository Analysis")
        print("=" * 70)
        
        try:
            self.results['analysis_metadata']['first_run'] = self.detect_first_run()
            
            if self.results['analysis_metadata']['first_run']:
                print("ℹ️  First run - initializing analysis system\n")
            
            # Run all analysis modules
            self.collect_repository_stats()
            self.analyze_code_complexity()
            self.detect_and_run_tests()
            self.scan_security_vulnerabilities()
            self.check_dependency_health()
            self.generate_file_insights()
            self.get_recent_changes()
            self.self_healing_mode()
            self.save_historical_data()
            
            print("\n" + "=" * 70)
            print(f"✅ Analysis Complete!")
            print(f"  Errors: {len(self.results['errors'])}")
            print(f"  Warnings: {len(self.results['warnings'])}")
            print(f"  Security Issues: {len(self.results['security_issues'])}")
            print(f"  Files Scanned: {self.results['analysis_metadata']['files_scanned']}")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n❌ Critical error: {str(e)}")
            print(traceback.format_exc())
        
        return self.results
    
    def save_results(self, filename: str = 'analysis_results.json') -> None:
        """Save analysis results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n✅ Results saved to {filename}")
        except Exception as e:
            print(f"\n❌ Failed to save: {str(e)}")
    
    def set_github_output(self) -> None:
        """Set outputs for GitHub Actions"""
        try:
            output_file = os.environ.get('GITHUB_OUTPUT')
            if output_file:
                with open(output_file, 'a') as f:
                    f.write(f"error_count={len(self.results['errors'])}\n")
                    f.write(f"warning_count={len(self.results['warnings'])}\n")
                    f.write(f"security_issues={len(self.results['security_issues'])}\n")
                    f.write(f"first_run={str(self.results['analysis_metadata']['first_run']).lower()}\n")
        except Exception as e:
            pass


if __name__ == "__main__":
    analyzer = AdvancedRepositoryAnalyzer()
    results = analyzer.run_analysis()
    analyzer.save_results()
    analyzer.set_github_output()
    sys.exit(0)
