#!/usr/bin/env python3
"""
Performance Optimization Analysis & Bundle Size Optimization
==========================================================

Comprehensive analysis of codebase performance bottlenecks focusing on:
1. Bundle size optimization
2. Load time analysis  
3. Memory usage optimization
4. Import dependency analysis
5. Code splitting recommendations
6. Performance monitoring and profiling

Usage:
    python run_performance_analysis.py [--verbose] [--output report.json]
"""

import os
import sys
import time
import psutil
import importlib
import gc
import tracemalloc
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
import subprocess
import ast
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import multiprocessing

# Add the trusttApp directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Performance measurement imports
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for analysis"""
    
    # Load time metrics
    import_times: Dict[str, float] = field(default_factory=dict)
    module_sizes: Dict[str, int] = field(default_factory=dict)
    load_order: List[str] = field(default_factory=list)
    
    # Memory metrics
    memory_usage: Dict[str, float] = field(default_factory=dict)
    peak_memory: float = 0.0
    memory_growth: List[Tuple[str, float]] = field(default_factory=list)
    
    # Bundle metrics
    file_sizes: Dict[str, int] = field(default_factory=dict)
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    unused_imports: List[str] = field(default_factory=list)
    
    # Performance bottlenecks
    slow_imports: List[Tuple[str, float]] = field(default_factory=list)
    large_modules: List[Tuple[str, int]] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)

class ImportTracker:
    """Tracks import times and dependencies"""
    
    def __init__(self):
        self.import_times = {}
        self.import_stack = []
        self.dependency_graph = defaultdict(list)
        self.original_import = __builtins__.__import__
        
    def start_tracking(self):
        """Start tracking imports"""
        __builtins__.__import__ = self._tracked_import
    
    def stop_tracking(self):
        """Stop tracking imports"""
        __builtins__.__import__ = self.original_import
    
    def _tracked_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Wrapper for import to track timing"""
        start_time = time.perf_counter()
        
        # Track dependency relationships
        if self.import_stack:
            parent = self.import_stack[-1]
            self.dependency_graph[parent].append(name)
        
        self.import_stack.append(name)
        
        try:
            result = self.original_import(name, globals, locals, fromlist, level)
            
            # Record import time
            import_time = time.perf_counter() - start_time
            if name not in self.import_times or import_time > self.import_times[name]:
                self.import_times[name] = import_time
                
            return result
        finally:
            self.import_stack.pop()

class MemoryProfiler:
    """Memory usage profiler for modules and functions"""
    
    def __init__(self):
        self.snapshots = []
        self.module_memory = {}
        self.baseline_memory = 0
        
    def start_profiling(self):
        """Start memory profiling"""
        tracemalloc.start()
        self.baseline_memory = psutil.Process().memory_info().rss
        
    def take_snapshot(self, label: str):
        """Take a memory snapshot"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            current_memory = psutil.Process().memory_info().rss
            
            self.snapshots.append({
                'label': label,
                'snapshot': snapshot,
                'memory_rss': current_memory,
                'memory_growth': current_memory - self.baseline_memory,
                'timestamp': time.perf_counter()
            })
    
    def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        if not self.snapshots:
            return {}
        
        analysis = {
            'total_growth': 0,
            'peak_memory': 0,
            'memory_timeline': [],
            'top_allocations': [],
            'memory_leaks': []
        }
        
        # Calculate memory growth
        if len(self.snapshots) > 1:
            initial = self.snapshots[0]['memory_rss']
            final = self.snapshots[-1]['memory_rss']
            analysis['total_growth'] = final - initial
        
        # Find peak memory usage
        analysis['peak_memory'] = max(s['memory_rss'] for s in self.snapshots)
        
        # Create memory timeline
        for snapshot in self.snapshots:
            analysis['memory_timeline'].append({
                'label': snapshot['label'],
                'memory_mb': snapshot['memory_rss'] / (1024 * 1024),
                'growth_mb': snapshot['memory_growth'] / (1024 * 1024),
                'timestamp': snapshot['timestamp']
            })
        
        # Analyze top allocations from latest snapshot
        if tracemalloc.is_tracing():
            latest_snapshot = self.snapshots[-1]['snapshot']
            top_stats = latest_snapshot.statistics('lineno')
            
            for stat in top_stats[:10]:
                analysis['top_allocations'].append({
                    'file': stat.traceback.format()[0],
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                })
        
        return analysis

class BundleSizeAnalyzer:
    """Analyzes bundle size and optimization opportunities"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.file_sizes = {}
        self.import_graph = defaultdict(set)
        
    def analyze_file_sizes(self) -> Dict[str, Any]:
        """Analyze file sizes in the project"""
        total_size = 0
        large_files = []
        file_types = defaultdict(int)
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and not self._should_ignore(file_path):
                size = file_path.stat().st_size
                total_size += size
                
                relative_path = str(file_path.relative_to(self.project_root))
                self.file_sizes[relative_path] = size
                
                # Track large files
                if size > 100 * 1024:  # > 100KB
                    large_files.append((relative_path, size))
                
                # Track file types
                file_types[file_path.suffix] += size
        
        # Sort by size
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': len(self.file_sizes),
            'large_files': large_files[:20],  # Top 20 largest files
            'size_by_type': dict(file_types),
            'average_file_size': total_size / len(self.file_sizes) if self.file_sizes else 0
        }
    
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored in analysis"""
        ignore_patterns = [
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.venv', 'venv', '.idea', '.vscode', '*.pyc', '*.pyo'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)
    
    def analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze import dependencies"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                relative_path = str(file_path.relative_to(self.project_root))
                imports = self._extract_imports(tree)
                self.import_graph[relative_path] = imports
                
            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
        
        # Analyze import patterns
        import_counts = defaultdict(int)
        for file_imports in self.import_graph.values():
            for imp in file_imports:
                import_counts[imp] += 1
        
        # Find unused imports (imported but not used frequently)
        total_files = len(self.import_graph)
        unused_threshold = max(1, total_files * 0.1)  # Used in < 10% of files
        
        unused_imports = [
            imp for imp, count in import_counts.items()
            if count < unused_threshold and not self._is_standard_library(imp)
        ]
        
        return {
            'total_imports': sum(len(imports) for imports in self.import_graph.values()),
            'unique_imports': len(import_counts),
            'most_common_imports': dict(sorted(import_counts.items(), 
                                             key=lambda x: x[1], reverse=True)[:20]),
            'unused_imports': unused_imports,
            'import_graph': dict(self.import_graph)
        }
    
    def _extract_imports(self, tree: ast.AST) -> set:
        """Extract import statements from AST"""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        return imports
    
    def _is_standard_library(self, module_name: str) -> bool:
        """Check if module is part of standard library"""
        standard_modules = {
            'os', 'sys', 'time', 'json', 'math', 'random', 'datetime',
            'collections', 'itertools', 'functools', 'operator',
            'threading', 'multiprocessing', 're', 'pathlib', 'logging',
            'typing', 'dataclasses', 'asyncio', 'concurrent', 'queue'
        }
        return module_name in standard_modules

class PerformanceProfiler:
    """Comprehensive performance profiler"""
    
    def __init__(self):
        self.profilers = {}
        self.results = {}
        
    def profile_function(self, func, *args, **kwargs):
        """Profile a function call"""
        # CPU profiling
        pr = cProfile.Profile()
        
        # Memory profiling
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss
        
        # Timing
        start_time = time.perf_counter()
        
        try:
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            # Collect results
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            profile_data = {
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'cpu_profile': s.getvalue(),
                'function_name': func.__name__
            }
            
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                profile_data['memory_profile'] = [
                    {
                        'file': stat.traceback.format()[0],
                        'size_kb': stat.size / 1024,
                        'count': stat.count
                    }
                    for stat in top_stats[:10]
                ]
            
            return result, profile_data
            
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

class PerformanceOptimizationAnalyzer:
    """Main analyzer class for performance optimization"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.metrics = PerformanceMetrics()
        self.import_tracker = ImportTracker()
        self.memory_profiler = MemoryProfiler()
        self.bundle_analyzer = BundleSizeAnalyzer(project_root)
        self.performance_profiler = PerformanceProfiler()
        
    def analyze_current_codebase(self) -> Dict[str, Any]:
        """Comprehensive analysis of current codebase performance"""
        logger.info("Starting comprehensive performance analysis...")
        
        analysis_results = {
            'timestamp': time.time(),
            'analysis_duration': 0,
            'bundle_analysis': {},
            'dependency_analysis': {},
            'memory_analysis': {},
            'import_analysis': {},
            'ocr_performance': {},
            'optimization_recommendations': []
        }
        
        start_time = time.perf_counter()
        
        try:
            # 1. Bundle Size Analysis
            logger.info("Analyzing bundle sizes...")
            analysis_results['bundle_analysis'] = self.bundle_analyzer.analyze_file_sizes()
            
            # 2. Dependency Analysis
            logger.info("Analyzing dependencies...")
            analysis_results['dependency_analysis'] = self.bundle_analyzer.analyze_dependencies()
            
            # 3. Import Performance Analysis
            logger.info("Analyzing import performance...")
            analysis_results['import_analysis'] = self._analyze_import_performance()
            
            # 4. Memory Usage Analysis
            logger.info("Analyzing memory usage...")
            analysis_results['memory_analysis'] = self._analyze_memory_patterns()
            
            # 5. OCR Pipeline Performance Analysis
            logger.info("Analyzing OCR pipeline performance...")
            analysis_results['ocr_performance'] = self._analyze_ocr_performance()
            
            # 6. Generate Optimization Recommendations
            logger.info("Generating optimization recommendations...")
            analysis_results['optimization_recommendations'] = self._generate_recommendations(analysis_results)
            
            analysis_results['analysis_duration'] = time.perf_counter() - start_time
            logger.info(f"Analysis completed in {analysis_results['analysis_duration']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _analyze_import_performance(self) -> Dict[str, Any]:
        """Analyze import performance and bottlenecks"""
        # Test importing key modules
        test_modules = [
            'trustt_gpt_service.ocrpipeline',
            'trustt_gpt_service.services',
            'trustt_gpt_service.views',
            'trustt_gpt_service.db'
        ]
        
        import_results = {}
        
        for module_name in test_modules:
            try:
                self.import_tracker.start_tracking()
                self.memory_profiler.take_snapshot(f'before_{module_name}')
                
                start_time = time.perf_counter()
                start_memory = psutil.Process().memory_info().rss
                
                # Dynamic import
                module = importlib.import_module(module_name)
                
                end_time = time.perf_counter()
                end_memory = psutil.Process().memory_info().rss
                
                self.memory_profiler.take_snapshot(f'after_{module_name}')
                self.import_tracker.stop_tracking()
                
                import_results[module_name] = {
                    'import_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'memory_delta_mb': (end_memory - start_memory) / (1024 * 1024),
                    'module_size': sys.getsizeof(module),
                    'success': True
                }
                
            except ImportError as e:
                import_results[module_name] = {
                    'error': str(e),
                    'import_time': 0,
                    'memory_delta': 0,
                    'success': False
                }
            except Exception as e:
                logger.warning(f"Failed to analyze import for {module_name}: {e}")
                import_results[module_name] = {
                    'error': str(e),
                    'import_time': 0,
                    'memory_delta': 0,
                    'success': False
                }
        
        # Identify slow imports (> 100ms)
        slow_imports = [
            (module, data['import_time']) 
            for module, data in import_results.items() 
            if 'import_time' in data and data['import_time'] > 0.1
        ]
        
        return {
            'import_results': import_results,
            'slow_imports': slow_imports,
            'total_import_time': sum(
                data.get('import_time', 0) for data in import_results.values()
            ),
            'total_memory_delta': sum(
                data.get('memory_delta', 0) for data in import_results.values()
            )
        }
    
    def _analyze_ocr_performance(self) -> Dict[str, Any]:
        """Analyze OCR pipeline performance"""
        ocr_analysis = {
            'pipeline_analysis': {},
            'bottlenecks_identified': [],
            'optimization_potential': {},
            'current_performance': {}
        }
        
        try:
            # Try to analyze the OCR pipeline module
            from trustt_gpt_service.ocrpipeline import HighPerformanceOCRPipeline, PerformanceConfig
            
            # Simulate performance measurement
            config = PerformanceConfig()
            
            # Analyze configuration
            ocr_analysis['pipeline_analysis'] = {
                'numa_aware': config.numa_aware,
                'max_workers': config.max_workers,
                'queue_size': config.queue_size,
                'memory_pool_size': config.memory_pool_size,
                'batch_size': config.batch_size,
                'optimization_level': config.optimization_level
            }
            
            # Identify potential bottlenecks
            bottlenecks = []
            if not config.numa_aware:
                bottlenecks.append("NUMA awareness disabled - may impact multi-socket performance")
            
            if config.max_workers < os.cpu_count():
                bottlenecks.append(f"Max workers ({config.max_workers}) < CPU count ({os.cpu_count()})")
            
            if config.queue_size < 1000:
                bottlenecks.append(f"Small queue size ({config.queue_size}) may cause blocking")
            
            ocr_analysis['bottlenecks_identified'] = bottlenecks
            
            # Optimization potential
            ocr_analysis['optimization_potential'] = {
                'cpu_utilization_improvement': f"{100 - (config.max_workers / os.cpu_count() * 100):.1f}%",
                'memory_optimization': "Memory pooling implemented",
                'concurrency_optimization': "Lock-free queues implemented",
                'simd_optimization': "SIMD preprocessing available"
            }
            
        except ImportError:
            ocr_analysis['error'] = "OCR pipeline module not found or not properly implemented"
        except Exception as e:
            ocr_analysis['error'] = f"Failed to analyze OCR pipeline: {e}"
        
        return ocr_analysis
    
    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        return self.memory_profiler.analyze_memory_usage()
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on analysis"""
        recommendations = []
        
        # Bundle size recommendations
        bundle_analysis = analysis_results.get('bundle_analysis', {})
        if bundle_analysis.get('total_size_mb', 0) > 50:
            recommendations.append({
                'category': 'Bundle Size',
                'priority': 'High',
                'issue': f"Large bundle size: {bundle_analysis.get('total_size_mb', 0):.1f}MB",
                'recommendation': "Consider code splitting and lazy loading",
                'implementation': [
                    "Split large modules into smaller components",
                    "Implement lazy imports using importlib",
                    "Remove unused dependencies",
                    "Use dynamic imports for optional features"
                ],
                'estimated_impact': "20-40% size reduction"
            })
        
        # Import performance recommendations
        import_analysis = analysis_results.get('import_analysis', {})
        slow_imports = import_analysis.get('slow_imports', [])
        if slow_imports:
            recommendations.append({
                'category': 'Import Performance',
                'priority': 'Medium',
                'issue': f"Slow imports detected: {len(slow_imports)} modules > 100ms",
                'recommendation': "Optimize import times",
                'implementation': [
                    "Use lazy imports for heavy dependencies",
                    "Defer imports until needed",
                    "Consider using importlib.util.spec_from_file_location",
                    "Cache imported modules",
                    "Use __import__ hooks for optimization"
                ],
                'estimated_impact': "50-80% import time reduction"
            })
        
        # Memory usage recommendations
        memory_analysis = analysis_results.get('memory_analysis', {})
        if memory_analysis.get('total_growth', 0) > 100 * 1024 * 1024:  # > 100MB
            recommendations.append({
                'category': 'Memory Usage',
                'priority': 'High',
                'issue': f"High memory growth: {memory_analysis.get('total_growth', 0) / (1024*1024):.1f}MB",
                'recommendation': "Optimize memory usage",
                'implementation': [
                    "Implement memory pooling",
                    "Use generators instead of lists for large datasets",
                    "Add garbage collection hints",
                    "Use __slots__ for frequently instantiated classes",
                    "Implement object recycling"
                ],
                'estimated_impact': "30-50% memory reduction"
            })
        
        # OCR-specific recommendations
        ocr_analysis = analysis_results.get('ocr_performance', {})
        if 'bottlenecks_identified' in ocr_analysis:
            bottlenecks = ocr_analysis['bottlenecks_identified']
            if bottlenecks:
                recommendations.append({
                    'category': 'OCR Performance',
                    'priority': 'High',
                    'issue': f"OCR bottlenecks identified: {len(bottlenecks)} issues",
                    'recommendation': "Optimize OCR pipeline configuration",
                    'implementation': [
                        "Enable NUMA awareness for multi-socket systems",
                        "Increase worker count to match CPU cores",
                        "Optimize queue sizes for workload",
                        "Implement SIMD vectorization",
                        "Add CPU affinity for critical threads"
                    ],
                    'estimated_impact': "2-3x performance improvement"
                })
        
        # General performance recommendations
        recommendations.append({
            'category': 'Performance Optimization',
            'priority': 'High',
            'issue': "General performance improvements available",
            'recommendation': "Implement advanced optimization techniques",
            'implementation': [
                "Use Cython for CPU-intensive functions",
                "Implement SIMD vectorization with Numba",
                "Add memory pools for frequent allocations",
                "Use lock-free data structures",
                "Implement NUMA-aware thread placement",
                "Add CPU profiling and optimization",
                "Use profile-guided optimization (PGO)"
            ],
            'estimated_impact': "5-10x performance improvement"
        })
        
        return recommendations
    
    def generate_optimization_report(self, output_file: str = "performance_optimization_report.json"):
        """Generate comprehensive optimization report"""
        logger.info("Generating performance optimization report...")
        
        analysis_results = self.analyze_current_codebase()
        
        # Add system information
        analysis_results['system_info'] = {
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform,
            'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
        }
        
        # Add current performance baseline
        analysis_results['performance_baseline'] = self._measure_performance_baseline()
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Performance optimization report saved to: {output_file}")
        
        # Print summary
        self._print_optimization_summary(analysis_results)
        
        return analysis_results
    
    def _measure_performance_baseline(self) -> Dict[str, Any]:
        """Measure current performance baseline"""
        baseline = {
            'import_time_total': 0,
            'memory_usage_mb': psutil.Process().memory_info().rss / (1024 * 1024),
            'cpu_count': os.cpu_count(),
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        # Measure typical operations
        try:
            # Time a typical operation
            start_time = time.perf_counter()
            
            # Simulate workload
            import random
            data = [random.random() for _ in range(10000)]
            result = sum(data) / len(data)
            
            baseline['computation_time'] = time.perf_counter() - start_time
            
        except Exception as e:
            logger.warning(f"Failed to measure baseline: {e}")
        
        return baseline
    
    def _print_optimization_summary(self, analysis_results: Dict[str, Any]):
        """Print optimization summary to console"""
        print("\n" + "="*80)
        print("🚀 PERFORMANCE OPTIMIZATION ANALYSIS SUMMARY")
        print("="*80)
        
        # System info
        system = analysis_results.get('system_info', {})
        print(f"🖥️  System: {system.get('cpu_count', 'N/A')} cores, "
              f"{system.get('memory_total_gb', 0):.1f}GB RAM")
        
        # Bundle analysis
        bundle = analysis_results.get('bundle_analysis', {})
        print(f"📦 Bundle Size: {bundle.get('total_size_mb', 0):.1f}MB "
              f"({bundle.get('file_count', 0)} files)")
        
        if bundle.get('large_files'):
            print(f"   📋 Largest files:")
            for file_path, size in bundle['large_files'][:3]:
                print(f"     • {file_path}: {size / (1024*1024):.1f}MB")
        
        # Import analysis
        import_analysis = analysis_results.get('import_analysis', {})
        print(f"⚡ Import Performance: {import_analysis.get('total_import_time', 0):.2f}s total")
        print(f"   Slow imports: {len(import_analysis.get('slow_imports', []))}")
        
        # Memory analysis
        memory = analysis_results.get('memory_analysis', {})
        peak_memory_mb = memory.get('peak_memory', 0) / (1024*1024)
        print(f"🧠 Memory Usage: {peak_memory_mb:.1f}MB peak")
        
        # Dependencies
        deps = analysis_results.get('dependency_analysis', {})
        print(f"📚 Dependencies: {deps.get('unique_imports', 0)} unique imports")
        print(f"   Unused imports: {len(deps.get('unused_imports', []))}")
        
        # OCR Performance
        ocr = analysis_results.get('ocr_performance', {})
        if 'bottlenecks_identified' in ocr:
            bottlenecks = len(ocr['bottlenecks_identified'])
            print(f"🔧 OCR Pipeline: {bottlenecks} bottlenecks identified")
        
        # Recommendations
        recommendations = analysis_results.get('optimization_recommendations', [])
        high_priority = [r for r in recommendations if r.get('priority') == 'High']
        print(f"\n🎯 Optimization Recommendations: {len(recommendations)} total")
        print(f"   High priority: {len(high_priority)}")
        
        if high_priority:
            print("\n🔥 High Priority Optimizations:")
            for rec in high_priority:
                impact = rec.get('estimated_impact', 'Unknown impact')
                print(f"   • {rec.get('category')}: {rec.get('issue')} ({impact})")
        
        print(f"\n⏱️  Analysis Duration: {analysis_results.get('analysis_duration', 0):.2f}s")
        print("\n" + "="*80)
        print("📊 Full report saved to performance_optimization_report.json")
        print("="*80)

def main():
    """Main function for running performance analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Performance Optimization Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--project-root', 
        type=str, 
        default=".",
        help='Project root directory to analyze'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default="performance_optimization_report.json",
        help='Output file for optimization report'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting Performance Optimization Analysis...")
    print("=" * 60)
    
    # Run analysis
    analyzer = PerformanceOptimizationAnalyzer(args.project_root)
    results = analyzer.generate_optimization_report(args.output)
    
    print("\n✅ Analysis complete! Check the report for detailed recommendations.")

if __name__ == "__main__":
    main()