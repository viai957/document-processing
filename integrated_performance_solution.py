"""
Integrated High-Performance OCR Solution
======================================

Complete integration of:
1. High-Performance OCR Pipeline with MLflow Monitoring
2. Intelligent Markdown-to-JSON Conversion System
3. NVIDIA-Style Performance Dashboard

Target Performance: 30 pages per 10 seconds (3 pages/second sustained)
Features: Real-time monitoring, extreme optimization, executive analytics
"""

import asyncio
import logging
import threading
import time
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import argparse
import json

# Import our high-performance components
from high_performance_ocr_pipeline import (
    HighPerformanceOCRPipeline, 
    PerformanceConfig, 
    PerformanceMonitor
)
from intelligent_markdown_to_json_converter import (
    IntelligentMarkdownToJsonConverter,
    ConversionConfig
)
from nvidia_style_performance_dashboard import (
    NVIDIAStyleDashboard,
    DashboardConfig,
    DashboardIntegration
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ocr_performance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedSolutionConfig:
    """Configuration for the complete integrated solution"""
    
    # Performance Targets
    target_pages_per_second: float = 3.0
    target_conversion_time_ms: float = 1.0
    
    # OCR Pipeline Configuration
    ocr_workers: int = 8
    ocr_dpi: int = 300
    enable_numa: bool = True
    enable_simd: bool = True
    memory_pool_size_mb: int = 512
    
    # JSON Conversion Configuration
    enable_ml_classification: bool = True
    enable_template_matching: bool = True
    cache_size_mb: int = 256
    
    # Dashboard Configuration
    dashboard_port: int = 8050
    dashboard_host: str = "0.0.0.0"
    real_time_updates: bool = True
    
    # Integration Settings
    enable_dashboard: bool = True
    enable_mlflow: bool = True
    output_directory: str = "ocr_output"
    
    # Performance Monitoring
    collect_detailed_metrics: bool = True
    performance_report_interval: int = 30  # seconds

class IntegratedHighPerformanceOCRSolution:
    """
    Complete high-performance OCR solution with real-time monitoring
    
    Architecture:
    - High-Performance OCR Pipeline (CPU-optimized, NUMA-aware)
    - Intelligent Markdown-to-JSON Converter (sub-millisecond conversion)
    - Real-time Performance Dashboard (NVIDIA-style analytics)
    - MLflow Integration for experiment tracking
    """
    
    def __init__(self, config: IntegratedSolutionConfig = None):
        self.config = config or IntegratedSolutionConfig()
        
        # Initialize components
        self.ocr_pipeline = None
        self.json_converter = None
        self.dashboard = None
        self.dashboard_integration = None
        
        # Performance tracking
        self.total_documents_processed = 0
        self.total_pages_processed = 0
        self.start_time = time.time()
        self.processing_times = []
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("Integrated High-Performance OCR Solution initialized")
    
    def initialize_components(self):
        """Initialize all solution components"""
        logger.info("Initializing high-performance components...")
        
        # 1. Initialize OCR Pipeline
        ocr_config = PerformanceConfig(
            target_pages_per_second=self.config.target_pages_per_second,
            numa_aware=self.config.enable_numa,
            simd_optimization=self.config.enable_simd,
            ocr_workers=self.config.ocr_workers,
            memory_pool_size=self.config.memory_pool_size_mb * 1024 * 1024,
            dpi=self.config.ocr_dpi,
            performance_logging_enabled=self.config.collect_detailed_metrics
        )
        
        self.ocr_pipeline = HighPerformanceOCRPipeline(ocr_config)
        logger.info("✓ OCR Pipeline initialized")
        
        # 2. Initialize JSON Converter
        conversion_config = ConversionConfig(
            target_conversion_time_ms=self.config.target_conversion_time_ms,
            enable_document_classification=self.config.enable_ml_classification,
            template_compilation=self.config.enable_template_matching,
            cache_size_mb=self.config.cache_size_mb
        )
        
        self.json_converter = IntelligentMarkdownToJsonConverter(conversion_config)
        logger.info("✓ JSON Converter initialized")
        
        # 3. Initialize Dashboard (if enabled)
        if self.config.enable_dashboard:
            dashboard_config = DashboardConfig(
                port=self.config.dashboard_port,
                host=self.config.dashboard_host,
                target_throughput=self.config.target_pages_per_second,
                auto_refresh=self.config.real_time_updates
            )
            
            self.dashboard = NVIDIAStyleDashboard(dashboard_config)
            self.dashboard_integration = DashboardIntegration(self.dashboard)
            logger.info("✓ Performance Dashboard initialized")
        
        logger.info("All components initialized successfully")
    
    async def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a single document through the complete pipeline
        
        Flow: PDF → OCR → Markdown → JSON → Performance Metrics
        """
        start_time = time.perf_counter()
        logger.info(f"Processing document: {document_path}")
        
        try:
            # Stage 1: High-Performance OCR Processing
            logger.info("Stage 1: OCR Processing")
            ocr_start = time.perf_counter()
            
            ocr_result = await self.ocr_pipeline.process_document_async(document_path)
            
            ocr_time = time.perf_counter() - ocr_start
            logger.info(f"OCR completed in {ocr_time:.2f}s")
            
            if 'error' in ocr_result:
                logger.error(f"OCR failed: {ocr_result['error']}")
                return ocr_result
            
            # Stage 2: Intelligent JSON Conversion
            logger.info("Stage 2: JSON Conversion")
            conversion_start = time.perf_counter()
            
            # Process each page's markdown
            converted_pages = []
            for page_result in ocr_result.get('results', []):
                if 'raw_text' in page_result:
                    json_result = await self.json_converter.convert_async(
                        page_result['raw_text'], 
                        document_type='bank_statement'
                    )
                    converted_pages.append(json_result)
            
            conversion_time = time.perf_counter() - conversion_start
            logger.info(f"JSON conversion completed in {conversion_time:.2f}s")
            
            # Stage 3: Aggregate results and metrics
            total_time = time.perf_counter() - start_time
            
            # Update statistics
            self.total_documents_processed += 1
            self.total_pages_processed += ocr_result.get('completed_pages', 0)
            self.processing_times.append(total_time)
            
            # Create comprehensive result
            final_result = {
                'document_path': document_path,
                'processing_summary': {
                    'total_time': total_time,
                    'ocr_time': ocr_time,
                    'conversion_time': conversion_time,
                    'pages_processed': ocr_result.get('completed_pages', 0),
                    'target_met': ocr_result.get('target_met', False),
                    'throughput': ocr_result.get('pages_per_second', 0)
                },
                'ocr_results': ocr_result,
                'json_results': converted_pages,
                'performance_metrics': self._calculate_performance_metrics(),
                'timestamp': time.time()
            }
            
            # Save results
            output_file = self._save_results(final_result)
            final_result['output_file'] = output_file
            
            # Update dashboard if enabled
            if self.dashboard_integration:
                await self._update_dashboard_metrics(final_result)
            
            logger.info(f"Document processing completed: {total_time:.2f}s total")
            return final_result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                'error': str(e),
                'document_path': document_path,
                'processing_time': time.perf_counter() - start_time
            }
    
    async def process_batch(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents in parallel
        
        Target: Achieve maximum throughput while maintaining quality
        """
        logger.info(f"Processing batch of {len(document_paths)} documents")
        batch_start = time.perf_counter()
        
        # Process documents concurrently
        tasks = [self.process_document(path) for path in document_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate batch results
        batch_time = time.perf_counter() - batch_start
        successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        failed_results = [r for r in results if isinstance(r, dict) and 'error' in r]
        
        total_pages = sum(r.get('processing_summary', {}).get('pages_processed', 0) 
                         for r in successful_results)
        
        batch_throughput = total_pages / batch_time if batch_time > 0 else 0
        
        batch_summary = {
            'batch_size': len(document_paths),
            'successful_documents': len(successful_results),
            'failed_documents': len(failed_results),
            'total_pages_processed': total_pages,
            'batch_processing_time': batch_time,
            'batch_throughput': batch_throughput,
            'target_throughput': self.config.target_pages_per_second,
            'performance_target_met': batch_throughput >= self.config.target_pages_per_second,
            'results': results,
            'timestamp': time.time()
        }
        
        logger.info(f"Batch completed: {batch_throughput:.2f} pages/sec "
                   f"(target: {self.config.target_pages_per_second})")
        
        return batch_summary
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        elapsed_time = time.time() - self.start_time
        
        metrics = {
            'uptime_seconds': elapsed_time,
            'total_documents_processed': self.total_documents_processed,
            'total_pages_processed': self.total_pages_processed,
            'average_throughput': self.total_pages_processed / max(elapsed_time, 0.001),
            'target_throughput': self.config.target_pages_per_second,
            'performance_ratio': 0.0,
            'processing_times': {
                'count': len(self.processing_times),
                'average': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        }
        
        # Calculate performance ratio
        if metrics['average_throughput'] > 0:
            metrics['performance_ratio'] = metrics['average_throughput'] / self.config.target_pages_per_second
        
        # Processing time statistics
        if self.processing_times:
            import statistics
            metrics['processing_times'].update({
                'average': statistics.mean(self.processing_times),
                'min': min(self.processing_times),
                'max': max(self.processing_times),
                'median': statistics.median(self.processing_times)
            })
        
        return metrics
    
    def _save_results(self, result: Dict[str, Any]) -> str:
        """Save processing results to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        doc_name = Path(result['document_path']).stem
        output_file = os.path.join(
            self.config.output_directory, 
            f"{doc_name}_processed_{timestamp}.json"
        )
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Results saved to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return ""
    
    async def _update_dashboard_metrics(self, result: Dict[str, Any]):
        """Update dashboard with latest processing metrics"""
        if not self.dashboard_integration:
            return
        
        try:
            processing_summary = result.get('processing_summary', {})
            
            # Extract stage timings from OCR results
            stage_timings = {}
            ocr_results = result.get('ocr_results', {})
            if 'performance_summary' in ocr_results:
                stage_timings = ocr_results['performance_summary'].get('stage_timings', {})
            
            # Update dashboard
            self.dashboard.update_pipeline_data(
                pages_processed=self.total_pages_processed,
                stage_timings=stage_timings,
                queue_depths={}  # Could extract from pipeline if needed
            )
            
        except Exception as e:
            logger.warning(f"Dashboard update failed: {e}")
    
    def start_dashboard_server(self):
        """Start the dashboard server in a separate thread"""
        if not self.config.enable_dashboard or not self.dashboard:
            return
        
        def run_dashboard():
            try:
                self.dashboard.start_dashboard()
            except Exception as e:
                logger.error(f"Dashboard server error: {e}")
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        logger.info(f"Dashboard server started on http://{self.config.dashboard_host}:{self.config.dashboard_port}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down integrated solution...")
        
        # Stop dashboard integration
        if self.dashboard_integration:
            self.dashboard_integration.stop_integration()
        
        # Shutdown OCR pipeline
        if self.ocr_pipeline:
            self.ocr_pipeline.shutdown()
        
        # Shutdown JSON converter
        if self.json_converter:
            self.json_converter.shutdown()
        
        # Stop dashboard
        if self.dashboard:
            self.dashboard.stop_dashboard()
        
        # Print final performance summary
        self._print_final_summary()
        
        logger.info("Shutdown complete")
    
    def _print_final_summary(self):
        """Print comprehensive performance summary"""
        metrics = self._calculate_performance_metrics()
        
        print("\n" + "="*80)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Total Runtime: {metrics['uptime_seconds']:.2f} seconds")
        print(f"Documents Processed: {metrics['total_documents_processed']}")
        print(f"Pages Processed: {metrics['total_pages_processed']}")
        print(f"Average Throughput: {metrics['average_throughput']:.2f} pages/second")
        print(f"Target Throughput: {metrics['target_throughput']} pages/second")
        print(f"Performance Ratio: {metrics['performance_ratio']:.2f}x")
        print(f"Target Achievement: {'✓ MET' if metrics['performance_ratio'] >= 1.0 else '✗ MISSED'}")
        
        if metrics['processing_times']['count'] > 0:
            print(f"\nProcessing Time Statistics:")
            print(f"  Average: {metrics['processing_times']['average']:.2f}s")
            print(f"  Minimum: {metrics['processing_times']['min']:.2f}s")
            print(f"  Maximum: {metrics['processing_times']['max']:.2f}s")
            print(f"  Median: {metrics['processing_times'].get('median', 0):.2f}s")
        
        print("="*80)

# Command Line Interface
def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="High-Performance OCR Pipeline with Real-Time Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single document
  python integrated_performance_solution.py --document document.pdf
  
  # Process multiple documents
  python integrated_performance_solution.py --batch doc1.pdf doc2.pdf doc3.pdf
  
  # Run with custom configuration
  python integrated_performance_solution.py --document document.pdf --workers 16 --target-throughput 5.0
  
  # Start dashboard only
  python integrated_performance_solution.py --dashboard-only --port 8080
        """
    )
    
    # Input options
    parser.add_argument('--document', type=str, help='Single document to process')
    parser.add_argument('--batch', nargs='+', help='Multiple documents to process')
    parser.add_argument('--input-dir', type=str, help='Directory containing documents to process')
    
    # Performance configuration
    parser.add_argument('--workers', type=int, default=8, help='Number of OCR workers')
    parser.add_argument('--target-throughput', type=float, default=3.0, help='Target pages per second')
    parser.add_argument('--dpi', type=int, default=300, help='OCR DPI setting')
    parser.add_argument('--memory-pool', type=int, default=512, help='Memory pool size in MB')
    
    # Feature toggles
    parser.add_argument('--no-numa', action='store_true', help='Disable NUMA optimization')
    parser.add_argument('--no-simd', action='store_true', help='Disable SIMD optimization')
    parser.add_argument('--no-dashboard', action='store_true', help='Disable dashboard')
    parser.add_argument('--no-mlflow', action='store_true', help='Disable MLflow tracking')
    
    # Dashboard configuration
    parser.add_argument('--dashboard-only', action='store_true', help='Start dashboard server only')
    parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Dashboard host')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='ocr_output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser

async def main():
    """Main async function"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = IntegratedSolutionConfig(
        target_pages_per_second=args.target_throughput,
        ocr_workers=args.workers,
        ocr_dpi=args.dpi,
        enable_numa=not args.no_numa,
        enable_simd=not args.no_simd,
        memory_pool_size_mb=args.memory_pool,
        enable_dashboard=not args.no_dashboard,
        enable_mlflow=not args.no_mlflow,
        dashboard_port=args.port,
        dashboard_host=args.host,
        output_directory=args.output_dir
    )
    
    # Initialize solution
    solution = IntegratedHighPerformanceOCRSolution(config)
    
    try:
        # Dashboard-only mode
        if args.dashboard_only:
            solution.config.enable_dashboard = True
            solution.initialize_components()
            solution.start_dashboard_server()
            
            print(f"Dashboard running at http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop...")
            
            # Keep running until interrupted
            while not solution.shutdown_requested:
                await asyncio.sleep(1)
            
            return
        
        # Initialize all components
        solution.initialize_components()
        
        # Start dashboard server
        solution.start_dashboard_server()
        
        # Determine documents to process
        documents_to_process = []
        
        if args.document:
            documents_to_process = [args.document]
        elif args.batch:
            documents_to_process = args.batch
        elif args.input_dir:
            # Find all PDF files in directory
            input_path = Path(args.input_dir)
            if input_path.exists():
                documents_to_process = list(input_path.glob("*.pdf"))
                documents_to_process.extend(input_path.glob("*.png"))
                documents_to_process.extend(input_path.glob("*.jpg"))
                documents_to_process = [str(p) for p in documents_to_process]
        
        if not documents_to_process:
            parser.error("No documents specified. Use --document, --batch, or --input-dir")
        
        # Process documents
        logger.info(f"Processing {len(documents_to_process)} documents")
        
        if len(documents_to_process) == 1:
            # Single document processing
            result = await solution.process_document(documents_to_process[0])
            print(f"\nProcessing completed: {result.get('output_file', 'No output file')}")
        else:
            # Batch processing
            batch_result = await solution.process_batch(documents_to_process)
            
            print(f"\nBatch processing completed:")
            print(f"  Successful: {batch_result['successful_documents']}")
            print(f"  Failed: {batch_result['failed_documents']}")
            print(f"  Throughput: {batch_result['batch_throughput']:.2f} pages/sec")
            print(f"  Target met: {'✓' if batch_result['performance_target_met'] else '✗'}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        solution.shutdown()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())