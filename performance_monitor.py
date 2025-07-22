import time
import statistics
from functools import wraps
from typing import Dict, List, Any
import json
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "pdf_processing": [],
            "text_chunking": [],
            "vector_store_creation": [],
            "vector_store_loading": [],
            "similarity_search": [],
            "llm_inference": [],
            "api_endpoints": {}
        }
    
    def timing_decorator(self, operation_name: str):
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds
                    
                    if operation_name not in self.metrics:
                        self.metrics[operation_name] = []
                    self.metrics[operation_name].append(latency)
                    
                    print(f"[PERFORMANCE] {operation_name}: {latency:.2f}ms")
                    return result
                except Exception as e:
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000
                    print(f"[PERFORMANCE] {operation_name} ERROR: {latency:.2f}ms - {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get aggregated performance metrics"""
        summary = {}
        for operation, latencies in self.metrics.items():
            if latencies:
                summary[operation] = {
                    "count": len(latencies),
                    "avg_ms": round(statistics.mean(latencies), 2),
                    "median_ms": round(statistics.median(latencies), 2),
                    "min_ms": round(min(latencies), 2),
                    "max_ms": round(max(latencies), 2),
                    "p95_ms": round(statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0], 2),
                    "std_dev_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2)
                }
        return summary
    
    def save_metrics_to_file(self, filename: str = None):
        """Save metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_metrics_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_metrics_summary(),
                "raw_data": self.metrics
            }, f, indent=2)
        print(f"Metrics saved to {filename}")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()