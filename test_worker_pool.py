"""
Test script for the worker pool system.
"""

import asyncio
import time
from datetime import datetime

from data.models import ProcessingTask, TaskType
from processors.worker_pool import WorkerPoolManager


def test_worker_pool():
    """Test the worker pool system with sample tasks."""
    print("Testing Worker Pool System")
    print("=" * 40)
    
    # Create worker pool manager
    with WorkerPoolManager(num_workers=2, checkpoint_interval=10) as pool:
        print(f"Started worker pool with {pool.num_workers} workers")
        
        # Create sample tasks
        tasks = []
        for i in range(5):
            task = ProcessingTask(
                task_id=f"test_task_{i}",
                task_type=TaskType.EXTRACT_METADATA,
                url=f"https://example.com/pdf_{i}.pdf",
                priority=i % 3 + 1,  # Priorities 1, 2, 3
                metadata={'file_path': f'/tmp/pdf_{i}.pdf'}
            )
            tasks.append(task)
        
        # Submit tasks
        submitted = pool.submit_bulk_tasks(tasks)
        print(f"Submitted {submitted} tasks")
        
        # Monitor progress
        start_time = time.time()
        while time.time() - start_time < 30:  # Wait up to 30 seconds
            stats = pool.get_statistics()
            
            print(f"\rProgress: {stats['completed_tasks']}/{len(tasks)} completed, "
                  f"{stats['failed_tasks']} failed, "
                  f"{stats['active_tasks']} active, "
                  f"{stats['pending_tasks']} pending", end="")
            
            if stats['completed_tasks'] + stats['failed_tasks'] >= len(tasks):
                break
            
            time.sleep(1)
        
        print("\n")
        
        # Final statistics
        final_stats = pool.get_statistics()
        print("Final Statistics:")
        print(f"  Total tasks processed: {final_stats['total_tasks_processed']}")
        print(f"  Completed: {final_stats['completed_tasks']}")
        print(f"  Failed: {final_stats['failed_tasks']}")
        print(f"  Average processing time: {final_stats['average_processing_time']:.3f}s")
        print(f"  Tasks per second: {final_stats['tasks_per_second']:.2f}")
        
        # Worker health
        print("\nWorker Health:")
        for worker_id, health in final_stats['worker_health'].items():
            print(f"  {worker_id}: {health['status']} "
                  f"(completed: {health['tasks_completed']}, "
                  f"uptime: {health['uptime']:.1f}s)")


if __name__ == "__main__":
    test_worker_pool()