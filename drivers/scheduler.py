#!/usr/bin/env python3
"""
Multithreaded Systematic Search Scheduler
Fast, scalable scheduler using parallel systematic search instead of ILP optimization.
"""

import json
import os
import subprocess
import time
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import signal
import sys
import re
import csv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from itertools import combinations, product
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# For fast systematic search
import random
from functools import partial

@dataclass
class ReferenceProfile:
    """Memory and operation profile from reference run"""
    op_to_memory: Dict[int, float]  # op_id -> memory_mb
    op_to_time: Dict[int, float]    # op_id -> relative_time_seconds
    total_ops: int
    total_time: float
    peak_memory: float
    memory_curve: List[float]  # memory at each operation
    single_period_ops: int = 0  # Operations in single period (for scheduling constraints)
    
    @classmethod
    def from_files(cls, log_file: str, resource_file: str, num_periods: int = 3):
        """Parse reference profile from LOG_CT output and ResourceMonitor CSV"""
        op_to_memory = {}
        op_to_time = {}
        
        # Parse LOG_CT operations and extract operation numbers
        log_ops = cls._parse_log_ct_file(log_file)
        
        # Parse ResourceMonitor CSV for memory over time
        memory_timeline = cls._parse_resource_monitor(resource_file)
        
        # Correlate operations with memory usage
        single_op_to_memory, single_op_to_time = cls._correlate_ops_with_memory(log_ops, memory_timeline)
        
        if not single_op_to_memory:
            print("Warning: No operations found, creating dummy profile")
            # Create a simple dummy profile for testing
            total_ops = 1000
            for i in range(total_ops):
                single_op_to_memory[i] = 100 + 50 * np.sin(i / 100)  # Dummy memory curve
                single_op_to_time[i] = i * 0.1
        
        single_period_ops = max(single_op_to_memory.keys()) if single_op_to_memory else 0
        single_period_time = max(single_op_to_time.values()) if single_op_to_time else 0
        single_peak_memory = max(single_op_to_memory.values()) if single_op_to_memory else 0
        
        # Build single period memory curve
        single_memory_curve = [single_op_to_memory.get(i, 0) for i in range(single_period_ops + 1)]
        
        print(f"Single period: {single_period_ops} ops, {single_period_time:.1f}s, {single_peak_memory:.1f}MB peak")
        
        # Extend to multiple periods for realistic scheduling
        extended_memory_curve = []
        extended_op_to_memory = {}
        extended_op_to_time = {}
        
        for period in range(num_periods):
            period_offset = period * single_period_ops
            time_offset = period * single_period_time
            
            for local_op in range(single_period_ops):
                global_op = period_offset + local_op
                extended_op_to_memory[global_op] = single_op_to_memory.get(local_op, 0)
                extended_op_to_time[global_op] = single_op_to_time.get(local_op, 0) + time_offset
            
            extended_memory_curve.extend(single_memory_curve)
        
        total_ops = len(extended_memory_curve) - 1
        total_time = num_periods * single_period_time
        
        print(f"Extended to {num_periods} periods: {total_ops} ops, {total_time:.1f}s total")
        
        return cls(
            op_to_memory=extended_op_to_memory,
            op_to_time=extended_op_to_time,
            total_ops=total_ops,
            total_time=total_time,
            peak_memory=single_peak_memory,
            memory_curve=extended_memory_curve,
            single_period_ops=single_period_ops  # Store for scheduling constraints
        )
    
    @staticmethod
    def _parse_log_ct_file(log_file: str) -> List[Tuple[int, float]]:
        """Extract (op_number, timestamp) from LOG_CT prints"""
        ops = []
        if not os.path.exists(log_file):
            print(f"Warning: Log file {log_file} not found")
            return ops
            
        with open(log_file, 'r') as f:
            op_count = 0
            for line in f:
                # Look for your new LOG_CT format: LOG_CT:ct23:TIME:0.369717:LINE:413
                if 'LOG_CT:' in line and ':TIME:' in line:
                    # Extract timestamp
                    time_match = re.search(r':TIME:([\d.]+):', line)
                    if time_match:
                        timestamp = float(time_match.group(1))
                        op_count += 1  # Sequential operation counter
                        ops.append((op_count, timestamp))
        
        print(f"Found {len(ops)} LOG_CT operations")
        return ops  # Already in sequential order
    
    @staticmethod
    def _parse_resource_monitor(resource_file: str) -> List[Tuple[float, float]]:
        """Parse ResourceMonitor CSV: (timestamp, memory_gb)"""
        timeline = []
        if not os.path.exists(resource_file):
            print(f"Warning: Resource file {resource_file} not found")
            return timeline
            
        try:
            with open(resource_file, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                print(f"Resource file headers: {headers}")
                
                for row_num, row in enumerate(reader):
                    try:
                        timestamp = float(row['Time'])
                        # Use RAM_Active_GB or RAM_Used_GB
                        memory_gb = float(row.get('RAM_Active_GB', row.get('RAM_Used_GB', 0)))
                        memory_mb = memory_gb * 1024
                        timeline.append((timestamp, memory_mb))
                    except (ValueError, KeyError) as e:
                        if row_num < 5:  # Only print first few errors
                            print(f"Warning: Could not parse row {row_num}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading resource file {resource_file}: {e}")
            return timeline
        
        print(f"Found {len(timeline)} memory measurements in resource file")
        return timeline
    
    @staticmethod
    def _correlate_ops_with_memory(ops: List[Tuple[int, float]], 
                                  memory_timeline: List[Tuple[float, float]]) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Correlate operations with memory usage by timestamp"""
        op_to_memory = {}
        op_to_time = {}
        
        if not memory_timeline:
            print("Warning: No memory timeline data, using fallback")
            # Fallback: assign increasing memory pattern
            for i, (op_num, timestamp) in enumerate(ops):
                op_to_memory[op_num] = 100 + (i % 100) * 2  # Simple pattern
                op_to_time[op_num] = timestamp
            return op_to_memory, op_to_time
        
        if not ops:
            print("Warning: No operations found")
            return op_to_memory, op_to_time
        
        # Sort memory timeline by timestamp
        memory_timeline.sort(key=lambda x: x[0])
        print(f"Memory timeline spans {memory_timeline[0][0]:.1f}s to {memory_timeline[-1][0]:.1f}s")
        
        for op_num, op_timestamp in ops:
            op_to_time[op_num] = op_timestamp
            
            # Find closest memory measurement
            closest_memory = memory_timeline[0][1]  # Default to first measurement
            min_time_diff = float('inf')
            
            for mem_timestamp, memory_mb in memory_timeline:
                time_diff = abs(mem_timestamp - op_timestamp)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_memory = memory_mb
            
            op_to_memory[op_num] = closest_memory
        
        print(f"Correlated {len(ops)} operations with memory data")
        return op_to_memory, op_to_time

@dataclass
class SystematicSearchResult:
    """Result of systematic search scheduling"""
    process_starts: List[int]  # Start operation for each process
    peak_memory: float
    total_processes: int
    schedule_duration: int  # In operations
    memory_timeline: List[float]  # Total memory at each operation
    search_time: float
    configurations_tested: int
    search_method: str

class SystematicScheduler:
    """Fast systematic search scheduler with multithreading"""
    
    def __init__(self, reference_profile: ReferenceProfile, memory_limit: float = None):
        self.ref_profile = reference_profile
        self.memory_limit = memory_limit or (reference_profile.peak_memory * 10)
        self.num_cores = multiprocessing.cpu_count()
        
    def find_optimal_schedule(self, n_processes: int, search_method: str = "smart_grid", 
                            time_limit: float = 300) -> SystematicSearchResult:
        """Find optimal schedule using systematic search"""
        
        print(f"Starting systematic search for {n_processes} processes using {search_method}")
        start_time = time.time()
        
        if search_method == "smart_grid":
            result = self._smart_grid_search(n_processes, time_limit)
        elif search_method == "memory_valley":
            result = self._memory_valley_search(n_processes, time_limit)
        elif search_method == "monte_carlo":
            result = self._monte_carlo_search(n_processes, time_limit)
        else:
            raise ValueError(f"Unknown search method: {search_method}")
        
        result.search_time = time.time() - start_time
        result.search_method = search_method
        
        print(f"Search completed in {result.search_time:.2f}s")
        print(f"Tested {result.configurations_tested} configurations")
        print(f"Best peak memory: {result.peak_memory:.1f} MB")
        print(f"Process starts: {result.process_starts}")
        
        return result
    
    def _smart_grid_search(self, n_processes: int, time_limit: float) -> SystematicSearchResult:
        """Intelligent grid search focusing on promising regions"""
        
        # Identify good candidate start positions
        candidates = self._find_candidate_positions(n_processes * 3)  # 3x candidates
        
        # Generate all combinations of start positions
        from itertools import combinations
        
        if len(candidates) <= 15:  # Small enough for exhaustive search
            search_space = list(combinations(candidates, n_processes))
        else:
            # Sample combinations for larger spaces
            max_combinations = min(50000, len(candidates) ** min(3, n_processes))
            search_space = []
            
            # Generate combinations systematically
            for _ in range(max_combinations):
                combo = tuple(sorted(random.sample(candidates, n_processes)))
                if combo not in search_space:
                    search_space.append(combo)
        
        print(f"Grid search space: {len(search_space)} combinations from {len(candidates)} candidates")
        
        # Parallel evaluation
        best_result = self._parallel_evaluate_combinations(search_space, time_limit)
        best_result.configurations_tested = len(search_space)
        
        return best_result
    
    def _memory_valley_search(self, n_processes: int, time_limit: float) -> SystematicSearchResult:
        """Search by placing processes in memory valleys/low regions"""
        
        # Find memory valleys
        memory_curve = np.array(self.ref_profile.memory_curve)
        
        # Find local minima
        valleys = []
        window_size = 20
        
        for i in range(window_size, len(memory_curve) - window_size):
            local_region = memory_curve[i-window_size:i+window_size+1]
            if memory_curve[i] == np.min(local_region):
                valleys.append(i)
        
        # Remove valleys that are too close to each other
        filtered_valleys = []
        min_gap = 50  # Minimum 50 operations between valleys
        
        for valley in sorted(valleys):
            if not filtered_valleys or valley - filtered_valleys[-1] >= min_gap:
                filtered_valleys.append(valley)
        
        print(f"Found {len(filtered_valleys)} memory valleys")
        
        # Generate combinations around valleys
        search_combinations = []
        
        if len(filtered_valleys) >= n_processes:
            # Select n_processes valleys with various strategies
            for spacing_strategy in range(min(5, len(filtered_valleys) - n_processes + 1)):
                valley_subset = filtered_valleys[spacing_strategy:spacing_strategy + n_processes]
                
                # Add small variations around each valley
                for variation in range(10):
                    varied_starts = []
                    for valley in valley_subset:
                        offset = random.randint(-20, 20)
                        start = max(0, min(valley + offset, self.ref_profile.total_ops - 100))
                        varied_starts.append(start)
                    search_combinations.append(tuple(sorted(varied_starts)))
        
        # Remove duplicates
        search_combinations = list(set(search_combinations))
        
        print(f"Valley search space: {len(search_combinations)} combinations")
        
        best_result = self._parallel_evaluate_combinations(search_combinations, time_limit)
        best_result.configurations_tested = len(search_combinations)
        
        return best_result
    
    def _monte_carlo_search(self, n_processes: int, time_limit: float) -> SystematicSearchResult:
        """Monte Carlo random search with guided sampling"""
        
        # Restrict to first 2 periods only
        if hasattr(self.ref_profile, 'single_period_ops') and self.ref_profile.single_period_ops > 0:
            max_start = self.ref_profile.single_period_ops
            print(f"Monte Carlo restricted to first 2 periods (ops 0-{max_start})")
        else:
            max_start = self.ref_profile.total_ops - 100
        
        max_start = min(max_start, self.ref_profile.total_ops - 100)
        search_combinations = []
        
        # Pure random sampling (within first 2 periods)
        random_samples = 10000
        for _ in range(random_samples):
            if max_start >= n_processes:
                starts = sorted(random.sample(range(max_start), n_processes))
                search_combinations.append(tuple(starts))
        
        # Guided sampling based on memory curve
        memory_curve = np.array(self.ref_profile.memory_curve)
        
        # Weight positions inversely by memory (prefer low memory regions)
        weights = 1.0 / (memory_curve[:max_start] + 1)
        weights = weights / np.sum(weights)
        
        guided_samples = 10000
        for _ in range(guided_samples):
            if max_start >= n_processes:
                starts = sorted(np.random.choice(max_start, n_processes, replace=False, p=weights).tolist())
                search_combinations.append(tuple(starts))
        
        # Remove duplicates
        search_combinations = list(set(search_combinations))
        
        print(f"Monte Carlo search space: {len(search_combinations)} combinations")
        
        best_result = self._parallel_evaluate_combinations(search_combinations, time_limit)
        best_result.configurations_tested = len(search_combinations)
        
        return best_result
    
    def _find_candidate_positions(self, num_candidates: int) -> List[int]:
        """Find promising start positions for processes"""
        
        memory_curve = np.array(self.ref_profile.memory_curve)
        
        # Restrict to first 1 periods only (can't start in the last period)
        if hasattr(self.ref_profile, 'single_period_ops') and self.ref_profile.single_period_ops > 0:
            max_start = self.ref_profile.single_period_ops  # First 2 periods only
            print(f"Restricting starts to first 1 period (ops 0-{max_start})")
        else:
            max_start = len(memory_curve) - 100  # Fallback
        
        max_start = min(max_start, len(memory_curve) - 100)
        
        candidates = set()
        
        # 1. Low memory positions (within first 2 periods)
        low_threshold = np.percentile(memory_curve[:max_start], 30)
        low_positions = np.where(memory_curve[:max_start] <= low_threshold)[0]
        if len(low_positions) > 0:
            candidates.update(random.sample(low_positions.tolist(), 
                                          min(num_candidates // 3, len(low_positions))))
        
        # 2. Memory valley positions (local minima) in first 2 periods
        for i in range(10, max_start - 10):
            if (memory_curve[i] < memory_curve[i-10:i].mean() and 
                memory_curve[i] < memory_curve[i+1:i+11].mean()):
                candidates.add(i)
        
        # 3. Regular spacing within first 2 periods
        spacing = max_start // (num_candidates // 3) if max_start > 0 else 1
        for i in range(0, max_start, spacing):
            candidates.add(i)
        
        # 4. Beginning positions
        candidates.update(range(0, min(50, max_start), 5))
        
        return sorted(list(candidates))
    
    def _parallel_evaluate_combinations(self, combinations: List[Tuple], time_limit: float) -> SystematicSearchResult:
        """Evaluate combinations in parallel and return best result"""
        
        best_peak = float('inf')
        best_combination = None
        
        # Use process pool for CPU-bound work
        chunk_size = max(1, len(combinations) // (self.num_cores * 4))
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            # Submit work in chunks
            evaluate_func = partial(self._evaluate_combination_batch, 
                                  memory_curve=self.ref_profile.memory_curve)
            
            # Split combinations into chunks
            chunks = [combinations[i:i+chunk_size] 
                     for i in range(0, len(combinations), chunk_size)]
            
            futures = [executor.submit(evaluate_func, chunk) for chunk in chunks]
            
            # Collect results as they complete
            for future in as_completed(futures, timeout=time_limit):
                try:
                    chunk_best_peak, chunk_best_combo = future.result()
                    if chunk_best_peak < best_peak:
                        best_peak = chunk_best_peak
                        best_combination = chunk_best_combo
                        
                except Exception as e:
                    print(f"Chunk evaluation failed: {e}")
                
                # Check time limit
                if time.time() - start_time > time_limit:
                    print("Time limit reached, terminating search")
                    break
        
        # Create result
        if best_combination is None:
            best_combination = tuple(range(0, len(combinations) * 100, 100))[:len(combinations[0])]
            best_peak = self._calculate_peak_memory(list(best_combination))
        
        memory_timeline = self._calculate_full_memory_timeline(list(best_combination))
        
        return SystematicSearchResult(
            process_starts=[int(x) for x in best_combination],  # Convert to regular Python ints
            peak_memory=float(best_peak),
            total_processes=len(best_combination),
            schedule_duration=len(memory_timeline),
            memory_timeline=[float(x) for x in memory_timeline],  # Convert to regular Python floats
            search_time=0,  # Will be filled by caller
            configurations_tested=0,  # Will be filled by caller
            search_method=""  # Will be filled by caller
        )
    
    @staticmethod
    def _evaluate_combination_batch(combinations_chunk: List[Tuple], memory_curve: List[float]) -> Tuple[float, Tuple]:
        """Evaluate a batch of combinations (used in parallel processing)"""
        
        best_peak = float('inf')
        best_combination = None
        
        for combination in combinations_chunk:
            peak = SystematicScheduler._calculate_peak_memory_static(combination, memory_curve)
            if peak < best_peak:
                best_peak = peak
                best_combination = combination
        
        return best_peak, best_combination
    
    @staticmethod
    def _calculate_peak_memory_static(process_starts: Tuple, memory_curve: List[float]) -> float:
        """Static method for calculating peak memory (used in parallel processing)"""
        
        if not process_starts:
            return 0.0
        
        max_time = max(process_starts) + len(memory_curve)
        peak_memory = 0.0
        
        # Sample every 5th operation for speed
        for t in range(0, max_time, 5):
            total_memory = 0.0
            for start in process_starts:
                process_op = t - start
                if 0 <= process_op < len(memory_curve):
                    total_memory += memory_curve[process_op]
            peak_memory = max(peak_memory, total_memory)
        
        return peak_memory
    
    def _calculate_peak_memory(self, process_starts: List[int]) -> float:
        """Calculate peak memory for given start positions"""
        return self._calculate_peak_memory_static(tuple(process_starts), self.ref_profile.memory_curve)
    
    def _calculate_full_memory_timeline(self, process_starts: List[int]) -> List[float]:
        """Calculate complete memory timeline for visualization"""
        
        if not process_starts:
            return []
        
        max_time = max(process_starts) + len(self.ref_profile.memory_curve)
        memory_timeline = [0.0] * max_time
        
        for p, start_op in enumerate(process_starts):
            for i, memory in enumerate(self.ref_profile.memory_curve):
                timeline_pos = start_op + i
                if timeline_pos < len(memory_timeline):
                    memory_timeline[timeline_pos] += memory
        
        return memory_timeline
    
    def create_schedule_visualization(self, result: SystematicSearchResult, save_path: str = None):
        """Create a beautiful visualization of the process schedule"""
        
        # Set up the plot with modern styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Individual process memory usage over time
        colors = sns.color_palette("husl", result.total_processes)
        
        max_timeline_length = max(start + len(self.ref_profile.memory_curve) 
                                for start in result.process_starts)
        
        for i, (start_op, color) in enumerate(zip(result.process_starts, colors)):
            # Create individual process timeline
            process_timeline = [0] * max_timeline_length
            
            for j, memory in enumerate(self.ref_profile.memory_curve):
                timeline_pos = start_op + j
                if timeline_pos < len(process_timeline):
                    process_timeline[timeline_pos] = memory
            
            # Convert operations to time
            time_axis = [op * 0.01 for op in range(len(process_timeline))]  # 10ms per op
            memory_gb = [mem / 1024 for mem in process_timeline]  # Convert MB to GB
            
            # Plot individual process
            ax1.fill_between(time_axis, memory_gb, alpha=0.7, color=color, 
                           label=f'Process {i+1} (start: op {start_op})')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory Usage (GB)')
        ax1.set_title('Individual Process Memory Usage Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Stacked total memory usage
        time_axis = [op * 0.01 for op in range(len(result.memory_timeline))]
        total_memory_gb = [mem / 1024 for mem in result.memory_timeline]
        
        # Create stacked area chart
        stack_data = np.zeros((result.total_processes, len(result.memory_timeline)))
        
        for i, start_op in enumerate(result.process_starts):
            for j, memory in enumerate(self.ref_profile.memory_curve):
                timeline_pos = start_op + j
                if timeline_pos < len(result.memory_timeline):
                    stack_data[i, timeline_pos] = memory / 1024  # Convert to GB
        
        # Plot stacked areas
        bottom = np.zeros(len(result.memory_timeline))
        for i in range(result.total_processes):
            ax2.fill_between(time_axis, bottom, bottom + stack_data[i], 
                           color=colors[i], alpha=0.8, 
                           label=f'Process {i+1}')
            bottom += stack_data[i]
        
        # Add peak memory line
        peak_gb = result.peak_memory / 1024
        ax2.axhline(y=peak_gb, color='red', linestyle='--', linewidth=2, 
                   label=f'Peak Memory: {peak_gb:.2f} GB')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Total Memory Usage (GB)')
        ax2.set_title(f'Stacked Memory Usage - {result.total_processes} Processes\n'
                     f'Peak: {peak_gb:.2f} GB, Method: {result.search_method}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Add optimization info text
        info_text = (f'Search Method: {result.search_method}\n'
                    f'Search Time: {result.search_time:.1f}s\n'
                    f'Configurations Tested: {result.configurations_tested:,}\n'
                    f'Peak Memory: {peak_gb:.2f} GB')
        
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return fig

def main():
    parser = argparse.ArgumentParser(description="Systematic Search Process Scheduler")
    parser.add_argument("--reference-log", required=True, help="Reference LOG_CT output")
    parser.add_argument("--reference-resources", required=True, help="Reference ResourceMonitor CSV")
    parser.add_argument("--max-processes", type=int, default=8, help="Maximum processes to consider")
    parser.add_argument("--memory-limit", type=float, help="Memory limit in MB (optional)")
    parser.add_argument("--time-limit", type=int, default=300, help="Time limit per search in seconds")
    parser.add_argument("--working-dir", default=".", help="Working directory")
    parser.add_argument("--output-dir", default="./systematic_scheduler_output", help="Output directory")
    parser.add_argument("--search-method", choices=["smart_grid", "memory_valley", "monte_carlo"], 
                       default="smart_grid", help="Search method to use")
    parser.add_argument("--create-visualization", action="store_true", help="Create schedule visualization")
    
    # Process count options
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument("--find-optimal-count", action="store_true", 
                            help="Find optimal process count by testing multiple configurations")
    count_group.add_argument("--num-processes", "-n", type=int, 
                            help="Specific number of processes to schedule")
    
    args = parser.parse_args()
    
    # Load reference profile (extended to 3 periods)
    print("Loading reference profile...")
    reference_profile = ReferenceProfile.from_files(args.reference_log, args.reference_resources, num_periods=3)
    print(f"Multi-period profile: {reference_profile.total_ops} ops, peak {reference_profile.peak_memory:.1f}MB")
    
    # Create scheduler
    scheduler = SystematicScheduler(reference_profile, args.memory_limit)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.find_optimal_count:
        # Find optimal process count
        results = {}
        
        for n_proc in range(1, args.max_processes + 1):
            print(f"\n=== Searching for {n_proc} processes ===")
            result = scheduler.find_optimal_schedule(n_proc, args.search_method, args.time_limit)
            results[n_proc] = result
            
            # Calculate efficiency
            efficiency = n_proc / (result.peak_memory / 1024)
            print(f"Efficiency: {efficiency:.2f} processes/GB")
        
        # Find best configuration
        best_n = max(results.keys(), key=lambda n: n / (results[n].peak_memory / 1024))
        best_result = results[best_n]
        print(f"\nOptimal configuration: {best_n} processes")
        
        # Save all results
        sweep_data = {str(k): asdict(v) for k, v in results.items()}
        with open(output_dir / "systematic_search_sweep.json", 'w') as f:
            json.dump(sweep_data, f, indent=2)
            
    else:
        # Search for specific number of processes
        best_result = scheduler.find_optimal_schedule(args.num_processes, args.search_method, args.time_limit)
    
    # Save best result
    with open(output_dir / "best_schedule.json", 'w') as f:
        json.dump(asdict(best_result), f, indent=2)
    
    # Create visualization if requested
    if args.create_visualization:
        viz_path = output_dir / "schedule_visualization.png"
        scheduler.create_schedule_visualization(best_result, str(viz_path))
    
    print(f"\nSystematic search completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
