import os
import json
import csv
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from src.context_window_experiment import ContextWindowExperiment
from src.metrics_collector import MetricsCollector
from src.rag_evaluator import RAGEvaluator
from src.utils import get_logger

logger = get_logger(__name__)

class BenchmarkRunner:
    def __init__(self, rag_engine, questions_path: str = "data/evaluation_questions.json"):
        self.experiment = ContextWindowExperiment(rag_engine)
        self.metrics = MetricsCollector()
        self.evaluator = RAGEvaluator()
        self.windows = [512, 1024, 2048, 4096]

        with open(questions_path, "r") as f:
            data = json.load(f)
            self.questions = [item["question"] for item in data]

        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.raw_results = []
        self.summary_results = []

    def run(self):
        logger.info(f"Starting Phase 3 Benchmark on {len(self.questions)} questions across {len(self.windows)} windows.")

        for window in self.windows:
            window_responses = []
            latencies = []
            memory_cpus = []
            memory_gpus = []

            for q in self.questions:
                # Measure memory before
                mem_cpu_before = self.metrics.get_cpu_memory_mb()
                mem_gpu_before = self.metrics.get_gpu_memory_mb()

                # Measure latency
                self.metrics.start_timer()
                result = self.experiment.ask_with_window(q, max_tokens=window)
                latency_ms = self.metrics.end_timer()

                # Measure memory after
                mem_cpu_after = self.metrics.get_cpu_memory_mb()
                mem_gpu_after = self.metrics.get_gpu_memory_mb()

                # We calculate the delta or simply track peak for this run
                peak_cpu = max(mem_cpu_before, mem_cpu_after)
                peak_gpu = max(mem_gpu_before, mem_gpu_after)

                latencies.append(latency_ms)
                memory_cpus.append(peak_cpu)
                memory_gpus.append(peak_gpu)
                window_responses.append(result)

                # Store raw result
                self.raw_results.append({
                    "question": q,
                    "window_size": window,
                    "latency": latency_ms,
                    "memory_cpu_mb": peak_cpu,
                    "memory_gpu_mb": peak_gpu,
                    "answer_found": self.evaluator.answer_found(result),
                    "retrieval_score": self.evaluator.average_retrieval_score(result),
                    "distance": self.evaluator.average_distance(result),
                    "answer_length": result.get("answer_length", 0),
                    "context_utilization_percent": result.get("context_utilization_percent", 0.0)
                })

            # Aggregate window summaries
            avg_latency = sum(latencies) / len(latencies)
            avg_cpu = sum(memory_cpus) / len(memory_cpus)
            avg_gpu = sum(memory_gpus) / len(memory_gpus)
            hit_rate = self.evaluator.retrieval_hit_rate(window_responses)
            avg_utilization = sum([r.get("context_utilization_percent", 0.0) for r in window_responses]) / len(window_responses)

            self.summary_results.append({
                "window_size": window,
                "avg_latency_ms": avg_latency,
                "avg_memory_cpu_mb": avg_cpu,
                "avg_memory_gpu_mb": avg_gpu,
                "hit_rate": hit_rate,
                "avg_utilization_percent": avg_utilization
            })

    def generate_outputs(self):
        """Generates the required CSVs, Plots, and Markdown analysis."""
        self._write_csvs()
        self._generate_plots()
        self._generate_markdown_analysis()
        logger.info("Generated all Phase 3 benchmark outputs in results/")

    def _write_csvs(self):
        # raw_benchmark_results.csv
        with open(f"{self.results_dir}/raw_benchmark_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.raw_results[0].keys())
            writer.writeheader()
            writer.writerows(self.raw_results)

        # Extract slices for specific files as per assignment rubric
        latency_data = [{"window_size": r["window_size"], "avg_latency_ms": r["avg_latency_ms"]} for r in self.summary_results]
        with open(f"{self.results_dir}/latency_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["window_size", "avg_latency_ms"])
            w.writeheader(); w.writerows(latency_data)

        memory_data = [{"window_size": r["window_size"], "avg_memory_cpu_mb": r["avg_memory_cpu_mb"], "avg_memory_gpu_mb": r["avg_memory_gpu_mb"]} for r in self.summary_results]
        with open(f"{self.results_dir}/memory_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["window_size", "avg_memory_cpu_mb", "avg_memory_gpu_mb"])
            w.writeheader(); w.writerows(memory_data)

        quality_data = [{"window_size": r["window_size"], "hit_rate": r["hit_rate"]} for r in self.summary_results]
        with open(f"{self.results_dir}/answer_quality_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["window_size", "hit_rate"])
            w.writeheader(); w.writerows(quality_data)

        with open(f"{self.results_dir}/context_window_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.summary_results[0].keys())
            w.writeheader(); w.writerows(self.summary_results)

    def _generate_plots(self):
        windows = [r["window_size"] for r in self.summary_results]
        latencies = [r["avg_latency_ms"] for r in self.summary_results]
        memories = [r["avg_memory_cpu_mb"] + r["avg_memory_gpu_mb"] for r in self.summary_results]
        hit_rates = [r["hit_rate"] for r in self.summary_results]

        # Latency Plot
        plt.figure()
        plt.plot(windows, latencies, marker='o')
        plt.title('Average Latency vs Context Window Size')
        plt.xlabel('Window Size (Tokens)')
        plt.ylabel('Latency (ms)')
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/latency_plot.png")
        plt.close()

        # Memory Plot
        plt.figure()
        plt.plot(windows, memories, marker='s', color='orange')
        plt.title('Total Memory Usage vs Context Window Size')
        plt.xlabel('Window Size (Tokens)')
        plt.ylabel('Memory (MB)')
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/memory_plot.png")
        plt.close()

        # Quality Plot
        plt.figure()
        plt.plot(windows, hit_rates, marker='^', color='green')
        plt.title('Answer Hit Rate vs Context Window Size')
        plt.xlabel('Window Size (Tokens)')
        plt.ylabel('Hit Rate')
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/answer_quality_plot.png")
        plt.close()

        # Utilization Plot
        utilizations = [r["avg_utilization_percent"] for r in self.summary_results]
        plt.figure()
        plt.plot(windows, utilizations, marker='D', color='purple')
        plt.title('Average Context Utilization vs Window Size')
        plt.xlabel('Window Size (Tokens)')
        plt.ylabel('Utilization (%)')
        plt.grid(True)
        plt.savefig(f"{self.results_dir}/context_utilization_plot.png")
        plt.close()

    def _generate_markdown_analysis(self):
        best_latency = min(self.summary_results, key=lambda x: x["avg_latency_ms"])
        best_memory = min(self.summary_results, key=lambda x: (x["avg_memory_cpu_mb"] + x["avg_memory_gpu_mb"]))
        best_quality = max(self.summary_results, key=lambda x: x["hit_rate"])

        # Simple heuristic for recommendation: highest hit rate, tie break with lowest latency
        overall_best = max(self.summary_results, key=lambda x: (x["hit_rate"], -x["avg_latency_ms"]))

        md_content = f"""# Phase 3: Context Window Study Analysis

## Best Configurations Discovered:
* **Best Latency Window:** {best_latency['window_size']} Tokens ({best_latency['avg_latency_ms']:.2f} ms)
* **Best Memory Window:** {best_memory['window_size']} Tokens
* **Best Answer Quality Window:** {best_quality['window_size']} Tokens (Hit Rate: {best_quality['hit_rate']:.2f})

### Overall Recommended Window: {overall_best['window_size']} Tokens

## Summary Table

| Window Size | Avg Latency (ms) | Avg CPU Mem (MB) | Avg GPU Mem (MB) | Hit Rate | Avg Utilization (%) |
|---|---|---|---|---|---|
"""
        for r in self.summary_results:
            md_content += f"| {r['window_size']} | {r['avg_latency_ms']:.2f} | {r['avg_memory_cpu_mb']:.2f} | {r['avg_memory_gpu_mb']:.2f} | {r['hit_rate']:.2f} | {r['avg_utilization_percent']:.2f}% |\n"

        md_content += """
## Context Utilization Analysis
The context utilization percentage demonstrates how much of the allocated token window is actively filled with retrieved documents before truncation.
As the window size increases, we often observe diminishing returns where the utilization percentage drops significantly because the available retrieved chunks do not contain enough tokens to fill the expanded boundary.

## Advantages & Disadvantages
* **512 Tokens:** Advantage: Fastest latency and lowest memory. Disadvantage: Risks truncating critical grounded context leading to hallucination.
* **1024-2048 Tokens:** Balances memory safely on Colab T4 while fitting most RAG context boundaries smoothly.
* **4096 Tokens:** Advantage: Captures maximum context. Disadvantage: Slower inference and highest probability of VRAM exhaustion on constrained hardware.
"""
        with open(f"{self.results_dir}/phase3_analysis.md", "w") as f:
            f.write(md_content)
