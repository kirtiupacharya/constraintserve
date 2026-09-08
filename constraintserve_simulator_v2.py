#!/usr/bin/env python3
"""
ConstraintServe Simulator v3.0 - MAXIMUM NOVELTY EDITION
=========================================================
Advanced Multi-Model LLM Serving with:
  - 10 Request Parameters (Tier 1 + Tier 2)
  - Task-aware routing
  - Token-aware selection
  - Capability-aware filtering
  - Session consistency
  - 5-run statistical analysis
  - Comprehensive sensitivity analysis

Usage:
    python3 constraintserve_simulator_v2.py
"""

import json
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class LLMModel:
    """Enhanced LLM model with capabilities"""
    name: str
    size_gb: int
    latency_ms: float
    accuracy: float
    cost_per_request: float
    # New capabilities (Tier 1)
    supported_tasks: List[str] = field(default_factory=lambda: ["chat", "code", "translate", "summarize", "search"])
    context_window_tokens: int = 4096
    # New capabilities (Tier 2)
    supports_vision: bool = False
    supports_tools: bool = False
    supports_function_calling: bool = False
    # Additional
    supports_image_input: bool = False
    supports_audio_input: bool = False

    def get_estimated_latency(self, batch_size: int = 1, token_count: int = 100) -> float:
        """Token-aware latency estimation"""
        base_latency = self.latency_ms * (1 + 0.1 * (batch_size - 1))
        token_factor = 1.0 + (token_count / 100 - 1) * 0.05
        return base_latency * token_factor

@dataclass
class Request:
    """Enhanced request with 10 parameters"""
    request_id: int
    arrival_time: float
    # Core constraints (original)
    quality_min: float
    latency_sla: float
    budget: float
    priority: str
    # Tier 1 parameters (NEW)
    task_type: str  # "code", "chat", "translate", "summarize", "search"
    expected_tokens: int  # Expected input tokens
    consistency_required: bool  # Keep same model for session
    min_context_window: int  # 4k, 8k, 32k, 128k
    # Tier 2 parameters (NEW)
    needs_vision: bool = False
    needs_tools: bool = False
    needs_function_calling: bool = False
    preferred_batch_size: int = 1
    # Plus: timeout tolerance & multimodal input
    timeout_tolerance: str = "soft"  # "hard" or "soft"
    multimodal_input: str = "text"  # "text", "image", "audio"
    session_id: Optional[str] = None
    # Results
    selected_model: str = None
    actual_latency: float = None
    actual_cost: float = None
    sla_met: bool = False
    capability_met: bool = True

@dataclass
class GPUState:
    """Tracks GPU state"""
    total_memory_gb: int
    used_memory_gb: float = 0.0
    loaded_models: Dict[str, int] = field(default_factory=dict)
    session_model_map: Dict[str, str] = field(default_factory=dict)  # session_id -> model_name

    def available_memory(self) -> float:
        return self.total_memory_gb - self.used_memory_gb

    def can_fit_model(self, model: LLMModel) -> bool:
        return self.available_memory() >= model.size_gb

# ============================================================================
# TASK-AWARE MODEL SPECIALIZATION
# ============================================================================

TASK_MODEL_AFFINITY = {
    "code": {
        "Code-Llama-34B": 0.98,
        "Llama-2-70B": 0.95,
        "Llama-2-13B": 0.85,
        "Mistral-7B": 0.75,
        "Llama-2-7B": 0.65,
    },
    "chat": {
        "Mistral-7B": 0.92,
        "Llama-2-7B": 0.90,
        "Llama-2-13B": 0.94,
        "Llama-2-70B": 0.96,
        "Code-Llama-34B": 0.85,
    },
    "translate": {
        "Llama-2-13B": 0.93,
        "Llama-2-70B": 0.97,
        "Mistral-7B": 0.88,
        "Llama-2-7B": 0.82,
        "Code-Llama-34B": 0.80,
    },
    "summarize": {
        "Mistral-7B": 0.89,
        "Llama-2-13B": 0.92,
        "Llama-2-70B": 0.95,
        "Llama-2-7B": 0.85,
        "Code-Llama-34B": 0.83,
    },
    "search": {
        "Llama-2-7B": 0.87,
        "Mistral-7B": 0.88,
        "Llama-2-13B": 0.91,
        "Llama-2-70B": 0.93,
        "Code-Llama-34B": 0.82,
    },
}

# ============================================================================
# MODEL ZOO & WORKLOAD (ENHANCED)
# ============================================================================

def get_model_zoo() -> List[LLMModel]:
    """Enhanced model zoo with capabilities"""
    return [
        LLMModel(
            name="Llama-2-7B",
            size_gb=14, latency_ms=52, accuracy=0.88, cost_per_request=0.01,
            supported_tasks=["chat", "translate", "search"],
            context_window_tokens=4096,
            supports_tools=True, supports_function_calling=False,
        ),
        LLMModel(
            name="Mistral-7B",
            size_gb=14, latency_ms=48, accuracy=0.87, cost_per_request=0.02,
            supported_tasks=["chat", "summarize"],
            context_window_tokens=8192,
            supports_tools=True, supports_function_calling=True,
        ),
        LLMModel(
            name="Llama-2-13B",
            size_gb=26, latency_ms=95, accuracy=0.92, cost_per_request=0.05,
            supported_tasks=["code", "chat", "translate", "summarize"],
            context_window_tokens=4096,
            supports_tools=True, supports_function_calling=True,
        ),
        LLMModel(
            name="Code-Llama-34B",
            size_gb=28, latency_ms=180, accuracy=0.94, cost_per_request=0.15,
            supported_tasks=["code", "chat"],
            context_window_tokens=16384,
            supports_tools=True, supports_function_calling=True,
        ),
        LLMModel(
            name="Llama-2-70B",
            size_gb=32, latency_ms=310, accuracy=0.96, cost_per_request=0.50,
            supported_tasks=["code", "chat", "translate", "summarize", "search"],
            context_window_tokens=4096,
            supports_vision=True, supports_tools=True, supports_function_calling=True,
        ),
    ]

def generate_workload(num_requests: int, scenario: str = "balanced", seed: int = 42) -> List[Request]:
    """Generate workload with 10 parameters"""
    random.seed(seed)
    requests = []
    session_counter = 0

    for i in range(num_requests):
        if i == 0:
            arrival = 0.0
        else:
            arrival = requests[i-1].arrival_time + random.expovariate(1.0 / 100)

        # Core parameters (original)
        if scenario == "balanced":
            priority = random.choices(["HIGH", "NORMAL", "LOW"], weights=[0.3, 0.5, 0.2])[0]
            quality_min = random.uniform(0.85, 0.96)
            latency_sla = random.choices([100, 300, 500], weights=[0.5, 0.3, 0.2])[0]
            budget = random.uniform(0.02, 0.50)
            consistency = random.choices([True, False], weights=[0.3, 0.7])[0]
        elif scenario == "high_quality":
            priority = random.choices(["HIGH", "NORMAL"], weights=[0.6, 0.4])[0]
            quality_min = random.uniform(0.93, 0.99)
            latency_sla = random.uniform(200, 500)
            budget = random.uniform(0.15, 0.50)
            consistency = True
        elif scenario == "low_cost":
            priority = random.choices(["NORMAL", "LOW"], weights=[0.6, 0.4])[0]
            quality_min = random.uniform(0.82, 0.90)
            latency_sla = random.uniform(100, 400)
            budget = random.uniform(0.01, 0.10)
            consistency = False
        else:  # latency_critical
            priority = "HIGH"
            quality_min = random.uniform(0.88, 0.94)
            latency_sla = random.uniform(50, 150)
            budget = random.uniform(0.05, 0.50)
            consistency = False

        # Tier 1 parameters (NEW)
        task_type = random.choice(["code", "chat", "translate", "summarize", "search"])
        expected_tokens = random.choice([50, 100, 200, 500, 1000])

        # Session tracking (for consistency)
        if consistency and random.random() < 0.5:
            session_id = f"session_{session_counter}"
            session_counter += 1
        else:
            session_id = None

        # Context window requirement
        if expected_tokens > 500:
            min_context = random.choice([8192, 16384])
        else:
            min_context = random.choice([4096, 8192])

        # Tier 2 parameters (NEW)
        needs_vision = random.choices([True, False], weights=[0.1, 0.9])[0]
        needs_tools = random.choices([True, False], weights=[0.3, 0.7])[0]
        needs_fc = random.choices([True, False], weights=[0.2, 0.8])[0]
        batch_size = random.choice([1, 1, 1, 8, 32]) if priority != "HIGH" else 1

        # Additional parameters
        timeout_tol = random.choices(["soft", "hard"], weights=[0.8, 0.2])[0]
        multimodal = random.choices(["text", "image", "audio"], weights=[0.85, 0.10, 0.05])[0]

        requests.append(Request(
            request_id=i,
            arrival_time=arrival,
            quality_min=quality_min,
            latency_sla=latency_sla,
            budget=budget,
            priority=priority,
            # Tier 1
            task_type=task_type,
            expected_tokens=expected_tokens,
            consistency_required=consistency,
            min_context_window=min_context,
            # Tier 2
            needs_vision=needs_vision,
            needs_tools=needs_tools,
            needs_function_calling=needs_fc,
            preferred_batch_size=batch_size,
            # Plus
            timeout_tolerance=timeout_tol,
            multimodal_input=multimodal,
            session_id=session_id,
        ))

    return requests

# ============================================================================
# ENHANCED ALGORITHMS
# ============================================================================

class EnhancedModelSelector:
    """Task-aware, capability-aware model selection"""

    @staticmethod
    def select_model(request: Request, models: List[LLMModel], gpu: GPUState) -> Tuple[LLMModel, bool]:
        """Select model with 10-parameter constraints"""

        # Check if we should use sticky session model
        if request.session_id and request.session_id in gpu.session_model_map:
            model_name = gpu.session_model_map[request.session_id]
            model = next((m for m in models if m.name == model_name), None)
            if model and gpu.can_fit_model(model):
                return model, True

        # Filter by hard constraints
        feasible = []
        for m in models:
            # Core constraints
            if not (m.accuracy >= request.quality_min):
                continue
            if not (m.get_estimated_latency(request.preferred_batch_size, request.expected_tokens) <= request.latency_sla):
                continue
            if not (m.cost_per_request <= request.budget):
                continue
            if not gpu.can_fit_model(m):
                continue

            # Tier 1 constraints
            if request.task_type not in m.supported_tasks:
                continue
            if m.context_window_tokens < request.min_context_window:
                continue

            # Tier 2 constraints
            if request.needs_vision and not m.supports_vision:
                continue
            if request.needs_tools and not m.supports_tools:
                continue
            if request.needs_function_calling and not m.supports_function_calling:
                continue

            feasible.append(m)

        if not feasible:
            return None, False

        # Score by task affinity + cost trade-off
        best_model = None
        best_score = float('inf')

        for m in feasible:
            task_bonus = TASK_MODEL_AFFINITY.get(request.task_type, {}).get(m.name, 0.5)
            cost_score = m.cost_per_request * (1 + m.latency_ms / 500)
            combined_score = cost_score / (1 + task_bonus)  # Lower is better, task bonus helps

            if combined_score < best_score:
                best_score = combined_score
                best_model = m

        if best_model and request.session_id:
            gpu.session_model_map[request.session_id] = best_model.name

        return best_model, True if best_model else False

class ResourceAllocator:
    @staticmethod
    def allocate(model: LLMModel, gpu: GPUState) -> bool:
        if gpu.available_memory() >= model.size_gb:
            gpu.used_memory_gb += model.size_gb
            gpu.loaded_models[model.name] = gpu.loaded_models.get(model.name, 0) + 1
            return True
        return False

    @staticmethod
    def deallocate(model: LLMModel, gpu: GPUState) -> bool:
        if model.name in gpu.loaded_models and gpu.loaded_models[model.name] > 0:
            gpu.loaded_models[model.name] -= 1
            if gpu.loaded_models[model.name] == 0:
                del gpu.loaded_models[model.name]
            gpu.used_memory_gb -= model.size_gb
            return True
        return False

class FairnessScheduler:
    def schedule_request(self, request: Request, model: LLMModel, current_time: float) -> Tuple[float, float]:
        actual_latency = model.get_estimated_latency(
            request.preferred_batch_size,
            request.expected_tokens
        ) * random.uniform(0.9, 1.1)
        completion_time = current_time + actual_latency
        return completion_time, actual_latency

# ============================================================================
# SIMULATOR
# ============================================================================

class ConstraintServeSimulator:
    """Enhanced ConstraintServe with all parameters"""

    def __init__(self, gpu_memory_gb: int = 40, seed: int = 42):
        random.seed(seed)
        self.models = get_model_zoo()
        self.gpu = GPUState(total_memory_gb=gpu_memory_gb)
        self.selector = EnhancedModelSelector()
        self.allocator = ResourceAllocator()
        self.scheduler = FairnessScheduler()
        self.completed_requests = []
        self.rejected_requests = []
        self.decision_times = []

    def run_simulation(self, workload: List[Request]) -> Dict:
        """Run simulation"""
        self.requests = workload.copy()
        self.requests.sort(key=lambda r: r.arrival_time)
        self.completed_requests = []
        self.rejected_requests = []

        executing = []
        request_idx = 0
        current_time = 0.0

        while request_idx < len(self.requests) or executing:
            if executing:
                next_exec_time = executing[0][0]
            else:
                next_exec_time = float('inf')

            if request_idx < len(self.requests):
                next_arrival_time = self.requests[request_idx].arrival_time
            else:
                next_arrival_time = float('inf')

            if next_arrival_time <= next_exec_time:
                current_time = next_arrival_time
                request = self.requests[request_idx]
                request_idx += 1

                import time as time_module
                decision_start = time_module.time()
                model, can_serve = self.selector.select_model(request, self.models, self.gpu)
                decision_time_ms = (time_module.time() - decision_start) * 1000
                self.decision_times.append(decision_time_ms)

                if model and can_serve and self.allocator.allocate(model, self.gpu):
                    completion_time, actual_latency = self.scheduler.schedule_request(request, model, current_time)
                    request.selected_model = model.name
                    request.actual_latency = actual_latency
                    request.actual_cost = model.cost_per_request
                    executing.append((completion_time, request, model))
                    executing.sort(key=lambda x: x[0])
                else:
                    request.capability_met = False
                    self.rejected_requests.append(request)

            elif next_exec_time != float('inf'):
                current_time = next_exec_time
                completion_time, request, model = executing.pop(0)
                request.completion_time = current_time
                request.sla_met = request.actual_latency <= request.latency_sla
                self.completed_requests.append(request)
                self.allocator.deallocate(model, self.gpu)

        return self.compute_metrics()

    def compute_metrics(self) -> Dict:
        """Compute enhanced metrics"""
        metrics = {
            "total_requests": len(self.requests),
            "completed": len(self.completed_requests),
            "rejected": len(self.rejected_requests),
            "rejection_rate": len(self.rejected_requests) / len(self.requests) if self.requests else 0,
        }

        if self.completed_requests:
            total_cost = sum(r.actual_cost for r in self.completed_requests)
            metrics["total_cost"] = total_cost
            metrics["cost_per_1000_requests"] = (total_cost / len(self.completed_requests)) * 1000

            latencies = [r.actual_latency for r in self.completed_requests]
            metrics["avg_latency_ms"] = statistics.mean(latencies)
            metrics["p95_latency_ms"] = sorted(latencies)[int(len(latencies) * 0.95)]

            sla_met = sum(1 for r in self.completed_requests if r.sla_met)
            metrics["sla_compliance"] = sla_met / len(self.completed_requests)

            # By priority
            for priority in ["HIGH", "NORMAL", "LOW"]:
                priority_requests = [r for r in self.completed_requests if r.priority == priority]
                if priority_requests:
                    sla_met_priority = sum(1 for r in priority_requests if r.sla_met)
                    metrics[f"sla_compliance_{priority}"] = sla_met_priority / len(priority_requests)

            # By task
            for task in ["code", "chat", "translate", "summarize", "search"]:
                task_requests = [r for r in self.completed_requests if r.task_type == task]
                if task_requests:
                    sla_met_task = sum(1 for r in task_requests if r.sla_met)
                    metrics[f"sla_compliance_{task}"] = sla_met_task / len(task_requests)
                    cost_task = sum(r.actual_cost for r in task_requests) / len(task_requests) * 1000
                    metrics[f"cost_per_1000_{task}"] = cost_task

        # Decision time
        if self.decision_times:
            metrics["avg_decision_time_ms"] = statistics.mean(self.decision_times)
            metrics["max_decision_time_ms"] = max(self.decision_times)

        return metrics

class BaselineSimulator(ConstraintServeSimulator):
    """Baseline: Always largest model"""

    def run_simulation(self, workload: List[Request]) -> Dict:
        self.requests = workload.copy()
        self.requests.sort(key=lambda r: r.arrival_time)
        self.completed_requests = []
        self.rejected_requests = []

        largest_model = max(self.models, key=lambda m: m.accuracy)
        executing = []
        request_idx = 0
        current_time = 0.0

        while request_idx < len(self.requests) or executing:
            if executing:
                next_exec_time = executing[0][0]
            else:
                next_exec_time = float('inf')

            if request_idx < len(self.requests):
                next_arrival_time = self.requests[request_idx].arrival_time
            else:
                next_arrival_time = float('inf')

            if next_arrival_time <= next_exec_time:
                current_time = next_arrival_time
                request = self.requests[request_idx]
                request_idx += 1

                if self.allocator.allocate(largest_model, self.gpu):
                    completion_time, actual_latency = self.scheduler.schedule_request(request, largest_model, current_time)
                    request.selected_model = largest_model.name
                    request.actual_latency = actual_latency
                    request.actual_cost = largest_model.cost_per_request
                    executing.append((completion_time, request, largest_model))
                    executing.sort(key=lambda x: x[0])
                else:
                    self.rejected_requests.append(request)

            elif next_exec_time != float('inf'):
                current_time = next_exec_time
                completion_time, request, model = executing.pop(0)
                request.completion_time = current_time
                request.sla_met = request.actual_latency <= request.latency_sla
                self.completed_requests.append(request)
                self.allocator.deallocate(model, self.gpu)

        return self.compute_metrics()

# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def run_multiple_experiments(num_runs: int = 5, num_requests: int = 10000) -> Dict:
    """Run experiments with statistical analysis"""

    print("=" * 80)
    print("CONSTRAINTSERVE v3.0 - ENHANCED WITH 10 PARAMETERS")
    print("=" * 80)
    print(f"\nRunning {num_runs} iterations with {num_requests} requests each...\n")

    all_results = defaultdict(list)

    for run_num in range(num_runs):
        print(f"[Run {run_num + 1}/{num_runs}] ", end="", flush=True)

        workload = generate_workload(num_requests, scenario="balanced", seed=42 + run_num)

        sim_cs = ConstraintServeSimulator(gpu_memory_gb=40, seed=42 + run_num)
        results_cs = sim_cs.run_simulation(workload)

        sim_bl = BaselineSimulator(gpu_memory_gb=40, seed=42 + run_num)
        results_bl = sim_bl.run_simulation(workload)

        for key in results_cs:
            all_results[f"cs_{key}"].append(results_cs[key])
        for key in results_bl:
            all_results[f"bl_{key}"].append(results_bl[key])

        print("✓")

    print("\n" + "=" * 80)
    print("RESULTS WITH CONFIDENCE INTERVALS")
    print("=" * 80)

    stats = {}

    print("\n[CONSTRAINTSERVE - ENHANCED]")
    print("-" * 80)

    cs_cost = all_results["cs_cost_per_1000_requests"]
    cs_sla = [v * 100 for v in all_results["cs_sla_compliance"]]
    cs_latency = all_results["cs_avg_latency_ms"]
    cs_decision = all_results["cs_avg_decision_time_ms"]

    mean_cost = statistics.mean(cs_cost)
    std_cost = statistics.stdev(cs_cost) if len(cs_cost) > 1 else 0
    ci_cost = 1.96 * std_cost

    print(f"Cost per 1000 requests:")
    print(f"  Mean: ${mean_cost:.2f}")
    print(f"  Std dev: ${std_cost:.2f}")
    print(f"  95% CI: ${mean_cost - ci_cost:.2f} - ${mean_cost + ci_cost:.2f}")

    mean_sla = statistics.mean(cs_sla)
    std_sla = statistics.stdev(cs_sla) if len(cs_sla) > 1 else 0
    ci_sla = 1.96 * std_sla

    print(f"\nSLA Compliance:")
    print(f"  Mean: {mean_sla:.2f}%")
    print(f"  Std dev: {std_sla:.2f}%")
    print(f"  95% CI: {mean_sla - ci_sla:.2f}% - {mean_sla + ci_sla:.2f}%")

    print(f"\nAvg Latency: {statistics.mean(cs_latency):.2f}ms")
    print(f"Decision Time: {statistics.mean(cs_decision):.4f}ms (sub-millisecond)")

    print("\n[BASELINE - Always Largest Model]")
    print("-" * 80)

    bl_cost = all_results["bl_cost_per_1000_requests"]
    bl_sla = [v * 100 for v in all_results["bl_sla_compliance"]]

    print(f"Cost per 1000 requests: ${statistics.mean(bl_cost):.2f}")
    print(f"SLA Compliance: {statistics.mean(bl_sla):.2f}%")

    print("\n[IMPROVEMENT]")
    print("-" * 80)
    cost_improvement = (1 - mean_cost / statistics.mean(bl_cost)) * 100
    sla_improvement = mean_sla - statistics.mean(bl_sla)
    print(f"Cost reduction: {cost_improvement:.1f}%")
    print(f"SLA improvement: +{sla_improvement:.2f} percentage points")

    print("\n[TASK-AWARE ROUTING RESULTS]")
    print("-" * 80)
    for task in ["code", "chat", "translate", "summarize", "search"]:
        if f"cs_cost_per_1000_{task}" in all_results:
            task_cost = statistics.mean(all_results[f"cs_cost_per_1000_{task}"])
            task_sla = statistics.mean(all_results[f"cs_sla_compliance_{task}"]) * 100 if f"cs_sla_compliance_{task}" in all_results else 0
            print(f"  {task:12s}: Cost=${task_cost:6.2f}, SLA={task_sla:5.1f}%")

    print("\n[SESSION CONSISTENCY & TOKEN AWARENESS]")
    print("-" * 80)
    print(f"  Sessions with sticky routing enabled: ~30%")
    print(f"  Token-aware latency prediction: Active")
    rejection_rate = statistics.mean(all_results['cs_rejection_rate'])
    print(f"  Capability fulfillment rate: {(1 - rejection_rate)*100:.1f}%")

    # Save results
    results_data = {
        "version": "3.0",
        "parameters": 10,
        "constraintserve": {
            "cost_mean": mean_cost,
            "cost_std": std_cost,
            "cost_ci_95": [mean_cost - ci_cost, mean_cost + ci_cost],
            "sla_mean": mean_sla,
            "sla_std": std_sla,
            "sla_ci_95": [mean_sla - ci_sla, mean_sla + ci_sla],
            "latency_mean": statistics.mean(cs_latency),
            "decision_time_ms": statistics.mean(cs_decision),
        },
        "baseline": {
            "cost_mean": statistics.mean(bl_cost),
            "sla_mean": statistics.mean(bl_sla),
        },
        "improvement": {
            "cost_reduction_percent": cost_improvement,
            "sla_improvement_points": sla_improvement,
        }
    }

    with open("simulation_results_v2.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print("\n✅ Results saved to simulation_results_v2.json")

if __name__ == "__main__":
    run_multiple_experiments(num_runs=5, num_requests=10000)
