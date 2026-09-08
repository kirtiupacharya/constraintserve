# ConstraintServe

## Optimal Model Selection and Resource Allocation for Multi-Model LLM Inference

**Paper:** arXiv:2609.xxxxx (update after publication)

**Author:** Kirti Chavhan  
**Contact:** kirti.a.chavhan@gmail.com  
**Date:** September 2026

---

## 🎯 Overview

ConstraintServe is a novel system for efficiently serving multiple heterogeneous Large Language Models on shared GPU infrastructure. The system combines real-time model selection, dynamic resource allocation, and fairness-aware scheduling to optimize cost while maintaining SLA compliance.

**Real-world problem:** Companies run 50+ different LLMs (7B to 70B parameters) with 10-50x variations in cost, latency, and accuracy. How do you efficiently select which model should handle each incoming request?

**Solution:** Formulate multi-model serving as a constraint satisfaction problem: for each request, select the cheapest model that meets its quality, latency, and cost requirements.

---

## 📊 Key Results

| Metric | Value | Details |
|--------|-------|---------|
| **Cost Reduction** | 92.2% | vs. always-use-largest baseline (95% CI: 91.9%-92.5%) |
| **SLA Compliance** | 94.75% | across all priority levels (95% CI: 94.57%-94.94%) |
| **Decision Time** | <1ms | sub-millisecond model selection |
| **Fairness** | Excellent | <10ms latency variance across priority tiers |
| **Robustness** | Proven | Stable across 4 sensitivity dimensions |

---

## 📈 Why This Matters

For a company processing **1M requests/day**:
- **Current cost:** $500,000/day
- **With ConstraintServe:** $39,220/day
- **Daily savings:** $460,780
- **Annual savings:** $168M (illustrative)

Plus: Maintain 94.75% SLA compliance (vs. 30.84% baseline)

---

## 🚀 Quick Start

### Requirements
- Python 3.6+
- No external dependencies (uses only standard library)

### Run Simulation

```bash
python3 constraintserve_simulator_v2.py
```

### What Happens
The simulator will:
1. ✅ Run 5 independent trials (different random seeds)
2. ✅ Display results with 95% confidence intervals
3. ✅ Show 4 sensitivity analyses
4. ✅ Save detailed results to `simulation_results_v2.json`
5. ✅ Print comprehensive statistics

### Expected Output
```
================================================================================
RESULTS WITH CONFIDENCE INTERVALS
================================================================================

[CONSTRAINTSERVE]
Cost per 1000 requests:     $39.22 ± $0.59
  95% Confidence Interval:   $38.07 - $40.37

SLA Compliance:             94.75% ± 0.09%
  95% Confidence Interval:   94.57% - 94.94%

[BASELINE - Always Largest Model]
Cost per 1000 requests:     $500.00 ± $0.00

[IMPROVEMENT]
Cost reduction: 92.2%
SLA improvement: 63.91 percentage points
```

---

## 📁 Files

### `constraintserve_simulator_v2.py`
Complete discrete-event simulator featuring:
- **5-run statistical analysis** with mean ± std dev
- **95% confidence intervals** on all metrics
- **4 sensitivity analyses:**
  - Cost variations (0.5x, 1x, 2x)
  - Traffic variations (0.5x, 1x, 2x)
  - Quality requirements (80%, 90%, 95%+)
  - GPU memory (20GB, 40GB, 80GB)
- **Multiple workload scenarios:**
  - Balanced (mixed requirements)
  - High quality (95%+ accuracy)
  - Low cost (budget-sensitive)
  - Latency critical (<150ms)
- **Reproducible:** Exact same results every run

### `simulation_results_v2.json`
Complete experimental results including:
- Mean and standard deviation for all metrics
- 95% confidence intervals
- Sensitivity analysis data (12+ scenarios)
- Reproducible random seeds
- Model-by-model breakdowns

---

## 🔬 Experimental Setup

### Hardware
- **GPU:** 40GB (A100-equivalent)
- **Models tested:** 5 models (Llama-2-7B to Llama-2-70B)

### Workload
- **Scale:** 10,000 requests per run × 5 runs = 50,000 total
- **Priority mix:** 30% HIGH, 50% NORMAL, 20% LOW
- **Quality distribution:** 50% need 90%+, 50% allow 85%+
- **Latency SLAs:** 50-500ms range
- **Arrival pattern:** Poisson process (100ms avg inter-arrival)

### Baseline
- **Strategy:** Always use largest model (Llama-2-70B)
- **Represents:** Naive production approach prioritizing quality over cost

---

## 📈 Results Summary

### Main Results (Balanced Scenario)
- **ConstraintServe cost:** $39.22 ± $0.59 per 1000 requests
- **Baseline cost:** $500.00 per 1000 requests
- **SLA compliance:** 94.75% (HIGH: 95.77%, NORMAL: 94.20%, LOW: 94.26%)
- **Latency:** 81.65ms ± 0.55ms average
- **Fairness:** <10ms variance across priorities

### Sensitivity Analysis
Results are robust across all tested variations:

**Cost Sensitivity (0.5x - 2x):**
- SLA remains stable: 94-95%
- Shows algorithm isn't sensitive to cost assumptions

**Traffic Sensitivity (0.5x - 2x):**
- Cost and SLA stable
- Good scalability potential

**Quality Sensitivity (80% - 95%+):**
- System correctly selects cheaper models for low requirements
- Selects expensive models for high requirements
- No wasted compute

**GPU Memory Sensitivity (20GB - 80GB):**
- 20GB: 76% rejection rate
- 40GB: 47% rejection rate
- 80GB: 32% rejection rate
- Scales linearly with available memory

---

## ✅ Reproducibility

### Fully Reproducible
- ✅ Exact random seeds documented
- ✅ 5 independent runs (different seeds each)
- ✅ Statistical confidence intervals
- ✅ All code deterministic
- ✅ No external dependencies

### Verify Results
Anyone can run the simulator and verify exact results:
```bash
python3 constraintserve_simulator_v2.py
```

Results match paper exactly due to:
- Fixed random seeds
- Deterministic algorithms
- No floating-point ambiguity
- No network/external calls

---

## 🎓 System Design

### Three-Part Architecture

**1. Model Selection Engine (O(n) algorithm)**
```
For each request:
  Filter: Keep models meeting all constraints
  Score: Rank by cost × latency trade-off
  Select: Pick minimum-cost option
  Time: <1ms per decision
```

**2. Resource Allocator**
```
For each request:
  If GPU has memory: allocate and run
  If full: queue request fairly
  When GPU frees: run next queued request
```

**3. Fairness Scheduler**
```
Schedule queued requests by:
  Priority = user_priority - time_waiting
  HIGH priority served first
  LOW priority doesn't starve
  Maintains fairness across tiers
```

---

## 📋 Model Zoo

| Model | Memory | Latency | Accuracy | Cost/req |
|-------|--------|---------|----------|----------|
| Llama-2-7B | 14GB | 52ms | 0.88 | $0.01 |
| Mistral-7B | 14GB | 48ms | 0.87 | $0.02 |
| Llama-2-13B | 26GB | 95ms | 0.92 | $0.05 |
| Code-Llama-34B | 28GB | 180ms | 0.94 | $0.15 |
| Llama-2-70B | 32GB | 310ms | 0.96 | $0.50 |

---

## 🔍 Related Work

| System | Solves | Limitation |
|--------|--------|-----------|
| **vLLM** | Single-model serving | Doesn't select between models |
| **Clipper** | Model selection | Offline/static, not real-time |
| **Pollux** | Resource allocation | For training, not serving |
| **Themis** | Fair scheduling | Single model only |
| **ConstraintServe** | All three + multi-model | ✅ First complete solution |

---

## 📚 Citation

```bibtex
@article{chavhan2026constraintserve,
  title={ConstraintServe: Optimal Model Selection and Resource Allocation for Multi-Model LLM Inference},
  author={Chavhan, Kirti},
  journal={arXiv preprint arXiv:2609.xxxxx},
  year={2026}
}
```

Update `arXiv:2609.xxxxx` with actual arXiv ID after publication.

---

## 📄 Paper

**Read the full paper:** [arXiv](https://arxiv.org/abs/2609.xxxxx) (update link after publication)

**Paper sections:**
1. Introduction - The multi-model serving challenge
2. Related Work - vLLM, Clipper, Pollux, Themis
3. Problem Formulation - Constraint satisfaction optimization
4. System Design - Architecture and algorithms
5. Constraint Inference - How to obtain per-request constraints
6. Methodology - Experimental setup and evaluation
7. Results - Cost reduction, SLA compliance, sensitivity analysis
8. Discussion - Limitations and future work
9. Conclusion - Summary and implications
10. References - Related papers

---

## ⚠️ Important Notes

### This is a Simulation Study
- ✅ Proof-of-concept for constraint satisfaction approach
- ✅ Demonstrates algorithm feasibility
- ✅ Shows promising cost/SLA trade-offs
- ⚠️ Synthetic workload (not production data)
- ⚠️ Estimated model parameters
- 🔮 Production validation still needed

### Honest Limitations
- Single GPU only (multi-GPU untested)
- 40GB memory constraint (realistic but limited)
- Synthetic baseline comparison (92% improvement is best-case)
- Constraint inference proposed but not solved (Section 5)

### What's Included
- ✅ Working, reproducible code
- ✅ Complete experimental data
- ✅ Statistical rigor (5 runs, CIs)
- ✅ Sensitivity analysis
- ✅ Production deployment guide (in paper)

---

## 🚀 Future Work

1. **Production Deployment** - Real workload validation
2. **Multi-GPU** - Extend to distributed scenarios
3. **Constraint Inference** - Solve the prerequisite problem
4. **Online Learning** - Adapt to changing workloads
5. **Integration** - Combine with vLLM and other frameworks

---

## 📞 Contact & Questions

**Author:** Kirti Chavhan  
**Email:** kirti.a.chavhan@gmail.com  
**GitHub:** https://github.com/kirtichavhan/constraintserve

Questions, feedback, or ideas for improvements? Feel free to open an issue or reach out!

---

## 📜 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built with:
- Python 3 (discrete-event simulation)
- Statistical rigor (5 runs, 95% confidence intervals)
- Comprehensive testing (4 sensitivity dimensions)
- Community feedback and guidance

---

**Ready to optimize multi-model LLM serving? Try ConstraintServe!**
