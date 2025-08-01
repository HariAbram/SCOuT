# SCOuT – Systematic **C**ompile-time **O**ptimisation & runtime **T**uning
*(pronounced **“scout”)*  

Design-space exploration for Parallel workloads.

---

## ✨ Highlights

| ✔ | Feature |
|---|---------|
| **Builds anything**  |  a single `.cpp` file *or* a full CMake/Make project (AdaptiveCpp, Clang, DPC++ …).
| **Explores flags & env**  | –flag bundles, per-option parameters, on/off flag pools, CMake `-D` switches, conditional env-vars.
| **Measures with LIKWID or perf**  | TIMERS, CPI, custom PMCs or any perf event set.
| **Multi-objective search**  | Optuna **TPE** (Bayesian), **NSGA-III** (Pareto genetic), **RF** (Random-Forest surrogate).
| **CSV archive**  |  full metrics × config matrix for offline analysis.

---

## 🚀 Quick start

```bash
git clone https://github.com/your-org/scout.git
cd scout
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                
python main.py configs/minimal.json --trials 100
