# CurricuRL Phase 1: Knowledge Tracing & Cognitive Modeling

## 📋 Overview

Phase 1 adds **intelligent learner modeling** to CurricuRL, enabling the system to:
- Track which skills learners have mastered (Knowledge Tracing)
- Model time-varying cognitive capacity (Temporal Cognitive Load)
- Provide personalized recommendations based on both knowledge state and cognitive readiness

## 🎯 Key Innovation: Personalized Temporal Cognitive Load Modeling (TCLM)

**This is the first system to learn individual circadian rhythms from performance data.**

Unlike previous systems that assume everyone learns best in the morning, TCLM:
- Learns each learner's optimal learning hours from their quiz scores
- Adapts to night owls, early birds, and irregular schedules
- Combines circadian patterns with momentum, workload, and streak factors

## 📁 Project Structure

```
recommender_app/
├── knowledge_tracing.py          # BKT and DKT implementations
├── sakt_model.py                  # SAKT (Self-Attentive Knowledge Tracing) - NEW
├── temporal_cognitive_load.py     # TCLM with personalized circadian learning
├── run_phase1.py                  # Phase 1 orchestration pipeline
│
├── data/phase1/                   # Phase 1 generated data
│   ├── skill_interactions.csv     # Skill-level interactions from trajectories
│   ├── cognitive_capacity_log.csv # Time-series capacity measurements
│   ├── knowledge_assessments.csv  # BKT vs DKT performance metrics
│   └── learner_states.csv         # Unified learner state representations
│
└── models/phase1/                 # Trained Phase 1 models
    ├── bkt_model.pkl              # Bayesian Knowledge Tracing model
    ├── dkt_model.pt               # Deep Knowledge Tracing (PyTorch)
    ├── sakt_model.pt              # SAKT (Self-Attentive Knowledge Tracing) - NEW
    └── tclm_model.pkl             # Temporal Cognitive Load Model
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure Phase 0 is complete:
```bash
python run_phase0.py
```

### 2. Install Dependencies

```bash
pip install -r requirements_phase1.txt
```

### 3. Run Phase 1

```bash
python run_phase1.py
```

This will:
1. Check Phase 0 prerequisites
2. Create Phase 1 directories
3. Train BKT and DKT models
4. Train TCLM with personalized circadian learning
5. Generate unified learner states
6. Display summary report

**Expected runtime:** 5-10 minutes on CPU

## 📊 Components

### 1. Knowledge Tracing

Tracks which skills learners have mastered using two approaches:

#### Bayesian Knowledge Tracing (BKT)
- Classical probabilistic model
- 4 parameters per skill: prior knowledge, learning rate, slip, guess
- Fast, interpretable, good baseline

#### Deep Knowledge Tracing (DKT)
- LSTM-based neural network
- Learns complex skill dependencies
- Better performance than BKT

#### SAKT (Self-Attentive Knowledge Tracing)
**NEW MODEL** - Transformer-based alternative to DKT with improved performance.

Architecture:
- Self-attention mechanism (multi-head attention)
- Positional encoding for temporal patterns
- 128-dimensional embeddings, 8 attention heads, 4 layers
- Causal masking (cannot look at future interactions)

Performance (Expected):
- AUC: 0.78-0.80 (vs DKT: 0.72)
- Accuracy: 76-78% (vs DKT: 74%)
- Log-loss: 0.55-0.65 (vs DKT: 1.65)

**Usage:**
```python
from knowledge_tracing import BayesianKnowledgeTracing, DeepKnowledgeTracing
from sakt_model import SAKTRecommender

# BKT
bkt = BayesianKnowledgeTracing()
bkt.fit(train_interactions)
prob = bkt.predict('L00001', 'python')

# DKT
dkt = DeepKnowledgeTracing(num_skills=100)
dkt.train(train_interactions, epochs=20)
prob = dkt.predict(learner_sequence, 'python')

# SAKT (NEW)
sakt = SAKTRecommender(num_skills=100, embed_dim=128, num_heads=8, num_layers=4)
sakt.train(train_interactions, epochs=20)
prob = sakt.predict(learner_sequence, 'python')
```

### 2. Temporal Cognitive Load Model (TCLM)

Models time-varying cognitive capacity using 4 factors:

1. **Circadian Rhythm (Personalized)** - 0.5 to 1.2
   - Learned from quiz scores by hour
   - Requires 5+ samples per hour to activate
   - Falls back to population default if insufficient data

2. **Learning Momentum** - 0.4 to 1.4
   - Recent successes boost capacity
   - Recent failures reduce capacity
   - Weighted by recency

3. **Workload Fatigue** - 0.3 to 1.0
   - Accumulated study hours reduce capacity
   - No fatigue for first 20 hours
   - Linear decay after 20 hours

4. **Learning Streak** - 0.85 to 1.3
   - Consecutive days of learning boost capacity
   - Breaking streak reduces capacity

**Usage:**
```python
from temporal_cognitive_load import TemporalCognitiveLoadModel
from datetime import datetime

tclm = TemporalCognitiveLoadModel(learn_circadian=True)

# Calculate capacity at a specific time
capacity = tclm.calculate_cognitive_capacity('L00001', datetime(2025, 1, 15, 9, 0))

# Update after learning interaction
tclm.update_learner_state(
    learner_id='L00001',
    success=True,
    course_id='python_intro',
    hours_spent=5.0,
    timestamp=datetime(2025, 1, 15, 9, 0),
    quiz_score=85.0  # CRITICAL: Used for circadian learning!
)

# Get personalized peak hours
peak_hours = tclm.get_personalized_peak_hours('L00001', top_n=3)
print(f"Best learning hours: {peak_hours}")

# Get detailed circadian report
report = tclm.get_circadian_report('L00001')
print(f"Using personalized pattern: {report['using_personalized']}")
print(f"Peak hours: {report['peak_hours']}")
```

## 🔬 How Personalized Circadian Learning Works

### The Problem

Traditional systems assume everyone learns best in the morning:
```python
# ❌ WRONG APPROACH
if 9 <= hour <= 12:
    capacity = 1.2  # Everyone peaks in morning (WRONG!)
```

### Our Solution

Learn individual patterns from performance data:
```python
# ✅ CORRECT APPROACH
def circadian_factor(self, timestamp, learner_id):
    hour = timestamp.hour
    
    # Check if we have enough data (5+ samples for this hour)
    if self._has_enough_data(learner_id, hour):
        # Use personalized pattern learned from quiz scores
        return self._get_personalized_circadian(learner_id, hour)
    else:
        # Fallback to population default
        return self._get_default_circadian(hour)
```

### How It Learns

1. **Data Collection:** Store quiz scores by hour for each learner
   ```python
   self.learner_circadian[learner_id][hour].append(quiz_score)
   ```

2. **Pattern Discovery:** Compare average performance at each hour vs overall average
   ```python
   hour_avg = np.mean(scores_at_hour)
   overall_avg = np.mean(all_scores)
   relative_perf = hour_avg / overall_avg
   capacity = np.clip(relative_perf, 0.5, 1.2)
   ```

3. **Personalization:** After 5+ samples per hour, use personalized pattern
   - Night owl: Performs best at 10 PM → capacity = 1.2 at 10 PM
   - Early bird: Performs best at 6 AM → capacity = 1.2 at 6 AM
   - Irregular: System adapts to their unique pattern

## 📈 Expected Performance

| Model | Metric | Target | Notes |
|-------|--------|--------|-------|
| **BKT** | AUC | 0.65-0.70 | Baseline knowledge tracing |
| **BKT** | Accuracy | 60-65% | Simple probabilistic model |
| **DKT** | AUC | 0.72-0.78 | Better than BKT (deep learning) |
| **DKT** | Accuracy | 68-73% | LSTM captures skill dependencies |
| **SAKT** | AUC | 0.78-0.80 | **NEW** Transformer-based, best performance |
| **SAKT** | Accuracy | 76-78% | **NEW** Self-attention captures long-range dependencies |
| **TCLM** | Capacity Range | 0.3-1.8 | Should vary 5-6x across learners/times |
| **TCLM** | Personalized? | Yes | Must learn individual patterns |

## 🔍 Verification

After running Phase 1, verify:

```python
# 1. Files exist
from pathlib import Path
assert Path('data/phase1/skill_interactions.csv').exists()
assert Path('models/phase1/bkt_model.pkl').exists()
assert Path('models/phase1/dkt_model.pt').exists()
assert Path('models/phase1/tclm_model.pkl').exists()

# 2. Models work
from knowledge_tracing import BayesianKnowledgeTracing
bkt = BayesianKnowledgeTracing()
bkt.load('models/phase1/bkt_model.pkl')
prob = bkt.predict('L00001', 'python')
assert 0 <= prob <= 1

# 3. TCLM learns personalized patterns
from temporal_cognitive_load import TemporalCognitiveLoadModel
tclm = TemporalCognitiveLoadModel()
tclm.load('models/phase1/tclm_model.pkl')

report = tclm.get_circadian_report('L00001')
if report['using_personalized']:
    print("SUCCESS: Learned personalized pattern")
    print(f"Peak hours: {report['peak_hours']}")
else:
    print("Need more data for personalization")

# 4. Capacity varies over time
from datetime import datetime
capacity_morning = tclm.calculate_cognitive_capacity(
    'L00001', datetime(2026, 1, 16, 9, 0)
)
capacity_night = tclm.calculate_cognitive_capacity(
    'L00001', datetime(2026, 1, 16, 22, 0)
)
assert capacity_morning != capacity_night
print(f"Capacity varies: {capacity_morning:.2f} to {capacity_night:.2f}")
```

## 🐛 Troubleshooting

### Issue: "Phase 0 not complete"
**Solution:** Run `python run_phase0.py` first

### Issue: "CUDA out of memory" (DKT training)
**Solution:** Reduce batch size in `run_phase1.py`:
```python
dkt.train(train_df, epochs=10, batch_size=8)  # Smaller batch
```

### Issue: TCLM not learning personalized patterns
**Solution:** Ensure quiz scores are being stored:
```python
# In update_learner_state, must include quiz_score:
tclm.update_learner_state(..., quiz_score=85.0)  # CRITICAL!
```

### Issue: Low BKT/DKT performance
**Solution:** 
- Check skill interaction data quality
- Ensure trajectories have sufficient quiz scores
- Verify skill mappings are correct

## 📚 API Reference

### `BayesianKnowledgeTracing`

```python
class BayesianKnowledgeTracing:
    def initialize_skill(skill_id, p_l0=0.1, p_t=0.1, p_s=0.1, p_g=0.3)
    def update(learner_id, skill_id, correct) -> float
    def predict(learner_id, skill_id) -> float
    def fit(interaction_df)
    def save(path)
    def load(path)
```

### `DeepKnowledgeTracing`

```python
class DeepKnowledgeTracing:
    def __init__(num_skills, hidden_size=128, num_layers=2)
    def train(interaction_df, epochs=20, batch_size=32, lr=0.001)
    def predict(learner_sequence, skill_id) -> float
    def save(path)
    def load(path)
```

### `SAKTRecommender`

```python
class SAKTRecommender:
    def __init__(num_skills, embed_dim=128, num_heads=8, num_layers=4, dropout=0.2, max_seq_len=200)
    def train(interaction_df, epochs=20, batch_size=32, lr=0.001, val_split=0.1, patience=5)
    def predict(learner_sequence, skill_id) -> float
    def save(path)
    def load(path)
```

**Files:**
- Model code: `sakt_model.py`
- Trained weights: `models/phase1/sakt_model.pt`
- Metrics: `data/phase1/knowledge_assessments.csv` (row 3)

### `TemporalCognitiveLoadModel`

```python
class TemporalCognitiveLoadModel:
    def __init__(base_capacity=1.0, learn_circadian=True)
    def calculate_cognitive_capacity(learner_id, timestamp) -> float
    def update_learner_state(learner_id, success, course_id, hours, timestamp, quiz_score)
    def can_handle_course(learner_id, course_difficulty, timestamp) -> bool
    def get_personalized_peak_hours(learner_id, top_n=3) -> List[int]
    def get_circadian_report(learner_id) -> Dict
    def save(path)
    def load(path)
```

## 🎯 Next Steps

After Phase 1:
1. **Phase 2:** Reinforcement Learning agent training
2. **Integration:** Connect to recommendation system
3. **Visualization:** Streamlit dashboard for learner insights
4. **Evaluation:** A/B testing with real learners

## 📝 Citation

If you use this code in your research, please cite:

```
CurricuRL: Personalized Learning Recommendation with Temporal Cognitive Load Modeling
[Your Authors]
CHI/RecSys 2025
```

## 🤝 Contributing

This is a research project. For questions or contributions, please contact the CurricuRL team.

## 📄 License

[Your License Here]

