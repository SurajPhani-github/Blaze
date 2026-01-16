# CurricuRL: Curriculum-Aware Reinforcement Learning for Personalized Learning Paths

##  Overview

**CurricuRL** is a research-grade personalized learning system that uses Reinforcement Learning (RL) to generate optimal learning paths. Unlike traditional recommendation systems that suggest courses independently, CurricuRL considers:

- **Prerequisite structures** - Courses must be taken in logical order
- **Skill dependencies** - What skills are needed before tackling advanced topics
- **Individual learner state** - Your current knowledge and learning pace
- **Multi-objective optimization** - Balance engagement, mastery, and efficiency

### Research Goal

This project aims to publish a paper at **KDD/RecSys/WWW 2025-2026** demonstrating that RL-based curriculum planning outperforms traditional recommendation approaches.

---

## Current Project Structure

```
recommender_app/
├── app.py                          # Streamlit UI (existing)
├── recommender.py                  # Baseline recommenders (existing)
├── requirements.txt                # Original dependencies (existing)
│
├── requirements.txt                # NEW:  dependencies
├── run_phase0.py                   # NEW: Master orchestrator
│
├── skill_extractor.py              # NEW: Extract skills from courses
├── prerequisite_miner.py           # NEW: Infer course prerequisites
├── trajectory_simulator.py         # NEW: Generate learner trajectories
│
├── coursera_courses.csv            # Input: Course catalog (existing)
├── augmented_learner_data.csv      # Input: Learner data (existing)
│
├── data/                           # Generated data (NEW)
│   ├── course_skills_raw.csv
│   ├── course_skills_clustered.csv
│   ├── skill_graph.gpickle
│   ├── course_prerequisites.csv
│   ├── prerequisite_graph.gpickle
│   ├── learner_trajectories.csv
│   ├── augmented_learner_data_v2.csv
│   └── neo4j/                      # Neo4j import files
│       ├── skills_nodes.csv
│       ├── skill_relationships.csv
│       ├── course_teaches_skill.csv
│       └── prerequisite_edges.csv
│
├── outputs/                        # Visualizations (NEW)
│   ├── skill_graph.png
│   └── prerequisite_graph.png
│
├── models/                         # Trained models (future)
└── README_PHASE0.md                # This file
```

---

## Quick Start - Phase 0

### Prerequisites

- Python 3.8+
- `coursera_courses.csv` and `augmented_learner_data.csv` in root directory

### Installation

```bash
# Install Phase 0 dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Run Complete Pipeline

```bash
# This runs all 3 steps automatically
python run_phase0.py
```

**Expected runtime:** 5-15 minutes depending on CPU

### Output

After successful execution, you'll have:

1. **Skill Graph** - Network of skills extracted from courses
2. **Prerequisite Graph** - Directed graph showing course dependencies
3. **Learner Trajectories** - 1,000 simulated realistic learning journeys
4. **Visualizations** - PNG files showing graphs
5. **Neo4j Data** - Ready for graph database import

---

## Phase 0 Components

### 1. Skill Extraction (`skill_extractor.py`)

**What it does:**
- Extracts skills from course titles and descriptions using NLP
- Maps skills to standardized taxonomy (based on O*NET)
- Builds skill co-occurrence graph
- Clusters similar skills

**Key Methods:**
- `extract_skills_from_text()` - NLP-based skill extraction
- `build_skill_graph()` - Create skill network
- `cluster_skills()` - Group similar skills

**Output:**
- `data/course_skills_raw.csv` - All course-skill mappings
- `data/skill_graph.gpickle` - NetworkX graph
- `outputs/skill_graph.png` - Visualization

**Run individually:**
```bash
python skill_extractor.py
```

---

### 2. Prerequisite Mining (`prerequisite_miner.py`)

**What it does:**
- Extracts explicit prerequisites from course descriptions
- Infers prerequisites based on difficulty progression
- Uses skill dependencies to infer relationships
- Validates and removes circular dependencies

**Key Methods:**
- `extract_explicit_prerequisites()` - Parse text for prereqs
- `infer_difficulty_based_prerequisites()` - Use difficulty levels
- `infer_skill_based_prerequisites()` - Use skill overlap
- `build_prerequisite_graph()` - Create DAG (Directed Acyclic Graph)

**Output:**
- `data/course_prerequisites.csv` - All prerequisite relationships
- `data/prerequisite_graph.gpickle` - NetworkX DiGraph
- `outputs/prerequisite_graph.png` - Visualization

**Run individually:**
```bash
python prerequisite_miner.py
```

---

### 3. Learner Trajectory Simulation (`trajectory_simulator.py`)

**What it does:**
- Simulates realistic learner behaviors using archetypes
- Generates course sequences respecting prerequisites
- Models quiz scores, engagement, and dropout
- Creates temporal dynamics (time between courses)

**Learner Archetypes:**
- **High Achiever** (15%) - Fast learner, high scores, low dropout
- **Steady Learner** (40%) - Consistent progress, moderate scores
- **Struggling Learner** (20%) - Lower scores, high dropout
- **Explorer** (15%) - Diverse interests, variable completion
- **Focused Specialist** (10%) - Deep dive into one domain

**Output:**
- `data/learner_trajectories.csv` - Full trajectories with metadata
- `data/augmented_learner_data_v2.csv` - Compatible with existing system

**Run individually:**
```bash
python trajectory_simulator.py
```

---

## Data Schema

### course_skills_raw.csv
```csv
course_id,course_title,skill,skill_category,skill_difficulty,confidence,extraction_method
C0001,Intro to ML,machine learning,data_science,intermediate,0.9,taxonomy_match
```

### course_prerequisites.csv
```csv
course_id,prereq_course_id,prereq_course_title,confidence,methods
C0100,C0001,Intro to ML,0.85,"explicit_match, difficulty_based"
```

### learner_trajectories.csv
```csv
learner_id,course_id,enrollment_date,quiz_score,engagement_score,completed,time_spent_hours,archetype
L00001,C0001,2024-01-15,82.5,78.3,True,24.5,high_achiever
```

---

## Data Validation

After running Phase 0, validate your data:

```python
import pandas as pd
import networkx as nx

# Check skills
skills = pd.read_csv('data/course_skills_raw.csv')
print(f"Total skills: {skills['skill'].nunique()}")
print(f"Courses with skills: {skills['course_id'].nunique()}")

# Check prerequisites
prereqs = pd.read_csv('data/course_prerequisites.csv')
print(f"Prerequisite relationships: {len(prereqs)}")

# Check graph
G = nx.read_gpickle('data/prerequisite_graph.gpickle')
print(f"Graph nodes: {G.number_of_nodes()}")
print(f"Graph edges: {G.number_of_edges()}")
print(f"Is DAG? {nx.is_directed_acyclic_graph(G)}")

# Check trajectories
traj = pd.read_csv('data/learner_trajectories.csv')
print(f"Learners: {traj['learner_id'].nunique()}")
print(f"Avg courses per learner: {len(traj) / traj['learner_id'].nunique():.1f}")
print(f"Completion rate: {traj['completed'].mean():.1%}")
```

---

## Visualization

The generated visualizations show:

### Skill Graph (`outputs/skill_graph.png`)
- Nodes = Skills (colored by category)
- Edges = Co-occurrence in courses
- Node size = Frequency in catalog

### Prerequisite Graph (`outputs/prerequisite_graph.png`)
- Nodes = Courses (colored by difficulty)
- Edges = Prerequisite relationships (arrows point from prereq → course)
- Shows top 30 most connected courses

---

##  Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### spaCy model missing
```bash
python -m spacy download en_core_web_sm
```

### Empty graphs or low extraction rates

**Symptom:** Very few skills or prerequisites extracted

**Solution:**
1. Check `coursera_courses.csv` has `title` and `description` columns
2. Descriptions should be detailed (>50 words)
3. Lower thresholds in code:
   ```python
   # In prerequisite_miner.py, line ~260
   if confidence >= 0.2:  # Lower from 0.3
   ```

### Memory errors with large datasets

**Solution:**
1. Reduce number of simulated learners:
   ```python
   # In trajectory_simulator.py main()
   num_learners=500  # Instead of 1000
   ```

2. Process courses in batches:
   ```python
   # In skill_extractor.py
   for i in range(0, len(courses_df), 100):
       batch = courses_df.iloc[i:i+100]
       # Process batch
   ```

---

## Next Steps: Phase 1

After completing Phase 0, you'll move to **Phase 1: Knowledge Tracing** (Week 3-4)

**Phase 1 will implement:**
1. Bayesian Knowledge Tracing (BKT)
2. Deep Knowledge Tracing (DKT)
3. Learner skill state modeling
4. Cognitive load estimation

**Stay tuned for:**
- `knowledge_tracing.py`
- `learner_model.py`
- `cognitive_load.py`
- Updated Streamlit app with skill radar

---

##  Contributing

This is a research project aiming for publication. If you want to contribute:

1. Focus on improving extraction accuracy
2. Add more sophisticated prerequisite inference
3. Expand learner archetypes with real data
4. Improve graph visualizations

---

## Contact

For questions or collaboration:
- Create an issue on GitHub
- Email: surajphaniharam@gmail.com

---



