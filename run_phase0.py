"""
Phase 0 Orchestrator - CurricuRL Data Enrichment
=================================================

Runs the complete Phase 0 pipeline:
1. Skill extraction from courses
2. Prerequisite mining
3. Learner trajectory simulation
4. Data validation and export

Author: CurricuRL Team
Date: January 2026
"""

import sys
from pathlib import Path
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required packages are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required = [
        'pandas', 'numpy', 'spacy', 'nltk', 
        'sklearn', 'networkx', 'matplotlib'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    # Check spaCy model
    try:
        import spacy
        spacy.load('en_core_web_sm')
        print("✓ spaCy model: en_core_web_sm")
    except:
        print("✗ spaCy model missing")
        print("Download with: python -m spacy download en_core_web_sm")
        return False
    
    print("\n✓ All dependencies satisfied!")
    return True

def check_input_files():
    """Check if required input files exist"""
    print_header("CHECKING INPUT FILES")
    
    required_files = [
        'coursera_courses.csv',
        'augmented_learner_data.csv'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Missing required input files!")
        print("Please ensure coursera_courses.csv and augmented_learner_data.csv are in the current directory")
        return False
    
    print("\n✓ All input files found!")
    return True

def create_directories():
    """Create output directories"""
    print_header("CREATING OUTPUT DIRECTORIES")
    
    dirs = ['data', 'data/neo4j', 'outputs', 'models']
    
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✓ {dir_path}/")
    
    print("\n✓ Directories created!")

def run_skill_extraction():
    """Run skill extraction pipeline"""
    print_header("STEP 1: SKILL EXTRACTION")
    
    try:
        from skill_extractor import main as skill_main
        skill_main()
        return True
    except Exception as e:
        print(f"\n✗ Error in skill extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_prerequisite_mining():
    """Run prerequisite mining pipeline"""
    print_header("STEP 2: PREREQUISITE MINING")
    
    try:
        from prerequisite_miner import main as prereq_main
        prereq_main()
        return True
    except Exception as e:
        print(f"\n✗ Error in prerequisite mining: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_trajectory_simulation():
    """Run trajectory simulation pipeline"""
    print_header("STEP 3: LEARNER TRAJECTORY SIMULATION")
    
    try:
        from trajectory_simulator import main as traj_main
        traj_main()
        return True
    except Exception as e:
        print(f"\n✗ Error in trajectory simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report():
    """Generate summary report of Phase 0 outputs"""
    print_header("PHASE 0 SUMMARY REPORT")
    
    import pandas as pd
    import networkx as nx
    
    try:
        # Load generated data
        skills_df = pd.read_csv('data/course_skills_raw.csv')
        prereqs_df = pd.read_csv('data/course_prerequisites.csv')
        trajectories_df = pd.read_csv('data/learner_trajectories.csv')
        
        import pickle
        with open('data/skill_graph.gpickle', 'rb') as f:
            skill_graph = pickle.load(f)
        with open('data/prerequisite_graph.gpickle', 'rb') as f:
            prereq_graph = pickle.load(f)
        
        print("📊 Data Statistics:")
        print(f"\n  Skills:")
        print(f"    • Course-skill mappings: {len(skills_df):,}")
        print(f"    • Unique skills: {skills_df['skill'].nunique():,}")
        print(f"    • Skill graph nodes: {skill_graph.number_of_nodes():,}")
        print(f"    • Skill graph edges: {skill_graph.number_of_edges():,}")
        
        print(f"\n  Prerequisites:")
        print(f"    • Prerequisite relationships: {len(prereqs_df):,}")
        print(f"    • Courses with prerequisites: {prereqs_df['course_id'].nunique():,}")
        print(f"    • Prerequisite graph nodes: {prereq_graph.number_of_nodes():,}")
        print(f"    • Prerequisite graph edges: {prereq_graph.number_of_edges():,}")
        
        print(f"\n  Learner Trajectories:")
        print(f"    • Total interactions: {len(trajectories_df):,}")
        print(f"    • Unique learners: {trajectories_df['learner_id'].nunique():,}")
        print(f"    • Unique courses taken: {trajectories_df['course_id'].nunique():,}")
        print(f"    • Avg courses per learner: {len(trajectories_df) / trajectories_df['learner_id'].nunique():.1f}")
        print(f"    • Overall completion rate: {trajectories_df['completed'].mean():.1%}")
        print(f"    • Avg quiz score: {trajectories_df['quiz_score'].mean():.1f}")
        
        print("\n📁 Generated Files:")
        files = [
            'data/course_skills_raw.csv',
            'data/course_skills_clustered.csv',
            'data/skill_graph.gpickle',
            'data/course_prerequisites.csv',
            'data/prerequisite_graph.gpickle',
            'data/learner_trajectories.csv',
            'data/augmented_learner_data_v2.csv',
            'outputs/skill_graph.png',
            'outputs/prerequisite_graph.png',
            'data/neo4j/skills_nodes.csv',
            'data/neo4j/skill_relationships.csv',
            'data/neo4j/course_teaches_skill.csv',
            'data/neo4j/prerequisite_edges.csv'
        ]
        
        for file in files:
            if Path(file).exists():
                size = Path(file).stat().st_size / 1024  # KB
                print(f"    ✓ {file} ({size:.1f} KB)")
            else:
                print(f"    ✗ {file} - MISSING")
        
        print("\n✨ Phase 0 Complete! Ready for Phase 1.")
        
    except Exception as e:
        print(f"\n⚠️  Could not generate full report: {e}")

def main():
    """
    Main orchestration pipeline
    """
    
    start_time = time.time()
    
    print("="*70)
    print("  CurricuRL Phase 0: Data Enrichment Pipeline")
    print("  Building Knowledge Graphs for Curriculum-Aware RL")
    print("="*70)
    
    # Step 0: Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Step 1: Check input files
    if not check_input_files():
        sys.exit(1)
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Run skill extraction
    if not run_skill_extraction():
        print("\n❌ Skill extraction failed. Please fix errors and retry.")
        sys.exit(1)
    
    # Step 4: Run prerequisite mining
    if not run_prerequisite_mining():
        print("\n Prerequisite mining failed. Please fix errors and retry.")
        sys.exit(1)
    
    # Step 5: Run trajectory simulation
    if not run_trajectory_simulation():
        print("\n Trajectory simulation failed. Please fix errors and retry.")
        sys.exit(1)
    
    # Step 6: Generate summary
    generate_summary_report()
    
    # Final timing
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n⏱️  Total execution time: {minutes}m {seconds}s")
    
    print("\n" + "="*70)
    print("  ✅ PHASE 0 COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review generated graphs in outputs/")
    print("  2. Validate data in data/")
    print("  3. Proceed to Phase 1: Knowledge Tracing")
    print("  4. Run: python app.py (to visualize in Streamlit)")
    print("\nDocumentation: See README.md for details")


if __name__ == "__main__":
    main()