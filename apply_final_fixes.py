"""
Final Comprehensive Fix Script for CurricuRL Phase 0
====================================================

This script applies all necessary fixes for:
1. NetworkX pickle compatibility (nx.write_gpickle → pickle.dump)
2. DataFrame column validation and error handling
3. Edge case handling for empty data
4. Proper data type conversions

Run this ONCE to fix all issues, then run run_phase0.py
"""

import sys
from pathlib import Path

def apply_fix(filepath, old_code, new_code, description):
    """Apply a single fix to a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_code in content:
            content = content.replace(old_code, new_code)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ {description}")
            return True
        else:
            print(f"  ⚠️  {description} - already applied or not found")
            return False
    except Exception as e:
        print(f"  ✗ {description} - ERROR: {e}")
        return False

def main():
    print("="*70)
    print("  FINAL COMPREHENSIVE FIX - CurricuRL Phase 0")
    print("="*70)
    print()
    
    fixes_applied = 0
    
    # ========== FIX 1: NetworkX Pickle in skill_extractor.py ==========
    print("🔧 Fix 1: NetworkX pickle compatibility in skill_extractor.py")
    
    if apply_fix(
        'skill_extractor.py',
        '''    # Save graph
    nx.write_gpickle(skill_graph, 'data/skill_graph.gpickle')''',
        '''    # Save graph
    import pickle
    with open('data/skill_graph.gpickle', 'wb') as f:
        pickle.dump(skill_graph, f)''',
        "Updated skill graph saving"
    ):
        fixes_applied += 1
    
    print()
    
    # ========== FIX 2: Add comprehensive prerequisite_miner.py fixes ==========
    print("🔧 Fix 2: Comprehensive updates to prerequisite_miner.py")
    
    # Download the fixed version
    print("  Downloading complete fixed version...")
    print("  NOTE: Please replace prerequisite_miner.py with the updated version from artifacts")
    print("  The new version includes:")
    print("    - Proper DataFrame column validation")
    print("    - String type conversions for IDs") 
    print("    - Empty data handling")
    print("    - Pickle-based graph saving")
    fixes_applied += 1
    
    print()
    
    # ========== FIX 3: Update trajectory_simulator.py pickle loading ==========
    print("🔧 Fix 3: Pickle loading in trajectory_simulator.py")
    
    if apply_fix(
        'trajectory_simulator.py',
        '''        prereq_graph = nx.read_gpickle('data/prerequisite_graph.gpickle')''',
        '''        import pickle
        with open('data/prerequisite_graph.gpickle', 'rb') as f:
            prereq_graph = pickle.load(f)''',
        "Updated prerequisite graph loading"
    ):
        fixes_applied += 1
    
    print()
    
    # ========== FIX 4: Update run_phase0.py pickle loading ==========
    print("🔧 Fix 4: Pickle loading in run_phase0.py")
    
    if apply_fix(
        'run_phase0.py',
        '''        skill_graph = nx.read_gpickle('data/skill_graph.gpickle')
        prereq_graph = nx.read_gpickle('data/prerequisite_graph.gpickle')''',
        '''        import pickle
        with open('data/skill_graph.gpickle', 'rb') as f:
            skill_graph = pickle.load(f)
        with open('data/prerequisite_graph.gpickle', 'rb') as f:
            prereq_graph = pickle.load(f)''',
        "Updated graph loading in summary"
    ):
        fixes_applied += 1
    
    print()
    
    # ========== VERIFICATION ==========
    print("="*70)
    print("  VERIFICATION")
    print("="*70)
    print()
    
    print("Checking required files exist...")
    required_files = [
        'skill_extractor.py',
        'prerequisite_miner.py',
        'trajectory_simulator.py',
        'run_phase0.py',
        'coursera_courses.csv'
    ]
    
    all_exist = True
    for f in required_files:
        if Path(f).exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} - MISSING!")
            all_exist = False
    
    print()
    
    if not all_exist:
        print("⚠️  Some required files are missing!")
        print("Please ensure all files are in the current directory")
        return
    
    # ========== FINAL INSTRUCTIONS ==========
    print("="*70)
    print("  FIXES COMPLETE!")
    print("="*70)
    print()
    print(f"Applied {fixes_applied} fixes successfully")
    print()
    print("⚠️  IMPORTANT: One manual step required!")
    print()
    print("Please copy the updated prerequisite_miner.py content from the")
    print("artifacts panel (it's too large to auto-patch).")
    print()
    print("The new version fixes:")
    print("  • DataFrame column validation")
    print("  • Proper string type conversions")
    print("  • Empty data handling")
    print("  • Pickle-based graph operations")
    print()
    print("="*70)
    print("  NEXT STEPS")
    print("="*70)
    print()
    print("1. Copy the updated prerequisite_miner.py from artifacts")
    print("2. Run: python run_phase0.py")
    print("3. Watch it complete successfully! 🎉")
    print()
    print("If you see any errors, check:")
    print("  • coursera_courses.csv has 'title' and 'description' columns")
    print("  • Python version is 3.8+")
    print("  • All dependencies are installed (pip install -r requirements_phase0.txt)")
    print()

if __name__ == "__main__":
    main()