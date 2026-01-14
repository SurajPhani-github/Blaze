"""
Quick hotfix for NetworkX compatibility
Run this to update all files for NetworkX 3.x compatibility
"""

import re
from pathlib import Path

def fix_file(filepath, replacements):
    """Apply replacements to a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    for old, new in replacements:
        content = content.replace(old, new)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Fixed {filepath}")
        return True
    else:
        print(f"  {filepath} already up to date")
        return False

def main():
    print("🔧 Applying NetworkX 3.x compatibility fixes...")
    print()
    
    # Replacements for write_gpickle
    write_replacements = [
        (
            "    nx.write_gpickle(skill_graph, 'data/skill_graph.gpickle')\n    print(f\"💾 Saved graph to data/skill_graph.gpickle\")",
            "    import pickle\n    with open('data/skill_graph.gpickle', 'wb') as f:\n        pickle.dump(skill_graph, f)\n    print(f\"💾 Saved graph to data/skill_graph.gpickle\")"
        ),
        (
            "        nx.write_gpickle(G, f'{output_dir}/prerequisite_graph.gpickle')\n        print(f\"  ✓ Saved graph to prerequisite_graph.gpickle\")",
            "        import pickle\n        with open(f'{output_dir}/prerequisite_graph.gpickle', 'wb') as f:\n            pickle.dump(G, f)\n        print(f\"  ✓ Saved graph to prerequisite_graph.gpickle\")"
        )
    ]
    
    # Replacements for read_gpickle
    read_replacements = [
        (
            "        prereq_graph = nx.read_gpickle('data/prerequisite_graph.gpickle')\n        print(f\"✓ Loaded prerequisite graph\")",
            "        import pickle\n        with open('data/prerequisite_graph.gpickle', 'rb') as f:\n            prereq_graph = pickle.load(f)\n        print(f\"✓ Loaded prerequisite graph\")"
        ),
        (
            "        skill_graph = nx.read_gpickle('data/skill_graph.gpickle')\n        prereq_graph = nx.read_gpickle('data/prerequisite_graph.gpickle')",
            "        import pickle\n        with open('data/skill_graph.gpickle', 'rb') as f:\n            skill_graph = pickle.load(f)\n        with open('data/prerequisite_graph.gpickle', 'rb') as f:\n            prereq_graph = pickle.load(f)"
        )
    ]
    
    # Files to fix
    files = [
        ('skill_extractor.py', write_replacements[:1]),
        ('prerequisite_miner.py', write_replacements[1:2] + read_replacements[:1]),
        ('trajectory_simulator.py', read_replacements[:1]),
        ('run_phase0.py', read_replacements[1:])
    ]
    
    fixed_count = 0
    for filepath, replacements in files:
        if Path(filepath).exists():
            if fix_file(filepath, replacements):
                fixed_count += 1
        else:
            print(f"⚠️  {filepath} not found")
    
    print()
    print("="*60)
    if fixed_count > 0:
        print(f"✅ Fixed {fixed_count} file(s)")
        print("You can now run: python run_phase0.py")
    else:
        print("✅ All files already up to date")
    print("="*60)

if __name__ == "__main__":
    main()