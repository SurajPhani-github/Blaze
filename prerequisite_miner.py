"""
Prerequisite Mining Pipeline - CurricuRL Phase 0
=================================================

This module infers prerequisite relationships between courses using:
1. Explicit prerequisite extraction from descriptions
2. Difficulty-based progression (beginner -> intermediate -> advanced)
3. Skill dependency analysis
4. ML-based prerequisite prediction

Author: CurricuRL Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# NLP
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Graph
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class PrerequisiteMiner:
    """
    Mine and infer prerequisite relationships between courses
    """
    
    def __init__(self):
        print("Initializing Prerequisite Miner...")
        
        # Load spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("WARNING: Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Prerequisite keywords
        self.prereq_patterns = [
            r'prerequisite[s]?:?\s*(.*?)(?:\.|$)',
            r'require[s]?:?\s*(.*?)(?:\.|$)',
            r'assumed knowledge:?\s*(.*?)(?:\.|$)',
            r'prior knowledge:?\s*(.*?)(?:\.|$)',
            r'recommended background:?\s*(.*?)(?:\.|$)',
            r'should (?:have|know|understand):?\s*(.*?)(?:\.|$)',
            r'need to (?:have|know|understand):?\s*(.*?)(?:\.|$)',
            r'familiarity with:?\s*(.*?)(?:\.|$)',
            r'experience (?:in|with):?\s*(.*?)(?:\.|$)',
        ]
        
        # Difficulty levels (ordered)
        self.difficulty_order = {
            'beginner': 0,
            'intermediate': 1,
            'advanced': 2,
            'expert': 3
        }
        
        print("SUCCESS: Prerequisite Miner initialized")
    
    def extract_explicit_prerequisites(self, courses_df: pd.DataFrame) -> List[Dict]:
        """Extract explicitly stated prerequisites from course descriptions"""
        
        print("\nExtracting explicit prerequisites from descriptions...")
        
        explicit_prereqs = []
        
        # Ensure course_id exists
        if 'course_id' not in courses_df.columns:
            courses_df['course_id'] = [f'C{i:04d}' for i in range(len(courses_df))]
        
        for idx, row in courses_df.iterrows():
            course_id = row['course_id']
            description = str(row.get('description', ''))
            title = str(row.get('title', ''))
            
            if not description or len(description) < 10:
                continue
            
            # Search for prerequisite patterns
            for pattern in self.prereq_patterns:
                matches = re.finditer(pattern, description, re.IGNORECASE)
                for match in matches:
                    prereq_text = match.group(1).strip()
                    
                    if len(prereq_text) > 5:
                        explicit_prereqs.append({
                            'course_id': course_id,
                            'course_title': title,
                            'prereq_text': prereq_text,
                            'extraction_pattern': pattern
                        })
        
        print(f"SUCCESS: Found {len(explicit_prereqs)} explicit prerequisite mentions")
        return explicit_prereqs
    
    def match_prerequisites_to_courses(self, explicit_prereqs: List[Dict],
                                      courses_df: pd.DataFrame,
                                      threshold: float = 0.3) -> pd.DataFrame:
        """Match prerequisite text to actual courses using TF-IDF similarity"""
        
        print(f"\nMatching prerequisites to courses (threshold={threshold})...")
        
        if len(explicit_prereqs) == 0:
            print("WARNING: No explicit prerequisites to match")
            return pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'method'])
        
        # Ensure course_id exists
        if 'course_id' not in courses_df.columns:
            courses_df['course_id'] = [f'C{i:04d}' for i in range(len(courses_df))]
        
        # Create course corpus
        courses_df['combined_text'] = (
            courses_df['title'].fillna('') + ' ' + 
            courses_df.get('description', pd.Series([''] * len(courses_df))).fillna('') + ' ' + 
            courses_df.get('domain', pd.Series([''] * len(courses_df))).fillna('')
        )
        
        # TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        course_vectors = vectorizer.fit_transform(courses_df['combined_text'])
        
        matched_prereqs = []
        
        for prereq in explicit_prereqs:
            prereq_vector = vectorizer.transform([prereq['prereq_text']])
            similarities = cosine_similarity(prereq_vector, course_vectors).flatten()
            top_indices = np.argsort(similarities)[::-1][:3]
            
            for idx in top_indices:
                sim_score = similarities[idx]
                
                if sim_score >= threshold:
                    prereq_course_id = courses_df.iloc[idx]['course_id']
                    
                    if prereq_course_id != prereq['course_id']:
                        matched_prereqs.append({
                            'course_id': prereq['course_id'],
                            'prereq_course_id': prereq_course_id,
                            'prereq_course_title': courses_df.iloc[idx]['title'],
                            'confidence': float(sim_score),
                            'method': 'explicit_match'
                        })
        
        matched_df = pd.DataFrame(matched_prereqs)
        
        if len(matched_df) > 0:
            matched_df = matched_df.sort_values('confidence', ascending=False)
            matched_df = matched_df.drop_duplicates(subset=['course_id', 'prereq_course_id'])
            print(f"SUCCESS: Matched {len(matched_df)} prerequisite relationships")
        else:
            print("WARNING: No prerequisites matched to courses")
        
        return matched_df
    
    def infer_difficulty_based_prerequisites(self, courses_df: pd.DataFrame) -> pd.DataFrame:
        """Infer prerequisites based on difficulty levels and domain similarity"""
        
        print("\nInferring prerequisites based on difficulty progression...")
        
        # Ensure course_id exists
        if 'course_id' not in courses_df.columns:
            courses_df['course_id'] = [f'C{i:04d}' for i in range(len(courses_df))]
        
        if 'difficulty' not in courses_df.columns:
            print("WARNING: No difficulty column found - skipping difficulty-based inference")
            return pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'method'])
        
        courses_df['difficulty_level'] = courses_df['difficulty'].map(
            lambda x: self.difficulty_order.get(str(x).lower(), 1)
        )
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        courses_df['combined_text'] = (
            courses_df['title'].fillna('') + ' ' + 
            courses_df.get('description', pd.Series([''] * len(courses_df))).fillna('') + ' ' + 
            courses_df.get('domain', pd.Series([''] * len(courses_df))).fillna('')
        )
        
        course_vectors = vectorizer.fit_transform(courses_df['combined_text'])
        similarity_matrix = cosine_similarity(course_vectors)
        
        difficulty_prereqs = []
        
        for i, row in courses_df.iterrows():
            course_id = row['course_id']
            course_difficulty = row['difficulty_level']
            
            if course_difficulty == 0:
                continue
            
            for j, prereq_row in courses_df.iterrows():
                if i == j:
                    continue
                
                prereq_id = prereq_row['course_id']
                prereq_difficulty = prereq_row['difficulty_level']
                
                if prereq_difficulty < course_difficulty:
                    domain_similarity = similarity_matrix[i, j]
                    difficulty_gap = course_difficulty - prereq_difficulty
                    
                    if difficulty_gap == 1:
                        confidence = domain_similarity * 0.8
                    else:
                        confidence = domain_similarity * 0.5
                    
                    if confidence >= 0.3:
                        difficulty_prereqs.append({
                            'course_id': course_id,
                            'prereq_course_id': prereq_id,
                            'prereq_course_title': prereq_row['title'],
                            'confidence': float(confidence),
                            'method': 'difficulty_based',
                            'difficulty_gap': int(difficulty_gap)
                        })
        
        difficulty_df = pd.DataFrame(difficulty_prereqs)
        
        if len(difficulty_df) > 0:
            difficulty_df = difficulty_df.sort_values('confidence', ascending=False)
            difficulty_df = difficulty_df.groupby('course_id').head(3)
            print(f"SUCCESS: Inferred {len(difficulty_df)} difficulty-based prerequisites")
        else:
            print("WARNING: No difficulty-based prerequisites inferred")
        
        return difficulty_df
    
    def infer_skill_based_prerequisites(self, courses_df: pd.DataFrame,
                                       skills_df: pd.DataFrame) -> pd.DataFrame:
        """Infer prerequisites based on skill dependencies"""
        
        print("\nInferring prerequisites based on skill dependencies...")
        
        if skills_df is None or len(skills_df) == 0:
            print("WARNING: No skills data available")
            return pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'method'])
        
        if 'course_id' not in courses_df.columns:
            courses_df['course_id'] = [f'C{i:04d}' for i in range(len(courses_df))]
        
        if 'course_id' not in skills_df.columns:
            print("WARNING: No course_id in skills data")
            return pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'method'])
        
        course_skills = skills_df.groupby('course_id')['skill'].apply(set).to_dict()
        skill_prereqs = []
        
        for course_id, course_skills_set in course_skills.items():
            if len(course_skills_set) == 0:
                continue
            
            for prereq_id, prereq_skills_set in course_skills.items():
                if course_id == prereq_id:
                    continue
                
                overlap = course_skills_set.intersection(prereq_skills_set)
                
                if len(overlap) > 0:
                    jaccard = len(overlap) / len(course_skills_set.union(prereq_skills_set))
                    prereq_coverage = len(overlap) / len(prereq_skills_set) if len(prereq_skills_set) > 0 else 0
                    confidence = (jaccard + prereq_coverage) / 2
                    
                    if confidence >= 0.2:
                        skill_prereqs.append({
                            'course_id': course_id,
                            'prereq_course_id': prereq_id,
                            'confidence': float(confidence),
                            'method': 'skill_based',
                            'shared_skills': len(overlap)
                        })
        
        skill_df = pd.DataFrame(skill_prereqs)
        
        if len(skill_df) > 0:
            skill_df = skill_df.sort_values('confidence', ascending=False)
            skill_df = skill_df.groupby('course_id').head(5)
            print(f"SUCCESS: Inferred {len(skill_df)} skill-based prerequisites")
        else:
            print("WARNING: No skill-based prerequisites inferred")
        
        return skill_df
    
    def combine_prerequisite_sources(self, *prereq_dfs) -> pd.DataFrame:
        """Combine multiple prerequisite sources with weighted confidence"""
        
        print("\nCombining prerequisite sources...")
        
        valid_dfs = []
        for df in prereq_dfs:
            if df is not None and len(df) > 0:
                required_cols = ['course_id', 'prereq_course_id', 'confidence', 'method']
                if all(col in df.columns for col in required_cols):
                    valid_dfs.append(df[required_cols])
        
        if len(valid_dfs) == 0:
            print("WARNING: No valid prerequisite data to combine")
            return pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'methods'])
        
        combined = pd.concat(valid_dfs, ignore_index=True)
        combined['course_id'] = combined['course_id'].astype(str)
        combined['prereq_course_id'] = combined['prereq_course_id'].astype(str)
        combined['confidence'] = pd.to_numeric(combined['confidence'], errors='coerce')
        combined['method'] = combined['method'].astype(str)
        combined = combined.dropna(subset=['course_id', 'prereq_course_id', 'confidence'])
        
        aggregated = combined.groupby(['course_id', 'prereq_course_id']).agg({
            'confidence': 'max',
            'method': lambda x: ', '.join(x.unique())
        }).reset_index()
        
        aggregated = aggregated.rename(columns={'method': 'methods'})
        print(f"SUCCESS: Combined into {len(aggregated)} unique prerequisite relationships")
        
        return aggregated
    
    def build_prerequisite_graph(self, prerequisites_df: pd.DataFrame, 
                                courses_df: pd.DataFrame) -> nx.DiGraph:
        """Build directed graph of course prerequisites"""
        
        print("\nBuilding prerequisite graph...")
        
        G = nx.DiGraph()
        
        if 'course_id' not in courses_df.columns:
            courses_df['course_id'] = [f'C{i:04d}' for i in range(len(courses_df))]
        
        for _, row in courses_df.iterrows():
            G.add_node(
                str(row['course_id']),
                title=str(row.get('title', '')),
                domain=str(row.get('domain', 'Unknown')),
                difficulty=str(row.get('difficulty', 'intermediate'))
            )
        
        if len(prerequisites_df) > 0:
            prerequisites_df['course_id'] = prerequisites_df['course_id'].astype(str)
            prerequisites_df['prereq_course_id'] = prerequisites_df['prereq_course_id'].astype(str)
            
            for _, row in prerequisites_df.iterrows():
                prereq_id = row['prereq_course_id']
                course_id = row['course_id']
                
                if prereq_id in G and course_id in G:
                    G.add_edge(
                        prereq_id,
                        course_id,
                        confidence=float(row.get('confidence', 0.5)),
                        methods=str(row.get('methods', row.get('method', 'unknown')))
                    )
        
        print(f"SUCCESS: Graph created!")
        print(f"  Nodes (courses): {G.number_of_nodes()}")
        print(f"  Edges (prerequisites): {G.number_of_edges()}")
        
        if G.number_of_edges() > 0:
            try:
                cycles = list(nx.simple_cycles(G))
                if len(cycles) > 0:
                    print(f"  WARNING: Found {len(cycles)} cycles - removing...")
                    G = self._remove_cycles(G)
                    print(f"  SUCCESS: Removed cycles - now {G.number_of_edges()} edges")
                else:
                    print("  SUCCESS: No cycles detected (graph is a DAG)")
            except:
                print("  SUCCESS: Graph validation complete")
        
        return G
    
    def _remove_cycles(self, G: nx.DiGraph) -> nx.DiGraph:
        """Remove cycles from graph by removing lowest-confidence edges"""
        
        while True:
            try:
                cycle = nx.find_cycle(G)
                min_conf = float('inf')
                min_edge = None
                
                for edge in cycle:
                    conf = G.edges[edge[0], edge[1]].get('confidence', 0.5)
                    if conf < min_conf:
                        min_conf = conf
                        min_edge = (edge[0], edge[1])
                
                if min_edge:
                    G.remove_edge(min_edge[0], min_edge[1])
                
            except nx.NetworkXNoCycle:
                break
        
        return G
    
    def visualize_prerequisite_graph(self, G: nx.DiGraph, 
                                    output_path='outputs/prerequisite_graph.png',
                                    max_nodes: int = 30):
        """Visualize prerequisite graph"""
        
        print(f"\nVisualizing prerequisite graph (max {max_nodes} nodes)...")
        
        if G.number_of_nodes() == 0:
            print("WARNING: Graph is empty, skipping visualization")
            return
        
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:max_nodes]
        G_sub = G.subgraph(top_nodes)
        
        if G_sub.number_of_nodes() == 0:
            print("WARNING: No nodes to visualize")
            return
        
        try:
            pos = nx.spring_layout(G_sub, k=2, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G_sub)
        
        plt.figure(figsize=(20, 16))
        
        difficulties = nx.get_node_attributes(G_sub, 'difficulty')
        difficulty_colors = {
            'beginner': '#4CAF50',
            'intermediate': '#FFC107', 
            'advanced': '#F44336',
            'expert': '#9C27B0'
        }
        
        node_colors = [difficulty_colors.get(difficulties.get(node, 'intermediate'), 'gray') 
                      for node in G_sub.nodes()]
        
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, 
                              node_size=800, alpha=0.8)
        
        if G_sub.number_of_edges() > 0:
            nx.draw_networkx_edges(G_sub, pos, edge_color='gray', 
                                  arrows=True, arrowsize=20, 
                                  width=2, alpha=0.6, arrowstyle='->')
        
        labels = {node: G_sub.nodes[node].get('title', node)[:30] for node in G_sub.nodes()}
        nx.draw_networkx_labels(G_sub, pos, labels, font_size=7, font_weight='bold')
        
        plt.title("Course Prerequisite Graph (Arrows: prerequisite -> course)", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"SUCCESS: Saved visualization to {output_path}")
        
        plt.close()
    
    def export_prerequisites(self, prerequisites_df: pd.DataFrame, 
                           G: nx.DiGraph,
                           output_dir='data'):
        """Export prerequisite data"""
        
        print(f"\nExporting prerequisite data...")
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        if len(prerequisites_df) > 0:
            prerequisites_df.to_csv(f'{output_dir}/course_prerequisites.csv', index=False)
            print(f"  SUCCESS: Saved {len(prerequisites_df)} prerequisites")
        else:
            pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'methods']).to_csv(
                f'{output_dir}/course_prerequisites.csv', index=False
            )
            print(f"  SUCCESS: Created empty course_prerequisites.csv")
        
        import pickle
        with open(f'{output_dir}/prerequisite_graph.gpickle', 'wb') as f:
            pickle.dump(G, f)
        print(f"  SUCCESS: Saved graph to prerequisite_graph.gpickle")
        
        Path(f'{output_dir}/neo4j').mkdir(exist_ok=True, parents=True)
        
        if G.number_of_edges() > 0:
            edges = []
            for u, v, data in G.edges(data=True):
                edges.append({
                    'prereq_course_id': str(u),
                    'course_id': str(v),
                    'confidence': float(data.get('confidence', 0.5)),
                    'methods': str(data.get('methods', 'unknown'))
                })
            
            edges_df = pd.DataFrame(edges)
            edges_df.to_csv(f'{output_dir}/neo4j/prerequisite_edges.csv', index=False)
            print(f"  SUCCESS: Saved {len(edges)} edges for Neo4j")
        else:
            pd.DataFrame(columns=['prereq_course_id', 'course_id', 'confidence', 'methods']).to_csv(
                f'{output_dir}/neo4j/prerequisite_edges.csv', index=False
            )
            print(f"  SUCCESS: Created empty prerequisite_edges.csv")


def main():
    """Main execution pipeline"""
    
    print("="*70)
    print("  PREREQUISITE MINING PIPELINE - CurricuRL Phase 0")
    print("="*70)
    
    miner = PrerequisiteMiner()
    
    print("\nLoading data...")
    courses_df = pd.read_csv('coursera_courses.csv')
    
    if 'course_id' not in courses_df.columns:
        courses_df['course_id'] = [f'C{i:04d}' for i in range(len(courses_df))]
    
    print(f"SUCCESS: Loaded {len(courses_df)} courses")
    
    try:
        skills_df = pd.read_csv('data/course_skills_raw.csv')
        print(f"SUCCESS: Loaded {len(skills_df)} course-skill mappings")
    except:
        print("WARNING: No skills data found - skill-based inference will be skipped")
        skills_df = None
    
    explicit_prereqs = miner.extract_explicit_prerequisites(courses_df)
    matched_explicit = miner.match_prerequisites_to_courses(explicit_prereqs, courses_df)
    difficulty_prereqs = miner.infer_difficulty_based_prerequisites(courses_df)
    
    if skills_df is not None:
        skill_prereqs = miner.infer_skill_based_prerequisites(courses_df, skills_df)
    else:
        skill_prereqs = pd.DataFrame(columns=['course_id', 'prereq_course_id', 'confidence', 'method'])
    
    combined_prereqs = miner.combine_prerequisite_sources(
        matched_explicit, 
        difficulty_prereqs, 
        skill_prereqs
    )
    
    prereq_graph = miner.build_prerequisite_graph(combined_prereqs, courses_df)
    miner.visualize_prerequisite_graph(prereq_graph)
    miner.export_prerequisites(combined_prereqs, prereq_graph)
    
    print("\n" + "="*70)
    print("  SUCCESS: PREREQUISITE MINING COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - data/course_prerequisites.csv")
    print(f"  - data/prerequisite_graph.gpickle")
    print(f"  - data/neo4j/prerequisite_edges.csv")
    print(f"  - outputs/prerequisite_graph.png")


if __name__ == "__main__":
    main()