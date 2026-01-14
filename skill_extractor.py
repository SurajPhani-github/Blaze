"""
Skill Extraction Pipeline - CurricuRL Phase 0
==============================================

This module extracts skills from course descriptions and maps them to
standardized taxonomies for building knowledge graphs.

Features:
1. NLP-based skill extraction from course text
2. Mapping to O*NET skill taxonomy
3. Skill clustering and hierarchy construction
4. Export to graph-ready format

Author: CurricuRL Team
Date: January 2026
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import json
from pathlib import Path

# NLP
import spacy
from spacy.matcher import PhraseMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Visualization
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SkillExtractor:
    """
    Extract and normalize skills from course descriptions
    """
    
    def __init__(self, taxonomy_path='data/onet_skills.json'):
        """
        Initialize skill extractor with taxonomy
        
        Args:
            taxonomy_path: Path to O*NET or custom skill taxonomy
        """
        print("🚀 Initializing Skill Extractor...")
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("✓ Loaded spaCy model: en_core_web_sm")
        except OSError:
            print("⚠️  Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.skill_taxonomy = self._load_or_create_taxonomy(taxonomy_path)
        self.phrase_matcher = self._build_phrase_matcher()
        
        # Skill patterns
        self.skill_indicators = [
            'learn', 'understand', 'master', 'develop', 'build',
            'create', 'design', 'implement', 'analyze', 'apply',
            'use', 'practice', 'gain', 'acquire', 'explore'
        ]
        
        print(f"✓ Loaded {len(self.skill_taxonomy)} skills from taxonomy")
    
    def _load_or_create_taxonomy(self, taxonomy_path):
        """Load existing taxonomy or create default one"""
        
        if Path(taxonomy_path).exists():
            with open(taxonomy_path, 'r') as f:
                taxonomy = json.load(f)
            print(f"✓ Loaded taxonomy from {taxonomy_path}")
            return taxonomy
        else:
            print("⚠️  No taxonomy found, creating default skill list...")
            # Default comprehensive skill taxonomy
            taxonomy = self._create_default_taxonomy()
            
            # Save for future use
            Path(taxonomy_path).parent.mkdir(exist_ok=True)
            with open(taxonomy_path, 'w') as f:
                json.dump(taxonomy, f, indent=2)
            
            print(f"✓ Created and saved taxonomy to {taxonomy_path}")
            return taxonomy
    
    def _create_default_taxonomy(self):
        """Create comprehensive skill taxonomy organized by domain"""
        
        taxonomy = {
            # Technical Skills
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust',
                'sql', 'r', 'matlab', 'scala', 'kotlin', 'swift', 'php', 'typescript',
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'data analysis', 'statistics',
                'data visualization', 'big data', 'predictive modeling', 'regression',
                'classification', 'clustering', 'neural networks', 'natural language processing',
                'computer vision', 'time series', 'feature engineering', 'model evaluation'
            ],
            'software_engineering': [
                'software development', 'web development', 'mobile development',
                'api development', 'microservices', 'devops', 'ci/cd', 'testing',
                'debugging', 'version control', 'git', 'agile', 'scrum', 'software architecture',
                'design patterns', 'object-oriented programming', 'functional programming'
            ],
            'cloud_computing': [
                'aws', 'azure', 'google cloud', 'cloud architecture', 'kubernetes',
                'docker', 'containerization', 'serverless', 'cloud security',
                'cloud migration', 'infrastructure as code', 'terraform'
            ],
            'cybersecurity': [
                'network security', 'cryptography', 'ethical hacking', 'penetration testing',
                'security analysis', 'risk assessment', 'compliance', 'incident response',
                'security architecture', 'vulnerability assessment'
            ],
            
            # Business & Management
            'business_analysis': [
                'business intelligence', 'data-driven decision making', 'strategic planning',
                'market analysis', 'competitive analysis', 'requirements gathering',
                'process improvement', 'stakeholder management'
            ],
            'project_management': [
                'project planning', 'risk management', 'resource allocation',
                'budgeting', 'scheduling', 'leadership', 'team management',
                'communication', 'negotiation', 'conflict resolution'
            ],
            'marketing': [
                'digital marketing', 'content marketing', 'social media marketing',
                'seo', 'sem', 'email marketing', 'marketing analytics',
                'brand management', 'customer acquisition', 'conversion optimization'
            ],
            'finance': [
                'financial analysis', 'accounting', 'financial modeling',
                'investment analysis', 'portfolio management', 'risk management',
                'corporate finance', 'valuation', 'financial reporting'
            ],
            
            # Design & Creative
            'design': [
                'ui design', 'ux design', 'graphic design', 'web design',
                'user research', 'prototyping', 'wireframing', 'design thinking',
                'visual design', 'interaction design', 'accessibility design'
            ],
            'multimedia': [
                'video editing', 'photography', 'animation', '3d modeling',
                'audio production', 'game design', 'illustration'
            ],
            
            # Core Competencies
            'analytical': [
                'critical thinking', 'problem solving', 'analytical thinking',
                'research', 'data interpretation', 'logical reasoning',
                'systems thinking', 'decision making'
            ],
            'communication': [
                'written communication', 'verbal communication', 'presentation skills',
                'technical writing', 'storytelling', 'active listening', 'collaboration'
            ],
            'mathematics': [
                'linear algebra', 'calculus', 'probability', 'optimization',
                'discrete mathematics', 'numerical analysis', 'mathematical modeling'
            ]
        }
        
        # Flatten and create skill metadata
        all_skills = {}
        for category, skills in taxonomy.items():
            for skill in skills:
                all_skills[skill] = {
                    'name': skill,
                    'category': category,
                    'aliases': [skill.replace('-', ' '), skill.replace(' ', '-')],
                    'difficulty': self._infer_skill_difficulty(skill, category)
                }
        
        return all_skills
    
    def _infer_skill_difficulty(self, skill, category):
        """Infer difficulty level of a skill"""
        
        advanced_indicators = [
            'deep learning', 'advanced', 'architecture', 'optimization',
            'kubernetes', 'microservices', 'neural networks', 'cryptography'
        ]
        
        beginner_indicators = [
            'introduction', 'basics', 'fundamentals', 'html', 'css'
        ]
        
        skill_lower = skill.lower()
        
        if any(ind in skill_lower for ind in advanced_indicators):
            return 'advanced'
        elif any(ind in skill_lower for ind in beginner_indicators):
            return 'beginner'
        else:
            return 'intermediate'
    
    def _build_phrase_matcher(self):
        """Build phrase matcher for known skills"""
        
        matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        patterns = [self.nlp.make_doc(skill) for skill in self.skill_taxonomy.keys()]
        matcher.add("SKILLS", patterns)
        
        return matcher
    
    def extract_skills_from_text(self, text: str) -> List[Dict]:
        """
        Extract skills from text using multiple strategies
        
        Args:
            text: Course description or content
            
        Returns:
            List of extracted skills with metadata
        """
        
        if not text or pd.isna(text):
            return []
        
        text = str(text).lower()
        doc = self.nlp(text)
        
        extracted_skills = []
        
        # Strategy 1: Phrase matching against taxonomy
        matches = self.phrase_matcher(doc)
        for match_id, start, end in matches:
            skill_text = doc[start:end].text
            if skill_text in self.skill_taxonomy:
                extracted_skills.append({
                    'skill': skill_text,
                    'source': 'taxonomy_match',
                    'confidence': 0.9,
                    **self.skill_taxonomy[skill_text]
                })
        
        # Strategy 2: Extract noun phrases near skill indicators
        for token in doc:
            if token.lemma_ in self.skill_indicators:
                # Look for noun phrases in vicinity
                for np in doc.noun_chunks:
                    if abs(np.start - token.i) <= 5:  # Within 5 tokens
                        skill_candidate = np.text.lower()
                        
                        # Filter out stopwords and common words
                        if (len(skill_candidate.split()) <= 4 and 
                            skill_candidate not in self.stop_words and
                            len(skill_candidate) > 3):
                            
                            extracted_skills.append({
                                'skill': skill_candidate,
                                'source': 'nlp_extraction',
                                'confidence': 0.6,
                                'category': 'extracted',
                                'difficulty': 'intermediate'
                            })
        
        # Strategy 3: Look for capitalized technical terms
        tech_terms = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE'] and len(ent.text) > 2:
                # Check if it's a known technology
                ent_lower = ent.text.lower()
                if ent_lower in self.skill_taxonomy:
                    tech_terms.append(ent_lower)
        
        for term in tech_terms:
            extracted_skills.append({
                'skill': term,
                'source': 'entity_recognition',
                'confidence': 0.7,
                **self.skill_taxonomy[term]
            })
        
        # Remove duplicates, keeping highest confidence
        unique_skills = {}
        for skill_dict in extracted_skills:
            skill_name = skill_dict['skill']
            if skill_name not in unique_skills or skill_dict['confidence'] > unique_skills[skill_name]['confidence']:
                unique_skills[skill_name] = skill_dict
        
        return list(unique_skills.values())
    
    def extract_skills_from_courses(self, courses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract skills from entire course catalog
        
        Args:
            courses_df: DataFrame with course information
            
        Returns:
            DataFrame with course_id, skill mappings
        """
        
        print("\n📚 Extracting skills from course catalog...")
        
        # Ensure required columns exist
        if 'description' not in courses_df.columns:
            courses_df['description'] = ''
        if 'title' not in courses_df.columns:
            courses_df['title'] = ''
        
        # Combine title and description for better extraction
        courses_df['combined_text'] = (
            courses_df['title'].fillna('') + ' ' + 
            courses_df['description'].fillna('')
        )
        
        all_course_skills = []
        
        for idx, row in courses_df.iterrows():
            course_id = row.get('course_id', f'C{idx:04d}')
            text = row['combined_text']
            
            # Extract skills
            skills = self.extract_skills_from_text(text)
            
            # Create records
            for skill in skills:
                all_course_skills.append({
                    'course_id': course_id,
                    'course_title': row['title'],
                    'skill': skill['skill'],
                    'skill_category': skill.get('category', 'unknown'),
                    'skill_difficulty': skill.get('difficulty', 'intermediate'),
                    'confidence': skill['confidence'],
                    'extraction_method': skill['source']
                })
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(courses_df)} courses...")
        
        skills_df = pd.DataFrame(all_course_skills)
        
        # Statistics
        print(f"\n✓ Extraction complete!")
        print(f"  Total course-skill mappings: {len(skills_df)}")
        print(f"  Unique skills found: {skills_df['skill'].nunique()}")
        print(f"  Avg skills per course: {len(skills_df) / len(courses_df):.1f}")
        
        # Show top skills
        print(f"\n  Top 10 most common skills:")
        top_skills = skills_df['skill'].value_counts().head(10)
        for skill, count in top_skills.items():
            print(f"    - {skill}: {count} courses")
        
        return skills_df
    
    def build_skill_graph(self, skills_df: pd.DataFrame, min_cooccurrence=2) -> nx.Graph:
        """
        Build skill co-occurrence graph
        
        Args:
            skills_df: DataFrame with course-skill mappings
            min_cooccurrence: Minimum times skills must appear together
            
        Returns:
            NetworkX graph of skill relationships
        """
        
        print("\n🕸️  Building skill co-occurrence graph...")
        
        # Group skills by course
        course_skills = skills_df.groupby('course_id')['skill'].apply(list).to_dict()
        
        # Count co-occurrences
        cooccurrence = defaultdict(int)
        for course_id, skills in course_skills.items():
            # For each pair of skills in the course
            for i, skill1 in enumerate(skills):
                for skill2 in skills[i+1:]:
                    pair = tuple(sorted([skill1, skill2]))
                    cooccurrence[pair] += 1
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes (all skills)
        all_skills = skills_df['skill'].unique()
        for skill in all_skills:
            skill_info = skills_df[skills_df['skill'] == skill].iloc[0]
            G.add_node(skill, 
                      category=skill_info['skill_category'],
                      difficulty=skill_info['skill_difficulty'],
                      frequency=len(skills_df[skills_df['skill'] == skill]))
        
        # Add edges (co-occurrences)
        for (skill1, skill2), count in cooccurrence.items():
            if count >= min_cooccurrence:
                G.add_edge(skill1, skill2, weight=count)
        
        print(f"✓ Graph created!")
        print(f"  Nodes (skills): {G.number_of_nodes()}")
        print(f"  Edges (relationships): {G.number_of_edges()}")
        print(f"  Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.1f}")
        
        return G
    
    def cluster_skills(self, skills_df: pd.DataFrame, n_clusters=10) -> pd.DataFrame:
        """
        Cluster similar skills using TF-IDF + hierarchical clustering
        
        Args:
            skills_df: DataFrame with skills
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        
        print(f"\n🎯 Clustering skills into {n_clusters} groups...")
        
        # Get unique skills
        unique_skills = skills_df['skill'].unique()
        
        # Vectorize skill names
        vectorizer = TfidfVectorizer(max_features=100)
        skill_vectors = vectorizer.fit_transform(unique_skills)
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        
        # Fit on dense matrix
        skill_vectors_dense = skill_vectors.toarray()
        cluster_labels = clustering.fit_predict(skill_vectors_dense)
        
        # Create mapping
        skill_clusters = pd.DataFrame({
            'skill': unique_skills,
            'cluster': cluster_labels
        })
        
        # Merge back
        skills_df = skills_df.merge(skill_clusters, on='skill', how='left')
        
        print(f"✓ Clustering complete!")
        
        # Show cluster examples
        for cluster_id in range(min(5, n_clusters)):
            cluster_skills = skill_clusters[skill_clusters['cluster'] == cluster_id]['skill'].tolist()
            print(f"\n  Cluster {cluster_id}: {cluster_skills[:5]}")
        
        return skills_df
    
    def visualize_skill_graph(self, G: nx.Graph, output_path='outputs/skill_graph.png', top_n=50):
        """
        Visualize skill graph
        
        Args:
            G: NetworkX graph
            output_path: Where to save visualization
            top_n: Show only top N most connected skills
        """
        
        print(f"\n📊 Visualizing top {top_n} skills...")
        
        # Get top nodes by degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:top_n]
        
        # Subgraph
        G_sub = G.subgraph(top_nodes)
        
        # Layout
        pos = nx.spring_layout(G_sub, k=0.5, iterations=50, seed=42)
        
        # Draw
        plt.figure(figsize=(20, 16))
        
        # Color by category
        categories = nx.get_node_attributes(G_sub, 'category')
        unique_categories = list(set(categories.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        category_colors = dict(zip(unique_categories, colors))
        
        node_colors = [category_colors.get(categories.get(node, 'unknown'), 'gray') 
                      for node in G_sub.nodes()]
        
        # Node sizes by frequency
        frequencies = nx.get_node_attributes(G_sub, 'frequency')
        node_sizes = [frequencies.get(node, 1) * 50 for node in G_sub.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.7)
        nx.draw_networkx_edges(G_sub, pos, alpha=0.2, width=1)
        nx.draw_networkx_labels(G_sub, pos, font_size=8, font_weight='bold')
        
        plt.title("Skill Co-occurrence Network (Top 50 Skills)", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        Path(output_path).parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
        
        plt.close()
    
    def export_for_neo4j(self, skills_df: pd.DataFrame, G: nx.Graph, output_dir='data/neo4j'):
        """
        Export data in Neo4j-compatible format
        
        Args:
            skills_df: Course-skill mappings
            G: Skill graph
            output_dir: Output directory
        """
        
        print(f"\n💾 Exporting data for Neo4j...")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Skills nodes
        skills_nodes = []
        for skill in G.nodes():
            node_data = G.nodes[skill]
            skills_nodes.append({
                'skill_id': skill.replace(' ', '_'),
                'skill_name': skill,
                'category': node_data.get('category', 'unknown'),
                'difficulty': node_data.get('difficulty', 'intermediate'),
                'frequency': node_data.get('frequency', 0)
            })
        
        skills_nodes_df = pd.DataFrame(skills_nodes)
        skills_nodes_df.to_csv(f'{output_dir}/skills_nodes.csv', index=False)
        print(f"  ✓ Exported {len(skills_nodes_df)} skill nodes")
        
        # 2. Skill relationships (co-occurrence)
        skill_edges = []
        for edge in G.edges(data=True):
            skill_edges.append({
                'skill1_id': edge[0].replace(' ', '_'),
                'skill2_id': edge[1].replace(' ', '_'),
                'weight': edge[2].get('weight', 1),
                'relationship_type': 'CO_OCCURS_WITH'
            })
        
        skill_edges_df = pd.DataFrame(skill_edges)
        skill_edges_df.to_csv(f'{output_dir}/skill_relationships.csv', index=False)
        print(f"  ✓ Exported {len(skill_edges_df)} skill relationships")
        
        # 3. Course-skill mappings
        course_skill_edges = skills_df[['course_id', 'skill', 'confidence']].copy()
        course_skill_edges['skill_id'] = course_skill_edges['skill'].str.replace(' ', '_')
        course_skill_edges = course_skill_edges[['course_id', 'skill_id', 'confidence']]
        course_skill_edges.to_csv(f'{output_dir}/course_teaches_skill.csv', index=False)
        print(f"  ✓ Exported {len(course_skill_edges)} course-skill mappings")
        
        print(f"\n✓ All data exported to {output_dir}/")
        print(f"  Ready for Neo4j import!")


def main():
    """
    Main execution pipeline
    """
    
    print("="*70)
    print("  SKILL EXTRACTION PIPELINE - CurricuRL Phase 0")
    print("="*70)
    
    # Initialize extractor
    extractor = SkillExtractor()
    
    # Load courses
    print("\n📂 Loading course data...")
    courses_df = pd.read_csv('coursera_courses.csv')
    print(f"✓ Loaded {len(courses_df)} courses")
    
    # Extract skills
    skills_df = extractor.extract_skills_from_courses(courses_df)
    
    # Save raw extractions
    skills_df.to_csv('data/course_skills_raw.csv', index=False)
    print(f"\n💾 Saved raw skills to data/course_skills_raw.csv")
    
    # Cluster skills
    skills_df = extractor.cluster_skills(skills_df, n_clusters=15)
    skills_df.to_csv('data/course_skills_clustered.csv', index=False)
    print(f"💾 Saved clustered skills to data/course_skills_clustered.csv")
    
    # Build skill graph
    skill_graph = extractor.build_skill_graph(skills_df, min_cooccurrence=2)
    
    # Save graph
    import pickle
    with open('data/skill_graph.gpickle', 'wb') as f:
        pickle.dump(skill_graph, f)
    print(f"💾 Saved graph to data/skill_graph.gpickle")
    
    # Visualize
    extractor.visualize_skill_graph(skill_graph, output_path='outputs/skill_graph.png', top_n=50)
    
    # Export for Neo4j
    extractor.export_for_neo4j(skills_df, skill_graph)
    
    print("\n" + "="*70)
    print("  ✓ SKILL EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - data/course_skills_raw.csv")
    print(f"  - data/course_skills_clustered.csv")
    print(f"  - data/skill_graph.gpickle")
    print(f"  - outputs/skill_graph.png")
    print(f"  - data/neo4j/skills_nodes.csv")
    print(f"  - data/neo4j/skill_relationships.csv")
    print(f"  - data/neo4j/course_teaches_skill.csv")
    

if __name__ == "__main__":
    main()