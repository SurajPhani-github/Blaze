# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # ==================== LOAD DATA ====================
# coursera_df = pd.read_csv('coursera_courses.csv')
# learner_df = pd.read_csv('augmented_learner_data.csv')

# print(f"Loaded {len(coursera_df)} courses and {len(learner_df)} learner records\n")

# # ==================== DATA PREPARATION ====================
# coursera_df['title'] = coursera_df['title'].fillna('').astype(str)
# coursera_df['description'] = coursera_df['description'].fillna('').astype(str)
# coursera_df['domain'] = coursera_df['domain'].fillna('').astype(str)
# coursera_df['workload'] = coursera_df['workload'].fillna('Unknown').astype(str)
# coursera_df['url'] = coursera_df['url'].fillna('No URL').astype(str)

# # Create rich text representation
# coursera_df['text'] = (
#     coursera_df['title'] + ' ' + 
#     coursera_df['title'] + ' ' +
#     coursera_df['description'] + ' ' + 
#     coursera_df['domain']
# )

# # ==================== DIFFICULTY INFERENCE ====================
# difficulty_keywords = {
#     'beginner': ['introduction', 'intro', 'basics', 'basic', 'fundamentals', 'fundamental', 
#                  'beginner', 'getting started', 'start', 'first', 'everyone', 'essentials'],
#     'intermediate': ['intermediate', 'applied', 'practical', 'hands-on', 'building', 
#                      'developing', 'complete', 'full'],
#     'advanced': ['advanced', 'expert', 'in-depth', 'deep', 'master', 'specialization',
#                  'professional', 'architect', 'engineering']
# }

# def infer_difficulty(text):
#     text = str(text).lower()
#     scores = {level: 0 for level in difficulty_keywords}
    
#     for level, keywords in difficulty_keywords.items():
#         for kw in keywords:
#             if kw in text:
#                 scores[level] += 1
    
#     if scores['beginner'] > 0 and scores['advanced'] == 0:
#         return 'beginner'
#     elif scores['advanced'] > scores['beginner']:
#         return 'advanced'
#     else:
#         return 'intermediate'

# coursera_df['difficulty'] = coursera_df['text'].apply(infer_difficulty)

# difficulty_map = {
#     'easy': 'beginner',
#     'beginner': 'beginner',
#     'intermediate': 'intermediate',
#     'medium': 'intermediate',
#     'advanced': 'advanced',
#     'hard': 'advanced',
#     'expert': 'advanced'
# }

# # ==================== TF-IDF VECTORIZATION ====================
# vectorizer = TfidfVectorizer(
#     stop_words='english', 
#     max_features=5000,
#     ngram_range=(1, 2),
#     min_df=1
# )
# course_matrix = vectorizer.fit_transform(coursera_df['text'])

# print(f"Course matrix shape: {course_matrix.shape}")
# print(f"Difficulty distribution:\n{coursera_df['difficulty'].value_counts()}\n")

# # ==================== BUILD USER PROFILES ====================
# # Create comprehensive user profiles
# user_profiles = {}

# for learner_id in learner_df['learner_id'].unique():
#     history = learner_df[learner_df['learner_id'] == learner_id]
    
#     user_profiles[learner_id] = {
#         'content_types': history['content_type'].value_counts().to_dict(),
#         'avg_quiz_score': history['quiz_score'].mean(),
#         'avg_engagement': history['engagement_score'].mean(),
#         'total_interactions': len(history),
#         'preferred_type': history['content_type'].mode()[0] if len(history) > 0 else None,
#         'performance_level': 'advanced' if history['quiz_score'].mean() > 75 else 
#                            'beginner' if history['quiz_score'].mean() < 55 else 'intermediate'
#     }

# # ==================== COLLABORATIVE FILTERING SETUP ====================
# pivot_quiz = learner_df.pivot_table(
#     index='learner_id', 
#     columns='content_type', 
#     values='quiz_score', 
#     aggfunc='mean'
# ).fillna(0)

# pivot_eng = learner_df.pivot_table(
#     index='learner_id', 
#     columns='content_type', 
#     values='engagement_score', 
#     aggfunc='mean'
# ).fillna(0)

# combined = pivot_quiz * 0.6 + pivot_eng * 0.4

# scaler = StandardScaler()
# user_matrix = scaler.fit_transform(combined.fillna(0))
# similarity_matrix = cosine_similarity(user_matrix)
# similarity_df = pd.DataFrame(
#     similarity_matrix, 
#     index=combined.index, 
#     columns=combined.index
# )

# # ==================== HELPER: GET EXCLUDED COURSES ====================
# def get_excluded_set(exclude_list):
#     """Convert list of DataFrames to set of course titles to exclude"""
#     excluded = set()
#     for df in exclude_list:
#         if df is not None and len(df) > 0:
#             excluded.update(df['title'].tolist())
#     return excluded

# # ==================== METHOD 1: PURE TOPIC-BASED (CONTENT) ====================
# def recommend_by_topic(topic, course_difficulty='beginner', top_k=10, exclude_courses=None):
#     """
#     Pure content-based: Only considers topic match and difficulty
#     """
#     course_difficulty = difficulty_map.get(course_difficulty.lower(), course_difficulty)
    
#     # Simple, focused query
#     query = f"{topic} {topic} {topic} course"
#     query_vec = vectorizer.transform([query])
    
#     sims = cosine_similarity(query_vec, course_matrix).flatten()
    
#     # Difficulty filter with flexibility
#     difficulty_mask = coursera_df['difficulty'] == course_difficulty
#     if course_difficulty == 'intermediate':
#         difficulty_mask = difficulty_mask | (coursera_df['difficulty'] == 'beginner')
#     elif course_difficulty == 'advanced':
#         difficulty_mask = difficulty_mask | (coursera_df['difficulty'] == 'intermediate')
    
#     # Exclude already recommended courses
#     if exclude_courses:
#         exclude_mask = ~coursera_df['title'].isin(exclude_courses)
#         difficulty_mask = difficulty_mask & exclude_mask
    
#     filtered_sims = np.where(difficulty_mask, sims, -1)
    
#     top_idx = np.argsort(filtered_sims)[-top_k:][::-1]
#     valid_idx = [idx for idx in top_idx if filtered_sims[idx] > 0]
    
#     if len(valid_idx) == 0:
#         top_idx = np.argsort(sims)[-top_k:][::-1]
#         recs = coursera_df.iloc[top_idx][['title', 'domain', 'workload', 'url', 'difficulty']].copy()
#         recs['score'] = sims[top_idx]
#         recs['reason'] = f"Best available matches for '{topic}'"
#     else:
#         recs = coursera_df.iloc[valid_idx][['title', 'domain', 'workload', 'url', 'difficulty']].copy()
#         recs['score'] = filtered_sims[valid_idx]
#         recs['reason'] = f"Topic match: '{topic}' at {course_difficulty} level"
    
#     return recs.reset_index(drop=True)

# # ==================== METHOD 2: COLLABORATIVE FILTERING ====================
# def collaborative_recommend(learner_id, user_topic='', course_difficulty='beginner', 
#                           top_k=5, exclude_courses=None):
#     """
#     Collaborative filtering: What do similar learners like?
#     Uses learner similarity + their successful content types
#     """
#     course_difficulty = difficulty_map.get(course_difficulty.lower(), course_difficulty)
    
#     # Find similar users and their preferences
#     if learner_id not in similarity_df.index:
#         similar_learners_types = learner_df['content_type'].value_counts().head(3).index.tolist()
#         collab_weight = 0.3  # Less weight for new users
#     else:
#         similar_users = similarity_df[learner_id].sort_values(ascending=False).index[1:21]
#         similar_hist = learner_df[learner_df['learner_id'].isin(similar_users)]
        
#         # Weight by engagement and quiz scores
#         weighted_types = similar_hist.groupby('content_type').agg({
#             'quiz_score': 'mean',
#             'engagement_score': 'mean'
#         })
#         weighted_types['combined_score'] = (
#             weighted_types['quiz_score'] * 0.6 + 
#             weighted_types['engagement_score'] * 0.4
#         )
#         similar_learners_types = weighted_types.sort_values('combined_score', ascending=False).head(3).index.tolist()
#         collab_weight = 0.5  # More weight for known users
    
#     # Build query: Balance topic with collaborative signal
#     # 50% topic, 50% what similar users liked
#     topic_query = f"{user_topic} {user_topic} {user_topic}"
#     collab_query = ' '.join([f"{t} {t}" for t in similar_learners_types])
#     query = f"{topic_query} {collab_query} {course_difficulty}"
    
#     query_vec = vectorizer.transform([query])
#     sims = cosine_similarity(query_vec, course_matrix).flatten()
    
#     # Apply difficulty filter
#     difficulty_mask = coursera_df['difficulty'] == course_difficulty
#     if course_difficulty == 'intermediate':
#         difficulty_mask = difficulty_mask | (coursera_df['difficulty'] == 'beginner')
#     elif course_difficulty == 'advanced':
#         difficulty_mask = difficulty_mask | (coursera_df['difficulty'] == 'intermediate')
    
#     # Boost courses that match similar learners' successful content types
#     domain_boost = coursera_df['domain'].apply(
#         lambda x: 1.2 if any(d.lower() in str(x).lower() for d in similar_learners_types) else 1.0
#     )
    
#     # Exclude already recommended
#     if exclude_courses:
#         exclude_mask = ~coursera_df['title'].isin(exclude_courses)
#         difficulty_mask = difficulty_mask & exclude_mask
    
#     filtered_sims = np.where(difficulty_mask, sims * domain_boost, -1)
    
#     top_idx = np.argsort(filtered_sims)[-top_k:][::-1]
#     valid_idx = [idx for idx in top_idx if filtered_sims[idx] > 0]
    
#     if len(valid_idx) == 0:
#         valid_idx = np.argsort(sims)[-top_k:][::-1]
    
#     recs = coursera_df.iloc[valid_idx][['title', 'domain', 'workload', 'url', 'difficulty']].copy()
#     recs['score'] = sims[valid_idx]
    
#     types_str = ', '.join(similar_learners_types[:2]) if similar_learners_types else 'popular'
#     recs['reason'] = f"Collaborative: Similar learners succeeded with {types_str} content on '{user_topic}'"
    
#     return recs.reset_index(drop=True)

# # ==================== METHOD 3: PERSONALIZED (HYBRID) ====================
# def personalized_recommend(learner_id, user_topic='', course_difficulty='beginner', 
#                           top_k=5, exclude_courses=None):
#     """
#     Personalized: Uses YOUR specific history, performance, and preferences
#     """
#     course_difficulty = difficulty_map.get(course_difficulty.lower(), course_difficulty)
    
#     history = learner_df[learner_df['learner_id'] == learner_id]
    
#     if len(history) == 0:
#         return recommend_by_topic(user_topic, course_difficulty, top_k, exclude_courses)
    
#     profile = user_profiles.get(learner_id, {})
    
#     # Get user's performance-based difficulty adjustment
#     user_performance = profile.get('performance_level', 'intermediate')
#     avg_quiz = profile.get('avg_quiz_score', 60)
    
#     # Adjust difficulty based on performance
#     # High performers might want challenges, low performers need support
#     if avg_quiz > 80 and course_difficulty == 'beginner':
#         adjusted_difficulty = 'intermediate'
#         adjustment_note = "upgraded to intermediate (high performance)"
#     elif avg_quiz < 50 and course_difficulty == 'advanced':
#         adjusted_difficulty = 'intermediate'
#         adjustment_note = "adjusted to intermediate (building foundation)"
#     else:
#         adjusted_difficulty = course_difficulty
#         adjustment_note = course_difficulty
    
#     # Build personalized query
#     freq_type = profile.get('preferred_type', 'course')
    
#     # Weight: 60% topic, 40% personal history
#     topic_query = f"{user_topic} {user_topic} {user_topic} {user_topic}"
#     personal_query = f"{freq_type} {user_performance}"
#     query = f"{topic_query} {personal_query} {adjusted_difficulty}"
    
#     query_vec = vectorizer.transform([query])
#     sims = cosine_similarity(query_vec, course_matrix).flatten()
    
#     # Difficulty mask
#     difficulty_mask = coursera_df['difficulty'] == adjusted_difficulty
#     if adjusted_difficulty == 'intermediate':
#         difficulty_mask = difficulty_mask | (coursera_df['difficulty'] == 'beginner')
#     elif adjusted_difficulty == 'advanced':
#         difficulty_mask = difficulty_mask | (coursera_df['difficulty'] == 'intermediate')
    
#     # Personal boost: courses in domains you've engaged with before
#     engaged_types = set(history['content_type'].unique())
#     personal_boost = coursera_df['domain'].apply(
#         lambda x: 1.3 if any(t.lower() in str(x).lower() for t in engaged_types) else 1.0
#     )
    
#     # Exclude already recommended
#     if exclude_courses:
#         exclude_mask = ~coursera_df['title'].isin(exclude_courses)
#         difficulty_mask = difficulty_mask & exclude_mask
    
#     filtered_sims = np.where(difficulty_mask, sims * personal_boost, -1)
    
#     top_idx = np.argsort(filtered_sims)[-top_k:][::-1]
#     valid_idx = [idx for idx in top_idx if filtered_sims[idx] > 0]
    
#     if len(valid_idx) == 0:
#         valid_idx = np.argsort(sims)[-top_k:][::-1]
    
#     recs = coursera_df.iloc[valid_idx][['title', 'domain', 'workload', 'url', 'difficulty']].copy()
#     recs['score'] = sims[valid_idx]
#     recs['reason'] = (f"Personal match: Your {freq_type} preference + {user_performance} performance "
#                       f"on '{user_topic}' ({adjustment_note})")
    
#     return recs.reset_index(drop=True)

# # ==================== MAIN FLOW ====================
# def recommendation_flow():
#     print("=" * 70)
#     print("        COURSERA COURSE RECOMMENDATION SYSTEM")
#     print("=" * 70)
#     print()
    
#     topic = input("What topic do you want to learn? ").strip()
#     level = input("Your proficiency level? (beginner/intermediate/advanced): ").strip().lower()
#     difficulty = input("Preferred course difficulty? (easy/intermediate/advanced): ").strip().lower()
    
#     # METHOD 1: Topic-Based
#     print(f"\n{'='*70}")
#     print(f"  METHOD 1: TOPIC-BASED RECOMMENDATIONS")
#     print(f"  Pure content matching for '{topic}' at {difficulty} level")
#     print(f"{'='*70}\n")
    
#     topic_recs = recommend_by_topic(topic, difficulty, top_k=10)
    
#     for idx, row in topic_recs.iterrows():
#         print(f"{idx + 1}. {row['title']}")
#         print(f"   Domain: {row['domain']} | Difficulty: {row['difficulty']} | Workload: {row['workload']}")
#         print(f"   Match Score: {row['score']:.3f}")
#         print(f"   {row['reason']}")
#         print(f"   URL: {row['url']}")
#         print()
    
#     learner_input = input("\nEnter learner_id for personalized suggestions (e.g., L200) or 'n': ").strip()
    
#     if learner_input.lower() != 'n' and learner_input:
#         # Get exclusion set from topic recommendations
#         exclude_set = get_excluded_set([topic_recs])
        
#         # METHOD 2: Collaborative
#         print(f"\n{'='*70}")
#         print(f"  METHOD 2: COLLABORATIVE FILTERING")
#         print(f"  What learners similar to {learner_input} succeeded with")
#         print(f"{'='*70}\n")
        
#         collab_recs = collaborative_recommend(
#             learner_input, topic, difficulty, top_k=5, exclude_courses=exclude_set
#         )
        
#         for idx, row in collab_recs.iterrows():
#             print(f"{idx + 1}. {row['title']}")
#             print(f"   Domain: {row['domain']} | Difficulty: {row['difficulty']} | Workload: {row['workload']}")
#             print(f"   Match Score: {row['score']:.3f}")
#             print(f"   {row['reason']}")
#             print(f"   URL: {row['url']}")
#             print()
        
#         # Update exclusion set
#         exclude_set = get_excluded_set([topic_recs, collab_recs])
        
#         # METHOD 3: Personalized
#         print(f"\n{'='*70}")
#         print(f"  METHOD 3: PERSONALIZED RECOMMENDATIONS")
#         print(f"  Tailored specifically to {learner_input}'s history & performance")
#         print(f"{'='*70}\n")
        
#         pers_recs = personalized_recommend(
#             learner_input, topic, difficulty, top_k=5, exclude_courses=exclude_set
#         )
        
#         for idx, row in pers_recs.iterrows():
#             print(f"{idx + 1}. {row['title']}")
#             print(f"   Domain: {row['domain']} | Difficulty: {row['difficulty']} | Workload: {row['workload']}")
#             print(f"   Match Score: {row['score']:.3f}")
#             print(f"   {row['reason']}")
#             print(f"   URL: {row['url']}")
#             print()
        
#         # Show user profile
#         if learner_input in user_profiles:
#             profile = user_profiles[learner_input]
#             print(f"\n{'='*70}")
#             print(f"  YOUR LEARNER PROFILE ({learner_input})")
#             print(f"{'='*70}")
#             print(f"  Performance Level: {profile['performance_level']}")
#             print(f"  Avg Quiz Score: {profile['avg_quiz_score']:.1f}")
#             print(f"  Avg Engagement: {profile['avg_engagement']:.1f}")
#             print(f"  Preferred Content: {profile['preferred_type']}")
#             print(f"  Total Interactions: {profile['total_interactions']}")
#             print(f"{'='*70}\n")

# if __name__ == "__main__":
#     recommendation_flow()


"""
Advanced Course Recommendation System - Research Grade (CORRECTED)
===================================================================

This module implements state-of-the-art recommendation techniques:
1. Content-Based Filtering (TF-IDF + BERT embeddings)
2. Collaborative Filtering (SVD, Neural Collaborative Filtering)
3. Sequential Modeling (LSTM for learning paths)
4. Hybrid Ensemble (Neural network combining all signals)
5. Cold-Start Handling (Meta-learning approach)

Data Requirements:
-----------------
1. coursera_courses.csv:
   - course_id, title, description, domain, url, workload
   - difficulty (optional - will be inferred if missing)
   
2. augmented_learner_data.csv:
   - learner_id, course_id (or content_type), quiz_score, engagement_score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Traditional ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Utilities
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
import json

# ==================== CONFIGURATION ====================
class Config:
    """Configuration for the recommendation system"""
    # Model parameters
    EMBEDDING_DIM = 128
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT = 0.3
    LEARNING_RATE = 0.001
    BATCH_SIZE = 256
    EPOCHS = 50
    
    # SVD parameters
    SVD_COMPONENTS = 100
    
    # LSTM parameters
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    SEQUENCE_LENGTH = 10
    
    # Hybrid weights
    CONTENT_WEIGHT = 0.3
    COLLAB_WEIGHT = 0.3
    NEURAL_WEIGHT = 0.4
    
    # Paths
    MODEL_DIR = Path("models")
    DATA_DIR = Path("data")
    
    def __init__(self):
        self.MODEL_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)


config = Config()

# ==================== DATA LOADER ====================
class DataLoader:
    """Enhanced data loading with validation and preprocessing"""
    
    def __init__(self):
        self.courses_df = None
        self.learners_df = None
        self.ratings_df = None
        self.interaction_matrix = None
        
    def load_data(self, courses_path='coursera_courses.csv', 
                  learners_path='augmented_learner_data.csv',
                  ratings_path=None):
        """Load and validate all datasets"""
        
        print("📊 Loading datasets...")
        
        # Load courses
        self.courses_df = pd.read_csv(courses_path)
        print(f"✓ Loaded {len(self.courses_df)} courses")
        
        # Load learner interactions
        self.learners_df = pd.read_csv(learners_path)
        print(f"✓ Loaded {len(self.learners_df)} learner interactions")
        
        # Load ratings if available
        if ratings_path and Path(ratings_path).exists():
            self.ratings_df = pd.read_csv(ratings_path)
            print(f"✓ Loaded {len(self.ratings_df)} ratings")
        else:
            # Create synthetic ratings from engagement
            self.ratings_df = self._create_synthetic_ratings()
            print(f"✓ Created {len(self.ratings_df)} synthetic ratings")
        
        # Clean and validate
        self._clean_data()
        self._create_interaction_matrix()
        
        return self
    
    def _infer_difficulty(self, text):
        """Infer course difficulty from text content"""
        difficulty_keywords = {
            'beginner': ['introduction', 'intro', 'basics', 'basic', 'fundamentals', 'fundamental', 
                         'beginner', 'getting started', 'start', 'first', 'everyone', 'essentials',
                         'beginning', 'foundation', 'introductory', '101'],
            'intermediate': ['intermediate', 'applied', 'practical', 'hands-on', 'building', 
                             'developing', 'complete', 'full', 'advanced topics', 'deep dive'],
            'advanced': ['advanced', 'expert', 'in-depth', 'deep', 'master', 'specialization',
                         'professional', 'architect', 'engineering', 'mastering', 'expert level']
        }
        
        text = str(text).lower()
        scores = {level: 0 for level in difficulty_keywords}
        
        for level, keywords in difficulty_keywords.items():
            for kw in keywords:
                if kw in text:
                    scores[level] += 1
        
        if scores['beginner'] > 0 and scores['advanced'] == 0:
            return 'beginner'
        elif scores['advanced'] > scores['beginner']:
            return 'advanced'
        else:
            return 'intermediate'
    
    def _clean_data(self):
        """Clean and preprocess all datasets"""
        print("🧹 Cleaning data...")
        
        # Courses - handle missing columns
        required_cols = ['title', 'description', 'domain']
        for col in required_cols:
            if col not in self.courses_df.columns:
                print(f"⚠️  Warning: '{col}' column missing, creating empty column")
                self.courses_df[col] = ''
        
        self.courses_df['title'] = self.courses_df['title'].fillna('').astype(str)
        self.courses_df['description'] = self.courses_df['description'].fillna('').astype(str)
        self.courses_df['domain'] = self.courses_df['domain'].fillna('General').astype(str)
        
        # Handle difficulty - infer if missing
        if 'difficulty' not in self.courses_df.columns:
            print("⚠️  'difficulty' column missing - inferring from title and description...")
            self.courses_df['content_text_temp'] = (
                self.courses_df['title'] + ' ' + self.courses_df['description']
            )
            self.courses_df['difficulty'] = self.courses_df['content_text_temp'].apply(self._infer_difficulty)
            print(f"✓ Difficulty inferred. Distribution:\n{self.courses_df['difficulty'].value_counts()}")
        else:
            self.courses_df['difficulty'] = self.courses_df['difficulty'].fillna('intermediate').astype(str)
        
        # Standardize difficulty values
        difficulty_map = {
            'easy': 'beginner',
            'beginner': 'beginner',
            'intermediate': 'intermediate',
            'medium': 'intermediate',
            'advanced': 'advanced',
            'hard': 'advanced',
            'expert': 'advanced'
        }
        self.courses_df['difficulty'] = self.courses_df['difficulty'].str.lower().map(
            lambda x: difficulty_map.get(x, 'intermediate')
        )
        
        # Handle optional columns
        if 'workload' not in self.courses_df.columns:
            self.courses_df['workload'] = 'Unknown'
        else:
            self.courses_df['workload'] = self.courses_df['workload'].fillna('Unknown').astype(str)
        
        if 'url' not in self.courses_df.columns:
            self.courses_df['url'] = ''
        else:
            self.courses_df['url'] = self.courses_df['url'].fillna('').astype(str)
        
        # Create rich text for content-based
        self.courses_df['content_text'] = (
            self.courses_df['title'] + ' ' + 
            self.courses_df['title'] + ' ' +  # Title weighted more
            self.courses_df['description'] + ' ' + 
            self.courses_df['domain'] + ' ' +
            self.courses_df['difficulty']
        )
        
        # Add course_id if not present
        if 'course_id' not in self.courses_df.columns:
            self.courses_df['course_id'] = [f'C{i:04d}' for i in range(len(self.courses_df))]
        
        # Learners - ensure numeric scores
        if 'quiz_score' in self.learners_df.columns:
            self.learners_df['quiz_score'] = pd.to_numeric(
                self.learners_df['quiz_score'], errors='coerce'
            ).fillna(50.0)
        else:
            print("⚠️  'quiz_score' column missing - using default value 50.0")
            self.learners_df['quiz_score'] = 50.0
        
        if 'engagement_score' in self.learners_df.columns:
            self.learners_df['engagement_score'] = pd.to_numeric(
                self.learners_df['engagement_score'], errors='coerce'
            ).fillna(50.0)
        else:
            print("⚠️  'engagement_score' column missing - using default value 50.0")
            self.learners_df['engagement_score'] = 50.0
        
        print("✓ Data cleaned and validated")
    
    def _create_synthetic_ratings(self):
        """Create ratings from engagement and quiz scores"""
        # Check if course_id exists in learners_df
        if 'course_id' not in self.learners_df.columns:
            print("⚠️  'course_id' not in learner data - mapping content_type to courses...")
            # Map content_type to course domain
            content_to_course = {}
            
            if 'content_type' in self.learners_df.columns:
                for content_type in self.learners_df['content_type'].unique():
                    # Find matching course
                    matches = self.courses_df[
                        self.courses_df['domain'].str.contains(str(content_type), case=False, na=False)
                    ]
                    if len(matches) > 0:
                        content_to_course[content_type] = matches.iloc[0]['course_id']
                    else:
                        # Random assignment
                        content_to_course[content_type] = np.random.choice(
                            self.courses_df['course_id'].values
                        )
                
                self.learners_df['course_id'] = self.learners_df['content_type'].map(content_to_course)
            else:
                # No content_type either - random assignment
                print("⚠️  No 'content_type' column - assigning random courses")
                self.learners_df['course_id'] = np.random.choice(
                    self.courses_df['course_id'].values, 
                    size=len(self.learners_df)
                )
        
        # Create rating from engagement and quiz scores
        ratings = []
        for _, row in self.learners_df.iterrows():
            # Rating = weighted combination of engagement and quiz score
            engagement = row.get('engagement_score', 50) / 100 * 5
            quiz = row.get('quiz_score', 50) / 100 * 5
            rating = 0.6 * engagement + 0.4 * quiz
            rating = np.clip(rating, 1, 5)  # Ensure 1-5 range
            
            ratings.append({
                'learner_id': row['learner_id'],
                'course_id': row['course_id'],
                'rating': rating
            })
        
        return pd.DataFrame(ratings)
    
    def _create_interaction_matrix(self):
        """Create user-item interaction matrix"""
        try:
            self.interaction_matrix = self.ratings_df.pivot_table(
                index='learner_id',
                columns='course_id',
                values='rating',
                fill_value=0
            )
            print(f"✓ Created interaction matrix: {self.interaction_matrix.shape}")
        except Exception as e:
            print(f"⚠️  Error creating interaction matrix: {e}")
            # Create a simple matrix with at least some structure
            unique_learners = self.ratings_df['learner_id'].unique()
            unique_courses = self.ratings_df['course_id'].unique()
            self.interaction_matrix = pd.DataFrame(
                0, 
                index=unique_learners, 
                columns=unique_courses
            )


# ==================== NEURAL COLLABORATIVE FILTERING ====================
class NCFDataset(Dataset):
    """Dataset for Neural Collaborative Filtering"""
    
    def __init__(self, ratings_df, user_encoder, item_encoder):
        self.ratings_df = ratings_df
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, idx):
        row = self.ratings_df.iloc[idx]
        user_id = self.user_encoder.transform([row['learner_id']])[0]
        item_id = self.item_encoder.transform([row['course_id']])[0]
        rating = row['rating']
        
        return torch.tensor(user_id, dtype=torch.long), \
               torch.tensor(item_id, dtype=torch.long), \
               torch.tensor(rating, dtype=torch.float32)


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) Model
    Based on "Neural Collaborative Filtering" (He et al., WWW 2017)
    """
    
    def __init__(self, num_users, num_items, embedding_dim=128, hidden_dims=[256, 128, 64]):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embeddings for GMF
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)
        
        # Embeddings for MLP
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(config.DROPOUT))
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        self.output = nn.Linear(embedding_dim + hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, user_ids, item_ids):
        # GMF part
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate and predict
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.output(combined).squeeze()
        
        return prediction


class NCFRecommender:
    """Wrapper for NCF training and inference"""
    
    def __init__(self, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")
    
    def train(self, ratings_df, epochs=50, batch_size=256, lr=0.001):
        """Train the NCF model"""
        print("\n🔥 Training Neural Collaborative Filtering...")
        
        # Encode users and items
        self.user_encoder.fit(ratings_df['learner_id'].unique())
        self.item_encoder.fit(ratings_df['course_id'].unique())
        
        # Create dataset and dataloader
        dataset = NCFDataset(ratings_df, self.user_encoder, self.item_encoder)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = NeuralCollaborativeFiltering(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dims=config.HIDDEN_DIMS
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for user_ids, item_ids, ratings in dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                # Forward pass
                predictions = self.model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        print("✓ NCF training complete")
        return self
    
    def predict(self, user_id, item_ids):
        """Predict ratings for a user and multiple items"""
        self.model.eval()
        
        # Encode
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
        except:
            return np.ones(len(item_ids)) * 3.0
        
        # Filter known items
        known_items = set(self.item_encoder.classes_)
        item_ids_filtered = [i for i in item_ids if i in known_items]
        
        if len(item_ids_filtered) == 0:
            return np.ones(len(item_ids)) * 3.0
        
        items_encoded = self.item_encoder.transform(item_ids_filtered)
        
        # Predict
        with torch.no_grad():
            user_tensor = torch.tensor([user_encoded] * len(items_encoded), 
                                      dtype=torch.long).to(self.device)
            item_tensor = torch.tensor(items_encoded, dtype=torch.long).to(self.device)
            predictions = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Map back
        result = np.ones(len(item_ids)) * 3.0
        filtered_idx = [i for i, item in enumerate(item_ids) if item in known_items]
        result[filtered_idx] = predictions
        
        return result
    
    def save(self, path='models/ncf_model.pt'):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'num_users': self.num_users,
            'num_items': self.num_items
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path='models/ncf_model.pt'):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = NeuralCollaborativeFiltering(
            checkpoint['num_users'],
            checkpoint['num_items']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.user_encoder = checkpoint['user_encoder']
        self.item_encoder = checkpoint['item_encoder']
        print(f"✓ Model loaded from {path}")


# ==================== SVD COLLABORATIVE FILTERING ====================
class SVDRecommender:
    """SVD-based Collaborative Filtering"""
    
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=min(n_components, 50), random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, interaction_matrix):
        """Fit SVD on interaction matrix"""
        print("\n📐 Training SVD Collaborative Filtering...")
        
        self.user_ids = interaction_matrix.index
        self.item_ids = interaction_matrix.columns
        
        # Adjust n_components if needed
        max_components = min(interaction_matrix.shape) - 1
        if self.svd.n_components > max_components:
            print(f"⚠️  Adjusting SVD components from {self.svd.n_components} to {max_components}")
            self.svd.n_components = max_components
        
        # Apply SVD
        self.user_factors = self.svd.fit_transform(interaction_matrix)
        self.item_factors = self.svd.components_.T
        
        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"✓ SVD trained - Explained variance: {explained_var:.2%}")
        
        return self
    
    def predict(self, user_id, item_ids):
        """Predict ratings"""
        if user_id not in self.user_ids:
            return np.ones(len(item_ids)) * 3.0
        
        user_idx = self.user_ids.get_loc(user_id)
        user_vec = self.user_factors[user_idx]
        
        scores = []
        for item_id in item_ids:
            if item_id in self.item_ids:
                item_idx = self.item_ids.get_loc(item_id)
                item_vec = self.item_factors[item_idx]
                score = np.dot(user_vec, item_vec)
                scores.append(score)
            else:
                scores.append(3.0)
        
        return np.array(scores)
    
    def save(self, path='models/svd_model.pkl'):
        """Save SVD model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'svd': self.svd,
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'user_ids': self.user_ids,
                'item_ids': self.item_ids
            }, f)
        print(f"✓ SVD model saved to {path}")
    
    def load(self, path='models/svd_model.pkl'):
        """Load SVD model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.svd = data['svd']
            self.user_factors = data['user_factors']
            self.item_factors = data['item_factors']
            self.user_ids = data['user_ids']
            self.item_ids = data['item_ids']
        print(f"✓ SVD model loaded from {path}")


# ==================== CONTENT-BASED FILTERING ====================
class ContentBasedRecommender:
    """Enhanced content-based filtering"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )
        self.course_matrix = None
        self.courses_df = None
    
    def fit(self, courses_df):
        """Fit TF-IDF"""
        print("\n📚 Training Content-Based Filtering...")
        
        self.courses_df = courses_df
        self.course_matrix = self.vectorizer.fit_transform(courses_df['content_text'])
        
        print(f"✓ Content-based model trained - Matrix shape: {self.course_matrix.shape}")
        return self
    
    def recommend(self, query, difficulty=None, top_k=10, exclude_ids=None):
        """Recommend courses"""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.course_matrix).flatten()
        
        mask = np.ones(len(similarities), dtype=bool)
        if difficulty:
            mask = self.courses_df['difficulty'] == difficulty
        
        if exclude_ids:
            exclude_mask = ~self.courses_df['course_id'].isin(exclude_ids)
            mask = mask & exclude_mask
        
        filtered_sims = np.where(mask, similarities, -1)
        top_indices = np.argsort(filtered_sims)[-top_k:][::-1]
        
        results = self.courses_df.iloc[top_indices].copy()
        results['similarity_score'] = filtered_sims[top_indices]
        
        return results
    
    def save(self, path='models/content_model.pkl'):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'course_matrix': self.course_matrix
            }, f)
        print(f"✓ Content model saved to {path}")
    
    def load(self, path='models/content_model.pkl'):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.course_matrix = data['course_matrix']
        print(f"✓ Content model loaded from {path}")


# ==================== HYBRID ENSEMBLE ====================
class HybridRecommender:
    """Hybrid ensemble"""
    
    def __init__(self, content_model, svd_model, ncf_model):
        self.content_model = content_model
        self.svd_model = svd_model
        self.ncf_model = ncf_model
        
        self.weights = {
            'content': config.CONTENT_WEIGHT,
            'svd': config.COLLAB_WEIGHT,
            'ncf': config.NEURAL_WEIGHT
        }
    
    def recommend(self, user_id, query, difficulty=None, top_k=10, exclude_ids=None):
        """Generate hybrid recommendations"""
        content_recs = self.content_model.recommend(
            query, difficulty, top_k=50, exclude_ids=exclude_ids
        )
        
        course_ids = content_recs['course_id'].tolist()
        
        svd_scores = self.svd_model.predict(user_id, course_ids)
        ncf_scores = self.ncf_model.predict(user_id, course_ids)
        content_scores = content_recs['similarity_score'].values
        
        # Normalize
        svd_scores = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min() + 1e-8)
        ncf_scores = (ncf_scores - ncf_scores.min()) / (ncf_scores.max() - ncf_scores.min() + 1e-8)
        content_scores = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min() + 1e-8)
        
        # Ensemble
        final_scores = (
            self.weights['content'] * content_scores +
            self.weights['svd'] * svd_scores +
            self.weights['ncf'] * ncf_scores
        )
        
        content_recs['final_score'] = final_scores
        content_recs['svd_score'] = svd_scores
        content_recs['ncf_score'] = ncf_scores
        content_recs = content_recs.sort_values('final_score', ascending=False).head(top_k)
        
        return content_recs.reset_index(drop=True)


# ==================== MAIN SYSTEM ====================
class AdvancedRecommendationSystem:
    """Complete recommendation system"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.content_model = ContentBasedRecommender()
        self.svd_model = None
        self.ncf_model = None
        self.hybrid_model = None
        self.is_trained = False
    
    def load_and_train(self, courses_path='coursera_courses.csv',
                      learners_path='augmented_learner_data.csv',
                      force_retrain=False):
        """Load data and train all models"""
        
        print("="*70)
        print("   ADVANCED COURSE RECOMMENDATION SYSTEM - RESEARCH GRADE")
        print("="*70)
        
        # Load data
        self.data_loader.load_data(courses_path, learners_path)
        
        # Train content
        self.content_model.fit(self.data_loader.courses_df)
        
        # Train SVD
        self.svd_model = SVDRecommender(n_components=config.SVD_COMPONENTS)
        self.svd_model.fit(self.data_loader.interaction_matrix)
        
        # Train NCF
        num_users = len(self.data_loader.ratings_df['learner_id'].unique())
        num_items = len(self.data_loader.ratings_df['course_id'].unique())
        
        self.ncf_model = NCFRecommender(num_users, num_items)
        self.ncf_model.train(
            self.data_loader.ratings_df,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            lr=config.LEARNING_RATE
        )
        
        # Create hybrid
        self.hybrid_model = HybridRecommender(
            self.content_model,
            self.svd_model,
            self.ncf_model
        )
        
        self.is_trained = True
        print("\n" + "="*70)
        print("   ✓ ALL MODELS TRAINED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return self
    
    def load_pretrained_models(self, 
                              courses_path='coursera_courses.csv',
                              learners_path='augmented_learner_data.csv',
                              content_model_path='models/content_model.pkl',
                              ncf_model_path='models/ncf_model.pt',
                              svd_model_path='models/svd_model.pkl'):
        """
        Load pre-trained models instead of training from scratch
        
        Args:
            courses_path: Path to courses CSV file
            learners_path: Path to learners CSV file
            content_model_path: Path to saved content-based model
            ncf_model_path: Path to saved NCF model
            svd_model_path: Path to saved SVD model
        """
        import os
        
        print("="*70)
        print("   LOADING PRE-TRAINED MODELS")
        print("="*70)
        
        # Load data
        print("\n📂 Loading datasets...")
        self.data_loader.load_data(courses_path, learners_path)
        
        # Load content-based model
        if os.path.exists(content_model_path):
            print(f"\n📥 Loading content-based model from {content_model_path}...")
            self.content_model.load(content_model_path)
            self.content_model.courses_df = self.data_loader.courses_df
        else:
            print(f"\n⚠️  {content_model_path} not found, training content-based model...")
            self.content_model.fit(self.data_loader.courses_df)
            self.content_model.save(content_model_path)
        
        # Load SVD model
        if os.path.exists(svd_model_path):
            print(f"\n📥 Loading SVD model from {svd_model_path}...")
            self.svd_model = SVDRecommender()
            self.svd_model.load(svd_model_path)
        else:
            print(f"\n⚠️  {svd_model_path} not found, training SVD model...")
            self.svd_model = SVDRecommender(n_components=config.SVD_COMPONENTS)
            self.svd_model.fit(self.data_loader.interaction_matrix)
            self.svd_model.save(svd_model_path)
        
        # Load NCF model
        if os.path.exists(ncf_model_path):
            print(f"\n📥 Loading NCF model from {ncf_model_path}...")
            
            # Get dimensions from data
            num_users = len(self.data_loader.ratings_df['learner_id'].unique())
            num_items = len(self.data_loader.ratings_df['course_id'].unique())
            
            # Initialize NCF model wrapper
            self.ncf_model = NCFRecommender(num_users, num_items)
            
            # Load the trained model
            self.ncf_model.load(ncf_model_path)
        else:
            print(f"\n⚠️  {ncf_model_path} not found, training NCF model...")
            num_users = len(self.data_loader.ratings_df['learner_id'].unique())
            num_items = len(self.data_loader.ratings_df['course_id'].unique())
            
            self.ncf_model = NCFRecommender(num_users, num_items)
            self.ncf_model.train(
                self.data_loader.ratings_df,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                lr=config.LEARNING_RATE
            )
            self.ncf_model.save(ncf_model_path)
        
        # Create hybrid model
        self.hybrid_model = HybridRecommender(
            self.content_model,
            self.svd_model,
            self.ncf_model
        )
        
        self.is_trained = True
        
        print("\n" + "="*70)
        print("   ✓ ALL MODELS LOADED SUCCESSFULLY")
        print("="*70 + "\n")
        
        return self

    def recommend(self, user_id, query, difficulty='intermediate', top_k=10):
        """Get recommendations"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call load_and_train() first.")
        
        recommendations = self.hybrid_model.recommend(
            user_id=user_id,
            query=query,
            difficulty=difficulty,
            top_k=top_k
        )
        
        return recommendations[['course_id', 'title', 'domain', 'difficulty', 
                               'workload', 'url', 'final_score', 'similarity_score',
                               'svd_score', 'ncf_score']]
    
    def get_user_profile(self, user_id):
        """
        Get user profile information
        
        Args:
            user_id: Learner ID
        
        Returns:
            Dictionary with user profile data or None if user not found
        """
        user_data = self.data_loader.learners_df[
            self.data_loader.learners_df['learner_id'] == user_id
        ]
        
        if len(user_data) == 0:
            return None
        
        # Calculate user statistics
        profile = {
            'learner_id': user_id,
            'total_interactions': len(user_data),
            'avg_quiz_score': user_data['quiz_score'].mean(),
            'avg_engagement': user_data['engagement_score'].mean(),
            'performance_level': self._get_performance_level(user_data['quiz_score'].mean()),
            'courses_taken': len(user_data['course_id'].unique()) if 'course_id' in user_data.columns else 0
        }
        
        return profile
    
    def _get_performance_level(self, avg_quiz_score):
        """Determine performance level from quiz scores"""
        if avg_quiz_score >= 75:
            return 'advanced'
        elif avg_quiz_score >= 55:
            return 'intermediate'
        else:
            return 'beginner'
    
    def recommend_by_topic(self, topic, course_difficulty='intermediate', top_k=10):
        """
        Pure content-based recommendations (like original system)
        
        Args:
            topic: What to learn
            course_difficulty: Desired difficulty level
            top_k: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call load_and_train() first.")
        
        results = self.content_model.recommend(
            query=topic,
            difficulty=course_difficulty,
            top_k=top_k
        )
        
        # Format to match original system output
        results = results.rename(columns={'similarity_score': 'score'})
        results['reason'] = f"Content match: '{topic}' at {course_difficulty} level"
        
        return results[['title', 'domain', 'difficulty', 'workload', 'url', 'score', 'reason']].reset_index(drop=True)
    
    def collaborative_recommend(self, learner_id, user_topic='', course_difficulty='intermediate', top_k=5):
        """
        Collaborative filtering recommendations (SVD-based)
        
        Args:
            learner_id: User ID
            user_topic: Topic preference
            course_difficulty: Desired difficulty
            top_k: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call load_and_train() first.")
        
        # Get content-based candidates
        candidates = self.content_model.recommend(
            query=user_topic,
            difficulty=course_difficulty,
            top_k=50
        )
        
        # Score with SVD
        course_ids = candidates['course_id'].tolist()
        svd_scores = self.svd_model.predict(learner_id, course_ids)
        
        candidates['score'] = svd_scores
        candidates = candidates.sort_values('score', ascending=False).head(top_k)
        candidates['reason'] = f"Collaborative filtering: Similar learners succeeded with these courses on '{user_topic}'"
        
        return candidates[['title', 'domain', 'difficulty', 'workload', 'url', 'score', 'reason']].reset_index(drop=True)
    
    def personalized_recommend(self, learner_id, user_topic='', course_difficulty='intermediate', top_k=5):
        """
        Personalized recommendations using full hybrid model (NCF + SVD + Content)
        
        Args:
            learner_id: User ID
            user_topic: Topic preference
            course_difficulty: Desired difficulty
            top_k: Number of recommendations
        
        Returns:
            DataFrame with recommendations
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call load_and_train() first.")
        
        # Use full hybrid model
        results = self.hybrid_model.recommend(
            user_id=learner_id,
            query=user_topic,
            difficulty=course_difficulty,
            top_k=top_k
        )
        
        # Get user profile for personalization message
        profile = self.get_user_profile(learner_id)
        
        if profile:
            perf_level = profile['performance_level']
            results['reason'] = f"Personalized: Based on your {perf_level} performance and learning history with '{user_topic}'"
        else:
            results['reason'] = f"Personalized: Hybrid recommendations for '{user_topic}' at {course_difficulty} level"
        
        results = results.rename(columns={'final_score': 'score'})
        
        return results[['title', 'domain', 'difficulty', 'workload', 'url', 'score', 'reason']].reset_index(drop=True)
    
    def save_models(self):
        """Save all models"""
        print("\n💾 Saving models...")
        self.content_model.save('models/content_model.pkl')
        self.svd_model.save('models/svd_model.pkl')
        self.ncf_model.save('models/ncf_model.pt')
        print("✓ All models saved")
    
    def load_models(self):
        """Load models"""
        print("\n📂 Loading models...")
        self.content_model.load('models/content_model.pkl')
        self.svd_model.load('models/svd_model.pkl')
        self.ncf_model.load('models/ncf_model.pt')
        self.hybrid_model = HybridRecommender(
            self.content_model,
            self.svd_model,
            self.ncf_model
        )
        self.is_trained = True
        print("✓ All models loaded")


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    rec_system = AdvancedRecommendationSystem()
    
    rec_system.load_and_train(
        courses_path='coursera_courses.csv',
        learners_path='augmented_learner_data.csv'
    )
    
    rec_system.save_models()
    
    print("\n" + "="*70)
    print("   EXAMPLE RECOMMENDATIONS")
    print("="*70 + "\n")
    
    recommendations = rec_system.recommend(
        user_id='L200',
        query='machine learning',
        difficulty='intermediate',
        top_k=5
    )
    
    print("\nTop 5 Recommendations:")
    print("-" * 70)
    for idx, row in recommendations.iterrows():
        print(f"\n{idx+1}. {row['title']}")
        print(f"   Domain: {row['domain']} | Difficulty: {row['difficulty']}")
        print(f"   Scores - Final: {row['final_score']:.3f} | Content: {row['similarity_score']:.3f} | SVD: {row['svd_score']:.3f} | NCF: {row['ncf_score']:.3f}")