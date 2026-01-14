"""
Learner Trajectory Simulator - CurricuRL Phase 0
=================================================

Generates realistic learner trajectories for training the RL agent.

Features:
1. Realistic course sequencing based on prerequisites
2. Temporal dynamics (time between courses)
3. Performance modeling (quiz scores, engagement)
4. Dropout simulation
5. Diverse learner archetypes

Author: CurricuRL Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import json
from datetime import datetime, timedelta
import networkx as nx


class LearnerArchetype:
    """
    Different types of learners with distinct behaviors
    """
    
    ARCHETYPES = {
        'high_achiever': {
            'description': 'Completes courses quickly with high scores',
            'avg_quiz_score': 85,
            'quiz_variance': 10,
            'avg_engagement': 85,
            'engagement_variance': 10,
            'completion_rate': 0.9,
            'avg_time_between_courses': 14,  # days
            'dropout_probability': 0.05,
            'difficulty_preference': 'progressive'  # Takes harder courses over time
        },
        'steady_learner': {
            'description': 'Consistent learner with moderate pace',
            'avg_quiz_score': 70,
            'quiz_variance': 15,
            'avg_engagement': 70,
            'engagement_variance': 15,
            'completion_rate': 0.7,
            'avg_time_between_courses': 30,
            'dropout_probability': 0.15,
            'difficulty_preference': 'balanced'
        },
        'struggling_learner': {
            'description': 'Faces challenges, lower completion rate',
            'avg_quiz_score': 55,
            'quiz_variance': 20,
            'avg_engagement': 50,
            'engagement_variance': 20,
            'completion_rate': 0.4,
            'avg_time_between_courses': 60,
            'dropout_probability': 0.35,
            'difficulty_preference': 'easier'  # Prefers easier courses
        },
        'explorer': {
            'description': 'Takes diverse courses but may not finish',
            'avg_quiz_score': 65,
            'quiz_variance': 25,
            'avg_engagement': 60,
            'engagement_variance': 25,
            'completion_rate': 0.5,
            'avg_time_between_courses': 45,
            'dropout_probability': 0.25,
            'difficulty_preference': 'varied'  # Random difficulty
        },
        'focused_specialist': {
            'description': 'Deep dives into specific domain',
            'avg_quiz_score': 80,
            'quiz_variance': 12,
            'avg_engagement': 80,
            'engagement_variance': 12,
            'completion_rate': 0.85,
            'avg_time_between_courses': 20,
            'dropout_probability': 0.08,
            'difficulty_preference': 'progressive',
            'domain_focused': True  # Sticks to one domain
        }
    }


class TrajectorySimulator:
    """
    Simulate realistic learner trajectories
    """
    
    def __init__(self, courses_df: pd.DataFrame, 
                 prerequisite_graph: nx.DiGraph = None,
                 skills_df: pd.DataFrame = None):
        """
        Initialize simulator
        
        Args:
            courses_df: DataFrame with course information
            prerequisite_graph: Graph of course prerequisites (optional)
            skills_df: Course-skill mappings (optional)
        """
        
        print("🎭 Initializing Trajectory Simulator...")
        
        self.courses_df = courses_df
        self.prerequisite_graph = prerequisite_graph
        self.skills_df = skills_df
        
        # Create course lookup - FIXED: using 'id' instead of 'course_id'
        self.course_lookup = courses_df.set_index('id').to_dict('index')
        
        # Group courses by domain and difficulty
        self._group_courses()
        
        print(f"✓ Initialized with {len(courses_df)} courses")
        if prerequisite_graph:
            print(f"✓ Loaded prerequisite graph with {prerequisite_graph.number_of_edges()} edges")
    
    def _group_courses(self):
        """Group courses by domain and difficulty for sampling"""
        
        self.courses_by_domain = {}
        self.courses_by_difficulty = {}
        
        for domain in self.courses_df['domain'].unique():
            self.courses_by_domain[domain] = \
                self.courses_df[self.courses_df['domain'] == domain]['id'].tolist()
        
        if 'difficulty' in self.courses_df.columns:
            for difficulty in self.courses_df['difficulty'].unique():
                self.courses_by_difficulty[difficulty] = \
                    self.courses_df[self.courses_df['difficulty'] == difficulty]['id'].tolist()
        else:
            self.courses_by_difficulty['intermediate'] = self.courses_df['id'].tolist()
    
    def simulate_learner_trajectory(self, 
                                   learner_id: str,
                                   archetype: str,
                                   num_courses: int = None,
                                   start_date: datetime = None) -> List[Dict]:
        """
        Simulate trajectory for a single learner
        
        Args:
            learner_id: Unique learner ID
            archetype: Learner archetype from LearnerArchetype.ARCHETYPES
            num_courses: Number of courses to attempt (None = random 3-15)
            start_date: Starting date (None = random in past year)
            
        Returns:
            List of course interaction records
        """
        
        if archetype not in LearnerArchetype.ARCHETYPES:
            raise ValueError(f"Unknown archetype: {archetype}")
        
        profile = LearnerArchetype.ARCHETYPES[archetype]
        
        # Random number of courses if not specified
        if num_courses is None:
            num_courses = np.random.randint(3, 16)
        
        # Random start date in past year if not specified
        if start_date is None:
            days_ago = np.random.randint(0, 365)
            start_date = datetime.now() - timedelta(days=days_ago)
        
        trajectory = []
        current_date = start_date
        completed_courses = set()
        learner_skill_level = 0  # Tracks learning progress
        
        # Choose initial domain if focused specialist
        if profile.get('domain_focused', False):
            focus_domain = np.random.choice(list(self.courses_by_domain.keys()))
        else:
            focus_domain = None
        
        for course_num in range(num_courses):
            # Check dropout
            if np.random.random() < profile['dropout_probability']:
                break  # Learner drops out
            
            # Select next course
            course_id = self._select_next_course(
                completed_courses=completed_courses,
                learner_skill_level=learner_skill_level,
                difficulty_preference=profile['difficulty_preference'],
                focus_domain=focus_domain
            )
            
            if course_id is None:
                break  # No suitable course found
            
            # Simulate course interaction
            course_info = self.course_lookup[course_id]
            
            # Quiz score (based on difficulty and learner ability)
            course_difficulty_map = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
            course_difficulty_num = course_difficulty_map.get(
                course_info.get('difficulty', 'intermediate').lower(), 1
            )
            
            difficulty_penalty = (course_difficulty_num - learner_skill_level) * 10
            base_score = np.random.normal(
                profile['avg_quiz_score'] - difficulty_penalty,
                profile['quiz_variance']
            )
            quiz_score = np.clip(base_score, 0, 100)
            
            # Engagement score
            engagement_score = np.clip(
                np.random.normal(profile['avg_engagement'], profile['engagement_variance']),
                0, 100
            )
            
            # Completion (based on completion rate and performance)
            completion_prob = profile['completion_rate'] * (1 + (quiz_score - 50) / 100)
            completed = np.random.random() < completion_prob
            
            # Time spent (correlated with engagement)
            estimated_hours = course_info.get('estimated_hours', 10)
            time_spent_hours = estimated_hours * (engagement_score / 100) * np.random.uniform(0.8, 1.2)
            
            # Create record
            record = {
                'learner_id': learner_id,
                'course_id': course_id,
                'course_name': course_info.get('name', 'Unknown'),
                'course_domain': course_info.get('domain', 'Unknown'),
                'course_difficulty': course_info.get('difficulty', 'intermediate'),
                'archetype': archetype,
                'course_number': course_num + 1,
                'enrollment_date': current_date.strftime('%Y-%m-%d'),
                'quiz_score': round(quiz_score, 1),
                'engagement_score': round(engagement_score, 1),
                'time_spent_hours': round(time_spent_hours, 1),
                'completed': completed,
                'learner_skill_level': learner_skill_level
            }
            
            trajectory.append(record)
            
            # Update state
            if completed:
                completed_courses.add(course_id)
                # Skill improvement based on performance
                skill_gain = (quiz_score / 100) * 0.3
                learner_skill_level = min(learner_skill_level + skill_gain, 2.0)
            
            # Move to next course date
            days_between = int(np.random.exponential(profile['avg_time_between_courses']))
            current_date += timedelta(days=days_between)
        
        return trajectory
    
    def _select_next_course(self, 
                           completed_courses: set,
                           learner_skill_level: float,
                           difficulty_preference: str,
                           focus_domain: str = None) -> str:
        """
        Select next course based on learner profile
        """
        
        # Get available courses (not yet completed)
        available_courses = [
            cid for cid in self.course_lookup.keys()
            if cid not in completed_courses
        ]
        
        if len(available_courses) == 0:
            return None
        
        # Filter by prerequisites if graph available
        if self.prerequisite_graph:
            available_courses = [
                cid for cid in available_courses
                if self._prerequisites_met(cid, completed_courses)
            ]
            
            if len(available_courses) == 0:
                return None
        
        # Filter by domain if focused
        if focus_domain and focus_domain in self.courses_by_domain:
            domain_courses = [
                cid for cid in available_courses
                if cid in self.courses_by_domain[focus_domain]
            ]
            if len(domain_courses) > 0:
                available_courses = domain_courses
        
        # Select based on difficulty preference
        if difficulty_preference == 'progressive':
            # Match difficulty to skill level
            target_difficulty = ['beginner', 'intermediate', 'advanced'][int(learner_skill_level)]
            preferred = [
                cid for cid in available_courses
                if self.course_lookup[cid].get('difficulty', 'intermediate').lower() == target_difficulty
            ]
            if len(preferred) > 0:
                return np.random.choice(preferred)
        
        elif difficulty_preference == 'easier':
            # Prefer beginner/intermediate
            preferred = [
                cid for cid in available_courses
                if self.course_lookup[cid].get('difficulty', 'intermediate').lower() in ['beginner', 'intermediate']
            ]
            if len(preferred) > 0:
                return np.random.choice(preferred)
        
        # Random selection as fallback
        return np.random.choice(available_courses)
    
    def _prerequisites_met(self, course_id: str, completed_courses: set) -> bool:
        """Check if prerequisites are met for a course"""
        
        if course_id not in self.prerequisite_graph:
            return True
        
        # Get all prerequisites
        prerequisites = list(self.prerequisite_graph.predecessors(course_id))
        
        # Check if all are completed
        return all(prereq in completed_courses for prereq in prerequisites)
    
    def generate_dataset(self, 
                        num_learners: int = 1000,
                        archetype_distribution: Dict[str, float] = None,
                        output_path: str = 'data/learner_trajectories.csv') -> pd.DataFrame:
        """
        Generate complete dataset of learner trajectories
        
        Args:
            num_learners: Number of learners to simulate
            archetype_distribution: Distribution of archetypes (None = uniform)
            output_path: Where to save data
            
        Returns:
            DataFrame with all trajectories
        """
        
        print(f"\n🎲 Generating {num_learners} learner trajectories...")
        
        # Default uniform distribution
        if archetype_distribution is None:
            archetypes = list(LearnerArchetype.ARCHETYPES.keys())
            archetype_distribution = {a: 1/len(archetypes) for a in archetypes}
        
        # Sample archetypes
        archetypes = list(archetype_distribution.keys())
        probs = list(archetype_distribution.values())
        learner_archetypes = np.random.choice(archetypes, size=num_learners, p=probs)
        
        all_trajectories = []
        
        for i in range(num_learners):
            learner_id = f'L{i+1:05d}'
            archetype = learner_archetypes[i]
            
            trajectory = self.simulate_learner_trajectory(
                learner_id=learner_id,
                archetype=archetype
            )
            
            all_trajectories.extend(trajectory)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{num_learners} learners...")
        
        # Create DataFrame
        df = pd.DataFrame(all_trajectories)
        
        # Statistics
        print(f"\n✓ Generation complete!")
        print(f"  Total interactions: {len(df)}")
        print(f"  Unique learners: {df['learner_id'].nunique()}")
        print(f"  Unique courses: {df['course_id'].nunique()}")
        print(f"  Avg courses per learner: {len(df) / num_learners:.1f}")
        print(f"  Overall completion rate: {df['completed'].mean():.1%}")
        print(f"  Avg quiz score: {df['quiz_score'].mean():.1f}")
        
        # Archetype breakdown
        print(f"\n  Archetype distribution:")
        archetype_counts = df.groupby('archetype')['learner_id'].nunique()
        for archetype, count in archetype_counts.items():
            print(f"    {archetype}: {count} learners ({count/num_learners:.1%})")
        
        # Save
        Path(output_path).parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved to {output_path}")
        
        return df
    
    def analyze_trajectories(self, df: pd.DataFrame):
        """
        Analyze generated trajectories
        """
        
        print("\n📊 Trajectory Analysis:")
        
        # Learning progression
        avg_scores_by_course_num = df.groupby('course_number')['quiz_score'].mean()
        print(f"\n  Learning progression (avg quiz score by course number):")
        for course_num, score in avg_scores_by_course_num.head(10).items():
            print(f"    Course {course_num}: {score:.1f}")
        
        # Domain diversity
        learner_domains = df.groupby('learner_id')['course_domain'].nunique()
        print(f"\n  Domain diversity:")
        print(f"    Avg domains per learner: {learner_domains.mean():.1f}")
        
        # Difficulty progression
        if 'course_difficulty' in df.columns:
            print(f"\n  Difficulty distribution:")
            diff_dist = df['course_difficulty'].value_counts()
            for diff, count in diff_dist.items():
                print(f"    {diff}: {count} ({count/len(df):.1%})")
        
        # Completion patterns
        completion_by_archetype = df.groupby('archetype')['completed'].mean()
        print(f"\n  Completion rate by archetype:")
        for archetype, rate in completion_by_archetype.items():
            print(f"    {archetype}: {rate:.1%}")


def main():
    """
    Main execution pipeline
    """
    
    print("="*70)
    print("  LEARNER TRAJECTORY SIMULATION - CurricuRL Phase 0")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    courses_df = pd.read_csv('coursera_courses.csv')
    print(f"✓ Loaded {len(courses_df)} courses")
    
    # Load prerequisite graph if available
    try:
        import pickle
        with open('data/prerequisite_graph.gpickle', 'rb') as f:
            prereq_graph = pickle.load(f)
        print(f"✓ Loaded prerequisite graph")
    except:
        print("⚠️  No prerequisite graph found")
        prereq_graph = None
    
    # Load skills if available
    try:
        skills_df = pd.read_csv('data/course_skills_raw.csv')
        print(f"✓ Loaded skills data")
    except:
        skills_df = None
    
    # Initialize simulator
    simulator = TrajectorySimulator(
        courses_df=courses_df,
        prerequisite_graph=prereq_graph,
        skills_df=skills_df
    )
    
    # Define archetype distribution (realistic mix)
    archetype_dist = {
        'high_achiever': 0.15,
        'steady_learner': 0.40,
        'struggling_learner': 0.20,
        'explorer': 0.15,
        'focused_specialist': 0.10
    }
    
    # Generate dataset
    trajectories_df = simulator.generate_dataset(
        num_learners=1000,
        archetype_distribution=archetype_dist,
        output_path='data/learner_trajectories.csv'
    )
    
    # Analyze
    simulator.analyze_trajectories(trajectories_df)
    
    # Also save a version compatible with original augmented_learner_data.csv format
    legacy_format = trajectories_df[['learner_id', 'course_id', 'quiz_score', 'engagement_score']].copy()
    # Add content_type from domain
    legacy_format['content_type'] = trajectories_df['course_domain']
    legacy_format.to_csv('data/augmented_learner_data_v2.csv', index=False)
    print(f"\n💾 Saved legacy format to data/augmented_learner_data_v2.csv")
    
    print("\n" + "="*70)
    print("  ✓ TRAJECTORY SIMULATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - data/learner_trajectories.csv (full trajectories)")
    print(f"  - data/augmented_learner_data_v2.csv (legacy format)")


if __name__ == "__main__":
    main()