# import streamlit as st
# import pandas as pd
# from typing import Dict, List, Optional

# # Page configuration
# st.set_page_config(
#     page_title="Course Recommendation System",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS for dark theme
# st.markdown("""
# <style>
#     /* Global dark theme */
#     .main {
#         background-color: #0e1117;
#         padding: 1.5rem 2.5rem;
#     }
    
#     .stApp {
#         background-color: #0e1117;
#     }
    
#     /* Header styling */
#     .header-container {
#         background: linear-gradient(135deg, #1a2332 0%, #1e293b 100%);
#         padding: 2.5rem 2rem;
#         border-radius: 16px;
#         margin-bottom: 2rem;
#         box-shadow: 0 8px 20px rgba(0, 0, 0, 0.5);
#         border: 1px solid rgba(100, 116, 139, 0.3);
#     }
    
#     .header-title {
#         color: #e2e8f0;
#         font-size: 2.5rem;
#         font-weight: 700;
#         margin: 0;
#         text-align: center;
#         text-shadow: 0 2px 8px rgba(0,0,0,0.3);
#     }
    
#     .header-subtitle {
#         color: rgba(148, 163, 184, 0.9);
#         font-size: 1.1rem;
#         text-align: center;
#         margin-top: 0.5rem;
#     }
    
#     /* Input section styling */
#     .input-section {
#         background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
#         padding: 2rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
#         margin-bottom: 2rem;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#     }
    
#     .section-title {
#         font-size: 1.4rem;
#         font-weight: 600;
#         color: #cbd5e1;
#         margin-bottom: 1.5rem;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }
    
#     /* Course card styling */
#     .course-card {
#         background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
#         padding: 1.75rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
#         margin-bottom: 1.25rem;
#         transition: all 0.3s ease;
#         border-left: 4px solid #475569;
#         position: relative;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#     }
    
#     .course-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
#         border-left-color: #64748b;
#     }
    
#     .course-number {
#         position: absolute;
#         top: 1rem;
#         right: 1rem;
#         background: linear-gradient(135deg, #334155 0%, #475569 100%);
#         color: #e2e8f0;
#         width: 32px;
#         height: 32px;
#         border-radius: 50%;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-weight: 700;
#         font-size: 0.9rem;
#         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
#     }
    
#     .course-title {
#         font-size: 1.25rem;
#         font-weight: 600;
#         color: #e2e8f0;
#         margin-bottom: 1rem;
#         padding-right: 40px;
#         line-height: 1.4;
#     }
    
#     .course-meta {
#         display: flex;
#         gap: 0.75rem;
#         flex-wrap: wrap;
#         margin-bottom: 1rem;
#     }
    
#     .meta-badge {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.35rem;
#         padding: 0.4rem 0.85rem;
#         border-radius: 20px;
#         font-size: 0.85rem;
#         font-weight: 500;
#     }
    
#     .badge-domain {
#         background: rgba(59, 130, 246, 0.15);
#         color: #93c5fd;
#         border: 1px solid rgba(59, 130, 246, 0.3);
#     }
    
#     .badge-difficulty {
#         background: rgba(251, 146, 60, 0.15);
#         color: #fdba74;
#         border: 1px solid rgba(251, 146, 60, 0.3);
#     }
    
#     .badge-workload {
#         background: rgba(34, 197, 94, 0.15);
#         color: #86efac;
#         border: 1px solid rgba(34, 197, 94, 0.3);
#     }
    
#     .badge-score {
#         background: rgba(168, 85, 247, 0.15);
#         color: #c084fc;
#         border: 1px solid rgba(168, 85, 247, 0.3);
#     }
    
#     .course-reason {
#         color: #94a3b8;
#         font-size: 0.9rem;
#         margin-bottom: 0.75rem;
#         line-height: 1.5;
#         font-style: italic;
#     }
    
#     .course-url {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.5rem;
#         color: #60a5fa;
#         text-decoration: none;
#         font-size: 0.9rem;
#         font-weight: 600;
#         transition: all 0.2s;
#     }
    
#     .course-url:hover {
#         color: #93c5fd;
#         gap: 0.75rem;
#     }
    
#     /* Profile card styling */
#     .profile-card {
#         background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
#         padding: 2rem;
#         border-radius: 12px;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
#         border: 1px solid rgba(100, 116, 139, 0.3);
#     }
    
#     .profile-header {
#         display: flex;
#         align-items: center;
#         gap: 1rem;
#         margin-bottom: 1.5rem;
#     }
    
#     .profile-icon {
#         width: 60px;
#         height: 60px;
#         background: linear-gradient(135deg, #334155 0%, #475569 100%);
#         border-radius: 50%;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-size: 1.8rem;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
#     }
    
#     .profile-title {
#         font-size: 1.5rem;
#         font-weight: 600;
#         color: #e2e8f0;
#     }
    
#     .profile-subtitle {
#         font-size: 0.9rem;
#         color: #94a3b8;
#     }
    
#     .profile-stats {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
#         gap: 1rem;
#         margin-top: 1rem;
#     }
    
#     .stat-card {
#         background: rgba(30, 41, 59, 0.6);
#         padding: 1.25rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
#         transition: transform 0.2s;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#     }
    
#     .stat-card:hover {
#         transform: translateY(-2px);
#     }
    
#     .stat-label {
#         font-size: 0.8rem;
#         color: #94a3b8;
#         font-weight: 500;
#         margin-bottom: 0.5rem;
#         text-transform: uppercase;
#         letter-spacing: 0.5px;
#     }
    
#     .stat-value {
#         font-size: 1.4rem;
#         color: #e2e8f0;
#         font-weight: 700;
#     }
    
#     .stat-icon {
#         margin-right: 0.5rem;
#     }
    
#     /* Tab styling enhancement */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 10px;
#         background-color: rgba(30, 41, 59, 0.5);
#         padding: 0.75rem;
#         border-radius: 12px;
#         box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         height: 55px;
#         padding: 0 2rem;
#         background-color: rgba(51, 65, 85, 0.5);
#         border-radius: 10px;
#         font-weight: 600;
#         font-size: 1rem;
#         color: #94a3b8;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#         transition: all 0.2s;
#     }
    
#     .stTabs [data-baseweb="tab"]:hover {
#         border-color: rgba(100, 116, 139, 0.4);
#         background-color: rgba(51, 65, 85, 0.7);
#     }
    
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #334155 0%, #475569 100%);
#         color: #e2e8f0;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
#         border-color: rgba(100, 116, 139, 0.4);
#     }
    
#     /* Empty state styling */
#     .empty-state {
#         text-align: center;
#         padding: 4rem 2rem;
#         color: #64748b;
#     }
    
#     .empty-state-icon {
#         font-size: 4rem;
#         margin-bottom: 1rem;
#         opacity: 0.5;
#     }
    
#     .empty-state-title {
#         font-size: 1.4rem;
#         font-weight: 600;
#         color: #94a3b8;
#         margin-bottom: 0.5rem;
#     }
    
#     .empty-state-text {
#         font-size: 1rem;
#         color: #64748b;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #334155 0%, #475569 100%);
#         color: #e2e8f0;
#         border: 1px solid rgba(100, 116, 139, 0.3);
#         border-radius: 10px;
#         padding: 0.75rem 2.5rem;
#         font-weight: 600;
#         font-size: 1rem;
#         transition: all 0.3s;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
#         background: linear-gradient(135deg, #3b4a5e 0%, #526375 100%);
#     }
    
#     /* Loading state */
#     .loading-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         padding: 3rem;
#     }
    
#     /* Remove default Streamlit padding */
#     .block-container {
#         padding-top: 1rem;
#         max-width: 1400px;
#     }
    
#     /* Input styling */
#     .stTextInput > div > div > input {
#         background-color: rgba(30, 41, 59, 0.5);
#         border-radius: 8px;
#         border: 1px solid rgba(100, 116, 139, 0.3);
#         padding: 0.75rem 1rem;
#         font-size: 1rem;
#         color: #e2e8f0;
#     }
    
#     .stTextInput > div > div > input:focus {
#         border-color: #64748b;
#         box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.2);
#         background-color: rgba(30, 41, 59, 0.7);
#     }
    
#     .stSelectbox > div > div {
#         background-color: rgba(30, 41, 59, 0.5);
#         border-radius: 8px;
#         border: 1px solid rgba(100, 116, 139, 0.3);
#         color: #e2e8f0;
#     }
    
#     .stSelectbox [data-baseweb="select"] {
#         background-color: rgba(30, 41, 59, 0.5);
#     }
    
#     /* Result count badge */
#     .result-count {
#         display: inline-block;
#         background: rgba(100, 116, 139, 0.3);
#         color: #cbd5e1;
#         padding: 0.35rem 1rem;
#         border-radius: 20px;
#         font-size: 0.9rem;
#         font-weight: 600;
#         margin-left: 0.75rem;
#         border: 1px solid rgba(100, 116, 139, 0.4);
#     }
    
#     /* Divider styling */
#     hr {
#         border-color: rgba(100, 116, 139, 0.2);
#         margin: 2rem 0;
#     }
    
#     /* Info box styling */
#     .stInfo {
#         background-color: rgba(59, 130, 246, 0.1);
#         border-left-color: #3b82f6;
#     }
    
#     /* Error box styling */
#     .stError {
#         background-color: rgba(239, 68, 68, 0.1);
#         border-left-color: #ef4444;
#     }
    
#     /* Success box styling */
#     .stSuccess {
#         background-color: rgba(34, 197, 94, 0.1);
#         border-left-color: #22c55e;
#     }
    
#     /* Tab description text */
#     .tab-description {
#         padding: 1rem 0;
#         color: #94a3b8;
#         background: rgba(30, 41, 59, 0.3);
#         padding: 1rem;
#         border-radius: 8px;
#         margin-bottom: 1.5rem;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#     }
# </style>
# """, unsafe_allow_html=True)


# def render_header():
#     """Render the application header"""
#     st.markdown("""
#     <div class="header-container">
#         <h1 class="header-title"> Course Recommendation System</h1>
#         <p class="header-subtitle">Discover personalized learning paths </p>
#     </div>
#     """, unsafe_allow_html=True)


# def render_course_card(course: pd.Series, index: int):
#     """Render a single course card with all details"""
#     # Format score as percentage if it's a decimal between 0 and 1
#     score = course.get('score', 0)
#     if 0 <= score <= 1:
#         score_display = f"{score * 100:.1f}%"
#     else:
#         score_display = f"{score:.2f}"
    
#     st.markdown(f"""
#     <div class="course-card">
#         <div class="course-number">{index}</div>
#         <div class="course-title">{course['title']}</div>
#         <div class="course-meta">
#             <span class="meta-badge badge-domain">📚 {course['domain']}</span>
#             <span class="meta-badge badge-difficulty">⚡ {course['difficulty'].title()}</span>
#             <span class="meta-badge badge-workload">⏱️ {course['workload']}</span>
#             <span class="meta-badge badge-score">⭐ Match: {score_display}</span>
#         </div>
#         <div class="course-reason">{course.get('reason', '')}</div>
#         <a href="{course['url']}" target="_blank" class="course-url">
#             View Course →
#         </a>
#     </div>
#     """, unsafe_allow_html=True)


# def render_profile_card(profile: Dict, learner_id: str):
#     """Render learner profile card with stats"""
#     st.markdown(f"""
#     <div class="profile-card">
#         <div class="profile-header">
#             <div class="profile-icon">👤</div>
#             <div>
#                 <div class="profile-title">Learner Profile</div>
#                 <div class="profile-subtitle">ID: {learner_id}</div>
#             </div>
#         </div>
#         <div class="profile-stats">
#             <div class="stat-card">
#                 <div class="stat-label">Performance Level</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">🎯</span>{profile.get('performance_level', 'N/A').title()}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Avg Quiz Score</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">📊</span>{profile.get('avg_quiz_score', 0):.1f}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Avg Engagement</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">💡</span>{profile.get('avg_engagement', 0):.1f}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Preferred Content</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">📖</span>{profile.get('preferred_type', 'N/A').title()}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Total Interactions</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">🔢</span>{profile.get('total_interactions', 0)}
#                 </div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)


# def render_empty_state(title: str, message: str, icon: str = "📭"):
#     """Render an empty state when no results are found"""
#     st.markdown(f"""
#     <div class="empty-state">
#         <div class="empty-state-icon">{icon}</div>
#         <div class="empty-state-title">{title}</div>
#         <div class="empty-state-text">{message}</div>
#     </div>
#     """, unsafe_allow_html=True)


# def main():
#     """Main application logic"""
    
#     # Render header
#     render_header()
    
#     # Input Section
#     st.markdown('<div class="input-section">', unsafe_allow_html=True)
#     st.markdown('<div class="section-title">🔍 Configure Your Learning Preferences</div>', unsafe_allow_html=True)
    
#     # Create two rows of inputs
#     col1, col2 = st.columns([2, 2])
    
#     with col1:
#         topic = st.text_input(
#             "📌 What topic do you want to learn?",
#             placeholder="e.g., Machine Learning, Python, Data Science",
#             key="topic_input",
#             help="Enter the subject or topic you're interested in learning"
#         )
    
#     with col2:
#         skill_level = st.selectbox(
#             "🎯 What is your current skill level?",
#             options=["Beginner", "Intermediate", "Advanced"],
#             index=0,
#             key="skill_level_select",
#             help="Your current proficiency in the topic"
#         )
    
#     col3, col4 = st.columns([2, 2])
    
#     with col3:
#         difficulty = st.selectbox(
#             "⚡ Preferred course difficulty",
#             options=["Beginner", "Intermediate", "Advanced"],
#             index=0,
#             key="difficulty_select",
#             help="The difficulty level you want for the recommended courses"
#         )
    
#     with col4:
#         learner_id = st.text_input(
#             "👤 Learner ID (Optional)",
#             placeholder="e.g., L200, L001",
#             key="learner_input",
#             help="Enter your learner ID for personalized recommendations"
#         )
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Get Recommendations Button
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         get_recommendations = st.button("🚀 Get Recommendations", use_container_width=True)
    
#     # Only proceed if button is clicked or we have results in session state
#     if get_recommendations or 'recommendations_ready' in st.session_state:
        
#         if get_recommendations:
#             # Validate inputs
#             if not topic:
#                 st.error("⚠️ Please enter a topic to get recommendations.")
#                 return
            
#             # Show loading state
#             with st.spinner("🔄 Generating personalized recommendations..."):
#                 try:
#                     # Import recommendation functions from the existing recommender.py
#                     from recommender import (
#                         recommend_by_topic,
#                         collaborative_recommend,
#                         personalized_recommend,
#                         user_profiles,
#                         get_excluded_set
#                     )
                    
#                     # Store inputs in session state
#                     st.session_state.topic = topic
#                     st.session_state.skill_level = skill_level
#                     st.session_state.difficulty = difficulty.lower()
#                     st.session_state.learner_id = learner_id.strip() if learner_id else None
                    
#                     # Get content-based recommendations
#                     st.session_state.content_recs = recommend_by_topic(
#                         topic=topic,
#                         course_difficulty=difficulty.lower(),
#                         top_k=10
#                     )
                    
#                     # Get collaborative and personalized recommendations if learner ID provided
#                     if st.session_state.learner_id:
#                         exclude_set = get_excluded_set([st.session_state.content_recs])
                        
#                         st.session_state.collab_recs = collaborative_recommend(
#                             learner_id=st.session_state.learner_id,
#                             user_topic=topic,
#                             course_difficulty=difficulty.lower(),
#                             top_k=5,
#                             exclude_courses=exclude_set
#                         )
                        
#                         exclude_set = get_excluded_set([
#                             st.session_state.content_recs,
#                             st.session_state.collab_recs
#                         ])
                        
#                         st.session_state.personal_recs = personalized_recommend(
#                             learner_id=st.session_state.learner_id,
#                             user_topic=topic,
#                             course_difficulty=difficulty.lower(),
#                             top_k=5,
#                             exclude_courses=exclude_set
#                         )
                        
#                         # Get user profile
#                         st.session_state.user_profile = user_profiles.get(st.session_state.learner_id)
#                     else:
#                         st.session_state.collab_recs = None
#                         st.session_state.personal_recs = None
#                         st.session_state.user_profile = None
                    
#                     st.session_state.recommendations_ready = True
                    
#                 except Exception as e:
#                     st.error(f"❌ Error generating recommendations: {str(e)}")
#                     st.error("Please ensure 'recommender.py', 'coursera_courses.csv', and 'augmented_learner_data.csv' are in the same directory.")
#                     return
        
#         # Display results if available
#         if st.session_state.get('recommendations_ready'):
            
#             # Show search summary
#             st.markdown("---")
#             summary_text = f"**Showing results for:** {st.session_state.topic} | **Your skill level:** {st.session_state.skill_level} | **Difficulty:** {st.session_state.difficulty.title()}"
#             if st.session_state.get('learner_id'):
#                 summary_text += f" | **Learner ID:** {st.session_state.learner_id}"
#             st.markdown(f'<div style="color: #94a3b8; padding: 1rem; text-align: center;">{summary_text}</div>', unsafe_allow_html=True)
            
#             # Show learner profile if available
#             if st.session_state.get('user_profile') and st.session_state.get('learner_id'):
#                 st.markdown("---")
#                 render_profile_card(st.session_state.user_profile, st.session_state.learner_id)
            
#             st.markdown("---")
            
#             # Create tabs for different recommendation types
#             if st.session_state.get('learner_id') and st.session_state.get('collab_recs') is not None:
#                 tabs = st.tabs([
#                     f"📚 Content-Based ({len(st.session_state.content_recs)})",
#                     f"🤝 Collaborative ({len(st.session_state.collab_recs)})",
#                     f"✨ Personalized ({len(st.session_state.personal_recs)})"
#                 ])
                
#                 # Content-Based Tab
#                 with tabs[0]:
#                     st.markdown("""
#                     <div class="tab-description">
#                         <strong>Content-Based Recommendations</strong><br>
#                         Courses that match your topic and difficulty preferences based on content similarity
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     if len(st.session_state.content_recs) > 0:
#                         for idx, row in st.session_state.content_recs.iterrows():
#                             render_course_card(row, idx + 1)
#                     else:
#                         render_empty_state(
#                             "No Results Found",
#                             "Try adjusting your search criteria or difficulty level",
#                             "🔍"
#                         )
                
#                 # Collaborative Tab
#                 with tabs[1]:
#                     st.markdown("""
#                     <div class="tab-description">
#                         <strong>Collaborative Filtering</strong><br>
#                         Courses recommended based on what learners similar to you have succeeded with
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     if len(st.session_state.collab_recs) > 0:
#                         for idx, row in st.session_state.collab_recs.iterrows():
#                             render_course_card(row, idx + 1)
#                     else:
#                         render_empty_state(
#                             "No Collaborative Results",
#                             "Not enough similar learner data available",
#                             "👥"
#                         )
                
#                 # Personalized Tab
#                 with tabs[2]:
#                     st.markdown("""
#                     <div class="tab-description">
#                         <strong>Personalized Recommendations</strong><br>
#                         Tailored specifically to your learning history, performance, and preferences
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     if len(st.session_state.personal_recs) > 0:
#                         for idx, row in st.session_state.personal_recs.iterrows():
#                             render_course_card(row, idx + 1)
#                     else:
#                         render_empty_state(
#                             "No Personalized Results",
#                             "Unable to generate personalized recommendations",
#                             "✨"
#                         )
            
#             else:
#                 # Only show content-based recommendations
#                 st.markdown(f"""
#                 <div style="padding: 1rem 0;">
#                     <h3 style="color: #cbd5e1;">📚 Content-Based Recommendations <span class="result-count">{len(st.session_state.content_recs)} courses</span></h3>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown("""
#                 <div class="tab-description">
#                     Courses that match your topic and difficulty preferences based on content similarity
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 if len(st.session_state.content_recs) > 0:
#                     for idx, row in st.session_state.content_recs.iterrows():
#                         render_course_card(row, idx + 1)
#                 else:
#                     render_empty_state(
#                         "No Results Found",
#                         "Try adjusting your search criteria or difficulty level",
#                         "🔍"
#                     )
                
#                 # Suggestion to add learner ID
#                 st.markdown("---")
#                 st.info("💡 **Tip**: Enter a Learner ID above to unlock Collaborative and Personalized recommendations tailored to your learning history!")


# if __name__ == "__main__":
#     main()


# import streamlit as st
# import pandas as pd
# from typing import Dict, List, Optional
# import sys

# # Page configuration
# st.set_page_config(
#     page_title="AI-Powered Course Recommender",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # Custom CSS for dark theme
# st.markdown("""
# <style>
#     /* Global dark theme */
#     .main {
#         background-color: #0e1117;
#         padding: 1.5rem 2.5rem;
#     }
    
#     .stApp {
#         background-color: #0e1117;
#     }
    
#     /* Header styling */
#     .header-container {
#         background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
#         padding: 3rem 2rem;
#         border-radius: 20px;
#         margin-bottom: 2.5rem;
#         box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
#         border: 1px solid rgba(38, 208, 206, 0.3);
#         position: relative;
#         overflow: hidden;
#     }
    
#     .header-container::before {
#         content: '';
#         position: absolute;
#         top: -50%;
#         right: -50%;
#         width: 200%;
#         height: 200%;
#         background: radial-gradient(circle, rgba(38, 208, 206, 0.15) 0%, transparent 70%);
#     }
    
#     .header-title {
#         color: #ffffff;
#         font-size: 2.8rem;
#         font-weight: 800;
#         margin: 0;
#         text-align: center;
#         text-shadow: 0 4px 10px rgba(0,0,0,0.5);
#         position: relative;
#         z-index: 1;
#         letter-spacing: -0.5px;
#     }
    
#     .header-subtitle {
#         color: rgba(255, 255, 255, 0.95);
#         font-size: 1.15rem;
#         text-align: center;
#         margin-top: 0.75rem;
#         position: relative;
#         z-index: 1;
#         font-weight: 300;
#     }
    
#     /* Input section styling */
#     .input-section {
#         background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
#         padding: 2.5rem;
#         border-radius: 16px;
#         box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
#         margin-bottom: 2.5rem;
#         border: 1px solid rgba(100, 116, 139, 0.3);
#     }
    
#     .section-title {
#         font-size: 1.5rem;
#         font-weight: 700;
#         color: #cbd5e1;
#         margin-bottom: 1.75rem;
#         display: flex;
#         align-items: center;
#         gap: 0.75rem;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     /* Course card styling */
#     .course-card {
#         background: linear-gradient(135deg, #1e293b 0%, #1a2433 100%);
#         padding: 2rem;
#         border-radius: 16px;
#         box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
#         margin-bottom: 1.5rem;
#         transition: all 0.3s ease;
#         border-left: 5px solid #26d0ce;
#         position: relative;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#     }
    
#     .course-card:hover {
#         transform: translateY(-6px);
#         box-shadow: 0 12px 28px rgba(38, 208, 206, 0.2);
#         border-left-color: #1a2980;
#     }
    
#     .course-number {
#         position: absolute;
#         top: 1.25rem;
#         right: 1.25rem;
#         background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
#         color: white;
#         width: 40px;
#         height: 40px;
#         border-radius: 50%;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-weight: 800;
#         font-size: 1rem;
#         box-shadow: 0 4px 12px rgba(38, 208, 206, 0.4);
#     }
    
#     .course-title {
#         font-size: 1.35rem;
#         font-weight: 700;
#         color: #f1f5f9;
#         margin-bottom: 1.25rem;
#         padding-right: 50px;
#         line-height: 1.5;
#     }
    
#     .course-meta {
#         display: flex;
#         gap: 0.85rem;
#         flex-wrap: wrap;
#         margin-bottom: 1.25rem;
#     }
    
#     .meta-badge {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.4rem;
#         padding: 0.5rem 1rem;
#         border-radius: 25px;
#         font-size: 0.875rem;
#         font-weight: 600;
#         transition: all 0.2s ease;
#     }
    
#     .meta-badge:hover {
#         transform: translateY(-2px);
#     }
    
#     .badge-domain {
#         background: rgba(59, 130, 246, 0.2);
#         color: #93c5fd;
#         border: 1.5px solid rgba(59, 130, 246, 0.4);
#     }
    
#     .badge-difficulty {
#         background: rgba(251, 146, 60, 0.2);
#         color: #fdba74;
#         border: 1.5px solid rgba(251, 146, 60, 0.4);
#     }
    
#     .badge-workload {
#         background: rgba(34, 197, 94, 0.2);
#         color: #86efac;
#         border: 1.5px solid rgba(34, 197, 94, 0.4);
#     }
    
#     .badge-score {
#         background: rgba(168, 85, 247, 0.2);
#         color: #c084fc;
#         border: 1.5px solid rgba(168, 85, 247, 0.4);
#     }
    
#     .course-reason {
#         color: #94a3b8;
#         font-size: 0.95rem;
#         margin-bottom: 1rem;
#         line-height: 1.6;
#         font-style: italic;
#         padding-left: 1rem;
#         border-left: 3px solid rgba(38, 208, 206, 0.3);
#     }
    
#     .course-url {
#         display: inline-flex;
#         align-items: center;
#         gap: 0.5rem;
#         color: #26d0ce;
#         text-decoration: none;
#         font-size: 0.95rem;
#         font-weight: 700;
#         transition: all 0.3s;
#         padding: 0.5rem 1rem;
#         border-radius: 8px;
#         background: rgba(38, 208, 206, 0.1);
#     }
    
#     .course-url:hover {
#         color: #ffffff;
#         background: rgba(38, 208, 206, 0.2);
#         gap: 0.85rem;
#     }
    
#     /* Profile card styling */
#     .profile-card {
#         background: linear-gradient(135deg, #1e293b 0%, #1a2433 100%);
#         padding: 2.5rem;
#         border-radius: 16px;
#         margin-bottom: 2.5rem;
#         box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
#         border: 1px solid rgba(38, 208, 206, 0.3);
#     }
    
#     .profile-header {
#         display: flex;
#         align-items: center;
#         gap: 1.5rem;
#         margin-bottom: 2rem;
#     }
    
#     .profile-icon {
#         width: 70px;
#         height: 70px;
#         background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
#         border-radius: 50%;
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         font-size: 2rem;
#         box-shadow: 0 6px 16px rgba(38, 208, 206, 0.4);
#     }
    
#     .profile-title {
#         font-size: 1.75rem;
#         font-weight: 700;
#         color: #f1f5f9;
#     }
    
#     .profile-subtitle {
#         font-size: 1rem;
#         color: #94a3b8;
#         margin-top: 0.25rem;
#     }
    
#     .profile-stats {
#         display: grid;
#         grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
#         gap: 1.25rem;
#         margin-top: 1.5rem;
#     }
    
#     .stat-card {
#         background: rgba(30, 41, 59, 0.7);
#         padding: 1.5rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
#         transition: all 0.3s;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#     }
    
#     .stat-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 8px 20px rgba(38, 208, 206, 0.2);
#         border-color: rgba(38, 208, 206, 0.4);
#     }
    
#     .stat-label {
#         font-size: 0.8rem;
#         color: #94a3b8;
#         font-weight: 600;
#         margin-bottom: 0.75rem;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     .stat-value {
#         font-size: 1.6rem;
#         color: #f1f5f9;
#         font-weight: 800;
#         display: flex;
#         align-items: center;
#         gap: 0.5rem;
#     }
    
#     .stat-icon {
#         font-size: 1.25rem;
#     }
    
#     /* Tab styling enhancement */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 12px;
#         background-color: rgba(30, 41, 59, 0.6);
#         padding: 1rem;
#         border-radius: 16px;
#         box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         height: 60px;
#         padding: 0 2.5rem;
#         background-color: rgba(51, 65, 85, 0.5);
#         border-radius: 12px;
#         font-weight: 700;
#         font-size: 1.05rem;
#         color: #94a3b8;
#         border: 2px solid rgba(100, 116, 139, 0.2);
#         transition: all 0.3s;
#     }
    
#     .stTabs [data-baseweb="tab"]:hover {
#         border-color: rgba(38, 208, 206, 0.4);
#         background-color: rgba(51, 65, 85, 0.7);
#     }
    
#     .stTabs [aria-selected="true"] {
#         background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
#         color: #ffffff;
#         box-shadow: 0 6px 16px rgba(38, 208, 206, 0.4);
#         border-color: rgba(38, 208, 206, 0.5);
#     }
    
#     /* Empty state styling */
#     .empty-state {
#         text-align: center;
#         padding: 5rem 2rem;
#         color: #64748b;
#     }
    
#     .empty-state-icon {
#         font-size: 5rem;
#         margin-bottom: 1.5rem;
#         opacity: 0.4;
#     }
    
#     .empty-state-title {
#         font-size: 1.6rem;
#         font-weight: 700;
#         color: #94a3b8;
#         margin-bottom: 0.75rem;
#     }
    
#     .empty-state-text {
#         font-size: 1.05rem;
#         color: #64748b;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
#         color: white;
#         border: none;
#         border-radius: 12px;
#         padding: 0.9rem 3rem;
#         font-weight: 700;
#         font-size: 1.05rem;
#         transition: all 0.3s;
#         box-shadow: 0 6px 16px rgba(38, 208, 206, 0.4);
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-3px);
#         box-shadow: 0 10px 25px rgba(38, 208, 206, 0.6);
#     }
    
#     /* Input styling */
#     .stTextInput > div > div > input {
#         background-color: rgba(30, 41, 59, 0.6);
#         border-radius: 10px;
#         border: 2px solid rgba(100, 116, 139, 0.3);
#         padding: 0.85rem 1.25rem;
#         font-size: 1rem;
#         color: #f1f5f9;
#         transition: all 0.3s;
#     }
    
#     .stTextInput > div > div > input:focus {
#         border-color: #26d0ce;
#         box-shadow: 0 0 0 3px rgba(38, 208, 206, 0.2);
#         background-color: rgba(30, 41, 59, 0.8);
#     }
    
#     .stTextInput > div > div > input::placeholder {
#         color: #64748b;
#     }
    
#     .stSelectbox > div > div {
#         background-color: rgba(30, 41, 59, 0.6);
#         border-radius: 10px;
#         border: 2px solid rgba(100, 116, 139, 0.3);
#         color: #f1f5f9;
#     }
    
#     .stSelectbox [data-baseweb="select"] {
#         background-color: rgba(30, 41, 59, 0.6);
#     }
    
#     /* Result count badge */
#     .result-count {
#         display: inline-block;
#         background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
#         color: white;
#         padding: 0.4rem 1.25rem;
#         border-radius: 25px;
#         font-size: 0.95rem;
#         font-weight: 700;
#         margin-left: 1rem;
#         box-shadow: 0 4px 12px rgba(38, 208, 206, 0.3);
#     }
    
#     /* Loading animation */
#     .loading-text {
#         color: #26d0ce;
#         font-size: 1.1rem;
#         font-weight: 600;
#         text-align: center;
#         padding: 2rem;
#     }
    
#     /* Divider */
#     hr {
#         border-color: rgba(100, 116, 139, 0.3);
#         margin: 2.5rem 0;
#     }
    
#     /* Info box */
#     .stInfo {
#         background-color: rgba(38, 208, 206, 0.15);
#         border-left-color: #26d0ce;
#         color: #cbd5e1;
#     }
    
#     /* Error box */
#     .stError {
#         background-color: rgba(239, 68, 68, 0.15);
#         border-left-color: #ef4444;
#     }
    
#     /* Success box */
#     .stSuccess {
#         background-color: rgba(34, 197, 94, 0.15);
#         border-left-color: #22c55e;
#     }
    
#     /* Tab description */
#     .tab-description {
#         padding: 1.25rem;
#         color: #94a3b8;
#         background: rgba(30, 41, 59, 0.4);
#         border-radius: 10px;
#         margin-bottom: 2rem;
#         border: 1px solid rgba(100, 116, 139, 0.2);
#         font-size: 0.95rem;
#         line-height: 1.6;
#     }
    
#     /* Remove default padding */
#     .block-container {
#         padding-top: 1rem;
#         max-width: 1400px;
#     }
    
#     /* Model badge */
#     .model-badge {
#         display: inline-block;
#         padding: 0.35rem 0.85rem;
#         border-radius: 20px;
#         font-size: 0.75rem;
#         font-weight: 700;
#         text-transform: uppercase;
#         letter-spacing: 0.5px;
#         margin-left: 0.5rem;
#     }
    
#     .model-ncf {
#         background: rgba(168, 85, 247, 0.2);
#         color: #c084fc;
#         border: 1px solid rgba(168, 85, 247, 0.4);
#     }
    
#     .model-svd {
#         background: rgba(59, 130, 246, 0.2);
#         color: #93c5fd;
#         border: 1px solid rgba(59, 130, 246, 0.4);
#     }
    
#     .model-content {
#         background: rgba(34, 197, 94, 0.2);
#         color: #86efac;
#         border: 1px solid rgba(34, 197, 94, 0.4);
#     }
# </style>
# """, unsafe_allow_html=True)


# def render_header():
#     """Render the application header"""
#     st.markdown("""
#     <div class="header-container">
#         <h1 class="header-title">🎓 AI-Powered Course Recommender</h1>
#         <p class="header-subtitle">Advanced hybrid recommendation system powered by Neural Collaborative Filtering, SVD, and Deep Learning</p>
#     </div>
#     """, unsafe_allow_html=True)


# def render_course_card(course: pd.Series, index: int, show_model_info: bool = False):
#     """Render a single course card with all details"""
#     # Format score as percentage
#     score = course.get('score', 0)
#     if 0 <= score <= 1:
#         score_display = f"{score * 100:.1f}%"
#     else:
#         score_display = f"{score:.2f}"
    
#     # Model badges if available
#     model_badges = ""
#     if show_model_info and 'final_score' in course:
#         model_badges = '<span class="model-badge model-ncf">NCF</span><span class="model-badge model-svd">SVD</span><span class="model-badge model-content">Content</span>'
    
#     st.markdown(f"""
#     <div class="course-card">
#         <div class="course-number">{index}</div>
#         <div class="course-title">{course['title']}{model_badges}</div>
#         <div class="course-meta">
#             <span class="meta-badge badge-domain">📚 {course['domain']}</span>
#             <span class="meta-badge badge-difficulty">⚡ {course['difficulty'].title()}</span>
#             <span class="meta-badge badge-workload">⏱️ {course['workload']}</span>
#             <span class="meta-badge badge-score">⭐ Match: {score_display}</span>
#         </div>
#         <div class="course-reason">{course.get('reason', '')}</div>
#         <a href="{course['url']}" target="_blank" class="course-url">
#             View Course →
#         </a>
#     </div>
#     """, unsafe_allow_html=True)


# def render_profile_card(profile: Dict, learner_id: str):
#     """Render learner profile card with stats"""
#     st.markdown(f"""
#     <div class="profile-card">
#         <div class="profile-header">
#             <div class="profile-icon">👤</div>
#             <div>
#                 <div class="profile-title">Learner Profile</div>
#                 <div class="profile-subtitle">ID: {learner_id}</div>
#             </div>
#         </div>
#         <div class="profile-stats">
#             <div class="stat-card">
#                 <div class="stat-label">Performance Level</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">🎯</span>{profile.get('performance_level', 'N/A').title()}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Avg Quiz Score</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">📊</span>{profile.get('avg_quiz_score', 0):.1f}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Avg Engagement</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">💡</span>{profile.get('avg_engagement', 0):.1f}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Courses Taken</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">📖</span>{profile.get('courses_taken', 0)}
#                 </div>
#             </div>
#             <div class="stat-card">
#                 <div class="stat-label">Total Interactions</div>
#                 <div class="stat-value">
#                     <span class="stat-icon">🔢</span>{profile.get('total_interactions', 0)}
#                 </div>
#             </div>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)


# def render_empty_state(title: str, message: str, icon: str = "📭"):
#     """Render an empty state"""
#     st.markdown(f"""
#     <div class="empty-state">
#         <div class="empty-state-icon">{icon}</div>
#         <div class="empty-state-title">{title}</div>
#         <div class="empty-state-text">{message}</div>
#     </div>
#     """, unsafe_allow_html=True)


# def main():
#     """Main application logic"""
    
#     # Render header
#     render_header()
    
#     # Input Section
#     st.markdown('<div class="input-section">', unsafe_allow_html=True)
#     st.markdown('<div class="section-title">🔍 Configure Your Learning Journey</div>', unsafe_allow_html=True)
    
#     # Create two rows of inputs
#     col1, col2 = st.columns([2, 2])
    
#     with col1:
#         topic = st.text_input(
#             "📌 What course/topic do you want to learn?",
#             placeholder="e.g., Machine Learning, Python, Data Science, Web Development",
#             key="topic_input",
#             help="Enter the subject or topic you're interested in learning"
#         )
    
#     with col2:
#         skill_level = st.selectbox(
#             "🎯 What is your current proficiency level?",
#             options=["Beginner", "Intermediate", "Advanced"],
#             index=1,
#             key="skill_level_select",
#             help="Your current proficiency/experience in this topic area"
#         )
    
#     col3, col4 = st.columns([2, 2])
    
#     with col3:
#         difficulty = st.selectbox(
#             "⚡ Preferred course difficulty level",
#             options=["Beginner", "Intermediate", "Advanced"],
#             index=1,
#             key="difficulty_select",
#             help="The difficulty level you want for the recommended courses"
#         )
    
#     with col4:
#         learner_id = st.text_input(
#             "👤 Learner ID (Optional)",
#             placeholder="e.g., L200, L001, L100",
#             key="learner_input",
#             help="Enter your learner ID for personalized AI-powered recommendations"
#         )
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Get Recommendations Button
#     col1, col2, col3 = st.columns([1, 1, 1])
#     with col2:
#         get_recommendations = st.button("🚀 Get AI Recommendations", use_container_width=True)
    
#     # Process recommendations
#     if get_recommendations or 'recommendations_ready' in st.session_state:
        
#         if get_recommendations:
#             # Validate inputs
#             if not topic:
#                 st.error("⚠️ Please enter a topic to get recommendations.")
#                 return
            
#             # Show loading state
#             with st.spinner("🔄 Training advanced AI models and generating personalized recommendations..."):
#                 try:
#                     # Import the advanced recommendation system
#                     from recommender import AdvancedRecommendationSystem
                    
#                     # Initialize or load from session
#                     if 'rec_system' not in st.session_state:
#                         rec_system = AdvancedRecommendationSystem()
#                         rec_system.load_and_train(
#                             courses_path='coursera_courses.csv',
#                             learners_path='augmented_learner_data.csv'
#                         )
#                         st.session_state.rec_system = rec_system
#                     else:
#                         rec_system = st.session_state.rec_system
                    
#                     # Store inputs
#                     st.session_state.topic = topic
#                     st.session_state.skill_level = skill_level
#                     st.session_state.difficulty = difficulty.lower()
#                     st.session_state.learner_id = learner_id.strip() if learner_id else None
                    
#                     # Get content-based recommendations
#                     st.session_state.content_recs = rec_system.recommend_by_topic(
#                         topic=topic,
#                         course_difficulty=difficulty.lower(),
#                         top_k=10
#                     )
                    
#                     # Get collaborative and personalized if learner ID provided
#                     if st.session_state.learner_id:
#                         st.session_state.collab_recs = rec_system.collaborative_recommend(
#                             learner_id=st.session_state.learner_id,
#                             user_topic=topic,
#                             course_difficulty=difficulty.lower(),
#                             top_k=5
#                         )
                        
#                         st.session_state.personal_recs = rec_system.personalized_recommend(
#                             learner_id=st.session_state.learner_id,
#                             user_topic=topic,
#                             course_difficulty=difficulty.lower(),
#                             top_k=5
#                         )
                        
#                         # Get user profile
#                         st.session_state.user_profile = rec_system.get_user_profile(
#                             st.session_state.learner_id
#                         )
#                     else:
#                         st.session_state.collab_recs = None
#                         st.session_state.personal_recs = None
#                         st.session_state.user_profile = None
                    
#                     st.session_state.recommendations_ready = True
                    
#                 except Exception as e:
#                     st.error(f"❌ Error generating recommendations: {str(e)}")
#                     st.error("Please ensure 'recommender.py', 'coursera_courses.csv', and 'augmented_learner_data.csv' are in the same directory.")
#                     import traceback
#                     st.code(traceback.format_exc())
#                     return
        
#         # Display results
#         if st.session_state.get('recommendations_ready'):
            
#             # Show search summary
#             st.markdown("---")
#             summary_text = f"**Search:** {st.session_state.topic} | **Your Level:** {st.session_state.skill_level} | **Course Difficulty:** {st.session_state.difficulty.title()}"
#             if st.session_state.get('learner_id'):
#                 summary_text += f" | **Learner ID:** {st.session_state.learner_id}"
#             st.markdown(f'<div style="color: #94a3b8; padding: 1.25rem; text-align: center; font-size: 1.05rem; background: rgba(30, 41, 59, 0.4); border-radius: 10px;">{summary_text}</div>', unsafe_allow_html=True)
            
#             # Show learner profile if available
#             if st.session_state.get('user_profile') and st.session_state.get('learner_id'):
#                 st.markdown("---")
#                 render_profile_card(st.session_state.user_profile, st.session_state.learner_id)
            
#             st.markdown("---")
            
#             # Create tabs for different recommendation types
#             if st.session_state.get('learner_id') and st.session_state.get('collab_recs') is not None:
#                 tabs = st.tabs([
#                     f"📚 Content-Based ({len(st.session_state.content_recs)})",
#                     f"🤝 Collaborative - SVD ({len(st.session_state.collab_recs)})",
#                     f"✨ Personalized - AI Hybrid ({len(st.session_state.personal_recs)})"
#                 ])
                
#                 # Content-Based Tab
#                 with tabs[0]:
#                     st.markdown("""
#                     <div class="tab-description">
#                         <strong>📚 Content-Based Filtering</strong><br>
#                         Uses TF-IDF vectorization to match courses based on title, description, and domain similarity to your query.
#                         <span class="model-badge model-content">TF-IDF + Cosine Similarity</span>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     if len(st.session_state.content_recs) > 0:
#                         for idx, row in st.session_state.content_recs.iterrows():
#                             render_course_card(row, idx + 1)
#                     else:
#                         render_empty_state(
#                             "No Results Found",
#                             "Try adjusting your search criteria or difficulty level",
#                             "🔍"
#                         )
                
#                 # Collaborative Tab
#                 with tabs[1]:
#                     st.markdown("""
#                     <div class="tab-description">
#                         <strong>🤝 Collaborative Filtering (SVD)</strong><br>
#                         Uses Singular Value Decomposition to analyze patterns from similar learners and predict courses you'll enjoy.
#                         <span class="model-badge model-svd">Matrix Factorization (SVD)</span>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     if len(st.session_state.collab_recs) > 0:
#                         for idx, row in st.session_state.collab_recs.iterrows():
#                             render_course_card(row, idx + 1)
#                     else:
#                         render_empty_state(
#                             "No Collaborative Results",
#                             "Not enough similar learner data available for this query",
#                             "👥"
#                         )
                
#                 # Personalized Tab
#                 with tabs[2]:
#                     st.markdown("""
#                     <div class="tab-description">
#                         <strong>✨ Personalized AI Recommendations (Hybrid Model)</strong><br>
#                         Combines Neural Collaborative Filtering (NCF), SVD, and Content-Based filtering for the most accurate predictions tailored to your learning history and performance.
#                         <span class="model-badge model-ncf">Neural CF</span>
#                         <span class="model-badge model-svd">SVD</span>
#                         <span class="model-badge model-content">Content</span>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     if len(st.session_state.personal_recs) > 0:
#                         for idx, row in st.session_state.personal_recs.iterrows():
#                             render_course_card(row, idx + 1, show_model_info=True)
#                     else:
#                         render_empty_state(
#                             "No Personalized Results",
#                             "Unable to generate personalized recommendations for this query",
#                             "✨"
#                         )
            
#             else:
#                 # Only show content-based recommendations
#                 st.markdown(f"""
#                 <div style="padding: 1.5rem 0;">
#                     <h3 style="color: #cbd5e1; font-size: 1.75rem; font-weight: 700;">📚 Content-Based Recommendations <span class="result-count">{len(st.session_state.content_recs)} courses</span></h3>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 st.markdown("""
#                 <div class="tab-description">
#                     <strong>Content-Based Filtering</strong> - Courses matched based on content similarity
#                     <span class="model-badge model-content">TF-IDF</span>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 if len(st.session_state.content_recs) > 0:
#                     for idx, row in st.session_state.content_recs.iterrows():
#                         render_course_card(row, idx + 1)
#                 else:
#                     render_empty_state(
#                         "No Results Found",
#                         "Try adjusting your search criteria or difficulty level",
#                         "🔍"
#                     )
                
#                 # Suggestion to add learner ID
#                 st.markdown("---")
#                 st.info("💡 **Unlock Advanced AI Recommendations!** Enter a Learner ID above to access:\n\n• **Collaborative Filtering** using SVD (Matrix Factorization)\n• **Personalized Hybrid Model** combining Neural Collaborative Filtering, SVD, and Content-Based approaches\n• **Your Learning Profile** with performance analytics")


# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import sys

# Page configuration
st.set_page_config(
    page_title="AI-Powered Course Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Global dark theme */
    .main {
        background-color: #0e1117;
        padding: 1.5rem 2.5rem;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(38, 208, 206, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(38, 208, 206, 0.15) 0%, transparent 70%);
    }
    
    .header-title {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        text-shadow: 0 4px 10px rgba(0,0,0,0.5);
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.15rem;
        text-align: center;
        margin-top: 0.75rem;
        position: relative;
        z-index: 1;
        font-weight: 300;
    }
    
    /* Input section styling */
    .input-section {
        background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 2.5rem;
        border: 1px solid rgba(100, 116, 139, 0.3);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #cbd5e1;
        margin-bottom: 1.75rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Course card styling */
    .course-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a2433 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border-left: 5px solid #26d0ce;
        position: relative;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    .course-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 28px rgba(38, 208, 206, 0.2);
        border-left-color: #1a2980;
    }
    
    .course-number {
        position: absolute;
        top: 1.25rem;
        right: 1.25rem;
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(38, 208, 206, 0.4);
    }
    
    .course-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1.25rem;
        padding-right: 50px;
        line-height: 1.5;
    }
    
    .course-meta {
        display: flex;
        gap: 0.85rem;
        flex-wrap: wrap;
        margin-bottom: 1.25rem;
    }
    
    .meta-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.875rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .meta-badge:hover {
        transform: translateY(-2px);
    }
    
    .badge-domain {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1.5px solid rgba(59, 130, 246, 0.4);
    }
    
    .badge-difficulty {
        background: rgba(251, 146, 60, 0.2);
        color: #fdba74;
        border: 1.5px solid rgba(251, 146, 60, 0.4);
    }
    
    .badge-workload {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1.5px solid rgba(34, 197, 94, 0.4);
    }
    
    .badge-score {
        background: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1.5px solid rgba(168, 85, 247, 0.4);
    }
    
    .course-reason {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        font-style: italic;
        padding-left: 1rem;
        border-left: 3px solid rgba(38, 208, 206, 0.3);
    }
    
    .course-url {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #26d0ce;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 700;
        transition: all 0.3s;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        background: rgba(38, 208, 206, 0.1);
    }
    
    .course-url:hover {
        color: #ffffff;
        background: rgba(38, 208, 206, 0.2);
        gap: 0.85rem;
    }
    
    /* Profile card styling */
    .profile-card {
        background: linear-gradient(135deg, #1e293b 0%, #1a2433 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2.5rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(38, 208, 206, 0.3);
    }
    
    .profile-header {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .profile-icon {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        box-shadow: 0 6px 16px rgba(38, 208, 206, 0.4);
    }
    
    .profile-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    .profile-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    .profile-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.25rem;
        margin-top: 1.5rem;
    }
    
    .stat-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.3s;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(38, 208, 206, 0.2);
        border-color: rgba(38, 208, 206, 0.4);
    }
    
    .stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 1.6rem;
        color: #f1f5f9;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stat-icon {
        font-size: 1.25rem;
    }
    
    /* Tab styling enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(30, 41, 59, 0.6);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 2.5rem;
        background-color: rgba(51, 65, 85, 0.5);
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #94a3b8;
        border: 2px solid rgba(100, 116, 139, 0.2);
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: rgba(38, 208, 206, 0.4);
        background-color: rgba(51, 65, 85, 0.7);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: #ffffff;
        box-shadow: 0 6px 16px rgba(38, 208, 206, 0.4);
        border-color: rgba(38, 208, 206, 0.5);
    }
    
    /* Empty state styling */
    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: #64748b;
    }
    
    .empty-state-icon {
        font-size: 5rem;
        margin-bottom: 1.5rem;
        opacity: 0.4;
    }
    
    .empty-state-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    
    .empty-state-text {
        font-size: 1.05rem;
        color: #64748b;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 3rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s;
        box-shadow: 0 6px 16px rgba(38, 208, 206, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(38, 208, 206, 0.6);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        border: 2px solid rgba(100, 116, 139, 0.3);
        padding: 0.85rem 1.25rem;
        font-size: 1rem;
        color: #f1f5f9;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #26d0ce;
        box-shadow: 0 0 0 3px rgba(38, 208, 206, 0.2);
        background-color: rgba(30, 41, 59, 0.8);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #64748b;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        border: 2px solid rgba(100, 116, 139, 0.3);
        color: #f1f5f9;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: rgba(30, 41, 59, 0.6);
    }
    
    /* Result count badge */
    .result-count {
        display: inline-block;
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        padding: 0.4rem 1.25rem;
        border-radius: 25px;
        font-size: 0.95rem;
        font-weight: 700;
        margin-left: 1rem;
        box-shadow: 0 4px 12px rgba(38, 208, 206, 0.3);
    }
    
    /* Loading animation */
    .loading-text {
        color: #26d0ce;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        padding: 2rem;
    }
    
    /* Divider */
    hr {
        border-color: rgba(100, 116, 139, 0.3);
        margin: 2.5rem 0;
    }
    
    /* Info box */
    .stInfo {
        background-color: rgba(38, 208, 206, 0.15);
        border-left-color: #26d0ce;
        color: #cbd5e1;
    }
    
    /* Error box */
    .stError {
        background-color: rgba(239, 68, 68, 0.15);
        border-left-color: #ef4444;
    }
    
    /* Success box */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.15);
        border-left-color: #22c55e;
    }
    
    /* Tab description */
    .tab-description {
        padding: 1.25rem;
        color: #94a3b8;
        background: rgba(30, 41, 59, 0.4);
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid rgba(100, 116, 139, 0.2);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }
    
    /* Model badge */
    .model-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-left: 0.5rem;
    }
    
    .model-ncf {
        background: rgba(168, 85, 247, 0.2);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    
    .model-svd {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .model-content {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the application header"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🎓 AI-Powered Course Recommender</h1>
        <p class="header-subtitle">Advanced hybrid recommendation system powered by Neural Collaborative Filtering, SVD, and Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)


def render_course_card(course: pd.Series, index: int, show_model_info: bool = False):
    """Render a single course card with all details"""
    # Format score as percentage
    score = course.get('score', 0)
    if 0 <= score <= 1:
        score_display = f"{score * 100:.1f}%"
    else:
        score_display = f"{score:.2f}"
    
    # Model badges if available
    model_badges = ""
    if show_model_info and 'final_score' in course:
        model_badges = '<span class="model-badge model-ncf">NCF</span><span class="model-badge model-svd">SVD</span><span class="model-badge model-content">Content</span>'
    
    st.markdown(f"""
    <div class="course-card">
        <div class="course-number">{index}</div>
        <div class="course-title">{course['title']}{model_badges}</div>
        <div class="course-meta">
            <span class="meta-badge badge-domain">📚 {course['domain']}</span>
            <span class="meta-badge badge-difficulty">⚡ {course['difficulty'].title()}</span>
            <span class="meta-badge badge-workload">⏱️ {course['workload']}</span>
            <span class="meta-badge badge-score">⭐ Match: {score_display}</span>
        </div>
        <div class="course-reason">{course.get('reason', '')}</div>
        <a href="{course['url']}" target="_blank" class="course-url">
            View Course →
        </a>
    </div>
    """, unsafe_allow_html=True)


def render_profile_card(profile: Dict, learner_id: str):
    """Render learner profile card with stats"""
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-header">
            <div class="profile-icon">👤</div>
            <div>
                <div class="profile-title">Learner Profile</div>
                <div class="profile-subtitle">ID: {learner_id}</div>
            </div>
        </div>
        <div class="profile-stats">
            <div class="stat-card">
                <div class="stat-label">Performance Level</div>
                <div class="stat-value">
                    <span class="stat-icon">🎯</span>{profile.get('performance_level', 'N/A').title()}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Quiz Score</div>
                <div class="stat-value">
                    <span class="stat-icon">📊</span>{profile.get('avg_quiz_score', 0):.1f}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Engagement</div>
                <div class="stat-value">
                    <span class="stat-icon">💡</span>{profile.get('avg_engagement', 0):.1f}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Courses Taken</div>
                <div class="stat-value">
                    <span class="stat-icon">📖</span>{profile.get('courses_taken', 0)}
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Interactions</div>
                <div class="stat-value">
                    <span class="stat-icon">🔢</span>{profile.get('total_interactions', 0)}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(title: str, message: str, icon: str = "📭"):
    """Render an empty state"""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-state-icon">{icon}</div>
        <div class="empty-state-title">{title}</div>
        <div class="empty-state-text">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application logic"""
    
    # Render header
    render_header()
    
    # Input Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Configure Your Learning Journey</div>', unsafe_allow_html=True)
    
    # Create two rows of inputs
    col1, col2 = st.columns([2, 2])
    
    with col1:
        topic = st.text_input(
            "📌 What course/topic do you want to learn?",
            placeholder="e.g., Machine Learning, Python, Data Science, Web Development",
            key="topic_input",
            help="Enter the subject or topic you're interested in learning"
        )
    
    with col2:
        skill_level = st.selectbox(
            "🎯 What is your current proficiency level?",
            options=["Beginner", "Intermediate", "Advanced"],
            index=1,
            key="skill_level_select",
            help="Your current proficiency/experience in this topic area"
        )
    
    col3, col4 = st.columns([2, 2])
    
    with col3:
        difficulty = st.selectbox(
            "⚡ Preferred course difficulty level",
            options=["Beginner", "Intermediate", "Advanced"],
            index=1,
            key="difficulty_select",
            help="The difficulty level you want for the recommended courses"
        )
    
    with col4:
        learner_id = st.text_input(
            "👤 Learner ID (Optional)",
            placeholder="e.g., L200, L001, L100",
            key="learner_input",
            help="Enter your learner ID for personalized AI-powered recommendations"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get Recommendations Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        get_recommendations = st.button("🚀 Get AI Recommendations", use_container_width=True)
    
    # Process recommendations
    if get_recommendations or 'recommendations_ready' in st.session_state:
        
        if get_recommendations:
            # Validate inputs
            if not topic:
                st.error("⚠️ Please enter a topic to get recommendations.")
                return
            
            # Show loading state
            with st.spinner("🔄 Loading pre-trained AI models and generating personalized recommendations..."):
                try:
                    # Import the advanced recommendation system
                    from recommender import AdvancedRecommendationSystem
                    
                    # Initialize or load from session
                    if 'rec_system' not in st.session_state:
                        rec_system = AdvancedRecommendationSystem()
                        # Load pre-trained models instead of training
                        rec_system.load_pretrained_models(
                            courses_path='coursera_courses.csv',
                            learners_path='augmented_learner_data.csv',
                            content_model_path='models/content_model.pkl',
                            ncf_model_path='models/ncf_model.pt',
                            svd_model_path='models/svd_model.pkl'
                        )
                        st.session_state.rec_system = rec_system
                    else:
                        rec_system = st.session_state.rec_system
                    
                    # Store inputs
                    st.session_state.topic = topic
                    st.session_state.skill_level = skill_level
                    st.session_state.difficulty = difficulty.lower()
                    st.session_state.learner_id = learner_id.strip() if learner_id else None
                    
                    # Get content-based recommendations
                    st.session_state.content_recs = rec_system.recommend_by_topic(
                        topic=topic,
                        course_difficulty=difficulty.lower(),
                        top_k=10
                    )
                    
                    # Get collaborative and personalized if learner ID provided
                    if st.session_state.learner_id:
                        st.session_state.collab_recs = rec_system.collaborative_recommend(
                            learner_id=st.session_state.learner_id,
                            user_topic=topic,
                            course_difficulty=difficulty.lower(),
                            top_k=5
                        )
                        
                        st.session_state.personal_recs = rec_system.personalized_recommend(
                            learner_id=st.session_state.learner_id,
                            user_topic=topic,
                            course_difficulty=difficulty.lower(),
                            top_k=5
                        )
                        
                        # Get user profile
                        st.session_state.user_profile = rec_system.get_user_profile(
                            st.session_state.learner_id
                        )
                    else:
                        st.session_state.collab_recs = None
                        st.session_state.personal_recs = None
                        st.session_state.user_profile = None
                    
                    st.session_state.recommendations_ready = True
                    
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {str(e)}")
                    st.error("Please ensure the following files exist:")
                    st.error("• recommender.py")
                    st.error("• coursera_courses.csv")
                    st.error("• augmented_learner_data.csv")
                    st.error("• models/content_model.pkl")
                    st.error("• models/ncf_model.pt")
                    st.error("• models/svd_model.pkl")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        # Display results
        if st.session_state.get('recommendations_ready'):
            
            # Show search summary
            st.markdown("---")
            summary_text = f"**Search:** {st.session_state.topic} | **Your Level:** {st.session_state.skill_level} | **Course Difficulty:** {st.session_state.difficulty.title()}"
            if st.session_state.get('learner_id'):
                summary_text += f" | **Learner ID:** {st.session_state.learner_id}"
            st.markdown(f'<div style="color: #94a3b8; padding: 1.25rem; text-align: center; font-size: 1.05rem; background: rgba(30, 41, 59, 0.4); border-radius: 10px;">{summary_text}</div>', unsafe_allow_html=True)
            
            # Show learner profile if available
            if st.session_state.get('user_profile') and st.session_state.get('learner_id'):
                st.markdown("---")
                render_profile_card(st.session_state.user_profile, st.session_state.learner_id)
            
            st.markdown("---")
            
            # Create tabs for different recommendation types
            if st.session_state.get('learner_id') and st.session_state.get('collab_recs') is not None:
                tabs = st.tabs([
                    f"📚 Content-Based ({len(st.session_state.content_recs)})",
                    f"🤝 Collaborative - SVD ({len(st.session_state.collab_recs)})",
                    f"✨ Personalized - AI Hybrid ({len(st.session_state.personal_recs)})"
                ])
                
                # Content-Based Tab
                with tabs[0]:
                    st.markdown("""
                    <div class="tab-description">
                        <strong>📚 Content-Based Filtering</strong><br>
                        Uses TF-IDF vectorization to match courses based on title, description, and domain similarity to your query.
                        <span class="model-badge model-content">TF-IDF + Cosine Similarity</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if len(st.session_state.content_recs) > 0:
                        for idx, row in st.session_state.content_recs.iterrows():
                            render_course_card(row, idx + 1)
                    else:
                        render_empty_state(
                            "No Results Found",
                            "Try adjusting your search criteria or difficulty level",
                            "🔍"
                        )
                
                # Collaborative Tab
                with tabs[1]:
                    st.markdown("""
                    <div class="tab-description">
                        <strong>🤝 Collaborative Filtering (SVD)</strong><br>
                        Uses Singular Value Decomposition to analyze patterns from similar learners and predict courses you'll enjoy.
                        <span class="model-badge model-svd">Matrix Factorization (SVD)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if len(st.session_state.collab_recs) > 0:
                        for idx, row in st.session_state.collab_recs.iterrows():
                            render_course_card(row, idx + 1)
                    else:
                        render_empty_state(
                            "No Collaborative Results",
                            "Not enough similar learner data available for this query",
                            "👥"
                        )
                
                # Personalized Tab
                with tabs[2]:
                    st.markdown("""
                    <div class="tab-description">
                        <strong>✨ Personalized AI Recommendations (Hybrid Model)</strong><br>
                        Combines Neural Collaborative Filtering (NCF), SVD, and Content-Based filtering for the most accurate predictions tailored to your learning history and performance.
                        <span class="model-badge model-ncf">Neural CF</span>
                        <span class="model-badge model-svd">SVD</span>
                        <span class="model-badge model-content">Content</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if len(st.session_state.personal_recs) > 0:
                        for idx, row in st.session_state.personal_recs.iterrows():
                            render_course_card(row, idx + 1, show_model_info=True)
                    else:
                        render_empty_state(
                            "No Personalized Results",
                            "Unable to generate personalized recommendations for this query",
                            "✨"
                        )
            
            else:
                # Only show content-based recommendations
                st.markdown(f"""
                <div style="padding: 1.5rem 0;">
                    <h3 style="color: #cbd5e1; font-size: 1.75rem; font-weight: 700;">📚 Content-Based Recommendations <span class="result-count">{len(st.session_state.content_recs)} courses</span></h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="tab-description">
                    <strong>Content-Based Filtering</strong> - Courses matched based on content similarity
                    <span class="model-badge model-content">TF-IDF</span>
                </div>
                """, unsafe_allow_html=True)
                
                if len(st.session_state.content_recs) > 0:
                    for idx, row in st.session_state.content_recs.iterrows():
                        render_course_card(row, idx + 1)
                else:
                    render_empty_state(
                        "No Results Found",
                        "Try adjusting your search criteria or difficulty level",
                        "🔍"
                    )
                
                # Suggestion to add learner ID
                st.markdown("---")
                st.info("💡 **Unlock Advanced AI Recommendations!** Enter a Learner ID above to access:\n\n• **Collaborative Filtering** using SVD (Matrix Factorization)\n• **Personalized Hybrid Model** combining Neural Collaborative Filtering, SVD, and Content-Based approaches\n• **Your Learning Profile** with performance analytics")


if __name__ == "__main__":
    main()