# """
# Interactive Recommendation Script
# ==================================
# This script uses the TRAINED models to get recommendations.
# No training happens here - only inference.

# Usage:
#     python get_recommendations.py

# Make sure you've already trained the models by running recommender.py first!
# """

# from recommender import AdvancedRecommendationSystem
# import pandas as pd

# def get_user_input():
#     """Get user inputs interactively"""
#     print("\n" + "="*70)
#     print("   🎓 AI COURSE RECOMMENDER - GET RECOMMENDATIONS")
#     print("="*70 + "\n")
    
#     # Get User ID
#     user_id = input("👤 Enter your User ID (e.g., L200): ").strip()
#     if not user_id:
#         user_id = "L200"
#         print(f"   → Using default: {user_id}")
    
#     # Get course query
#     query = input("📖 What do you want to learn? (e.g., machine learning): ").strip()
#     if not query:
#         query = "machine learning"
#         print(f"   → Using default: {query}")
    
#     # Get difficulty
#     print("\n📊 Select difficulty level:")
#     print("   1. Beginner")
#     print("   2. Intermediate")
#     print("   3. Advanced")
#     diff_choice = input("Enter choice (1-3): ").strip()
    
#     difficulty_map = {
#         '1': 'beginner',
#         '2': 'intermediate',
#         '3': 'advanced'
#     }
#     difficulty = difficulty_map.get(diff_choice, 'intermediate')
#     print(f"   → Selected: {difficulty.title()}")
    
#     # Get number of recommendations
#     try:
#         top_k = int(input("\n🔢 How many recommendations? (default 5): ").strip() or "5")
#         top_k = max(1, min(top_k, 20))  # Limit between 1-20
#     except:
#         top_k = 5
#     print(f"   → Getting {top_k} recommendations")
    
#     return user_id, query, difficulty, top_k


# def display_recommendations(recommendations):
#     """Display recommendations in a formatted way"""
#     print("\n" + "="*70)
#     print("   ✨ YOUR PERSONALIZED RECOMMENDATIONS")
#     print("="*70 + "\n")
    
#     for idx, row in recommendations.iterrows():
#         print(f"\n{'='*70}")
#         print(f"  🎓 RECOMMENDATION #{idx + 1}")
#         print(f"{'='*70}")
#         print(f"\n  📚 Title: {row['title']}")
#         print(f"  🏷️  Domain: {row['domain']}")
#         print(f"  📊 Difficulty: {row['difficulty'].title()}")
#         print(f"  ⏱️  Workload: {row['workload']}")
        
#         if pd.notna(row['url']) and row['url'].strip():
#             print(f"  🔗 URL: {row['url']}")
        
#         print(f"\n  📈 SCORES:")
#         print(f"     ⭐ Final Score:      {row['final_score']:.4f}")
#         print(f"     📝 Content Score:    {row['similarity_score']:.4f}")
#         print(f"     📐 SVD Score:        {row['svd_score']:.4f}")
#         print(f"     🧠 Neural Score:     {row['ncf_score']:.4f}")
    
#     print("\n" + "="*70 + "\n")


# def save_recommendations(recommendations, user_id, query):
#     """Save recommendations to CSV"""
#     filename = f"recommendations_{user_id}_{query.replace(' ', '_')[:20]}.csv"
#     recommendations.to_csv(filename, index=False)
#     print(f"💾 Recommendations saved to: {filename}")


# def main():
#     """Main function"""
#     print("\n🚀 Loading trained models...")
    
#     try:
#         # Initialize system
#         rec_system = AdvancedRecommendationSystem()
        
#         # Load DATA and PRE-TRAINED MODELS
#         rec_system.data_loader.load_data(
#             'coursera_courses.csv',
#             'augmented_learner_data.csv'
#         )
        
#         rec_system.load_models()  # Load pre-trained models
        
#         print("✅ Models loaded successfully!\n")
        
#     except FileNotFoundError as e:
#         print("\n❌ ERROR: Model files not found!")
#         print("Please train the models first by running:")
#         print("   python recommender.py")
#         print("\nOr make sure these files exist in the 'models/' folder:")
#         print("   - content_model.pkl")
#         print("   - svd_model.pkl")
#         print("   - ncf_model.pt")
#         return
#     except Exception as e:
#         print(f"\n❌ ERROR loading models: {str(e)}")
#         return
    
#     # Interactive loop
#     while True:
#         try:
#             # Get user inputs
#             user_id, query, difficulty, top_k = get_user_input()
            
#             # Get recommendations
#             print("\n🔍 Finding the best courses for you...\n")
#             recommendations = rec_system.recommend(
#                 user_id=user_id,
#                 query=query,
#                 difficulty=difficulty,
#                 top_k=top_k
#             )
            
#             # Display results
#             display_recommendations(recommendations)
            
#             # Ask to save
#             save_choice = input("💾 Save recommendations to CSV? (y/n): ").strip().lower()
#             if save_choice == 'y':
#                 save_recommendations(recommendations, user_id, query)
            
#             # Ask to continue
#             print("\n" + "-"*70)
#             continue_choice = input("\n🔄 Get more recommendations? (y/n): ").strip().lower()
#             if continue_choice != 'y':
#                 print("\n👋 Thank you for using AI Course Recommender!")
#                 print("="*70 + "\n")
#                 break
                
#         except KeyboardInterrupt:
#             print("\n\n👋 Goodbye!")
#             break
#         except Exception as e:
#             print(f"\n❌ Error: {str(e)}")
#             continue_choice = input("\n🔄 Try again? (y/n): ").strip().lower()
#             if continue_choice != 'y':
#                 break


# if __name__ == "__main__":
#     main()