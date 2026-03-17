import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from colorama import init , Fore
import time
import sys

init(autoreset=True)

def load_data(file_path='imdb_top_1000.csv'):
    try:
        df = pd.read_csv(file_path)
        df['combined_features'] = df['Genre'].fillna('Nan') + '' + df['Overview'].fillna('Nan')
        return df 
    except FileNotFoundError:
        print(Fore.RED + f" It seems like you have Ran into an Error : The File path {file_path} was not found ")
        exit()

print("Loading data.......")
movies_df = load_data()


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def list_genres(df):
    return sorted(set(genre.strip() for sublist in df['Genre'].dropna().str.split(',') for genre in sublist))

genres = list_genres(movies_df)

def recommend_movie(genre=None, Mood=None, rating=None, top_n=5):
    filtered_df = movies_df
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    if rating:
        filtered_df = filtered_df[filtered_df['IMDB_rating'] >= rating]

    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)  # randomized the order

    recommendations = []
    for idx, row in filtered_df.iterrows():
        overview = row["Overview"]
        if pd.isna(overview):
            continue
        polarity = TextBlob(overview).sentiment.polarity
        if (Mood and ((TextBlob(Mood).sentiment.polarity < 0 and polarity > 0) or polarity >= 0)) or not Mood:
            recommendations.append((row["Series_Title"], polarity))
        if len(recommendations) == top_n:
            break

    return recommendations if recommendations else "No suitable Recommendations found"

def display_recommendations(recs, Name):
    print(Fore.YELLOW + f"\n AI Analyzed Movie Recommendations for {Name}:")  
    for idx, (title, polarity) in enumerate(recs, 1):
        sentiment = "Positive " if polarity > 0 else "Negative " if polarity < 0 else "Neutral"
        print(f"{Fore.CYAN}{idx}. {title} (polarity : {polarity:.2f}, {sentiment})")

def proccesing_anim():
    for _ in range(3):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(0.5)

def handle_ai(name):
    print(Fore.BLUE + "\n Let's find the perfect movie for you ")

    print(Fore.GREEN + "Available Genres: ", end="")
   
    for idx, g in enumerate(genres, 1):
        print(f"{Fore.CYAN}{idx}. {g}")

    while True:
        genre_input = input(Fore.YELLOW + "Enter Genre name or number: ").strip()
        if genre_input.isdigit() and 1 <= int(genre_input) <= len(genres):
            genre = genres[int(genre_input) - 1]
            break
        elif genre_input.title() in genres:
            genre = genre_input.title()
            break
        print(Fore.RED + "Invalid input please try again ")

    mood = input(Fore.YELLOW + "How do you feel today (Describe your mood): ").strip()

    print(Fore.BLUE + "\n Analyzing your mood", end="", flush=True)
    proccesing_anim()
    polarity = TextBlob(mood).sentiment.polarity
    moon_desc = "Positive :D" if polarity > 0 else "Negative" if polarity < 0 else "Neutral :|"
    print(f"\n {Fore.GREEN}Your mood is {moon_desc} (polarity = {polarity:.2f})\n")

    while True:
        rating_input = input(Fore.YELLOW + "Enter min IMDB Rating 7.6 - 9.3 or 'skip': ").strip()
        if rating_input.lower() == 'skip':
            rating = None
            break
        try:
            rating_input = float(rating_input)
            if 7.6 <= rating_input <= 9.3:
                rating = rating_input
                break
            print(Fore.RED + "Rating out of range..\n")
        except ValueError:
            print(Fore.RED + "Invalid ..")

    print(f"{Fore.BLUE}\n Finding movies for {name}", end="", flush=True)
    proccesing_anim()
    recs = recommend_movie(genre=genre, Mood=mood, rating=rating, top_n=5)

    if isinstance(recs, str):
        print(Fore.RED + recs + "\n")
    else:
        display_recommendations(recs, name)

    while True:
        action = input(Fore.YELLOW + "\n Would you like more movie recommendations? (yes/no): ").strip().lower()
        if action == 'no':
            print(Fore.GREEN + f"Enjoy your movie picks, {name}")
            break
        elif action == 'yes':
            recs = recommend_movie(genre=genre, Mood=mood, rating=rating, top_n=5)
            if isinstance(recs, str):
                print(Fore.RED + recs + "\n")
            else:
                display_recommendations(recs, name)
        else:
            print(Fore.RED + "Invalid Choice Try again \n")

def main(name):
    print(Fore.BLUE + "Welcome to your personal Movie thingy?")
    print(f"\n {Fore.GREEN} Great to meet you, {name}")
    handle_ai(name)

if __name__ == "__main__":
    name = input(Fore.YELLOW + "What's your Name? ").strip()
    main(name)