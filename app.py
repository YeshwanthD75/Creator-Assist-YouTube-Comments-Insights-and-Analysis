import os
import re
from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from dotenv import load_dotenv
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
import markdown
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llm import LLM  # Assuming your LLM class is properly set up

# Load API keys from .env file
load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Initialize the SentenceTransformer model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model

# Helper classes

class YouTubeCommentFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def get_comments(self, video_id, max_results=1000):
        comments = []
        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100  # Fetch 100 comments per request
            )
            response = request.execute()
            
            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            while 'nextPageToken' in response and len(comments) < max_results:
                next_page_token = response['nextPageToken']
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                    comments.append(comment)

        except Exception as e:
            print(f"An error occurred: {e}")

        return comments[:max_results]


class CommentCleaner:
    @staticmethod
    def clean_comment(comment):
        clean_text = re.sub(r'<.*?>', '', comment)  # Remove HTML tags
        clean_text = re.sub(r'[^\x00-\x7F]+', ' ', clean_text)  # Remove non-ASCII characters
        return clean_text


class SentimentAnalyzer:
    @staticmethod
    def analyze_sentiment(comment):
        blob = TextBlob(comment)
        return blob.sentiment.polarity  # Sentiment value between -1 (negative) and 1 (positive)

    @staticmethod
    def summarize_sentiments(comments):
        positive, negative, neutral = 0, 0, 0
        total_sentiment = 0

        for comment in comments:
            cleaned_comment = CommentCleaner.clean_comment(comment)
            sentiment_score = SentimentAnalyzer.analyze_sentiment(cleaned_comment)

            if sentiment_score > 0:
                positive += 1
            elif sentiment_score < 0:
                negative += 1
            else:
                neutral += 1

            total_sentiment += sentiment_score

        average_sentiment = total_sentiment / len(comments) if comments else 0

        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "average_sentiment": average_sentiment
        }


class InsightGenerator:
    def __init__(self, llm_class):
        self.llm_class = llm_class

    def generate_insight(self, comments, similarity_threshold=0.85):
        cleaned_comments = [CommentCleaner.clean_comment(comment) for comment in comments]
        
        # Encode the comments to get embeddings
        embeddings = model.encode(cleaned_comments)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Set a threshold to remove similar comments
        unique_comments = []
        for i in range(len(comments)):
            if all(similarity_matrix[i][j] < similarity_threshold for j in range(i)):
                unique_comments.append(comments[i])

        removed_count = len(comments) - len(unique_comments)
        print(f"Removed {removed_count} similar comments based on cosine similarity.")

        all_comments_text = " ".join(unique_comments)

        prompt = (
            f"Here are the comments from a YouTube video:\n"
            f"{all_comments_text}\n"
            f"Can you provide insights on how the audience feels about the video? "
            f"Give detailed suggestions to improve, based on these comments."
        )

        llm_instance = self.llm_class()
        insight = llm_instance.model(prompt)
        insight = markdown.markdown(insight)

        return insight


# Flask App
app = Flask(__name__)

youtube_fetcher = YouTubeCommentFetcher(YOUTUBE_API_KEY)
sentiment_analyzer = SentimentAnalyzer()
insight_generator = InsightGenerator(LLM)

comments = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_comments', methods=['POST'])
def fetch_comments():
    video_url = request.form.get('video_url')
    max_comments = int(request.form.get('max_comments', 1000))
    similarity_threshold = float(request.form.get('similarity_threshold', 0.85))  # Get the similarity threshold

    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid URL. Couldn't extract video ID."})

    global comments
    comments = youtube_fetcher.get_comments(video_id, max_results=max_comments)
    sentiment_summary = sentiment_analyzer.summarize_sentiments(comments)

    # Generate insights with the cosine similarity threshold
    insight = insight_generator.generate_insight(comments, similarity_threshold=similarity_threshold)

    # Send the sentiment summary and insight as response
    return jsonify({
        "sentiment_summary": sentiment_summary,
        "insight": insight
    })

@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form.get('user_message').strip()
    if user_message == "":
        return jsonify({"response": "Please enter a message."})

    # Only answer the user's question based on the comments
    all_comments_text = "\n".join(comments)  # Join all comments into a single text block
    prompt = (
        f"Here are the comments from a YouTube video:\n"
        f"{all_comments_text}\n\n"
        f"User question: {user_message}\n"
        "Based on the above comments, please provide a detailed response to the user's question.\n"
        "Format the response with proper spacing, and use **bold** for key points and *italics* for emphasis."
    )

    llm_instance = insight_generator.llm_class()
    response = llm_instance.model(prompt)

    # Now, format the response based on what we learned from the example:
    formatted_response = markdown.markdown(response)
    
    return jsonify({"response": formatted_response})


def extract_video_id(url):
    regex = r"^(?:https?\:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/]+\/[^\/]+\/|(?:v|e(?:mbed)?)\/|(?:watch\?v=|(?:e(?:mbed)?\/)))([a-zA-Z0-9_-]{11}))"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


if __name__ == "__main__":
    app.run(debug=True)
