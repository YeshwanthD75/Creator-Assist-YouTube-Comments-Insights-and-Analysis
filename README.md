# Creator Assist: AI-Powered YouTube Comment Insights and Analysis

## **Overview**
The **Creator Assist** is an AI-powered web application designed to provide content creators with in-depth insights into YouTube video comments. By leveraging advanced natural language processing (NLP) techniques and machine learning models, the project categorizes sentiments, identifies toxic content, and offers valuable engagement metrics. Additionally, users can query the data using an intelligent Large Language Model (LLM), generating tailored insights from the comments.

This project is particularly useful for content creators, businesses, and marketers to:
- Analyze audience sentiments.
- Manage comment toxicity.
- Gauge content impact.
- Improve engagement and decision-making based on audience feedback.

**Note:** Due to YouTube API constraints, the application processes up to **1,000 comments** per video.

In simple, this project combines sentiment analysis, clustering, and engagement metrics to deliver powerful insights, enabling content creators and businesses to better understand their audience and improve their strategies.

---

## **Features**

### 1. **YouTube Comment Extraction**
- Integrates with the YouTube Data API to fetch comments from public videos.
- Retrieves video metadata such as title, description, views, likes, and comment count.

### 2. **Sentiment Analysis**
- **TextBlob**: Performs baseline sentiment scoring, categorizing comments as positive, negative, or neutral.
- **Sentence Transformers**: Provides deep contextual analysis of comments, ensuring accurate sentiment classification based on the semantic meaning of text.

### 3. **Engagement Metrics**
- Measures sentiment trends (positive vs. negative comments).
- Evaluates engagement quality based on the frequency and type of sentiments expressed.
- Offers insights into how the audience perceives the content, helping creators identify engaging topics and address dissatisfaction.

### 4. **Querying with LLM**
- Enables users to query specific insights via a Large Language Model (LLM).
- Example queries include:
  - “What are the most common sentiments in the comments?”
  - “Which comments received the most engagement?”
  - “Are there specific suggestions or concerns mentioned by users?”

### 5. **Toxic Comment Identification**
- Identifies and filters out toxic, spammy, or offensive comments using sentiment analysis and text classification.
- Helps maintain a positive environment in the comment section.

### 6. **Cosine Similarity for Comment Clustering**
- **Cosine Similarity (scikit-learn)**: Calculates semantic similarity between comments by measuring the cosine angle between vector representations.
- Groups similar comments, enabling the detection of recurring themes, trends, or key feedback points.

---

## **Tools, Libraries, and Languages Used**

### **Backend and NLP**
- **Python**: Core language for data processing and backend operations.
- **Flask**: Lightweight web framework to build and serve the application.
- **TextBlob**: Performs baseline sentiment analysis.
- **Sentence Transformers**: Provides deep contextual sentiment understanding.
- **Cosine Similarity (scikit-learn)**: Measures similarity between comments for clustering.

### **Frontend**
- **HTML5 and CSS3**: Builds a responsive and user-friendly interface.
- **JavaScript**: Adds interactivity and dynamic functionality.

### **APIs**
- **YouTube Data API**: Fetches YouTube comments and video metadata.
- **Gemini API**: Powers the LLM for answering user queries based on sentiment analysis and clustering results.

### **Environment Management**
- **Miniconda**: Manages dependencies and virtual environments.
- **Jupyter Notebook**: Facilitates model development and experimentation.

---

## **Detailed Workflow**

1. **Data Fetching**
   - Retrieves up to **1,000 comments** and metadata from YouTube using the YouTube API.

2. **Preprocessing**
   - Cleans comment text by removing special characters, URLs, and irrelevant content.

3. **Sentiment Analysis**
   - Applies **TextBlob** for basic sentiment categorization.
   - Utilizes **Sentence Transformers** for context-aware sentiment analysis.

4. **Cosine Similarity and Clustering**
   - Converts comments into vector representations.
   - Groups similar comments based on their semantic meaning using Cosine Similarity.

5. **Engagement Metrics**
   - Analyzes sentiment distribution and trends.
   - Provides insights into how well the content resonates with the audience.

6. **Query System**
   - Allows users to interact with an LLM to extract actionable insights from the comments.

---

## **Environment Variables**
The project requires a `.env` file to securely store API keys:
- `YOUTUBE_API_KEY`: Key for accessing the YouTube Data API.
- `GEMINI_API_KEY`: Key for accessing the Gemini-powered LLM API.

Ensure these keys are correctly set in the `.env` file for smooth application functionality.

---

## **Screenshots**
### 1. Homepage
![Homepage](path_to_homepage_screenshot.png)

---

## Contributing
Contributions are welcome! Please create a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
