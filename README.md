# GoogleGeminiSearchEngine - Clinical Trials Relevance Finder

This Python program processes a dataset of clinical trials, creating embeddings for each trial's conditions and inclusion/exclusion criteria to match the most relevant studies to a user's query. Using the Gemini API, it generates vector embeddings for clinical trial data and a user query, then finds the most similar trials based on cosine similarity.

## Features

- Loads clinical trials from a CSV file and preprocesses the data.
- Uses the Gemini API to create embeddings for each trial's condition criteria.
- Stores embeddings in a file to avoid redundant API calls.
- Limits API calls to prevent rate limit issues.
- Processes user queries and finds relevant clinical trials based on cosine similarity.

## Requirements

- Python 3.7+
- Google Gemini API key
- Required packages:
  - `pandas`
  - `google-generativeai`
  - `numpy`
  - `concurrent.futures`
  - `scikit-learn`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/clinical-trials-relevance-finder.git
   cd clinical-trials-relevance-finder
2. Install the dependencies:
   pip install pandas google-generativeai numpy scikit-learn
3. Set up your Gemini API key: Replace the placeholder API key in the script:

api_key = "YOUR_GEMINI_API_KEY"

4. Prepare your clinical trials CSV file. Ensure it has columns named:

  NCT ID
  Conditions
  Criterion
  Study Link
  
5. Usage
Load Data and Generate Embeddings: Run the script with your CSV file to generate embeddings for the clinical trials.

Query and Find Relevant Studies: After embedding generation, you can input a user query to find the top 3 most relevant clinical trials based on cosine similarity.

## Coding Explanation
Embedding Generation: The function get_embedding creates an embedding vector for each clinical trial's conditions using the Gemini API.
Similarity Calculation: Cosine similarity is used to measure the relevance of clinical trials to the user's query.
API Rate Management: The manage_api_rate_limit function prevents exceeding the maximum number of API calls.
Troubleshooting
Rate Limit Error: The script includes a limit on API calls to avoid hitting the rate limit.
File Not Found Error: Ensure the CSV file exists in the directory or update the file path in the code.






