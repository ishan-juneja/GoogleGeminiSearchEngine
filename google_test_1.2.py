# chromadb: This is the vector database where embeddings will be stored.
# pandas (pd): Used for reading the CSV file containing clinical trial data.
# google.generativeai: The API is used to generate embeddings for the content via Google’s Gemini API.
import concurrent
import time

# Next times actionables:
# we are running with a dual list setup which relies on the order of our materials
# we need to store our chroma db properly
# print our found functions properly
# fix comments and functions documentation
# we need to also make some proper testing

import chromadb
import pandas as pd
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure the Gemini API
api_key = "insert"
genai.configure(api_key=api_key)
import time
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity


MAX_QUERY_LENGTH = 2000  # Limit the max token size for a query
MAX_API_CALLS = 60  # Limit the number of API calls to avoid hitting rate limits


# Embedding function
def get_embedding(text):
	"""Generate an embedding for a given text using the Gemini API."""

	embedding = [genai.embed_content(
		model='models/embedding-004',
		content=text,
		task_type="retrieval_document"
	)['embedding']]

	return embedding


# Step 1: Load and preprocess clinical trials data
def load_trials(csv_file):
	"""Load clinical trials CSV data."""
	df = pd.read_csv(csv_file)
	return df[['NCT ID', 'Conditions', 'Criterion', 'Study Link']].dropna()  # Assuming 'Link' is a column in your CSV


# Step 2: Create embeddings for all clinical trials based on inclusion/exclusion criteria
def create_embeddings(trial_text):
	"""Generate an embedding for a given clinical trial text."""
	# try:
	print("3 Made it here 0")
	return get_embedding(trial_text)


# except openai.error.OpenAIError as e:
# 	print(f"Error generating embedding: {e}")
# 	return None


def generate_trial_embeddings(trials_df):
	"""Generate embeddings for all trials in parallel."""
	with ThreadPoolExecutor(
			max_workers=8) as executor:  # change max workers to edit the number in the parallel embeddings
		print("2 Made it here 0")
		trial_texts = trials_df['Conditions'].tolist()
		print("2 Made it here 1")
		trial_embeddings = list(executor.map(create_embeddings, trial_texts))
		print("2 Made it here 2")

	print("2 Made it here 3")
	# Add the embeddings back to the DataFrame
	trials_df['Embedding'] = trial_embeddings
	print("2 Made it here 4")
	return trials_df.dropna(subset=['Embedding'])


# Step 3: Handle user queries
def process_query(query):
	"""Process a user query to ensure it's within OpenAI token limits."""
	if len(query) > MAX_QUERY_LENGTH:
		raise ValueError(f"Query too long: {len(query)} characters. Please shorten your query.")

	# Generate embedding for the query
	return create_embeddings(query)


def find_relevant_trials(query_embedding, trials_df):
	"""Find the most relevant clinical trials based on cosine similarity."""
	trial_embeddings = np.vstack(trials_df['Embedding'].values)
	similarities = cosine_similarity([query_embedding], trial_embeddings)

	trials_df['Similarity'] = similarities[0]

	# Return top 3 most similar trials with NCT ID and Link
	return trials_df.nlargest(3, 'Similarity')[['NCT ID', 'Link', 'Similarity']]


# Step 4: Main program
def main(csv_file, user_query):
	# Load the clinical trials dataset
	print("Made it here 0")
	trials_df = load_trials(csv_file)
	print("Made it here 1")
	# Generate embeddings for all trials
	trials_df = generate_trial_embeddings(trials_df)
	print("Made it here 2")
	# Process user query
	try:
		query_embedding = process_query(user_query)
	except ValueError as e:
		return str(e)

	# Find and return the top 3 most relevant trials
	relevant_trials = find_relevant_trials(query_embedding, trials_df)
	return relevant_trials


# Edge cases: Too many API calls
def manage_api_rate_limit(calls_made, max_calls=MAX_API_CALLS):
	if calls_made >= max_calls:
		raise RuntimeError("API call limit reached. Please try again later.")


if __name__ == "__main__":
	# Example CSV and query
	csv_file_path = "clinical_trials_Georgia_data_complete.csv"
	user_query = "What studies have inclusion and exclusion criteria related to cancer in older adults?"

	# Run the program
	try:
		results = main(csv_file_path, user_query)
		print("hello")
		print(results)
	except Exception as e:
		print(f"Error: {e}")
