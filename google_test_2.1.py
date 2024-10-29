import pandas as pd
import google.generativeai as genai
# Configure the Gemini API
api_key = "insert"
genai.configure(api_key=api_key)
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import os, ast # Import the abstract syntax tree module to safely evaluate list from string
MAX_QUERY_LENGTH = 2000  # Limit the max token size for a query
MAX_API_CALLS = 60  # Limit the number of API calls to avoid hitting rate limits
MODEL = "text-embedding-004" # previous one was embedding-001
NUMBER_OF_STUDIES_RELATED = 3 # How many studies to return

# # Embedding function
# def get_embedding(text):
# 	"""Generate an embedding for a given text using the Gemini API."""
#
# 	embedding = [genai.embed_content(
# 		model='models/embedding-003',
# 		content=text,
# 		task_type="retrieval_document"
# 	)['embedding']]
# 	return embedding


def get_embedding(condition):
	"""Generate an embedding for a given text using the Gemini API."""
	embedding = genai.embed_content(
		model=f'models/{MODEL}',
		content=condition,
		task_type="retrieval_document"
	)['embedding']

	print(f"Embedding for the condition: {embedding}")  # Debugging
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

# Function to generate embeddings and save/load from file
def generate_trial_embeddings(trials_df, embeddings_file='embeddings.csv'):
	"""Generate embeddings for all trials, only for new entries, and store them."""

	# Check if embeddings file already exists
	if os.path.exists(embeddings_file):
		# If the file exists, load it
		existing_embeddings_df = pd.read_csv(embeddings_file)

		# Convert 'Embedding' column back to a list of floats
		existing_embeddings_df['Embedding'] = existing_embeddings_df['Embedding'].apply(
			lambda x: np.array(ast.literal_eval(x)))

		# Merge the existing embeddings with the trials dataframe
		merged_df = pd.merge(trials_df, existing_embeddings_df, on='NCT ID', how='left')
		# Filter out trials that already have embeddings
		new_trials_df = merged_df[merged_df['Embedding'].isna()]
	else:
		# If the file doesn't exist, all trials are considered new
		new_trials_df = trials_df.copy()

	if not new_trials_df.empty:
		# Generate embeddings for new trials only
		with ThreadPoolExecutor(max_workers=8) as executor:
			trial_texts = new_trials_df['Conditions'].tolist()
			new_trial_embeddings = list(executor.map(create_embeddings, trial_texts))

		# Store new embeddings as string representation of lists
		new_trials_df['Embedding'] = new_trial_embeddings

		# Append new embeddings to file (ensure Embedding column is saved as string)
		new_trials_df[['NCT ID', 'Embedding']].to_csv(embeddings_file, mode='a',
		                                              header=not os.path.exists(embeddings_file), index=False)
	else:
		print("No new trials found, using existing embeddings.")

	# Load the updated embeddings for all trials (existing + new)
	embeddings_df = pd.read_csv(embeddings_file)

	# Convert 'Embedding' column back to a list of floats for processing
	embeddings_df['Embedding'] = embeddings_df['Embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

	trials_df = pd.merge(trials_df, embeddings_df, on='NCT ID', how='left')

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
	return trials_df.nlargest(NUMBER_OF_STUDIES_RELATED, 'Similarity')[['NCT ID', 'Study Link', 'Similarity']]


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
	user_query = "What studies have inclusion and exclusion criteria relate to diabetes and cardio vascular health?"

	# Run the program
	try:
		results = main(csv_file_path, user_query)
		print("hello")
		print(results)
	except Exception as e:
		print(f"Error: {e}")
