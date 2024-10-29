# chromadb: This is the vector database where embeddings will be stored.
# pandas (pd): Used for reading the CSV file containing clinical trial data.
# google.generativeai: The API is used to generate embeddings for the content via Google’s Gemini API.
import concurrent
import time
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

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

# Load the CSV data
df = pd.read_csv("clinical_trials_Georgia_data_complete.csv")

# Extract only the 'conditions' column
conditions = df['Criterion'].tolist()
total_conditions = len(conditions)
# Store other relevant metadata (e.g., event number, study details) in a dictionary
metadata = df.to_dict(orient='records')  # Store entire row as metadata for each condition

# pair them up initially, so they should have the correct metadata and conditions matched up
condition_metadata_pairs = [{'condition': condition, 'metadata': metadata[i]} for i, condition in enumerate(conditions)]

def split_condition(condition, max_payload_size):
    chunks = []
    current_chunk = ""

    words = condition.split()  # Split the condition by spaces to get individual words

    for word in words:
        word_size = len((word + ' ').encode('utf-8'))  # Calculate size of word + space in bytes
        current_chunk_size = len(current_chunk.encode('utf-8')) # the length of our current chunk size

        # If adding the word exceeds the max payload size, store the current chunk and start a new one
        if current_chunk_size + word_size > max_payload_size:
            chunks.append(current_chunk.strip())  # Add the current chunk to the chunks list
            current_chunk = word + ' '  # Start a new chunk with the current word
        else:
            current_chunk += word + ' '  # Add the word to the current chunk

    # Add the last chunk if any words remain
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def validate_embedding_metadata_alignment(embeddings_with_metadata, metadata):
    if len(embeddings_with_metadata) != len(metadata):
        print(f"Warning: The number of embeddings ({len(embeddings_with_metadata)}) does not match metadata ({len(metadata)})")
        return False
    for i in range(len(embeddings_with_metadata)):
        embedding_metadata = embeddings_with_metadata[i]
        if embedding_metadata['metadata'] != metadata[i]:
	        # may have to check why its disagreeing
            print(f"Mismatch found at index {i}: Embedding metadata and stored metadata are not aligned.")
            return False
    print("All embeddings are correctly aligned with their metadata.")
    return True

def summarize_condition(condition, model="models/summarization-001"):
    try:
        summary = genai.summarize(
            model=model,
            content=condition
        )['summary']
        return summary
    except Exception as e:
        print(f"Error while summarizing: {e}")
        return condition  # Fallback to original condition if summarization fails

def preprocess_condition(condition):
    stop_words = set(stopwords.words('english'))
    condition = condition.lower()  # Lowercase
    condition = re.sub(r'\s+', ' ', condition)  # Remove extra spaces
    words = condition.split()  # Tokenize
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    preprocessed_condition = ' '.join(filtered_words)
    return preprocessed_condition

def preprocess_query(query):
    return preprocess_condition(query)  # Use the same

def split_and_batch_conditions_by_study(conditions, metadata, max_payload_size=9000):
	batched_conditions = []
	batched_metadata = []

	for i, condition in enumerate(conditions):
		study_metadata = metadata[i]  # Retrieve metadata for this condition
		condition_size = len(str(condition).encode('utf-8'))

		if condition_size > max_payload_size:
			print(f"Condition {i} is larger than the max payload size, splitting it.")
			# Split the condition into smaller chunks
			condition_chunks = split_condition(condition, max_payload_size)
			for chunk in condition_chunks:
				batched_conditions.append(chunk)
				batched_metadata.append(study_metadata)  # Add metadata for each chunk
		else:
			# Add the whole condition if it's within the size limit
			batched_conditions.append(condition)
			batched_metadata.append(study_metadata)

	return batched_conditions, batched_metadata

# Remove or replace NaN values in the conditions list
conditions = [str(condition) if pd.notnull(condition) else "" for condition in conditions]

# Use the split_and_batch_conditions_by_study function to split any conditions larger than the payload size
conditions, metadata = split_and_batch_conditions_by_study(conditions, metadata, max_payload_size=9000)

# Convert our rows of data into JSON format and store it in a list variable, documents
documents = df.apply(lambda row: row.to_json(), axis=1).tolist()


# We then create batches of documents and send it into the Gemini API for embedding
# We have a specific model we are using and task for the model

# Embed a single batch, also known as finding the numeric value
def embed_batch(batch):
	total_size = sum(len(str(condition).encode('utf-8')) for condition in batch)  # Log the batch size
	print(f"Embedding batch of size: {total_size} bytes with {len(batch)} conditions.")

	embeddings = [genai.embed_content(
		model='models/embedding-001',
		content=condition,
		task_type="retrieval_document"
	)['embedding'] for condition in batch]

	print(f"Embeddings for the batch: {embeddings}")  # Print the embeddings for debugging
	return embeddings


# Function to embed a single condition and preserve its metadata
def embed_condition(condition):
    condition_size = len(str(condition).encode('utf-8'))
    condition_size = len(str(condition).encode('utf-8'))
    print(f"Embedding condition of size: {condition_size} bytes.")

    embedding = genai.embed_content(
        model='models/embedding-001',
        content=condition,
        task_type="retrieval_document"
    )['embedding']

    print(f"Embedding for the condition: {embedding}")  # Debugging
    return embedding


# Embed conditions while handling payload size and metadata preservation
def embed_conditions_with_metadata(conditions, max_payload_size=9000, delay=0):
    embeddings_with_metadata = []
    counter = 0

    for condition in conditions:
        counter += 1
        print(f"We are on condition: {counter} and there are {len(conditions) - counter} left. Good luck!")

        # Ensure the condition isn't empty
        if not condition.strip():
            print(f"Skipping empty condition: {condition}")
            continue

        condition_size = len(str(condition).encode('utf-8'))

        # If the condition size exceeds the max payload, handle it (for large text content)
        if condition_size > max_payload_size:
            print(f"Condition exceeds max payload size of {max_payload_size} bytes. Skipping.")
            continue

        # Embed the individual condition
        embedding = embed_condition(condition)

        # Add metadata to the embeddings
        embeddings_with_metadata.append({
            'condition': condition,
            'embedding': embedding,
            # You can add additional metadata here if needed
        })

        time.sleep(delay)  # Add delay between embedding requests

    return embeddings_with_metadata


# Helper function for embedding and adding metadata
def process_condition(condition, max_payload_size=9000, delay=0):
	if not condition.strip():
		print(f"Skipping empty condition: {condition}")
		return None

	condition_size = len(str(condition).encode('utf-8'))

	if condition_size > max_payload_size:
		print(f"Condition exceeds max payload size of {max_payload_size} bytes. Skipping.")
		return None

	# Embed the individual condition
	embedding = embed_condition(condition)

	# Add delay if needed
	if delay > 0:
		time.sleep(delay)

	return {
		'condition': condition,
		'embedding': embedding
	}


# Main function to parallelize
def embed_conditions_with_metadata_parallel(conditions, max_payload_size=9000, delay=0.05, max_workers=5):
	embeddings_with_metadata = []
	counter = 0

	with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
		# Use a list comprehension to map conditions to parallel tasks
		futures = [executor.submit(process_condition, condition, max_payload_size, delay) for condition in conditions]

		for future in concurrent.futures.as_completed(futures):
			result = future.result()
			if result:
				counter += 1
				print(f"Processed condition {counter}/{len(conditions)}")
				embeddings_with_metadata.append(result)

	return embeddings_with_metadata

# We wrap the embed_content method for use while embedding documents to
# Define the embedding function externally for gemeni to use
# Embedding is a process where documents (textual data) are transformed into numerical vectors (a series of numbers)
# that represent the semantic meaning of the text
class GeminiEmbeddingFunction:
	def __call__(self, input):
		model = 'models/text-embedding-004'
		return genai.embed_content(
			model=model,
			content=input,
			task_type="retrieval_document"
		)['embedding']


# The Chroma DB is a vector database that allows you to store and retrieve high-dimensional vectors (like embeddings)
# efficiently. It's used for tasks such as semantic search or similarity matching.
def create_chroma_db(conditions, metadata, name):
	# Initialize the Chroma client with persistent storage
	chroma_client = chromadb.Client(
		chromadb.config.Settings(
			persist_directory="./chroma_storage"  # Specify the folder where you want to store the DB
		)
	)

	# Create a new collection for the embeddings
	db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

	# Embed conditions and store metadata along with embeddings
	condition_embeddings_with_metadata = embed_conditions_with_metadata_parallel(conditions)

	# Ensure embeddings are present
	if not condition_embeddings_with_metadata:
		print("No valid embeddings found. Skipping database creation.")
		return None

	print(f"Adding {len(condition_embeddings_with_metadata)} embeddings to the database.")

	# Process each condition one by one (no batching)
	for i in range(len(condition_embeddings_with_metadata)):
		condition_data = condition_embeddings_with_metadata[i]

		# Extract the condition, embedding, and metadata from the dictionary
		condition = condition_data['condition']
		embedding = condition_data['embedding']
		condition_metadata = metadata[i]  # Assuming metadata aligns with the condition index

		# Add the embedding and associated metadata to the database
		db.add(
			documents=[condition],  # The condition itself
			embeddings=[embedding],  # Corresponding embedding
			metadatas=[condition_metadata],  # Store additional metadata
			ids=[str(i)]  # Unique ID for each condition
		)

	return db

def load_and_query_chroma_db(query, db_name="clinical_trials_db", persist_directory="./chroma_storage", n_results=5):
	"""
		Load an existing ChromaDB and perform a query to find matching conditions.

		Parameters:
		- query (str): The search term to query the database.
		- db_name (str): The name of the ChromaDB collection.
		- persist_directory (str): The folder where the ChromaDB is stored.
		- n_results (int): The number of results to return (default is 5).

		Returns:
		- List of matching conditions.
		"""
	# Initialize the Chroma client with persistent storage
	chroma_client = chromadb.Client(
		chromadb.config.Settings(
			persist_directory=persist_directory  # Load from the storage directory
		)
	)

	# Load the collection by name
	try:
		db = chroma_client.get_collection(name=db_name)
	except KeyError:
		print(f"Collection '{db_name}' not found in the database.")
		return []

	# Perform a query on the loaded database
	try:
		results = db.query(query_texts=[query], n_results=n_results)
		matching_conditions = results['documents']
	except Exception as e:
		print(f"Error while querying the database: {e}")
		return []

	# Return the matching conditions
	return matching_conditions

def load_existing_chroma_db(name, persist_directory="./chroma_storage"):
	chroma_client = chromadb.Client(
		chromadb.config.Settings(
			persist_directory=persist_directory  # Specify the folder where you want to store the DB
		)
	)

	try:
		# Check if the collection already exists, load it
		db = chroma_client.get_collection(name=name)
		print(f"Returning DB")
		return db
	except KeyError:
		print("Didn't find any!")
		return None

def add_to_existing_chroma_db(conditions, metadata, name, persist_directory="./chroma_storage"):
    # Initialize the Chroma client with persistent storage
    chroma_client = chromadb.Client(
        chromadb.config.Settings(
            persist_directory=persist_directory  # Specify the folder where you want to store the DB
        )
    )

    try:
        # Check if the collection already exists, load it
        db = chroma_client.get_collection(name=name)
        print(f"Collection '{name}' found. Adding to existing database.")
    except KeyError:
        # If the collection doesn't exist, create a new one
        print(f"Collection '{name}' not found. Creating a new collection.")
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    # Embed conditions and store metadata along with embeddings
    condition_embeddings_with_metadata = embed_conditions_with_metadata_parallel(conditions)

    # Ensure embeddings are present
    if not condition_embeddings_with_metadata:
        print("No valid embeddings found. Skipping database creation.")
        return None

    print(f"Adding {len(condition_embeddings_with_metadata)} embeddings to the database.")

    # Add each condition and its metadata
    for i in range(len(condition_embeddings_with_metadata)):
        condition_data = condition_embeddings_with_metadata[i]
        condition = condition_data['condition']
        embedding = condition_data['embedding']
        condition_metadata = metadata[i]

        # Add to the database
        db.add(
            documents=[condition],
            embeddings=[embedding],
            metadatas=[condition_metadata],
            ids=[str(i)]  # Ensure unique ID generation (modify as needed)
        )

    return db


def rank_results(results, keywords):
    ranked_results = []
    for result in results:
        score = sum([result.lower().count(keyword.lower()) for keyword in keywords])  # Simple keyword frequency count
        ranked_results.append((result, score))
    # Sort by score in descending order
    ranked_results = sorted(ranked_results, key=lambda x: x[1], reverse=True)
    # Return only the results, not the scores
    return [result[0] for result in ranked_results]


def query_with_list(my_chroma_db, list_of_queries):
	for current_query in list_of_queries:
		# Load the database and run the query
		results = load_and_query_chroma_db(current_query, db_name=my_chroma_db, n_results=5)

		if not results:
			print(f"No results found for query: {current_query}")
			return

		# Output results for manual inspection
		print(f"Results for query '{current_query}':")
		for result in results:
			print(f"Condition: {result['documents']}, Metadata: {result['metadatas']}")
		print("Next!")

if __name__ == "__main__":
	# Save timestamp
	start = time.time()

	# Initialize the DB with conditions and their metadata
	db = create_chroma_db(conditions, metadata, "clinical_trials_db")



	#db = load_existing_chroma_db("clinical_trials_db")



	# Save timestamp
	end = time.time()

	print(f"\nThis takes this long: {end - start}")

	# Query for conditions related to cancer
	# queries = [ "Apoptosis in cancer cells" , "Cell necrosis in liver disease" , "Therapies targeting cell death" ,
	#             "Mitochondrial dysfunction in neurodegeneration", "Regenerative medicine for dying cells",
	#             "Clinical trials on programmed cell death inhibitors"]

	queries = ["adult patients with Type 2 Diabetes that involve cardiovascular interventions",
	           " cancer treatments that include supportive care as an intervention",
	           " observational studies conducted in European countries, such as Belgium or Italy",
	           "trials studying infectious diseases such as influenza, with antiviral interventions",
	           "terminated studies involving drug interventions for heart failure."]

	for query in queries:
		relevant_conditions = db.query(query_texts=[query], n_results=3)['documents'][0]
		relevant_conditions_2 = db.query(query_texts=[query], n_results=3)['documents'][0]
		# Print the results
		print(f"For query: {query}")

		print(len(relevant_conditions))

		for condition in relevant_conditions:
			print(condition)
			print("\n")



	# Print the results

# study_details = condition['metadata']  # Get associated metadata for the condition
# print(f"Condition: {condition['documents']}, Study Details: {study_details}")
