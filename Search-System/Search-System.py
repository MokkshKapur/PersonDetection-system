import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import re # Still useful for basic cleaning

# --- Configuration ---
CSV_FILE_PATH = 'image_descriptions.csv'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_CACHE_FILE = 'description_embeddings.pkl'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 1. Load Model ---
print(f"Loading SentenceTransformer model: {EMBEDDING_MODEL_NAME}...")
sentence_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
print("Model loaded.")

# --- 2. Load Data ---
print(f"Loading data from {CSV_FILE_PATH}...")
try:
    df = pd.read_csv(CSV_FILE_PATH)
    df['description'] = df['description'].astype(str).fillna('')
    filename_to_description = pd.Series(df.description.values, index=df.image_filename).to_dict()
    print(f"Loaded {len(df)} descriptions.")
except FileNotFoundError:
    print(f"Error: CSV file not found at {CSV_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 3. Generate or Load Embeddings ---
# ... (Keep the embedding loading/generation logic exactly the same) ...
embeddings = None
filenames = None
if os.path.exists(EMBEDDINGS_CACHE_FILE):
    print(f"Loading embeddings from cache: {EMBEDDINGS_CACHE_FILE}...")
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
            cache_data = pickle.load(f)
            if cache_data.get('model_name') == EMBEDDING_MODEL_NAME:
                embeddings = cache_data['embeddings']
                filenames = cache_data['filenames']
                print(f"Loaded {len(filenames)} embeddings from cache.")
            else:
                print("Cached embeddings were created with a different model. Regenerating...")
    except Exception as e:
        print(f"Error loading cache file: {e}. Regenerating embeddings...")

if embeddings is None or filenames is None:
    print("Generating text embeddings for descriptions...")
    descriptions = df['description'].tolist()
    embeddings = sentence_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=True)
    filenames = df['image_filename'].tolist()
    print(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    print(f"Saving embeddings to cache: {EMBEDDINGS_CACHE_FILE}...")
    cache_data = {
        'model_name': EMBEDDING_MODEL_NAME,
        'embeddings': embeddings,
        'filenames': filenames
    }
    try:
        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Embeddings saved successfully.")
    except Exception as e:
        print(f"Error saving embeddings: {e}")


# --- Helper Function for Basic Text Cleaning ---
def basic_clean(text):
    """Basic cleaning: remove extra whitespace and markdown."""
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate whitespace
    text = text.replace('**', '') # Remove bold markers
    # Add other simple cleaning if needed
    return text

# --- 5. Enhanced Search Function (Using Newline Splitting) ---
def search_descriptions_with_lines(query, model, embeddings_matrix, filenames_list, desc_map, top_n=5):
    """
    Searches descriptions, splits by lines, identifies best matching line segment.
    """
    if embeddings_matrix is None or filenames_list is None:
        print("Error: Embeddings not available for search.")
        return []

    print(f"\nSearching for: '{query}'")
    query_embedding = model.encode([query], convert_to_numpy=True)
    doc_similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
    actual_top_n = min(top_n, len(filenames_list))
    top_n_doc_indices = np.argsort(doc_similarities)[-actual_top_n:][::-1]

    results = []
    for idx in top_n_doc_indices:
        filename = filenames_list[idx]
        full_description = desc_map.get(filename, "")
        doc_score = doc_similarities[idx]

        best_matching_line = "(No relevant lines found)" # Default message
        best_line_score = 0.0

        if full_description:
            # Split the description into lines
            lines = full_description.splitlines()

            # Clean and filter lines
            valid_lines = [basic_clean(line) for line in lines if len(basic_clean(line)) > 5] # Clean then check length

            if valid_lines:
                try:
                    # Embed the valid lines of this specific document
                    line_embeddings = model.encode(valid_lines, convert_to_numpy=True, show_progress_bar=False)

                    # Calculate similarity between query and these line embeddings
                    line_sims = cosine_similarity(query_embedding, line_embeddings)[0]

                    # Find the best matching line within this document
                    best_line_index_in_doc = np.argmax(line_sims)
                    best_matching_line = valid_lines[best_line_index_in_doc] # Use the cleaned line
                    best_line_score = line_sims[best_line_index_in_doc]

                except Exception as emb_err:
                    print(f"Warning: Error embedding lines for {filename}: {emb_err}")
                    best_matching_line = "(Line embedding failed)"
            # else: No need for an else here, default message handles it

        # Append result
        results.append((filename, full_description, doc_score, best_matching_line, best_line_score))

    return results

# --- Example Usage ---
if __name__ == "__main__":
    if embeddings is not None and filenames is not None:
        # Example Search Queries
        search_query_1 = "person wearing a red dress"
        results_1 = search_descriptions_with_lines(
            search_query_1, sentence_model, embeddings, filenames, filename_to_description, top_n=3
        )

        print("\n--- Top Results for Query 1 ---")
        for filename, description, doc_score, line, line_score in results_1:
            print(f"Filename: {filename}")
            print(f"Overall Doc Score: {doc_score:.4f}")
            # Only show line score if matching was successful
            if line_score > 0.0:
                 print(f"Best Matching Line (Score: {line_score:.4f}): {line}")
            else:
                 print(f"Best Matching Line: {line}") # Shows the status/default message
            print("-" * 20)

        search_query_2 = "black tank on the sidewalk"
        results_2 = search_descriptions_with_lines(
            search_query_2, sentence_model, embeddings, filenames, filename_to_description, top_n=3
        )
        print("\n--- Top Results for Query 2 ---")
        for filename, description, doc_score, line, line_score in results_2:
            print(f"Filename: {filename}")
            print(f"Overall Doc Score: {doc_score:.4f}")
            if line_score > 0.0:
                 print(f"Best Matching Line (Score: {line_score:.4f}): {line}")
            else:
                 print(f"Best Matching Line: {line}")
            print("-" * 20)


        search_query_3 = "walking side by side"
        results_3 = search_descriptions_with_lines(
            search_query_3, sentence_model, embeddings, filenames, filename_to_description, top_n=3
        )
        print("\n--- Top Results for Query 3 ---")
        for filename, description, doc_score, line, line_score in results_3:
            print(f"Filename: {filename}")
            print(f"Overall Doc Score: {doc_score:.4f}")
            if line_score > 0.0:
                 print(f"Best Matching Line (Score: {line_score:.4f}): {line}")
            else:
                 print(f"Best Matching Line: {line}")
            print("-" * 20)

    else:
        print("Could not run search examples because embeddings were not loaded/generated.")