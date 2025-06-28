#!/usr/bin/env python3
import torch
from PIL import Image
import open_clip
import os
import io
import chromadb
import argparse

db_name = 'IMAGE_DB'

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
tokenizer = open_clip.get_tokenizer('ViT-B-32')

client = chromadb.PersistentClient(path='.')
collection = client.get_or_create_collection(name=db_name)

def create_file_index(start_dir='.'):
   image_extensions = [
      '.jpg', '.jpeg', '.png', '.gif', '.bmp',
      '.tiff', '.tif', '.webp', '.svg', '.heic'
   ]
    
   image_files = []
    
   # Walk through all directories starting from start_dir
   for root, dirs, files in os.walk(start_dir):
      for file in files:
         # Check if the file has an image extension
         if any(file.lower().endswith(ext) for ext in image_extensions):
            # Get the full path
            full_path = os.path.join(root, file)
            # Convert to relative path
            rel_path = os.path.relpath(full_path, start_dir)
            image_files.append(rel_path)
    
   # Sort the list for better readability
   image_files.sort()
    
   # Create output filename
   output_file = 'image_files_list.txt'
    
   # Write the list to a text file
   with open(output_file, 'w', encoding='utf-8') as f:
      for image_file in image_files:
         f.write(f"{image_file}\n")
    
   print(f"Found {len(image_files)} image files.")
   print(f"List saved to: {output_file}")


def save_embeddings_in_vectordatabase():
   # Path to the image list file
   image_list_path = 'image_files_list.txt'
    
   # Check if the image list file exists
   if not os.path.isfile(image_list_path):
      print(f"Error: The file '{image_list_path}' does not exist.")
      return 1
    
   # Read the image list
   try:
      with open(image_list_path, 'r', encoding='utf-8') as f:
         image_paths = [line.strip() for line in f if line.strip()]
        
      print(f"Found {len(image_paths)} images in the list.")
     
      # Process each image
      results = []
      for i, image_path in enumerate(image_paths):
         print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
         embedding = process_image(image_path)
         if embedding:
            print(f"Successfully generated embeddings for '{image_path}'")
            embed_in_chroma(image_path, embedding)
         else:
            print(f"Failed to generate embeddings for '{image_path}'")
        
      print(f"\nProcessed {len(results)} images successfully out of {len(image_paths)} total.")
      return 0
   except Exception as e:
      print(f"Error processing images: {e}")
      return 1

def process_image(image_path):
   """Process a single image and generate embeddings"""
   # Convert relative path to absolute path
   abs_image_path = os.path.abspath(image_path)
    
   # Check if the file exists
   if not os.path.isfile(abs_image_path):
      print(f"Error: The file '{image_path}' does not exist.")
      print(f"Absolute path: '{abs_image_path}'")
      return None
    
   # Read the image bytes
   try:
      with open(abs_image_path, 'rb') as image_file:
         image_bytes = image_file.read()
        
      # Print information about the bytes read
      print(f"Successfully read {len(image_bytes)} bytes from '{image_path}'")
      print(f"Absolute path: '{abs_image_path}'")
        
      # Generate embeddings for the image
      return classify_file(image_bytes)
   except Exception as e:
      print(f"Error reading the image: {e}")
      return None


def classify_file(bytes):
   try:
      image = Image.open(io.BytesIO(bytes))
      image = preprocess(image).unsqueeze(0)  # Process image, then move and convert
   except Exception as e:
      print(f"Error processing image: {e}")
      raise

   # Generate embeddings
   with torch.no_grad():
      try:
         image_features = model.encode_image(image)
         image_features /= image_features.norm(dim=-1, keepdim=True)
         embeddings = image_features.cpu().numpy().tolist()
         print("Embeddings generated successfully.")
      except Exception as e:
         print(f"Error generating embeddings: {e}")
         raise

   return embeddings


def embed_in_chroma(filename, embedding):
    try:
        # Add the photo's embeddings to the Chroma collection
        item = collection.get(ids=[filename])
        if item['ids'] !=[]:
            print(f"Already stored {filename}")
            return
        collection.add(
            embeddings=embedding,
            documents=[filename],
            ids=[filename]
        )

        #items = collection.get()
        #print(items)

        print(f"Added embedding to Chroma for {filename}")
    except Exception as e:
        # Log an error if the addition to Chroma fails
        print(f"Failed to add embedding to Chroma for {filename}: {e}")



def build_index():
   print("start building index")
   create_file_index()
   save_embeddings_in_vectordatabase()


def search_files(search_string, num_results):
   print("searching for: ", search_string)

   # Split search string into words if it contains multiple words
   search_words = search_string.split()

   items = collection.get()
   #print(items)
   print(len(items["ids"]), " images in database")
    
   # Tokenize the search words
   text = tokenizer(search_words)
    
   with torch.no_grad():
      text_features = model.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      embeddings = text_features.cpu().numpy().tolist()
    
   print(len(embeddings))

   # Use the specified number of results
   results = collection.query(query_embeddings=embeddings, n_results=(num_results))
    
   for ids in results["ids"]:
      for id in ids:
         print(id)


def main():
   parser = argparse.ArgumentParser(description='Search your local computer for images in natural language')
   subparsers = parser.add_subparsers(dest='command', help='Commands to run', required=True)

   parser_build_index = subparsers.add_parser('index')
   parser_build_index.set_defaults(func=build_index)

   parser_search_files = subparsers.add_parser('search', help='"search string" --num_results 10')
   parser_search_files.add_argument('search_string', type=str, help='Text to search for (words will be split if multiple)')
   parser_search_files.add_argument('--num_results', type=int, default=10, help='Number of results to return (default: 10)')
   parser_search_files.set_defaults(func=search_files)

   args = parser.parse_args()
   args_ = vars(args).copy()
   args_.pop('command', None)
   args_.pop('func', None)
   args.func(**args_)

if __name__ == "__main__":
   main()