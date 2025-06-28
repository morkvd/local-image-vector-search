#!/usr/bin/env python3
import torch
from PIL import Image
import open_clip
import os
import io
import chromadb
import argparse

"""
1. place this script in the root folder of the directory structure you want to search through
2. run `python .\photo-search.py index` to:
   - go through all images in this directory structure
   - turn them into vectors
   - save these into a local sqlite database
   (this may take a while)
3. after that you can run `python .\photo-search.py search "green goblin"`
   to open the N amount of photos that match your search term.

TODO: Conda this shit so it uses the same packages wherever it is executed
TODO: Turn it into a docker image that includes the model so it doesn't depend on huggingface existing. 

"""

db_name = 'IMAGE_DB'
image_files_list_filename = 'image_files_list.txt'

# https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/tree/main
local_model_path = "C:/Users/markv/Code/python/models/open_clip_pytorch_model.bin"


# btw open_clip saves this model in ~/.cache/huggingface/hub 
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=local_model_path)
model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active (i dont know what this means :D)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

client = chromadb.PersistentClient(path='.') # creates the database in the folder you run this

"""
TODO: play arount with these settings

Configuring Chroma Collections

You can configure the embedding space, HNSW index parameters, and embedding function of a collection by setting the collection configuration. These configurations will help you customize your Chroma collections for different data, accuracy and performance requirements.

The space parameter defines the distance function of the embedding space. The default is l2 (squared L2 norm), and other possible values are cosine (cosine similarity), and ip (inner product).
Distance	parameter	Equation
Squared L2	               d =  \sum\left(A_i-B_i\right)^2 
Inner product	            d = 1.0 - \sum\left(A_i \times B_i\right) 
Cosine similarity	cosine	d = 1.0 - \frac{\sum\left(A_i \times B_i\right)}{\sqrt{\sum\left(A_i^2\right)} \cdot \sqrt{\sum\left(B_i^2\right)}} 

SW Index Configuration#

The HNSW index parameters include:

    ef_construction determines the size of the candidate list used to select neighbors during index creation. A higher value improves index quality at the cost of more memory and time, while a lower value speeds up construction with reduced accuracy. The default value is 100.
    ef_search determines the size of the dynamic candidate list used while searching for the nearest neighbors. A higher value improves recall and accuracy by exploring more potential neighbors but increases query time and computational cost, while a lower value results in faster but less accurate searches. The default value is 100.
    max_neighbors is the maximum number of neighbors (connections) that each node in the graph can have during the construction of the index. A higher value results in a denser graph, leading to better recall and accuracy during searches but increases memory usage and construction time. A lower value creates a sparser graph, reducing memory usage and construction time but at the cost of lower search accuracy and recall. The default value is 16.
    num_threads specifies the number of threads to use during index construction or search operations. The default value is multiprocessing.cpu_count() (available CPU cores).
    batch_size controls the number of vectors to process in each batch during index operations. The default value is 100.
    sync_threshold determines when to synchronize the index with persistent storage. The default value is 1000.
    resize_factor controls how much the index grows when it needs to be resized. The default value is 1.2.
"""

# I guess cosine is supposed to be better but not seeing much difference.
collection = client.get_or_create_collection(name=db_name, configuration={"hnsw":{"space": "cosine"}})


def create_file_index(start_dir='.'):
   image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.heic']

   # gather files in directory (and subdirectories) 
   image_files = [] 
   for root, _, files in os.walk(start_dir):
      for file in files:
         if any(file.lower().endswith(ext) for ext in image_extensions):
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, start_dir)
            image_files.append(rel_path)
    
   # write to file    
   image_files.sort()
   with open(image_files_list_filename, 'w', encoding='utf-8') as f:
      for image_file in image_files:
         f.write(f"{image_file}\n")
    
   print(f"Found {len(image_files)} image files.")
   print(f"List saved to: {image_files_list_filename}")


def save_embeddings_in_vectordatabase():
   # Check if the image list file exists
   if not os.path.isfile(image_files_list_filename):
      print(f"Error: The file '{image_files_list_filename}' does not exist.")
      return 1
    
   # Read the image list
   try:
      with open(image_files_list_filename, 'r', encoding='utf-8') as f:
         image_paths = [line.strip() for line in f if line.strip()]
        
      print(f"Found {len(image_paths)} images in the list.")
     
      # Process each image
      succeeded = 0
      for i, image_path in enumerate(image_paths):
         print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
         image_bytes = get_bytes(image_path)
         embedding = generate_embedding(image_bytes)
         if embedding:
            #print(f"Successfully generated embeddings for '{image_path}'")
            if (embed_in_chroma(image_path, embedding)):
               succeeded = succeeded + 1
         else:
            print(f"Failed to generate embeddings for '{image_path}'")
        
      print(f"\nProcessed {succeeded} images successfully out of {len(image_paths)} total.")
      return 0

   except Exception as e:
      print(f"Error processing images: {e}")
      return 1

def get_bytes(image_path):
   abs_image_path = os.path.abspath(image_path)
    
   if not os.path.isfile(abs_image_path):
      print(f"Error: The file '{image_path}' does not exist.")
      print(f"Absolute path: '{abs_image_path}'")
      return None

   try:
      with open(abs_image_path, 'rb') as image_file:
         image_bytes = image_file.read()
        
      #print(f"Successfully read {len(image_bytes)} bytes from '{abs_image_path}'")

      return image_bytes

   except Exception as e:
      print(f"Error reading the image: {e}")
      return None


def generate_embedding(bytes):
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
         #print("Embeddings generated successfully.")
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

        #print(f"Added embedding to Chroma for {filename}")
        return True
    except Exception as e:
        # Log an error if the addition to Chroma fails
        print(f"Failed to add embedding to Chroma for {filename}: {e}")
        return False



def build_index():
   print("start building index")
   create_file_index()
   save_embeddings_in_vectordatabase()


def search_files(search_string, num_results):
   print("searching for: ", search_string)

   items = collection.get()
   #print(items)
   print(len(items["ids"]), " images in database")
    
   # Tokenize the entire search string as a single unit
   text = tokenizer([search_string])
   
   with torch.no_grad():
      text_features = model.encode_text(text)
      text_features /= text_features.norm(dim=-1, keepdim=True)
      embedding = text_features.cpu().numpy().tolist()
    
   print("Created single embedding for the entire search string")

   # Use the specified number of results with a single embedding
   results = collection.query(query_embeddings=embedding, n_results=(num_results))
    
   for ids in results["ids"]:
      for id in ids:
         print(id)
         os.startfile(id)



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