import numpy as np
import pickle
import argparse
import os
from itertools import combinations

def calculate_similarity(embedding_a, embedding_b):
    """Calculate cosine similarity between two embeddings"""
    return np.dot(embedding_a, embedding_b) / (np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b))

def similarity_to_percentage(similarity):
    """Convert cosine similarity to a percentage value"""
    # Map the similarity from [-1,1] to [0,100]
    # For image embeddings, values are typically positive, so we can simplify
    percentage = max(0, similarity * 100)
    return percentage

def find_most_similar_pairs(embeddings, top_n=10):
    """Find the most similar pairs of images based on their embeddings"""
    # Create a list of all image pairs and their similarity scores
    image_pairs = []
    for (image1, emb1), (image2, emb2) in combinations(embeddings.items(), 2):
        similarity = calculate_similarity(emb1, emb2)
        image_pairs.append((image1, image2, similarity))
    
    # Sort pairs by similarity score (highest first)
    image_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return the top N most similar pairs
    return image_pairs[:top_n]

def find_similar_to_image(target_image, embeddings):
    """Find images most similar to a target image"""
    target_embedding = embeddings[target_image]
    similarities = []
    
    for image, embedding in embeddings.items():
        if image != target_image:  # Skip comparing with itself
            similarity = calculate_similarity(target_embedding, embedding)
            similarities.append((image, similarity))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare image embeddings to find similar images")
    parser.add_argument("--embeddings", required=True, help="Path to the embeddings pickle file")
    parser.add_argument("--target", help="Target image to compare others against (optional)")
    parser.add_argument("--top", type=int, default=10, help="Number of top similar pairs to show")
    
    args = parser.parse_args()
    
    # Load embeddings
    with open(args.embeddings, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Loaded embeddings for {len(embeddings)} images")
    
    if args.target:
        # Find images similar to the target
        if args.target not in embeddings:
            print(f"Error: Target image '{args.target}' not found in embeddings")
            exit(1)
        
        similar_images = find_similar_to_image(args.target, embeddings)
        
        print(f"\nImages most similar to {args.target}:")
        for i, (image, similarity) in enumerate(similar_images[:args.top], 1):
            percentage = similarity_to_percentage(similarity)
            print(f"{i}. {image} (similarity: {similarity:.4f}, {percentage:.1f}%)")
    else:
        # Find most similar pairs
        similar_pairs = find_most_similar_pairs(embeddings, args.top)
        
        print("\nMost similar image pairs:")
        for i, (image1, image2, similarity) in enumerate(similar_pairs, 1):
            percentage = similarity_to_percentage(similarity)
            print(f"{i}. {image1} and {image2} (similarity: {similarity:.4f}, {percentage:.1f}%)") 