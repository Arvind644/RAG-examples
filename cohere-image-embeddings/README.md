# Image Similarity with Cohere Embeddings

This project demonstrates how to use Cohere's image embedding capabilities to identify similar images.

## Setup

1. Install the required packages:
```
pip install cohere pillow numpy
```

2. Make sure you have a Cohere API key. You can get one by signing up at [cohere.com](https://cohere.com).

## Usage

### 1. Generate Embeddings

First, generate embeddings for your images using the `image_to_embeddings.py` script:

```
python image_to_embeddings.py --api_key YOUR_API_KEY --input path/to/images --output embeddings.pkl
```

Parameters:
- `--api_key`: Your Cohere API key (required)
- `--input`: Path to an image file or directory of images (required)
- `--output`: Path to save the embeddings (default: embeddings.pkl)
- `--model`: Cohere embedding model to use (default: embed-v4.0)
- `--dimension`: Output dimension for embeddings (256, 512, 1024, or 1536) (default: 1024)

### 2. Compare Embeddings

After generating embeddings, use the `compare_image_embeddings.py` script to find similar images:

```
python compare_image_embeddings.py --embeddings embeddings.pkl
```

Parameters:
- `--embeddings`: Path to the embeddings pickle file (required)
- `--target`: Target image to compare others against (optional)
- `--top`: Number of top similar pairs to show (default: 10)

### Examples

Generate embeddings for all images in a directory:
```
python image_to_embeddings.py --api_key YOUR_API_KEY --input ./my_images --output my_embeddings.pkl
```

Find the most similar image pairs:
```
python compare_image_embeddings.py --embeddings my_embeddings.pkl --top 5
```

Find images most similar to a specific image:
```
python compare_image_embeddings.py --embeddings my_embeddings.pkl --target cat.jpg
```

## How It Works

1. The first script converts images to base64 data URLs and uses Cohere's API to generate vector embeddings.
2. The second script calculates cosine similarity between embeddings to determine how similar images are to each other.
3. Similar images will have embeddings with high cosine similarity (closer to 1.0).

## Notes

- The Cohere embedding model supports multiple dimensions (256, 512, 1024, or 1536).
- Larger dimensions may provide better similarity results but will use more memory.
- The free tier of Cohere's API has rate limits, so processing large image collections may require pauses. 