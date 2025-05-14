import cohere
import base64
import os
import argparse
from PIL import Image
from io import BytesIO

def image_to_base64_data_url(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Create a BytesIO object to hold the image data in memory
        buffered = BytesIO()
        # Save the image as PNG to the BytesIO object
        img.save(buffered, format="PNG")
        # Encode the image data in base64
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # Create the Data URL
        data_url = f"data:image/png;base64,{img_base64}"
        return data_url

def get_image_embedding(api_key, image_path, model="embed-v4.0", output_dimension=1024):
    # Initialize the Cohere client
    co = cohere.ClientV2(api_key=api_key)
    
    # Convert image to base64 data URL
    processed_image = image_to_base64_data_url(image_path)
    
    # Get embedding from Cohere API
    res = co.embed(
        images=[processed_image],
        model=model,
        embedding_types=["float"],
        input_type="image",
        output_dimension=output_dimension
    )
    
    return res.embeddings.float[0]

def process_directory(api_key, directory, output_file, model="embed-v4.0", output_dimension=1024):
    """Process all images in a directory and save their embeddings to a file"""
    import pickle
    
    embeddings = {}
    
    # Process all image files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
            image_path = os.path.join(directory, filename)
            print(f"Processing {filename}")
            try:
                embedding = get_image_embedding(
                    api_key, 
                    image_path, 
                    model=model, 
                    output_dimension=output_dimension
                )
                embeddings[filename] = embedding
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    # Save embeddings to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
        
    print(f"Saved embeddings for {len(embeddings)} images to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for images using Cohere API")
    parser.add_argument("--api_key", required=True, help="Cohere API key")
    parser.add_argument("--input", required=True, help="Input image path or directory")
    parser.add_argument("--output", default="embeddings.pkl", help="Output file for embeddings")
    parser.add_argument("--model", default="embed-v4.0", help="Cohere embedding model to use")
    parser.add_argument("--dimension", type=int, default=1024, 
                        help="Output dimension for embeddings (256, 512, 1024, or 1536)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Process all images in the directory
        process_directory(
            args.api_key, 
            args.input, 
            args.output, 
            model=args.model, 
            output_dimension=args.dimension
        )
    else:
        # Process a single image
        embedding = get_image_embedding(
            args.api_key, 
            args.input, 
            model=args.model, 
            output_dimension=args.dimension
        )
        
        # Save embedding to file
        import pickle
        with open(args.output, 'wb') as f:
            pickle.dump({os.path.basename(args.input): embedding}, f)
            
        print(f"Saved embedding to {args.output}") 