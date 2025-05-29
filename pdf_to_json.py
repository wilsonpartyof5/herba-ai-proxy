import os
import json
import argparse
from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=4000):
    """Split text into smaller chunks for processing."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_chunk_with_openai(chunk, api_key):
    """Process text chunk with OpenAI to extract herbal information."""
    client = OpenAI(api_key=api_key)
    
    prompt = """Analyze the following text about herbs and herbal medicine. Provide a detailed summary of each herb mentioned, including:
    - Common name
    - Properties
    - Uses
    - Preparation methods
    - Cautions (if any)

    Please synthesize the information in your own words, avoiding direct quotes from the text. Include specific examples of remedies or uses for each herb, and explain how they can be beneficial. Format the response as a JSON object where each key is the herb's common name and the value is a detailed description.

    Text to analyze:
    {text}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are an expert herbalist and natural medicine practitioner. Extract and structure information about herbs from the provided text."},
                {"role": "user", "content": prompt.format(text=chunk)}
            ],
            temperature=0.3
        )
        
        # Extract the JSON from the response
        content = response.choices[0].message.content
        # Find JSON content between curly braces
        json_str = content[content.find("{"):content.rfind("}")+1]
        return json.loads(json_str)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Convert PDF to structured JSON for herbal knowledge base')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('--output', default='herbal_knowledge_synthesized.json', help='Output JSON file path')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    args = parser.parse_args()
    
    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(args.pdf_path)
    
    # Split into chunks
    print("Splitting text into chunks...")
    chunks = chunk_text(text)
    
    # Process each chunk
    print("Processing chunks with OpenAI...")
    all_herbs = {}
    for chunk in tqdm(chunks):
        herbs = process_chunk_with_openai(chunk, args.api_key)
        all_herbs.update(herbs)
    
    # Save to JSON file
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(all_herbs, f, indent=4)
    
    print(f"Processed {len(all_herbs)} herbs successfully!")

if __name__ == "__main__":
    main() 