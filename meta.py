from transformers import VitsModel, AutoTokenizer
from pydub import AudioSegment
import numpy as np
import PyPDF2
import torch
import re
import os
import scipy

input_name = "input.pdf"

def pdf2text(pdf_filepath, chunk_size=100, portion=1.0):
    """
    portion: float, portion of pages to be processed, e.g., 0.1 means process 10% of all pages.
    """
    print("pdf2texting...")

    pdfFileObj = open(pdf_filepath, 'rb')
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    
    # Determine the range of pages to process
    total_pages = len(pdfReader.pages)
    end_page = int(total_pages * portion)  # Up to which page we want to process
    
    text = ""
    for page_num in range(end_page):
        pageObj = pdfReader.pages[page_num]
        text += pageObj.extract_text()
    pdfFileObj.close()

    text = re.sub(r'[^\x00-\x7F]+', '', text)
    for char in ['/', '+', '(', ')', '"', "@", "-", ",", ";"]:
        text = text.replace(char, " ")

    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Split into words and create chunks
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

tts_count = 0
def tts(text, model, tokenizer):
    global tts_count  # Use the global variable, not create a local one
    print("tts-ing...", tts_count)

    tts_count += 1

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    
    return np.squeeze(output.float().numpy())

# Model loading once
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

# Generate text chunks directly from PDF
# Extract and process only the first 10% of pages for quick testing
text_chunks = pdf2text(input_name, chunk_size=100, portion=1.0)    

# Initialize an empty numpy array for storing the concatenated audio data
concatenated_audio_np = np.array([]).astype(np.float32)

total_chunks = len(text_chunks)  # Total number of chunks

# Using enumerate to get both the index (idx) and the chunk
for idx, chunk in enumerate(text_chunks):
    concatenated_audio_np = np.concatenate([concatenated_audio_np, tts(chunk, model, tokenizer)])
    
    # Calculate and print the percentage completed so far
    percentage_completed = ((idx + 1) / total_chunks) * 100
    print(f'Processed {idx + 1} of {total_chunks} chunks ({percentage_completed:.2f}%)')

scipy.io.wavfile.write("book.wav", int(model.config.sampling_rate), concatenated_audio_np)