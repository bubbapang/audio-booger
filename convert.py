import PyPDF2
from gtts import gTTS
from tqdm import tqdm

def pdf_to_texts(file_path, pages_per_chunk=10):
    texts = []
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        total_pages = len(reader.pages)
        print(f"Extracting text from {total_pages} pages...")

        for start_page in tqdm(range(0, total_pages, pages_per_chunk)):
            end_page = start_page + pages_per_chunk
            text_chunk = ""
            
            for page_number in range(start_page, min(end_page, total_pages)):
                text_chunk += reader.pages[page_number].extract_text()
            
            texts.append(text_chunk)
    return texts

def text_to_speech(text, output_filename):
    print(f"Converting text to speech for {output_filename}...")
    tts = gTTS(text, lang="en")
    tts.save(output_filename)
    print("Conversion to mp3 complete!")

def pdf_to_mp3(pdf_file_path, mp3_file_base_path):
    print(f"Processing file: {pdf_file_path}")
    texts = pdf_to_texts(pdf_file_path)

    for idx, text in enumerate(texts):
        output_filename = f"{mp3_file_base_path}_{idx}.mp3"
        text_to_speech(text, output_filename)

    print("Processing complete!")

# Usage Example:
pdf_to_mp3("input.pdf", "output")
