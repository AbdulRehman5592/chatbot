import io
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import glob
from pytesseract import Output
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter


async def extract_text_from_pdfs(files, session_id=None):
    """
    Process uploaded PDFs using PyMuPDF to generate image files in the output directory.
    Then process all .jpeg files in the output_dir using OCR.
    Returns all OCR text concatenated, and a list of processed file names, chunks, and metadatas.
    """
    pdf_names = []
    output_dir = f"pdf_output/{session_id or 'default'}"
    os.makedirs(output_dir, exist_ok=True)
    ocr_text = ""

    # Only process images from PDF
    for file in files:
        pdf_names.append(file.filename)
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")
        
        for page_num, page in enumerate(doc.pages()):
            # --- Extract images from the page ---
            img_list = page.get_images(full=True)
            for img_num, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                ext = base_image["ext"]
                img_path = os.path.join(output_dir, f"{file.filename}_page{page_num+1}_img{img_num+1}.{ext}")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)
        doc.close()

    # Only process .jpeg files in the output_dir (OCR with coordinate mapping)
    jpeg_files = glob.glob(os.path.join(output_dir, "*.jpeg"))
    ocr_full_text = ""
    ocr_chunks = []
    ocr_metadatas = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)

    for jpeg_file in jpeg_files:
        img = Image.open(jpeg_file)
        
        # Use image_to_data to get detailed OCR output with coordinates
        ocr_df = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
        ocr_df = ocr_df.dropna(subset=["text"])
        ocr_df["text"] = ocr_df["text"].astype(str)
        ocr_df = ocr_df[ocr_df.text.str.strip() != '']
        
        # Sort words by their vertical and then horizontal position
        sorted_df = ocr_df.sort_values(by=['top', 'left'], ascending=True)
        
        # Reconstruct text from sorted words
        full_page_text = ' '.join(sorted_df['text'])
        ocr_full_text += f"File: {os.path.basename(jpeg_file)}\n{full_page_text}\n\n"

        # Write OCR result to a separate .txt file
        ocr_txt_file = os.path.splitext(jpeg_file)[0] + "_coordinates.txt"
        sorted_df.to_csv(ocr_txt_file, index=False, sep='\t')

        # Chunk the full text
        chunks = text_splitter.split_text(full_page_text)
        
        # Map chunks back to words and their bounding boxes
        word_idx = 0
        for chunk in chunks:
            chunk_words = []
            chunk_text_covered = ""
            
            # Find the words from the dataframe that make up the current chunk
            while word_idx < len(sorted_df) and len(chunk_text_covered) < len(chunk):
                word_data = sorted_df.iloc[word_idx]
                chunk_words.append(word_data)
                chunk_text_covered += word_data['text'] + ' '
                word_idx += 1

            if not chunk_words:
                continue

            # Calculate the bounding box for the entire chunk
            x_min = min(int(w['left']) for w in chunk_words)
            y_min = min(int(w['top']) for w in chunk_words)
            x_max = max(int(w['left'] + w['width']) for w in chunk_words)
            y_max = max(int(w['top'] + w['height']) for w in chunk_words)
            
            # Extract image number from filename
            base = os.path.basename(jpeg_file)
            img_num = None
            import re
            match = re.search(r'_img(\d+)', base)
            if match:
                img_num = int(match.group(1))

            ocr_chunks.append(chunk)
            meta = {
                "source": base,
                "image_file": base,  # <-- Ensures downstream code always has the correct filename
                "image_number": int(img_num) if img_num is not None else None,
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                "page_width": int(img.width),
                "page_height": int(img.height)
            }
            ocr_metadatas.append(meta)

    # Write all OCR text to a single file in the output directory
    ocr_full_text_path = os.path.join(output_dir, "ocr_full_text.txt")
    with open(ocr_full_text_path, "w", encoding="utf-8") as f:
        f.write(ocr_full_text)
        
    return ocr_full_text, ocr_full_text, pdf_names, ocr_chunks, ocr_metadatas 