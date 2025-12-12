import fitz  # This is PyMuPDF
import easyocr
import numpy as np

def extract_text_from_image_pdf(pdf_path):
    # 1. Initialize the OCR reader
    # 'en' indicates English. You can add more languages ['en', 'fr']
    # gpu=False ensures it runs even if you don't have a CUDA graphics card
    reader = easyocr.Reader(['en'], gpu=False) 
    
    # 2. Open the PDF file
    doc = fitz.open(pdf_path)
    full_text = ""

    print(f"Processing {len(doc)} pages...")

    for page_num, page in enumerate(doc):
        # 3. Render the page to an image (pixmap)
        # matrix=fitz.Matrix(2, 2) zooms in 2x for better OCR accuracy
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        
        # 4. Convert the image structure to a numpy array for EasyOCR
        # PyMuPDF gives us bytes; we need to format it for EasyOCR
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        # If the image has an alpha channel (transparency), remove it
        if pix.n == 4:
            img_data = img_data[:, :, :3]

        # 5. Extract text from the image array
        # detail=0 gives us simple paragraph output
        page_results = reader.readtext(img_data, detail=0)
        
        # Join the list of strings found on the page
        page_text = "\n".join(page_results)
        
        print(f"--- Page {page_num + 1} Done ---")
        full_text += f"\n\n--- Page {page_num + 1} ---\n{page_text}"

    return full_text

# Usage
pdf_file = "documents\FlightDetails.pdf"  # Replace with your file path
try:
    extracted_text = extract_text_from_image_pdf(pdf_file)
    
    # Save to a file
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
        
    print("Extraction complete! Saved to output.txt")
    
except Exception as e:
    print(f"An error occurred: {e}")