from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar

def extract_and_clean_text(pdf_path):
    text_content = []

    # Iterate over all the pages
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                # Filter out non-text elements and empty text strings
                text_lines = [text_line.strip() for text_line in element.get_text().splitlines() if text_line.strip()]
                text_content.extend(text_lines)

    return text_content

def process_text_lines(text_lines):
    processed_entries = []
    # Assuming that each name entry is followed by the language abbreviation and then the meaning
    for i in range(0, len(text_lines), 3):
        # Skip page numbers and headers
        if text_lines[i].isdigit() or text_lines[i].upper() == text_lines[i]:
            continue
        name = text_lines[i]
        language = text_lines[i + 1] if i + 1 < len(text_lines) else ''
        meaning = text_lines[i + 2] if i + 2 < len(text_lines) else ''
        processed_entries.append((name, language, meaning))

    return processed_entries
# File path
pdf_path = './books/pdfcoffee.com_complete-english-latin-dictionary-pdf-free.pdf'

# Extract text from the PDF
text_lines = extract_and_clean_text(pdf_path)

# Further cleaning and structuring the text
entries = process_text_lines(text_lines)

# Print entries to check if they are extracted correctly
for entry in entries:
    print(entry)
    print('---------------------------')
