# loader.py
# responsible for reading raw HR documents from data/hr_docs/ and returning plain text
# needs to handle multiple file types:
#   - PDF files using pypdf
#   - Word documents (.docx) using python-docx
# output: raw text string + metadata dict (filename, file type, date loaded)
# does not do any chunking — that is chunker.py's job