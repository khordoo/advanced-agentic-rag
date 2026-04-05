from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 1. A dummy document with headers and one very long paragraph
markdown_document = """
# Introduction
Welcome to the company. This section is short.

## Remote Work Policy
Employees are allowed to work from anywhere in the world. 
However, you must have a stable internet connection at all times. This is a very long paragraph that we are intentionally making long so that the recursive splitter is forced to step in and chop it up. If we don't make it long, the recursive splitter will just look at it and say 'this is fine' and leave it alone. But we want to see the sliding window overlap in action!
"""

# 2. First Pass: Split by the Markdown headers
headers_to_split_on = [
    ("#", "Header_1"),
    ("##", "Header_2"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_splits = markdown_splitter.split_text(markdown_document)

print("--- AFTER MARKDOWN SPLITTING ---")
for i, chunk in enumerate(md_splits):
    print(f"Chunk {i+1} Metadata: {chunk.metadata}")
    print(f"Chunk {i+1} Text Length: {len(chunk.page_content)}")
    print("-" * 20)

# 3. Second Pass: The Recursive Splitter
# We set a tiny chunk size (100) just to force it to split our text
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20 # 20 characters of sliding window overlap!
)

final_chunks = recursive_splitter.split_documents(md_splits)

print("\n--- AFTER RECURSIVE SPLITTING ---")
for i, chunk in enumerate(final_chunks):
    print(f"Chunk {i+1} Metadata: {chunk.metadata}")
    print(f"Chunk {i+1} Content: {chunk.page_content}")
    print("-" * 20)

