
import os
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from dotenv import dotenv_values
config=dotenv_values()

headers_to_split_on= [
    ('#','Header_1'),
    ('##','Header_2')
]
PAGE_BREAK_PLACE_HOLDER='[[PAGE_BREAK]]"'
path='docs/2024_form_10k.pdf'
path='docs/demo3.pdf'
path='docs/demo_1_5.pdf'
output_path = os.path.join(config['PROCCESED_DOCS_DIR'], 'demo.md')

converter = DocumentConverter()
print('Converting documented started...')
result=converter.convert(path)
# print('converting to markdown...')
# markdown = result.document.export_to_markdown(
#     page_break_placeholder=PAGE_BREAK_PLACE_HOLDER
# )

import re
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

PAGE_TMPL = "<!-- PAGE_{n} -->"
pattern_str = re.escape(PAGE_TMPL).replace(r"\{n\}", r"(\d+)")
page_pattern = re.compile(pattern_str)

# Put a visible marker between pages first
markdown = result.document.export_to_markdown(
    page_break_placeholder=PAGE_TMPL
)
print('FULL MARKDOWN--------------',markdown)
# Replace generic breaks with numbered markers
parts = markdown.split(PAGE_TMPL)
print(' BReak ON pAGES----',parts)
numbered_markdown = "\n\n".join(
    f"{PAGE_TMPL.format(n=i+1)}\n{part.strip()}"
    for i, part in enumerate(parts)
    if part.strip()
)
with open ('docs/demo_num.md','w',encoding='utf-8') as f:
    f.write(numbered_markdown)

print('----NUmbered MD------------',numbered_markdown)
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)

docs = splitter.split_text(numbered_markdown)

# page_pattern = re.compile(r"<!-- PAGE_(\d+) -->")

last_seen_page=1
for d in docs:
    pages = [int(x) for x in page_pattern.findall(d.page_content)]

    if pages:
        d.metadata["page_start"] = min(pages)
        d.metadata["page_end"] = max(pages)
        d.metadata["pages"] = sorted(set(pages))
        last_seen_page= d.metadata["page_end"]
    else:
        print('********************No Page found:')
        d.metadata["page_start"] = last_seen_page
        d.metadata["page_end"] = last_seen_page
        d.metadata["pages"] = [last_seen_page]
        print(d)
    # remove markers from chunk text after extracting metadata
    d.page_content = page_pattern.sub("", d.page_content).strip()
   
print(docs)
# print('===============doc wit hpage meta=================================')
# print(all)
