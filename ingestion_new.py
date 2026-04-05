import os
import re
import uuid
import pickle
import json
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import dotenv_values
config=dotenv_values()
print('config:',config)

document_store={}
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(emb)

chroma_store = Chroma(
    collection_name=config['CHROMA_COLLECTION_NAME'],
    embedding_function=emb,
    persist_directory=config['CHROMA_PERSIST_DIRECTORY']
)
# headers_to_split_on= [
#     ('#','Header_1'),
#     ('##','Header_2')
# ]
PAGE_TMPL = "<!-- PAGE_{n} -->"
pattern_str = re.escape(PAGE_TMPL).replace(r"\{n\}", r"(\d+)")
page_pattern = re.compile(pattern_str)
headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
header_keys=[v for k,v in headers_to_split_on]

def get_clean_name(path):
    filename = os.path.basename(path)
    # Optional: Remove extension and underscores for better LLM reading
    return filename.replace(".pdf", "").replace("_", " ")
# PAGE_BREAK_PLACE_HOLDER='[[PAGE_BREAK]]"'
path='docs/2024_form_10k.pdf'
path='docs/demo3.pdf'
path='docs/demo_1_5.pdf'
output_path = os.path.join(config['PROCCESED_DOCS_DIR'], 'demo.md')

def convert_to_markdown(input_path,output_path):
    converter = DocumentConverter()
    print('Converting documented started...')
    result=converter.convert(path)
    PAGE_TMPL = "<!-- PAGE_{n} -->"
    pattern_str = re.escape(PAGE_TMPL).replace(r"\{n\}", r"(\d+)")
    page_pattern = re.compile(pattern_str)

    # Put a visible marker between pages first
    markdown = result.document.export_to_markdown(
        page_break_placeholder=PAGE_TMPL
    )
    parts = markdown.split(PAGE_TMPL)
    print(' BReak ON pAGES----',parts)
    numbered_markdown = "\n\n".join(
        f"{PAGE_TMPL.format(n=i+1)}\n{part.strip()}"
        for i, part in enumerate(parts)
        if part.strip()
    )
    with open (output_path,'w',encoding='utf-8') as f:
        f.write(numbered_markdown)
    print('write scuccsfuly')
    print('----NUmbered MD------------',numbered_markdown)
    print('FULL MARKDOWN--------------',markdown)
    tables = list(result.document.tables)
    print('total tables detected:',len(tables))

    return numbered_markdown, tables


def chunk_and_store_markdown(full_file_path,md,tables=None, delete_store=True):
    if delete_store:
       # chroma_store.delete_collection()
       chroma_store.reset_collection()
       print('Old collection deleted: ')
    print('Starting markdown splitter')

    # Replace generic breaks with numbered markers



    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        # strip_headers=False,
    )

    parents_md_splitts = splitter.split_text(md)
    print('==================PArent splitted markdown==============')
    # page_pattern = re.compile(r"<!-- PAGE_(\d+) -->")
    file_name=get_clean_name(path)
    last_seen_page=1
    for d in parents_md_splitts:
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
        d.metadata['source_path']=path
        d.metadata['file_name']=file_name

    # md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=int(config['PARENT_CHUNK_SIZE']),
        chunk_overlap=int(config['PARENT_CHUNK_VOERLAP'])
    )
    child_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=int(config['CHILD_CHUNK_SIZE']),
        chunk_overlap=int(config['CHILD_CHUNK_VOERLAP'])
    )
    # parents_md_splitts = md_splitter.split_text(md)
    # header_keys = ["h1", "h2", "h3"]
    for parent_md_split in parents_md_splitts:
        print('====PArent MD=======================================')
        print(parent_md_split)
        # print('===='*50)
        parent_chunk_splitts = parent_splitter.split_documents([parent_md_split])
        print('---------------Parent MD Split-------------------')
        for idx,parent_chunk in enumerate(parent_chunk_splitts):
            print('Parent idx=',idx,' ', '*'*50)
            print(parent_chunk)
            parent_id = uuid.uuid4().hex
            # breadcrumb = " > ".join(
            #     str(parent_chunk.metadata[k])
            #     for k in parent_chunk.metadata.values()
            #     # if k in parent_chunk.metadata
            # )
            # # breadcrumb = '>'.join(parent_chunk.metadata.values())
            # print('breadcrumb:',breadcrumb)
            # enriched_parent_text =f"DOCUMENTN SECTION:{breadcrumb}\n\n {parent_chunk.page_content}"
            # header_keys = ["h1", "h2", "h3"]
            breadcrumb = " > ".join(
                str(parent_chunk.metadata[k])
                for k in header_keys
                if k in parent_chunk.metadata
            )

            page_start = parent_chunk.metadata.get("page_start")
            page_end = parent_chunk.metadata.get("page_end")

            page_label = f"Page {page_start}" if page_start == page_end else f"Pages {page_start}-{page_end}"
            header=f"[Doc: {parent_chunk.metadata.get('file_name')}] [Section: {breadcrumb}] [Location: {page_label}]\n\n"
            # enriched_parent_text = (
            #     f"SOURCE: {parent_chunk.metadata.get('file_name')}\n"
            #     f"DOCUMENT SECTION: {breadcrumb}\n"
            #     f"SOURCE LOCATION: {page_label}\n\n"
            #     f"{parent_chunk.page_content}"
            # )
            enriched_parent_text = header + parent_chunk.page_content
            document_store[parent_id]= {'content':enriched_parent_text, 'metadata':json.dumps(parent_chunk.metadata)}
            child_splitts = child_splitter.split_text(parent_chunk.page_content)
            print('--------------Child Splitts-----------------------')
            childs=[]
            for idx,child_split in enumerate(child_splitts):
                # child_content=f"""Context: {breadcrumb}\n\n{child_split}"""
                header=f"[Doc: {parent_chunk.metadata.get('file_name')}] [Section: {breadcrumb}] [Location: {page_label}]\n\n"
                # enriched_child_text = (
                #     f"SOURCE: {parent_chunk.metadata.get('file_name')}\n"
                #     f"SECTION: {breadcrumb}\n"
                #     f"PAGE: {page_label}\n\n"
                #     f"{child_split}"
                # )
                #
                enriched_child_text = header+ child_split
                print('CHILD IDX=',idx, '---'*20)
                print(enriched_child_text)
                doc = Document(
                    page_content=enriched_child_text,
                    metadata={"parent":parent_id}
                )
                childs.append(doc)
            vector_store.add_documents(childs)
            chroma_store.add_documents(childs)
            print(f'Parent:{idx}: All childs added to vector store')

        with open(config['DOCUMENT_STORE_PERSIST_PATH'], 'wb') as f:
            pickle.dump(document_store,f)
            print('Successfuly persist document store to file')

def parent_child_search(query,k=3, inmemory=True):
    print('Searching Parent-Child databse')
    if inmemory:
        print('Searching in memory store...')
        childs = vector_store.similarity_search(query,k=k)
    else:
        print('Searching Chroma db....')
        childs = chroma_store.similarity_search(query,k=k)
        print('Chroma childs:',childs)
    seen_ids=set()
    deduplicated=[]
    print('parent ids in sotre:',document_store.keys())
    for child in childs:
        parent_id =child.metadata['parent']
        print('parent id:',parent_id)
        if parent_id not in seen_ids:
           seen_ids.add(parent_id)
           if parent_id not in document_store:
               print(f'Parent ID {parent_id} not found in Store')
           deduplicated.append(document_store.get(parent_id,''))
    print(f'Deduplicated results: {len(childs)} -> {len(deduplicated)}')
    print(deduplicated)
    return deduplicated
def run_pipeline():
    md, tables=convert_to_markdown(path,output_path)
    chunk_and_store_markdown(path,md)
if __name__=='__main__':
    run_pipeline()
    print('======================== PIPELINE DONE=======================')
    query ='Rule 405 of Regulation S-T'
    print('Searc hvector store:')
    results = parent_child_search(query,k=5,inmemory=False)
    print('Search results:')
    for doc in results:
        print('Doc----',doc)
