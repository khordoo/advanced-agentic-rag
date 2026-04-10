import uuid
from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document

from rag_core.schemas import ParentChunk
from rag_core.agents import PromptLoader, TableSummarizer

# def summarize_table(model: BaseChatModel, table_markdown: str) -> str:
#     """Summarizes a table using the provided model and the 'table_summary' prompt."""
#         # 1. Load the prompt (only when this function is actually called)
#     prompt_config = PromptLoader.load("table_summary",table_markdown=table_markdown)
#         # 2. Format the template
#     # prompt = prompt_config.template.format(table_markdown=table_markdown)
#         # 3. Invoke the LLM
#     print('Prompt:',prompt_config.template)
#     response = model.invoke(prompt_config.template)
#     return response.content.strip()

def summerize_and_chunk_table(table_makrdowns: list[str], model:TableSummarizer,summerize_enabled=True) -> list[ParentChunk]:
    print(f'Summerziing the {len(table_makrdowns)} tables...')
    parent_chunks=[]
    for i,table_md in enumerate(table_makrdowns):
        table_id = f"DATA_TABLE_{i:03d}"
        table_summary =model.summarize(table_markdown=table_md) if summerize_enabled else table_md
        parent_id = uuid.uuid4().hex
        # Create the Child (The Searchable Vector)
        # We explicitly put the table_id in the text so the vector DB indexes it!
        child_doc = Document(
                    page_content=f"[SECTION: TABLE SUMMARY FOR {table_id}]\n\n{table_summary}",
                    metadata={
                        "parent": parent_id, 
                        "table_id": table_id # Added for strict metadata filtering
                    }
                )
        # Create the Parent (The LLM Payload)
        # We append the ID to the top so the LLM has a visual anchor
        enriched_table_md = f"[{table_id}]\n\n{table_md}"
        parent_chunks.append(
            ParentChunk(
                parent_id=parent_id,
                enriched_text=enriched_table_md,
                metadata=child_doc.metadata,
                children=[child_doc],
            )
        )

    # print(f'Parent chunk {len(parent_chunks)}:',parent_chunks[-1])
    print(f'Parent chunk {len(parent_chunks)}:')
    return parent_chunks
