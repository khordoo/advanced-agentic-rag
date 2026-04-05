from pydantic import BaseModel, Field
from typing import List,Optional,Literal
import json
import asyncio
import time
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import dotenv_values
import pickle
from langchain_aws import ChatBedrockConverse
from agents import AgentFactory,PromptLoader
from schemas import QAPair,SyntheticsTestSet,QARecord
config =dotenv_values()

with open(config['DOCUMENT_STORE_PERSIST_PATH'],'rb') as f:
   document_store = pickle.load(f)
   print('Succefully loaded the document store.')

schema = SyntheticsTestSet.model_json_schema()
prompt_template=PromptLoader.load('qa_gen',schema=json.dumps(schema),context='')
# llm=AgentFactory.create('aws',temperature=prompt_template.temperature)
llm=AgentFactory.get_judge_model()
system_prompt=prompt_template.template

def save_to_file(path,qa_pairs):
    # final_pairs = SyntheticsTestSet(items=qa_pairs)
    with open(path,'w', encoding='utf-8') as f:
        json.dump(
            qa_pairs,
            f,
            ensure_ascii=False,
            indent=2,
        )
    print('Successfuly wrote file:',path)

# def generate_questions_sync():
#     qa_pairs=[]
#     for parent_id, md_doc in document_store.items():
#         if len(md_doc) > 300:
#             messages = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Context Texts:\n\n{md_doc}"}
#             ]
#             try:
#                 print('API calling for id:',parent_id)
#                 res= llm.invoke(messages)
#                 print('----------Document:-----------',md_doc)
#                 print('-----------Questions------------')
#                 text= res.content
#                 print('text:',text)
#                 data= json.loads(text)
#                 print('data json:',data)
#                 parsed = SyntheticsTestSet.model_validate(data)
#                 if parsed.refusal:
#                     continue
#                 print('Parsed:',parsed)
#                 qa_pairs.extend(parsed.items)
#             except Exception as e:
#                 print('Error, Skipped for parent id:',parent_id)
#                 print(e)

#     save_to_file(qa_pairs)



async def invoke_agent(llm, prompt, md_doc, parent_id,semaphore)->QARecord:
    async with semaphore:
        t0=time.time()
        try:
            print('Calling API for Parent ID:',parent_id)
            response = await llm.ainvoke(prompt)
            text= response.content
            data= json.loads(text)
            parsed = SyntheticsTestSet.model_validate(data)
            print('Parsed:',parsed)
            print(f'API call completed for parent ID:{parent_id} in {time.time()-t0:0.2f} seconds' )
            return parsed
            # return [QARecord(
            #     user_input=item.question,
            # response=item.ground_truth,
            # retrieved_contexts=[md_doc],
            # refusal=None
            # )
            # for item in parsed.items
            # ]
        except Exception as e:
            print(f'Error: Skipped parent ID:{parent_id} in {time.time()-t0:0.2f} seconds, Info:{e}' )
            print(e)
            empty_response=SyntheticsTestSet(
                items=[],
                refusal='Insifficient text')
            return empty_response
            # return [QARecord(refusal='Insifficient text to generate question',
            #     user_input='N/A',
            #     response='N/A',
            #     retrieved_contexts=md_doc,
            # )]

async def parallel_excution():
    print('Building paralle calls')
    semaphore = asyncio.Semaphore(int(config['LLM_MAX_PARALLEL_REQUEST']))
    tasks=[]
    # print('system prompt:',system_prompt)
    for parent_id, md_doc in document_store.items():
        # print('doc md:',md_doc.get('content'))
        print('reading doc store...')
        content=md_doc['content']
        if len(content) < 300:
            print('skiping sotre shrot lenght..')
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document Texts:\n\n{content}"}
        ]
        print('Messages:',messages)
        tasks.append(
            invoke_agent(llm,messages,md_doc,parent_id,semaphore)
        )
    print('executing parallel calls')
    t0=time.time()
    results= await asyncio.gather(*tasks)
    print(f'Completed {len(results)} tasks in {time.time()-t0:0.2f} seconds')
    return results
    print('All tasks are done')

if __name__=='__main__':
    # generate_questions_sync()
    results=asyncio.run(parallel_excution())
    print('results:',results)
    qa_pairs = [ qa.model_dump(exclude='refusal')
        for r in results
        for qa in r.items
        ]
    print('Final Pairs:',qa_pairs)
    save_to_file(config['SYNTHETIC_QA_PERSIST_PATH'],qa_pairs)
