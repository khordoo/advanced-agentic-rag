import yaml
from pathlib import Path
from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from dotenv import dotenv_values
from schemas import PromptConfig

config = dotenv_values()

class PromptLoader:
    PROMPTA_DIR=Path('prompts')
    @classmethod
    def load(cls,name:str,**kwargs)->PromptConfig:
        file_path=cls.PROMPTA_DIR / f"{name}.yaml"
        with open(file_path,'r',encoding='utf-8') as f:
             raw_data=yaml.safe_load(f)
        #parse for validation
        config = PromptConfig(**raw_data)
        config.template = config.template.format(**kwargs)
        return config

class AgentFactory:
    DEFAULT_TEMP=0.2
    DEFAULT_AWS_MODEL = 'amazon.nova-micro-v1:0'
    DEFAULT_KIMI_MODEL = 'moonshotai/kimi-k2.5'
    DEFAULT_OLLAMA_MODEL='gemma4:e2b'
    @classmethod
    def create(cls,provider:str, temperature:float=DEFAULT_TEMP, model_name=None)->BaseChatModel:
        provider=provider.lower().strip()
        #Using lamdba for lazy initialization
        builders= {
          'aws': lambda:ChatBedrockConverse(

               model=model_name or cls.DEFAULT_AWS_MODEL,
               region_name="us-east-1",
               temperature=temperature,
          ),
        'lmstudio': lambda: ChatOpenAI(
                model=model_name or 'dummy', #name is irrelevent in lmstudio
                base_url=config['OPEN_AI_COMPATIBLE_BASE_URL'],
                temperature=temperature
            ),
        'ollama': lambda: ChatOllama(
                model=model_name or cls.DEFAULT_OLLAMA_MODEL, #name is irrelevent in lmstudio
                base_url=config['OLLAMA_BASE_URL'],
                temperature=temperature
            ),
        'kimi':lambda: ChatOpenAI(
                model=model_name or cls.DEFAULT_KIMI_MODEL,
                base_url=config['NVIDIA_KIMI_K2_BASE_URL'],
                api_key=config['NVIDIA_KIMI_API_KEY'],
                temperature=temperature
            )
        }
        if provider not in builders:
           raise ValueError(f'UNknown provider:{provider}')

        return builders[provider]()

    @classmethod
    def get_judge_model(cls, provider='aws')->BaseChatModel:
        return cls.create(provider,model_name='amazon.nova-pro-v1:0',temperature=0)

# llm=AgentFactory.create('aws')
# print(type(llm))
# res=llm.invoke('hi')
# print(res.content)
if __name__=='__main__':
    # llm=AgentFactory.get_judge_model()
    # print(llm.invoke('hi'))
    # p=PromptLoader.load('qa_gen',schema='{items:[]}',context='fdd')
    # print(p)
    # llm_judge = AgentFactory.create('lmstudio', temperature=0)
    # print(llm_judge.invoke("hi"))
    llm_judge = AgentFactory.create('ollama', temperature=0)
    print(llm_judge.invoke("hi"))
