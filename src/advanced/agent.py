from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()


import os
from advanced.embed import embed_index,retrieve,detect_query_type,load_model

def score_chunk(chunk, query):
    query_words = set(query.lower().split())
    chunk_words = set(chunk.lower().split())
    return len(query_words & chunk_words)

def get_llm(on_device = False, model_choice = "llama-3.1-8b-instant"):
    if not on_device:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model = model_choice,
            temperature=0.2,
            api_key=os.getenv("GROQ_KEY")
        )
    else:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model = model_choice,
            temperature=0.2
        )
    return llm

def build_agent(model,index,sections,conv,on_device=False,model_choice = "llama-3.1-8b-instant"):

    # Tools for agent
    @tool
    def search_paper(query):
        indices = retrieve(query,model,index,n_res=10)
        allowed_sections = detect_query_type(query,sections)
        retrieved = []
        filtered = []
        for i,item in enumerate(conv):
            if i in indices:
                retrieved.append(item)
        scored = [(doc,score_chunk(doc[0],query)) for doc in retrieved]
        scored.sort(key=lambda x:x[1],reverse=True)
        retrieved = [doc for doc,score in scored[:3]]
        filtered = []
        for t,section,src in retrieved:
            for section_ in allowed_sections:
                if section in section_:
                    filtered.append((t,section,src))
        if not filtered:
            filtered = retrieved
        prompt = f"Use only the provided cosmology context to answer the query : {query}\n\nContext:\n"
        for t in filtered:
            prompt += f"Source :{t[2]}, text : {t[0]}"
        prompt += "\nProvide a clear and brief answer.\nDo NOT add information not explicitly mentioned.\nDo NOT assume common cosmological datasets."
        return prompt
    @tool
    def summarize_section(section):
        section = section.lower().strip()
        matches = [f"[{src}] {t}" for t,sec,src in conv if section in sec.lower()]
        if not matches:
            return f"No section named '{section}' found."
        return "\n\n".join(matches)
    
    @tool
    def general_answer(query):
        llm = get_llm(on_device=on_device,model_choice=model_choice)
        response = llm.invoke(f"Answer this cosmology question concisely : {query}")
        return response.content if hasattr(response,"content") else str(response)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a cosmology research assistant with access to a few cosmology papers.
        
For specific questions about datasets, models, or results mentioned in the papers, use the tool search_paper.
For requests to summarize a section, use summarize_section.
For general background concepts not likely in the papers, use general_answer.

Always cite the source when using search_paper
         """),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
    ])
    llm = get_llm(on_device=on_device, model_choice=model_choice)
    tools = [search_paper, summarize_section, general_answer]
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=4)
  

