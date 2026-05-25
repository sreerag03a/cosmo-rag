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
        """
        Search the available papers for relevant chunks to the user query. Use this tool for questions specific to the work
        like about datasets, models or results. Returns relevant chunks and the source it is from.
        """
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
        # prompt = f"Use only the provided cosmology context to answer the query : {query}\n\nContext:\n"
        prompt = ""
        for t in filtered:
            prompt += f"Source :{t[2]}, text : {t[0]}"
        # prompt += "\nProvide a clear and brief answer.\nDo NOT add information not explicitly mentioned.\nDo NOT assume common cosmological datasets."
        return prompt
    @tool
    def summarize_section(section):
        """
        Returns the full text of a named section (eg. 'Results', 'Method', 'Introduction', etc.). Use this tool when the user asks
        to summarize or explain a whole section.
        """
        section = section.lower().strip()
        matches = [f"[{src}] {t}" for t,sec,src in conv if section in sec.lower()]
        if not matches:
            return f"No section named '{section}' found."
        return "\n\n".join(matches)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a cosmology research assistant with access to a corpus of cosmology papers.

TOOL SELECTION RULES — follow these strictly:
- Use search_papers for ANY question about specific facts, datasets, models, methods, or results.
- Use summarize_section ONLY when the user explicitly says "summarize" or "overview of the [section] section".
- For general background concepts, answer directly from your own knowledge without using any tool.
- If a tool returns no results, answer with what you know and DO NOT retry with a different tool.

RESPONSE RULES — follow these strictly:
- Never mention tools, function calls, or internal processes in your response.
- Never say things like "the function returned..." or "based on the retrieved chunks...".
- Just answer the question directly and concisely as if you already knew the answer."""),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
    ])
    llm = get_llm(on_device=on_device, model_choice=model_choice)
    tools = [search_paper, summarize_section]
    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6)
  

