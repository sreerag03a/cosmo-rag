import os
import time 

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq

from advanced.agent import build_agent
from advanced.pipeline import load_or_buildIndex, load_model
from advanced.embed import retrieve,detect_query_type,load_model
from advanced.agent import score_chunk

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper


load_dotenv(os.path.join(os.getcwd(),'src','advanced','.env'))
ragas_llm = LangchainLLMWrapper(
    ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_KEY")
    )
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
)
evalpath = os.path.join(os.getcwd(),'eval')
def run_eval(on_device=False,model_choice = "llama-3.1-8b-instant"):
    with open(os.path.join(evalpath,'eval_set.json')) as f:
        eval_set = json.load(f)
        index,sections,docs_formatted = load_or_buildIndex()
        model = load_model()
        agent_executor = build_agent(model,index,sections,docs_formatted,on_device,model_choice)

        questions, answers, contexts, references = [], [], [], []

        for pair in eval_set:
            query = pair["question"]
            reference_answer = pair["answer"]

            print(f"\n Running {query[:60]}...")
            result = agent_executor.invoke({"input":query})["output"]
            indices = retrieve(query,model,index,n_res=10)
            allowed_sections = detect_query_type(query,sections)

            retrieved = []
            filtered = []
            for i,item in enumerate(docs_formatted):
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
            context_t = [t for t,sec,src in filtered]

            questions.append(query)
            answers.append(result)
            contexts.append(context_t)
            references.append(reference_answer)
            time.sleep(5)
            print(f"Answer : {result[:80]}...")

        data = Dataset.from_dict({
            "question" : questions,
            "answer" : answers,
            "contexts" : contexts,
            "reference" : references
        })
        scores = evaluate(data,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            metrics=[Faithfulness(llm = ragas_llm),AnswerRelevancy(llm = ragas_llm),ContextPrecision(llm = ragas_llm)]
            )
        
        print("\n RAGAS Scores")
        print(scores)







if __name__ == "__main__":
    run_eval(on_device=False, model_choice="llama-3.1-8b-instant")