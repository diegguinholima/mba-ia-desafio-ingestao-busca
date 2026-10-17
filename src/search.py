import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


REQUIRED_VARS = ("OPENAI_API_KEY", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME")


def missing_vars():
    return [var for var in REQUIRED_VARS if not os.getenv(var)]


def get_store(pre_delete_collection=False):
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
        pre_delete_collection=pre_delete_collection,
    )


def search_prompt(question=None):
    if missing := missing_vars():
        print(f"Variáveis de ambiente não definidas: {', '.join(missing)}")
        return None

    try:
        store = get_store()
    except Exception as e:
        print(f"Erro ao conectar no banco vetorial: {e}")
        return None

    def build_context(pergunta):
        results = store.similarity_search_with_score(pergunta, k=10)
        contexto = "\n\n".join(doc.page_content.strip() for doc, _score in results)
        return {"contexto": contexto, "pergunta": pergunta}

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-nano"), temperature=0)

    return RunnableLambda(build_context) | prompt | llm | StrOutputParser()
