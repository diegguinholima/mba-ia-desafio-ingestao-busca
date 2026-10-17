import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_store, missing_vars

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH", "document.pdf")


def resolve_pdf_path():
    # Caminhos relativos são resolvidos a partir da raiz do projeto
    path = Path(PDF_PATH)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path.resolve()


def ingest_pdf():
    if missing := missing_vars():
        raise RuntimeError(f"Variáveis de ambiente não definidas: {', '.join(missing)}")

    pdf_path = resolve_pdf_path()
    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)

    if not splits:
        print("Nenhum conteúdo encontrado no PDF.")
        return

    # Recria a collection a cada ingestão para não duplicar chunks
    store = get_store(pre_delete_collection=True)
    store.add_documents(splits)

    print(f"Ingestão concluída: {len(splits)} chunks de {pdf_path.name} salvos.")


if __name__ == "__main__":
    ingest_pdf()
