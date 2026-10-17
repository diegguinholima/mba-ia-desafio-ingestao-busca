# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e busca semântica de um PDF usando LangChain, OpenAI e PostgreSQL + pgVector.

- `src/ingest.py`: lê o `document.pdf`, divide em chunks de 1000 caracteres (overlap de 150), gera os embeddings e salva no Postgres (pgVector).
- `src/search.py`: monta a chain de busca: vetoriza a pergunta, busca os 10 chunks mais relevantes (`similarity_search_with_score(k=10)`), monta o prompt e chama a LLM.
- `src/chat.py`: CLI para conversar com o conteúdo do PDF.

## Modelos utilizados (OpenAI)

- Embeddings: `text-embedding-3-small`
- LLM: `gpt-4.1-nano` (com `temperature=0`)

Ambos podem ser trocados pelas variáveis `OPENAI_EMBEDDING_MODEL` e `OPENAI_LLM_MODEL` no `.env`.

> Se trocar o modelo de embeddings depois da primeira ingestão, apague o volume do banco (`docker compose down -v`) e rode a ingestão de novo, pois a dimensão dos vetores muda.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Uma API Key da OpenAI

## Como executar

### 1. Criar e ativar o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar as variáveis de ambiente

Copie o template e preencha a sua `OPENAI_API_KEY`:

```bash
cp .env.example .env
```

| Variável | Descrição |
| --- | --- |
| `OPENAI_API_KEY` | Sua API Key da OpenAI |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings (padrão `text-embedding-3-small`) |
| `OPENAI_LLM_MODEL` | Modelo de chat (padrão `gpt-4.1-nano`) |
| `DATABASE_URL` | Conexão com o Postgres (já configurada para o `docker-compose.yml`) |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector |
| `PDF_PATH` | Caminho do PDF, relativo à raiz do projeto (padrão `document.pdf`) |

### 4. Subir o banco de dados

```bash
docker compose up -d
```

### 5. Executar a ingestão do PDF

```bash
python src/ingest.py
```

A ingestão recria a collection a cada execução, então pode ser rodada mais de uma vez sem duplicar os dados.

### 6. Rodar o chat

```bash
python src/chat.py
```

Digite `sair` (ou `Ctrl+C`) para encerrar.

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da empresa Alfa Energia S.A.?
RESPOSTA: R$ 722.875.391,46

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```
