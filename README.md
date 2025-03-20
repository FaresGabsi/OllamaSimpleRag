# PDF Processing and Retrieval System

This project processes PDF documents for semantic search and retrieval using embeddings and a vector database.

## Installation

Ensure Python is installed and set up a virtual environment:
```sh
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

Install dependencies:
```sh
pip install langchain langchain-community langchain-ollama chromadb ollama
```

Pull the required model:
```sh
ollama pull nomic-embed-text
```

## Project Structure
```
.
├── Data/                  # PDF files
├── main.py                # Main script
├── README.md              # Documentation
└── requirements.txt        # Dependencies
```

## Workflow

### Load and Process PDF
```python
from langchain_community.document_loaders import PyPDFLoader

doc_path = "./Data/BOI.pdf"
loader = PyPDFLoader(file_path=doc_path)
data = loader.load_and_split()
```

### Generate Embeddings and Store in Database
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
```

### Retrieve Relevant Chunks
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2")
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
    Generate five alternative versions of the user's query for better retrieval.
    Original question: {question}
    """
)
retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=QUERY_PROMPT)
```

### Generate Answer from Retrieved Chunks
```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the following context:
{context}
Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("What is the document about?",))
print(res)
```

## Usage
Run the script:
```sh
python main.py
```

Modify the query as needed:
```python
res = chain.invoke(input=("your question here",))
print(res)
```

## Features
- PDF text extraction
- Semantic search with vector similarity
- Query expansion for better retrieval
- LLM-generated responses
- Efficient storage using ChromaDB

## Future Improvements
- Add a web interface
- Improve chunking strategies
- Support multiple documents

## License
MIT License

