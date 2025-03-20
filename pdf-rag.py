## 1. Ingest pdf files
## 2. Extract text from pdf files and split into smaller chunks
## 3. Send the chunks to the embedding model
## 4. Save the embeddings to a vector databse
## 5. Perform similarity search on the vector database to find similar
## 6. retrieve the similar documents and present them to the user

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path="./Data/BOI.pdf"
model="llama3.2"

# Load the pdf file
if doc_path:
    loader= PyPDFLoader(file_path=doc_path)
    data= loader.load_and_split()
    print("Data loaded successfully")
else:
    print("No document path provided")

# #Preview first page
# content= data[0].page_content
# print(content[:1000])
# ============= END OF PDF INGESTION =============


# Extract text from the pdf file and split into smaller chunks
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Split and chunk
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300) # chunk overlap is used to ensure that the chunks are not split in the middle of a word
chunks= text_splitter.split_documents(data)
print("done splitting...")

# print(f"Number of chunks: {len(chunks)}")
# # Preview the first chunk
# print(f"Example Chunk : {chunks[0]}")

# ============= ADD Vector DataBase =============
import ollama
ollama.pull("nomic-embed-text")

vector_db=Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database...")

# ====== Retrieval =========
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm= ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever= MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm,prompt=QUERY_PROMPT
)
print(retriever)
template= """Answer the question based ONLY on the following context:
{context}
Question: {question}"""

prompt= ChatPromptTemplate.from_template(template)

chain= (
{"context": retriever, "question": RunnablePassthrough()}
| prompt
| llm
| StrOutputParser()
)

res= chain.invoke(input=("what is the document about?",))

print(res)