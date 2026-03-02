from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Read URLs
with open("C:/Users/vanashree g hegde/OneDrive/Documents/AI_project/urls.txt", "r") as f:
    file1 = [line.strip() for line in f.readlines() if line.strip()]

print("Number of URLs:", len(file1))

# Load URL docs
url_loader = UnstructuredURLLoader(urls=file1)
url_docs = url_loader.load()

# Load PDF
pdf_loader = PyPDFLoader("C:/Users/vanashree g hegde/OneDrive/Documents/AI_project/data.pdf")
pdf_docs = pdf_loader.load()

# Combine
text = url_docs + pdf_docs
print("Total documents loaded:", len(text))

# Split into chunks
ts = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

chunks = ts.split_documents(text)
print("Total chunks created:", len(chunks))

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma DB
database = Chroma(
    persist_directory="Database",
    embedding_function=embeddings,
    collection_name="ds_interview_bot"
)
database.delete_collection() 
print("Old data cleared.")

# 3. NOW create the fresh database from your chunks
database = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="Database",
    collection_name="ds_interview_bot"
)
print("Data is ingested successfully.")