from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
from flask import send_file
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    pdf = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(filepath)

    loader = PyPDFLoader(filepath)
    pages = loader.load()

    # Set correct page numbers in metadata
    docs_with_pages = []
    for i, doc in enumerate(pages):
        doc.metadata["page"] = i + 1
        doc.metadata["source"] = pdf.filename  # Add filename to metadata
        docs_with_pages.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs_with_pages)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory="db")
    vectordb.persist()

    return jsonify({"message": f"{pdf.filename} uploaded and processed."})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")

    vectordb = Chroma(persist_directory="db", embedding_function=OpenAIEmbeddings())
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(question)
    answer = result["result"]
    sources = []

    for doc in result["source_documents"]:
        meta = doc.metadata
        filename = meta.get("source", "unknown.pdf")
        page = meta.get("page", "N/A")
        sources.append(f"{filename} (Page {page})")

    return jsonify({"answer": answer, "sources": list(set(sources))})

@app.route("/")
def serve_frontend():
    return send_file("index.html")

if __name__ == "__main__":
    app.run(debug=True)