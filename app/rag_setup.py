import os

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def build_vectorstore():
    # 1. Load the PDFs (metadata keeps the source path and page number)
    docs = PyPDFDirectoryLoader(DOCUMENTS_DIR, recursive=True).load()

    # tag each chunk with its area
    for d in docs:
        d.metadata["area"] = d.metadata["source"].split("/")[-2]

    # 2. Chunk
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)

    # 3. Embed and store, persisted on disk
    return Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(api_key=OPENAI_API_KEY),
        persist_directory=CHROMA_DIR,
    )


def load_vectorstore():
    # Reload without re-embedding
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings(api_key=OPENAI_API_KEY))


# Build only when there is no saved index, so running again does not duplicate chunks
vs = load_vectorstore() if os.path.isdir(CHROMA_DIR) else build_vectorstore()
retriever = vs.as_retriever(search_kwargs={"k": 8})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

memory = ConversationBufferWindowMemory(k=5, memory_key="history", return_messages=False)


def get_history(_):
    return memory.load_memory_variables({})["history"]


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the IronStore enterprise assistant. Answer ONLY using the context below. "
     "If the answer is not in the context, say you don't know. "
     "Cite the source file name for each fact.\n\n"
     "Use the chat history only to understand what the user is referring to in follow-up "
     "questions (for example 'it', 'that', 'and for part-time staff?'). "
     "Facts in your answer must still come from the context, not from the history.\n\n"
     "Chat history:\n{history}\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    parts = []
    for d in docs:
        name = os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page")
        label = f"{name}, p.{page + 1}" if isinstance(page, int) else name
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n".join(parts)


rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough(),
     "history": get_history}
    | prompt
    | llm
    | StrOutputParser()
)


def ask(question):
    answer = rag_chain.invoke(question)
    memory.save_context({"input": question}, {"output": answer})
    return answer


def run(questions):
    for q in questions:
        print(f"Q: {q}\nA: {ask(q)}\n{'-'*60}")
