import easyocr
reader =easyocr.Reader(["en"])
result =reader.readtext("work-policy-D13896.png")
for detection in result:
    ocr_text=detection[1]
    print(ocr_text,end="")

from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import base64
image_path=Path("work-policy-D13896.png")
image_data=base64.b64encode(image_path.read_bytes()).decode("utf-8")

import os
llm = ChatGroq(model="qwen/qwen3.6-27B",
               api_key=os.getenv("groq_api_key_"))

message ={
    "role":"user",
    "content":[{"type":"text",
                "text":"describe this image in detail."},
                {"type":"image_url",
                 "image_url":{"url":f"data:image/png;base64,{image_data}"}}]
}



res=llm.invoke([message])
vision_text=res.content

# combining ocr+visuion
combined_text=f"""
OCR TEXT:
{ocr_text}
visiontext:
{vision_text}
"""
print(combined_text)

# make document 
from langchain_core.documents import Document
docs=[
    Document(page_content=combined_text,metadata={"source":str(image_path)})

]
print(docs)
# print(docs[0].metadata)

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter =RecursiveCharacterTextSplitter(chunk_size=500,
                                              chunk_overlap=40)

chunks =text_splitter.split_documents(docs)
print(len(chunks))

# for i ,chunk in enumerate(chunks):
#     print(f"\nchunk{i+1}:")
#     print(chunk.page_content)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectors =embeddings.embed_documents(
    [chunk.page_content for chunk in chunks]
)
print("number of embeddings:",len(vectors))
print(len(vectors[0]))

from langchain_chroma import Chroma

vector_store =Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("embiddings stored succesfuly in database")

retriver =vector_store.as_retriever(
    search_kwargs={"k":2}
)
quetion="give me the introdution "
retrived_docs=retriver.invoke(quetion)
for i,doc in enumerate(retrived_docs):
    print(f"\nDocument{i+1}:")
    print(doc.page_content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"Answer is not available in the image."
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


question = "What is the safety policy?"

retrieved_docs = retriver.invoke(question)

context = format_docs(retrieved_docs)

messages = prompt.invoke({
    "context": context,
    "question": question
})

response = llm.invoke(messages)

print(response.content)    