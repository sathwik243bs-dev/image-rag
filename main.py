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

for i ,chunk in enumerate(chunks):
    print(f"\nchunk{i+1}:")
    print(chunk.page_content)