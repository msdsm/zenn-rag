from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# print(os.getenv("OPENAI_API_KEY"))

loader = TextLoader("./rule.md", encoding='utf8')
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=10,
    chunk_overlap=0,
)

docs = text_splitter.split_documents(documents)
print("="*10 + "docs" + "="*10)
print(docs)
print("="*10 + "len(docs)" + "="*10)
print(len(docs))

embeddings = OpenAIEmbeddings()
db = DocArrayInMemorySearch.from_documents(docs, embeddings)
retriever = db.as_retriever()
print("="*10 + "retriever" + "="*10)
print(retriever)

question = "基本勤務時間は、何時から、何時ですか？"
print("="*10 + "retriever.invoke(question)" + "="*10)
print(retriever.invoke(question))
template = """
# ゴール
私は、就業規則の文章と質問を提供します。
あなたは、就業規則に基づいて、質問に対する回答を生成してください。
# 質問
{question}
# 就業規則
{context}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

chain = setup_and_retrieval | prompt | model | output_parser
res_1 = chain.invoke("基本勤務時間は、何時から、何時ですか？")
print("="*10 + "result" + "="*10)
print(res_1)