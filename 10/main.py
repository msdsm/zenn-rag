from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.evaluation import load_evaluator

def generate(question):
    # RAGで回答を得る関数, 03と全く同じ
    loader = TextLoader("./rule.md", encoding='utf8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10,
        chunk_overlap=0,
    )
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    retriever = db.as_retriever()
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
    res_1 = chain.invoke(question)
    return res_1

def evaluate(question, result, reference):
    model = ChatOpenAI()
    evaluator = load_evaluator("score_string", llm=model)
    print(f"{question=}")
    print(f"{result=}")
    print(f"{reference=}")
    eval_result_method_1 = evaluator.evaluate_strings(
        prediction=result,
        input=question,
        reference=reference,
    )
    print(f"{eval_result_method_1['score']=}")
    print(f"{eval_result_method_1['reasoning']=}")

if __name__ == "__main__":
    question = "基本勤務時間は、何時から、何時ですか？"
    result = generate(question)
    reference = "基本勤務時間は、午前9時から午後3時までです。"
    evaluate(question, result, reference)