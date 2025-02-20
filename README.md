# サクッと始めるRAG開発
- https://zenn.dev/umi_mori/books/llm-rag-langchain-python/viewer/intro


## 構成
- 03 : 簡単なRAG
- 10 : langchain evaluationによるRAGの評価
## 文脈内学習(In-Context Learning)


- プロンプトの末尾に検索拡張した知識を挿入すること
```
# ゴール
私は、就業規則の文章を提供します。
あなたは、就業規則に基づいて、質問に対する回答を生成してください。

# 質問
基本勤務時間は、何時から、何時ですか？

# 就業規則
・この就業規則は、株式会社Galirage（以下、「当社」という）の全従業員に適用され、
　従業員が遵守すべき基本的なルールと指針を定めるものです。
・基本勤務時間は、午前9時から午後6時までとします。休憩時間は12時から1時までの1時間です。
・時間外労働については、事前に上司の承認を必要とします。
・定休日は土曜日、日曜日、および国が定める祝日です。
・年次有給休暇は、入社半年後に10日間付与されます。以後、勤続年数に応じて加算されます。
```
- 就業規則の文章をプロンプトの中に挿入することで知識を拡張して文脈(プロンプト)の中で学習している
- RAGはこの文脈内学習をするための知識検索機能

## RAG実装
- ローカルで実装する際のベクトルDBの選択肢は以下
  - ChromaDB
  - Faiss
  - DocArray
- クラウド環境でRAGを実装する際の選択肢は以下
  - Azure AI Search (Azure)
  - Amazon Kendra (AWS)
  - Amazon Bedrock Knowledge Bases (AWS)
  - Vertex AI Search (GCP)

## RAGの大分類
- Document RAG : ドキュメントに対してベクトル検索して文章生成をするRAG手法
- SQL RAG : LLMを用いてプロンプトからSQLクエリを生成して検索を行い文章生成をするRAG手法
- Graph RAG

## Ragasとは
- RAG Assessmentの略
- LLMによるRAGの評価フレームワーク