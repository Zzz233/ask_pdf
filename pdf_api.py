import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from flask import Flask, jsonify, request
from langchain import ElasticVectorSearch
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
# from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open("./config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

app = Flask(__name__)  # define app using Flask
OPENAI_API_KEY = cfg["openai"]["OPENAI_API_KEY"]
ELASTICSEARCH_URL = cfg["es"]["elasticsearch_url"]
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
# llm_test = OpenAI(openai_api_key=OPENAI_API_KEY)


def _load_pdf_text(dir_path):
    loader = DirectoryLoader(dir_path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)  # TextLoader
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    return texts


@app.route('/put', methods=["POST"])
def create_or_add():
    """
    create a new es index, or add data to an existing es index
    """
    try:
        if request.headers.get("token") != 'citexs':
            return jsonify({"status": "error"})
        my_data = request.form.to_dict()
        index_name = my_data["index_name"]
        dir_path = my_data["dir_path"]
        action = my_data["action"]  # create or add
    except Exception:
        return jsonify({"status": "error"})

    try:
        texts = _load_pdf_text(dir_path)
        elastic_vector_search = ElasticVectorSearch(
            elasticsearch_url=ELASTICSEARCH_URL,
            index_name=index_name,
            embedding=embeddings,
        )
        # 新建索引
        if action == "create":
            elastic_vector_search.from_texts(
                texts=[t.page_content for t in texts],
                embedding=embeddings,
                metadatas=[t.metadata for t in texts],
                elasticsearch_url=ELASTICSEARCH_URL,
                index_name=index_name,
            )
        # 向已存在的索引中添加数据
        elif action == "add":
            elastic_vector_search.add_texts(
                texts=[t.page_content for t in texts],
                metadatas=[t.metadata for t in texts],
                index_name=index_name,
            )
        else:
            return jsonify({"status": "error"})
    except Exception as e:
        print(e)
        return jsonify({"status": "error"})

    # 递归删除目录以及所有的内容
    try:
        shutil.rmtree(dir_path)
    except Exception:
        return jsonify({"status": "error"})

    return jsonify({"status": "success"})


@app.route('/ask', methods=["POST"])
def ask_or_summarize():
    """
    summarize or ask a document
    """
    try:
        if request.headers.get("token") != 'citexs':
            return jsonify({"status": "error"})
        my_data = request.form.to_dict()
        index_name = my_data["index_name"]
        query = my_data["query"]
        filter = {"file_path": my_data["filter"]} if my_data["filter"] else None·
        action = my_data["action"]  # qa or summarize
    except Exception:
        return jsonify({"status": "error"})

    try:
        elastic_vector_search = ElasticVectorSearch(
            elasticsearch_url=ELASTICSEARCH_URL,
            index_name=index_name,
            embedding=embeddings,
        )
        if action == "qa":
            chain = load_qa_chain(llm, chain_type="stuff")
            query = query
            docs_and_scores = elastic_vector_search.similarity_search_with_score(query=query, filter=filter)
            # print(docs_and_scores)
            with get_openai_callback() as cb:
                res = chain.run(input_documents=[d[0] for d in docs_and_scores], question=query)
        # elif action == "summarize":
        #     chain = load_summarize_chain(llm, chain_type="stuff")
        #     query = "Summarise introduction of this paper"
        #     docs_and_scores = elastic_vector_search.similarity_search_with_score(query=query)
        #     with get_openai_callback() as cb:
        #         res = chain.run([d[0] for d in docs_and_scores])
        else:
            return jsonify({"status": "error"})
    except Exception:
        return jsonify({"status": "error"})

    return jsonify(
        {"status": "success",
         "result": res,
         "total_tokens": cb.total_tokens,
         "prompt_tokens": cb.prompt_tokens,
         "completion_tokens": cb.completion_tokens,
         "total_cost": cb.total_cost}
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086, debug=True)