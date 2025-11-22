"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import constants as ct


############################################################
# 設定関連
############################################################
# 「.env」ファイルで定義した環境変数の読み込み
load_dotenv()


############################################################
# 関数定義
############################################################

def get_source_icon(source):
    """
    メッセージと一緒に表示するアイコンの種類を取得

    Args:
        source: 参照元のありか

    Returns:
        メッセージと一緒に表示するアイコンの種類
    """
    # 参照元がWebページの場合とファイルの場合で、取得するアイコンの種類を変える
    if source.startswith("http"):
        icon = ct.LINK_SOURCE_ICON
    else:
        icon = ct.DOC_SOURCE_ICON
    
    return icon


def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def get_llm_response(chat_message):
    """
    LLMからの回答取得

    Args:
        chat_message: ユーザー入力値

    Returns:
        LLMからの回答
    """
    import logging
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    logger.info(f"[DEBUG] get_llm_response開始 - mode: {st.session_state.mode}, message: {chat_message}")
    
    # LLMのオブジェクトを用意
    llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE)
    logger.info(f"[DEBUG] LLMオブジェクト作成完了")

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのプロンプトテンプレートを作成
    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # モードによってLLMから回答を取得する用のプロンプトを変更
    if st.session_state.mode == ct.ANSWER_MODE_1:
        # モードが「社内文書検索」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_DOC_SEARCH
    else:
        # モードが「社内問い合わせ」の場合のプロンプト
        question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # LLMから回答を取得する用のプロンプトテンプレートを作成
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    # 会話履歴なしでもLLMに理解してもらえる、独立した入力テキストを取得するためのRetrieverを作成
    logger.info(f"[DEBUG] Retriever存在確認: {'retriever' in st.session_state}")
    logger.info(f"[DEBUG] Retrieverタイプ: {type(st.session_state.retriever) if 'retriever' in st.session_state else 'None'}")
    
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, question_generator_prompt
    )
    logger.info(f"[DEBUG] history_aware_retriever作成完了")

    # LLMから回答を取得する用のChainを作成
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    # 「RAG x 会話履歴の記憶機能」を実現するためのChainを作成
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # LLMへのリクエストとレスポンス取得
    logger.info(f"[DEBUG] Chain実行開始 - chat_history件数: {len(st.session_state.chat_history)}")
    try:
        llm_response = chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        logger.info(f"[DEBUG] Chain実行完了 - response keys: {llm_response.keys() if llm_response else 'None'}")
        logger.info(f"[DEBUG] LLM回答: {llm_response.get('answer', 'answer key not found')[:100]}...")
        logger.info(f"[DEBUG] context件数: {len(llm_response.get('context', []))}")
        # 取得されたcontextのソースと内容を記録
        for i, doc in enumerate(llm_response.get('context', [])):
            source = doc.metadata.get('source', 'unknown')
            content_preview = doc.page_content[:150] if hasattr(doc, 'page_content') else 'no content'
            logger.info(f"[DEBUG] context[{i}]のソース: {source}")
            logger.info(f"[DEBUG] context[{i}]の内容(先頭150文字): {content_preview}...")
    except Exception as e:
        logger.error(f"[DEBUG] Chain実行エラー: {type(e).__name__}: {str(e)}")
        raise
    
    # LLMレスポンスを会話履歴に追加
    st.session_state.chat_history.extend([HumanMessage(content=chat_message), llm_response["answer"]])
    logger.info(f"[DEBUG] get_llm_response完了")

    return llm_response