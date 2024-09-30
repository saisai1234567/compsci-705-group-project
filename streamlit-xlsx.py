import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag0 import ChatXLSX
import pandas as pd
st.set_page_config(page_title="ChatXLSX")


def read_and_save_file():
    st.session_state["assistant"].clear()  # 清除之前的文档内容
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    if st.session_state["file_uploader"]:
        for file in st.session_state["file_uploader"]:
            file_name = file.name.lower()
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            # 处理 .xlsx 文件
            if file_name.endswith(".xlsx"):
                try:
                    # 读取 Excel 文件
                    data = pd.read_excel(file_path)
                    st.session_state["excel_data"] = data

                    # 在 Streamlit 应用中输出数据
                    st.write(data)

                    # 成功加载文件后使用 ChatXLSX
                    st.success(f"Successfully loaded {file.name} as an Excel file!")
                except Exception as e:
                    st.error(f"Error loading .xlsx file: {e}")

            os.remove(file_path)  # 删除临时文件

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            # 检查是否上传了文件内容
            if st.session_state.get("excel_data") is not None:
                # 根据上传的 Excel 文件内容进行处理
                excel_data = st.session_state["excel_data"]
                result = excel_data[excel_data["tweet"].str.contains(user_text, case=False, na=False)]

                if not result.empty:
                    agent_text = f"Found matching data: {result.to_dict(orient='records')}"
                else:
                    # 使用 Ollama 的 Mistral 模型回答没有文档的普通问题
                    agent_response = st.session_state["assistant"].model.invoke(user_text)
                    agent_text = agent_response.content  # 直接访问 content 属性
            else:
                # 使用 Ollama 的 Mistral 模型回答没有文档的普通问题
                agent_response = st.session_state["assistant"].model.invoke(user_text)
                agent_text = agent_response.content  # 直接访问 content 属性

        # 确保返回的是字符串
        if not isinstance(agent_text, str):
            agent_text = str(agent_text)

        # 显示用户和代理的消息
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

        # 清空输入框
        st.session_state["user_input"] = ""

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatXLSX()

    st.header("ChatXLSX")

    st.subheader("Upload an Excel file")
    st.file_uploader(
        "Upload document",
        type=["xlsx"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()