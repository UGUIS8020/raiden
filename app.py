import gradio as gr
from chatbot_engine import chat, create_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()


def respond(message, chat_history):
        history = ChatMessageHistory()
        for [user_message, ai_message] in chat_history:
              history.add_user_message(user_message)
              history.add_ai_message(ai_message)

        bot_message = chat(
    f"""Pineconeに保存されているデータから検索を行い、以下の優先順位で回答を作成してください：

        1. 質問内容に完全に一致する情報があれば、それを使用して回答してください。
        2. 完全一致がない場合でも、質問に含まれる重要なキーワードや概念に関連する情報があれば、それらを組み合わせて回答を作成してください。
        3. 部分的な情報しか見つからない場合は、見つかった情報を基に、その範囲で回答を作成してください。
        4. 関連する情報が全く見つからない場合のみ、OpenAIを使用して回答を生成してください。

        回答は必ず日本語で作成し、見つかった情報を可能な限り活用してください。質問: {message}""",
            history,
            index
        )
        chat_history.append((message, bot_message))

    # 履歴の長さを制限
        MAX_HISTORY_LENGTH = 5
        if len(chat_history) > MAX_HISTORY_LENGTH:
            chat_history = chat_history[-MAX_HISTORY_LENGTH:]
            history.messages = history.messages[-(MAX_HISTORY_LENGTH * 2):]

        return "", chat_history, history


with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; }") as demo:
    gr.Markdown("# 渋谷歯科技工所 自動応答BOT")
    gr.Markdown("# 弊社に関すること、自家歯牙移植、歯科に関するご質問にお答えします")
    # 連絡先情報を追加
    gr.Markdown("""
    ### チャットボットに関するご意見、ご要望は:070-6633-0363  **email**:shibuya8020@gmail.com    
    """)    

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="メッセージを入力してください", label="conversation")
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])



index = create_index()
demo.launch(server_name="0.0.0.0", server_port=7860)
