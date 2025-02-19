import gradio as gr
from chatbot_engine import chat, create_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

index = create_index()

def respond(message, chat_history):
        history = ChatMessageHistory()
        for [user_message, ai_message] in chat_history:
              history.add_user_message(user_message)
              history.add_ai_message(ai_message)

        bot_message = chat(
    f"""Pineconeに保存されているデータから検索を行い、以下の優先順位で回答を作成してください：

        1. 質問内容におおよそ一致する情報があれば、それを要約して回答してください。
        2. 完全一致がない場合でも、質問に含まれる重要なキーワードや概念に関連する情報があれば、それらを組み合わせて要約して回答を作成してください。
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


# with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; border: 2px solid #2c3e50; }") as demo:
with gr.Blocks(css=".gradio-container {background-color:rgb(248, 230, 199)}") as demo:
    gr.Markdown("## 渋谷歯科技工所 自動応答BOT")
    gr.Markdown("## 自家歯牙移植、歯牙再植について専門的に応対するチャットボット")
    # 連絡先情報を追加
    gr.Markdown("## 機能を追加 RAIDEN.v2:近日公開予定")
    gr.Markdown("""
    ### チャットボットに関するご意見、ご要望は:070-6633-0363  **email**:shibuya8020@gmail.com    
    """)    

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="メッセージを入力してください", label="conversation")
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    demo.launch()