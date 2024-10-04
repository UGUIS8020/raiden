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

        bot_message = chat(message, history, index)
        bot_message = chat(f"大規模言語モデルchatGPTのあなたは渋谷歯科技工所の代表渋谷正彦を演じてください。日本語で答えてください{message}", history, index)
        chat_history.append((message, bot_message))

    # 履歴の長さを制限
        MAX_HISTORY_LENGTH = 5
        if len(chat_history) > MAX_HISTORY_LENGTH:
            chat_history = chat_history[-MAX_HISTORY_LENGTH:]
            history.messages = history.messages[-(MAX_HISTORY_LENGTH * 2):]

        return "", chat_history, history


with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; }") as demo:
    gr.Markdown("# 渋谷歯科技工所 自動応答BOT TEST")
    gr.Markdown("# 弊社に関すること、歯科に関するご質問にお答えします")
     # 連絡先情報を追加
    gr.Markdown("""
    ### チャットボットに関するご意見、ご要望は:070-6633-0363  **email**:shibuya8020@gmail.com    
    """)
    gr.Markdown("""### ただいまメモリを最小限にしてテスト運用中ですので反応が遅いですがご了承ください""")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="メッセージを入力してください", label="conversation")
    clear = gr.ClearButton([msg, chatbot])   

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# インデックスの作成、読み込み、または更新
# 既存のインデックスを使用する場合
index = create_index()  

# 新しいデータを追加する場合
# index = create_index(add_new_data=True) 


demo.launch()