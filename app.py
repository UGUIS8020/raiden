import gradio as gr
from chatbot_engine import chat, get_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

index = get_index()

def respond(message, chat_history):
        history = ChatMessageHistory()
        for [user_message, ai_message] in chat_history:
              history.add_user_message(user_message)
              history.add_ai_message(ai_message)      

        bot_message = chat(f"""
        1. 専門知識に基づき、質問に最も関連する情報を要約して回答してください。        
        2. 回答は日本語で作成し、結論と臨床的な参考事例を簡潔に含めてください。
        3. 直接関連する情報がない場合は、最も近い情報を提供し、その旨を明示してください。        

        質問: {message}""",
            history,
            index
        )
        chat_history.append((message, bot_message))

    # 履歴の長さを制限
        MAX_HISTORY_LENGTH = 3
        if len(chat_history) > MAX_HISTORY_LENGTH:
            # インプレースで余分な履歴を削除（リストを再代入しない）
            while len(chat_history) > MAX_HISTORY_LENGTH:
                chat_history.pop(0)
            
            # history.messagesも同様に処理
            while len(history.messages) > MAX_HISTORY_LENGTH * 2:
                history.messages.pop(0)

        return "", chat_history


# with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; border: 2px solid #2c3e50; }") as demo:
with gr.Blocks(css=".gradio-container {background-color:rgb(248, 230, 199)}") as demo:    
    gr.Markdown("## 自家歯牙移植、歯牙再植について専門的に応対します")
    # 連絡先情報を追加
    gr.Markdown("## RAIDEN.v1.52 :RAIDEN.v2.0:近日公開予定")
    gr.Markdown("""
    ### チャットボットに関するご意見、ご要望は:070-6633-0363  **email**:shibuya8020@gmail.com    
    """)    

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="メッセージを入力してください", label="conversation")
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
    