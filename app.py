import gradio as gr
from chatbot_engine import chat, get_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from chatbot_utils import store_response_in_pinecone, search_cached_answer
from cache_manager import setup_cache_cleanup_scheduler
import time

load_dotenv()

def respond(message, chat_history):
    start_time = time.time()    

    # ChatMessageHistory オブジェクトに現在の履歴を追加
    history = ChatMessageHistory()
    for [user_message, ai_message] in chat_history:
        history.add_user_message(user_message)
        history.add_ai_message(ai_message)

      # 1. キャッシュ検索（過去回答の検索）
    cached_result = search_cached_answer(message)
    # cached_result = {"found": False}

    if cached_result.get("found"):        
        bot_message = cached_result["answer"]
        # 応答時間を計測して表示
        elapsed_time = time.time() - start_time
        # print(f"キャッシュヒット！保存済み回答を返します (応答時間: {elapsed_time:.3f}秒)")

    else:
        # 3. キャッシュヒットしなかった場合 → 新規回答を生成
        print("キャッシュヒットなし。LLMで新規回答を生成します")

        prompt = f"""
        1. 回答は日本語で行い、結論、理由、リスク、臨床的な参考事例を必ず含めてください。       
        2. 歯科医療の質問は専門性が高いため、必ずベクトル検索ツールを使用し、その結果のみを参考にして回答を作成してください。自身の知識だけで回答せず、必ずツールを使用してください。

        質問: {message}
        """

        # LLMから回答を取得
        bot_message = chat(prompt, history, index)

        # 4. 回答をPineconeに保存
        store_result = store_response_in_pinecone(message, bot_message)
        if store_result:
            print("新規回答を正常にPineconeに保存しました")

    # 5. チャット履歴を更新
    chat_history.append((message, bot_message))

    MAX_HISTORY_LENGTH = 3
    chat_history = chat_history[-MAX_HISTORY_LENGTH:]
    history.messages = history.messages[-MAX_HISTORY_LENGTH * 2:]


    # 6. チャット履歴の最大保持数を制限
    # MAX_HISTORY_LENGTH = 3
    # if len(chat_history) > MAX_HISTORY_LENGTH:
    #     while len(chat_history) > MAX_HISTORY_LENGTH:
    #         chat_history.pop(0)

        # history.messagesも同様に制限
        # while len(history.messages) > MAX_HISTORY_LENGTH * 2:
        #     history.messages.pop(0)

    return "", chat_history

# with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; border: 2px solid #2c3e50; }") as demo:
with gr.Blocks(css=".gradio-container {background-color:rgb(248, 230, 199)}") as demo:    
    # gr.Markdown("## 自家歯牙移植、歯牙再植に専門的に応答します")
    # 連絡先情報を追加
    gr.Markdown("## RAIDEN v2.0")  # バージョン番号を更新
    gr.Markdown("""
    ### Chatbotに関するご意見、ご要望は:070-6633-0363  **email**:shibuya8020@gmail.com    
    """)    

    chatbot = gr.Chatbot(autoscroll=True)
    msg = gr.Textbox(placeholder="メッセージを入力してください", label="conversation")
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    index = get_index()
    demo.launch(
        server_name="127.0.0.1",     # 外部にはバインドしない
        server_port=7860,
        share=False,                 # Gradioの外部トンネル機能を無効化
        inbrowser=False              # 自動でブラウザを開かない（サーバー用途）
    )

# if __name__ == "__main__":
#     index = get_index()
#     setup_cache_cleanup_scheduler(index,expiration_days=90)
#     demo.launch(server_name="127.0.0.1", server_port=7860)
    