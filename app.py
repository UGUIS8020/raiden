import gradio as gr
from chatbot_engine import chat, get_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from cache_manager import setup_cache_cleanup_scheduler
from prompts import SYSTEM_PROMPT_TEMPLATE
from chatbot_utils import search_cached_answer, store_response_in_qdrant
import time

load_dotenv()

# キャッシュ機能の切り替え（True: 有効 / False: 無効）
CACHE_ENABLED = True

index = None

def respond(message, chat_history):
    global index
    start_time = time.time()
    
    # 入力検証
    if not message or not message.strip():
        return "", chat_history
    
    if len(message) > 5000:
        error_msg = "申し訳ありませんが、質問が長すぎます。5000文字以内でお願いします。"
        return "", chat_history + [("", error_msg)]
    
    try:
        # ChatMessageHistory オブジェクトに現在の履歴を追加
        history = ChatMessageHistory()
        for [user_message, ai_message] in chat_history:
            history.add_user_message(user_message)
            history.add_ai_message(ai_message)

        # キャッシュ検索
        cached_result = search_cached_answer(message) if CACHE_ENABLED else {"found": False}

        if cached_result.get("found"):
            bot_message = cached_result["answer"]
            elapsed_time = time.time() - start_time
            print(f"✅ キャッシュヒット (応答時間: {elapsed_time:.2f}秒)")
        else:
            # 新規回答を生成
            prompt = SYSTEM_PROMPT_TEMPLATE.format(question=message)
            bot_message = chat(prompt, history, index)

            elapsed_time = time.time() - start_time
            print(f"✅ 回答生成完了 (応答時間: {elapsed_time:.2f}秒)")

            # 新規回答をキャッシュに保存
            if CACHE_ENABLED:
                store_response_in_qdrant(message, bot_message)
    
    except Exception as e:
        print(f"❌ エラー発生: {str(e)}")
        import traceback
        traceback.print_exc()
        
        bot_message = "申し訳ありません。エラーが発生しました。もう一度お試しください。"

    # チャット履歴を更新
    chat_history.append((message, bot_message))
    
    MAX_HISTORY_LENGTH = 3
    chat_history = chat_history[-MAX_HISTORY_LENGTH:]
    history.messages = history.messages[-MAX_HISTORY_LENGTH * 2:]

    return "", chat_history

# with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; border: 2px solid #2c3e50; }") as demo:
with gr.Blocks(css=".gradio-container {background-color:rgb(248, 230, 199)}") as demo:    
    # gr.Markdown("## 自家歯牙移植、歯牙再植に専門的に応答します")
    # 連絡先情報を追加
    gr.Markdown("## RAIDEN v2.0 (Qdrant版20251205_2025)")  # バージョン番号を更新
    gr.Markdown("""
    ### Chatbotに関するご意見、ご要望は:070-6633-0363  **email**:shibuya8020@gmail.com
     
    """)    

    chatbot = gr.Chatbot(autoscroll=True)
    msg = gr.Textbox(placeholder="メッセージを入力してください", label="conversation")
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    # RAGインデックスを初期化
    print("🔄 Initializing RAG index...")
    index = get_index()
    print("✅ RAG index initialized")
    
    # キャッシュクリーンアップスケジューラーを起動
    print("🧹 Starting cache cleanup scheduler...")
    setup_cache_cleanup_scheduler()
    print("✅ Cache cleanup scheduler started")
    
    # ========== API機能を起動（オプション） ==========
    try:
        print("🔧 Starting API server...")
        from api import start_api_background
        
        # APIモジュールにindexを渡す
        import api
        api.index = index
        
        # APIサーバーをバックグラウンドで起動
        api_started = start_api_background(host='0.0.0.0', port=5001)
        
        if api_started:
            print("🎉 Gradio UI + API Server both running!")
            print("💬 Gradio UI: http://127.0.0.1:7860")
            print("🔗 API: http://127.0.0.1:5001/api/question")
        else:
            print("⚠️ API Server startup failed, continuing with Gradio only...")
            
    except ImportError:
        print("📝 api.py not found - running Gradio only...")
        print("💡 To enable API: create api.py and install 'flask flask-cors'")
    except Exception as e:
        print(f"⚠️ API startup error: {str(e)}")
        print("📝 Continuing with Gradio only...")
    
    # ========== Gradioインターフェースの起動 ==========
    print("🚀 Starting Gradio interface...")
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",     # 外部にはバインドしない
        # server_name="0.0.0.0",
        server_port=7860,
        share=False,                 # Gradioの外部トンネル機能を無効化
        inbrowser=False              # 自動でブラウザを開かない（サーバー用途）
    )