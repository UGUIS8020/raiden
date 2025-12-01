import gradio as gr
from chatbot_engine import chat, get_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from chatbot_utils import store_response_in_qdrant, search_cached_answer
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
        print(f"キャッシュヒット！保存済み回答を返します (応答時間: {elapsed_time:.3f}秒)")

    else:
        # 3. キャッシュヒットしなかった場合 → 新規回答を生成
        prompt = f"""
        あなたは自家歯牙移植の専門知識を持つ歯科医師として回答してください。

        ## 基本原則
        1. 回答は日本語で行い、医学的正確性を最優先としてください
        2. 必ずベクトル検索ツールを使用し、検索結果のみを参考に回答してください
        3. 検索結果の上位文書から重要な情報（数値データ、症例、図表情報）を見逃さないよう特に注意してください
        4. 文書に記載がない情報については「検索した文書には記載されていません」と明記してください

        ## 回答構成（該当する要素を含めてください）
        - **直接的な回答**: 質問に対する明確な答え
        - **詳細な説明**: メカニズム、手順、基準など
        - **根拠となるデータ**: 数値、期間、成功率、症例データなど
        - **臨床的意義**: 実践的な応用や注意点
        - **リスクや合併症**: 該当する場合
        - **参考症例**: 具体的な事例がある場合

        ## 情報の優先順位
        【最優先】数値データ、図表付き症例、実験結果
        【高優先】メカニズム、診断基準、治療方法、合併症
        【標準】定義、分類、一般的説明

        ## 複合質問の処理
        質問に複数の要素が含まれる場合：
        **STEP 1**: 質問を個別要素に分解
        **STEP 2**: 各要素について上記の構成で回答
        **STEP 3**: 要素間の関連性があれば説明

        ## 専門用語への配慮
        - 重要な専門用語には簡潔な説明を付加
        - 略語の正式名称を併記
        - 患者説明に使える表現も提供（適切な場合）

        ## 専門性向上のための指示
        - 検索結果に含まれる専門用語は積極的に使用
        - メカニズムの詳細説明を重視
        - 臨床的意義や病態との関連も含める
        - 段階的なプロセスは番号付きで整理       

        質問: {message}

        上記の原則に従って、検索結果を最大限活用した包括的で実用的な回答を提供してください。
        """

        # LLMから回答を取得
        bot_message = chat(prompt, history, index)

        # 4. 回答をQdrantに保存
        store_result = store_response_in_qdrant(message, bot_message)
        if store_result:
            print("新規回答を正常にQdrantに保存しました")

    # 5. チャット履歴を更新
    chat_history.append((message, bot_message))

    MAX_HISTORY_LENGTH = 3
    chat_history = chat_history[-MAX_HISTORY_LENGTH:]
    history.messages = history.messages[-MAX_HISTORY_LENGTH * 2:]

    return "", chat_history

# with gr.Blocks(css=".custom-textbox { width: 100%; height: 100px; border: 2px solid #2c3e50; }") as demo:
with gr.Blocks(css=".gradio-container {background-color:rgb(248, 230, 199)}") as demo:    
    # gr.Markdown("## 自家歯牙移植、歯牙再植に専門的に応答します")
    # 連絡先情報を追加
    gr.Markdown("## RAIDEN v2.0 (Qdrant版)")  # バージョン番号を更新
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
    demo.launch(
        server_name="127.0.0.1",     # 外部にはバインドしない
        # server_name="0.0.0.0",
        server_port=7860,
        share=False,                 # Gradioの外部トンネル機能を無効化
        inbrowser=False              # 自動でブラウザを開かない（サーバー用途）
    )