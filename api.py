"""
RAIDEN API Server
既存のapp.pyに追加するAPI機能

使用方法:
1. app.pyと同じディレクトリにこのファイルを配置
2. pip install flask flask-cors
3. python api.py で単独起動、またはapp.pyから自動起動
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot_engine import chat, get_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
import time
import threading
import sys
import os

# 環境変数を読み込み
load_dotenv()

# Flask アプリケーションを作成
app = Flask(__name__)
CORS(app)  # CORS設定

# グローバル変数
index = None

def initialize_index():
    """インデックスを初期化"""
    global index
    try:
        print("🔄 Initializing RAG index...")
        index = get_index()
        print("✅ RAG index initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize index: {str(e)}")
        return False

def create_prompt(question):
    """質問用のプロンプトを生成"""
    return f"""
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

    質問: {question}

    上記の原則に従って、検索結果を最大限活用した包括的で実用的な回答を提供してください。
    """

@app.route('/api/question', methods=['POST'])
def api_question():
    """質問APIエンドポイント"""
    start_time = time.time()
    
    try:
        # リクエストデータの検証
        data = request.json
        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "error": "質問が提供されていません",
                "error_code": "MISSING_QUESTION"
            }), 400
        
        question = data['question']
        if not question.strip():
            return jsonify({
                "success": False,
                "error": "空の質問です",
                "error_code": "EMPTY_QUESTION"
            }), 400
        
        if len(question.strip()) < 2:
            return jsonify({
                "success": False,
                "error": "質問が短すぎます",
                "error_code": "QUESTION_TOO_SHORT"
            }), 400
        
        # インデックスの確認
        if index is None:
            return jsonify({
                "success": False,
                "error": "RAGインデックスが初期化されていません",
                "error_code": "INDEX_NOT_INITIALIZED"
            }), 500
        
        # 質問を処理
        history = ChatMessageHistory()
        prompt = create_prompt(question)
        
        print(f"📝 Processing question: {question[:50]}...")
        
        # RAG検索 + LLM回答を実行
        answer = chat(prompt, history, index)
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Response generated in {elapsed_time:.3f}s")
        
        return jsonify({
            "success": True,
            "answer": answer,
            "question": question,
            "response_time": round(elapsed_time, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "service": "raiden",
            "version": "2.0"
        })
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        
        print(f"❌ API Error after {elapsed_time:.3f}s: {error_msg}")
        
        return jsonify({
            "success": False,
            "error": f"処理中にエラーが発生しました: {error_msg}",
            "error_code": "PROCESSING_ERROR",
            "response_time": round(elapsed_time, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェックエンドポイント"""
    try:
        status = {
            "status": "healthy",
            "service": "raiden",
            "version": "2.0",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "index_initialized": index is not None,
            "uptime": time.time() - start_time if 'start_time' in globals() else 0
        }
        
        # インデックスの状態をチェック
        if index is None:
            status["status"] = "degraded"
            status["warnings"] = ["RAG index not initialized"]
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """API情報エンドポイント"""
    return jsonify({
        "service": "RAIDEN API",
        "description": "自家歯牙移植専門AI",
        "version": "2.0",
        "endpoints": {
            "POST /api/question": {
                "description": "質問に対する回答を取得",
                "parameters": {
                    "question": "質問内容（文字列）"
                }
            },
            "GET /api/health": {
                "description": "サービスの健全性を確認"
            },
            "GET /api/info": {
                "description": "API情報を取得"
            }
        },
        "example": {
            "request": {
                "method": "POST",
                "url": "/api/question",
                "body": {
                    "question": "40歳でも自家歯牙移植は可能ですか？"
                }
            }
        }
    })

@app.errorhandler(404)
def not_found(error):
    """404エラーハンドラー"""
    return jsonify({
        "success": False,
        "error": "エンドポイントが見つかりません",
        "error_code": "ENDPOINT_NOT_FOUND",
        "available_endpoints": ["/api/question", "/api/health", "/api/info"]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """405エラーハンドラー"""
    return jsonify({
        "success": False,
        "error": "許可されていないHTTPメソッドです",
        "error_code": "METHOD_NOT_ALLOWED"
    }), 405

@app.errorhandler(500)
def internal_error(error):
    """500エラーハンドラー"""
    return jsonify({
        "success": False,
        "error": "内部サーバーエラーが発生しました",
        "error_code": "INTERNAL_SERVER_ERROR"
    }), 500

def run_api_server(host='127.0.0.1', port=5001, debug=False):
    """APIサーバーを起動"""
    global start_time
    start_time = time.time()
    
    print(f"🚀 RAIDEN API Server starting...")
    print(f"📍 Host: {host}:{port}")
    print(f"🔗 Endpoints:")
    print(f"   • POST http://{host}:{port}/api/question - 質問API")
    print(f"   • GET  http://{host}:{port}/api/health - ヘルスチェック")
    print(f"   • GET  http://{host}:{port}/api/info - API情報")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=False,  # 重要: Gradioと併用時はreloaderを無効化
            threaded=True
        )
    except Exception as e:
        print(f"❌ Failed to start API server: {str(e)}")
        sys.exit(1)

def start_api_background(host='127.0.0.1', port=5001):
    """APIサーバーをバックグラウンドで起動"""
    api_thread = threading.Thread(
        target=run_api_server,
        args=(host, port),
        daemon=True,
        name="RAIDENAPIServer"
    )
    api_thread.start()
    
    # サーバーが起動するまで少し待機
    time.sleep(2)
    
    # ヘルスチェックで起動を確認
    try:
        import requests
        response = requests.get(f'http://{host}:{port}/api/health', timeout=5)
        if response.status_code == 200:
            print(f"✅ API Server is running at http://{host}:{port}")
            return True
        else:
            print(f"⚠️ API Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ Could not verify API server startup: {str(e)}")
        return False

if __name__ == "__main__":
    # スタンドアローンでの起動
    print("🔧 Initializing RAIDEN API...")
    
    if initialize_index():
        print("✅ Initialization complete")
        run_api_server(host='127.0.0.1', port=5001, debug=False)
    else:
        print("❌ Failed to initialize. Exiting...")
        sys.exit(1)