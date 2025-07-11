import gradio as gr
from chatbot_engine import chat, get_index
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from chatbot_utils import store_response_in_pinecone, search_cached_answer, get_separated_adaptive_response
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
        elapsed_time = time.time() - start_time
        print(f"キャッシュヒット！保存済み回答を返します (応答時間: {elapsed_time:.3f}秒)")

    else:
        print("キャッシュヒットなし。AIアダプティブシステムで新規回答を生成します")
        
        try:
            # 高度版AIアダプティブシステムを使用
            bot_message = get_separated_adaptive_response(message, chat, history, index)
            
        except Exception as e:
            print(f"AIアダプティブシステムでエラー発生: {e}")
            print("フォールバック: 標準プロンプトで回答生成します")
            
            # フォールバック用の標準プロンプト（元のプロンプトを使用）
            fallback_prompt = f"""
            あなたは経験豊富な歯科医師として、以下の質問に専門的かつ実践的に回答してください。

            ## 必須要件
            1. **必ずベクトル検索ツールを使用**し、検索結果に基づいて回答を構成してください
            2. 自身の知識のみでは回答せず、検索結果を主要な情報源として活用してください
            3. 回答は日本語で行い、臨床現場で即座に活用できる内容にしてください

            ## 回答構成（以下の項目を必ず含めてください）
            ### 1. 【結論・要点】
            - 質問に対する明確で簡潔な結論
            - 最も重要なポイントを3つ以内で要約

            ### 2. 【科学的根拠・理論的背景】
            - なぜその方法・考え方が正しいのかの根拠
            - 関連する解剖学的・生理学的知識
            - 可能であれば文献や研究結果への言及

            ### 3. 【具体的な実践方法】
            - 臨床で実際に行う手技・手順
            - 使用する器具や材料の選択基準
            - 成功のための具体的なコツやテクニック

            ### 4. 【注意点・リスク管理】
            - 避けるべき操作や手技
            - 起こりうる合併症とその予防法
            - 失敗例から学ぶべきポイント

            ### 5. 【臨床的な参考事例・応用】
            - 典型的な症例での適用例
            - 特殊なケースでの対応法
            - 他の治療法との比較・選択基準

            ### 6. 【追加的な臨床情報】
            - 患者への説明のポイント
            - 術後管理や経過観察のポイント
            - 最新の動向や今後の展望（該当する場合）

            ## 回答スタイル
            - 専門用語は適切に使用し、必要に応じて簡潔な説明を併記
            - 箇条書きと段落を適切に使い分け、読みやすい構成にする
            - 重要なポイントは**太字**で強調する
            - 数値や具体的な基準がある場合は明記する

            質問: {message}
            """
            
            bot_message = chat(fallback_prompt, history, index)

        # 4. 回答をPineconeに保存
        store_result = store_response_in_pinecone(message, bot_message)
        if store_result:
            print("新規回答を正常にPineconeに保存しました")

    # 5. チャット履歴を更新
    chat_history.append((message, bot_message))

    # 6. チャット履歴の最大保持数を制限
    MAX_HISTORY_LENGTH = 3
    chat_history = chat_history[-MAX_HISTORY_LENGTH:]
    history.messages = history.messages[-MAX_HISTORY_LENGTH * 2:]

    return "", chat_history

# 高度版AIアダプティブシステムの実装
def get_adaptive_response(message, chat_function, history, index):
    """高度版AIアダプティブシステムのメイン関数"""
    
    try:
        # Step 1: AIによる質問分析
        print("🤖 AIによる質問分析を実行中...")
        analysis_prompt = f"""
以下の歯科医療に関する質問を専門的に分析し、最適な回答スタイルを判定してください。

質問: 「{message}」

以下の観点で詳細に分析してください：

## 1. 質問タイプの分類
- **DEFINITION**: 用語の定義、基本概念、「〜とは」系の質問
- **METHOD**: 具体的方法、手技、治療手順、「〜するには」系の質問
- **COMPARISON**: 比較検討、選択基準、「〜と〜の違い」系の質問
- **CLINICAL**: 症例対応、診断、治療計画、実際の臨床応用に関する質問

## 2. 必要な詳細度レベル
- **BRIEF**: 基本的な定義や概要のみで十分（2-3段落）
- **STANDARD**: 適度な説明と実用的な情報が必要（4-5段落）
- **DETAILED**: 包括的で専門的な詳細情報が必要（6段落以上）

必ず以下の形式で回答してください：
タイプ: [DEFINITION/METHOD/COMPARISON/CLINICAL]
詳細度: [BRIEF/STANDARD/DETAILED]
理由: [判定の根拠を具体的に1-2文で説明]
"""
        
        # AIによる分析実行（ベクトル検索は使わない）
        analysis_result = chat_function(analysis_prompt, [], None)
        print(f"📊 分析結果: {analysis_result}")
        
        # Step 2: 分析結果の解析
        question_type, detail_level = parse_analysis_result(analysis_result)
        print(f"✅ 判定結果 - タイプ: {question_type}, 詳細度: {detail_level}")
        
        # Step 3: 最適化されたメインプロンプトを生成
        main_prompt = generate_optimized_prompt(message, question_type, detail_level)
        
        # Step 4: 最適化されたプロンプトで最終回答を生成
        print("🎯 最適化されたプロンプトで回答生成中...")
        final_response = chat_function(main_prompt, history, index)
        
        return final_response
        
    except Exception as e:
        print(f"❌ AIアダプティブシステムでエラー: {e}")
        raise e  # エラーを上位に伝播

def parse_analysis_result(analysis_result):
    """AI分析結果を解析して質問タイプと詳細度を抽出"""
    
    result_text = analysis_result.upper()
    
    # 質問タイプの判定
    question_type = "standard"
    if "DEFINITION" in result_text:
        question_type = "definition"
    elif "METHOD" in result_text:
        question_type = "method"
    elif "COMPARISON" in result_text:
        question_type = "comparison"
    elif "CLINICAL" in result_text:
        question_type = "clinical"
    
    # 詳細度レベルの判定
    detail_level = "standard"
    if "BRIEF" in result_text:
        detail_level = "brief"
    elif "DETAILED" in result_text:
        detail_level = "detailed"
    
    return question_type, detail_level

def generate_optimized_prompt(message, question_type, detail_level):
    """質問タイプと詳細度に基づいて最適化されたプロンプトを生成"""
    
    base_prompt = f"""
あなたは経験豊富な歯科医師として、以下の質問に専門的に回答してください。

## 必須要件
- 必ずベクトル検索ツールを使用し、検索結果に基づいて回答してください
- 回答は日本語で行ってください
- 質問タイプ: {question_type.upper()}
- 詳細度レベル: {detail_level.upper()}

質問: {message}

"""
    
    # 質問タイプと詳細度の組み合わせに応じた回答構成
    response_structures = {
        # DEFINITION タイプ
        ("definition", "brief"): """
## 回答構成【定義・簡潔モード】
1. **基本定義**: 用語の正確で簡潔な定義
2. **主要特徴**: 最も重要な特徴2-3点
3. **臨床的意義**: なぜ重要なのか

※2-3段落で要点を絞って回答してください。
""",
        ("definition", "standard"): """
## 回答構成【定義・標準モード】
1. **基本定義**: 用語の正確な定義と概念
2. **分類・種類**: 主な分類や種類
3. **特徴・機能**: 重要な特徴や機能
4. **臨床的意義**: 診療での重要性
5. **関連事項**: 関連する重要な情報

※4-5段落で体系的に回答してください。
""",
        ("definition", "detailed"): """
## 回答構成【定義・詳細モード】
1. **基本定義**: 用語の正確で包括的な定義
2. **歴史的背景**: 概念の発展や変遷
3. **分類・種類**: 詳細な分類と各特徴
4. **メカニズム**: 作用機序や原理
5. **臨床的意義**: 診療での重要性と応用
6. **最新の知見**: 近年の研究や動向

※6段落以上で学術的に詳細に回答してください。
""",
        
        # METHOD タイプ
        ("method", "brief"): """
## 回答構成【方法・簡潔モード】
1. **核心ポイント**: 成功の最重要要素
2. **基本手順**: 主要なステップ
3. **注意点**: 最も重要な注意事項

※実用的な要点を3段落程度で回答してください。
""",
        ("method", "standard"): """
## 回答構成【方法・標準モード】
1. **成功の要点**: 最重要ポイント
2. **基本原理**: 理論的根拠
3. **具体的手順**: 実際のステップ
4. **重要なコツ**: 成功のテクニック
5. **注意・リスク**: 避けるべき点

※実践的な内容を4-5段落で回答してください。
""",
        ("method", "detailed"): """
## 回答構成【方法・詳細モード】
1. **成功の核心**: 最重要成功要因
2. **理論的基盤**: 科学的根拠と原理
3. **詳細手順**: ステップバイステップの方法
4. **高度なテクニック**: 経験に基づくコツ
5. **リスク管理**: 合併症予防と対処法
6. **症例応用**: 具体的な臨床応用例
7. **トラブルシューティング**: 問題発生時の対応

※臨床で即座に活用できる詳細な内容で回答してください。
""",
        
        # COMPARISON タイプ
        ("comparison", "brief"): """
## 回答構成【比較・簡潔モード】
1. **主な違い**: 最重要な相違点
2. **選択基準**: 基本的な使い分け
3. **推奨**: 一般的な推奨事項

※判断に必要な要点を3段落程度で回答してください。
""",
        ("comparison", "standard"): """
## 回答構成【比較・標準モード】
1. **比較概要**: 主要な違いの整理
2. **各選択肢の特徴**: 利点・欠点
3. **選択基準**: 使い分けの判断基準
4. **実際の適用**: 臨床での使い分け例
5. **推奨事項**: 専門的な推奨

※判断に必要な情報を4-5段落で回答してください。
""",
        ("comparison", "detailed"): """
## 回答構成【比較・詳細モード】
1. **包括的比較**: 全面的な比較分析
2. **各選択肢の詳細**: 詳細な特徴分析
3. **科学的根拠**: エビデンスに基づく比較
4. **選択アルゴリズム**: 系統的な選択方法
5. **症例別適用**: 具体的症例での選択例
6. **長期的考慮**: 予後や長期的な観点
7. **専門的見解**: 最新の専門的推奨

※包括的な判断材料を6段落以上で提供してください。
""",
        
        # CLINICAL タイプ
        ("clinical", "brief"): """
## 回答構成【臨床・簡潔モード】
1. **典型例**: 代表的な症例
2. **基本対応**: 基本的な対応方法
3. **重要ポイント**: 最重要な注意点

※臨床の要点を3段落程度で回答してください。
""",
        ("clinical", "standard"): """
## 回答構成【臨床・標準モード】
1. **症例特徴**: 典型的な症例や背景
2. **診断アプローチ**: 診断の進め方
3. **治療戦略**: 基本的な治療方針
4. **実施ポイント**: 重要な実施上の注意
5. **予後管理**: 経過観察のポイント

※臨床実践に役立つ内容を4-5段落で回答してください。
""",
        ("clinical", "detailed"): """
## 回答構成【臨床・詳細モード】
1. **症例背景**: 詳細な症例特徴と疫学
2. **診断プロセス**: 系統的な診断アプローチ
3. **治療計画**: 包括的な治療戦略
4. **手技詳細**: 具体的な実施方法
5. **合併症管理**: リスク評価と予防対策
6. **予後評価**: 長期的な予後と管理
7. **実症例**: 具体的な症例提示
8. **最新動向**: 最新の治療法や研究

※専門的で包括的な臨床情報を6段落以上で提供してください。
"""
    }
    
    # デフォルトの標準構成
    default_structure = """
## 回答構成【標準モード】
1. **要点**: 質問に対する明確な回答
2. **根拠**: 理論的背景と科学的根拠
3. **実践**: 臨床での重要な注意点
4. **補足**: 関連する有用な情報

※バランスの取れた4-5段落程度で回答してください。
"""
    
    # 該当する構成を取得
    structure = response_structures.get((question_type, detail_level), default_structure)
    
    return base_prompt + structure

# def test_question_analysis():
#     """分析システムをテストする関数"""
    
#     test_questions = [
#         "自家歯牙移植とは",                          # → definition, brief
#         "自家歯牙移植を成功させる方法",                # → method, detailed  
#         "自家歯牙移植のメリットとデメリット",          # → comparison, standard
#         "難治性根尖病変の症例対応について詳しく",      # → clinical, detailed
#         "根管治療の基本的な手順を簡単に",             # → method, brief
#     ]
    
#     print("=== 質問分析テスト ===")
#     for question in test_questions:
#         q_type, detail = analyze_question_type(question)
#         print(f"質問: {question}")
#         print(f"判定: タイプ={q_type}, 詳細度={detail}")
#         print("-" * 50)

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
#     setup_cache_cleanup_scheduler(index,expiration_days=60)
#     demo.launch(server_name="127.0.0.1", server_port=7860)
    