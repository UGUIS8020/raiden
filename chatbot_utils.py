from uuid import uuid4
import time
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 環境変数のロード
load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
enhancement_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

CACHE_INDEX_NAME = "raiden-cache"

SIMILARITY_THRESHOLD = 0.8  # 希望通りに0.85に設定

def enhance_with_ai(question, answer):
    """
    質問と回答にAIを使って類義語や要約を追加する
    
    Parameters:
    -----------
    question : str
        ユーザーからの質問
    answer : str
        チャットボットの回答
    
    Returns:
    --------
    dict
        拡張された情報を含む辞書
    """
    try:
        print(f"===== AI拡張処理開始 =====")
        print(f"元の質問: {question}")
        
        # システムプロンプトを設定
        prompt = f"""
以下の歯科医療に関する質問と回答のペアに対して、次の拡張情報を生成してください:

1. 質問の要約 (30文字以内)
2. 回答の要約 (50文字以内)
3. 質問のキーワード (5つまで)
4. 回答のカテゴリ（例: 治療法、診断、予防、症状、技術、材料）

質問: {question}

回答: {answer}

出力は以下のJSON形式で返してください:
{{
  "question_summary": "質問の要約",
  "answer_summary": "回答の要約",
  "keywords": ["キーワード1", "キーワード2", "キーワード3", "キーワード4", "キーワード5"],
  "category": "カテゴリ"
}}

出力はJSON形式のみにしてください。説明などは不要です。
類義語はできるだけ多様にしてください。例えば「自家歯牙移植のメリットは?」と「自家歯の移植の利点は?」のように異なる言い回しや言葉を使ってください。
        """
        
        # LLMに処理を依頼
        response = enhancement_llm.invoke(prompt)
        
        # 応答をパースしてJSONに変換
        enhanced_data = json.loads(response.content)
        
        print(f"AI拡張結果:")
        print(f"  要約: {enhanced_data.get('question_summary', 'なし')}")
        print(f"  類義語: {enhanced_data.get('alternative_questions', [])}")
        print(f"  キーワード: {enhanced_data.get('keywords', [])}")
        print(f"  カテゴリ: {enhanced_data.get('category', '未分類')}")
        print(f"===== AI拡張処理完了 =====")
        
        return enhanced_data
    except Exception as e:
        print(f"AI拡張処理エラー: {e}")
        # エラー時はデフォルト値を返す
        return {
            "question_summary": question[:30] + "..." if len(question) > 30 else question,
            "answer_summary": answer[:50] + "..." if len(answer) > 50 else answer,
            "alternative_questions": [],
            "keywords": [],
            "category": "未分類"
        }

def store_response_in_pinecone(question, answer, index_name=CACHE_INDEX_NAME):
    """
    質問と回答のペアをPineconeに保存する関数。AIで拡張した情報も保存。
    
    Parameters:
    -----------
    question : str
        ユーザーからの質問
    answer : str
        チャットボットの回答
    index_name : str
        Pineconeのインデックス名（デフォルトは"raiden-cache"）
    
    Returns:
    --------
    bool
        保存が成功したらTrue、失敗したらFalse
    """
    try:
        # Pineconeの初期化
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # インデックスが存在するか確認し、なければ作成を試みる
        try:
            pinecone_index = pc.Index(index_name)
            print(f"インデックス {index_name} に接続しました")
        except Exception as e:
            print(f"インデックス {index_name} が見つかりません: {e}")
            # インデックスが存在するか確認
            indexes = pc.list_indexes()
            print(f"利用可能なインデックス: {indexes}")
            if not indexes or index_name not in [idx.name for idx in indexes]:
                print(f"インデックス {index_name} が存在しません。作成してください。")
                # 代替としてraidenインデックスを使用
                print(f"代替として 'raiden-main' インデックスを使用します")
                index_name = "raiden-main"
                try:
                    pinecone_index = pc.Index(index_name)
                except Exception as e:
                    print(f"代替インデックスへの接続も失敗: {e}")
                    return False
            else:
                try:
                    pinecone_index = pc.Index(index_name)
                except Exception as e:
                    print(f"インデックス接続エラー: {e}")
                    return False
        
        # AI拡張情報を取得
        enhanced_data = enhance_with_ai(question, answer)
        
        # Q&Aペア用の一意のIDを作成
        unique_id = str(uuid4())
        
        # 質問の埋め込みを取得
        question_embedding = embedding_model.embed_query(question)
        print(f"質問の埋め込みベクトル生成完了 (長さ: {len(question_embedding)})")
        
        # 質問と回答を含むメタデータを準備
        metadata = {
            "text": answer,  # 検索用にtextフィールドに回答を保存
            "question": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "chatbot_response",
            "question_summary": enhanced_data.get("question_summary", ""),
            "answer_summary": enhanced_data.get("answer_summary", ""),
            "alternative_questions": enhanced_data.get("alternative_questions", []),
            "keywords": enhanced_data.get("keywords", []),
            "category": enhanced_data.get("category", "未分類")
        }
        
        # ベクトルをPineconeにアップサート
        pinecone_index.upsert(
            vectors=[
                {
                    "id": unique_id,
                    "values": question_embedding,
                    "metadata": metadata
                }
            ]
        )
        print(f"オリジナル質問ベクトルをアップサート: {unique_id}")
        
        # 類義語のインデックスも追加
        alt_questions = enhanced_data.get("alternative_questions", [])
        print(f"類義語の数: {len(alt_questions)}")
        
        # 元の質問と類義語の類似度の計算と出力
        if alt_questions:
            original_embedding = np.array(question_embedding).reshape(1, -1)
            print(f"===== 類義語の類似度分析 =====")
            
            for i, alt_question in enumerate(alt_questions):
                if alt_question and len(alt_question) > 5:  # 短すぎる類義語は除外
                    print(f"類義語 {i+1}: '{alt_question}'")
                    
                    # 類義語の埋め込みベクトルを取得
                    alt_embedding = embedding_model.embed_query(alt_question)
                    alt_embedding_array = np.array(alt_embedding).reshape(1, -1)
                    
                    # 元の質問との類似度を計算
                    similarity = cosine_similarity(original_embedding, alt_embedding_array)[0][0]
                    print(f"  元の質問との類似度: {similarity:.4f}")
                    
                    # 類義語をアップサート
                    alt_id = f"{unique_id}-alt-{i}"
                    pinecone_index.upsert(
                        vectors=[
                            {
                                "id": alt_id,
                                "values": alt_embedding,
                                "metadata": metadata  # 同じメタデータを使用
                            }
                        ]
                    )
                    print(f"  類義語ベクトルをアップサート: {alt_id}")
                else:
                    print(f"類義語 {i+1}: '{alt_question}' - 短すぎるためスキップ")
        
        print(f"拡張Q&AをIDで保存しました: {unique_id} (インデックス: {index_name})")
        return True
    except Exception as e:
        print(f"Pineconeへの応答保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_previous_responses(query, index_name=CACHE_INDEX_NAME):
    print(f"DEBUG: 渡された検索クエリ → {query}")
    """
    以前に類似の質問が答えられているかチェックする関数
    拡張された検索機能を使用
    
    Parameters:
    -----------
    query : str
        ユーザークエリ
    index_name : str
        Pineconeのインデックス名（デフォルトは"raiden-cache"）
    
    Returns:
    --------
    dict
        類似の質問が見つかった場合は質問と回答を含む辞書
        見つからなかった場合は {"found": False}
    """
    print(f"===== 類似質問検索開始 =====")
    print(f"検索クエリ: '{query}'")
    print(f"検索インデックス: {index_name}")
    
    try:
        # クエリの埋め込みを取得
        query_embedding = embedding_model.embed_query(query)
        print(f"埋め込みベクトル生成完了 (長さ: {len(query_embedding)})")
        
        # インデックスに対して類似の質問をクエリ
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # インデックスが存在するか確認
        try:
            index = pc.Index(index_name)
            print(f"インデックス {index_name} に接続成功")
            
            # インデックス統計の取得
            stats = index.describe_index_stats()
            print(f"インデックス統計: ベクトル数={stats.total_vector_count}")
            
            if stats.total_vector_count == 0:
                print(f"インデックス {index_name} にベクトルが存在しません")
                # 代替としてraidenインデックスを試す
                print(f"代替インデックス 'raiden' を試みます")
                try:
                    alt_index = pc.Index("raiden")
                    alt_stats = alt_index.describe_index_stats()
                    print(f"代替インデックス統計: ベクトル数={alt_stats.total_vector_count}")
                    
                    if alt_stats.total_vector_count > 0:
                        index = alt_index
                        index_name = "raiden"
                        print(f"代替インデックス 'raiden' を使用します")
                    else:
                        print(f"代替インデックスにもベクトルがありません")
                        return {"found": False}
                except Exception as e:
                    print(f"代替インデックスへのアクセスエラー: {e}")
                    return {"found": False}
                
        except Exception as e:
            print(f"インデックス {index_name} への接続エラー: {e}")
            # 利用可能なインデックスを表示
            indexes = pc.list_indexes()
            print(f"利用可能なインデックス: {indexes}")
            
            # 代替としてraidenインデックスを試す
            if "raiden" in [idx.name for idx in indexes]:
                try:
                    index = pc.Index("raiden")
                    index_name = "raiden"
                    print(f"代替インデックス 'raiden' を使用します")
                except Exception as e:
                    print(f"代替インデックスへのアクセスエラー: {e}")
                    return {"found": False}
            else:
                return {"found": False}
        
        # 類似度しきい値を出力
        print(f"類似度閾値: {SIMILARITY_THRESHOLD}")
        
        # 類似の質問を検索
        query_results = index.query(
            vector=query_embedding,
            top_k=5,  # より多くの候補を取得
            include_metadata=True,
            filter={"type": "chatbot_response"}
        )
        
        print(f"検索結果: {len(query_results.matches)}件")
        
        # もし結果がなければ、フィルターなしで再試行
        if not query_results.matches:
            print("フィルターなしで再検索します")
            query_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            print(f"フィルターなし検索結果: {len(query_results.matches)}件")
        
        # 見つからない場合は早期リターン
        if not query_results.matches:
            print("マッチする質問が見つかりませんでした")
            return {"found": False}
            
        # 検索結果を処理
        for i, match in enumerate(query_results.matches):
            print(f"マッチ {i+1}:")
            print(f"  ID: {match.id}")
            print(f"  スコア: {match.score}")
            
            if 'question' in match.metadata:
                print(f"  質問: {match.metadata['question']}")
                
                # 類義語リスト
                alt_questions = match.metadata.get("alternative_questions", [])
                if alt_questions:
                    print(f"  類義語:")
                    for j, alt in enumerate(alt_questions):
                        print(f"    {j+1}: '{alt}'")
            
            print(f"  タイムスタンプ: {match.metadata.get('timestamp', 'なし')}")
        
        # 良いマッチがあるかチェック
        if query_results.matches and len(query_results.matches) > 0:
            best_match = query_results.matches[0]
            
            print(f"最良マッチ - スコア: {best_match.score}, 閾値: {SIMILARITY_THRESHOLD}")
            
            # 類似度スコアがしきい値以上なら良いマッチとみなす
            if best_match.score > SIMILARITY_THRESHOLD:
                print(f"閾値を超えるマッチが見つかりました: {best_match.score} > {SIMILARITY_THRESHOLD}")
                
                return {
                    "found": True,
                    "question": best_match.metadata["question"],
                    "answer": best_match.metadata["text"],
                    "similarity": best_match.score,
                    "timestamp": best_match.metadata.get("timestamp", "不明"),
                    "category": best_match.metadata.get("category", "未分類"),
                    "summary": best_match.metadata.get("answer_summary", "")
                }
            else:
                print(f"類似度が閾値未満: {best_match.score} < {SIMILARITY_THRESHOLD}")
        else:
            print("マッチする質問が見つかりませんでした")
        
        print(f"===== 類似質問検索終了 =====")
        return {"found": False}
    except Exception as e:
        print(f"過去の応答チェックエラー: {e}")
        import traceback
        traceback.print_exc()
        return {"found": False}
    
def search_cached_answer(question: str):
    """
    質問から類似質問を検索し、キャッシュ回答を返す。
    
    Parameters:
    - question (str): ユーザーの質問
    
    Returns:
    - dict: {
        "found": True/False,
        "answer": 回答テキスト,
        "question": 質問テキスト,
        "similarity": 類似度スコア,
        "timestamp": 保存日時
    }
    """
    search_result = check_previous_responses(question)
    
    if search_result.get("found"):
        print(f"キャッシュヒット: {search_result['question']}")
        print(f"類似度スコア: {search_result['similarity']}")
        print(f"保存された回答: {search_result['timestamp']}")
        return search_result
    
    print("キャッシュは見つかりませんでした")
    return {"found": False}

def analyze_question_type(message):
    """
    質問タイプと詳細度を分析する関数（改善版）
    優先度順にパターンマッチングを実行
    """
    
    message_lower = message.lower()
    
    # 優先度1: 症例・臨床系の質問（最優先で判定）
    clinical_patterns = [
        "症例", "ケース", "患者", "診断", "治療計画", "対応", "管理", 
        "臨床", "実際", "経験", "適応", "禁忌", "病変", "疾患"
    ]
    
    # 優先度2: 方法・手技系の質問（手順を含む）
    method_patterns = [
        "成功させる", "方法", "やり方", "手順", "ステップ", "コツ", "ポイント", 
        "どうやって", "どのように", "技術", "テクニック", "実施", "実行",
        "治療", "処置", "操作", "手技"
    ]
    
    # 優先度3: 比較・選択系の質問
    comparison_patterns = [
        "違い", "比較", "メリット", "デメリット", "利点", "欠点", "どちらが", 
        "選択", "使い分け", "判断", "vs", "対", "長所", "短所"
    ]
    
    # 優先度4: 定義・概要系の質問（最後に判定）
    definition_patterns = [
        "とは", "って何", "の定義", "について教えて", "どんな", "意味", 
        "概要", "基本", "基礎", "原理", "仕組み"
    ]
    
    # 詳細度を判定するキーワード
    brief_indicators = ["簡単に", "要約", "概要", "まとめ", "簡潔に"]
    detailed_indicators = ["詳しく", "詳細に", "具体的に", "徹底的に", "完全に", "包括的に"]
    
    # 質問タイプの判定（優先度順）
    question_type = "standard"  # デフォルト
    
    # 優先度順で判定（早くマッチしたものが優先）
    if any(pattern in message_lower for pattern in clinical_patterns):
        question_type = "clinical"
    elif any(pattern in message_lower for pattern in method_patterns):
        question_type = "method"
    elif any(pattern in message_lower for pattern in comparison_patterns):
        question_type = "comparison"
    elif any(pattern in message_lower for pattern in definition_patterns):
        question_type = "definition"
    
    # 詳細度の判定
    detail_level = "standard"  # デフォルト
    
    if any(pattern in message_lower for pattern in brief_indicators):
        detail_level = "brief"
    elif any(pattern in message_lower for pattern in detailed_indicators):
        detail_level = "detailed"
    else:
        # 質問タイプに基づくデフォルト詳細度
        if question_type == "definition":
            detail_level = "brief"      # 定義は通常簡潔でよい
        elif question_type == "method":
            detail_level = "detailed"   # 方法は詳細が必要
        elif question_type == "comparison":
            detail_level = "standard"   # 比較は標準的
        elif question_type == "clinical":
            detail_level = "detailed"   # 臨床は詳細が必要
    
    return question_type, detail_level

def test_question_analysis():
    """分析システムをテストする関数（改善版）"""
    
    test_questions = [
        "自家歯牙移植とは",                          # → definition, brief
        "自家歯牙移植を成功させる方法",                # → method, detailed  
        "自家歯牙移植のメリットとデメリット",          # → comparison, standard
        "難治性根尖病変の症例対応について詳しく",      # → clinical, detailed
        "根管治療の基本的な手順を簡単に",             # → method, brief
        "インプラントと自家歯牙移植の違い",           # → comparison, standard
        "歯内療法の基礎について教えて",               # → definition, brief
        "抜歯の手技を詳細に説明",                    # → method, detailed
        "歯周病患者の診断プロセス",                  # → clinical, detailed
    ]
    
    print("=== 改善版質問分析テスト ===")
    for question in test_questions:
        q_type, detail = analyze_question_type(question)
        print(f"質問: {question}")
        print(f"判定: タイプ={q_type}, 詳細度={detail}")
        print("-" * 60)

# より詳細なテスト（デバッグ用）
def debug_question_analysis(message):
    """質問分析の詳細なデバッグ情報を表示"""
    
    message_lower = message.lower()
    
    clinical_patterns = ["症例", "ケース", "患者", "診断", "治療計画", "対応", "管理", "臨床", "実際", "経験", "適応", "禁忌", "病変", "疾患"]
    method_patterns = ["成功させる", "方法", "やり方", "手順", "ステップ", "コツ", "ポイント", "どうやって", "どのように", "技術", "テクニック", "実施", "実行", "治療", "処置", "操作", "手技"]
    comparison_patterns = ["違い", "比較", "メリット", "デメリット", "利点", "欠点", "どちらが", "選択", "使い分け", "判断", "vs", "対", "長所", "短所"]
    definition_patterns = ["とは", "って何", "の定義", "について教えて", "どんな", "意味", "概要", "基本", "基礎", "原理", "仕組み"]
    
    print(f"\n=== デバッグ: {message} ===")
    print(f"小文字変換: {message_lower}")
    
    # マッチしたパターンを表示
    clinical_matches = [p for p in clinical_patterns if p in message_lower]
    method_matches = [p for p in method_patterns if p in message_lower]
    comparison_matches = [p for p in comparison_patterns if p in message_lower]
    definition_matches = [p for p in definition_patterns if p in message_lower]
    
    print(f"Clinical matches: {clinical_matches}")
    print(f"Method matches: {method_matches}")
    print(f"Comparison matches: {comparison_matches}")
    print(f"Definition matches: {definition_matches}")
    
    q_type, detail = analyze_question_type(message)
    print(f"最終判定: タイプ={q_type}, 詳細度={detail}")

# テスト実行
if __name__ == "__main__":
    test_question_analysis()
    
    # 問題があった質問をデバッグ
    print("\n" + "="*80)
    print("問題があった質問のデバッグ:")
    debug_question_analysis("難治性根尖病変の症例対応について詳しく")
    debug_question_analysis("根管治療の基本的な手順を簡単に")

def get_separated_adaptive_response(message, chat_function, history, index):
    """分析段階を分離したAIアダプティブシステム"""
    
    try:
        # Step 1: 質問タイプ分析（Pineconeを使わない）
        print("🤖 質問タイプを分析中...")
        question_type, detail_level = analyze_question_type(message)
        print(f"✅ 判定結果 - タイプ: {question_type}, 詳細度: {detail_level}")
        
        # Step 2: 最適化されたプロンプト生成
        optimized_prompt = generate_optimized_prompt(message, question_type, detail_level)
        
        # Step 3: ベクトル検索を使った回答生成
        print("🎯 最適化されたプロンプトで回答生成中...")
        final_response = chat_function(optimized_prompt, history, index)
        
        return final_response
        
    except Exception as e:
        print(f"❌ 分離システムでエラー: {e}")
        raise e

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
