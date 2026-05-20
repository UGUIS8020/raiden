from uuid import uuid4
import time
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# 環境変数のロード
load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
enhancement_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

CACHE_COLLECTION_NAME = "raiden-cache"

SIMILARITY_THRESHOLD = 0.8  # 希望通りに0.8に設定

def get_qdrant_client():
    """Qdrantクライアントを取得"""
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

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
4. 質問の類義語（異なる言い回し）を3つ
5. 回答のカテゴリ（例: 治療法、診断、予防、症状、技術、材料）

質問: {question}

回答: {answer}

出力は以下のJSON形式で返してください:
{{
  "question_summary": "質問の要約",
  "answer_summary": "回答の要約",
  "keywords": ["キーワード1", "キーワード2", "キーワード3", "キーワード4", "キーワード5"],
  "alternative_questions": ["類義語1", "類義語2", "類義語3"],
  "category": "カテゴリ"
}}

重要: 必ずJSON形式のみで返してください。説明文や追加のテキストは一切含めないでください。
類義語はできるだけ多様にしてください。例えば「自家歯牙移植のメリットは?」と「自家歯の移植の利点は?」のように異なる言い回しや言葉を使ってください。
        """
        
        # LLMに処理を依頼
        response = enhancement_llm.invoke(prompt)
        response_text = response.content.strip()
        
        # JSONブロックを抽出（```json と ``` で囲まれている場合）
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        
        # 応答をパースしてJSONに変換
        enhanced_data = json.loads(response_text)
        
        print(f"AI拡張結果:")
        print(f"  質問要約: {enhanced_data.get('question_summary', 'なし')}")
        print(f"  回答要約: {enhanced_data.get('answer_summary', 'なし')}")
        print(f"  類義語: {enhanced_data.get('alternative_questions', [])}")
        print(f"  キーワード: {enhanced_data.get('keywords', [])}")
        print(f"  カテゴリ: {enhanced_data.get('category', '未分類')}")
        print(f"===== AI拡張処理完了 =====")
        
        return enhanced_data
    except Exception as e:
        print(f"AI拡張処理エラー: {e}")
        print(f"応答内容: {response.content if 'response' in locals() else 'なし'}")
        # エラー時はデフォルト値を返す
        return {
            "question_summary": question[:30] + "..." if len(question) > 30 else question,
            "answer_summary": answer[:50] + "..." if len(answer) > 50 else answer,
            "alternative_questions": [],
            "keywords": [],
            "category": "未分類"
        }

def store_response_in_qdrant(question, answer, collection_name=CACHE_COLLECTION_NAME):
    """
    質問と回答のペアをQdrantに保存する関数。AIで拡張した情報も保存。
    """
    try:
        # Qdrantクライアントの初期化
        client = get_qdrant_client()
        
        # コレクションが存在するか確認
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            print(f"コレクション {collection_name} に接続しました")
            print(f"現在のポイント数: {collection_info.points_count}")
        except Exception as e:
            print(f"コレクション {collection_name} が見つかりません: {e}")
            collections = client.get_collections()
            print(f"利用可能なコレクション: {[c.name for c in collections.collections]}")
            
            print(f"代替として 'raiden-main' コレクションを使用します")
            collection_name = "raiden-main"
            try:
                collection_info = client.get_collection(collection_name=collection_name)
            except Exception as e:
                print(f"代替コレクションへの接続も失敗: {e}")
                return False
        
        # AI拡張情報を取得
        enhanced_data = enhance_with_ai(question, answer)
        
        # Q&Aペア用の一意のIDを作成
        unique_id = str(uuid4())
        
        # 質問の埋め込みを取得
        question_embedding = embedding_model.embed_query(question)
        print(f"質問の埋め込みベクトル生成完了 (長さ: {len(question_embedding)})")
        
        # 質問と回答を含むメタデータを準備
        payload = {
            "text": answer,
            "question": question,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "chatbot_response",
            "question_summary": enhanced_data.get("question_summary", ""),
            "answer_summary": enhanced_data.get("answer_summary", ""),
            "alternative_questions": enhanced_data.get("alternative_questions", []),
            "keywords": enhanced_data.get("keywords", []),
            "category": enhanced_data.get("category", "未分類")
        }
        
        # ポイントをQdrantにアップサート
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=unique_id,
                    vector=question_embedding,
                    payload=payload
                )
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
            
            alt_points = []
            for i, alt_question in enumerate(alt_questions):
                if alt_question and len(alt_question) > 5:
                    print(f"類義語 {i+1}: '{alt_question}'")
                    
                    # 類義語の埋め込みベクトルを取得
                    alt_embedding = embedding_model.embed_query(alt_question)
                    alt_embedding_array = np.array(alt_embedding).reshape(1, -1)
                    
                    # 元の質問との類似度を計算
                    similarity = cosine_similarity(original_embedding, alt_embedding_array)[0][0]
                    print(f"  元の質問との類似度: {similarity:.4f}")
                    
                    # 類義語用の新しいUUIDを生成
                    synonym_id = str(uuid.uuid4())  # ← 修正：新しいUUIDを生成
                    
                    # 類義語をポイントリストに追加
                    alt_points.append(
                        PointStruct(
                            id=synonym_id,  # ← 修正：alt_id → synonym_id
                            vector=alt_embedding,
                            payload=payload
                        )
                    )
                    print(f"  類義語ベクトルを準備: {synonym_id}")  # ← 修正：alt_id → synonym_id
                else:
                    print(f"類義語 {i+1}: '{alt_question}' - 短すぎるためスキップ")
            
            # 類義語を一括アップサート
            if alt_points:
                client.upsert(
                    collection_name=collection_name,
                    points=alt_points
                )
                print(f"{len(alt_points)}個の類義語ベクトルをアップサートしました")
        
        print(f"拡張Q&AをIDで保存しました: {unique_id} (コレクション: {collection_name})")
        return True
    except Exception as e:
        print(f"Qdrantへの応答保存エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_previous_responses(query, collection_name=CACHE_COLLECTION_NAME):
    """
    以前に類似の質問が答えられているかチェックする関数
    拡張された検索機能を使用
    
    Parameters:
    -----------
    query : str
        ユーザークエリ
    collection_name : str
        Qdrantのコレクション名（デフォルトは"raiden-cache"）
    
    Returns:
    --------
    dict
        類似の質問が見つかった場合は質問と回答を含む辞書
        見つからなかった場合は {"found": False}
    """
    print(f"===== 類似質問検索開始 =====")
    print(f"検索クエリ: '{query}'")
    print(f"検索コレクション: {collection_name}")
    
    try:
        # クエリの埋め込みを取得
        query_embedding = embedding_model.embed_query(query)
        print(f"埋め込みベクトル生成完了 (長さ: {len(query_embedding)})")
        
        # Qdrantクライアントの初期化
        client = get_qdrant_client()
        
        # コレクションが存在するか確認
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            print(f"コレクション {collection_name} に接続成功")
            print(f"コレクション統計: ポイント数={collection_info.points_count}")
            
            if collection_info.points_count == 0:
                print(f"コレクション {collection_name} にポイントが存在しません")
                # 代替としてraiden-mainコレクションを試す
                print(f"代替コレクション 'raiden-main' を試みます")
                try:
                    alt_collection_info = client.get_collection(collection_name="raiden-main")
                    print(f"代替コレクション統計: ポイント数={alt_collection_info.points_count}")
                    
                    if alt_collection_info.points_count > 0:
                        collection_name = "raiden-main"
                        print(f"代替コレクション 'raiden-main' を使用します")
                    else:
                        print(f"代替コレクションにもポイントがありません")
                        return {"found": False}
                except Exception as e:
                    print(f"代替コレクションへのアクセスエラー: {e}")
                    return {"found": False}
                
        except Exception as e:
            print(f"コレクション {collection_name} への接続エラー: {e}")
            # 利用可能なコレクションを表示
            collections = client.get_collections()
            print(f"利用可能なコレクション: {[c.name for c in collections.collections]}")
            
            # 代替としてraiden-mainコレクションを試す
            if "raiden-main" in [c.name for c in collections.collections]:
                try:
                    collection_name = "raiden-main"
                    print(f"代替コレクション 'raiden-main' を使用します")
                except Exception as e:
                    print(f"代替コレクションへのアクセスエラー: {e}")
                    return {"found": False}
            else:
                return {"found": False}
        
        # 類似度しきい値を出力
        print(f"類似度閾値: {SIMILARITY_THRESHOLD}")
        
        # 類似の質問を検索（フィルター付き）- 正しいメソッド名を使用
        try:
            search_results = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=5,  # より多くの候補を取得
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="chatbot_response")
                        )
                    ]
                )
            ).points
            print(f"検索結果: {len(search_results)}件")
        except Exception as e:
            print(f"フィルター付き検索エラー: {e}")
            search_results = []
        
        # もし結果がなければ、フィルターなしで再試行
        if not search_results:
            print("フィルターなしで再検索します")
            search_results = client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=5
            ).points
            print(f"フィルターなし検索結果: {len(search_results)}件")
        
        # 見つからない場合は早期リターン
        if not search_results:
            print("マッチする質問が見つかりませんでした")
            return {"found": False}
            
        # 検索結果を処理
        for i, result in enumerate(search_results):
            print(f"マッチ {i+1}:")
            print(f"  ID: {result.id}")
            print(f"  スコア: {result.score}")
            
            if 'question' in result.payload:
                print(f"  質問: {result.payload['question']}")
                
                # 類義語リスト
                alt_questions = result.payload.get("alternative_questions", [])
                if alt_questions:
                    print(f"  類義語:")
                    for j, alt in enumerate(alt_questions):
                        print(f"    {j+1}: '{alt}'")
            
            print(f"  タイムスタンプ: {result.payload.get('timestamp', 'なし')}")
        
        # 良いマッチがあるかチェック
        if search_results and len(search_results) > 0:
            best_match = search_results[0]
            
            print(f"最良マッチ - スコア: {best_match.score}, 閾値: {SIMILARITY_THRESHOLD}")
            
            # 類似度スコアがしきい値以上なら良いマッチとみなす
            if best_match.score > SIMILARITY_THRESHOLD:
                print(f"閾値を超えるマッチが見つかりました: {best_match.score} > {SIMILARITY_THRESHOLD}")
                
                return {
                    "found": True,
                    "question": best_match.payload["question"],
                    "answer": best_match.payload["text"],
                    "similarity": best_match.score,
                    "timestamp": best_match.payload.get("timestamp", "不明"),
                    "category": best_match.payload.get("category", "未分類"),
                    "summary": best_match.payload.get("answer_summary", "")
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