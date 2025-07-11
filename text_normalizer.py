"""
テキスト正規化のためのユーティリティモジュール
歯科関連テキストの表記ゆれや形式の統一を行う関数を提供
"""

import re
import unicodedata
import hashlib
import json
import os
from pathlib import Path
import time
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

# AI正規化用のキャッシュディレクトリ
CACHE_DIR = Path("ai_normalization_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def basic_normalize_text(text):
    """
    基本的なテキスト正規化を行う関数
    
    Parameters:
    -----------
    text : str
        正規化する元のテキスト
    
    Returns:
    --------
    str
        正規化されたテキスト
    """
    if not text or not isinstance(text, str):
        return ""
    
    # NFKC正規化（全角英数字→半角、全角カタカナ→半角カタカナなどのUnicode正規化）
    normalized = unicodedata.normalize('NFKC', text)
    
    # 空白文字の正規化（全角スペース→半角スペース）
    normalized = normalized.replace('　', ' ')
    
    # 複数の空白を1つに
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 句読点の正規化
    normalized = normalized.replace('、', ',').replace('。', '.')
    
    # カッコの正規化（全角→半角）
    normalized = normalized.replace('（', '(').replace('）', ')')
    normalized = normalized.replace('「', '"').replace('」', '"')
    
    # 疑問符・感嘆符の正規化（全角→半角）
    normalized = normalized.replace('？', '?').replace('！', '!')
    
    # 前後の空白を削除
    normalized = normalized.strip()
    
    return normalized

def normalize_with_ai(text, enhancement_llm):
    """
    AIを使って質問テキストを正規化する関数
    
    Parameters:
    -----------
    text : str
        正規化する元の質問テキスト
    enhancement_llm : object
        LLM呼び出し用のオブジェクト
    
    Returns:
    --------
    str
        AIによって正規化された質問テキスト
    """
    try:
        logger.info(f"===== AI正規化処理開始 =====")
        logger.info(f"元のテキスト: {text}")
        
        # テキストのハッシュを計算
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = CACHE_DIR / f"{text_hash}.json"
        
        # キャッシュにあればそれを使用
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"キャッシュから正規化結果を読み込みました")
                    return cache_data['ai_normalized']
            except Exception as e:
                logger.error(f"キャッシュ読み込みエラー: {e}")
        
        # システムプロンプトを設定
        prompt = f"""
以下の歯科医療に関する質問テキストを正規化してください。

目的:
- 句読点、空白、記号などの表記ゆれを統一する
- 歯科用語の表記ゆれを統一する（例: むし歯→虫歯、しいしゃ→歯医者）
- 同じ意味を持つ言葉を統一する（例: メリット→利点、デメリット→欠点）

元のテキスト: {text}

出力は正規化された文のみを返してください。説明は不要です。元の内容や意味は変えずに、表記の統一だけを行ってください。
        """
        
        # LLMに処理を依頼
        response = enhancement_llm.invoke(prompt)
        
        # 応答から正規化されたテキストを取得
        normalized_text = response.content.strip()
        
        # キャッシュに保存
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'original': text,
                    'ai_normalized': normalized_text,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"キャッシュ保存エラー: {e}")
        
        logger.info(f"AI正規化結果: {normalized_text}")
        logger.info(f"===== AI正規化処理完了 =====")
        
        return normalized_text
    except Exception as e:
        logger.error(f"AI正規化処理エラー: {e}")
        # エラー時は元のテキストを返す
        return text

def hybrid_normalize_text(text, enhancement_llm=None, force_ai=False):
    """
    基本的な正規化とAI正規化を組み合わせたハイブリッドアプローチ
    
    Parameters:
    -----------
    text : str
        正規化する元のテキスト
    enhancement_llm : object, optional
        LLM呼び出し用のオブジェクト（Noneの場合はAI正規化を行わない）
    force_ai : bool, optional
        AIによる正規化を強制するかどうか（デフォルトはFalse）
    
    Returns:
    --------
    str
        正規化されたテキスト
    """
    # 基本的な正規化を適用
    basic_normalized = basic_normalize_text(text)
    
    # enhancement_llmがなければ基本正規化のみ返す
    if enhancement_llm is None:
        logger.info("LLMが指定されていないため、基本正規化のみを適用します")
        return basic_normalized
    
    # AIによる正規化が必要かどうかを判断
    needs_ai_normalization = force_ai
    
    if not force_ai:
        # 歯科関連の専門用語を含むか確認
        dental_terms = [
            'むし歯', '虫歯', 'むしば', '歯医者', 'しいしゃ', '歯科', 
            '親知らず', '矯正', 'インプラント', 'ブリッジ', '入れ歯', 
            '歯垢', 'プラーク', '知覚過敏', 'メリット', 'デメリット',
            '痛み', 'いたみ', '腫れ', 'はれ', '治療', 'ちりょう'
        ]
        
        # 歯科用語を含む場合はAI正規化を適用
        for term in dental_terms:
            if term in text:
                needs_ai_normalization = True
                break
        
        # 疑問文かどうかもチェック
        if '?' in text or '？' in text or 'か？' in text or 'ですか' in text or 'しょうか' in text:
            needs_ai_normalization = True
    
    # AIによる正規化が必要と判断された場合
    if needs_ai_normalization:
        logger.info("AIによる正規化を適用します")
        return normalize_with_ai(basic_normalized, enhancement_llm)
    else:
        logger.info("基本正規化のみを適用します（AI正規化は不要と判断）")
        return basic_normalized

# テスト用コード
if __name__ == "__main__":
    # ロギングの設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # テスト用のテキスト
    test_texts = [
        "自家歯牙移植のメリットは？　実際に行った症例を教えてください。（患者さんからの質問）",
        "こんにちは、むし歯が痛いです。どうすれば良いですか？",
        "インプラント治療と自家歯牙移植術の違いについて教えてください。",
        "前歯のブリッジが取れてしまいました。どうすれば良いでしょうか？"
    ]
    
    print("=== 基本正規化テスト ===")
    for text in test_texts:
        print(f"元の文: {text}")
        print(f"正規化後: {basic_normalize_text(text)}")
        print()
        
    # LLMがあればAI正規化もテスト可能
    # enhancement_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    # print("=== AI正規化テスト ===")
    # for text in test_texts:
    #     print(f"元の文: {text}")
    #     print(f"正規化後: {hybrid_normalize_text(text, enhancement_llm)}")
    #     print()