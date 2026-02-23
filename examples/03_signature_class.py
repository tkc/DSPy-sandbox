"""
03: Signature Class - クラスベースのSignature定義
詳細な説明やフィールドの制約を指定できる、より柔軟なSignature定義方法です。
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- クラスベースの Signature ---
class Summarize(dspy.Signature):
    """与えられた文章を要約してください。"""

    text: str = dspy.InputField(desc="要約対象の文章")
    summary: str = dspy.OutputField(desc="3文以内の簡潔な要約")


summarizer = dspy.Predict(Summarize)

long_text = """
DSPyは、Stanford NLPグループが開発した宣言的なAIプログラミングフレームワークです。
従来のプロンプトエンジニアリングでは、LLMへの指示をテキスト文字列として手動で作成・調整する必要がありました。
DSPyはこのアプローチを根本的に変え、プログラマーがモジュール式のコードとして
AI動作を定義できるようにします。さらに、DSPyのオプティマイザーは、
定義されたメトリクスに基づいてプロンプトやモデルの重みを自動的に最適化します。
これにより、AIシステムの開発がより効率的で再現可能になり、
モデルの変更にも柔軟に対応できるようになります。
"""

result = summarizer(text=long_text)
print("=== Summarization ===")
print(f"Summary: {result.summary}")
print()


# --- 感情分析 Signature ---
class SentimentAnalysis(dspy.Signature):
    """テキストの感情を分析してください。"""

    text: str = dspy.InputField(desc="分析対象のテキスト")
    sentiment: str = dspy.OutputField(desc="positive, negative, neutralのいずれか")
    confidence: float = dspy.OutputField(desc="確信度 (0.0 ~ 1.0)")
    reason: str = dspy.OutputField(desc="判定理由の簡潔な説明")


analyzer = dspy.Predict(SentimentAnalysis)

texts = [
    "この製品は素晴らしい！買ってよかったです。",
    "配送が遅くて、商品も期待はずれでした。",
    "普通に使えます。特に問題はありません。",
]

print("=== Sentiment Analysis ===")
for text in texts:
    result = analyzer(text=text)
    print(f"Text: {text}")
    print(f"  Sentiment: {result.sentiment} (confidence: {result.confidence})")
    print(f"  Reason: {result.reason}")
    print()


# --- エンティティ抽出 Signature ---
class EntityExtraction(dspy.Signature):
    """テキストから固有表現を抽出してください。"""

    text: str = dspy.InputField(desc="解析対象のテキスト")
    persons: list[str] = dspy.OutputField(desc="人名のリスト")
    organizations: list[str] = dspy.OutputField(desc="組織名のリスト")
    locations: list[str] = dspy.OutputField(desc="場所のリスト")


extractor = dspy.Predict(EntityExtraction)
result = extractor(
    text="田中太郎はGoogleの東京オフィスで働いている。先週、彼はスタンフォード大学のカンファレンスに参加するためにサンフランシスコを訪れた。"
)
print("=== Entity Extraction ===")
print(f"Persons: {result.persons}")
print(f"Organizations: {result.organizations}")
print(f"Locations: {result.locations}")
