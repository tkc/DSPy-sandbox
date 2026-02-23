"""
01: Basic Predict - DSPyの最も基本的な使い方
dspy.Predict を使って、シンプルな質問応答を行います。
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

# LMの設定
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# --- 基本的な Predict ---
# Signature: "question -> answer" で質問応答タスクを定義
predict = dspy.Predict("question -> answer")

# 実行
result = predict(question="Pythonの作者は誰ですか？")
print("=== Basic Predict ===")
print(f"Answer: {result.answer}")
print()

# --- 型付き Predict ---
# 出力の型を指定できる
math_predict = dspy.Predict("question -> answer: float")
result = math_predict(question="2つのサイコロを振って、合計が7になる確率は？")
print("=== Typed Predict ===")
print(f"Answer: {result.answer}")
print()

# --- 複数フィールドの Signature ---
translate = dspy.Predict("text, target_language -> translated_text")
result = translate(
    text="DSPy is a framework for programming language models.",
    target_language="Japanese",
)
print("=== Translation ===")
print(f"Translated: {result.translated_text}")
print()

# --- LMの直接呼び出し ---
print("=== Direct LM Call ===")
response = lm("こんにちは！DSPyについて一言で説明してください。")
print(f"Response: {response}")
