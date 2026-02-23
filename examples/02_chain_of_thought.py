"""
02: Chain of Thought - 推論プロセスを伴う応答
dspy.ChainOfThought を使って、ステップバイステップの推論を行います。
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# --- ChainOfThought ---
# 自動的に reasoning フィールドが追加される
cot = dspy.ChainOfThought("question -> answer")

result = cot(question="ある店で、りんごが3個で150円、みかんが5個で200円です。りんご6個とみかん10個を買うと合計いくらですか？")
print("=== Chain of Thought ===")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
print()

# --- 数学問題 ---
math_cot = dspy.ChainOfThought("question -> answer: float")
result = math_cot(question="A train travels 120 km in 2 hours. Then it travels 180 km in 3 hours. What is the average speed for the entire journey in km/h?")
print("=== Math with CoT ===")
print(f"Reasoning: {result.reasoning}")
print(f"Answer: {result.answer}")
print()

# --- 論理パズル ---
logic_cot = dspy.ChainOfThought("puzzle -> solution")
result = logic_cot(
    puzzle="AはBより背が高い。CはBより背が低い。DはAより背が高い。背の高い順に並べてください。"
)
print("=== Logic Puzzle ===")
print(f"Reasoning: {result.reasoning}")
print(f"Solution: {result.solution}")
