"""
05: ReAct Agent - ツールを使ったエージェント
dspy.ReAct を使って、ツールを活用するエージェントを構築します。
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- ツールの定義 ---
def calculator(expression: str) -> str:
    """数式を計算します。四則演算、累乗(**)、剰余(%)に対応。"""
    try:
        # 安全な評価のため、許可する名前を制限
        allowed = {"__builtins__": {}}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"計算エラー: {e}"


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """単位変換を行います。km<->miles, kg<->lbs, celsius<->fahrenheit に対応。"""
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    else:
        return f"未対応の変換: {from_unit} -> {to_unit}"


# --- ReAct エージェント ---
agent = dspy.ReAct(
    "question -> answer",
    tools=[calculator, unit_converter],
)

# 計算が必要な質問
print("=== ReAct Agent: Math ===")
result = agent(question="100kmは何マイルですか？また、その値の2乗はいくつですか？")
print(f"Answer: {result.answer}")
print()

# 複数ツールの組み合わせ
print("=== ReAct Agent: Multi-tool ===")
result = agent(
    question="体重75kgの人が、マラソン(42.195km)を走りました。距離をマイルに変換し、体重をポンドに変換してください。"
)
print(f"Answer: {result.answer}")
