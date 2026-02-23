"""
04: Custom Module - カスタムモジュールの作成
dspy.Module を継承して、複数のステップを組み合わせたパイプラインを構築します。
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- カスタムモジュール: 多段階QA ---
class MultiStepQA(dspy.Module):
    """質問を分解し、各サブ質問に回答してから最終回答を生成するモジュール"""

    def __init__(self):
        self.decompose = dspy.Predict("question -> sub_questions: list[str]")
        self.answer_sub = dspy.ChainOfThought("question, context -> answer")
        self.synthesize = dspy.ChainOfThought(
            "question, sub_answers: list[str] -> final_answer"
        )

    def forward(self, question: str):
        # Step 1: 質問をサブ質問に分解
        decomposed = self.decompose(question=question)
        print(f"  Sub-questions: {decomposed.sub_questions}")

        # Step 2: 各サブ質問に回答
        sub_answers = []
        for sub_q in decomposed.sub_questions:
            ans = self.answer_sub(question=sub_q, context="一般知識に基づいて回答")
            sub_answers.append(f"{sub_q}: {ans.answer}")

        # Step 3: サブ回答を統合して最終回答を生成
        final = self.synthesize(question=question, sub_answers=sub_answers)
        return final


qa = MultiStepQA()
result = qa(question="東京とニューヨークの人口と面積を比較してください。")
print("=== Multi-Step QA ===")
print(f"Final Answer: {result.final_answer}")
print()


# --- カスタムモジュール: コードレビュー ---
class CodeReviewer(dspy.Module):
    """コードをレビューし、改善提案を行うモジュール"""

    def __init__(self):
        self.analyze = dspy.ChainOfThought(
            "code, language -> issues: list[str], quality_score: float"
        )
        self.suggest = dspy.Predict(
            "code, language, issues: list[str] -> improved_code: str, explanation: str"
        )

    def forward(self, code: str, language: str = "python"):
        # Step 1: コード分析
        analysis = self.analyze(code=code, language=language)

        # Step 2: 改善提案
        suggestion = self.suggest(
            code=code, language=language, issues=analysis.issues
        )

        return dspy.Prediction(
            issues=analysis.issues,
            quality_score=analysis.quality_score,
            improved_code=suggestion.improved_code,
            explanation=suggestion.explanation,
        )


reviewer = CodeReviewer()
sample_code = """
def calc(x,y):
    r = x+y
    if r > 100:
        print("big")
        return r
    else:
        print("small")
        return r
"""

result = reviewer(code=sample_code, language="python")
print("=== Code Review ===")
print(f"Issues: {result.issues}")
print(f"Quality Score: {result.quality_score}")
print(f"Explanation: {result.explanation}")
print(f"Improved Code:\n{result.improved_code}")
