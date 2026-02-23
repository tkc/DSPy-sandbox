"""
06: Optimizer - プロンプトの最適化
DSPyのオプティマイザーを使って、少数の例からプロンプトを自動最適化します。
"""

import dspy
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- 分類タスクの定義 ---
class Classify(dspy.Signature):
    """テキストをカテゴリに分類してください。"""

    text: str = dspy.InputField(desc="分類対象のテキスト")
    category: str = dspy.OutputField(
        desc="tech, sports, politics, entertainment のいずれか"
    )


# --- 訓練データ ---
trainset = [
    dspy.Example(
        text="新しいiPhoneが来月発売される予定です。",
        category="tech",
    ).with_inputs("text"),
    dspy.Example(
        text="ワールドカップの決勝戦が来週行われます。",
        category="sports",
    ).with_inputs("text"),
    dspy.Example(
        text="首相が新しい経済政策を発表しました。",
        category="politics",
    ).with_inputs("text"),
    dspy.Example(
        text="人気俳優が新作映画に主演することが決定しました。",
        category="entertainment",
    ).with_inputs("text"),
    dspy.Example(
        text="AIスタートアップが1億ドルの資金調達に成功しました。",
        category="tech",
    ).with_inputs("text"),
    dspy.Example(
        text="オリンピック代表選手が新記録を達成しました。",
        category="sports",
    ).with_inputs("text"),
    dspy.Example(
        text="国会で新法案が可決されました。",
        category="politics",
    ).with_inputs("text"),
    dspy.Example(
        text="有名歌手のコンサートチケットが即完売しました。",
        category="entertainment",
    ).with_inputs("text"),
]

# --- テストデータ ---
testset = [
    dspy.Example(
        text="量子コンピューターの新しいブレークスルーが発表されました。",
        category="tech",
    ).with_inputs("text"),
    dspy.Example(
        text="プロ野球のドラフト会議が開催されました。",
        category="sports",
    ).with_inputs("text"),
    dspy.Example(
        text="市長選挙の投票率が過去最高を記録しました。",
        category="politics",
    ).with_inputs("text"),
    dspy.Example(
        text="新しいアニメシリーズが世界的なヒットとなっています。",
        category="entertainment",
    ).with_inputs("text"),
]


# --- メトリクスの定義 ---
def classification_metric(example, prediction, trace=None):
    return example.category.lower().strip() == prediction.category.lower().strip()


# --- 最適化前の評価 ---
classifier = dspy.Predict(Classify)

print("=== Before Optimization ===")
evaluator = dspy.Evaluate(devset=testset, metric=classification_metric, num_threads=4)
score = evaluator(classifier)
print(f"Accuracy: {score}%")
print()

# --- BootstrapFewShot で最適化 ---
print("=== Optimizing with BootstrapFewShot ===")
optimizer = dspy.BootstrapFewShot(
    metric=classification_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized_classifier = optimizer.compile(classifier, trainset=trainset)

# --- 最適化後の評価 ---
print("=== After Optimization ===")
score = evaluator(optimized_classifier)
print(f"Accuracy: {score}%")
print()

# --- 最適化されたモデルでテスト ---
print("=== Test Predictions ===")
test_texts = [
    "GPT-5の性能がリークされました。",
    "サッカー日本代表が決勝進出を決めました。",
]
for text in test_texts:
    result = optimized_classifier(text=text)
    print(f"Text: {text}")
    print(f"Category: {result.category}")
    print()
