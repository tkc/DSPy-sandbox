# DSPy Sandbox

[DSPy](https://dspy.ai/) のサンプルコード集です。  
DSPy は Stanford NLP が開発した、LLM をプログラム的に扱うための宣言的フレームワークです。

## セットアップ

### 前提条件

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python パッケージマネージャー)
- OpenAI API キー

### インストール

```bash
# 依存関係のインストール
uv sync

# 環境変数の設定
cp .env.example .env
# .env ファイルを編集して OPENAI_API_KEY を設定
```

## サンプル一覧

| # | ファイル | 内容 |
|---|---------|------|
| 01 | `examples/01_basic_predict.py` | **Basic Predict** - `dspy.Predict` を使ったシンプルな質問応答・翻訳 |
| 02 | `examples/02_chain_of_thought.py` | **Chain of Thought** - `dspy.ChainOfThought` によるステップバイステップ推論 |
| 03 | `examples/03_signature_class.py` | **Signature Class** - クラスベースの Signature で要約・感情分析・エンティティ抽出 |
| 04 | `examples/04_custom_module.py` | **Custom Module** - `dspy.Module` を継承した多段階 QA・コードレビュー |
| 05 | `examples/05_react_agent.py` | **ReAct Agent** - ツール（計算機・単位変換）を活用するエージェント |
| 06 | `examples/06_optimizer.py` | **Optimizer** - `BootstrapFewShot` によるプロンプト自動最適化 |

## 実行方法

```bash
# 各サンプルを個別に実行
uv run python examples/01_basic_predict.py
uv run python examples/02_chain_of_thought.py
uv run python examples/03_signature_class.py
uv run python examples/04_custom_module.py
uv run python examples/05_react_agent.py
uv run python examples/06_optimizer.py
```

## DSPy の主要コンセプト

### Signature
タスクの入出力を宣言的に定義する仕組み。文字列形式 (`"question -> answer"`) またはクラス形式で定義可能。

### Module
- **`dspy.Predict`** - 基本的な予測モジュール
- **`dspy.ChainOfThought`** - 推論ステップ付きの予測
- **`dspy.ReAct`** - ツール使用可能なエージェント
- **`dspy.Module`** - カスタムモジュールの基底クラス

### Optimizer
訓練データとメトリクスに基づいて、プロンプトやモデルの重みを自動最適化するアルゴリズム。

## ドキュメント

- [DSPy 内部実装（概観）](docs/INTERNALS.md)

## 参考リンク

- [DSPy 公式サイト](https://dspy.ai/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [DSPy チュートリアル](https://dspy.ai/tutorials/)
- [DSPy API リファレンス](https://dspy.ai/api/)
