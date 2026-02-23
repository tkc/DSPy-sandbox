# DSPy 内部実装（概観）

このドキュメントは、DSPy の“内部の考え方”を把握するための概観です。  
実装の詳細は DSPy 本体リポジトリで常に更新されるため、ここでは**安定的な概念モデル**に焦点を当てています。

> 注意: ここに書かれている内容は、公式ドキュメントの公開情報に基づく**高レベルの整理**です。
> 厳密なソースコードの位置やクラス構造は、DSPy 本体の変更により変わる可能性があります。

## 1. DSPy の層構造（概念）

DSPy は大きく次の層に分けて考えると理解しやすいです。

1. **モデル層 (LM / Embedder)**
   - 実際の LLM／埋め込みモデルにアクセスする層。
   - `dspy.LM` が抽象化ポイントとなり、OpenAI や Anthropic、ローカルモデルなどを統一的に扱う。

2. **アダプタ層 (Adapter)**
   - **Signature で定義された入出力**を、実際のプロンプト／出力パースへ変換する層。
   - `ChatAdapter`, `JSONAdapter` などがある。

3. **Signature 層**
   - `"question -> answer"` のような宣言的入出力定義。
   - 入力フィールドは `InputField`、出力フィールドは `OutputField` として明示できる。

4. **Module 層**
   - `Predict`, `ChainOfThought`, `ReAct` などの高レベルモジュール。
   - Signature を使って **具体的な LLM 呼び出し戦略**を実装する。

5. **Evaluation / Optimization 層**
   - `Evaluate` でメトリクスを定義し、
     `BootstrapFewShot`, `MIPROv2` などの最適化器でプロンプトや重みを改善。

---

## 2. 実行フローの全体像

典型的な DSPy 実行は以下の流れで進みます。

1. **LM を設定**
   - `dspy.configure(lm=...)` でグローバルに LM を登録。

2. **Signature を定義**
   - 文字列 or クラスで入出力を定義。

3. **Module を作成**
   - `Predict`, `ChainOfThought`, `ReAct` などを使ってモジュールを作る。

4. **Adapter がプロンプト化**
   - Signature に基づき、入力をプロンプト形式に整形。

5. **LM 呼び出し**
   - `dspy.LM` が実際の LLM に問い合わせ。

6. **出力をパースして Prediction を生成**
   - 出力は `Prediction` として返却され、`prediction.answer` などでアクセス。

---

## 3. Signature の内部イメージ

Signature は **入出力の仕様書**です。内部的には次の役割を持ちます。

- **入力の検証・整形**
- **プロンプトテンプレートの基礎**
- **出力フォーマットの制約**

クラスベース Signature を使うと、
`InputField` / `OutputField` によって説明文や型情報が入り、
プロンプト生成時のヒントになります。

---

## 4. Module の内部イメージ

### Predict
- 最も基本的なモジュール。
- Signature に従い 1 回の LM 呼び出しで予測を返す。

### ChainOfThought
- Signature に `reasoning` フィールドを追加するイメージ。
- 推論ステップを含めた出力を誘導する。

### ReAct
- **ツール呼び出し**が可能。
- 内部では「思考 → ツール実行 → 観察 → 再思考」のループが走る。

### Module（カスタム）
- `forward()` を実装し、複数のモジュールを組み合わせる。
- 内部的には `dspy.Prediction` に結果をまとめる設計が多い。

---

## 5. Adapter の内部イメージ

Adapter は **Signature と LM の橋渡し**です。

- **入力 → プロンプト** の変換
- **出力 → Prediction** の変換

たとえば JSONAdapter を使うと、
モデルに JSON 形式の出力を要求し、出力を JSON としてパースします。

---

## 6. Evaluation と Optimization

### Evaluation
- `Evaluate` は、モデル出力をメトリクスで評価してスコア化。
- これは最適化器の入力に利用される。

### Optimization
- `BootstrapFewShot` などの最適化器は、
  **プロンプトを改善**するためにデモ例を追加・選択する。
- 進化的な探索で「より良いプロンプト設定」を生成する。

---

## 7. まとめ

DSPy の内部構造は、

- **Signature (仕様)**
- **Adapter (変換)**
- **Module (戦略)**
- **Optimizer (改善)**

という分離に強い特徴があります。  
この分離により、LM を切り替えたり、プロンプト最適化を追加したりしても、
システム全体の構造を壊さずに拡張できます。

---

## 参考

- [DSPy 公式サイト](https://dspy.ai/)
- [DSPy API リファレンス](https://dspy.ai/api/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
