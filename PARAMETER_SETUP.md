## 参数设置说明（先关键参数，后次要参数）

本文档说明如何运行 `eval_mcq.py` 并正确配置参数。

**重要说明（必须看）**：
- **本代码对所有题型（包括选择题与非选择题）统一采用 LLM 作为裁判（LLM-as-a-judge）来判定对错**。
- **不再使用本地“规则提取/字符串匹配”逻辑作为最终得分依据**（日志中可能仍会保留提取字段用于排查）。

---

## 关键参数（必须/优先配置）

### 1) 数据集输入：`--input`（或 YAML 的 `input_path`）
- **作用**：指定待评测的 Excel（`.xlsx`）文件。
- **示例**：`--input data/dataset.xlsx`

同时可选：
- **`--sheet`**：指定工作表名称；不填则读取第一个 sheet。

### 2) 被测模型（Model）配置
这组参数决定“要评测的模型是谁、去哪调用”。

- **`--model-provider`**：模型厂商/协议，支持 `openai` / `gemini` / `claude`
- **`--model-base-url`**：API Base URL
  - OpenAI 兼容接口：一般是 `https://api.openai.com`
  - 也可以是你自建/转发的 OpenAI 兼容网关
- **`--model-name`**：模型名（必填）
- **`--model-api-key`**：API Key
  - 若 `--model-provider openai` 且未传此参数，会尝试读取环境变量 **`OPENAI_API_KEY`**
  - `gemini` / `claude` 也需要有效的 key（只是校验逻辑略宽松，没填通常会在调用时失败）

最小可用示例（OpenAI 兼容）：

```bash
python eval_mcq.py \
  --input data/dataset.xlsx \
  --model-provider openai \
  --model-base-url https://api.openai.com \
  --model-name gpt-4.1-mini \
  --model-api-key $OPENAI_API_KEY
```

### 3) 裁判模型（Judge）开关与配置（非选择题必读）
- **裁判模型现在默认始终启用，并用于所有题型评分**。
- `--judge-enable` 仅作为历史参数保留（兼容旧命令行），即使不传也会启用裁判评分。
- 如果你不单独配置 Judge，则会**默认使用被测模型的配置作为裁判**（同 provider/base_url/api_key/model）。

裁判相关参数（与 model 类似，可单独指定裁判模型）：
- **`--judge-provider`**、**`--judge-base-url`**、**`--judge-name`**、**`--judge-api-key`**

示例（同一个模型既当作被测模型，也当裁判；你也可以不传任何 `--judge-*` 参数走默认兜底）：

```bash
python eval_mcq.py \
  --input data/dataset.xlsx \
  --model-provider openai \
  --model-base-url https://api.openai.com \
  --model-name gpt-4.1-mini \
  --model-api-key $OPENAI_API_KEY \
  --judge-enable \
  --judge-provider openai \
  --judge-base-url https://api.openai.com \
  --judge-name gpt-4.1-mini \
  --judge-api-key $OPENAI_API_KEY
```

### 4) 输出目录：`--out-dir`
- **作用**：保存评测产物（jsonl 逐题日志、summary、输出 Excel）。
- 默认：`out_eval`

---

## 数据集字段要求（影响评测逻辑）

脚本会从 Excel 中读取以下列名（区分大小写以实际表头为准，代码会对列名做 strip）：

- **`id`**：题目唯一标识（可空，但建议提供）
- **`Question`**：题干
- **`Question_Type`**：题型
  - **选择题必须为**：`Multiple Choice`
  - **非选择题**：任何不等于 `Multiple Choice` 的值都会走“LLM 裁判”流程
- **`Options`**：选择题选项
  - 推荐格式：JSON 数组字符串，例如 `["A: ...","B: ...", ...]`
  - 也支持用 `|` 或 `;` 分隔的兜底解析
- **`Answer`**：标准答案
  - 选择题：建议为 `A` 或 `A,B,C`（不含空格更稳）
  - 非选择题：直接写参考答案文本
- **`Image`**：图片相对路径或绝对路径（可空）
- **`Image_Dependency`**：是否强依赖图片（0/1）
  - 为 1 且图片缺失时，默认会跳过该题（skipped）

---

## 次要参数（可选/调优项）

### 1) `--cot`（影响选择题输出格式约束）
- **`--cot off`（默认）**：要求模型**只输出** `A` 或 `A,B,C`（严格，不允许多余文字）。
- **`--cot on`**：答题模型输出将被强制为 **JSON**（仅 JSON、无 markdown、无额外文本），包含明确的 `answer` 字段，便于裁判稳定抽取并提升正确率。

> 注意：当前版本**最终得分与答案抽取都由裁判模型完成**，因此 `--cot` 不再影响最终评分；它只影响“答题模型被如何提示/约束输出”（降低裁判抽取难度、减少歧义）。

### 2) 并发与重试（稳定性/速度）
- **并发配额已拆分为两套（流水线）**：答题模型并发与裁判并发互不抢占，更稳定也更快。
- **`--model-concurrency`**：答题模型最大并发请求数
- **`--judge-concurrency`**：裁判模型最大并发请求数
- **`--concurrency`**：历史参数（默认 8），当你不单独指定 `--model-concurrency/--judge-concurrency` 时，会作为两者的默认值
- **`--max-retries`**：失败重试次数，默认 4
- **`--retry-base-delay-s`**：指数退避基准延迟，默认 1.0
- **`--retry-max-delay-s`**：最大退避延迟，默认 16.0

### 3) 抽样/限量：`--limit`
- 只评测前 N 行，便于快速冒烟测试。

### 4) 网络代理（VPN/Proxy）
- **`--vpn off`（默认）**：直连，并且忽略环境代理变量（`trust_env=False`）。
- **`--vpn on`**：启用代理；若不填 `--proxy`，默认使用 `http://127.0.0.1:7897`。
- **`--proxy`**：例如 `http://127.0.0.1:7897` 或 `socks5://127.0.0.1:7897`

### 5) Token/温度/超时（Model/Judge 都可配）
- `--model-max-tokens` / `--judge-max-tokens`
- `--model-temperature` / `--judge-temperature`
- `--model-timeout-s` / `--judge-timeout-s`

---

## 用 YAML 配置（可选，更适合固定跑法）

你可以用 `--config` 传一个 YAML 文件，然后只在命令行覆盖少数参数。

YAML 示例（字段名与代码读取一致）：

```yaml
input_path: data/dataset.xlsx
sheet_name: Sheet1
images_root: data
out_dir: out_eval
concurrency: 8
max_retries: 4
retry_base_delay_s: 1.0
retry_max_delay_s: 16.0
limit:

cot: off
vpn: off
proxy: ""

model:
  provider: openai
  base_url: https://api.openai.com
  api_key: ${OPENAI_API_KEY}  # 注意：YAML 不会自动展开环境变量；请改成真实 key，或改用命令行/环境变量传入
  model: gpt-4.1-mini
  timeout_s: 60
  temperature: 0
  max_tokens: 256

judge:
  enable: true
  provider: openai
  base_url: https://api.openai.com
  api_key: ${OPENAI_API_KEY}  # 同上：请改成真实 key，或改用命令行/环境变量传入
  model: gpt-4.1-mini
  timeout_s: 60
  temperature: 0
  max_tokens: 256
```

命令行示例：

```bash
python eval_mcq.py --config config.yaml
```

---

## 输出结果会生成什么？

在 `--out-dir` 下会生成：
- `results_*.jsonl`：逐题完整日志（包含 model/judge 调用的脱敏请求结构与响应摘要）
- `summary.json`：整体统计（最终口径为裁判模型评分；选择题/非选择题不再分开用不同判分规则）
- `evaluated_*.xlsx`：把逐题 `model_correct` 写回到 Excel
