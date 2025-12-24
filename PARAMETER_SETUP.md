## 参数设置说明（先关键参数，后次要参数）

本文档说明如何运行 `eval_questions.py` 并正确配置参数。

**重要说明（必须看）**：
- **本代码对所有题型都采用 LLM 作为裁判（LLM-as-a-judge）来判定对错**（包括选择题）。最终写回到 Excel 的 `model_correct` 也统一来自裁判结果。
- **为降低 token**：答题模型被要求输出**严格 JSON**，脚本会先本地解析出 `answer` 再交给裁判（不把完整输出/推理过程交给裁判）。
- **选择题的裁判输入最小化**：裁判**只会看到标准答案选项**与**模型作答选项**（不再发送题干/选项文本）。
- **非选择题不发图片给裁判**：即使题目依赖图片，裁判调用也**不会携带图片/base64**；同时会对发送给裁判的文本字段做 base64 片段清洗（防止超长 blob 进入裁判 prompt）。
- **选择题识别是“鲁棒/宽松”的**：优先看 `Question_Type`（英文/中文多种写法均可），若 `Question_Type` 缺失或脏数据，则会退化为“只要 `Options` 里有 ≥2 个有效选项就当选择题”。

---

## 关键参数（必须/优先配置）

### 1) 数据集输入：`--input` / `--root`（或 YAML 的 `input_path` / `root_path`）
- **推荐（更省事）**：只给一个“上一级目录” `--root`  
  - 约定：数据集在 `<root>/dataset.xlsx`，图片在 `<root>/images/`
- **兼容旧用法**：继续支持 `--input` 指定任意 `.xlsx` 路径

示例：
- `--root data/待审核数据集`
- `--input data/dataset.xlsx`

同时可选：
- **`--sheet`**：指定工作表名称；不填则读取第一个 sheet。
- **`--images-root`**：图片根目录（可选；如果你使用了 `--root` 且不传它，会自动用 `--root` 作为图片根目录）

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
python eval_questions.py \
  --input data/dataset.xlsx \
  --model-provider openai \
  --model-base-url https://api.openai.com \
  --model-name gpt-4.1-mini \
  --model-api-key $OPENAI_API_KEY
```

### 3) 裁判模型（Judge）开关与配置（非选择题必读）
- **裁判模型会用于所有题型**（包括选择题），用于给出最终 `judge_correct`（True/False/None）。
- 你可以显式指定裁判模型参数；如果不指定，则会默认“尽量复用 model 配置”（provider/base_url/model），同时裁判 key 会优先使用 `--judge-api-key` / `judge.api_key`，否则回退到 model key。
- **`--judge-enable` 已废弃**：保留兼容但不影响当前逻辑（当前逻辑会一直使用裁判模型）。

裁判相关参数（与 model 类似）：
- **`--judge-provider`**、**`--judge-base-url`**、**`--judge-name`**、**`--judge-api-key`**

示例（同一个模型既当作被测模型，也当裁判）：

```bash
python eval_questions.py \
  --input data/dataset.xlsx \
  --model-provider openai \
  --model-base-url https://api.openai.com \
  --model-name gpt-4.1-mini \
  --model-api-key $OPENAI_API_KEY \
  --judge-provider openai \
  --judge-base-url https://api.openai.com \
  --judge-name gpt-4.1-mini \
  --judge-api-key $OPENAI_API_KEY
```

### 4) 输出目录：`--out-dir`
- **作用**：保存评测产物（jsonl 逐题日志、summary、输出 Excel）。
- 默认：`out_run`
- **输出结构**：每次运行会在 `--out-dir` 下创建一个时间戳子目录，例如 `out_run/20251222_101500/`，本次运行的 `results_*.jsonl` / `summary_*.json` / `evaluated_*.xlsx` 都会写入该子目录。

---

## 数据集字段要求（影响评测逻辑）

脚本会从 Excel 中读取以下列名（区分大小写以实际表头为准，代码会对列名做 strip）：

- **`id`**：题目唯一标识（可空，但建议提供）
- **`Question`**：题干
- **`Question_Type`**：题型
  - 支持英文/中文多种写法（例如 `Multiple Choice` / `mcq` / `单选` / `多选` / `选择题` 等）
  - 若该列缺失/不可靠，会根据 `Options` 是否存在自动推断
- **`Options`**：选择题选项
  - 推荐格式：JSON 数组字符串，例如 `["A: ...","B: ...", ...]`
  - 也支持用 `|` 或 `;` 分隔的兜底解析
- **`Answer`**：标准答案
  - 选择题：建议为 `A` 或 `A,B,C`（不含空格更稳）
  - 非选择题：直接写参考答案文本
- **`Image`**：图片相对路径或绝对路径（可空）
- **`Image.1`**：第二张图片（可选；当前脚本会优先使用 `Image`，缺失时尝试 `Image.1`）
- **`Image_Dependency`**：图片依赖程度（0/1/2）
  - 0：不依赖图片（或无图）
  - 1：图片可能有帮助，但即使图片缺失也**不跳过**
  - 2：强依赖图片；图片缺失时默认会跳过该题（skipped）
- **`Image_Complexity`**：图片复杂度（0/1/2）
  - 0：无图
  - 1：图细节简单
  - 2：图细节丰富

---

## 次要参数（可选/调优项）

### 1) `--cot`（思维链开关）
本项目仍然要求答题模型输出**严格 JSON**，以便脚本本地解析出 `answer` 再交给裁判。

- **`--cot on`**：答题模型输出 JSON 中会包含 `cot` 字段（思维链/推理过程），例如 `{"answer":"B","cot":"..."}`
- **`--cot off`（默认）**：答题模型只输出答案 JSON（不输出思维链），例如 `{"answer":"B"}`

说明：
- 无论开关如何，裁判侧仍然只依据最终 `answer` 判定对错（不会把图片/大段内容发给裁判）。

### 1.5) `--mcq-cardinality-hint`（选择题提示：告知单选/多选）
- **作用**：在选择题提示词中告知答题模型“本题答案是单选还是多选（多于一个选项）”。  
  - 单选：提示 “EXACTLY ONE option”
  - 多选：提示 “MORE THAN ONE option”（**不透露具体选几个**）
- **默认**：`off`
- **取值**：`on` / `off`
- **YAML**：`mcq_cardinality_hint: off`

### 2) `--answer-json-max-attempts`（答题 JSON 解析失败重试次数）
- **作用**：当答题模型输出无法解析为 JSON 时，脚本会对同一题进行“内容级重试”（重新询问答题模型要求严格 JSON）。
- **默认**：3
- **YAML**：`answer_json_max_attempts: 3`

### 3) 并发与重试（稳定性/速度）
- **`--model-concurrency`**：被测模型并发（推荐显式设置）
- **`--judge-concurrency`**：裁判模型并发（推荐显式设置）
- **`--concurrency`**：legacy 参数；当你未显式设置 `--model-concurrency/--judge-concurrency` 时，会作为二者的默认值（默认 8）
- **`--max-retries`**：失败重试次数（默认 4）
- **`--retry-base-delay-s`**：指数退避基准延迟（默认 1.0）
- **`--retry-max-delay-s`**：最大退避延迟（默认 16.0）

### 4) 抽样/限量：`--limit`
- 只评测前 N 行，便于快速测试。

### 5) 多数投票：`--majority-vote`
- **作用**：同一题对答题模型调用 N 次，取**多数答案**作为最终答案（只对最终答案调用一次裁判）。
- **默认**：1（不投票）
- **注意**：N 越大，成本与耗时近似线性增加。

### 6) 网络代理（VPN/Proxy）
- **`--vpn off`（默认）**：直连，并且忽略环境代理变量（`trust_env=False`）。
- **`--vpn on`**：启用代理；若不填 `--proxy`，默认使用 `http://127.0.0.1:7897`。
- **`--proxy`**：例如 `http://127.0.0.1:7897` 或 `socks5://127.0.0.1:7897`

### 7) Token/温度/超时（Model/Judge 都可配）
- `--model-max-tokens` / `--judge-max-tokens`
- `--model-temperature` / `--judge-temperature`
- `--model-timeout-s` / `--judge-timeout-s`

---

## 用 YAML 配置（可选，更适合固定跑法）

你可以用 `--config` 传一个 YAML 文件，然后只在命令行覆盖少数参数。

YAML 示例（字段名与代码读取一致）：

```yaml
# 方式1（推荐）：只给 root_path（约定 root 下有 dataset.xlsx 与 images/）
# root_path: data/待审核数据集

# 方式2：显式给 input_path / images_root
input_path: data/dataset.xlsx
sheet_name: Sheet1
images_root: data
out_dir: out_run
concurrency: 8
model_concurrency:
judge_concurrency:
max_retries: 4
retry_base_delay_s: 1.0
retry_max_delay_s: 16.0
limit:

cot: off
mcq_cardinality_hint: off
answer_json_max_attempts: 3
vpn: off
proxy: ""

model:
  provider: openai
  base_url: https://api.openai.com
  api_key: ${OPENAI_API_KEY}  # 注意：YAML 不会自动展开环境变量；请改成真实 key，或改用命令行/环境变量传入
  model: gpt-4.1-mini
  timeout_s: 60
  temperature: 0
  max_tokens: 10000

judge:
  provider: openai
  base_url: https://api.openai.com
  api_key: ${OPENAI_API_KEY}  # 同上：请改成真实 key，或改用命令行/环境变量传入
  model: gpt-4.1-mini
  timeout_s: 60
  temperature: 0
  max_tokens: 10000
```

命令行示例：

```bash
python eval_questions.py --config config.yaml
```

---

## 输出结果会生成什么？

在 `--out-dir/<timestamp>/` 下会生成：
- `results_*.jsonl`：逐题完整日志（包含 model/judge 调用的脱敏请求结构与响应摘要）
- `summary_*.json`：整体统计（包含 overall 分数、网络/配置摘要等）
- `summary_*.json` 还会包含 `breakdowns`：按题型/难度/领域/Image_Dependency（若数据列存在）分组的正确率统计
- `evaluated_*.xlsx`：把逐题 `model_correct` 写回到 Excel

**指标口径（重要）**：
- 只保留**单一准确率**（overall accuracy）
- **裁判返回 unjudgeable 一律按错误计入**（即不会再单独统计 unjudgeable，也不会再区分 judged/all 两套 accuracy）
