# Chat 评分请求严格 Schema 方案

状态：已确认（2026-11-13，维护者确认 Q1–Q6 按推荐）。代码已实现，总开关默认关闭；Vertex 实测与 QWK 未做（见第 10 节）。

范围：Chat 协议（`openai_compatible`）下 Core 评分的两个请求——单条规则 `score_core_envelope` 与互斥组
`score_core_group`。不改 v1 兼容路径，不改 Responses 与 Messages 适配器，没有迁移。

## 1. 背景

#103 定位的那次失败：Vertex 上的 gemini-2.5-flash 对规则 `t03.v1` 的回答没有通过字段校验。#103 加了
“重问一次”，是事后补救。这份方案从源头减少这类回答：把评分请求的输出约束从“一段文字说明”升到
“厂商按 Schema 约束生成”。

## 2. 现状

Core 评分为每条规则生成一份严格 Schema（`core_view.build_core_request` 的 `request.schema`）：字段名、
`status` 枚举、档位枚举、允许的发现代码、可引用的证据别名都在里面。三个协议用它的方式不同：

| 协议 | 评分请求怎么用这份 Schema |
|---|---|
| Responses | `text.format: json_schema` + `strict: true`，始终发送 |
| Messages（Claude） | 连接选项 `structured_output`，核实过的主机默认发（`output_config.format`），发送前删掉 Claude 不支持的约束 |
| Chat | **不发。** 连接选项 `response_format_json` 打开时发 `json_object`，默认关闭，即什么都不发 |

所以 Chat 下 Schema 已经算出来了，只是没有交给厂商；模型只看到系统提示里的文字描述。

Chat 适配器的起草请求（评分标准侧）已经在发 `response_format: json_schema`，评分请求没有跟上。

生产那个连接有没有打开 `response_format_json`，我没有查（读生产需要你来运行）。无论开没开，都没有
Schema 约束。

## 3. Vertex 支不支持

查了 Google 的官方文档（2026-11-13），结论分三层：

**文档写明的：**

- Vertex 的 OpenAI 兼容 Chat Completions 接口，`response_format` 支持 `json_object`、`json_schema`、`text`；
  对 `json_schema` 只有一句说明：不支持完全递归的 Schema，支持 `additional_properties`。
- 不认识的参数会被忽略，不报错。
- Gemini 2.5 Flash 在该接口的支持模型列表里，也在“控制生成输出”的支持模型列表里。
- Schema 太复杂会返回 400。文档点名的因素有：属性名很长、数组长度限制很大、枚举值很多、可选属性很多。
  没有给任何数字上限。
- 枚举只支持非空字符串。
- Schema 的大小计入输入 token。

**文档没写的：**

- `strict` 字段是被接受、被执行，还是被忽略。
- “gemini-2.5-flash + 兼容接口 + json_schema”这个组合能用（两张列表分别列了，没有一句话合在一起说）。
- 可空类型写成 `["string","null"]` 是否接受（Vertex 的示例用的是 `nullable: true`）。
- `minLength`、`const` 是否接受。

**对我们的 Schema 意味着什么：**

| 我们用到的写法 | 在 Vertex 上的风险 |
|---|---|
| `level_code`：`type: ["string","null"]`，枚举里含 `null` | 两处都可能被拒绝：类型数组没有文档依据，枚举只允许非空字符串 |
| `evidence_ref`：枚举是本次可引用的全部证据别名 | 一篇论文的一条规则可能有几十个别名，属于文档点名的“枚举值很多” |
| `quote`、`rationale` 的 `minLength: 1` | 不在支持列表里，按文档会被忽略，无害 |
| `additionalProperties: false` | 兼容接口写明支持 |

结论：**文档层面可行，但我们现有的 Schema 不能原样发给 Vertex，而且能不能用最终要对真实接口发几次
请求才知道。** 这需要你授权（见 Q5）。

## 4. 方案

### 4.1 连接选项

给 Chat 连接加一个选项 `structured_output`，取值 `json_schema` / `json_object` / `off`，与 Claude 连接已有
的同名选项一致。旧的 `response_format_json` 保留并继续生效（`true` 等同 `json_object`），不迁移数据。

没有显式设置时的默认值按主机族决定：只有核实过的主机族默认 `json_schema`，其余保持现状。核实过的
清单与来源、日期写在代码里（与思考能力表、Claude 的 `_STRUCTURED_OUTPUT_HOSTS` 同一做法）。上线时这张
清单是空的，由总开关控制（见 4.5）。

### 4.2 Schema 方言

发送前按主机族把 Schema 转成对方接受的写法，放在适配器层，内核与 `core_view` 生成的 Schema 不变：

- **Google（Vertex、Gemini API）**：可空类型改写成 `nullable: true`，枚举里去掉 `null`；删掉 `minLength`；
  `evidence_ref` 的枚举超过上限时退成普通字符串（上限见 Q3）。
- **其它主机族**：原样发送。实现时逐个核实，核实不了的不默认开启。

被方言放宽的约束仍由本地校验把关（`decode_core_response`、`normalize_core_provider_response`），
所以放宽只会让“厂商拦不住、我们拦住”的回答多一些，不会让不合格的回答被采用。

### 4.3 请求与身份

- 请求体：`response_format: {"type":"json_schema","json_schema":{"name":…,"strict":true,"schema":…}}`，
  名字与 Responses 用的相同（`semantic_rule_response` / `semantic_group_response`）。
- 信封里的 `provider.response_format` 记为 `json_schema`（契约已允许这个值）。它是运行身份的一部分，
  所以打开之后缓存与判定账本自然不复用旧结果，不会混用两种请求得到的判定。
- 方言有版本号，写进 `model_version`。以后改某个主机族的改写规则就升这个版本。
- 系统提示与用户消息一个字不改。

### 4.4 失败时怎么办

- 厂商拒绝 Schema（400）：照常报 `PROVIDER_INVALID_REQUEST`，**不**自动退回 `json_object` 重发。
- 回答仍然不合格：#103 的重问照常生效。重问请求同样带 Schema。
- 测试连接时，`structured_output=json_schema` 的连接按评分的形状带一份小 Schema 发探测，接口不接受就在
  测试时失败，而不是等到评分。

### 4.5 上线顺序

1. 合并代码，总开关 `SCORING_CHAT_STRICT_SCHEMA_ENABLED` 默认关闭。关闭时请求逐字节不变。
2. 经你授权，对 Vertex 的真实接口做一次小规模核实（Q5）：接口是否接受改写后的 Schema、枚举上限在哪、
   不合格回答的比例有没有下降。
3. 核实通过后把 Vertex 记入默认清单，开关仍然关着。
4. 并入已经在排队的那次合并评测（判定范围、证据不足、判定依据、思考能力表的评分范围），QWK 通过后
   再打开。

## 5. 影响

- 开关关闭时没有任何行为变化。
- 打开后：受影响连接的运行身份改变，已有缓存与账本不复用，第一次评分按全量计费。
- 每个请求的输入 token 增加一份 Schema 的大小（Vertex 明确计费）。具体多少在第 2 步量。
- 评分结果可能变化——约束生成会改变模型的输出分布，这正是需要 QWK 的原因。

## 6. 放弃了什么

- **直接默认打开。** 改的是评分请求构造，上线清单要求真实留出集 QWK；而且 Vertex 对我们的 Schema
  是否接受还没有实测。
- **被拒绝时自动退回 `json_object`。** 同一个连接会在两种请求之间漂移，运行身份说的是 `json_schema`、
  实际发的不是，结果无法复现。09-30 否决“调用时换协议”是同一个理由。
- **为了兼容 Vertex 去改 `core_view` 里的 Schema。** 那会同时改变 Responses 与 Claude 的请求，把一个
  主机族的限制扩散到所有协议。方言留在 Chat 适配器里。
- **只开 `json_object`。** 只保证是 JSON，不约束字段；`t03.v1` 那类“字段不对的 JSON”照样出现。
- **只靠 #103 的重问。** 重问每次多花一次调用，每篇有上限，而且同一个模型在温度 0 下重问未必改口。
  两者不冲突：Schema 减少发生，重问兜底。

## 7. 测试

全部用假的 HTTP 响应：

- 开关关闭时，单条与互斥组的请求体与现在逐字节相同；
- 打开且 `structured_output=json_schema` 时请求体带 Schema，信封身份为 `json_schema`；
- Google 方言：可空类型、枚举去 `null`、删 `minLength`、别名枚举超上限时退成字符串；其它主机族原样；
- 方言改写后的 Schema 仍然接受本地校验认可的合格回答（用现有的合格回答样例逐个验）；
- 400 不重发；重问请求带 Schema；
- 连接选项校验、旧 `response_format_json` 的兼容、测试连接的探测形状。

## 8. 文档与依赖

- 三文档：ARCHITECTURE（Chat 的输出约束）、DECISIONS（含上面的放弃项）、RUNBOOK（新开关、Schema 被拒
  时的 400 怎么认）。
- 与 #103 改同一个文件（`openai_compatible_adapter.py`），等 #103 合并后再从 develop 切分支。
- 与《连接协议优先 Responses 方案》互补：那份让“有 Responses 的平台”用上严格 Schema，这份让“只有 Chat 的
  平台”也用上。两份都落地后，协议之间在输出约束上的差距基本消失。

## 9. 待确认

| # | 问题 | 推荐 |
|---|---|---|
| Q1 | 新选项 `structured_output` 与旧 `response_format_json` 并存，还是替换？ | 并存，旧的继续生效，不迁移数据。 |
| Q2 | 默认值按“核实过的主机族清单”定，清单初始为空？ | 是。 |
| Q3 | `evidence_ref` 的别名枚举超过多少个就退成普通字符串？ | 先不定数字，第 4.5 节第 2 步实测 Vertex 在哪里开始拒绝，取一个留有余量的值。 |
| Q4 | 厂商拒绝 Schema 时是否自动退回？ | 不退回，报错。 |
| Q5 | 是否授权对 Vertex 真实接口做小规模核实？预算多少？ | 建议授权。规模：一篇论文的 20 条规则，带与不带 Schema 各跑一遍，约 40 次 gemini-2.5-flash 调用，预算上限 1 美元。不授权的话代码可以合并，但 Vertex 进不了默认清单，开关打不开。 |
| Q6 | QWK 是否并入已排队的合并评测？ | 是，不单独评一次。 |

## 10. 实施记录

2026-11-13：

- 已实现：`llm/schema_dialects`（约束解析、Google 与原样两种方言）、Chat 适配器的 Core 请求、身份字段、连接选项校验、测试连接的探测、`test_chat_strict_schema.py`、三文档。
- 与方案的差别：
  - 总开关关闭时，连接上的 `structured_output` 可以保存但不生效（方案没有写明）。原因：否则一个连接可以绕过 QWK 门禁在生产打开。
  - `SCORING_CHAT_SCHEMA_ENUM_LIMIT` 先给了暂定值 64，没有实测依据，等第 4.5 节第 2 步。
  - Google 方言多做了一处：必须为空的数组不带空对象的条目定义（文档要求属性名非空，空对象定义没有依据）。
  - 界面没有加这个选项的入口，只能通过接口设置。
- 没做：Vertex 真实接口核实（Q5 已授权，上限 1 美元）、QWK。`VERIFIED_STRICT_SCHEMA_FAMILIES` 为空。

### Vertex 实测（2026-11-13，维护者运行 `probe_chat_schema`）

对象：Vertex 兼容接口上的 `google/gemini-2.5-flash`，请求内容全部是合成的。共 13 次请求，约 5500 输入、650 输出 token。

| 问题 | 结果 |
|---|---|
| 接口接不接受我们的 Schema | 接受。原样的（类型数组、枚举含 `null`、`minLength`）与改写后的都返回 200 |
| 枚举多大开始被拒绝 | 8 / 32 / 64 / 128 / 256 个别名都没有被拒绝 |
| Schema 的输入开销 | 每个别名约 3.5 token（8 个别名 190，256 个别名 1090） |
| `strict` 是否被执行 | 提示词要求 `status=maybe`、`level_code=Z` 并多加一个字段，带 Schema 时回答仍然合规（原样与改写后各 1 次）。**缺对照**：没有测同一句提示词不带 Schema 时模型会不会违反，所以还不能写成“已核实” |

没有测到的：没有可引用证据时的写法（必须为空的数组）、带判定依据的 Schema、互斥组的 Schema、真实规则的规模。脚本的 `--only-violate` 已包含对照与空数组两项，尚未运行。

据此暂不改代码：`VERIFIED_STRICT_SCHEMA_FAMILIES` 仍为空，Google 方言保留。对照通过后的打算是把 Vertex 列入清单、对 Google 改为原样发送（实测原样写法被接受，改写反而丢约束）、删掉枚举上限配置——那是一次身份变化，要在打开开关之前做完。
