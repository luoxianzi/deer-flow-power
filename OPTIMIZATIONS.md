# DeerFlow Power Optimizations

本文记录本仓库相对官方 `bytedance/deer-flow` 的本地优化、可量化效率提升和发布前验证结果。

## 小米开发者认证摘要

```text
我基于开源 DeerFlow 2.0 构建了一个面向真实研发流程的 Agent 协作平台，核心解决长任务开发中模型切换不一致、API 网关不稳定、子任务并发不足、sandbox 容易失联和调试成本高的问题。相比官方版本，我新增了 Claude/Codex/API Pool 统一模型路由、多 gateway/多 key 轮转、409 并发满载重试、Claude OAuth 自动续期、Opus 4.7 adaptive thinking、Docker power profile、sandbox 预热与自动恢复，并把 subagent 默认并发从 3 提升到 6、硬上限从 4 提升到 8、执行线程池从 3 提升到 8。实际使用时，主 Agent 负责需求分析和任务编排，多个 subagent 并行完成代码阅读、实现、测试和文档整理。对于 6 个可并行子任务，理想墙钟耗时可下降约 50%，执行容量提升约 2.67 倍。项目已通过后端 2094 项测试、前端 typecheck/lint/unit test，可作为 AI 驱动研发效率提升的工程化实践。
```

## 对比基线

- 官方远端：`https://github.com/bytedance/deer-flow.git`
- 本次拉取到的官方 `main`：`6bd88fe1`
- 本地优化分叉点：`259a6844`
- 本地优化 HEAD：`7e4320d8`，另含本次公开发布前的清理提交
- 相对分叉点改动规模：88 个文件，约 `+11,563 / -319` 行
- 测试覆盖增量：21 个测试文件新增或扩展

说明：截至本次整理时，官方 `main` 已在分叉点之后前进 54 个提交。直接合并官方最新主线会在 agent、subagent、model factory、task tool、frontend agent 类型等核心文件产生冲突，所以本次公开发布保留本地稳定优化分支，并在后续单独做上游 rebase。

## 我们优化了什么

### 1. 模型与认证能力

- 增加 `StandardAPIChatModel`，支持 OpenAI-compatible API pool、多 gateway、多 key 轮转、Chat Completions / Responses 双协议、reasoning 内容抽取和工具调用协议修复。
- 增强 Claude OAuth provider：支持 Claude Code OAuth 凭证读取、过期自动刷新、Bearer 鉴权、Claude billing header、prompt caching、Opus 4.7 adaptive thinking 和 `reasoning_effort` 映射。
- 增强 Codex provider：支持 Codex CLI token、Responses API streaming、reasoning effort 透传和长流式超时配置。
- 增加统一 engine/model 解析层，让前端选择的模型优先于 agent 固定模型，并支持 Codex / Claude / API pool 三类运行时。

### 2. 失败恢复与可用性

- API pool 对 `401/403/429` 做 key 轮转，对 `408/409/502/503/504/524` 做瞬时错误重试。
- `409` 并发满载场景默认至少重试 `10` 次，并对重试延迟使用额外倍率，减少高并发任务直接失败。
- LLM middleware 增加 provider busy、quota、auth、network egress block 分类，并支持 API pool 自动 failover 到本地可用 fallback model。
- 对空内容响应、工具调用历史截断、compaction 后缺失 tool result 等常见网关异常做修复。

### 3. Agent 并发与长任务能力

- subagent 默认并发从 `3` 提升到 `6`。
- subagent middleware 硬上限从 `4` 提升到 `8`。
- subagent scheduler / execution / isolated-loop 线程池从 `3` worker 提升到 `8` worker。
- custom subagent 递归预算支持 `1x -> 1.5x -> 2x` 渐进重试，减少复杂子任务因为固定 recursion limit 过早失败。
- subagent 可继承父 agent 的 model、thinking 和 reasoning effort，避免 UI 已选择的高质量模型被子任务降级。

### 4. Power Profile 与 Sandbox

- 增加 `config.power.yaml`、`docker/docker-compose.power.yaml` 和 `scripts/power-profile.sh`，形成独立 power profile，不覆盖默认 `config.yaml`。
- AIO sandbox 默认预热 `2` 个 replica，新 thread 可直接复用 warm container。
- 增加 sandbox HTTP bridge 健康检查、连接失败自动 recovery、诊断计数和本地 fallback 控制项。
- 增加受控 host mount、SSH/Git/Claude/Codex 凭证桥接、API pool SSH tunnel 管理脚本。
- 增加 `agents_api.enabled=true`，让前端能列出和管理自定义 agents。

### 5. 前端体验

- workspace 输入框、消息组、agent card、model hooks 增加 runtime profile / engine / model / reasoning effort 相关状态。
- 前端选择模型优先级修正：UI 选择会覆盖 agent pinned model，避免用户切换模型后实际运行仍使用旧模型。
- i18n 增加模型与 runtime profile 相关文案。

## 效率提升

这些数字是基于代码和配置的可验证上限，不把外部 LLM 网络、账号配额、供应商排队时间算作固定收益。

| 项目 | 官方当前主线 | 本优化版 | 可量化收益 |
| --- | --- | --- | --- |
| subagent 默认并发 | 3 | 6 | 独立子任务吞吐最高 `2.0x` |
| subagent 硬上限 | 4 | 8 | 单轮最大 fan-out 最高 `2.0x` |
| executor 线程池 | 3 | 8 | 调度/执行 worker 容量 `2.67x` |
| 6 个等时长子任务 | `ceil(6/3) = 2` 批 | `ceil(6/6) = 1` 批 | 理想墙钟耗时约降低 `50%` |
| 8 个等时长子任务（显式上限 8） | `ceil(8/4) = 2` 批 | `ceil(8/8) = 1` 批 | 理想墙钟耗时约降低 `50%` |
| sandbox 冷启动 | 新任务通常等待容器启动 | 预热 2 个 replica | 前两个新 thread 可避开约 `3-5s` Docker 冷启动等待 |
| API pool 409 并发满载 | 容易快速失败或交给外层重试 | provider 内部至少 10 次耐心重试 | 提升成功率，减少人工重跑；不是固定延迟收益 |

结论：对多子任务、I/O bound、长工具链任务，本优化版的主要收益来自更高并发和更少失败重跑。理论上 6-8 个独立子任务的墙钟时间可下降约 `50%`；实际收益取决于模型供应商限流、账号 quota、任务是否真的能并行。

## 发布前验证

本次公开发布前已在本地完成：

```bash
cd backend
./.venv/bin/python -m pytest tests
# 2094 passed, 15 skipped

cd frontend
pnpm typecheck
pnpm lint
pnpm test
# 4 test files passed, 19 tests passed
```

注意：后端全量测试结束后，进程退出阶段出现过一次 memory updater 的非阻塞日志：`cannot schedule new futures after shutdown`。pytest 退出码为 0，未影响测试结果；后续可以单独把 memory updater 的 shutdown 行为做得更安静。

## 公开发布安全处理

- 已移除硬编码 API key，`config.power.yaml` 中的 GLM key 改为环境变量占位。
- 已移除会读取本机 Claude 凭证并联网调用的临时调试脚本。
- 已移除个人本机路径和默认邮箱。
- 发布公开仓库时应使用清理后的单提交历史，避免旧提交中的密钥进入公开 Git history。

本地使用 GLM / Modal Research 时，请在 `.env` 中自行配置：

```bash
GLM_5_1_API_KEYS=your-comma-separated-keys
```
