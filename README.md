# openaiadk
Google ADK 原生仅支持 Gemini 模型，无法直接对接 OpenAI 兼容 API（DeepSeek、硅基流动、智谱、Ollama 等）。  本包作为适配层，将 OpenAI Chat Completions 协议（含 SSE 流式、function calling）转换为 ADK 标准的 `adkmodel.LLM` 接口， 使 Agent 可无缝切换供应商，无需修改上层业务逻辑。
