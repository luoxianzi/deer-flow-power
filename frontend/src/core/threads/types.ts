import type { Message, Thread } from "@langchain/langgraph-sdk";

import type { Todo } from "../todos";

export type AgentEngine = "codex" | "claude" | "api_pool";
export type RuntimeProfileName = "flash" | "thinking" | "pro" | "ultra";
export type RuntimeReasoningEffort = "minimal" | "low" | "medium" | "high";

export interface AgentThreadState extends Record<string, unknown> {
  title: string;
  messages: Message[];
  artifacts: string[];
  todos?: Todo[];
}

export interface AgentThreadContext extends Record<string, unknown> {
  thread_id: string;
  engine?: AgentEngine;
  model_name: string | undefined;
  runtime_profile?: RuntimeProfileName;
  thinking_enabled: boolean;
  is_plan_mode: boolean;
  subagent_enabled: boolean;
  reasoning_effort?: RuntimeReasoningEffort;
  agent_name?: string;
}

export interface AgentThread extends Thread<AgentThreadState> {
  context?: AgentThreadContext;
}
