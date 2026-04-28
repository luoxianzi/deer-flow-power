export interface Model {
  id: string;
  name: string;
  model: string;
  display_name: string;
  description?: string | null;
  engine?: "codex" | "claude" | "api_pool" | null;
  credential_label?: string | null;
  supports_thinking?: boolean;
  supports_reasoning_effort?: boolean;
}

export interface TokenUsageSettings {
  enabled: boolean;
}

export interface ModelsResponse {
  models: Model[];
  token_usage: TokenUsageSettings;
}
