export interface Agent {
  name: string;
  description: string;
  model: string | null;
  tool_groups: string[] | null;
  tags?: string[] | null;
  role?: string | null;
  mission?: string | null;
  in_scope?: string[] | null;
  out_of_scope?: string[] | null;
  tool_permissions?: string[] | null;
  constraints?: string[] | null;
  escalation_rules?: string[] | null;
  input_schema?: Record<string, unknown> | null;
  output_schema?: Record<string, unknown> | null;
  completion_definition?: string[] | null;
  soul?: string | null;
}

export interface CreateAgentRequest {
  name: string;
  description?: string;
  model?: string | null;
  tool_groups?: string[] | null;
  tags?: string[] | null;
  role?: string | null;
  mission?: string | null;
  in_scope?: string[] | null;
  out_of_scope?: string[] | null;
  tool_permissions?: string[] | null;
  constraints?: string[] | null;
  escalation_rules?: string[] | null;
  input_schema?: Record<string, unknown> | null;
  output_schema?: Record<string, unknown> | null;
  completion_definition?: string[] | null;
  soul?: string;
}

export interface UpdateAgentRequest {
  description?: string | null;
  model?: string | null;
  tool_groups?: string[] | null;
  tags?: string[] | null;
  role?: string | null;
  mission?: string | null;
  in_scope?: string[] | null;
  out_of_scope?: string[] | null;
  tool_permissions?: string[] | null;
  constraints?: string[] | null;
  escalation_rules?: string[] | null;
  input_schema?: Record<string, unknown> | null;
  output_schema?: Record<string, unknown> | null;
  completion_definition?: string[] | null;
  soul?: string | null;
}
