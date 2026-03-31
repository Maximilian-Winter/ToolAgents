export interface Project {
  id: number;
  name: string;
  slug: string;
  description: string | null;
  working_dir: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProjectStats {
  room_count: number;
  open_issue_count: number;
  total_issue_count: number;
}

export interface Agent {
  id: number;
  name: string;
  display_name: string | null;
  role: string | null;
  persona_id: number | null;
  created_at: string;
}

export interface ProjectAgent {
  id: number;
  project_id: number;
  agent_id: number;
  system_prompt: string | null;
  initial_task: string | null;
  model: string | null;
  allowed_tools: string | null;
  prompt_source: string; // "append" | "override"
  runtime: string | null;
  extra_flags: string | null;
  added_at: string;
  // Flattened from agent
  agent_name: string;
  agent_display_name: string | null;
  agent_role: string | null;
}

export interface Room {
  id: number;
  project_id: number;
  name: string;
  topic: string | null;
  current_round: number;
  created_at: string;
}

export interface Message {
  id: number;
  room_id: number;
  sender: string;
  content: string;
  message_type: string;
  reply_to: number | null;
  to: string | null;
  edited_at: string | null;
  edit_history: unknown[] | null;
  created_at: string;
  reactions: ReactionSummary[];
}

export interface ReactionSummary {
  emoji: string;
  count: number;
  senders: string[];
}

export interface PollResponse {
  messages: Message[];
  receipts: Receipt[];
}

export interface Receipt {
  agent: string;
  last_read: number;
  updated_at: string;
}

export interface Issue {
  id: number;
  project_id: number;
  number: number;
  title: string;
  body: string | null;
  state: string;
  priority: string;
  assignee: string | null;
  reporter: string;
  milestone_id: number | null;
  created_at: string;
  updated_at: string;
  closed_at: string | null;
  labels: Label[];
  comment_count: number;
}

export interface Label {
  id: number;
  project_id: number;
  name: string;
  color: string | null;
  description: string | null;
}

export interface Comment {
  id: number;
  issue_id: number;
  author: string;
  body: string;
  created_at: string;
  updated_at: string;
}

export interface RoomStatus {
  room: Room;
  message_count: number;
  members: Agent[];
  receipts: Receipt[];
  presence: { agent: string; status: string }[];
  typing: string[];
}

export interface LaunchConfig {
  agentName: string;
  role: string;
  systemPrompt: string;
  initialTask: string;
  workingDir: string;
  serverUrl: string;
  projectSlug: string;
  model: string;
  allowedTools: string;
  promptSource: 'append' | 'override';
  runtime: string | null;
  extraFlags: string | null;
}

// Custom Fields
export interface CustomFieldDefinition {
  id: number;
  name: string;
  label: string;
  field_type: "string" | "number" | "boolean" | "enum";
  entity_type: "agent" | "project";
  options_json: string | null;
  default_value: string | null;
  required: boolean;
  sort_order: number;
  created_at: string;
  updated_at: string;
}

// Document Templates
export interface DocumentTemplate {
  id: number;
  name: string;
  description: string | null;
  type_tag: string | null;
  content: string;
  project_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface RenderResponse {
  rendered_content: string;
  unresolved_variables: string[];
}

// Knowledge Base
export interface KBDocument {
  id: number;
  project_id: number;
  path: string;
  title: string;
  tags: string | null;
  content: string;
  created_by: string;
  updated_by: string;
  created_at: string;
  updated_at: string;
}

export interface KBDocumentSummary {
  path: string;
  title: string;
  tags: string | null;
  updated_by: string;
  updated_at: string;
}

export interface KBSearchResult {
  path: string;
  title: string;
  snippet: string;
  rank: number;
}

export interface KBTreeNode {
  name: string;
  path?: string;
  title?: string;
  children?: KBTreeNode[];
}

// Mentions
export interface MentionRef {
  id: number;
  source_type: string;
  source_id: number;
  mention_type: string;
  target_path: string | null;
  target_issue_number: number | null;
  created_at: string;
}
