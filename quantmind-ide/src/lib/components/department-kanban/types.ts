/**
 * Department Kanban Types
 *
 * Type definitions for the Department Kanban sub-page UI.
 */

export type TaskStatus = 'TODO' | 'IN_PROGRESS' | 'BLOCKED' | 'DONE';

export type TaskPriority = 'HIGH' | 'MEDIUM' | 'LOW';

export type DepartmentName = 'research' | 'development' | 'risk' | 'trading' | 'portfolio' | 'flowforge' | 'shared-assets';

export interface DepartmentTask {
  task_id: string;
  task_name: string;
  description?: string;
  department: DepartmentName;
  priority: TaskPriority;
  status: TaskStatus;
  source_dept?: string;
  message_type?: string;
  workflow_id?: string;
  kanban_card_id?: string;
  mail_message_id?: string;
  created_at: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
}

export interface TaskUpdate {
  task_id: string;
  status: TaskStatus;
  timestamp: string;
}

export interface InitialTaskUpdate {
  type: 'initial';
  tasks: DepartmentTask[];
}

export interface HeartbeatTaskUpdate {
  type: 'heartbeat';
  timestamp: string;
}

export type DepartmentTaskEvent = TaskUpdate | InitialTaskUpdate | HeartbeatTaskUpdate;

export interface DepartmentTasksResponse {
  department: DepartmentName;
  tasks: DepartmentTask[];
}
