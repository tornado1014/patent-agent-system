/**
 * TypeScript type definitions matching Python backend state.
 */

// Workflow types
export type WorkflowType =
  | "spec_writing"
  | "oa_response"
  | "prior_art"
  | "translation"
  | "analysis";

export type ConfidenceLevel = "확실" | "가능" | "불확실";

// Quality assessment
export interface QualityScore {
  overall: number;
  completeness: number;
  accuracy: number;
  compliance: number;
  comments: string[];
}

// Review comment from Reflexion
export interface ReviewComment {
  section: string;
  issue: string;
  severity: "critical" | "high" | "medium" | "low";
  suggestion: string;
}

// Human feedback record
export interface HumanFeedback {
  checkpoint: string;
  approved: boolean;
  comments: string;
  timestamp: string;
  reviewer: string | null;
}

// Evidence for traceability (LAW-7)
export interface Evidence {
  source: string;
  page: number | null;
  paragraph: number | null;
  claim_number: number | null;
  verbatim_text: string;
  confidence: ConfidenceLevel;
  timestamp: string;
}

// Patent claim structure
export interface Claim {
  claim_number: number;
  claim_type: "independent" | "dependent";
  depends_on: number | null;
  preamble: string;
  body: string;
  full_text: string;
}

export interface ClaimSet {
  claims: Claim[];
  total_independent: number;
  total_dependent: number;
}

// Amendment for OA response
export interface Amendment {
  amendment_id: string;
  strategy_name: string;
  original_claim: string;
  amended_claim: string;
  added_limitations: string[];
  specification_support: Evidence[];
  new_matter_risk: "low" | "medium" | "high";
  effectiveness: ConfidenceLevel;
}

// Main agent state (matches PatentAgentCopilotKitState)
export interface PatentAgentState {
  // Core workflow state
  current_workflow?: WorkflowType;
  current_step: string;

  // Quality control
  iteration_count: number;
  max_iterations?: number;
  quality_score: QualityScore;
  review_comments?: ReviewComment[];

  // Human-in-the-Loop
  human_approval_required?: boolean;
  human_feedback?: HumanFeedback[];
  pending_approval_checkpoint?: string | null;

  // Evidence chain
  evidence_chain?: Evidence[];

  // Error handling
  error_messages?: string[];
  is_error_state?: boolean;

  // Metadata
  created_at?: string;
  updated_at?: string;
  session_id?: string;

  // Workflow-specific data
  draft_claims?: ClaimSet;
  draft_specification?: Record<string, string>;
  amendment_options?: Amendment[];
}

// Interrupt event types
export type InterruptType =
  | "claim_review"
  | "spec_review"
  | "final_approval"
  | "amendment_review"
  | "report_approval"
  | "prior_art_review"
  | "approval_request";

export interface BaseInterruptEvent {
  type: InterruptType;
  checkpoint: string;
  message: string;
  quality_score?: QualityScore;
}

export interface ClaimReviewEvent extends BaseInterruptEvent {
  type: "claim_review";
  claims: ClaimSet;
}

export interface SpecReviewEvent extends BaseInterruptEvent {
  type: "spec_review";
  specification: Record<string, string>;
}

export interface AmendmentReviewEvent extends BaseInterruptEvent {
  type: "amendment_review";
  amendments: Amendment[];
  recommended: string;
}

export type InterruptEvent =
  | ClaimReviewEvent
  | SpecReviewEvent
  | AmendmentReviewEvent
  | BaseInterruptEvent;

// Workflow metadata
export interface WorkflowMetadata {
  id: WorkflowType;
  name: string;
  description: string;
  steps: string[];
  hitl_checkpoints: string[];
}

export const WORKFLOWS: WorkflowMetadata[] = [
  {
    id: "spec_writing",
    name: "명세서 작성",
    description: "발명 정보로부터 특허 명세서 자동 생성",
    steps: ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"],
    hitl_checkpoints: ["E4", "E6", "E9"],
  },
  {
    id: "oa_response",
    name: "OA 대응",
    description: "거절이유통지서 분석 및 대응서 작성",
    steps: ["P1", "P2", "P3", "P4", "P5"],
    hitl_checkpoints: ["P4", "P5"],
  },
  {
    id: "prior_art",
    name: "선행기술조사",
    description: "다중 데이터베이스 선행기술 검색",
    steps: ["S1", "S2", "S3", "S4", "S5"],
    hitl_checkpoints: ["S5"],
  },
  {
    id: "translation",
    name: "특허번역",
    description: "영문 특허 → 한국어 특허 번역",
    steps: ["T1", "T2", "T3", "T4", "T5", "T6"],
    hitl_checkpoints: ["T6"],
  },
  {
    id: "analysis",
    name: "특허분석",
    description: "특허 포트폴리오 분석 및 경쟁사 분석",
    steps: ["A1", "A2", "A3", "A4", "A5"],
    hitl_checkpoints: ["A5"],
  },
];
