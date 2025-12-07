"use client";

import type { PatentAgentState, WorkflowType } from "@/lib/types";
import { QualityScoreCard } from "@/components/hitl/QualityScoreCard";

interface WorkflowStateDisplayProps {
  state?: PatentAgentState;
}

const WORKFLOW_STEPS: Record<WorkflowType, { step: string; label: string }[]> = {
  spec_writing: [
    { step: "E1", label: "발명 개요" },
    { step: "E2", label: "핵심 정보" },
    { step: "E3", label: "선행기술" },
    { step: "E4", label: "청구항 초안" },
    { step: "E5", label: "명세서 구조" },
    { step: "E6", label: "본문 작성" },
    { step: "E7", label: "도면 생성" },
    { step: "E8", label: "품질 검증" },
    { step: "E9", label: "최종 완성" },
  ],
  oa_response: [
    { step: "P1", label: "문서 수집" },
    { step: "P2", label: "거절이유 분석" },
    { step: "P3", label: "비판적 검토" },
    { step: "P4", label: "보정안 생성" },
    { step: "P5", label: "최종 보고서" },
  ],
  prior_art: [
    { step: "S1", label: "쿼리 생성" },
    { step: "S2", label: "DB 검색" },
    { step: "S3", label: "관련성 분석" },
    { step: "S4", label: "청구항 비교" },
    { step: "S5", label: "보고서 생성" },
  ],
  translation: [
    { step: "T1", label: "문서 입수" },
    { step: "T2", label: "구조 분석" },
    { step: "T3", label: "용어집 구축" },
    { step: "T4", label: "번역 수행" },
    { step: "T5", label: "품질 검증" },
    { step: "T6", label: "최종 출력" },
  ],
  analysis: [
    { step: "A1", label: "데이터 수집" },
    { step: "A2", label: "청구항 매핑" },
    { step: "A3", label: "트렌드 분석" },
    { step: "A4", label: "경쟁사 분석" },
    { step: "A5", label: "보고서 생성" },
  ],
};

const WORKFLOW_NAMES: Record<WorkflowType, string> = {
  spec_writing: "명세서 작성",
  oa_response: "OA 대응",
  prior_art: "선행기술조사",
  translation: "특허번역",
  analysis: "특허분석",
};

export function WorkflowStateDisplay({ state }: WorkflowStateDisplayProps) {
  if (!state?.current_workflow) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8 text-center">
        <div className="text-gray-400 text-4xl mb-4">🚀</div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          워크플로우를 시작하세요
        </h3>
        <p className="text-gray-500">
          채팅창에서 원하는 작업을 요청하거나<br />
          왼쪽에서 워크플로우를 선택하세요.
        </p>
      </div>
    );
  }

  const steps = WORKFLOW_STEPS[state.current_workflow] || [];
  const currentStepIndex = steps.findIndex((s) => s.step === state.current_step);

  return (
    <div className="space-y-6">
      {/* Workflow Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold text-gray-900">
            {WORKFLOW_NAMES[state.current_workflow]}
          </h2>
          {state.session_id && (
            <span className="text-xs text-gray-400 font-mono">
              {state.session_id}
            </span>
          )}
        </div>

        {/* Progress Steps */}
        <div className="flex items-center gap-1 overflow-x-auto pb-2">
          {steps.map((step, index) => (
            <div key={step.step} className="flex items-center">
              <div
                className={`flex flex-col items-center min-w-[70px] ${
                  index < currentStepIndex
                    ? "text-green-600"
                    : index === currentStepIndex
                    ? "text-blue-600"
                    : "text-gray-400"
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    index < currentStepIndex
                      ? "bg-green-100"
                      : index === currentStepIndex
                      ? "bg-blue-100 ring-2 ring-blue-400 ring-offset-2"
                      : "bg-gray-100"
                  }`}
                >
                  {index < currentStepIndex ? "✓" : step.step}
                </div>
                <span className="text-xs mt-1 text-center whitespace-nowrap">
                  {step.label}
                </span>
              </div>
              {index < steps.length - 1 && (
                <div
                  className={`h-0.5 w-4 flex-shrink-0 ${
                    index < currentStepIndex ? "bg-green-300" : "bg-gray-200"
                  }`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Iteration Info */}
        <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between text-sm">
          <span className="text-gray-500">
            Reflexion 반복: {state.iteration_count}/{state.max_iterations || 3}
          </span>
          {state.pending_approval_checkpoint && (
            <span className="text-orange-500 font-medium">
              ⏸ {state.pending_approval_checkpoint} 검토 대기 중
            </span>
          )}
        </div>
      </div>

      {/* Quality Score */}
      {state.quality_score && state.quality_score.overall > 0 && (
        <QualityScoreCard score={state.quality_score} />
      )}

      {/* Error State */}
      {state.is_error_state && state.error_messages && state.error_messages.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-medium mb-2">오류 발생</h3>
          <ul className="text-red-600 text-sm space-y-1">
            {state.error_messages.map((error, index) => (
              <li key={index}>• {error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Human Feedback History */}
      {state.human_feedback && state.human_feedback.length > 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="font-medium text-gray-900 mb-4">검토 이력</h3>
          <div className="space-y-3">
            {state.human_feedback.map((feedback, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg ${
                  feedback.approved ? "bg-green-50" : "bg-orange-50"
                }`}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">
                    {feedback.checkpoint} 체크포인트
                  </span>
                  <span
                    className={`text-xs px-2 py-0.5 rounded ${
                      feedback.approved
                        ? "bg-green-100 text-green-700"
                        : "bg-orange-100 text-orange-700"
                    }`}
                  >
                    {feedback.approved ? "승인" : "수정 요청"}
                  </span>
                </div>
                {feedback.comments && (
                  <p className="text-sm text-gray-600">{feedback.comments}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
