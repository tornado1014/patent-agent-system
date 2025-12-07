"use client";

import { WORKFLOWS, type WorkflowType } from "@/lib/types";

interface WorkflowSelectorProps {
  selectedWorkflow?: WorkflowType;
  onSelect?: (workflow: WorkflowType) => void;
}

const WORKFLOW_ICONS: Record<WorkflowType, string> = {
  spec_writing: "📝",
  oa_response: "📋",
  prior_art: "🔍",
  translation: "🌐",
  analysis: "📊",
};

export function WorkflowSelector({ selectedWorkflow, onSelect }: WorkflowSelectorProps) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="px-4 py-3 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">워크플로우 선택</h2>
        <p className="text-sm text-gray-500">채팅창에서 요청하거나 아래에서 선택하세요</p>
      </div>

      <div className="p-4 space-y-2">
        {WORKFLOWS.map((workflow) => (
          <button
            key={workflow.id}
            onClick={() => onSelect?.(workflow.id)}
            className={`w-full text-left p-4 rounded-lg border transition-all ${
              selectedWorkflow === workflow.id
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
            }`}
          >
            <div className="flex items-start gap-3">
              <span className="text-2xl">{WORKFLOW_ICONS[workflow.id]}</span>
              <div className="flex-1">
                <h3 className="font-medium text-gray-900">{workflow.name}</h3>
                <p className="text-sm text-gray-500 mt-0.5">{workflow.description}</p>
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-xs text-gray-400">
                    {workflow.steps.length}단계
                  </span>
                  <span className="text-xs text-gray-400">•</span>
                  <span className="text-xs text-blue-500">
                    {workflow.hitl_checkpoints.length}개 검토 포인트
                  </span>
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
