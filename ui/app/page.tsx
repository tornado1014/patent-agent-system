"use client";

import { CopilotPopup } from "@copilotkit/react-ui";
import { useCoAgent } from "@copilotkit/react-core";
import { WorkflowSelector } from "@/components/workflows/WorkflowSelector";
import { WorkflowStateDisplay } from "@/components/workflows/WorkflowStateDisplay";
import { useHumanApproval } from "@/hooks/useHumanApproval";
import type { PatentAgentState } from "@/lib/types";

export default function HomePage() {
  // Connect to the patent agent
  const { state, setState } = useCoAgent<PatentAgentState>({
    name: "patent_agent",
    initialState: {
      current_workflow: undefined,
      current_step: "",
      iteration_count: 0,
      quality_score: { overall: 0, completeness: 0, accuracy: 0, compliance: 0, comments: [] },
    },
  });

  // Register Human-in-the-Loop handlers
  useHumanApproval();

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">P</span>
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Patent Agent System</h1>
              <p className="text-sm text-gray-500">AI-powered Patent Workflow Automation</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">v0.2.0</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Workflow Selection */}
          <div className="lg:col-span-1">
            <WorkflowSelector />
          </div>

          {/* Right: Workflow State */}
          <div className="lg:col-span-2">
            <WorkflowStateDisplay state={state} />
          </div>
        </div>
      </main>

      {/* CopilotKit Chat Popup */}
      <CopilotPopup
        labels={{
          title: "Patent Agent",
          initial: "안녕하세요! 특허 업무를 도와드리겠습니다.\n\n어떤 작업을 원하시나요?\n• 명세서 작성\n• OA 대응\n• 선행기술조사\n• 특허번역\n• 특허분석",
        }}
        clickOutsideToClose={false}
      />
    </div>
  );
}
