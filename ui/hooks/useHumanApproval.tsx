"use client";

import { useLangGraphInterrupt } from "@copilotkit/react-core";
import { ClaimReviewPanel } from "@/components/hitl/ClaimReviewPanel";
import { OAResponseReviewPanel } from "@/components/hitl/OAResponseReviewPanel";
import type {
  InterruptEvent,
  ClaimReviewEvent,
  AmendmentReviewEvent,
  ReviewComment,
} from "@/lib/types";

// Type for human feedback response
interface HumanFeedbackResponse {
  approved: boolean;
  comments: string;
  reviewer?: string;
  selected_amendment?: string;
  revision_comments?: ReviewComment[];
}

/**
 * Hook to handle Human-in-the-Loop interrupts from LangGraph backend.
 *
 * Registers handlers for different interrupt types:
 * - claim_review: E4 checkpoint (청구항 검토)
 * - spec_review: E6 checkpoint (명세서 검토)
 * - amendment_review: P4 checkpoint (보정안 선택)
 * - approval_request: Generic approval request
 */
export function useHumanApproval() {
  // Handle claim review interrupts (E4 checkpoint)
  useLangGraphInterrupt<ClaimReviewEvent>({
    enabled: ({ eventValue }) => {
      const event = eventValue as InterruptEvent | undefined;
      return event?.type === "claim_review";
    },
    render: ({ event, resolve }) => {
      const value = event.value as ClaimReviewEvent;
      const typedResolve = resolve as unknown as (response: HumanFeedbackResponse) => void;

      return (
        <ClaimReviewPanel
          claims={value.claims}
          qualityScore={value.quality_score!}
          message={value.message}
          checkpoint={value.checkpoint}
          onApprove={(comments: string) =>
            typedResolve({
              approved: true,
              comments,
              reviewer: "user",
            })
          }
          onReject={(comments: string, revisionComments: ReviewComment[]) =>
            typedResolve({
              approved: false,
              comments,
              revision_comments: revisionComments,
              reviewer: "user",
            })
          }
        />
      );
    },
  });

  // Handle specification review interrupts (E6 checkpoint)
  useLangGraphInterrupt<InterruptEvent>({
    enabled: ({ eventValue }) => {
      const event = eventValue as InterruptEvent | undefined;
      return event?.type === "spec_review";
    },
    render: ({ event, resolve }) => {
      const value = event.value as InterruptEvent;
      const typedResolve = resolve as unknown as (response: HumanFeedbackResponse) => void;

      // For now, use a simple approval dialog
      // TODO: Create SpecReviewPanel component
      return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-2">명세서 검토</h2>
            <p className="text-gray-600 mb-4">{value.message}</p>
            <p className="text-sm text-gray-500 mb-6">
              체크포인트: {value.checkpoint}
            </p>
            <div className="flex justify-end gap-4">
              <button
                onClick={() => typedResolve({ approved: false, comments: "수정 필요" })}
                className="px-6 py-2 border border-orange-500 text-orange-600 rounded-lg hover:bg-orange-50"
              >
                수정 요청
              </button>
              <button
                onClick={() => typedResolve({ approved: true, comments: "" })}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                승인
              </button>
            </div>
          </div>
        </div>
      );
    },
  });

  // Handle amendment review interrupts (P4 checkpoint)
  useLangGraphInterrupt<AmendmentReviewEvent>({
    enabled: ({ eventValue }) => {
      const event = eventValue as InterruptEvent | undefined;
      return event?.type === "amendment_review";
    },
    render: ({ event, resolve }) => {
      const value = event.value as AmendmentReviewEvent;
      const typedResolve = resolve as unknown as (response: HumanFeedbackResponse) => void;

      return (
        <OAResponseReviewPanel
          amendments={value.amendments}
          recommended={value.recommended}
          message={value.message}
          checkpoint={value.checkpoint}
          onApprove={(selectedAmendment: string, comments: string) =>
            typedResolve({
              approved: true,
              selected_amendment: selectedAmendment,
              comments,
              reviewer: "user",
            })
          }
          onReject={(comments: string) =>
            typedResolve({
              approved: false,
              comments,
              reviewer: "user",
            })
          }
        />
      );
    },
  });

  // Handle generic approval requests
  useLangGraphInterrupt<InterruptEvent>({
    enabled: ({ eventValue }) => {
      const event = eventValue as InterruptEvent | undefined;
      return event?.type === "approval_request" || event?.type === "final_approval" || event?.type === "report_approval";
    },
    render: ({ event, resolve }) => {
      const value = event.value as InterruptEvent;
      const typedResolve = resolve as unknown as (response: HumanFeedbackResponse) => void;

      return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-lg w-full p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                <span className="text-2xl">✓</span>
              </div>
              <div>
                <h2 className="text-xl font-bold text-gray-900">승인 요청</h2>
                <p className="text-sm text-gray-500">{value.checkpoint} 체크포인트</p>
              </div>
            </div>

            <p className="text-gray-700 mb-6">{value.message}</p>

            {value.quality_score && (
              <div className="bg-gray-50 p-4 rounded-lg mb-6">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">품질 점수</span>
                  <span className={`text-xl font-bold ${
                    value.quality_score.overall >= 80 ? "text-green-600" : "text-yellow-600"
                  }`}>
                    {value.quality_score.overall}/100
                  </span>
                </div>
              </div>
            )}

            <div className="flex justify-end gap-4">
              <button
                onClick={() => typedResolve({ approved: false, comments: "" })}
                className="px-6 py-2.5 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
              >
                거절
              </button>
              <button
                onClick={() => typedResolve({ approved: true, comments: "" })}
                className="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                승인
              </button>
            </div>
          </div>
        </div>
      );
    },
  });
}
