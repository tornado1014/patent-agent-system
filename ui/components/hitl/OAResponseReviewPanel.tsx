"use client";

import { useState } from "react";
import type { Amendment, ConfidenceLevel } from "@/lib/types";

interface OAResponseReviewPanelProps {
  amendments: Amendment[];
  recommended: string;
  message: string;
  checkpoint: string;
  onApprove: (selectedAmendment: string, comments: string) => void;
  onReject: (comments: string) => void;
}

export function OAResponseReviewPanel({
  amendments,
  recommended,
  message,
  checkpoint,
  onApprove,
  onReject,
}: OAResponseReviewPanelProps) {
  const [selectedAmendment, setSelectedAmendment] = useState(recommended);
  const [comments, setComments] = useState("");

  const getRiskBadgeColor = (risk: string) => {
    switch (risk) {
      case "low":
        return "bg-green-100 text-green-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "high":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getConfidenceIcon = (confidence: ConfidenceLevel) => {
    switch (confidence) {
      case "확실":
        return "🟢";
      case "가능":
        return "🟡";
      case "불확실":
        return "🔴";
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-purple-600 text-white px-6 py-4 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">보정안 검토</h2>
              <p className="text-purple-100 text-sm mt-1">{checkpoint} 체크포인트</p>
            </div>
          </div>
          <p className="text-purple-100 mt-2">{message}</p>
        </div>

        {/* Info Banner */}
        <div className="bg-purple-50 border-b border-purple-100 px-6 py-3 flex-shrink-0">
          <p className="text-sm text-purple-700">
            <span className="font-medium">LAW-3 준수:</span> 최소 2개의 보정안 중 적합한 안을 선택해주세요.
          </p>
        </div>

        {/* Amendment Options */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {amendments.map((amendment) => (
            <div
              key={amendment.amendment_id}
              className={`border-2 rounded-xl p-5 cursor-pointer transition-all ${
                selectedAmendment === amendment.amendment_id
                  ? "border-purple-500 bg-purple-50 shadow-md"
                  : "border-gray-200 hover:border-gray-300"
              }`}
              onClick={() => setSelectedAmendment(amendment.amendment_id)}
            >
              {/* Amendment Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <input
                    type="radio"
                    checked={selectedAmendment === amendment.amendment_id}
                    onChange={() => setSelectedAmendment(amendment.amendment_id)}
                    className="h-5 w-5 text-purple-600"
                  />
                  <div>
                    <span className="font-bold text-lg text-gray-900">
                      보정안 {amendment.amendment_id}
                    </span>
                    <span className="text-gray-500 ml-2">{amendment.strategy_name}</span>
                  </div>
                  {amendment.amendment_id === recommended && (
                    <span className="px-2 py-1 bg-purple-600 text-white text-xs rounded-full">
                      추천
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className={`px-3 py-1 rounded-full text-sm ${getRiskBadgeColor(amendment.new_matter_risk)}`}>
                    신규사항 위험: {amendment.new_matter_risk}
                  </span>
                  <span className="text-lg">
                    {getConfidenceIcon(amendment.effectiveness)}
                    <span className="text-sm ml-1 text-gray-600">{amendment.effectiveness}</span>
                  </span>
                </div>
              </div>

              {/* Original vs Amended */}
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-500 mb-2">원문 청구항</h4>
                  <div className="bg-gray-100 p-4 rounded-lg text-sm whitespace-pre-wrap max-h-40 overflow-y-auto">
                    {amendment.original_claim}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-gray-500 mb-2">보정 청구항</h4>
                  <div className="bg-green-50 p-4 rounded-lg text-sm whitespace-pre-wrap border border-green-200 max-h-40 overflow-y-auto">
                    {amendment.amended_claim}
                  </div>
                </div>
              </div>

              {/* Added Limitations */}
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-500 mb-2">추가된 한정사항</h4>
                <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                  {amendment.added_limitations.map((limitation, idx) => (
                    <li key={idx}>{limitation}</li>
                  ))}
                </ul>
              </div>

              {/* Specification Support */}
              {amendment.specification_support.length > 0 && (
                <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                  <h4 className="text-sm font-medium text-blue-800 mb-2">
                    명세서 지원 근거 (LAW-7)
                  </h4>
                  <div className="space-y-2">
                    {amendment.specification_support.map((evidence, idx) => (
                      <div key={idx} className="text-sm text-blue-700">
                        <span className="font-mono bg-blue-100 px-1 rounded">
                          [{evidence.source}:{evidence.page}:{evidence.paragraph}]
                        </span>
                        <span className="italic ml-2">&ldquo;{evidence.verbatim_text}&rdquo;</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Comments */}
        <div className="px-6 py-4 border-t flex-shrink-0">
          <textarea
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            className="w-full border rounded-lg p-3 h-20 resize-none text-sm"
            placeholder="보정안에 대한 추가 의견이 있으시면 입력해주세요..."
          />
        </div>

        {/* Actions */}
        <div className="px-6 py-4 bg-gray-50 flex justify-end gap-4 flex-shrink-0">
          <button
            onClick={() => onReject(comments)}
            className="px-6 py-2.5 border-2 border-red-500 text-red-600 rounded-lg hover:bg-red-50 font-medium"
          >
            전체 재검토 요청
          </button>
          <button
            onClick={() => onApprove(selectedAmendment, comments)}
            className="px-6 py-2.5 bg-purple-600 text-white rounded-lg hover:bg-purple-700 font-medium disabled:opacity-50"
            disabled={!selectedAmendment}
          >
            보정안 {selectedAmendment} 승인
          </button>
        </div>
      </div>
    </div>
  );
}
