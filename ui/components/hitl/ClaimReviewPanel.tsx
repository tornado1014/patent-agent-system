"use client";

import { useState } from "react";
import type { ClaimSet, QualityScore, ReviewComment } from "@/lib/types";
import { QualityScoreCard } from "./QualityScoreCard";

interface ClaimReviewPanelProps {
  claims: ClaimSet;
  qualityScore: QualityScore;
  message: string;
  checkpoint: string;
  onApprove: (comments: string) => void;
  onReject: (comments: string, revisionComments: ReviewComment[]) => void;
}

export function ClaimReviewPanel({
  claims,
  qualityScore,
  message,
  checkpoint,
  onApprove,
  onReject,
}: ClaimReviewPanelProps) {
  const [comments, setComments] = useState("");
  const [revisionRequests, setRevisionRequests] = useState<ReviewComment[]>([]);
  const [activeTab, setActiveTab] = useState<"claims" | "feedback">("claims");
  const [newIssue, setNewIssue] = useState({ section: "", issue: "", severity: "medium" });

  const handleAddRevision = () => {
    if (newIssue.section && newIssue.issue) {
      setRevisionRequests([
        ...revisionRequests,
        {
          section: newIssue.section,
          issue: newIssue.issue,
          severity: newIssue.severity as "critical" | "high" | "medium" | "low",
          suggestion: "",
        },
      ]);
      setNewIssue({ section: "", issue: "", severity: "medium" });
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="bg-blue-600 text-white px-6 py-4 flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">청구항 검토</h2>
              <p className="text-blue-100 text-sm mt-1">{checkpoint} 체크포인트</p>
            </div>
            <QualityScoreCard score={qualityScore} compact />
          </div>
          <p className="text-blue-100 mt-2">{message}</p>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b flex-shrink-0">
          <button
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === "claims"
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
            onClick={() => setActiveTab("claims")}
          >
            청구항 ({claims.total_independent}독 + {claims.total_dependent}종)
          </button>
          <button
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === "feedback"
                ? "border-b-2 border-blue-600 text-blue-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
            onClick={() => setActiveTab("feedback")}
          >
            수정 요청 ({revisionRequests.length})
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === "claims" ? (
            <div className="space-y-4">
              {claims.claims.map((claim) => (
                <div
                  key={claim.claim_number}
                  className={`p-4 rounded-lg border-2 ${
                    claim.claim_type === "independent"
                      ? "border-blue-200 bg-blue-50"
                      : "border-gray-200 bg-gray-50 ml-6"
                  }`}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <span className="font-bold text-gray-800">
                      청구항 {claim.claim_number}
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        claim.claim_type === "independent"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-400 text-white"
                      }`}
                    >
                      {claim.claim_type === "independent" ? "독립항" : "종속항"}
                    </span>
                    {claim.depends_on && (
                      <span className="text-gray-500 text-sm">
                        → 청구항 {claim.depends_on} 종속
                      </span>
                    )}
                  </div>
                  <p className="text-gray-800 whitespace-pre-wrap leading-relaxed">
                    {claim.full_text}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              {/* Add new revision request */}
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h4 className="font-medium text-gray-700 mb-3">수정 요청 추가</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <input
                    type="text"
                    placeholder="대상 청구항 (예: claim_1)"
                    value={newIssue.section}
                    onChange={(e) => setNewIssue({ ...newIssue, section: e.target.value })}
                    className="border rounded-lg px-3 py-2 text-sm"
                  />
                  <input
                    type="text"
                    placeholder="수정 필요 사항"
                    value={newIssue.issue}
                    onChange={(e) => setNewIssue({ ...newIssue, issue: e.target.value })}
                    className="border rounded-lg px-3 py-2 text-sm"
                  />
                  <div className="flex gap-2">
                    <select
                      value={newIssue.severity}
                      onChange={(e) => setNewIssue({ ...newIssue, severity: e.target.value })}
                      className="border rounded-lg px-3 py-2 text-sm flex-1"
                    >
                      <option value="low">낮음</option>
                      <option value="medium">중간</option>
                      <option value="high">높음</option>
                      <option value="critical">심각</option>
                    </select>
                    <button
                      onClick={handleAddRevision}
                      className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-blue-700"
                    >
                      추가
                    </button>
                  </div>
                </div>
              </div>

              {/* Revision list */}
              {revisionRequests.length > 0 ? (
                <div className="space-y-2">
                  {revisionRequests.map((req, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-3 bg-orange-50 border border-orange-200 rounded-lg"
                    >
                      <div>
                        <span className="font-medium text-gray-700">{req.section}</span>
                        <span className="mx-2 text-gray-400">•</span>
                        <span className="text-gray-600">{req.issue}</span>
                        <span
                          className={`ml-2 text-xs px-2 py-0.5 rounded ${
                            req.severity === "critical"
                              ? "bg-red-100 text-red-700"
                              : req.severity === "high"
                              ? "bg-orange-100 text-orange-700"
                              : req.severity === "medium"
                              ? "bg-yellow-100 text-yellow-700"
                              : "bg-gray-100 text-gray-700"
                          }`}
                        >
                          {req.severity}
                        </span>
                      </div>
                      <button
                        onClick={() =>
                          setRevisionRequests(revisionRequests.filter((_, i) => i !== index))
                        }
                        className="text-red-500 hover:text-red-700"
                      >
                        삭제
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  수정 요청이 없습니다. 위에서 추가하거나 승인하세요.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Comments */}
        <div className="px-6 py-4 border-t flex-shrink-0">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            전체 의견 (선택사항)
          </label>
          <textarea
            value={comments}
            onChange={(e) => setComments(e.target.value)}
            className="w-full border rounded-lg p-3 h-20 resize-none text-sm"
            placeholder="청구항 전체에 대한 의견이 있으시면 입력해주세요..."
          />
        </div>

        {/* Actions */}
        <div className="px-6 py-4 bg-gray-50 flex justify-end gap-4 flex-shrink-0">
          <button
            onClick={() => onReject(comments, revisionRequests)}
            className="px-6 py-2.5 border-2 border-orange-500 text-orange-600 rounded-lg hover:bg-orange-50 font-medium"
          >
            수정 요청 ({revisionRequests.length}건)
          </button>
          <button
            onClick={() => onApprove(comments)}
            className="px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 font-medium"
          >
            승인 및 다음 단계
          </button>
        </div>
      </div>
    </div>
  );
}
