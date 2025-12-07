"use client";

import type { QualityScore } from "@/lib/types";

interface QualityScoreCardProps {
  score: QualityScore;
  compact?: boolean;
}

export function QualityScoreCard({ score, compact = false }: QualityScoreCardProps) {
  const getScoreColor = (value: number) => {
    if (value >= 80) return "text-green-600 bg-green-100";
    if (value >= 60) return "text-yellow-600 bg-yellow-100";
    return "text-red-600 bg-red-100";
  };

  const getProgressColor = (value: number) => {
    if (value >= 80) return "bg-green-500";
    if (value >= 60) return "bg-yellow-500";
    return "bg-red-500";
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <span className="text-sm text-gray-500">품질:</span>
        <span className={`font-bold ${getScoreColor(score.overall)} px-2 py-0.5 rounded`}>
          {score.overall}/100
        </span>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-medium text-gray-900">품질 점수</h3>
        <span className={`text-2xl font-bold ${getScoreColor(score.overall)} px-3 py-1 rounded-lg`}>
          {score.overall}
        </span>
      </div>

      {/* Overall Progress Bar */}
      <div className="mb-6">
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className={`h-3 rounded-full transition-all duration-500 ${getProgressColor(score.overall)}`}
            style={{ width: `${score.overall}%` }}
          />
        </div>
        <div className="flex justify-between mt-1 text-xs text-gray-400">
          <span>0</span>
          <span>통과 기준: 80</span>
          <span>100</span>
        </div>
      </div>

      {/* Sub-scores */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center">
          <div className={`text-xl font-bold ${getScoreColor(score.completeness)}`}>
            {score.completeness}
          </div>
          <div className="text-xs text-gray-500">완성도</div>
        </div>
        <div className="text-center">
          <div className={`text-xl font-bold ${getScoreColor(score.accuracy)}`}>
            {score.accuracy}
          </div>
          <div className="text-xs text-gray-500">정확성</div>
        </div>
        <div className="text-center">
          <div className={`text-xl font-bold ${getScoreColor(score.compliance)}`}>
            {score.compliance}
          </div>
          <div className="text-xs text-gray-500">규정 준수</div>
        </div>
      </div>

      {/* Comments */}
      {score.comments && score.comments.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-2">검토 의견</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            {score.comments.map((comment, index) => (
              <li key={index} className="flex items-start gap-2">
                <span className="text-gray-400">•</span>
                <span>{comment}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
