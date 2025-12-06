"""OA Response agents module.

5-phase workflow based on PALLAS-EVIDENCE v3.0:
- P1: DocParserAgent - 문서 수집 및 검증
- P2: RejectionAnalyzerAgent - 거절이유 분석
- P3: RebuttalWriterAgent - 비판적 검토
- P4: AmendmentDrafterAgent - 보정안 생성
- P5: ReportGeneratorAgent - 최종 보고서
"""

from patent_agent.agents.oa_response.base import BaseOAResponseAgent
from patent_agent.agents.oa_response.doc_parser import DocParserAgent
from patent_agent.agents.oa_response.rejection_analyzer import RejectionAnalyzerAgent
from patent_agent.agents.oa_response.rebuttal_writer import RebuttalWriterAgent
from patent_agent.agents.oa_response.amendment_drafter import AmendmentDrafterAgent
from patent_agent.agents.oa_response.report_generator import ReportGeneratorAgent

__all__ = [
    "BaseOAResponseAgent",
    "DocParserAgent",
    "RejectionAnalyzerAgent",
    "RebuttalWriterAgent",
    "AmendmentDrafterAgent",
    "ReportGeneratorAgent",
]
