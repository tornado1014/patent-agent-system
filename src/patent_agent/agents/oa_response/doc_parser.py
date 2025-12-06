"""Document Parser Agent for OA Response workflow.

Phase 1: 문서 수집 및 검증 (원문 보존 추출)

Responsibilities:
- Parse OA document, cited references, and original specification
- Extract verbatim text preserving exact wording (LAW-6)
- Verify document completeness
- Build document page/paragraph index
"""

from typing import Any

from patent_agent.agents.oa_response.base import BaseOAResponseAgent
from patent_agent.state.oa_response import (
    ApplicationInfo,
    CitedReference,
    OADocument,
    OAResponseState,
)


class DocParserAgent(BaseOAResponseAgent):
    """Agent for parsing and extracting information from OA-related documents.

    Phase 1 of PALLAS-EVIDENCE workflow:
    - Parses OA document (의견제출통지서)
    - Extracts cited references (D1, D2, etc.)
    - Preserves original text verbatim (LAW-6)
    - Builds document verification checklist
    """

    @property
    def name(self) -> str:
        return "DocParser"

    @property
    def description(self) -> str:
        return "문서 파싱 에이전트: OA 관련 문서를 분석하고 원문을 정확하게 추출합니다."

    @property
    def phase(self) -> str:
        return "P1"

    def get_system_prompt(self) -> str:
        return """당신은 특허 문서 파싱 전문가입니다.

## 역할
의견제출통지서(OA)와 관련 문서들을 분석하여 필요한 정보를 정확하게 추출합니다.

## 핵심 원칙: 원문 보존 (LAW-6)
- 모든 텍스트는 **원문 그대로** 추출해야 합니다
- 요약, 의역, 수정 절대 금지
- 오타나 문법 오류가 있어도 그대로 보존

## 추출 대상

### 1. 출원 정보 (ApplicationInfo)
- 출원번호
- 출원일
- 발명의 명칭
- 출원인
- 발명자
- 대리인

### 2. OA 문서 정보 (OADocument)
- OA 번호/발송일
- 심사관 정보
- 거절이유 원문 (전문)
- 거절된 청구항 번호 목록
- 총 페이지 수

### 3. 인용문헌 정보 (CitedReference)
각 인용문헌(D1, D2 등)에 대해:
- 문헌 번호
- 발명의 명칭
- 출원인
- 출원일/공개일
- 관련 구절 원문 (심사관이 인용한 부분)

### 4. 출원 명세서
- 청구항 원문 (전체)
- 관련 명세서 구절

## 출력 형식
구조화된 데이터로 정보를 정리하되, 모든 텍스트 필드는 원문 그대로 포함합니다.
페이지 번호와 단락 번호를 함께 기록하여 추후 인용 시 사용할 수 있도록 합니다."""

    async def process(self, state: OAResponseState) -> dict[str, Any]:
        """Parse documents and extract information.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with parsed document information
        """
        self._logger.info("starting_document_parsing", phase=self.phase)

        # Check for required input documents
        oa_text = state.get("oa_document_text", "")
        spec_text = state.get("specification_text", "")

        if not oa_text:
            self._logger.warning("no_oa_document_provided")
            return {
                "is_error_state": True,
                "error_messages": ["OA 문서가 제공되지 않았습니다."],
                "current_step": "P1",
            }

        # Parse OA document
        oa_document = await self._parse_oa_document(oa_text)

        # Parse application info
        application_info = await self._parse_application_info(oa_text, spec_text)

        # Parse cited references
        cited_refs_text = state.get("cited_references_text", {})
        cited_references = await self._parse_cited_references(cited_refs_text)

        # Parse original claims
        original_claims = await self._extract_original_claims(spec_text)

        # Build document verification status
        documents_verified = self._verify_documents(
            oa_document, application_info, cited_references
        )

        # Calculate total pages by document
        total_pages_by_doc = self._calculate_total_pages(
            oa_text, spec_text, cited_refs_text
        )

        self._logger.info(
            "document_parsing_complete",
            cited_refs_count=len(cited_references),
            claims_count=len(original_claims),
            verified=all(documents_verified.values()),
        )

        return {
            "application_info": application_info,
            "oa_document": oa_document,
            "cited_references": cited_references,
            "original_claims": original_claims,
            "documents_verified": documents_verified,
            "total_pages_by_doc": total_pages_by_doc,
            "current_step": "P2",
            "is_error_state": False,
        }

    async def _parse_oa_document(self, oa_text: str) -> OADocument:
        """Parse OA document and extract structured information.

        Args:
            oa_text: Raw OA document text

        Returns:
            Parsed OA document structure
        """
        prompt = f"""다음 의견제출통지서를 분석하여 정보를 추출하세요.

## 원문
{oa_text}

## 추출할 정보
1. OA 번호 또는 발송일
2. 심사관 이름
3. 거절이유 원문 (전문 - 수정 없이)
4. 거절된 청구항 번호 목록
5. 법적 근거 (예: 특허법 제29조 제2항)

각 정보를 명확하게 구분하여 제공하세요.
거절이유는 반드시 **원문 그대로** 포함하세요."""

        response = await self._invoke_llm(prompt)

        # Parse LLM response to extract structured data
        # In production, would use structured output or parsing
        oa_doc: OADocument = {
            "oa_number": self._extract_field(response, "OA 번호", ""),
            "issue_date": self._extract_field(response, "발송일", ""),
            "examiner": self._extract_field(response, "심사관", ""),
            "rejection_reasons": self._extract_rejection_reasons(response),
            "cited_claims": self._extract_claim_numbers(response),
            "total_pages": self._estimate_pages(oa_text),
        }

        return oa_doc

    async def _parse_application_info(
        self, oa_text: str, spec_text: str
    ) -> ApplicationInfo:
        """Parse application information from documents.

        Args:
            oa_text: OA document text
            spec_text: Specification text

        Returns:
            Application information structure
        """
        combined_text = f"OA 문서:\n{oa_text[:2000]}\n\n명세서:\n{spec_text[:2000]}"

        prompt = f"""다음 문서에서 출원 정보를 추출하세요.

{combined_text}

## 추출할 정보
1. 출원번호
2. 출원일
3. 발명의 명칭
4. 출원인
5. 발명자
6. 대리인 (있는 경우)
7. 의견서 제출기한"""

        response = await self._invoke_llm(prompt)

        app_info: ApplicationInfo = {
            "application_number": self._extract_field(response, "출원번호", ""),
            "filing_date": self._extract_field(response, "출원일", ""),
            "title": self._extract_field(response, "발명의 명칭", ""),
            "applicant": self._extract_field(response, "출원인", ""),
            "inventor": self._extract_field(response, "발명자", ""),
            "agent": self._extract_field(response, "대리인", None),
            "response_deadline": self._extract_field(response, "제출기한", ""),
        }

        return app_info

    async def _parse_cited_references(
        self, cited_refs_text: dict[str, str]
    ) -> list[CitedReference]:
        """Parse cited reference documents.

        Args:
            cited_refs_text: Dict of reference ID -> document text

        Returns:
            List of parsed cited references
        """
        cited_refs = []

        for ref_id, ref_text in cited_refs_text.items():
            prompt = f"""다음 인용문헌 {ref_id}를 분석하세요.

{ref_text[:3000]}

## 추출할 정보
1. 문헌번호 (공개번호 또는 등록번호)
2. 발명의 명칭
3. 출원인
4. 출원일
5. 공개일/등록일
6. 주요 기술 내용 (심사관이 인용할 만한 구절)

기술 내용은 **원문 그대로** 추출하세요."""

            response = await self._invoke_llm(prompt)

            cited_ref: CitedReference = {
                "reference_id": ref_id,
                "document_number": self._extract_field(response, "문헌번호", ""),
                "title": self._extract_field(response, "발명의 명칭", ""),
                "applicant": self._extract_field(response, "출원인", ""),
                "filing_date": self._extract_field(response, "출원일", ""),
                "publication_date": self._extract_field(response, "공개일", ""),
                "relevant_passages": self._extract_passages(response),
                "total_pages": self._estimate_pages(ref_text),
            }

            cited_refs.append(cited_ref)

        return cited_refs

    async def _extract_original_claims(self, spec_text: str) -> list[str]:
        """Extract original claims from specification.

        Args:
            spec_text: Specification text

        Returns:
            List of original claim texts (verbatim)
        """
        prompt = f"""다음 명세서에서 청구항을 모두 추출하세요.

{spec_text}

## 중요
- 각 청구항을 **원문 그대로** 추출
- 번호, 구두점, 띄어쓰기 모두 보존
- 수정, 요약, 의역 절대 금지

청구항 1, 청구항 2 순서로 각각 전문을 제공하세요."""

        response = await self._invoke_llm(prompt)

        # Extract claims from response
        claims = self._parse_claims_from_response(response)

        return claims

    def _verify_documents(
        self,
        oa_document: OADocument,
        application_info: ApplicationInfo,
        cited_references: list[CitedReference],
    ) -> dict[str, bool]:
        """Verify document completeness.

        Returns:
            Dict of document name -> verified status
        """
        verified = {}

        # Verify OA document
        verified["oa_document"] = bool(
            oa_document.get("rejection_reasons")
            and oa_document.get("cited_claims")
        )

        # Verify application info
        verified["application_info"] = bool(
            application_info.get("application_number")
            and application_info.get("title")
        )

        # Verify cited references
        for ref in cited_references:
            ref_id = ref.get("reference_id", "unknown")
            verified[f"cited_ref_{ref_id}"] = bool(
                ref.get("document_number")
            )

        return verified

    def _calculate_total_pages(
        self,
        oa_text: str,
        spec_text: str,
        cited_refs_text: dict[str, str],
    ) -> dict[str, int]:
        """Calculate total pages by document."""
        pages = {
            "oa_document": self._estimate_pages(oa_text),
            "specification": self._estimate_pages(spec_text),
        }

        for ref_id, ref_text in cited_refs_text.items():
            pages[f"cited_ref_{ref_id}"] = self._estimate_pages(ref_text)

        return pages

    def _estimate_pages(self, text: str, chars_per_page: int = 2000) -> int:
        """Estimate page count from text length."""
        if not text:
            return 0
        return max(1, len(text) // chars_per_page)

    def _extract_field(
        self, text: str, field_name: str, default: str | None
    ) -> str | None:
        """Extract a field value from text."""
        import re

        patterns = [
            rf"{field_name}\s*[:\-]\s*(.+?)(?:\n|$)",
            rf"{field_name}\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return default

    def _extract_rejection_reasons(self, text: str) -> list[str]:
        """Extract rejection reasons from parsed text."""
        import re

        reasons = []

        # Look for numbered reasons or sections
        patterns = [
            r"거절이유[:\s]*(.+?)(?=\n\n|\Z)",
            r"(\d+\.\s*.+?(?=\n\d+\.|\Z))",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                reasons.extend([m.strip() for m in matches if m.strip()])
                break

        return reasons if reasons else [text]

    def _extract_claim_numbers(self, text: str) -> list[int]:
        """Extract claim numbers from text."""
        import re

        # Find patterns like "청구항 1, 2, 3" or "claims 1-5"
        numbers = []

        # Pattern for explicit claim mentions
        matches = re.findall(r"청구항\s*(\d+(?:\s*,\s*\d+)*)", text)
        for match in matches:
            for num in re.findall(r"\d+", match):
                numbers.append(int(num))

        # Pattern for ranges
        range_matches = re.findall(r"청구항\s*(\d+)\s*[-~]\s*(\d+)", text)
        for start, end in range_matches:
            numbers.extend(range(int(start), int(end) + 1))

        return sorted(set(numbers)) if numbers else [1]

    def _extract_passages(self, text: str) -> list[str]:
        """Extract relevant passages from text."""
        import re

        passages = []

        # Look for quoted text
        quoted = re.findall(r'"([^"]+)"', text)
        passages.extend(quoted)

        # Look for marked passages
        marked = re.findall(r"(?:구절|인용|관련\s*부분)[:\s]*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
        passages.extend([m.strip() for m in marked if m.strip()])

        return passages

    def _parse_claims_from_response(self, response: str) -> list[str]:
        """Parse individual claims from LLM response."""
        import re

        claims = []

        # Pattern for "청구항 N. ..." format
        pattern = r"청구항\s*(\d+)\.\s*(.+?)(?=청구항\s*\d+\.|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)

        for num, text in matches:
            full_claim = f"청구항 {num}. {text.strip()}"
            claims.append(full_claim)

        return claims if claims else [response.strip()]
