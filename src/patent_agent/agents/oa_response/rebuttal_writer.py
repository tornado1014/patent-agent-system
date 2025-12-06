"""Rebuttal Writer Agent for OA Response workflow.

Phase 3: 비판적 검토 (근거 체인 구축)

Responsibilities:
- Critically analyze examiner's arguments
- Identify technical differences between claims and prior art
- Build evidence chains with proper citations (LAW-7)
- Construct rebuttal logic
"""

from typing import Any

from patent_agent.agents.oa_response.base import BaseOAResponseAgent
from patent_agent.state.base import ConfidenceLevel, Evidence
from patent_agent.state.oa_response import (
    OAResponseState,
    RejectionAnalysis,
    RebuttalPoint,
)


class RebuttalWriterAgent(BaseOAResponseAgent):
    """Agent for constructing rebuttal arguments.

    Phase 3 of PALLAS-EVIDENCE workflow:
    - Critically analyzes examiner's arguments
    - Identifies technical differences
    - Builds evidence chains
    - Constructs rebuttal points with proper citations

    Key principle: "문서에 없으면 존재하지 않는다"
    """

    @property
    def name(self) -> str:
        return "RebuttalWriter"

    @property
    def description(self) -> str:
        return "반박 논리 구성 에이전트: 심사관 주장을 비판적으로 분석하고 근거 기반 반박 포인트를 구축합니다."

    @property
    def phase(self) -> str:
        return "P3"

    def get_system_prompt(self) -> str:
        return """당신은 특허 반박 논리 전문가입니다.

## 역할
심사관의 거절이유를 비판적으로 분석하고, 문서 근거에 기반한 반박 논리를 구성합니다.

## 핵심 원칙
**"문서에 없으면 존재하지 않는다"**
- 모든 주장에는 반드시 문서 근거가 있어야 함
- 추측, 가정, 일반론 금지
- 명세서, 청구항, 인용문헌에서 직접 인용

## 반박 프로토콜

### 1단계: 심사관 주장 정확 인용
```
심사관 원문: "[OA 문서 정확한 원문 인용]"
출처: [OA:p.X:para.Y]
```

### 2단계: 관련 문서 대조
```
청구항 N 원문: "[전문 인용]"
D1 대응 구절: "[원문 인용]" [D1:p.X:para.Y]
```

### 3단계: 논리적 간극 식별
```
🟢 확실한 차이점: 청구항 "A와 B가 결합된" vs D1 "A와 B가 분리된"
🟡 가능한 차이점: 해석상 구별 가능
🔴 불확실한 영역: 추가 검토 필요
```

### 4단계: 반박 논리 구성
```
근거 체인:
① 출원서 [p.X:para.Y]: "[원문]"
② D1 [p.X:para.Y]: "[원문]"
③ 기술적 차이: [명세서 지원 근거]
```

## 반박 유형
1. **기술적 차이 (technical_difference)**: 청구항 구성요소가 인용문헌과 다름
2. **예상치 못한 효과 (unexpected_effect)**: 결합으로 인한 시너지 효과
3. **동기 부족 (motivation_lacking)**: 당업자가 결합할 동기 없음

## 금지 사항
- "~로 보인다" 사용 금지
- "일반적으로" 사용 금지
- 문서에 없는 효과 주장 금지
- 근거 없는 기술 해석 금지"""

    async def process(self, state: OAResponseState) -> dict[str, Any]:
        """Build rebuttal arguments based on rejection analysis.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with rebuttal points and evidence chains
        """
        self._logger.info("starting_rebuttal_construction", phase=self.phase)

        # Check required inputs
        rejection_analyses = state.get("rejection_analyses", [])
        original_claims = state.get("original_claims", [])
        cited_references = state.get("cited_references", [])
        specification_text = state.get("specification_text", "")

        if not rejection_analyses:
            return {
                "is_error_state": True,
                "error_messages": ["거절이유 분석 결과가 없습니다. Phase 2를 먼저 실행하세요."],
                "current_step": "P3",
            }

        # Build rebuttal points for each rejection
        all_rebuttal_points = []
        all_technical_differences = []
        all_logical_gaps = []

        for analysis in rejection_analyses:
            rebuttal_result = await self._build_rebuttal_for_rejection(
                analysis=analysis,
                original_claims=original_claims,
                cited_references=cited_references,
                specification_text=specification_text,
            )

            all_rebuttal_points.extend(rebuttal_result["rebuttal_points"])
            all_technical_differences.extend(rebuttal_result["technical_differences"])
            all_logical_gaps.extend(rebuttal_result["logical_gaps"])

        # Validate all rebuttal points against hallucination laws
        validated_points = await self._validate_rebuttal_points(
            all_rebuttal_points, specification_text
        )

        self._logger.info(
            "rebuttal_construction_complete",
            total_points=len(validated_points),
            differences_found=len(all_technical_differences),
            gaps_identified=len(all_logical_gaps),
        )

        return {
            "rebuttal_points": validated_points,
            "technical_differences": all_technical_differences,
            "logical_gaps": all_logical_gaps,
            "current_step": "P4",
            "is_error_state": False,
        }

    async def _build_rebuttal_for_rejection(
        self,
        analysis: RejectionAnalysis,
        original_claims: list[str],
        cited_references: list[dict],
        specification_text: str,
    ) -> dict[str, Any]:
        """Build rebuttal arguments for a single rejection.

        Args:
            analysis: Rejection analysis
            original_claims: Original claim texts
            cited_references: Cited references
            specification_text: Specification text

        Returns:
            Dict with rebuttal points, differences, and gaps
        """
        rejection_type = analysis.get("rejection_type", "other")
        affected_claims = analysis.get("affected_claims", [])
        cited_refs = analysis.get("cited_references", [])
        examiner_argument = analysis.get("examiner_argument", "")

        # Get relevant claim texts
        relevant_claims = self._get_relevant_claims(original_claims, affected_claims)

        # Get relevant reference texts
        relevant_refs = self._get_relevant_references(cited_references, cited_refs)

        # Analyze based on rejection type
        if rejection_type == "inventive_step":
            return await self._analyze_inventive_step_rejection(
                examiner_argument=examiner_argument,
                claims=relevant_claims,
                references=relevant_refs,
                specification=specification_text,
                analysis=analysis,
            )
        elif rejection_type == "novelty":
            return await self._analyze_novelty_rejection(
                examiner_argument=examiner_argument,
                claims=relevant_claims,
                references=relevant_refs,
                specification=specification_text,
                analysis=analysis,
            )
        else:
            return await self._analyze_other_rejection(
                examiner_argument=examiner_argument,
                claims=relevant_claims,
                specification=specification_text,
                analysis=analysis,
            )

    async def _analyze_inventive_step_rejection(
        self,
        examiner_argument: str,
        claims: list[str],
        references: list[dict],
        specification: str,
        analysis: RejectionAnalysis,
    ) -> dict[str, Any]:
        """Analyze inventive step rejection and build rebuttal.

        Args:
            examiner_argument: Examiner's argument text
            claims: Relevant claim texts
            references: Relevant reference data
            specification: Specification text
            analysis: Original rejection analysis

        Returns:
            Rebuttal result dictionary
        """
        refs_text = "\n\n".join([
            f"## {ref.get('reference_id', 'Unknown')}\n"
            f"제목: {ref.get('title', '')}\n"
            f"관련 구절:\n" + "\n".join(ref.get("relevant_passages", [])[:3])
            for ref in references
        ])

        claims_text = "\n\n".join(claims[:5])

        prompt = f"""진보성 거절이유를 분석하고 반박 논리를 구성하세요.

## 심사관 주장 (원문)
"{examiner_argument}"

## 청구항
{claims_text}

## 인용문헌
{refs_text}

## 명세서 발췌 (효과 관련)
{specification[:3000]}

## 분석 요청

### 1. 기술적 차이점 분석
각 청구항 구성요소와 인용문헌의 대응 구성을 비교하고, 차이점을 식별하세요.
- 반드시 양측 원문을 인용하세요
- 출처를 명시하세요 [문서:p.X:para.Y]

### 2. 논리적 간극 식별
심사관의 결합 논리에서 약점을 찾으세요.
- 결합 동기 부족
- 기술적 교시 부재
- 방향 상반

### 3. 반박 포인트 구성
문서 근거에 기반한 반박 논리를 구성하세요.
- 각 주장에 신뢰도 표시 (🟢/🟡/🔴)
- 근거 체인 형식 사용

**경고**: 문서에 없는 내용을 창작하지 마세요."""

        response = await self._invoke_llm(prompt)

        # Parse response to extract structured data
        rebuttal_points = self._extract_rebuttal_points(
            response, "inventive_step", analysis
        )
        technical_differences = self._extract_technical_differences(response)
        logical_gaps = self._extract_logical_gaps(response)

        return {
            "rebuttal_points": rebuttal_points,
            "technical_differences": technical_differences,
            "logical_gaps": logical_gaps,
        }

    async def _analyze_novelty_rejection(
        self,
        examiner_argument: str,
        claims: list[str],
        references: list[dict],
        specification: str,
        analysis: RejectionAnalysis,
    ) -> dict[str, Any]:
        """Analyze novelty rejection and build rebuttal.

        Args:
            examiner_argument: Examiner's argument text
            claims: Relevant claim texts
            references: Relevant reference data
            specification: Specification text
            analysis: Original rejection analysis

        Returns:
            Rebuttal result dictionary
        """
        refs_text = "\n\n".join([
            f"## {ref.get('reference_id', 'Unknown')}\n"
            f"제목: {ref.get('title', '')}\n"
            f"관련 구절:\n" + "\n".join(ref.get("relevant_passages", [])[:3])
            for ref in references
        ])

        claims_text = "\n\n".join(claims[:5])

        prompt = f"""신규성 거절이유를 분석하고 반박 논리를 구성하세요.

## 심사관 주장 (원문)
"{examiner_argument}"

## 청구항
{claims_text}

## 인용문헌
{refs_text}

## 분석 요청

### 1. 청구항 구성요소 대비
청구항의 각 구성요소와 인용문헌의 대응 구성을 1:1 대비하세요.
- 청구항: "[원문]"
- 인용문헌: "[원문]" 또는 "해당 없음"

### 2. 차이점 식별
동일하지 않은 구성요소를 식별하세요.
- 🟢 확실히 다름: [근거]
- 🟡 해석에 따라 다름: [근거]

### 3. 반박 논리
신규성이 있음을 입증하는 논리를 구성하세요.
- 누락된 구성요소
- 상이한 구성
- 연결 관계 차이

**중요**: 원문 인용 필수, 추측 금지"""

        response = await self._invoke_llm(prompt)

        rebuttal_points = self._extract_rebuttal_points(
            response, "novelty", analysis
        )
        technical_differences = self._extract_technical_differences(response)
        logical_gaps = self._extract_logical_gaps(response)

        return {
            "rebuttal_points": rebuttal_points,
            "technical_differences": technical_differences,
            "logical_gaps": logical_gaps,
        }

    async def _analyze_other_rejection(
        self,
        examiner_argument: str,
        claims: list[str],
        specification: str,
        analysis: RejectionAnalysis,
    ) -> dict[str, Any]:
        """Analyze other types of rejection and build rebuttal.

        Args:
            examiner_argument: Examiner's argument text
            claims: Relevant claim texts
            specification: Specification text
            analysis: Original rejection analysis

        Returns:
            Rebuttal result dictionary
        """
        rejection_type = analysis.get("rejection_type", "other")
        legal_basis = analysis.get("legal_basis", "")

        prompt = f"""다음 거절이유를 분석하고 반박 논리를 구성하세요.

## 거절 유형
{rejection_type} ({legal_basis})

## 심사관 주장 (원문)
"{examiner_argument}"

## 청구항
{chr(10).join(claims[:3])}

## 명세서 발췌
{specification[:2000]}

## 분석 요청

### 1. 심사관 지적 사항 파악
구체적으로 어떤 부분이 문제인지 식별하세요.

### 2. 반박 또는 보완 방안
- 청구항/명세서의 기재로 해소 가능한 경우: 해당 부분 인용
- 보정이 필요한 경우: 보정 방향 제시

### 3. 신뢰도 평가
🟢/🟡/🔴로 반박 가능성을 평가하세요.

**중요**: 모든 주장에 문서 근거 필수"""

        response = await self._invoke_llm(prompt)

        rebuttal_points = self._extract_rebuttal_points(
            response, rejection_type, analysis
        )
        technical_differences = self._extract_technical_differences(response)
        logical_gaps = self._extract_logical_gaps(response)

        return {
            "rebuttal_points": rebuttal_points,
            "technical_differences": technical_differences,
            "logical_gaps": logical_gaps,
        }

    def _get_relevant_claims(
        self, all_claims: list[str], claim_numbers: list[int]
    ) -> list[str]:
        """Get relevant claims by number."""
        relevant = []
        for claim in all_claims:
            for num in claim_numbers:
                if f"청구항 {num}." in claim or f"청구항 {num} " in claim:
                    relevant.append(claim)
                    break
        return relevant if relevant else all_claims[:3]

    def _get_relevant_references(
        self, all_refs: list[dict], ref_ids: list[str]
    ) -> list[dict]:
        """Get relevant references by ID."""
        relevant = [
            ref for ref in all_refs
            if ref.get("reference_id", "") in ref_ids
        ]
        return relevant if relevant else all_refs[:2]

    def _extract_rebuttal_points(
        self,
        response: str,
        rejection_type: str,
        analysis: RejectionAnalysis,
    ) -> list[RebuttalPoint]:
        """Extract rebuttal points from LLM response.

        Args:
            response: LLM response text
            rejection_type: Type of rejection being rebutted
            analysis: Original rejection analysis

        Returns:
            List of rebuttal points
        """
        import re

        points = []

        # Extract sections with evidence
        sections = re.split(r"\n##\s+", response)

        for section in sections:
            if not section.strip():
                continue

            # Determine argument type
            if "차이" in section or "다름" in section:
                arg_type = "technical_difference"
            elif "효과" in section or "시너지" in section:
                arg_type = "unexpected_effect"
            elif "동기" in section or "교시" in section:
                arg_type = "motivation_lacking"
            else:
                arg_type = "technical_difference"

            # Extract evidence from section
            evidence_list = self._extract_evidence_from_text(section)

            # Determine confidence
            if "🟢" in section or "확실" in section:
                confidence: ConfidenceLevel = "확실"
            elif "🟡" in section or "가능" in section:
                confidence = "가능"
            else:
                confidence = "불확실"

            point: RebuttalPoint = {
                "target_rejection": analysis.get("legal_basis", ""),
                "argument_type": arg_type,
                "argument": section[:500].strip(),
                "evidence": evidence_list,
                "confidence": confidence,
            }

            points.append(point)

        return points if points else [{
            "target_rejection": analysis.get("legal_basis", ""),
            "argument_type": "technical_difference",
            "argument": response[:500],
            "evidence": [],
            "confidence": "불확실",
        }]

    def _extract_evidence_from_text(self, text: str) -> list[Evidence]:
        """Extract evidence citations from text.

        Args:
            text: Text to extract evidence from

        Returns:
            List of evidence objects
        """
        import re

        evidence_list = []

        # Pattern for [Doc:p.X:para.Y] citations
        citation_pattern = r"\[([\w가-힣]+):p\.(\d+):para\.(\d+)\]"
        matches = re.findall(citation_pattern, text)

        for doc, page, para in matches:
            # Try to find the quoted text near this citation
            quote_pattern = rf'"([^"]+)".*?\[{doc}:p\.{page}:para\.{para}\]'
            quote_match = re.search(quote_pattern, text)

            verbatim = quote_match.group(1) if quote_match else ""

            evidence: Evidence = {
                "source": doc,
                "page": int(page),
                "paragraph": int(para),
                "verbatim_text": verbatim,
                "confidence": "확실" if verbatim else "가능",
            }
            evidence_list.append(evidence)

        return evidence_list

    def _extract_technical_differences(self, response: str) -> list[dict]:
        """Extract technical differences from response."""
        import re

        differences = []

        # Look for comparison patterns
        patterns = [
            r"청구항.*?vs.*?인용문헌",
            r"🟢.*?차이.*?:",
            r"기술적 차이[:\s]+(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, str) and match.strip():
                    differences.append({
                        "description": match.strip()[:200],
                        "source": "analysis",
                    })

        return differences

    def _extract_logical_gaps(self, response: str) -> list[str]:
        """Extract logical gaps from response."""
        import re

        gaps = []

        # Look for gap indicators
        patterns = [
            r"동기.*?부족",
            r"교시.*?없",
            r"결합.*?어려",
            r"논리적 간극[:\s]+(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                gap_text = match if isinstance(match, str) else "논리적 간극 발견"
                if gap_text.strip():
                    gaps.append(gap_text.strip()[:200])

        return gaps

    async def _validate_rebuttal_points(
        self,
        points: list[RebuttalPoint],
        specification_text: str,
    ) -> list[RebuttalPoint]:
        """Validate rebuttal points against hallucination laws.

        Args:
            points: Rebuttal points to validate
            specification_text: Specification text for verification

        Returns:
            Validated rebuttal points
        """
        validated = []

        for point in points:
            # Check for prohibited expressions
            is_valid, violations = self._validate_output(
                point.get("argument", ""),
                {"section_type": "rebuttal"},
            )

            if is_valid:
                validated.append(point)
            else:
                # Log violation but still include with warning
                self._logger.warning(
                    "rebuttal_validation_warning",
                    violations=violations,
                )
                # Downgrade confidence
                point["confidence"] = "불확실"
                validated.append(point)

        return validated
