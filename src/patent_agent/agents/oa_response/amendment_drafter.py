"""Amendment Drafter Agent for OA Response workflow.

Phase 4: 보정안 생성 (명세서 지원 검증)

Responsibilities:
- Generate at least 2 amendment options (LAW-3)
- Verify all limitations are supported by specification
- Check for new matter addition
- Assess amendment effectiveness
"""

from typing import Any

from patent_agent.agents.oa_response.base import BaseOAResponseAgent
from patent_agent.rules.oa_response_laws import AmendmentGuideline
from patent_agent.state.base import ConfidenceLevel, Evidence
from patent_agent.state.oa_response import (
    Amendment,
    OAResponseState,
    RebuttalPoint,
)


class AmendmentDrafterAgent(BaseOAResponseAgent):
    """Agent for drafting claim amendments.

    Phase 4 of PALLAS-EVIDENCE workflow:
    - Generates at least 2 amendment options
    - Verifies specification support for all limitations
    - Checks for new matter addition
    - Assesses amendment effectiveness

    Requirements:
    - LAW-3: Minimum 2 amendment options
    - LAW-6: Original claim text preserved verbatim
    - LAW-8: No hallucinated limitations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use specialized amendment guidelines
        self._guidelines = AmendmentGuideline()

    @property
    def name(self) -> str:
        return "AmendmentDrafter"

    @property
    def description(self) -> str:
        return "보정안 작성 에이전트: 명세서에 근거한 청구항 보정안을 최소 2개 이상 생성합니다."

    @property
    def phase(self) -> str:
        return "P4"

    def get_system_prompt(self) -> str:
        return """당신은 특허 청구항 보정 전문가입니다.

## 역할
거절이유를 해소하기 위한 청구항 보정안을 작성합니다.
모든 보정 내용은 명세서에 의해 뒷받침되어야 합니다.

## 핵심 원칙

### 1. 최소 2개 보정안 필수 (LAW-3)
각 보정안은 서로 다른 전략을 사용해야 합니다:
- 보정안 A: 보수적 접근 (최소 한정)
- 보정안 B: 적극적 접근 (구체적 한정)
- 보정안 C: (선택) 대안적 접근

### 2. 명세서 지원 필수
- 모든 추가 한정사항은 명세서에서 근거를 찾아야 함
- 도면 부호, 페이지, 단락 명시
- 원문 인용 필수

### 3. 신규사항 금지
- 명세서에 명시적으로 기재되지 않은 내용 추가 금지
- 도면에 없는 구성 추가 금지
- 효과의 과장 또는 창작 금지

## 보정안 형식

```
━━━━━━━━━━━━━━━━━━━━━━━━
보정안 [A/B/C]: [보정 전략 명칭]
━━━━━━━━━━━━━━━━━━━━━━━━

[현재 청구항]
청구항 N. "[원문 전체 - 수정 불가]"

[보정 후 청구항]
청구항 N. "[보정된 전문]"

[보정 근거 - 명세서 매핑]
추가 한정 1: "[한정사항]"
└─ 근거: [명세서:p.X:para.Y] "[원문 인용]"
└─ 도면: [도면N:부호XXX] "[설명 인용]"

[신규사항 체크]
✅ 명세서 명시적 기재: YES/NO
✅ 도면 지원: YES/NO
⚠️ 신규사항 리스크: LOW/MEDIUM/HIGH

[보정 효과]
- 거절이유 해소: 🟢확실/🟡가능/🔴불확실
- 진보성 논리: "[명세서 기재 효과 인용]"
```

## 금지 사항
- 명세서에 없는 구성요소 추가
- 도면에 없는 특징 창작
- 효과의 과장 또는 창작
- 실시예 없는 조합 제안"""

    async def process(self, state: OAResponseState) -> dict[str, Any]:
        """Generate amendment options.

        Args:
            state: Current workflow state

        Returns:
            Dictionary with amendment options
        """
        self._logger.info("starting_amendment_drafting", phase=self.phase)

        # Check required inputs
        rebuttal_points = state.get("rebuttal_points", [])
        original_claims = state.get("original_claims", [])
        specification_text = state.get("specification_text", "")
        rejection_analyses = state.get("rejection_analyses", [])

        if not original_claims:
            return {
                "is_error_state": True,
                "error_messages": ["원 청구항 정보가 없습니다."],
                "current_step": "P4",
            }

        if not specification_text:
            return {
                "is_error_state": True,
                "error_messages": ["명세서 텍스트가 없습니다."],
                "current_step": "P4",
            }

        # Identify claims that need amendment
        claims_to_amend = self._identify_claims_to_amend(rejection_analyses)

        # Generate amendments for each claim
        all_amendments = []

        for claim_num in claims_to_amend:
            claim_text = self._get_claim_by_number(original_claims, claim_num)
            if not claim_text:
                continue

            amendments = await self._generate_amendments_for_claim(
                claim_number=claim_num,
                claim_text=claim_text,
                specification=specification_text,
                rebuttal_points=rebuttal_points,
                rejection_analyses=rejection_analyses,
            )

            all_amendments.extend(amendments)

        # Ensure minimum 2 amendments (LAW-3)
        if len(all_amendments) < 2:
            self._logger.warning(
                "insufficient_amendments",
                count=len(all_amendments),
                required=2,
            )
            # Generate additional amendment if needed
            if all_amendments:
                additional = await self._generate_alternative_amendment(
                    all_amendments[0], specification_text
                )
                if additional:
                    all_amendments.append(additional)

        # Determine recommended amendment
        recommended = self._recommend_amendment(all_amendments)

        self._logger.info(
            "amendment_drafting_complete",
            total_amendments=len(all_amendments),
            recommended=recommended,
        )

        return {
            "amendment_options": all_amendments,
            "recommended_amendment": recommended,
            "current_step": "P5",
            "is_error_state": False,
        }

    def _identify_claims_to_amend(
        self, rejection_analyses: list[dict]
    ) -> list[int]:
        """Identify which claims need amendment.

        Args:
            rejection_analyses: List of rejection analyses

        Returns:
            List of claim numbers that need amendment
        """
        claims = set()
        for analysis in rejection_analyses:
            claims.update(analysis.get("affected_claims", []))
        return sorted(claims)

    def _get_claim_by_number(
        self, claims: list[str], claim_number: int
    ) -> str | None:
        """Get claim text by number.

        Args:
            claims: List of claim texts
            claim_number: Claim number to find

        Returns:
            Claim text or None if not found
        """
        for claim in claims:
            if f"청구항 {claim_number}." in claim or f"청구항 {claim_number} " in claim:
                return claim
        return None

    async def _generate_amendments_for_claim(
        self,
        claim_number: int,
        claim_text: str,
        specification: str,
        rebuttal_points: list[RebuttalPoint],
        rejection_analyses: list[dict],
    ) -> list[Amendment]:
        """Generate amendment options for a single claim.

        Args:
            claim_number: Claim number
            claim_text: Original claim text
            specification: Specification text
            rebuttal_points: Rebuttal points
            rejection_analyses: Rejection analyses

        Returns:
            List of amendment options
        """
        # Get relevant rejection for this claim
        relevant_rejection = None
        for analysis in rejection_analyses:
            if claim_number in analysis.get("affected_claims", []):
                relevant_rejection = analysis
                break

        # Get relevant rebuttal points
        relevant_rebuttals = [
            p for p in rebuttal_points
            if p.get("target_rejection") == relevant_rejection.get("legal_basis", "")
        ] if relevant_rejection else []

        prompt = f"""청구항 {claim_number}에 대한 보정안을 작성하세요.

## 현재 청구항 (원문 - LAW-6)
"{claim_text}"

## 거절이유
{relevant_rejection.get('examiner_argument', '정보 없음') if relevant_rejection else '정보 없음'}

## 명세서 (보정 근거 검색용)
{specification[:5000]}

## 요청사항

### 보정안 A: 보수적 접근
- 최소한의 한정 추가
- 권리범위 최대 유지
- 명세서에서 핵심 특징 1-2개만 추가

### 보정안 B: 적극적 접근
- 구체적인 한정 추가
- 거절이유 확실한 해소
- 명세서의 구체적 실시예 활용

각 보정안에 대해:
1. [현재 청구항] - 원문 그대로 (수정 불가)
2. [보정 후 청구항] - 보정된 전체 청구항
3. [보정 근거] - 명세서 인용 (페이지, 단락, 원문)
4. [신규사항 체크] - YES/NO
5. [보정 효과] - 🟢/🟡/🔴

**중요**: 명세서에 없는 내용 절대 추가 금지"""

        response = await self._invoke_llm(prompt)

        # Parse response to extract amendments
        amendments = self._parse_amendments(response, claim_number, claim_text)

        # Verify specification support for each amendment
        verified_amendments = []
        for amendment in amendments:
            verified = await self._verify_specification_support(
                amendment, specification
            )
            if verified:
                verified_amendments.append(amendment)
            else:
                # Still include but mark as higher risk
                amendment["new_matter_risk"] = "high"
                verified_amendments.append(amendment)

        return verified_amendments

    def _parse_amendments(
        self,
        response: str,
        claim_number: int,
        original_claim: str,
    ) -> list[Amendment]:
        """Parse amendments from LLM response.

        Args:
            response: LLM response text
            claim_number: Claim number
            original_claim: Original claim text

        Returns:
            List of Amendment objects
        """
        import re

        amendments = []

        # Split by amendment headers
        amendment_sections = re.split(r"보정안\s*([A-C])", response)

        for i in range(1, len(amendment_sections), 2):
            if i + 1 >= len(amendment_sections):
                break

            amendment_id = amendment_sections[i].strip()
            section = amendment_sections[i + 1]

            # Extract strategy name
            strategy_match = re.search(r"[:\s]+(.+?)(?:\n|$)", section)
            strategy_name = strategy_match.group(1).strip() if strategy_match else f"보정안 {amendment_id}"

            # Extract amended claim
            amended_match = re.search(
                r"\[보정\s*후\s*청구항\].*?청구항\s*\d+\.\s*(.+?)(?=\[|$)",
                section,
                re.DOTALL,
            )
            amended_claim = amended_match.group(1).strip() if amended_match else ""

            # Extract added limitations
            limitations = self._extract_limitations(section)

            # Extract specification support
            support = self._extract_specification_support(section)

            # Determine new matter risk
            risk = self._assess_new_matter_risk(section)

            # Determine effectiveness
            effectiveness = self._assess_effectiveness(section)

            amendment: Amendment = {
                "amendment_id": amendment_id,
                "strategy_name": strategy_name,
                "original_claim": original_claim,
                "amended_claim": f"청구항 {claim_number}. {amended_claim}" if amended_claim else "",
                "added_limitations": limitations,
                "specification_support": support,
                "new_matter_risk": risk,
                "effectiveness": effectiveness,
            }

            if amendment["amended_claim"]:
                amendments.append(amendment)

        return amendments

    def _extract_limitations(self, section: str) -> list[str]:
        """Extract added limitations from section."""
        import re

        limitations = []
        patterns = [
            r"추가\s*한정[:\s]*(.+?)(?:\n|└|$)",
            r"한정사항[:\s]*(.+?)(?:\n|└|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, section)
            limitations.extend([m.strip() for m in matches if m.strip()])

        return limitations

    def _extract_specification_support(self, section: str) -> list[Evidence]:
        """Extract specification support evidence from section."""
        import re

        support = []

        # Pattern for [명세서:p.X:para.Y] citations
        pattern = r"\[(명세서|출원서):p\.(\d+):para\.(\d+)\]\s*[\"']([^\"']+)[\"']"
        matches = re.findall(pattern, section)

        for doc, page, para, text in matches:
            evidence: Evidence = {
                "source": doc,
                "page": int(page),
                "paragraph": int(para),
                "verbatim_text": text,
                "confidence": "확실",
            }
            support.append(evidence)

        return support

    def _assess_new_matter_risk(self, section: str) -> str:
        """Assess new matter risk from section."""
        section_lower = section.lower()

        if "신규사항 리스크: high" in section_lower or "⚠️" in section:
            return "high"
        elif "신규사항 리스크: medium" in section_lower:
            return "medium"
        elif "yes" in section_lower and "명세서" in section:
            return "low"
        else:
            return "medium"

    def _assess_effectiveness(self, section: str) -> ConfidenceLevel:
        """Assess amendment effectiveness from section."""
        if "🟢" in section or "확실" in section:
            return "확실"
        elif "🟡" in section or "가능" in section:
            return "가능"
        else:
            return "불확실"

    async def _verify_specification_support(
        self,
        amendment: Amendment,
        specification: str,
    ) -> bool:
        """Verify that amendment is supported by specification.

        Args:
            amendment: Amendment to verify
            specification: Specification text

        Returns:
            True if properly supported
        """
        added_limitations = amendment.get("added_limitations", [])

        if not added_limitations:
            return True

        # Check if limitation text appears in specification
        for limitation in added_limitations:
            # Extract key terms from limitation
            key_terms = [
                term for term in limitation.split()
                if len(term) > 2 and term not in ["상기", "포함", "하는", "있는"]
            ]

            found = False
            for term in key_terms:
                if term in specification:
                    found = True
                    break

            if not found:
                self._logger.warning(
                    "limitation_not_found_in_spec",
                    limitation=limitation[:50],
                )
                return False

        return True

    async def _generate_alternative_amendment(
        self,
        existing_amendment: Amendment,
        specification: str,
    ) -> Amendment | None:
        """Generate an alternative amendment when fewer than 2 exist.

        Args:
            existing_amendment: Existing amendment to contrast with
            specification: Specification text

        Returns:
            Alternative amendment or None
        """
        original_claim = existing_amendment.get("original_claim", "")
        existing_strategy = existing_amendment.get("strategy_name", "")

        prompt = f"""다른 전략의 보정안을 제안하세요.

## 현재 청구항
"{original_claim}"

## 기존 보정안 전략
{existing_strategy}

## 명세서
{specification[:3000]}

## 요청
기존 전략과 다른 새로운 보정안을 작성하세요.
- 다른 한정사항 사용
- 다른 구성요소 강조
- 명세서 근거 필수

형식은 동일하게:
[보정 후 청구항], [보정 근거], [신규사항 체크], [보정 효과]"""

        response = await self._invoke_llm(prompt)

        # Parse single amendment
        amendments = self._parse_amendments(
            f"보정안 B: 대안적 접근\n{response}",
            1,  # Default claim number
            original_claim,
        )

        return amendments[0] if amendments else None

    def _recommend_amendment(self, amendments: list[Amendment]) -> str | None:
        """Recommend the best amendment option.

        Args:
            amendments: List of amendments

        Returns:
            Recommended amendment ID or None
        """
        if not amendments:
            return None

        # Score each amendment
        scored = []
        for amendment in amendments:
            score = 0

            # Effectiveness score
            effectiveness = amendment.get("effectiveness", "불확실")
            if effectiveness == "확실":
                score += 30
            elif effectiveness == "가능":
                score += 15

            # New matter risk score (lower is better)
            risk = amendment.get("new_matter_risk", "medium")
            if risk == "low":
                score += 20
            elif risk == "medium":
                score += 10

            # Specification support score
            support = amendment.get("specification_support", [])
            score += min(len(support) * 5, 20)

            scored.append((amendment.get("amendment_id", ""), score))

        # Return highest scored
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None
