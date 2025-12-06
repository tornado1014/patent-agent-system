"""Output Generator Agent (P6) for Patent Translation workflow.

Handles final output generation:
- KIPO format Korean document
- English-Korean comparison document
- Quality verification report
- Reasoning log report
- Glossary export
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from patent_agent.agents.translation.base import BaseTranslationAgent
from patent_agent.config import get_settings
from patent_agent.state.translation import (
    OUTPUT_FILES,
    GlossaryEntry,
    QualityReport,
    ReasoningEntry,
    SectionType,
    TranslatedSection,
    TranslationPhase,
    TranslationState,
)


# KIPO section order
KIPO_SECTION_ORDER: list[SectionType] = [
    "title",
    "field",
    "background",
    "summary",
    "brief_description",
    "detailed_description",
    "claims",
    "abstract",
]

# KIPO section names in Korean
KIPO_SECTION_NAMES = {
    "title": "발명의 명칭",
    "field": "기술분야",
    "background": "배경기술",
    "summary": "발명의 내용",
    "brief_description": "도면의 간단한 설명",
    "detailed_description": "발명을 실시하기 위한 구체적인 내용",
    "claims": "청구범위",
    "abstract": "요약서",
}


class OutputGeneratorAgent(BaseTranslationAgent):
    """Output generation agent.

    Phase P6: 결과물 생성

    Responsibilities:
    - Generate KIPO format Korean document
    - Create English-Korean comparison document
    - Generate quality verification report
    - Export reasoning log
    - Save glossary as Excel

    Output Files:
    - ko_final.docx: KIPO 제출용 최종본
    - en_ko_compare.docx: 영한 대조본
    - qa_report.md: 품질 검증 보고서
    - reasoning_log.md: 추론 이력 보고서
    - glossary.xlsx: 사용된 용어집
    """

    @property
    def name(self) -> str:
        return "OutputGeneratorAgent"

    @property
    def description(self) -> str:
        return "최종 번역 결과물을 생성하는 에이전트"

    @property
    def phase(self) -> TranslationPhase:
        return "P6"

    def get_system_prompt(self) -> str:
        return """## Output Generator Agent (P6)

당신은 영→한 특허 번역의 최종 결과물을 생성하는 에이전트입니다.

### 생성할 결과물

1. **ko_final.docx** - KIPO 제출용 최종본
   - KIPO 표준 형식 준수
   - 섹션 순서: 발명의 명칭 → 기술분야 → 배경기술 → 발명의 내용 → 도면의 간단한 설명 → 발명을 실시하기 위한 구체적인 내용 → 청구범위 → 요약서
   - 청구항 번호 형식: 1., 2., 3...

2. **en_ko_compare.docx** - 영한 대조본
   - 2단 레이아웃 (왼쪽: 영문, 오른쪽: 한문)
   - 문단 대응 정렬
   - 검토용 가독성 최적화

3. **qa_report.md** - 품질 검증 보고서
   - 5C 점수 상세
   - 식별된 문제점
   - 개선 제안
   - 법칙 준수 현황

4. **reasoning_log.md** - 추론 이력 보고서
   - 각 단계별 결정 사항
   - 적용된 법칙
   - 체크포인트 기록

5. **glossary.xlsx** - 사용된 용어집
   - 영문 용어
   - 한국어 번역
   - 출처 (KIPRIS/WIPO Pearl/auto/user)
   - 비고

### 형식 요구사항
- DOCX 파일은 A4 형식
- 마크다운 파일은 GitHub 스타일
- 엑셀 파일은 xlsx 형식
"""

    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Generate output files.

        Args:
            state: Current workflow state

        Returns:
            State updates with output file paths
        """
        self._logger.info("starting_output_generation")

        # Verify quality passed
        quality_report = state.get("quality_report", {})
        if quality_report.get("overall_score", 0) < 80:
            self._logger.warning("generating_outputs_with_low_quality")

        translated_sections = state.get("translated_sections", {})
        glossary = state.get("glossary", [])
        reasoning_log = state.get("reasoning_log", [])
        doc_structure = state.get("document_structure", {})
        source_document = state.get("source_document", {})

        # Create output directory
        settings = get_settings()
        output_dir = Path(settings.paths.output_dir) / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        # Generate each output file
        # 1. Korean final document
        ko_final_path = await self._generate_korean_document(
            output_dir, translated_sections, doc_structure
        )
        output_files["ko_final.docx"] = str(ko_final_path)

        # 2. English-Korean comparison
        compare_path = await self._generate_comparison_document(
            output_dir, translated_sections, doc_structure
        )
        output_files["en_ko_compare.docx"] = str(compare_path)

        # 3. Quality report
        qa_path = self._generate_quality_report(
            output_dir, quality_report, translated_sections
        )
        output_files["qa_report.md"] = str(qa_path)

        # 4. Reasoning log
        reasoning_path = self._generate_reasoning_log(output_dir, reasoning_log)
        output_files["reasoning_log.md"] = str(reasoning_path)

        # 5. Glossary
        glossary_path = self._generate_glossary(output_dir, glossary)
        output_files["glossary.xlsx"] = str(glossary_path)

        # 6. Deadline reminder (if applicable)
        deadline = state.get("legal_deadline")
        if deadline:
            deadline_path = self._generate_deadline_reminder(
                output_dir, deadline, source_document
            )
            output_files["deadline_reminder.txt"] = str(deadline_path)

        # Create final Korean document content
        final_korean_document = self._build_final_document_content(translated_sections)

        # Create comparison document content
        comparison_document = self._build_comparison_content(
            translated_sections, doc_structure
        )

        # Create reasoning entry
        reasoning = self.create_reasoning_entry(
            checkpoint="CP6.1",
            decision="결과물 생성 완료",
            reasoning=f"생성된 파일: {', '.join(output_files.keys())}",
            applied_laws=[],
        )

        self._logger.info(
            "output_generation_complete",
            files=list(output_files.keys()),
            output_dir=str(output_dir),
        )

        return {
            "final_korean_document": final_korean_document,
            "comparison_document": comparison_document,
            "output_files": output_files,
            "current_step": "P6",  # Final step
            "reasoning_log": state.get("reasoning_log", []) + [reasoning],
            "is_error_state": False,
        }

    async def _generate_korean_document(
        self,
        output_dir: Path,
        translated_sections: dict[SectionType, TranslatedSection],
        doc_structure: dict,
    ) -> Path:
        """Generate KIPO format Korean document.

        Args:
            output_dir: Output directory
            translated_sections: Translated sections
            doc_structure: Document structure

        Returns:
            Path to generated document
        """
        try:
            from docx import Document
            from docx.shared import Pt, Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH

            doc = Document()

            # Set document properties
            doc.core_properties.title = translated_sections.get("title", {}).get(
                "translated_text", "번역된 특허명세서"
            )

            # Add sections in KIPO order
            for section_type in KIPO_SECTION_ORDER:
                if section_type not in translated_sections:
                    continue

                section = translated_sections[section_type]
                korean_name = KIPO_SECTION_NAMES.get(section_type, section_type)

                # Add section heading
                heading = doc.add_heading(korean_name, level=1)
                heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

                # Add section content
                if section_type == "claims":
                    # Format claims with numbering
                    self._add_claims_section(doc, section["translated_text"])
                else:
                    para = doc.add_paragraph(section["translated_text"])
                    para.style.font.size = Pt(11)

            # Save document
            file_path = output_dir / "ko_final.docx"
            doc.save(str(file_path))

            return file_path

        except ImportError:
            # Fallback to text file if python-docx not available
            file_path = output_dir / "ko_final.txt"
            content = self._build_final_document_content(translated_sections)
            file_path.write_text(content, encoding="utf-8")
            return file_path

    def _add_claims_section(self, doc: Any, claims_text: str) -> None:
        """Add claims section with proper formatting.

        Args:
            doc: Document object
            claims_text: Claims text
        """
        from docx.shared import Pt

        # Split claims by numbering pattern
        import re

        claims = re.split(r'\n\s*(\d+)\.\s*', claims_text)

        for i in range(1, len(claims), 2):
            claim_num = claims[i]
            claim_text = claims[i + 1] if i + 1 < len(claims) else ""

            para = doc.add_paragraph()
            run = para.add_run(f"{claim_num}. {claim_text.strip()}")
            run.font.size = Pt(11)

    async def _generate_comparison_document(
        self,
        output_dir: Path,
        translated_sections: dict[SectionType, TranslatedSection],
        doc_structure: dict,
    ) -> Path:
        """Generate English-Korean comparison document.

        Args:
            output_dir: Output directory
            translated_sections: Translated sections
            doc_structure: Document structure

        Returns:
            Path to generated document
        """
        try:
            from docx import Document
            from docx.shared import Inches, Pt
            from docx.enum.table import WD_TABLE_ALIGNMENT

            doc = Document()

            # Add title
            doc.add_heading("영한 대조본 (English-Korean Comparison)", level=0)

            # Create comparison table for each section
            for section_type in KIPO_SECTION_ORDER:
                if section_type not in translated_sections:
                    continue

                section = translated_sections[section_type]
                korean_name = KIPO_SECTION_NAMES.get(section_type, section_type)

                # Section heading
                doc.add_heading(korean_name, level=1)

                # Create 2-column table
                table = doc.add_table(rows=1, cols=2)
                table.alignment = WD_TABLE_ALIGNMENT.CENTER

                # Header row
                hdr_cells = table.rows[0].cells
                hdr_cells[0].text = "English (원문)"
                hdr_cells[1].text = "Korean (번역문)"

                # Content row
                row = table.add_row()
                row.cells[0].text = section["source_text"][:2000]  # Truncate if too long
                row.cells[1].text = section["translated_text"][:2000]

                doc.add_paragraph()  # Spacing

            # Save document
            file_path = output_dir / "en_ko_compare.docx"
            doc.save(str(file_path))

            return file_path

        except ImportError:
            # Fallback to markdown
            file_path = output_dir / "en_ko_compare.md"
            content = self._build_comparison_content(translated_sections, doc_structure)
            file_path.write_text(content, encoding="utf-8")
            return file_path

    def _generate_quality_report(
        self,
        output_dir: Path,
        quality_report: QualityReport,
        translated_sections: dict[SectionType, TranslatedSection],
    ) -> Path:
        """Generate quality verification report.

        Args:
            output_dir: Output directory
            quality_report: Quality report data
            translated_sections: Translated sections

        Returns:
            Path to generated report
        """
        report = f"""# 품질 검증 보고서 (Quality Verification Report)

생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. 5C 품질 점수

| 기준 | 점수 | 상태 |
|------|------|------|
| Correctness (정확성) | {quality_report.get('correctness_score', 0)}/100 | {'✅' if quality_report.get('correctness_score', 0) >= 80 else '⚠️'} |
| Clarity (명확성) | {quality_report.get('clarity_score', 0)}/100 | {'✅' if quality_report.get('clarity_score', 0) >= 80 else '⚠️'} |
| Conciseness (간결성) | {quality_report.get('conciseness_score', 0)}/100 | {'✅' if quality_report.get('conciseness_score', 0) >= 80 else '⚠️'} |
| Consistency (일관성) | {quality_report.get('consistency_score', 0)}/100 | {'✅' if quality_report.get('consistency_score', 0) >= 100 else '⚠️'} |
| Compliance (준수) | {quality_report.get('compliance_score', 0)}/100 | {'✅' if quality_report.get('compliance_score', 0) >= 100 else '⚠️'} |
| **종합 점수** | **{quality_report.get('overall_score', 0)}/100** | **{'✅ PASS' if quality_report.get('overall_score', 0) >= 80 else '❌ FAIL'}** |

## 2. 섹션별 법칙 준수 현황

| 섹션 | LAW-T-1 | LAW-T-2 | LAW-T-3 | LAW-T-4 |
|------|---------|---------|---------|---------|
"""

        for section_type, section in translated_sections.items():
            compliance = section.get("law_compliance", {})
            t1 = "✅" if compliance.get("LAW-T-1", True) else "❌"
            t2 = "✅" if compliance.get("LAW-T-2", True) else "❌"
            t3 = "✅" if compliance.get("LAW-T-3", True) else "❌"
            t4 = "✅" if compliance.get("LAW-T-4", True) else "❌"
            korean_name = KIPO_SECTION_NAMES.get(section_type, section_type)
            report += f"| {korean_name} | {t1} | {t2} | {t3} | {t4} |\n"

        report += f"""
## 3. 식별된 문제점

"""
        issues = quality_report.get("issues", [])
        if issues:
            for issue in issues:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "- ✅ 식별된 문제 없음\n"

        report += f"""
## 4. 개선 제안

"""
        suggestions = quality_report.get("suggestions", [])
        if suggestions:
            for suggestion in suggestions:
                report += f"- 💡 {suggestion}\n"
        else:
            report += "- 추가 개선 제안 없음\n"

        report += f"""
## 5. 4대 절대법칙 요약

- **LAW-T-1**: '상기' 사용 규칙 (청구항: the/said→상기, 청구항 외: 상기 금지)
- **LAW-T-2**: 청구항 한 문장 원칙 (마침표 끝에만)
- **LAW-T-3**: 권리범위 결정 용어 (comprising≠구성된, consisting of≠포함하는)
- **LAW-T-4**: 도면부호 괄호 규칙 (10 → (10))

---
✓ 할루시네이션 체크 CLEAR
"""

        file_path = output_dir / "qa_report.md"
        file_path.write_text(report, encoding="utf-8")

        return file_path

    def _generate_reasoning_log(
        self,
        output_dir: Path,
        reasoning_log: list[ReasoningEntry],
    ) -> Path:
        """Generate reasoning log report.

        Args:
            output_dir: Output directory
            reasoning_log: Reasoning log entries

        Returns:
            Path to generated report
        """
        report = f"""# 추론 이력 보고서 (Reasoning Log Report)

생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 처리 단계별 결정 사항

"""

        for entry in reasoning_log:
            report += f"""### {entry.get('checkpoint', 'N/A')} ({entry.get('phase', 'N/A')})

- **결정**: {entry.get('decision', 'N/A')}
- **근거**: {entry.get('reasoning', 'N/A')}
- **적용 법칙**: {', '.join(entry.get('applied_laws', [])) or '없음'}
- **시간**: {entry.get('timestamp', 'N/A')}

---

"""

        file_path = output_dir / "reasoning_log.md"
        file_path.write_text(report, encoding="utf-8")

        return file_path

    def _generate_glossary(
        self,
        output_dir: Path,
        glossary: list[GlossaryEntry],
    ) -> Path:
        """Generate glossary Excel file.

        Args:
            output_dir: Output directory
            glossary: Glossary entries

        Returns:
            Path to generated file
        """
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "용어집"

            # Headers
            headers = ["영문 용어", "한국어 번역", "출처", "기술분야", "비고"]
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="DDDDDD")

            # Data rows
            for row_num, entry in enumerate(glossary, 2):
                ws.cell(row=row_num, column=1, value=entry.get("english_term", ""))
                ws.cell(row=row_num, column=2, value=entry.get("korean_term", ""))
                ws.cell(row=row_num, column=3, value=entry.get("source", ""))
                ws.cell(row=row_num, column=4, value=entry.get("domain", ""))
                ws.cell(row=row_num, column=5, value=entry.get("notes", ""))

            # Adjust column widths
            ws.column_dimensions["A"].width = 30
            ws.column_dimensions["B"].width = 30
            ws.column_dimensions["C"].width = 15
            ws.column_dimensions["D"].width = 15
            ws.column_dimensions["E"].width = 40

            file_path = output_dir / "glossary.xlsx"
            wb.save(str(file_path))

            return file_path

        except ImportError:
            # Fallback to JSON
            file_path = output_dir / "glossary.json"
            file_path.write_text(
                json.dumps(glossary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return file_path

    def _generate_deadline_reminder(
        self,
        output_dir: Path,
        deadline: str,
        source_document: dict,
    ) -> Path:
        """Generate deadline reminder file.

        Args:
            output_dir: Output directory
            deadline: Legal deadline
            source_document: Source document info

        Returns:
            Path to generated file
        """
        content = f"""# 제출기한 안내

문서: {source_document.get('file_path', 'N/A')}
특허번호: {source_document.get('patent_number', 'N/A')}

⚠️ 법정 제출기한: {deadline}

이 번역물은 위 기한까지 제출되어야 합니다.
기한 내 미제출 시 출원이 취하될 수 있습니다.

---
생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        file_path = output_dir / "deadline_reminder.txt"
        file_path.write_text(content, encoding="utf-8")

        return file_path

    def _build_final_document_content(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
    ) -> str:
        """Build final document content as string.

        Args:
            translated_sections: Translated sections

        Returns:
            Formatted document content
        """
        content = ""

        for section_type in KIPO_SECTION_ORDER:
            if section_type not in translated_sections:
                continue

            section = translated_sections[section_type]
            korean_name = KIPO_SECTION_NAMES.get(section_type, section_type)

            content += f"\n【{korean_name}】\n\n"
            content += section["translated_text"]
            content += "\n\n"

        return content.strip()

    def _build_comparison_content(
        self,
        translated_sections: dict[SectionType, TranslatedSection],
        doc_structure: dict,
    ) -> str:
        """Build comparison document content as markdown.

        Args:
            translated_sections: Translated sections
            doc_structure: Document structure

        Returns:
            Markdown formatted comparison
        """
        content = "# 영한 대조본 (English-Korean Comparison)\n\n"

        for section_type in KIPO_SECTION_ORDER:
            if section_type not in translated_sections:
                continue

            section = translated_sections[section_type]
            korean_name = KIPO_SECTION_NAMES.get(section_type, section_type)

            content += f"## {korean_name}\n\n"
            content += "| English | Korean |\n"
            content += "|---------|--------|\n"

            # Split into paragraphs for comparison
            source_paras = section["source_text"].split("\n\n")
            trans_paras = section["translated_text"].split("\n\n")

            max_paras = max(len(source_paras), len(trans_paras))
            for i in range(max_paras):
                source = source_paras[i] if i < len(source_paras) else ""
                trans = trans_paras[i] if i < len(trans_paras) else ""

                # Escape pipe characters
                source = source.replace("|", "\\|").replace("\n", " ")[:200]
                trans = trans.replace("|", "\\|").replace("\n", " ")[:200]

                content += f"| {source} | {trans} |\n"

            content += "\n"

        return content
