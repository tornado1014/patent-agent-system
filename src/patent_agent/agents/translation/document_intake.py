"""Document Intake Agent (P1) for Patent Translation workflow.

Handles initial document processing:
- PDF/DOCX file parsing
- OCR requirement detection
- Metadata extraction
- Legal deadline identification
"""

from pathlib import Path
from typing import Any

from patent_agent.agents.translation.base import BaseTranslationAgent
from patent_agent.state.translation import (
    SourceDocument,
    TranslationPhase,
    TranslationState,
)


class DocumentIntakeAgent(BaseTranslationAgent):
    """Document intake and initial processing agent.

    Phase P1: 문서 접수 및 초기 추론

    Responsibilities:
    - Parse source document (PDF/DOCX)
    - Detect if OCR is needed
    - Extract metadata (patent number, filing date, etc.)
    - Identify legal deadlines
    - Initial file validation
    """

    @property
    def name(self) -> str:
        return "DocumentIntakeAgent"

    @property
    def description(self) -> str:
        return "문서 접수 및 초기 추론을 담당하는 에이전트"

    @property
    def phase(self) -> TranslationPhase:
        return "P1"

    def get_system_prompt(self) -> str:
        return """## Document Intake Agent (P1)

당신은 영문 특허 문서를 접수하고 초기 분석을 수행하는 에이전트입니다.

### 주요 임무
1. 문서 파일 형식 확인 (PDF, DOCX, TXT)
2. OCR 필요 여부 판단 (스캔 문서 여부)
3. 특허 메타데이터 추출
   - 특허번호/출원번호
   - 출원일/공개일
   - 출원인/발명자
   - IPC/CPC 분류
4. 법정 제출기한 확인

### 출력 요구사항
- 파일 정보 및 페이지 수
- OCR 필요 여부와 판단 근거
- 추출된 메타데이터
- 잠재적 문제점 (파일 손상, 해상도 등)
"""

    async def process(self, state: TranslationState) -> dict[str, Any]:
        """Process document intake.

        Args:
            state: Current workflow state

        Returns:
            State updates with source document info and metadata
        """
        self._logger.info("starting_document_intake")

        # Get source file path from state
        source_path = state.get("source_file_path", "")

        if not source_path:
            self._logger.error("no_source_file_provided")
            return {
                "is_error_state": True,
                "error_messages": ["소스 파일 경로가 제공되지 않았습니다."],
                "current_step": "P1",
            }

        # Analyze the document
        source_doc = await self._analyze_document(source_path)

        if not source_doc:
            return {
                "is_error_state": True,
                "error_messages": [f"문서 분석 실패: {source_path}"],
                "current_step": "P1",
            }

        # Extract metadata using LLM if we have text content
        metadata = {}
        text_content = state.get("source_text_content", "")

        if text_content:
            metadata = await self._extract_metadata(text_content)

        # Check for legal deadline
        legal_deadline = await self._extract_legal_deadline(text_content, metadata)

        # Create reasoning entry
        reasoning = self.create_reasoning_entry(
            checkpoint="CP1.1",
            decision="문서 접수 완료",
            reasoning=f"파일 형식: {source_doc['file_type']}, "
            f"페이지 수: {source_doc['total_pages']}, "
            f"OCR 필요: {source_doc['requires_ocr']}",
            applied_laws=[],
        )

        self._logger.info(
            "document_intake_complete",
            file_type=source_doc["file_type"],
            pages=source_doc["total_pages"],
            requires_ocr=source_doc["requires_ocr"],
        )

        return {
            "source_document": source_doc,
            "ocr_performed": False,
            "metadata_extracted": metadata,
            "legal_deadline": legal_deadline,
            "current_step": "P2",
            "reasoning_log": [reasoning],
        }

    async def _analyze_document(self, file_path: str) -> SourceDocument | None:
        """Analyze the source document.

        Args:
            file_path: Path to the source document

        Returns:
            SourceDocument info or None if analysis fails
        """
        path = Path(file_path)

        if not path.exists():
            self._logger.error("file_not_found", path=file_path)
            return None

        # Determine file type
        suffix = path.suffix.lower()
        file_type_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".doc": "docx",
            ".txt": "txt",
        }

        file_type = file_type_map.get(suffix)
        if not file_type:
            self._logger.error("unsupported_file_type", suffix=suffix)
            return None

        # Get basic info
        total_pages = await self._get_page_count(path, file_type)
        requires_ocr = await self._check_ocr_needed(path, file_type)

        return {
            "file_path": str(path.absolute()),
            "file_type": file_type,
            "total_pages": total_pages,
            "requires_ocr": requires_ocr,
            "patent_number": None,
            "filing_date": None,
        }

    async def _get_page_count(self, path: Path, file_type: str) -> int:
        """Get the page count of a document.

        Args:
            path: Path to the document
            file_type: Type of document

        Returns:
            Number of pages
        """
        try:
            if file_type == "pdf":
                # Try to get PDF page count
                try:
                    import pymupdf

                    with pymupdf.open(str(path)) as doc:
                        return len(doc)
                except ImportError:
                    # Fallback: estimate based on file size
                    size_kb = path.stat().st_size / 1024
                    return max(1, int(size_kb / 50))  # Rough estimate

            elif file_type == "docx":
                try:
                    from docx import Document

                    doc = Document(str(path))
                    # Estimate pages from paragraphs (rough)
                    return max(1, len(doc.paragraphs) // 30)
                except ImportError:
                    return 1

            else:
                return 1

        except Exception as e:
            self._logger.warning("page_count_error", error=str(e))
            return 1

    async def _check_ocr_needed(self, path: Path, file_type: str) -> bool:
        """Check if OCR is needed for the document.

        Args:
            path: Path to the document
            file_type: Type of document

        Returns:
            True if OCR is needed
        """
        if file_type != "pdf":
            return False

        try:
            import pymupdf

            with pymupdf.open(str(path)) as doc:
                if len(doc) == 0:
                    return True

                # Check first few pages for text
                text_found = False
                for page_num in range(min(3, len(doc))):
                    page = doc[page_num]
                    text = page.get_text()
                    if len(text.strip()) > 100:
                        text_found = True
                        break

                return not text_found

        except ImportError:
            # Can't check without pymupdf
            return False
        except Exception as e:
            self._logger.warning("ocr_check_error", error=str(e))
            return False

    async def _extract_metadata(self, text_content: str) -> dict[str, str]:
        """Extract metadata from document text using LLM.

        Args:
            text_content: Text content of the document

        Returns:
            Dictionary of extracted metadata
        """
        if not text_content or len(text_content) < 100:
            return {}

        # Take first portion of text for metadata extraction
        excerpt = text_content[:5000]

        prompt = f"""다음 영문 특허 문서에서 메타데이터를 추출하세요.

문서 내용:
{excerpt}

다음 정보를 JSON 형식으로 추출하세요 (없으면 null):
- patent_number: 특허번호 또는 출원번호
- filing_date: 출원일 (YYYY-MM-DD 형식)
- publication_date: 공개일 (YYYY-MM-DD 형식)
- applicant: 출원인
- inventor: 발명자
- title: 발명의 명칭
- ipc_codes: IPC 분류 코드 목록

JSON만 출력하세요:"""

        try:
            response = await self._invoke_llm(prompt)

            # Parse JSON from response
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                metadata = json.loads(json_match.group())
                return {k: v for k, v in metadata.items() if v is not None}

        except Exception as e:
            self._logger.warning("metadata_extraction_error", error=str(e))

        return {}

    async def _extract_legal_deadline(
        self, text_content: str, metadata: dict[str, str]
    ) -> str | None:
        """Extract or calculate legal deadline.

        Args:
            text_content: Document text content
            metadata: Extracted metadata

        Returns:
            Legal deadline string or None
        """
        # If there's a filing date, we might calculate translation deadline
        filing_date = metadata.get("filing_date")

        if filing_date:
            # This is a simplified example - actual deadline calculation
            # would depend on the specific patent jurisdiction and requirements
            try:
                from datetime import datetime, timedelta

                filed = datetime.strptime(filing_date, "%Y-%m-%d")
                # Example: 30 months from filing for PCT national phase
                deadline = filed + timedelta(days=30 * 30)
                return deadline.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        return None
