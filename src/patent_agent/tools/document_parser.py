"""Document parser for patent documents (PDF, HWP, DOCX).

Provides structured extraction of patent document content including:
- Claims (독립항/종속항)
- Abstract (요약)
- Description (발명의 설명)
- Drawings (도면)

Supports Korean patent document formats (KIPO submissions).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO

import structlog

logger = structlog.get_logger(__name__)


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    HWP = "hwp"
    DOCX = "docx"
    TXT = "txt"


class SectionType(str, Enum):
    """Patent document section types."""

    TITLE = "title"
    ABSTRACT = "abstract"
    TECHNICAL_FIELD = "technical_field"
    BACKGROUND = "background"
    PROBLEM_TO_SOLVE = "problem_to_solve"
    SOLUTION = "solution"
    EFFECTS = "effects"
    DETAILED_DESCRIPTION = "detailed_description"
    CLAIMS = "claims"
    DRAWINGS_DESCRIPTION = "drawings_description"
    DRAWINGS = "drawings"
    SEQUENCE_LISTING = "sequence_listing"
    UNKNOWN = "unknown"


@dataclass
class DocumentSection:
    """A section of a patent document."""

    section_type: SectionType
    title: str
    content: str
    page_start: int | None = None
    page_end: int | None = None
    subsections: list[DocumentSection] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_type": self.section_type.value,
            "title": self.title,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "subsections": [s.to_dict() for s in self.subsections],
            "metadata": self.metadata,
        }


@dataclass
class Claim:
    """A patent claim."""

    number: int
    text: str
    is_independent: bool
    depends_on: list[int] = field(default_factory=list)
    claim_type: str = "method"  # method, apparatus, composition, etc.

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "text": self.text,
            "is_independent": self.is_independent,
            "depends_on": self.depends_on,
            "claim_type": self.claim_type,
        }


@dataclass
class Drawing:
    """A patent drawing."""

    number: int
    description: str
    reference_numbers: list[str] = field(default_factory=list)
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "number": self.number,
            "description": self.description,
            "reference_numbers": self.reference_numbers,
            "image_path": self.image_path,
        }


@dataclass
class ParsedDocument:
    """Complete parsed patent document."""

    file_path: str
    document_type: DocumentType
    title: str = ""
    abstract: str = ""
    sections: list[DocumentSection] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    drawings: list[Drawing] = field(default_factory=list)
    raw_text: str = ""
    page_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "document_type": self.document_type.value,
            "title": self.title,
            "abstract": self.abstract,
            "sections": [s.to_dict() for s in self.sections],
            "claims": [c.to_dict() for c in self.claims],
            "drawings": [d.to_dict() for d in self.drawings],
            "raw_text": self.raw_text,
            "page_count": self.page_count,
            "metadata": self.metadata,
            "parse_errors": self.parse_errors,
        }

    def get_section(self, section_type: SectionType) -> DocumentSection | None:
        """Get a specific section by type."""
        for section in self.sections:
            if section.section_type == section_type:
                return section
        return None

    def get_independent_claims(self) -> list[Claim]:
        """Get all independent claims."""
        return [c for c in self.claims if c.is_independent]

    def get_dependent_claims(self, independent_claim_num: int) -> list[Claim]:
        """Get all claims dependent on a specific independent claim."""
        return [c for c in self.claims if independent_claim_num in c.depends_on]


class DocumentParser:
    """Parser for patent documents.

    Supports PDF, HWP, and DOCX formats with specialized handling
    for Korean patent document structure (KIPO format).

    Example:
        >>> parser = DocumentParser()
        >>> doc = await parser.parse("patent_application.pdf")
        >>> print(doc.title)
        >>> for claim in doc.claims:
        ...     print(f"Claim {claim.number}: {claim.text[:100]}...")
    """

    # Korean section headers (KIPO format)
    KOREAN_SECTION_PATTERNS = {
        SectionType.TITLE: [r"발명의\s*명칭", r"명\s*칭"],
        SectionType.ABSTRACT: [r"요\s*약", r"요약서"],
        SectionType.TECHNICAL_FIELD: [r"기\s*술\s*분\s*야", r"발명의\s*기술\s*분야"],
        SectionType.BACKGROUND: [r"배\s*경\s*기\s*술", r"종래\s*기술"],
        SectionType.PROBLEM_TO_SOLVE: [r"해결하고자\s*하는\s*과제", r"발명이\s*해결하고자\s*하는\s*과제"],
        SectionType.SOLUTION: [r"과제\s*해결\s*수단", r"과제의\s*해결\s*수단"],
        SectionType.EFFECTS: [r"발명의\s*효과", r"효\s*과"],
        SectionType.DETAILED_DESCRIPTION: [
            r"발명을\s*실시하기\s*위한\s*구체적인\s*내용",
            r"발명의\s*실시를\s*위한\s*구체적인\s*내용",
            r"실시예",
        ],
        SectionType.CLAIMS: [r"특허청구범위", r"청\s*구\s*범\s*위", r"청구항"],
        SectionType.DRAWINGS_DESCRIPTION: [r"도면의\s*간단한\s*설명"],
        SectionType.DRAWINGS: [r"도\s*면"],
    }

    # English section headers (USPTO format)
    ENGLISH_SECTION_PATTERNS = {
        SectionType.TITLE: [r"TITLE\s*OF\s*(?:THE\s*)?INVENTION"],
        SectionType.ABSTRACT: [r"ABSTRACT(?:\s*OF\s*THE\s*DISCLOSURE)?"],
        SectionType.TECHNICAL_FIELD: [r"TECHNICAL\s*FIELD", r"FIELD\s*OF\s*(?:THE\s*)?INVENTION"],
        SectionType.BACKGROUND: [
            r"BACKGROUND(?:\s*OF\s*THE\s*INVENTION)?",
            r"DESCRIPTION\s*OF\s*(?:THE\s*)?RELATED\s*ART",
        ],
        SectionType.DETAILED_DESCRIPTION: [
            r"DETAILED\s*DESCRIPTION",
            r"DESCRIPTION\s*OF\s*(?:THE\s*)?(?:PREFERRED\s*)?EMBODIMENTS?",
        ],
        SectionType.CLAIMS: [r"CLAIMS?", r"WHAT\s*IS\s*CLAIMED"],
        SectionType.DRAWINGS_DESCRIPTION: [r"BRIEF\s*DESCRIPTION\s*OF\s*(?:THE\s*)?DRAWINGS?"],
    }

    def __init__(self) -> None:
        """Initialize document parser."""
        self._pdf_parser: Any = None
        self._hwp_parser: Any = None
        self._docx_parser: Any = None

        # Try to import PDF parser (docling or PyMuPDF)
        try:
            import fitz  # PyMuPDF

            self._pdf_parser = "pymupdf"
            logger.info("Using PyMuPDF for PDF parsing")
        except ImportError:
            try:
                from docling.document_converter import DocumentConverter

                self._pdf_parser = "docling"
                logger.info("Using Docling for PDF parsing")
            except ImportError:
                logger.warning("No PDF parser available (install pymupdf or docling)")

        # Try to import HWP parser
        try:
            import olefile

            self._hwp_parser = "olefile"
            logger.info("Using olefile for HWP parsing")
        except ImportError:
            logger.warning("HWP parser not available (install olefile)")

        # Try to import DOCX parser
        try:
            import docx

            self._docx_parser = "python-docx"
            logger.info("Using python-docx for DOCX parsing")
        except ImportError:
            logger.warning("DOCX parser not available (install python-docx)")

    async def parse(
        self,
        file_path: str | Path,
        extract_images: bool = False,
    ) -> ParsedDocument:
        """Parse a patent document.

        Args:
            file_path: Path to the document file
            extract_images: Whether to extract drawing images

        Returns:
            ParsedDocument with structured content
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        doc_type = self._detect_document_type(file_path)
        logger.info("Parsing document", path=str(file_path), type=doc_type.value)

        loop = asyncio.get_event_loop()

        if doc_type == DocumentType.PDF:
            return await loop.run_in_executor(
                None, self._parse_pdf, file_path, extract_images
            )
        elif doc_type == DocumentType.HWP:
            return await loop.run_in_executor(
                None, self._parse_hwp, file_path, extract_images
            )
        elif doc_type == DocumentType.DOCX:
            return await loop.run_in_executor(
                None, self._parse_docx, file_path, extract_images
            )
        else:
            return await loop.run_in_executor(None, self._parse_text, file_path)

    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type from extension."""
        ext = file_path.suffix.lower()
        type_map = {
            ".pdf": DocumentType.PDF,
            ".hwp": DocumentType.HWP,
            ".docx": DocumentType.DOCX,
            ".doc": DocumentType.DOCX,
            ".txt": DocumentType.TXT,
        }
        return type_map.get(ext, DocumentType.TXT)

    def _parse_pdf(self, file_path: Path, extract_images: bool) -> ParsedDocument:
        """Parse PDF document."""
        doc = ParsedDocument(
            file_path=str(file_path),
            document_type=DocumentType.PDF,
        )

        if self._pdf_parser == "pymupdf":
            doc = self._parse_pdf_pymupdf(file_path, doc, extract_images)
        elif self._pdf_parser == "docling":
            doc = self._parse_pdf_docling(file_path, doc, extract_images)
        else:
            doc.parse_errors.append("No PDF parser available")

        # Post-process: extract structure
        if doc.raw_text:
            self._extract_structure(doc)

        return doc

    def _parse_pdf_pymupdf(
        self, file_path: Path, doc: ParsedDocument, extract_images: bool
    ) -> ParsedDocument:
        """Parse PDF using PyMuPDF."""
        import fitz

        try:
            pdf = fitz.open(str(file_path))
            doc.page_count = len(pdf)

            text_parts = []
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                text_parts.append(text)

                # Extract images if requested
                if extract_images:
                    images = page.get_images()
                    for img_idx, img in enumerate(images):
                        doc.drawings.append(
                            Drawing(
                                number=len(doc.drawings) + 1,
                                description=f"Image from page {page_num + 1}",
                            )
                        )

            doc.raw_text = "\n".join(text_parts)
            pdf.close()

        except Exception as e:
            doc.parse_errors.append(f"PyMuPDF error: {e}")
            logger.error(f"Failed to parse PDF with PyMuPDF: {e}")

        return doc

    def _parse_pdf_docling(
        self, file_path: Path, doc: ParsedDocument, extract_images: bool
    ) -> ParsedDocument:
        """Parse PDF using Docling."""
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(file_path))

            doc.raw_text = result.document.export_to_markdown()
            doc.page_count = len(result.document.pages) if hasattr(result.document, "pages") else 0

        except Exception as e:
            doc.parse_errors.append(f"Docling error: {e}")
            logger.error(f"Failed to parse PDF with Docling: {e}")

        return doc

    def _parse_hwp(self, file_path: Path, extract_images: bool) -> ParsedDocument:
        """Parse HWP (Hangul Word Processor) document."""
        doc = ParsedDocument(
            file_path=str(file_path),
            document_type=DocumentType.HWP,
        )

        if self._hwp_parser != "olefile":
            doc.parse_errors.append("HWP parser not available")
            return doc

        try:
            import olefile
            import zlib

            ole = olefile.OleFileIO(str(file_path))

            # Extract text from HWP
            if ole.exists("PrvText"):
                # Preview text (simple extraction)
                encoded_text = ole.openstream("PrvText").read()
                doc.raw_text = encoded_text.decode("utf-16", errors="ignore")
            elif ole.exists("BodyText/Section0"):
                # Try to extract from body text sections
                text_parts = []
                section_idx = 0
                while ole.exists(f"BodyText/Section{section_idx}"):
                    stream = ole.openstream(f"BodyText/Section{section_idx}")
                    data = stream.read()
                    try:
                        decompressed = zlib.decompress(data, -15)
                        # HWP body text is complex - this is simplified
                        text_parts.append(decompressed.decode("utf-16", errors="ignore"))
                    except Exception:
                        pass
                    section_idx += 1

                doc.raw_text = "\n".join(text_parts)

            ole.close()

        except Exception as e:
            doc.parse_errors.append(f"HWP parsing error: {e}")
            logger.error(f"Failed to parse HWP: {e}")

        if doc.raw_text:
            self._extract_structure(doc)

        return doc

    def _parse_docx(self, file_path: Path, extract_images: bool) -> ParsedDocument:
        """Parse DOCX document."""
        doc = ParsedDocument(
            file_path=str(file_path),
            document_type=DocumentType.DOCX,
        )

        if self._docx_parser != "python-docx":
            doc.parse_errors.append("DOCX parser not available")
            return doc

        try:
            import docx

            document = docx.Document(str(file_path))

            text_parts = []
            for para in document.paragraphs:
                text_parts.append(para.text)

            doc.raw_text = "\n".join(text_parts)

            # Extract tables (often contain claims or structured data)
            for table in document.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text for cell in row.cells)
                    text_parts.append(row_text)

        except Exception as e:
            doc.parse_errors.append(f"DOCX parsing error: {e}")
            logger.error(f"Failed to parse DOCX: {e}")

        if doc.raw_text:
            self._extract_structure(doc)

        return doc

    def _parse_text(self, file_path: Path) -> ParsedDocument:
        """Parse plain text file."""
        doc = ParsedDocument(
            file_path=str(file_path),
            document_type=DocumentType.TXT,
        )

        try:
            doc.raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                doc.raw_text = file_path.read_text(encoding="cp949")
            except Exception as e:
                doc.parse_errors.append(f"Text encoding error: {e}")

        if doc.raw_text:
            self._extract_structure(doc)

        return doc

    def _extract_structure(self, doc: ParsedDocument) -> None:
        """Extract structured content from raw text."""
        # Detect language
        is_korean = self._detect_korean(doc.raw_text)
        patterns = self.KOREAN_SECTION_PATTERNS if is_korean else self.ENGLISH_SECTION_PATTERNS

        # Find sections
        doc.sections = self._find_sections(doc.raw_text, patterns)

        # Extract title
        title_section = doc.get_section(SectionType.TITLE)
        if title_section:
            doc.title = title_section.content.strip()

        # Extract abstract
        abstract_section = doc.get_section(SectionType.ABSTRACT)
        if abstract_section:
            doc.abstract = abstract_section.content.strip()

        # Extract claims
        claims_section = doc.get_section(SectionType.CLAIMS)
        if claims_section:
            doc.claims = self._extract_claims(claims_section.content, is_korean)

        # Extract drawings
        drawings_section = doc.get_section(SectionType.DRAWINGS_DESCRIPTION)
        if drawings_section:
            doc.drawings = self._extract_drawings(drawings_section.content, is_korean)

    def _detect_korean(self, text: str) -> bool:
        """Detect if text is primarily Korean."""
        korean_chars = len(re.findall(r"[\uac00-\ud7af]", text))
        total_chars = len(text.replace(" ", "").replace("\n", ""))
        if total_chars == 0:
            return False
        return korean_chars / total_chars > 0.3

    def _find_sections(
        self, text: str, patterns: dict[SectionType, list[str]]
    ) -> list[DocumentSection]:
        """Find and extract sections from text."""
        sections: list[DocumentSection] = []

        # Build combined pattern for finding section boundaries
        all_patterns = []
        pattern_to_type: dict[str, SectionType] = {}

        for section_type, type_patterns in patterns.items():
            for p in type_patterns:
                all_patterns.append(p)
                pattern_to_type[p] = section_type

        if not all_patterns:
            return sections

        # Find all section headers
        combined_pattern = "|".join(f"({p})" for p in all_patterns)
        matches = list(re.finditer(combined_pattern, text, re.IGNORECASE | re.MULTILINE))

        if not matches:
            # No sections found, treat entire text as unknown section
            sections.append(
                DocumentSection(
                    section_type=SectionType.UNKNOWN,
                    title="Document Content",
                    content=text,
                )
            )
            return sections

        # Extract content between sections
        for i, match in enumerate(matches):
            # Find which pattern matched
            section_type = SectionType.UNKNOWN
            for p, st in pattern_to_type.items():
                if re.search(p, match.group(), re.IGNORECASE):
                    section_type = st
                    break

            # Get content until next section or end
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_pos:end_pos].strip()

            sections.append(
                DocumentSection(
                    section_type=section_type,
                    title=match.group().strip(),
                    content=content,
                )
            )

        return sections

    def _extract_claims(self, claims_text: str, is_korean: bool) -> list[Claim]:
        """Extract individual claims from claims section."""
        claims: list[Claim] = []

        if is_korean:
            # Korean claim pattern: 【청구항 1】 or 청구항 1.
            pattern = r"(?:【청구항\s*(\d+)】|청구항\s*(\d+)\.?)"
        else:
            # English claim pattern: 1. or Claim 1:
            pattern = r"(?:(\d+)\.|Claim\s*(\d+):?)"

        matches = list(re.finditer(pattern, claims_text, re.IGNORECASE | re.MULTILINE))

        for i, match in enumerate(matches):
            # Get claim number
            claim_num = int(match.group(1) or match.group(2))

            # Get claim text
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(claims_text)
            claim_text = claims_text[start_pos:end_pos].strip()

            # Detect dependency
            depends_on: list[int] = []
            is_independent = True

            if is_korean:
                dep_match = re.search(r"제?\s*(\d+)\s*항에\s*있어서", claim_text)
            else:
                dep_match = re.search(r"(?:claim|claims?)\s*(\d+)", claim_text, re.IGNORECASE)

            if dep_match:
                depends_on = [int(dep_match.group(1))]
                is_independent = False

            # Detect claim type
            claim_type = self._detect_claim_type(claim_text, is_korean)

            claims.append(
                Claim(
                    number=claim_num,
                    text=claim_text,
                    is_independent=is_independent,
                    depends_on=depends_on,
                    claim_type=claim_type,
                )
            )

        return claims

    def _detect_claim_type(self, claim_text: str, is_korean: bool) -> str:
        """Detect the type of claim (method, apparatus, etc.)."""
        if is_korean:
            if re.search(r"방법|단계|~하는\s*단계", claim_text):
                return "method"
            elif re.search(r"장치|시스템|포함하는\s*장치", claim_text):
                return "apparatus"
            elif re.search(r"조성물|화합물", claim_text):
                return "composition"
            elif re.search(r"매체|기록매체|저장매체", claim_text):
                return "medium"
        else:
            if re.search(r"method|process|step", claim_text, re.IGNORECASE):
                return "method"
            elif re.search(r"apparatus|device|system", claim_text, re.IGNORECASE):
                return "apparatus"
            elif re.search(r"composition|compound", claim_text, re.IGNORECASE):
                return "composition"
            elif re.search(r"medium|readable", claim_text, re.IGNORECASE):
                return "medium"

        return "unknown"

    def _extract_drawings(self, drawings_text: str, is_korean: bool) -> list[Drawing]:
        """Extract drawing descriptions."""
        drawings: list[Drawing] = []

        if is_korean:
            # Korean pattern: 도 1은... or 【도 1】
            pattern = r"(?:【?\s*도\s*(\d+)】?|도\s*(\d+)(?:은|는))"
        else:
            # English pattern: FIG. 1 or Figure 1
            pattern = r"(?:FIG\.?\s*(\d+)|Figure\s*(\d+))"

        matches = list(re.finditer(pattern, drawings_text, re.IGNORECASE))

        for i, match in enumerate(matches):
            fig_num = int(match.group(1) or match.group(2))

            # Get description
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(drawings_text)
            description = drawings_text[start_pos:end_pos].strip()

            # Extract reference numbers (e.g., (10), (20))
            ref_nums = re.findall(r"\((\d+)\)", description)

            drawings.append(
                Drawing(
                    number=fig_num,
                    description=description,
                    reference_numbers=ref_nums,
                )
            )

        return drawings

    async def parse_from_bytes(
        self,
        data: bytes | BinaryIO,
        document_type: DocumentType,
        filename: str = "document",
    ) -> ParsedDocument:
        """Parse document from bytes or file-like object.

        Args:
            data: Document data as bytes or file-like object
            document_type: Type of document
            filename: Optional filename for reference

        Returns:
            ParsedDocument with structured content
        """
        import tempfile

        # Write to temp file and parse
        suffix = f".{document_type.value}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            if isinstance(data, bytes):
                tmp.write(data)
            else:
                tmp.write(data.read())
            tmp_path = tmp.name

        try:
            doc = await self.parse(tmp_path)
            doc.file_path = filename
            return doc
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# LangChain Tool wrappers
def create_document_parser_tools() -> list[Any]:
    """Create LangChain tools for document parsing.

    Returns a list of tools that can be used by LangChain agents.
    """
    from langchain_core.tools import StructuredTool

    parser = DocumentParser()

    async def parse_patent_document(file_path: str) -> str:
        """Parse a patent document and extract structured content.

        Args:
            file_path: Path to the patent document (PDF, HWP, or DOCX)

        Returns:
            JSON string with parsed document content
        """
        import json

        doc = await parser.parse(file_path)
        return json.dumps(doc.to_dict(), ensure_ascii=False, indent=2)

    async def extract_claims_from_document(file_path: str) -> str:
        """Extract claims from a patent document.

        Args:
            file_path: Path to the patent document

        Returns:
            JSON string with extracted claims
        """
        import json

        doc = await parser.parse(file_path)
        return json.dumps([c.to_dict() for c in doc.claims], ensure_ascii=False, indent=2)

    return [
        StructuredTool.from_function(
            coroutine=parse_patent_document,
            name="parse_patent_document",
            description="Parse a patent document (PDF, HWP, DOCX) and extract structured content including title, abstract, claims, and drawings",
        ),
        StructuredTool.from_function(
            coroutine=extract_claims_from_document,
            name="extract_claims_from_document",
            description="Extract patent claims from a document, identifying independent and dependent claims",
        ),
    ]
