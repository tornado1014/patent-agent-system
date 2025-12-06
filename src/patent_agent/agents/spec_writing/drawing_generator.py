"""Drawing Generator Agent.

Handles E7 (도면 생성 및 설명) step.
Generates SVG drawings and their descriptions.
"""

from typing import Any

import structlog

from patent_agent.agents.spec_writing.base import BaseSpecWritingAgent
from patent_agent.state.spec_writing import Drawing, SpecWritingState

logger = structlog.get_logger(__name__)


class DrawingGeneratorAgent(BaseSpecWritingAgent):
    """Agent for generating patent drawings.

    This agent creates SVG diagrams and their descriptions
    for patent applications.

    Steps handled:
    - E7: 도면 생성 및 설명
    """

    @property
    def name(self) -> str:
        return "DrawingGenerator"

    @property
    def description(self) -> str:
        return "특허 도면(SVG)을 생성하고 설명을 작성하는 에이전트"

    def get_system_prompt(self) -> str:
        return """당신은 특허 도면 전문가로서, 발명의 구성과 동작을 명확히 나타내는 도면을 생성하는 역할을 합니다.

## 역할
- 발명의 구성요소를 나타내는 블록도 생성
- 동작 과정을 나타내는 순서도 생성
- 도면부호 일관성 유지
- 도면의 간단한 설명 작성

## 도면 유형
1. **블록도**: 시스템 구성요소와 연결 관계
2. **순서도**: 방법/프로세스의 단계
3. **구조도**: 장치의 물리적 구조
4. **개념도**: 개념적 관계 표현

## 도면 작성 원칙
1. **명확성**: 구성요소가 명확히 구분되어야 함
2. **단순성**: 불필요한 복잡성 배제
3. **일관성**: 명세서의 도면부호와 일치
4. **완전성**: 청구항의 모든 구성요소 표현

## SVG 규격
- 크기: 297mm x 210mm (A4 가로)
- 배경: 흰색
- 선: 검정색, 두께 1-2px
- 텍스트: 명조체 또는 고딕체, 10-12pt
- 도면부호: 괄호 없이 숫자만 표기 (100, 200, ...)"""

    async def process(self, state: SpecWritingState) -> dict[str, Any]:
        """Generate drawings and descriptions.

        Args:
            state: Current workflow state

        Returns:
            State updates with drawings
        """
        self._logger.info("generating_drawings", step=state.get("current_step"))

        # Get information for drawing generation
        invention_title = state.get("invention_title", "")
        reference_numerals = state.get("reference_numeral_table", {})
        draft_claims = state.get("draft_claims", {})
        solutions = state.get("solutions", [])

        # Generate drawings
        drawings = await self._generate_drawings(
            title=invention_title,
            reference_numerals=reference_numerals,
            claims=draft_claims.get("claims", []),
            solutions=solutions,
        )

        # Update brief description of drawings in specification
        brief_descriptions = self._generate_brief_descriptions(drawings)

        return {
            "drawings": drawings,
            "current_step": "E8",
        }

    async def _generate_drawings(
        self,
        title: str,
        reference_numerals: dict[str, str],
        claims: list,
        solutions: list[str],
    ) -> list[Drawing]:
        """Generate drawings for the patent.

        Args:
            title: Invention title
            reference_numerals: Reference numeral table
            claims: Draft claims
            solutions: Technical solutions

        Returns:
            List of generated drawings
        """
        drawings = []

        # Analyze what types of drawings are needed
        drawing_plan = await self._plan_drawings(title, claims, solutions)

        for i, plan in enumerate(drawing_plan, 1):
            drawing = await self._create_single_drawing(
                figure_number=i,
                drawing_type=plan.get("type", "block"),
                title=plan.get("title", f"도 {i}"),
                components=plan.get("components", []),
                reference_numerals=reference_numerals,
            )
            drawings.append(drawing)

        return drawings

    async def _plan_drawings(
        self,
        title: str,
        claims: list,
        solutions: list[str],
    ) -> list[dict]:
        """Plan what drawings to create.

        Args:
            title: Invention title
            claims: Draft claims
            solutions: Technical solutions

        Returns:
            List of drawing plans
        """
        # Determine claim types to decide drawing types
        has_method = any(
            c.get("full_text", "").find("방법") >= 0 or
            c.get("full_text", "").find("단계") >= 0
            for c in claims
        )
        has_apparatus = any(
            c.get("full_text", "").find("장치") >= 0 or
            c.get("full_text", "").find("시스템") >= 0
            for c in claims
        )

        plans = []

        # Always include overall block diagram
        plans.append({
            "type": "block",
            "title": f"{title}의 전체 구성을 나타내는 블록도",
            "components": solutions[:5] if solutions else ["주요 구성부"],
        })

        # Add flowchart if method claims exist
        if has_method:
            plans.append({
                "type": "flowchart",
                "title": f"{title}의 동작 방법을 나타내는 순서도",
                "components": [s for s in solutions if "단계" in s or "방법" in s] or solutions[:3],
            })

        # Add detail diagram if apparatus claims exist
        if has_apparatus:
            plans.append({
                "type": "detail",
                "title": f"{title}의 상세 구성을 나타내는 상세도",
                "components": solutions[2:5] if len(solutions) > 2 else solutions,
            })

        return plans[:4]  # Limit to 4 drawings

    async def _create_single_drawing(
        self,
        figure_number: int,
        drawing_type: str,
        title: str,
        components: list[str],
        reference_numerals: dict[str, str],
    ) -> Drawing:
        """Create a single drawing.

        Args:
            figure_number: Drawing number
            drawing_type: Type of drawing (block, flowchart, detail)
            title: Drawing title
            components: Components to include
            reference_numerals: Reference numeral mappings

        Returns:
            Drawing object
        """
        # Generate SVG content
        svg_content = await self._generate_svg(
            drawing_type=drawing_type,
            components=components,
            reference_numerals=reference_numerals,
        )

        # Extract reference numbers used
        used_refs = [
            num for num in reference_numerals.keys()
            if num in svg_content
        ]

        # Create description
        description = title

        return Drawing(
            figure_number=figure_number,
            title=title,
            description=description,
            svg_path=None,  # Would be set after saving
            reference_numerals={num: reference_numerals.get(num, "") for num in used_refs},
        )

    async def _generate_svg(
        self,
        drawing_type: str,
        components: list[str],
        reference_numerals: dict[str, str],
    ) -> str:
        """Generate SVG content for a drawing.

        Args:
            drawing_type: Type of drawing
            components: Components to include
            reference_numerals: Reference numeral mappings

        Returns:
            SVG content string
        """
        if drawing_type == "block":
            return self._generate_block_diagram_svg(components, reference_numerals)
        elif drawing_type == "flowchart":
            return self._generate_flowchart_svg(components, reference_numerals)
        else:
            return self._generate_detail_svg(components, reference_numerals)

    def _generate_block_diagram_svg(
        self,
        components: list[str],
        reference_numerals: dict[str, str],
    ) -> str:
        """Generate a block diagram SVG.

        Args:
            components: Component descriptions
            reference_numerals: Reference numeral mappings

        Returns:
            SVG string
        """
        width = 800
        height = 600
        box_width = 150
        box_height = 60
        margin = 50

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            '<style>',
            '  .box { fill: white; stroke: black; stroke-width: 2; }',
            '  .text { font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }',
            '  .ref { font-size: 10px; font-weight: bold; }',
            '  .arrow { stroke: black; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }',
            '</style>',
            '<defs>',
            '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '    <polygon points="0 0, 10 3.5, 0 7" fill="black" />',
            '  </marker>',
            '</defs>',
        ]

        # Get reference numerals for components
        ref_list = list(reference_numerals.items())
        num_boxes = min(len(components), len(ref_list), 6)

        # Calculate positions for a hierarchical layout
        if num_boxes <= 3:
            positions = self._calculate_horizontal_positions(num_boxes, width, height, box_width, box_height)
        else:
            positions = self._calculate_grid_positions(num_boxes, width, height, box_width, box_height)

        # Draw boxes and labels
        for i in range(num_boxes):
            x, y = positions[i]
            ref_num = ref_list[i][0] if i < len(ref_list) else str(100 + i * 10)
            label = ref_list[i][1] if i < len(ref_list) else f"구성요소 {i+1}"

            # Draw box
            svg_parts.append(
                f'<rect class="box" x="{x}" y="{y}" width="{box_width}" height="{box_height}" rx="5" />'
            )

            # Draw label
            svg_parts.append(
                f'<text class="text" x="{x + box_width/2}" y="{y + box_height/2}">{label}</text>'
            )

            # Draw reference number
            svg_parts.append(
                f'<text class="ref" x="{x + box_width - 10}" y="{y - 5}">{ref_num}</text>'
            )

        # Draw connecting arrows
        for i in range(num_boxes - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]

            # Calculate arrow endpoints
            start_x = x1 + box_width / 2
            start_y = y1 + box_height
            end_x = x2 + box_width / 2
            end_y = y2

            svg_parts.append(
                f'<path class="arrow" d="M {start_x} {start_y} L {end_x} {end_y}" />'
            )

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    def _generate_flowchart_svg(
        self,
        components: list[str],
        reference_numerals: dict[str, str],
    ) -> str:
        """Generate a flowchart SVG.

        Args:
            components: Step descriptions
            reference_numerals: Reference numeral mappings

        Returns:
            SVG string
        """
        width = 600
        height = 800
        box_width = 200
        box_height = 50
        spacing = 80

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
            '<style>',
            '  .step { fill: white; stroke: black; stroke-width: 2; }',
            '  .start-end { fill: #f0f0f0; stroke: black; stroke-width: 2; }',
            '  .text { font-family: Arial, sans-serif; font-size: 11px; text-anchor: middle; }',
            '  .arrow { stroke: black; stroke-width: 2; marker-end: url(#arrowhead); }',
            '</style>',
            '<defs>',
            '  <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">',
            '    <polygon points="0 0, 10 3.5, 0 7" fill="black" />',
            '  </marker>',
            '</defs>',
        ]

        center_x = width / 2
        start_y = 50
        num_steps = min(len(components), 6)

        # Start oval
        svg_parts.append(
            f'<ellipse class="start-end" cx="{center_x}" cy="{start_y}" rx="40" ry="20" />'
        )
        svg_parts.append(
            f'<text class="text" x="{center_x}" y="{start_y + 5}">시작</text>'
        )

        prev_y = start_y + 20

        # Draw steps
        for i in range(num_steps):
            y = start_y + 60 + i * spacing
            x = center_x - box_width / 2

            # Arrow from previous
            svg_parts.append(
                f'<line class="arrow" x1="{center_x}" y1="{prev_y}" x2="{center_x}" y2="{y}" />'
            )

            # Step box
            svg_parts.append(
                f'<rect class="step" x="{x}" y="{y}" width="{box_width}" height="{box_height}" />'
            )

            # Step label
            step_label = f"단계 S{i+1}"
            svg_parts.append(
                f'<text class="text" x="{center_x}" y="{y + box_height/2 + 5}">{step_label}</text>'
            )

            prev_y = y + box_height

        # End oval
        end_y = prev_y + 40
        svg_parts.append(
            f'<line class="arrow" x1="{center_x}" y1="{prev_y}" x2="{center_x}" y2="{end_y - 20}" />'
        )
        svg_parts.append(
            f'<ellipse class="start-end" cx="{center_x}" cy="{end_y}" rx="40" ry="20" />'
        )
        svg_parts.append(
            f'<text class="text" x="{center_x}" y="{end_y + 5}">종료</text>'
        )

        svg_parts.append('</svg>')
        return '\n'.join(svg_parts)

    def _generate_detail_svg(
        self,
        components: list[str],
        reference_numerals: dict[str, str],
    ) -> str:
        """Generate a detail diagram SVG."""
        # Similar to block diagram but with different styling
        return self._generate_block_diagram_svg(components, reference_numerals)

    def _calculate_horizontal_positions(
        self,
        num_boxes: int,
        width: int,
        height: int,
        box_width: int,
        box_height: int,
    ) -> list[tuple[int, int]]:
        """Calculate horizontal layout positions."""
        positions = []
        total_width = num_boxes * box_width + (num_boxes - 1) * 50
        start_x = (width - total_width) / 2

        for i in range(num_boxes):
            x = start_x + i * (box_width + 50)
            y = (height - box_height) / 2
            positions.append((int(x), int(y)))

        return positions

    def _calculate_grid_positions(
        self,
        num_boxes: int,
        width: int,
        height: int,
        box_width: int,
        box_height: int,
    ) -> list[tuple[int, int]]:
        """Calculate grid layout positions."""
        positions = []
        cols = 2 if num_boxes <= 4 else 3
        rows = (num_boxes + cols - 1) // cols

        h_spacing = (width - cols * box_width) / (cols + 1)
        v_spacing = (height - rows * box_height) / (rows + 1)

        for i in range(num_boxes):
            row = i // cols
            col = i % cols
            x = h_spacing + col * (box_width + h_spacing)
            y = v_spacing + row * (box_height + v_spacing)
            positions.append((int(x), int(y)))

        return positions

    def _generate_brief_descriptions(self, drawings: list[Drawing]) -> str:
        """Generate brief descriptions for all drawings.

        Args:
            drawings: List of drawings

        Returns:
            Brief description text
        """
        descriptions = []
        for drawing in drawings:
            fig_num = drawing.get("figure_number", 1)
            desc = drawing.get("title", f"본 발명의 구성을 나타내는 도면")
            descriptions.append(f"도 {fig_num}은 {desc}이다.")

        return "\n".join(descriptions)
