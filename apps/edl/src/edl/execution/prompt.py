from __future__ import annotations

from edl.api.schemas import DispatchRequest
from dullahan_shared.schemas.expert import ExpertProfile


class ExpertPromptBuilder:
    def build(self, request: DispatchRequest, expert: ExpertProfile) -> str:
        context_lines = [
            f"[{document.id}] {document.text.strip()}"
            for document in request.context.documents[:5]
        ]
        context = "\n".join(context_lines) if context_lines else "No supporting context was provided."
        return "\n".join(
            [
                "Role:",
                expert.role_context.strip(),
                "",
                "Subquery:",
                request.subquery,
                "",
                "Context:",
                context,
                "",
                "Answer with only the information needed to satisfy the subquery.",
            ]
        )
