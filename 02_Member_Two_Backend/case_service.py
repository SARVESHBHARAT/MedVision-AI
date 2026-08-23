"""Small case service used while persistence is being integrated."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

@dataclass
class Case:
    owner_id: UUID
    reference_code: str
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class CaseService:
    def __init__(self):
        self._cases: dict[UUID, Case] = {}

    def create(self, owner_id: UUID, reference_code: str) -> Case:
        if any(case.reference_code == reference_code for case in self._cases.values()):
            raise ValueError("Case reference already exists")
        case = Case(owner_id=owner_id, reference_code=reference_code)
        self._cases[case.id] = case
        return case

    def list_for_owner(self, owner_id: UUID, limit: int = 20, offset: int = 0) -> list[Case]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid pagination values")
        records = [case for case in self._cases.values() if case.owner_id == owner_id]
        return sorted(records, key=lambda item: item.created_at, reverse=True)[offset:offset + limit]
