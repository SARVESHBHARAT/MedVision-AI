from uuid import uuid4
from case_service import CaseService

def test_case_list_is_scoped_to_owner():
    service, owner, other = CaseService(), uuid4(), uuid4()
    service.create(owner, 'MV-2001')
    service.create(other, 'MV-2002')
    assert [case.reference_code for case in service.list_for_owner(owner)] == ['MV-2001']

def test_case_reference_must_be_unique():
    service, owner = CaseService(), uuid4()
    service.create(owner, 'MV-2001')
    try:
        service.create(owner, 'MV-2001')
    except ValueError as error:
        assert 'already exists' in str(error)
    else:
        assert False, 'duplicate case reference was allowed'
