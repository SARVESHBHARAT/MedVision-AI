# Implementation security checklist

- [x] Case owner row-level policy defined.
- [x] Image formats and maximum upload size constrained at database level.
- [x] Case and analysis lifecycle events recorded through audit tables.
- [x] Analysis completion cannot be persisted without a result and completion timestamp.
- [ ] Encrypt object storage and configure retention policy.
- [ ] Create application database role with least-privilege grants.
- [ ] Run a schema migration review in a disposable development database.
