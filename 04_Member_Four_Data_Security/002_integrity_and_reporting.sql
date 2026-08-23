-- Follow-up migration: tighter lifecycle rules and review dashboard query.
ALTER TABLE analysis_jobs
  ADD CONSTRAINT completed_job_has_result CHECK (status <> 'completed' OR result IS NOT NULL),
  ADD CONSTRAINT completed_job_has_time CHECK (status <> 'completed' OR completed_at IS NOT NULL);

CREATE INDEX cases_owner_created_idx ON cases (owner_id, created_at DESC);
CREATE INDEX audit_events_entity_idx ON audit_events (entity_type, entity_id, occurred_at DESC);

CREATE VIEW case_analysis_summary AS
SELECT c.id AS case_id, c.reference_code, c.owner_id, j.status, j.model_version, j.created_at AS analysis_created_at
FROM cases c
LEFT JOIN medical_images i ON i.case_id = c.id
LEFT JOIN analysis_jobs j ON j.image_id = i.id;

-- Grant only after creating the application role in the target environment.
-- GRANT SELECT, INSERT, UPDATE ON cases, medical_images, analysis_jobs TO medvision_app;
