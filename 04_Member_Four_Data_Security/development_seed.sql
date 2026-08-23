-- Local-only sample users and cases for testing the schema.
INSERT INTO app_users (id, email, display_name) VALUES
('11111111-1111-1111-1111-111111111111', 'clinician.one@medvision.local', 'Clinician One'),
('22222222-2222-2222-2222-222222222222', 'clinician.two@medvision.local', 'Clinician Two');

INSERT INTO cases (id, owner_id, reference_code) VALUES
('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '11111111-1111-1111-1111-111111111111', 'MV-1024'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '22222222-2222-2222-2222-222222222222', 'MV-1023');

CREATE INDEX analysis_jobs_status_created_idx ON analysis_jobs (status, created_at);
CREATE INDEX medical_images_case_id_idx ON medical_images (case_id);
