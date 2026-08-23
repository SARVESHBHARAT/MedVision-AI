-- MedVision AI: initial PostgreSQL schema (development only)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE analysis_status AS ENUM ('queued', 'processing', 'completed', 'failed');

CREATE TABLE app_users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES app_users(id),
    reference_code VARCHAR(32) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE medical_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    storage_key TEXT NOT NULL UNIQUE,
    content_type VARCHAR(64) NOT NULL CHECK (content_type IN ('image/jpeg', 'image/png')),
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0 AND size_bytes <= 10485760),
    checksum_sha256 CHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE analysis_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID NOT NULL REFERENCES medical_images(id) ON DELETE CASCADE,
    status analysis_status NOT NULL DEFAULT 'queued',
    model_version VARCHAR(64),
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE audit_events (
    id BIGSERIAL PRIMARY KEY,
    actor_id UUID REFERENCES app_users(id),
    event_type VARCHAR(80) NOT NULL,
    entity_type VARCHAR(80) NOT NULL,
    entity_id UUID NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Application assigns this setting after authenticating each request.
ALTER TABLE cases ENABLE ROW LEVEL SECURITY;
CREATE POLICY case_owner_access ON cases
    USING (owner_id = current_setting('app.current_user_id', true)::uuid)
    WITH CHECK (owner_id = current_setting('app.current_user_id', true)::uuid);

CREATE OR REPLACE FUNCTION audit_case_change() RETURNS trigger AS $$
BEGIN
    INSERT INTO audit_events (event_type, entity_type, entity_id, metadata)
    VALUES (TG_OP, 'case', COALESCE(NEW.id, OLD.id), jsonb_build_object('reference_code', COALESCE(NEW.reference_code, OLD.reference_code)));
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER cases_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON cases
FOR EACH ROW EXECUTE FUNCTION audit_case_change();
