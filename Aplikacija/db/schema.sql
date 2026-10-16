CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT,
    password_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;

CREATE TABLE IF NOT EXISTS threads (
    thread_id UUID PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    title TEXT,
    agent_variant TEXT NOT NULL,          
    extraction_mode TEXT,                 
    memory_types TEXT,                    
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE threads ADD COLUMN IF NOT EXISTS memory_types TEXT;

CREATE TABLE IF NOT EXISTS procedural_prompts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    version INTEGER NOT NULL,
    prompt_text TEXT NOT NULL,
    source_thread_ids UUID[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, version)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id SERIAL PRIMARY KEY,
    scenario_name TEXT NOT NULL,
    agent_variant TEXT NOT NULL,
    extraction_mode TEXT,
    user_id INTEGER NOT NULL REFERENCES users(id),
    thread_ids UUID[] NOT NULL,
    metrics_json JSONB NOT NULL,
    run_at TIMESTAMPTZ NOT NULL DEFAULT now()
);