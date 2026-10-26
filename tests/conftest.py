"""
Shared test setup.

Settings are read when `app` is first imported, so the fake values must be
set here, before any test module imports `app`. Environment variables win
over .env, so tests never use the real keys or reach real services.
"""

import os

os.environ.update({
    "SUPABASE_URL": "http://localhost:54321",
    "SUPABASE_PUBLISHABLE_KEY": "test-publishable-key",
    "SUPABASE_SECRET_KEY": "test-secret-key",
    "OPENAI_API_KEY": "test-openai-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "AI_PROVIDER": "openai",
    "ADMIN_API_KEY": "test-admin-key",
    "MIN_SIMILARITY": "0.3",
})
