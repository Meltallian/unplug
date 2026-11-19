import os

os.environ["DATABASE_URL"] = "sqlite:////tmp/unplug-test.db"
os.environ["DEV_USER_EMAIL"] = "tester@example.com"
os.environ["ADMIN_EMAILS"] = "admin@example.com"

import pytest
from fastapi.testclient import TestClient

from app import main
from app.db import Base, engine, init_db


@pytest.fixture
def client(monkeypatch):
    init_db()
    enqueued = []
    monkeypatch.setattr(main, "_enqueue", enqueued.append)
    with TestClient(main.app) as c:
        c.enqueued = enqueued
        yield c
    Base.metadata.drop_all(engine)
