"""
src/db/connection.py
--------------------
Single point of entry for MongoDB Atlas.

Usage:
    from src.db.connection import get_db

    db = get_db()
    collection = db["images_metadata"]
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

DB_NAME = "rdds"


@lru_cache(maxsize=1)
def _get_client() -> MongoClient:
    """
    Return a cached MongoClient.
    The client is thread-safe and reused across all calls in the same process.
    """
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise EnvironmentError(
            "MONGO_URI is not set. "
            "Copy .env.example to .env and fill in the connection string."
        )
    return MongoClient(uri)


def get_db() -> Database:
    """Return the 'rdds' database handle."""
    return _get_client()[DB_NAME]
