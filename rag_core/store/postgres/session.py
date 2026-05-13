import logging
from rag_core.core.config import get_postgres_settings, PostgresSettings
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from functools import lru_cache

logger = logging.getLogger(__name__)

class Database:
    """Owns the Postgres engine and session factory for application use.

    The SQLAlchemy engine is intentionally long-lived because it manages the
    connection pool. Individual sessions should be created through
    :meth:`get_session` and kept scoped to a single unit of work.
    """

    def __init__(self,settings:PostgresSettings| None=None) -> None:
        """Create a database wrapper from explicit or environment settings."""
        self.settings = settings or get_postgres_settings()
        self.engine = self._create_engine()
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine)

    def _create_engine(self):
        """Create the SQLAlchemy engine using the configured pool settings."""
        return create_engine(
            url=self.settings.uri,
            pool_pre_ping=self.settings.pool_pre_ping,
            max_overflow=self.settings.max_overflow,
            pool_size=self.settings.pool_size,
            pool_recycle=self.settings.pool_recycle,
            pool_timeout=self.settings.pool_timeout,
            connect_args=self.settings.connect_args,
        )

    @contextmanager
    def get_session(self):
        """Yield a raw, open session. The caller MUST manage transactions."""
        with self.session_factory() as session:
             yield session

@lru_cache(maxsize=1)
def get_database() ->Database:
    """Return the process-wide database wrapper and connection pool."""
    logger.info("Initializing Database connection pool...")
    return Database()
