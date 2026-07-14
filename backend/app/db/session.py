from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.config.config import settings
from backend.app.shared.logging import logger
from backend.app.shared.state import mock_db_store

engine = create_async_engine(settings.DATABASE_URL, echo=False, future=True)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class MockSession:
    async def add(self, item):
        mock_db_store.append(item)

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def close(self):
        pass

    async def execute(self, query):
        class MockResult:
            def scalars(self):
                class MockScalars:
                    def all(self):
                        return mock_db_store

                return MockScalars()

        return MockResult()


async def get_db():
    try:
        # Check connection or try to use Postgres Session
        async with SessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except Exception as e:
        logger.warning(
            "Database connection failed, falling back to mock in-memory DB",
            error=str(e),
        )
        yield MockSession()
