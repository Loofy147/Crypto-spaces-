"""Initialize the PostgreSQL schema before the API is promoted."""
from src.storage.postgres_repository import PostgresRepository


def main() -> None:
    with PostgresRepository.from_dsn() as repository:
        repository.initialize_schema()


if __name__ == "__main__":
    main()
