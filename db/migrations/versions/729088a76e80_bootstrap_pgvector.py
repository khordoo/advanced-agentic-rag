"""bootstrap_pgvector

Revision ID: 729088a76e80
Revises:
Create Date: 2026-04-26 12:42:10.602607

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

# revision identifiers, used by Alembic.
revision: str = '729088a76e80'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

def downgrade() -> None:
    op.execute('DROP EXTENSION IF EXISTS vector')
