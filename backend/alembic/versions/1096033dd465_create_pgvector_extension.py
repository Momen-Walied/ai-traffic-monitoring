"""create pgvector extension

Revision ID: 1096033dd465
Revises: 8feac1ca7289
Create Date: 2025-09-21 18:24:46.501068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1096033dd465'
down_revision: Union[str, Sequence[str], None] = '8feac1ca7289'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
