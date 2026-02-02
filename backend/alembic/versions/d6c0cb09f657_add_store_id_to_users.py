"""add_store_id_to_users

Revision ID: d6c0cb09f657
Revises: 47430731dfc2
Create Date: 2026-01-30 11:21:08.675163

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd6c0cb09f657'
down_revision: Union[str, Sequence[str], None] = '47430731dfc2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add store_id column to users table
    op.add_column('users', sa.Column('store_id', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Remove store_id column from users table
    op.drop_column('users', 'store_id')
