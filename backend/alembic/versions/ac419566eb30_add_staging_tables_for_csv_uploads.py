"""Add staging tables for CSV uploads

Revision ID: ac419566eb30
Revises: f1ccf32abd39
Create Date: 2026-02-01 11:15:45.347563

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ac419566eb30'
down_revision: Union[str, Sequence[str], None] = 'f1ccf32abd39'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create staging_uploads table
    op.create_table(
        'staging_uploads',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('uploaded_by', sa.String(length=100), nullable=False),
        sa.Column('uploaded_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('row_count', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('valid_rows', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('invalid_rows', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('validation_summary', sa.Text(), nullable=True),
        sa.Column('min_date', sa.Date(), nullable=True),
        sa.Column('max_date', sa.Date(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create staging_transactions table
    op.create_table(
        'staging_transactions',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('upload_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('store_id', sa.String(length=10), nullable=False),
        sa.Column('product_id', sa.String(length=50), nullable=False),
        sa.Column('product_category', sa.String(length=10), nullable=True),
        sa.Column('event_type', sa.String(length=20), nullable=False),
        sa.Column('quantity', sa.Float(), nullable=False),
        sa.Column('on_hand_before', sa.Integer(), nullable=True),
        sa.Column('on_hand_after', sa.Integer(), nullable=True),
        sa.Column('source', sa.String(length=50), nullable=True),
        sa.Column('destination', sa.String(length=50), nullable=True),
        sa.Column('price', sa.Float(), nullable=True),
        sa.Column('holiday_flag', sa.Integer(), nullable=True),
        sa.Column('weather', sa.String(length=50), nullable=True),
        sa.Column('is_valid', sa.Integer(), nullable=True, server_default='1'),
        sa.Column('validation_error', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['upload_id'], ['staging_uploads.id'], ondelete='CASCADE')
    )
    
    # Create index on upload_id for faster queries
    op.create_index('ix_staging_transactions_upload_id', 'staging_transactions', ['upload_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_staging_transactions_upload_id', table_name='staging_transactions')
    op.drop_table('staging_transactions')
    op.drop_table('staging_uploads')
