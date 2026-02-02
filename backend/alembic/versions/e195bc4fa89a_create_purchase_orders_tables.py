"""create_purchase_orders_tables

Revision ID: e195bc4fa89a
Revises: d6c0cb09f657
Create Date: 2026-01-30 11:24:46.743482

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e195bc4fa89a'
down_revision: Union[str, Sequence[str], None] = 'd6c0cb09f657'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create purchase_orders table
    op.create_table(
        'purchase_orders',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('po_number', sa.String(length=50), nullable=False),
        sa.Column('store_id', sa.String(length=10), nullable=False),
        sa.Column('destination', sa.String(length=50), nullable=False),
        sa.Column('source', sa.String(length=50), nullable=True),
        sa.Column('total_items', sa.Integer(), nullable=False),
        sa.Column('total_quantity', sa.Float(), nullable=False),
        sa.Column('total_amount', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('status', sa.Enum('draft', 'pending', 'approved', 'in_transit', 'delivered', 'cancelled', name='postatus'), nullable=False),
        sa.Column('created_by_user_id', sa.Integer(), nullable=False),
        sa.Column('approved_by_user_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('approved_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expected_delivery_date', sa.Date(), nullable=True),
        sa.Column('actual_delivery_date', sa.Date(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['approved_by_user_id'], ['users.id'], ),
        sa.ForeignKeyConstraint(['created_by_user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_purchase_orders_id'), 'purchase_orders', ['id'], unique=False)
    op.create_index(op.f('ix_purchase_orders_po_number'), 'purchase_orders', ['po_number'], unique=True)
    op.create_index(op.f('ix_purchase_orders_store_id'), 'purchase_orders', ['store_id'], unique=False)

    # Create purchase_order_items table
    op.create_table(
        'purchase_order_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('purchase_order_id', sa.Integer(), nullable=False),
        sa.Column('sku', sa.String(length=50), nullable=False),
        sa.Column('product_category', sa.String(length=10), nullable=True),
        sa.Column('quantity_requested', sa.Float(), nullable=False),
        sa.Column('quantity_delivered', sa.Float(), nullable=True),
        sa.Column('unit_price', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('line_total', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['purchase_order_id'], ['purchase_orders.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_purchase_order_items_id'), 'purchase_order_items', ['id'], unique=False)
    op.create_index(op.f('ix_purchase_order_items_sku'), 'purchase_order_items', ['sku'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop tables
    op.drop_index(op.f('ix_purchase_order_items_sku'), table_name='purchase_order_items')
    op.drop_index(op.f('ix_purchase_order_items_id'), table_name='purchase_order_items')
    op.drop_table('purchase_order_items')
    
    op.drop_index(op.f('ix_purchase_orders_store_id'), table_name='purchase_orders')
    op.drop_index(op.f('ix_purchase_orders_po_number'), table_name='purchase_orders')
    op.drop_index(op.f('ix_purchase_orders_id'), table_name='purchase_orders')
    op.drop_table('purchase_orders')
    
    # Drop enum type
    op.execute('DROP TYPE postatus')
