"""add_ai_scenario_fields_to_adversarial_risk

Revision ID: f1ccf32abd39
Revises: e195bc4fa89a
Create Date: 2026-01-31 15:56:13.265334

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1ccf32abd39'
down_revision: Union[str, Sequence[str], None] = 'e195bc4fa89a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add AI scenario fields to adversarial_risk table
    op.add_column('adversarial_risk', sa.Column('sku_id', sa.String(), nullable=True))
    op.add_column('adversarial_risk', sa.Column('scenario_name', sa.String(), nullable=True))
    op.add_column('adversarial_risk', sa.Column('scenario_id', sa.String(), nullable=True))
    op.add_column('adversarial_risk', sa.Column('probability', sa.Float(), nullable=True))
    op.add_column('adversarial_risk', sa.Column('strategies', sa.Text(), nullable=True))
    op.add_column('adversarial_risk', sa.Column('priority_level', sa.String(), nullable=True))
    op.add_column('adversarial_risk', sa.Column('current_inventory', sa.Integer(), nullable=True))
    
    # Create indexes for better query performance
    op.create_index('ix_adversarial_risk_sku_id', 'adversarial_risk', ['sku_id'])
    op.create_index('ix_adversarial_risk_scenario_id', 'adversarial_risk', ['scenario_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index('ix_adversarial_risk_scenario_id', 'adversarial_risk')
    op.drop_index('ix_adversarial_risk_sku_id', 'adversarial_risk')
    
    # Drop columns
    op.drop_column('adversarial_risk', 'current_inventory')
    op.drop_column('adversarial_risk', 'priority_level')
    op.drop_column('adversarial_risk', 'strategies')
    op.drop_column('adversarial_risk', 'probability')
    op.drop_column('adversarial_risk', 'scenario_id')
    op.drop_column('adversarial_risk', 'scenario_name')
    op.drop_column('adversarial_risk', 'sku_id')
