"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-08-12
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chats_id", "chats", ["id"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("doc_type", sa.String(length=100), nullable=False),
        sa.Column("indexing_status", sa.String(length=20), nullable=False),
        sa.Column("indexing_progress", sa.Integer(), nullable=False),
        sa.Column("indexing_error", sa.String(), nullable=True),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("storage_path", sa.String(length=255), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_documents_id", "documents", ["id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("previous_id", sa.Integer(), nullable=True),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=True),
        sa.Column("content", sa.JSON(), nullable=True),
        sa.Column("meta_data", sa.JSON(), nullable=True),
        sa.Column("token_usage", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["previous_id"], ["messages.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_chat_id", "messages", ["chat_id"])
    op.create_index("ix_messages_id", "messages", ["id"])
    op.create_index("ix_messages_previous_id", "messages", ["previous_id"])


def downgrade() -> None:
    op.drop_index("ix_messages_previous_id", table_name="messages")
    op.drop_index("ix_messages_id", table_name="messages")
    op.drop_index("ix_messages_chat_id", table_name="messages")
    op.drop_table("messages")

    op.drop_index("ix_documents_id", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_chats_id", table_name="chats")
    op.drop_table("chats")
