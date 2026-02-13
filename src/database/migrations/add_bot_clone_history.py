# Migration script to add bot_clone_history table
from src.database.db_manager import DBManager
from src.database.models import Base, BotCloneHistory

db = DBManager()

def upgrade():
    Base.metadata.create_all(bind=db.engine, tables=[BotCloneHistory.__table__])

def downgrade():
    BotCloneHistory.__table__.drop(db.engine, checkfirst=True)

if __name__ == "__main__":
    upgrade()
