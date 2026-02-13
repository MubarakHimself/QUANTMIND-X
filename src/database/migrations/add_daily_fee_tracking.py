# Migration script to add daily_fee_tracking table
from src.database.db_manager import DBManager
from src.database.models import Base, DailyFeeTracking

db = DBManager()

def upgrade():
    Base.metadata.create_all(bind=db.engine, tables=[DailyFeeTracking.__table__])

def downgrade():
    DailyFeeTracking.__table__.drop(db.engine, checkfirst=True)

if __name__ == "__main__":
    upgrade()
