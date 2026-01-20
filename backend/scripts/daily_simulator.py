"""
Daily Transaction Simulator

This script generates realistic simulated transactions based on 
historical patterns from 2023-2024 data.

Logic:
- For each date from last simulated date to today
- For each store and product
- Look at the same day-of-week from historical data
- Generate transactions with similar patterns (with some randomness)

Usage:
    python scripts/daily_simulator.py --catch-up
"""

import sys
import random
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL
from app.models.transaction import Transaction, DailyDemand
from app.database import Base


# Weather options
WEATHER_OPTIONS = ['Sunny', 'Cloudy', 'Rainy', 'Stormy']
WEATHER_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


def get_historical_pattern(session, product_id: str, store_id: str, day_of_week: int):
    """
    Get the historical demand pattern for a product on a specific day of week.
    """
    query = session.query(
        func.avg(DailyDemand.total_quantity).label('avg_demand'),
        func.avg(DailyDemand.avg_price).label('avg_price'),
        func.count(DailyDemand.id).label('data_points')
    ).filter(
        DailyDemand.product_id == product_id,
        DailyDemand.store_id == store_id,
        func.extract('dow', DailyDemand.date) == (day_of_week + 1) % 7
    ).first()
    
    avg_demand = float(query.avg_demand) if query.avg_demand else 0
    avg_price = float(query.avg_price) if query.avg_price else 100
    data_points = query.data_points or 0
    
    return avg_demand, avg_price, data_points


def get_product_category(session, product_id: str) -> str:
    """Get the category for a product."""
    result = session.query(DailyDemand.product_category).filter(
        DailyDemand.product_id == product_id,
        DailyDemand.product_category.isnot(None)
    ).first()
    return result[0] if result else product_id.split('_')[1][:4] if '_' in product_id else 'UNKN'


def is_holiday(target_date: date) -> int:
    """Simple holiday detection."""
    # Major holidays
    holidays = [
        (1, 1),   # New Year
        (1, 26),  # Republic Day
        (8, 15),  # Independence Day
        (10, 2),  # Gandhi Jayanti
        (12, 25), # Christmas
    ]
    return 1 if (target_date.month, target_date.day) in holidays else 0


def generate_daily_transactions(session, target_date: date, products: list, stores: list):
    """Generate simulated transactions for a specific date."""
    
    day_of_week = target_date.weekday()
    holiday_flag = is_holiday(target_date)
    weather = random.choices(WEATHER_OPTIONS, weights=WEATHER_WEIGHTS)[0]
    
    transactions = []
    daily_demands = []
    
    for store_id in stores:
        for product_id, category in products:
            # Get historical pattern
            avg_demand, avg_price, data_points = get_historical_pattern(
                session, product_id, store_id, day_of_week
            )
            
            if avg_demand == 0 or data_points < 5:
                avg_demand = random.uniform(5, 15)
                avg_price = random.uniform(50, 500)
            
            # Add randomness (±30%)
            noise = random.uniform(0.7, 1.3)
            
            # Holiday boost
            if holiday_flag:
                noise *= random.uniform(1.1, 1.4)
            
            # Weather effect
            if weather == 'Stormy':
                noise *= 0.8
            elif weather == 'Rainy':
                noise *= 0.9
            
            daily_quantity = max(0, round(avg_demand * noise, 1))
            
            if daily_quantity == 0:
                continue
            
            # Track on_hand simulation
            on_hand = random.randint(50, 200)
            
            # Generate individual transactions throughout the day
            num_transactions = max(1, int(daily_quantity / random.uniform(1, 3)))
            remaining_quantity = daily_quantity
            total_revenue = 0
            prices = []
            
            for i in range(num_transactions):
                if remaining_quantity <= 0:
                    break
                
                # Random time of day (weighted towards afternoon/evening)
                hour = random.choices(
                    range(8, 22),
                    weights=[1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3]
                )[0]
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                
                timestamp = datetime.combine(target_date, datetime.min.time().replace(
                    hour=hour, minute=minute, second=second
                ))
                
                # Quantity for this transaction
                if i == num_transactions - 1:
                    qty = remaining_quantity
                else:
                    qty = min(remaining_quantity, random.uniform(1, max(1, remaining_quantity / 2)))
                
                qty = round(qty, 1)
                remaining_quantity -= qty
                
                # Price with small variation
                price = round(avg_price * random.uniform(0.95, 1.05), 2)
                prices.append(price)
                total_revenue += price * qty
                
                on_hand_before = on_hand
                on_hand = max(0, on_hand - int(qty))
                on_hand_after = on_hand
                
                # Create transaction with exact CSV column names
                transaction = Transaction(
                    timestamp=timestamp,
                    date=target_date,
                    store_id=store_id,
                    product_id=product_id,
                    product_category=category,
                    event_type='sale',
                    quantity=qty,
                    on_hand_before=on_hand_before,
                    on_hand_after=on_hand_after,
                    source='POS',
                    destination='customer',
                    price=price,
                    holiday_flag=holiday_flag,
                    weather=weather,
                    is_simulated=1
                )
                transactions.append(transaction)
            
            # Create daily demand record
            avg_tx_price = sum(prices) / len(prices) if prices else avg_price
            daily_demands.append(DailyDemand(
                date=target_date,
                store_id=store_id,
                product_id=product_id,
                product_category=category,
                total_quantity=daily_quantity,
                transaction_count=num_transactions,
                total_revenue=round(total_revenue, 2),
                avg_price=round(avg_tx_price, 2)
            ))
    
    return transactions, daily_demands


def run_simulator(catch_up: bool = True, days_to_simulate: int = 1):
    """
    Run the daily simulator.
    """
    
    print("🎲 Daily Transaction Simulator")
    print("=" * 50)
    
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Get the last date in the database
    last_date_query = session.query(func.max(Transaction.date)).first()
    last_date = last_date_query[0] if last_date_query[0] else date(2024, 12, 31)
    
    print(f"📅 Last data date: {last_date}")
    print(f"📅 Today's date: {date.today()}")
    
    # Get all products with their categories
    products_query = session.query(
        DailyDemand.product_id,
        DailyDemand.product_category
    ).distinct().all()
    products = [(p[0], p[1]) for p in products_query]
    print(f"📦 Products to simulate: {len(products)}")
    
    # Get all stores
    stores = session.query(DailyDemand.store_id).distinct().all()
    stores = [s[0] for s in stores]
    print(f"🏪 Stores: {stores}")
    
    if catch_up:
        start_date = last_date + timedelta(days=1)
        end_date = date.today()
    else:
        start_date = last_date + timedelta(days=1)
        end_date = start_date + timedelta(days=days_to_simulate - 1)
    
    if start_date > end_date:
        print("✅ Database is already up to date!")
        session.close()
        return
    
    days_to_generate = (end_date - start_date).days + 1
    print(f"\n🔄 Generating data for {days_to_generate} days: {start_date} to {end_date}")
    
    total_transactions = 0
    total_demand_records = 0
    
    current_date = start_date
    while current_date <= end_date:
        print(f"   {current_date}...", end=" ", flush=True)
        
        transactions, daily_demands = generate_daily_transactions(
            session, current_date, products, stores
        )
        
        # Bulk insert
        if transactions:
            session.bulk_save_objects(transactions)
        if daily_demands:
            session.bulk_save_objects(daily_demands)
        
        session.commit()
        
        print(f"{len(transactions)} transactions")
        
        total_transactions += len(transactions)
        total_demand_records += len(daily_demands)
        current_date += timedelta(days=1)
    
    print(f"\n✅ Generated {total_transactions:,} transactions")
    print(f"✅ Generated {total_demand_records:,} daily demand records")
    
    # Verify
    new_max_date = session.query(func.max(Transaction.date)).first()[0]
    print(f"📅 New last date: {new_max_date}")
    
    session.close()
    print("\n🎉 Simulation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate simulated transactions")
    parser.add_argument("--catch-up", action="store_true", default=True,
                       help="Generate data from last date to today")
    parser.add_argument("--days", type=int, default=1,
                       help="Number of days to simulate (if not catching up)")
    
    args = parser.parse_args()
    
    run_simulator(catch_up=args.catch_up, days_to_simulate=args.days)
