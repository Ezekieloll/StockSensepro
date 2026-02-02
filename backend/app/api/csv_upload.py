"""
CSV Upload API for importing transaction data via staging tables.
Implements the pipeline: Upload → Staging DB → Validate → Approve → Master DB
"""

import csv
import io
import json
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.models.staging import StagingUpload, StagingTransaction
from app.models.transaction import Transaction, DailyDemand
from app.models.user import User


router = APIRouter(prefix="/csv-upload", tags=["CSV Upload"])


# Expected CSV columns (matching import_transactions.py)
REQUIRED_COLUMNS = ['date', 'store_id', 'sku', 'quantity']
OPTIONAL_COLUMNS = ['timestamp', 'product_category', 'event_type', 'price', 'holiday_flag', 'weather']


def validate_csv_row(row: dict, row_num: int) -> tuple[bool, Optional[str]]:
    """
    Validate a single CSV row.
    Returns (is_valid, error_message)
    """
    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in row or not row[col]:
            return False, f"Row {row_num}: Missing required column '{col}'"
    
    # Validate date format
    try:
        datetime.strptime(row['date'], '%Y-%m-%d')
    except ValueError:
        return False, f"Row {row_num}: Invalid date format (expected YYYY-MM-DD)"
    
    # Validate quantity is numeric
    try:
        qty = float(row['quantity'])
        if qty < 0:
            return False, f"Row {row_num}: Quantity cannot be negative"
    except ValueError:
        return False, f"Row {row_num}: Invalid quantity value"
    
    return True, None


@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    uploaded_by: str = "admin@stocksense.com",  # TODO: Get from auth session
    db: Session = Depends(get_db)
):
    """
    Upload CSV file to staging area for validation.
    Creates StagingUpload and StagingTransaction records.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read CSV content
    content = await file.read()
    decoded = content.decode('utf-8')
    csv_reader = csv.DictReader(io.StringIO(decoded))
    
    # Normalize column names (lowercase, strip spaces)
    fieldnames = [col.strip().lower() for col in csv_reader.fieldnames]
    
    # Check required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing_cols)}"
        )
    
    # Create staging upload record
    staging_upload = StagingUpload(
        filename=file.filename,
        uploaded_by=uploaded_by,
        status='pending'
    )
    db.add(staging_upload)
    db.flush()  # Get upload_id
    
    # Parse and validate rows
    staging_transactions = []
    row_count = 0
    valid_rows = 0
    invalid_rows = 0
    validation_errors = []
    dates = []
    
    for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 (row 1 is header)
        row_count += 1
        
        # Normalize keys
        normalized_row = {k.strip().lower(): v.strip() if v else None for k, v in row.items()}
        
        # Validate row
        is_valid, error_msg = validate_csv_row(normalized_row, row_num)
        
        if not is_valid:
            invalid_rows += 1
            validation_errors.append(error_msg)
            if len(validation_errors) > 50:  # Limit error messages
                validation_errors.append(f"... and {row_count - row_num} more errors")
                break
        else:
            valid_rows += 1
        
        # Parse date
        try:
            row_date = datetime.strptime(normalized_row['date'], '%Y-%m-%d').date()
            dates.append(row_date)
        except:
            row_date = None
        
        # Create staging transaction
        staging_tx = StagingTransaction(
            upload_id=staging_upload.id,
            date=row_date,
            store_id=normalized_row.get('store_id'),
            product_id=normalized_row.get('sku'),  # CSV uses 'sku', DB uses 'product_id'
            product_category=normalized_row.get('product_category'),
            event_type=normalized_row.get('event_type', 'sale'),
            quantity=float(normalized_row['quantity']) if normalized_row.get('quantity') else 0,
            price=float(normalized_row['price']) if normalized_row.get('price') else None,
            holiday_flag=int(normalized_row['holiday_flag']) if normalized_row.get('holiday_flag') else None,
            weather=normalized_row.get('weather'),
            is_valid=1 if is_valid else 0,
            validation_error=error_msg
        )
        staging_transactions.append(staging_tx)
    
    # Bulk insert staging transactions
    if staging_transactions:
        db.bulk_save_objects(staging_transactions)
    
    # Update staging upload summary
    staging_upload.row_count = row_count
    staging_upload.valid_rows = valid_rows
    staging_upload.invalid_rows = invalid_rows
    staging_upload.min_date = min(dates) if dates else None
    staging_upload.max_date = max(dates) if dates else None
    
    if invalid_rows > 0:
        staging_upload.status = 'error'
        staging_upload.error_message = '\n'.join(validation_errors)
    else:
        staging_upload.status = 'pending'
    
    staging_upload.validation_summary = json.dumps({
        'total_rows': row_count,
        'valid_rows': valid_rows,
        'invalid_rows': invalid_rows,
        'date_range': {
            'min': staging_upload.min_date.isoformat() if staging_upload.min_date else None,
            'max': staging_upload.max_date.isoformat() if staging_upload.max_date else None
        }
    })
    
    db.commit()
    db.refresh(staging_upload)
    
    return {
        "upload_id": staging_upload.id,
        "filename": staging_upload.filename,
        "status": staging_upload.status,
        "row_count": row_count,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "date_range": {
            "min": staging_upload.min_date.isoformat() if staging_upload.min_date else None,
            "max": staging_upload.max_date.isoformat() if staging_upload.max_date else None
        },
        "errors": validation_errors if validation_errors else None
    }


@router.get("/staging")
async def get_staging_queue(db: Session = Depends(get_db)):
    """
    Get all pending/error uploads in staging queue.
    """
    uploads = db.query(StagingUpload).filter(
        StagingUpload.status.in_(['pending', 'error'])
    ).order_by(StagingUpload.uploaded_at.desc()).all()
    
    return [{
        "id": upload.id,
        "filename": upload.filename,
        "uploaded_by": upload.uploaded_by,
        "uploaded_at": upload.uploaded_at.isoformat(),
        "status": upload.status,
        "row_count": upload.row_count,
        "valid_rows": upload.valid_rows,
        "invalid_rows": upload.invalid_rows,
        "date_range": {
            "min": upload.min_date.isoformat() if upload.min_date else None,
            "max": upload.max_date.isoformat() if upload.max_date else None
        },
        "error_message": upload.error_message
    } for upload in uploads]


@router.get("/staging/{upload_id}/preview")
async def preview_staging_data(upload_id: int, limit: int = 20, db: Session = Depends(get_db)):
    """
    Get preview of staging data for review before approval.
    """
    upload = db.query(StagingUpload).filter(StagingUpload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Get sample transactions
    transactions = db.query(StagingTransaction).filter(
        StagingTransaction.upload_id == upload_id
    ).limit(limit).all()
    
    return {
        "upload": {
            "id": upload.id,
            "filename": upload.filename,
            "status": upload.status,
            "row_count": upload.row_count,
            "valid_rows": upload.valid_rows,
            "invalid_rows": upload.invalid_rows
        },
        "preview": [{
            "date": tx.date.isoformat() if tx.date else None,
            "store_id": tx.store_id,
            "product_id": tx.product_id,
            "quantity": tx.quantity,
            "is_valid": bool(tx.is_valid),
            "error": tx.validation_error
        } for tx in transactions]
    }


@router.post("/staging/{upload_id}/approve")
async def approve_staging_upload(upload_id: int, db: Session = Depends(get_db)):
    """
    Approve staged upload and transfer to master DB.
    Inserts into transactions and aggregates to daily_demand.
    """
    upload = db.query(StagingUpload).filter(StagingUpload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    if upload.status not in ['pending', 'error']:
        raise HTTPException(status_code=400, detail=f"Cannot approve upload with status: {upload.status}")
    
    if upload.invalid_rows > 0:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve upload with {upload.invalid_rows} invalid rows. Fix errors first."
        )
    
    # Get all valid staging transactions
    staging_txs = db.query(StagingTransaction).filter(
        StagingTransaction.upload_id == upload_id,
        StagingTransaction.is_valid == 1
    ).all()
    
    if not staging_txs:
        raise HTTPException(status_code=400, detail="No valid transactions to import")
    
    # Transfer to master transactions table
    master_transactions = []
    for stx in staging_txs:
        tx = Transaction(
            timestamp=stx.timestamp or datetime.combine(stx.date, datetime.min.time()),
            date=stx.date,
            store_id=stx.store_id,
            product_id=stx.product_id,
            product_category=stx.product_category or 'Unknown',
            event_type=stx.event_type,
            quantity=stx.quantity,
            on_hand_before=stx.on_hand_before,
            on_hand_after=stx.on_hand_after,
            source=stx.source,
            destination=stx.destination,
            price=stx.price,
            holiday_flag=stx.holiday_flag or 0,
            weather=stx.weather
        )
        master_transactions.append(tx)
    
    db.bulk_save_objects(master_transactions)
    
    # Aggregate to daily_demand
    # Group by (date, store_id, product_id)
    aggregation = db.query(
        StagingTransaction.date,
        StagingTransaction.store_id,
        StagingTransaction.product_id,
        func.sum(StagingTransaction.quantity).label('total_quantity')
    ).filter(
        StagingTransaction.upload_id == upload_id,
        StagingTransaction.is_valid == 1
    ).group_by(
        StagingTransaction.date,
        StagingTransaction.store_id,
        StagingTransaction.product_id
    ).all()
    
    for agg in aggregation:
        # Check if daily_demand entry exists
        existing = db.query(DailyDemand).filter(
            DailyDemand.date == agg.date,
            DailyDemand.store_id == agg.store_id,
            DailyDemand.product_id == agg.product_id
        ).first()
        
        if existing:
            # Update existing demand
            existing.demand += agg.total_quantity
        else:
            # Create new demand entry
            daily_demand = DailyDemand(
                date=agg.date,
                store_id=agg.store_id,
                product_id=agg.product_id,
                demand=agg.total_quantity
            )
            db.add(daily_demand)
    
    # Update staging upload status
    upload.status = 'approved'
    upload.processed_at = datetime.now()
    
    db.commit()
    
    return {
        "message": "Upload approved and imported successfully",
        "upload_id": upload_id,
        "rows_imported": len(staging_txs),
        "daily_demand_updated": len(aggregation)
    }


@router.post("/staging/{upload_id}/reject")
async def reject_staging_upload(upload_id: int, db: Session = Depends(get_db)):
    """
    Reject and delete staged upload.
    """
    upload = db.query(StagingUpload).filter(StagingUpload.id == upload_id).first()
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    upload.status = 'rejected'
    upload.processed_at = datetime.now()
    
    # Delete staging transactions (CASCADE will handle this)
    db.delete(upload)
    db.commit()
    
    return {
        "message": "Upload rejected and deleted",
        "upload_id": upload_id
    }
