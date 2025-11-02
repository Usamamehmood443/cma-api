from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import sys

# Import your existing script functions
from free_top_deals_cma_report import run_cma_from_params

app = FastAPI(
    title="CMA Report API",
    description="Generate Comparative Market Analysis reports for real estate properties",
    version="1.0.0"
)

class SubjectProperty(BaseModel):
    """Subject property details"""
    id: Optional[str] = "form address"
    lot_size_m2: float
    built_up_size_m2: float
    bedrooms: int
    baths: int
    latitude: float
    longitude: float
    price: Optional[float] = None
    image_url: Optional[str] = None

class CMARequest(BaseModel):
    """Request model for CMA generation"""
    subject: SubjectProperty
    top_n: Optional[int] = 5
    start_radius_km: Optional[float] = 1.0
    step_km: Optional[float] = 1.0
    max_radius_km: Optional[float] = 25.0
    size_tol_land: Optional[float] = 200.0
    size_tol_built: Optional[float] = 100.0
    min_required: Optional[int] = 3
    price_iqr_k: Optional[float] = 0.5

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CMA Report Generator API",
        "version": "1.0.0"
    }

@app.post("/generate-cma")
async def generate_cma(request: CMARequest):
    """
    Generate a CMA report for a subject property
    
    Returns analysis results including:
    - Average comparable price
    - Discount percentage
    - PDF report (optional)
    """
    try:
        # Prepare subject data in CSV format
        subject_csv = {
            "ID": request.subject.id,
            "Lot size (M^2)": request.subject.lot_size_m2,
            "Built up size (M^2)": request.subject.built_up_size_m2,
            "Bedrooms": request.subject.bedrooms,
            "Baths": request.subject.baths,
            "Latitude": request.subject.latitude,
            "Longitude": request.subject.longitude,
        }
        
        if request.subject.price:
            subject_csv["Price ($)"] = request.subject.price
        
        if request.subject.image_url:
            subject_csv["Image URL"] = request.subject.image_url
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            output_path = tmp.name
        
        # Prepare parameters
        params = {
            "csv": "properties.csv",  # Should be uploaded or configured
            "out": output_path,
            "subject_csv": subject_csv,
            "top_n": request.top_n,
            "start_radius_km": request.start_radius_km,
            "step_km": request.step_km,
            "max_radius_km": request.max_radius_km,
            "size_tol_land": request.size_tol_land,
            "size_tol_built": request.size_tol_built,
            "min_required": request.min_required,
            "price_iqr_k": request.price_iqr_k,
        }
        
        # Generate CMA report
        result = run_cma_from_params(params)
        
        # Return results (without PDF for now)
        return {
            "success": True,
            "data": {
                "target_id": result["target_id"],
                "asking_price": result["ask"],
                "average_comparable_price": result["avg_price"],
                "discount_percentage": result["discount_pct"],
                "discount_amount": result["discount_abs"],
                "comparable_price_range": {
                    "low": result["comp_low"],
                    "high": result["comp_high"]
                }
            },
            "pdf_generated": os.path.exists(output_path),
            "message": "CMA report generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CMA: {str(e)}")

@app.post("/generate-cma-with-pdf")
async def generate_cma_with_pdf(request: CMARequest):
    """
    Generate a CMA report and return the PDF file
    """
    try:
        # Similar to above but return the PDF file
        subject_csv = {
            "ID": request.subject.id,
            "Lot size (M^2)": request.subject.lot_size_m2,
            "Built up size (M^2)": request.subject.built_up_size_m2,
            "Bedrooms": request.subject.bedrooms,
            "Baths": request.subject.baths,
            "Latitude": request.subject.latitude,
            "Longitude": request.subject.longitude,
        }
        
        if request.subject.price:
            subject_csv["Price ($)"] = request.subject.price
        if request.subject.image_url:
            subject_csv["Image URL"] = request.subject.image_url
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            output_path = tmp.name
        
        params = {
            "csv": "properties.csv",
            "out": output_path,
            "subject_csv": subject_csv,
            "top_n": request.top_n,
            "start_radius_km": request.start_radius_km,
            "step_km": request.step_km,
            "max_radius_km": request.max_radius_km,
            "size_tol_land": request.size_tol_land,
            "size_tol_built": request.size_tol_built,
            "min_required": request.min_required,
            "price_iqr_k": request.price_iqr_k,
        }
        
        result = run_cma_from_params(params)
        
        # Return the PDF file
        if os.path.exists(output_path):
            return FileResponse(
                output_path,
                media_type="application/pdf",
                filename=f"cma_report_{result['target_id']}.pdf"
            )
        else:
            raise HTTPException(status_code=500, detail="PDF generation failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating CMA: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "csv_exists": os.path.exists("properties.csv"),
        "temp_dir_writable": os.access(tempfile.gettempdir(), os.W_OK)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
