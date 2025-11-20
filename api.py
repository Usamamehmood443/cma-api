from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import sys
import math

# Import your existing script functions
from free_top_deals_cma_report import run_cma_from_params

app = FastAPI(
    title="CMA Report API",
    description="Generate Comparative Market Analysis reports for real estate properties",
    version="1.0.0"
)

def clean_float(value):
    """Convert NaN and Infinity to None for JSON compliance"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
    return value

def clean_result_dict(data: dict) -> dict:
    """Recursively clean all float values in a dictionary"""
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, dict):
            cleaned[key] = clean_result_dict(value)
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [clean_float(v) if isinstance(v, (int, float)) else v for v in value]
        else:
            cleaned[key] = clean_float(value)
    return cleaned

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
    top_n: Optional[int] = 3
    start_radius_km: Optional[float] = 1.0
    step_km: Optional[float] = 1.0
    max_radius_km: Optional[float] = 10.0
    size_tol_land: Optional[float] = 0.2
    size_tol_built: Optional[float] = 0.2
    min_required: Optional[int] = 3

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
        # Get absolute path to CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "properties.csv")
        
        # Verify CSV exists
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=500, 
                detail=f"CSV file not found at {csv_path}. Working directory: {os.getcwd()}"
            )
        
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
            "csv": csv_path,  # Use absolute path
            "out": output_path,
            "subject_csv": subject_csv,
            "top_n": request.top_n,
            "start_radius_km": request.start_radius_km,
            "step_km": request.step_km,
            "max_radius_km": request.max_radius_km,
            "size_tol_land": request.size_tol_land,
            "size_tol_built": request.size_tol_built,
            "min_required": request.min_required,
        }
        
        # Generate CMA report
        result = run_cma_from_params(params)
        
        # Clean the results to handle NaN/Infinity values
        response_data = {
            "success": True,
            "data": {
                "target_id": result["target_id"],
                "asking_price": clean_float(result.get("ask")),
                "average_comparable_price": clean_float(result.get("avg_price")),
                "discount_percentage": clean_float(result.get("discount_pct")),
                "discount_amount": clean_float(result.get("discount_abs")),
                "comparable_price_range": {
                    "low": clean_float(result.get("comp_low")),
                    "high": clean_float(result.get("comp_high"))
                }
            },
            "pdf_generated": os.path.exists(output_path),
            "message": "CMA report generated successfully"
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error generating CMA: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

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
        
        # Get absolute path to CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "properties.csv")
        
        params = {
            "csv": csv_path,  # Use absolute path
            "out": output_path,
            "subject_csv": subject_csv,
            "top_n": request.top_n,
            "start_radius_km": request.start_radius_km,
            "step_km": request.step_km,
            "max_radius_km": request.max_radius_km,
            "size_tol_land": request.size_tol_land,
            "size_tol_built": request.size_tol_built,
            "min_required": request.min_required,
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

@app.post("/generate-cma-base64")
async def generate_cma_base64(request: CMARequest):
    """
    Generate a CMA report and return the PDF as base64 string
    
    Returns:
    {
        "success": true,
        "data": { ... CMA analysis data ... },
        "pdf_base64": "JVBERi0xLjQKJeLjz9MKMyAwIG9iago8PC...",
        "filename": "cma_report_form_address.pdf"
    }
    """
    try:
        import base64
        
        # Get absolute path to CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "properties.csv")
        
        # Verify CSV exists
        if not os.path.exists(csv_path):
            raise HTTPException(
                status_code=500, 
                detail=f"CSV file not found at {csv_path}"
            )
        
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
            "csv": csv_path,
            "out": output_path,
            "subject_csv": subject_csv,
            "top_n": request.top_n,
            "start_radius_km": request.start_radius_km,
            "step_km": request.step_km,
            "max_radius_km": request.max_radius_km,
            "size_tol_land": request.size_tol_land,
            "size_tol_built": request.size_tol_built,
            "min_required": request.min_required,
        }
        
        # Generate CMA report
        result = run_cma_from_params(params)
        
        # Read the PDF file and convert to base64
        pdf_base64 = None
        if os.path.exists(output_path):
            with open(output_path, 'rb') as pdf_file:
                pdf_bytes = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Clean up the temporary file
            try:
                os.unlink(output_path)
            except:
                pass
        
        # Clean the results and prepare response
        response_data = {
            "success": True,
            "data": {
                "target_id": result["target_id"],
                "asking_price": clean_float(result.get("ask")),
                "average_comparable_price": clean_float(result.get("avg_price")),
                "discount_percentage": clean_float(result.get("discount_pct")),
                "discount_amount": clean_float(result.get("discount_abs")),
                "comparable_price_range": {
                    "low": clean_float(result.get("comp_low")),
                    "high": clean_float(result.get("comp_high"))
                }
            },
            "pdf_base64": pdf_base64,
            "filename": f"cma_report_{result['target_id']}.pdf",
            "message": "CMA report generated successfully"
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error generating CMA: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/health")
async def health_check():
    """Detailed health check"""
    csv_path = os.path.join(os.path.dirname(__file__), "properties.csv")
    return {
        "status": "healthy",
        "working_directory": os.getcwd(),
        "script_directory": os.path.dirname(__file__),
        "csv_path": csv_path,
        "csv_exists": os.path.exists(csv_path),
        "temp_dir_writable": os.access(tempfile.gettempdir(), os.W_OK),
        "files_in_directory": os.listdir(os.path.dirname(__file__) or '.')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
