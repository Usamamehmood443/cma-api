# Base64 PDF API Examples

## 🎯 Overview
The `/generate-cma-base64` endpoint returns the PDF as a base64-encoded string along with the analysis data.

---

## 📋 API Endpoint

**POST** `https://cma-api-d3cm.onrender.com/generate-cma-base64`

---

## 🔧 JavaScript/TypeScript Examples

### Example 1: Fetch API (Vanilla JS)

```javascript
async function generateCMAWithBase64() {
  const response = await fetch('https://cma-api-d3cm.onrender.com/generate-cma-base64', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      subject: {
        lot_size_m2: 459,
        built_up_size_m2: 133,
        bedrooms: 4,
        baths: 4,
        latitude: 12.5440,
        longitude: -70.0130
      }
    })
  });

  const result = await response.json();
  
  console.log('CMA Data:', result.data);
  console.log('Filename:', result.filename);
  
  // The PDF is in result.pdf_base64
  const pdfBase64 = result.pdf_base64;
  
  // Option 1: Download as file
  downloadBase64PDF(pdfBase64, result.filename);
  
  // Option 2: Display in iframe
  displayPDFInIframe(pdfBase64);
  
  // Option 3: Save to Wix Media Manager (if using Wix)
  // await uploadToWixMedia(pdfBase64, result.filename);
}

// Download the PDF as a file
function downloadBase64PDF(base64String, filename) {
  const linkSource = `data:application/pdf;base64,${base64String}`;
  const downloadLink = document.createElement('a');
  downloadLink.href = linkSource;
  downloadLink.download = filename;
  downloadLink.click();
}

// Display PDF in an iframe
function displayPDFInIframe(base64String) {
  const iframe = document.getElementById('pdf-viewer');
  iframe.src = `data:application/pdf;base64,${base64String}`;
}

// Convert base64 to Blob (useful for uploads)
function base64ToBlob(base64String, contentType = 'application/pdf') {
  const byteCharacters = atob(base64String);
  const byteArrays = [];

  for (let i = 0; i < byteCharacters.length; i++) {
    byteArrays.push(byteCharacters.charCodeAt(i));
  }

  const byteArray = new Uint8Array(byteArrays);
  return new Blob([byteArray], { type: contentType });
}
```

---

### Example 2: React Component

```jsx
import React, { useState } from 'react';

function CMAGenerator() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [pdfUrl, setPdfUrl] = useState(null);

  const generateCMA = async (propertyData) => {
    setLoading(true);
    
    try {
      const response = await fetch('https://cma-api-d3cm.onrender.com/generate-cma-base64', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subject: propertyData
        })
      });

      const data = await response.json();
      setResult(data);
      
      // Create object URL for the PDF
      const pdfDataUrl = `data:application/pdf;base64,${data.pdf_base64}`;
      setPdfUrl(pdfDataUrl);
      
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const propertyData = {
      lot_size_m2: 459,
      built_up_size_m2: 133,
      bedrooms: 4,
      baths: 4,
      latitude: 12.5440,
      longitude: -70.0130
    };
    
    generateCMA(propertyData);
  };

  return (
    <div>
      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Generating...' : 'Generate CMA Report'}
      </button>

      {result && (
        <div>
          <h3>Results</h3>
          <p>Average Price: ${result.data.average_comparable_price?.toLocaleString()}</p>
          <p>Discount: {result.data.discount_percentage?.toFixed(2)}%</p>
          
          {pdfUrl && (
            <>
              <a href={pdfUrl} download={result.filename}>
                Download PDF Report
              </a>
              
              <iframe 
                src={pdfUrl} 
                width="100%" 
                height="600px"
                title="CMA Report"
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default CMAGenerator;
```

---

### Example 3: Wix Velo (with Media Manager Upload)

```javascript
// In your Wix page code

import wixData from 'wix-data';
import { fetch } from 'wix-fetch';
import wixMediaBackend from 'wix-media-backend';

$w.onReady(function () {
  
  $w('#generateButton').onClick(async () => {
    $w('#loadingBar').show();
    
    const propertyData = {
      lot_size_m2: parseFloat($w('#lotSizeInput').value),
      built_up_size_m2: parseFloat($w('#builtSizeInput').value),
      bedrooms: parseInt($w('#bedroomsInput').value),
      baths: parseInt($w('#bathsInput').value),
      latitude: parseFloat($w('#latitudeInput').value),
      longitude: parseFloat($w('#longitudeInput').value)
    };

    try {
      // Call the API
      const response = await fetch('https://cma-api-d3cm.onrender.com/generate-cma-base64', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          subject: propertyData
        })
      });

      const result = await response.json();

      if (result.success) {
        // Display results
        $w('#avgPriceText').text = `$${Math.round(result.data.average_comparable_price).toLocaleString()}`;
        $w('#discountText').text = `${result.data.discount_percentage.toFixed(2)}%`;
        
        // Convert base64 to blob
        const pdfBlob = base64ToBlob(result.pdf_base64);
        
        // Upload to Wix Media Manager (backend function required)
        const mediaUrl = await uploadToMediaManager(result.pdf_base64, result.filename);
        
        // Show download link
        $w('#downloadButton').link = mediaUrl;
        $w('#downloadButton').show();
        
        // Or open in new tab
        const pdfDataUrl = `data:application/pdf;base64,${result.pdf_base64}`;
        window.open(pdfDataUrl, '_blank');
      }
      
    } catch (error) {
      console.error('Error:', error);
      $w('#errorMessage').text = 'Failed to generate report';
      $w('#errorMessage').show();
    } finally {
      $w('#loadingBar').hide();
    }
  });
});

// Helper function
function base64ToBlob(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return new Blob([bytes], { type: 'application/pdf' });
}

// Backend function to upload to Wix Media Manager
// (backend/media.jsw)
import wixMediaBackend from 'wix-media-backend';

export async function uploadToMediaManager(base64String, filename) {
  // Convert base64 to buffer
  const buffer = Buffer.from(base64String, 'base64');
  
  // Upload to media manager
  const uploadResult = await wixMediaBackend.mediaManager.upload(
    "/cma-reports",  // folder path
    buffer,
    filename,
    {
      "mediaOptions": {
        "mimeType": "application/pdf",
        "mediaType": "document"
      }
    }
  );
  
  return uploadResult.fileUrl;
}
```

---

### Example 4: Node.js (Backend)

```javascript
const axios = require('axios');
const fs = require('fs');

async function generateAndSaveCMA() {
  try {
    const response = await axios.post(
      'https://cma-api-d3cm.onrender.com/generate-cma-base64',
      {
        subject: {
          lot_size_m2: 459,
          built_up_size_m2: 133,
          bedrooms: 4,
          baths: 4,
          latitude: 12.5440,
          longitude: -70.0130
        }
      }
    );

    const { pdf_base64, filename, data } = response.data;
    
    // Save PDF to file
    const pdfBuffer = Buffer.from(pdf_base64, 'base64');
    fs.writeFileSync(filename, pdfBuffer);
    
    console.log(`PDF saved as ${filename}`);
    console.log('CMA Data:', data);
    
    return filename;
    
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

generateAndSaveCMA();
```

---

### Example 5: Python

```python
import requests
import base64

def generate_cma_with_base64():
    url = 'https://cma-api-d3cm.onrender.com/generate-cma-base64'
    
    payload = {
        'subject': {
            'lot_size_m2': 459,
            'built_up_size_m2': 133,
            'bedrooms': 4,
            'baths': 4,
            'latitude': 12.5440,
            'longitude': -70.0130
        }
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    if result['success']:
        # Decode base64 and save to file
        pdf_data = base64.b64decode(result['pdf_base64'])
        
        with open(result['filename'], 'wb') as f:
            f.write(pdf_data)
        
        print(f"PDF saved as {result['filename']}")
        print(f"Average Price: ${result['data']['average_comparable_price']:,.2f}")
        print(f"Discount: {result['data']['discount_percentage']:.2f}%")
    
    return result

if __name__ == '__main__':
    generate_cma_with_base64()
```

---

## 📦 Response Format

```json
{
  "success": true,
  "data": {
    "target_id": "form address",
    "asking_price": null,
    "average_comparable_price": 451666.6666666667,
    "discount_percentage": null,
    "discount_amount": null,
    "comparable_price_range": {
      "low": 185000,
      "high": 550000
    }
  },
  "pdf_base64": "JVBERi0xLjQKJeLjz9MKMyAwIG9iago8PC9UeXBlL...[very long string]",
  "filename": "cma_report_form_address.pdf",
  "message": "CMA report generated successfully"
}
```

---

## 🎯 Common Use Cases

### 1. Email the PDF
```javascript
async function emailCMAReport(email, propertyData) {
  const response = await fetch('https://cma-api-d3cm.onrender.com/generate-cma-base64', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ subject: propertyData })
  });
  
  const result = await response.json();
  
  // Send via your email service
  await sendEmail({
    to: email,
    subject: 'Your CMA Report',
    body: `Average comparable price: $${result.data.average_comparable_price}`,
    attachments: [{
      filename: result.filename,
      content: result.pdf_base64,
      encoding: 'base64'
    }]
  });
}
```

### 2. Store in Database
```javascript
// Store the base64 string directly in your database
await database.reports.create({
  propertyId: '123',
  pdfBase64: result.pdf_base64,
  avgPrice: result.data.average_comparable_price,
  createdAt: new Date()
});
```

### 3. Cloud Storage (AWS S3, Google Cloud, etc.)
```javascript
const blob = base64ToBlob(result.pdf_base64);
await uploadToS3(blob, result.filename);
```

---

## ⚠️ Important Notes

1. **Large Response**: Base64 encoded PDFs can be large (typically 1-3MB). Ensure your client can handle this.

2. **Memory**: Base64 encoding increases size by ~33%. Consider using the `/generate-cma-with-pdf` endpoint for direct downloads if size is a concern.

3. **Timeout**: First request after inactivity may take 30-40 seconds (cold start). Consider showing a loading indicator.

4. **Rate Limits**: On free tier, be mindful of concurrent requests.

---

## 🚀 Quick Test

Test the endpoint using cURL:

```bash
curl -X POST https://cma-api-d3cm.onrender.com/generate-cma-base64 \
  -H "Content-Type: application/json" \
  -d '{
    "subject": {
      "lot_size_m2": 459,
      "built_up_size_m2": 133,
      "bedrooms": 4,
      "baths": 4,
      "latitude": 12.5440,
      "longitude": -70.0130
    }
  }' > response.json

# Extract and save PDF
# (requires jq installed)
jq -r '.pdf_base64' response.json | base64 -d > report.pdf
```

Good luck! 🎉
