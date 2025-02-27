let isProcessing = false;

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480 
            } 
        });
        video.srcObject = stream;
    } catch (error) {
        showError('Error accessing webcam: ' + error.message);
    }
}

async function captureAndDetect() {
    if (isProcessing) return;
    isProcessing = true;
    
    try {
        // Capture frame from webcam
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        // Convert to JPEG with quality 80%
        const blob = await new Promise(resolve => 
            canvas.toBlob(resolve, 'image/jpeg', 0.8)
        );
        
        // Show processing status
        showStatus('Processing face detection...', 'blue');
        
        // Send to face detection API
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        
        const detectionResponse = await fetch('/start', {
            method: 'POST',
            body: formData
        });
        
        const detectionResult = await detectionResponse.json();
        
        if (!detectionResult.success) {
            throw new Error(detectionResult.message);
        }
        
        if (detectionResult.student_id) {
            // Mark attendance if face recognized
            const attendanceResponse = await fetch('/mark_attendance', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    student_id: detectionResult.student_id 
                })
            });
            
            const attendanceResult = await attendanceResponse.json();
            
            if (!attendanceResult.success) {
                throw new Error(attendanceResult.message);
            }
            
            showStatus(`Attendance marked for ID: ${detectionResult.student_id}`, 'green');
        } else {
            showStatus('No recognized student found', 'orange');
        }
        
    } catch (error) {
        showError(error.message);
    } finally {
        isProcessing = false;
    }
}

function showStatus(message, color = 'black') {
    resultDiv.textContent = message;
    resultDiv.style.color = color;
}

function showError(message) {
    resultDiv.textContent = 'Error: ' + message;
    resultDiv.style.color = 'red';
}

// Initialize webcam when page loads
if (video) {
    startWebcam();
}