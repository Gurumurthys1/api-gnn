const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_API_URL = 'http://localhost:5001';

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, 'audio-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const fileFilter = (req, file, cb) => {
    const allowedTypes = ['.wav', '.wave'];
    const ext = path.extname(file.originalname).toLowerCase();
    
    if (allowedTypes.includes(ext)) {
        cb(null, true);
    } else {
        cb(new Error('Only WAV files are allowed!'), false);
    }
};

const upload = multer({
    storage: storage,
    fileFilter: fileFilter,
    limits: { fileSize: 50 * 1024 * 1024 } // 50MB limit
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', message: 'Backend server is running' });
});

// Upload and predict endpoint
app.post('/api/upload', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No file uploaded' });
        }

        console.log('File uploaded:', req.file.filename);

        // Send file to Python ML service
        const formData = new FormData();
        const fileBuffer = fs.readFileSync(req.file.path);
        const blob = new Blob([fileBuffer]);
        formData.append('file', blob, req.file.filename);

        try {
            const response = await axios.post(`${PYTHON_API_URL}/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
                maxBodyLength: Infinity,
                maxContentLength: Infinity
            });

            // Clean up uploaded file
            fs.unlinkSync(req.file.path);

            res.json({
                success: true,
                prediction: response.data
            });

        } catch (pythonError) {
            console.error('Python service error:', pythonError.message);
            
            // Clean up uploaded file
            if (fs.existsSync(req.file.path)) {
                fs.unlinkSync(req.file.path);
            }

            res.status(500).json({
                error: 'Failed to process audio file',
                details: pythonError.response?.data || pythonError.message
            });
        }

    } catch (error) {
        console.error('Upload error:', error);
        
        // Clean up uploaded file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        res.status(500).json({
            error: 'Server error',
            details: error.message
        });
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File size too large. Maximum 50MB allowed.' });
        }
        return res.status(400).json({ error: error.message });
    }
    
    res.status(500).json({ error: error.message });
});

// WhatsApp Webhook endpoint
app.post('/api/whatsapp', express.urlencoded({ extended: false }), async (req, res) => {
    const { MessagingResponse } = require('twilio').twiml;
    const twiml = new MessagingResponse();
    
    try {
        const numMedia = req.body.NumMedia;
        
        if (numMedia > 0) {
            const mediaUrl = req.body.MediaUrl0;
            const contentType = req.body.MediaContentType0;
            
            console.log(`📩 Received WhatsApp media: ${contentType} from ${req.body.From}`);
            
            // Generate unique filename
            const ext = contentType.split('/')[1] || 'wav'; // Default to wav if unknown, though likely ogg/amr
            const filename = `whatsapp-${Date.now()}.${ext}`;
            const filepath = path.join(__dirname, 'uploads', filename);
            
            // Download the file
            const writer = fs.createWriteStream(filepath);
            const response = await axios({
                url: mediaUrl,
                method: 'GET',
                responseType: 'stream'
            });
            
            response.data.pipe(writer);
            
            await new Promise((resolve, reject) => {
                writer.on('finish', resolve);
                writer.on('error', reject);
            });
            
            console.log(`💾 Saved WhatsApp media to: ${filepath}`);
            
            // Send to Python Service
            try {
                const formData = new FormData();
                const fileBuffer = fs.readFileSync(filepath);
                const blob = new Blob([fileBuffer]);
                formData.append('file', blob, filename);
                
                const predResponse = await axios.post(`${PYTHON_API_URL}/predict`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                });
                
                const result = predResponse.data;
                const prediction = result.predicted_class;
                const confidence = (result.confidence * 100).toFixed(1);
                
                const message = `🔬 *Analysis Result*\n\n` +
                              `Condition: *${prediction}*\n` +
                              `Confidence: *${confidence}%*\n\n` +
                              `_Disclaimer: This is for research purposes only._`;
                              
                twiml.message(message);
                
            } catch (predError) {
                console.error('Prediction error:', predError.message);
                twiml.message('⚠️ Sorry, I could not analyze that audio file. Please try again.');
            } finally {
                // Cleanup
                if (fs.existsSync(filepath)) fs.unlinkSync(filepath);
            }
            
        } else {
            twiml.message('👋 Welcome to Respiratory Sound Classifier!\n\nPlease send me an audio file (voice note or upload) to analyze.');
        }
        
    } catch (error) {
        console.error('WhatsApp Error:', error);
        twiml.message('❌ An error occurred while processing your request.');
    }
    
    res.type('text/xml').send(twiml.toString());
});

app.listen(PORT, () => {
    console.log(`✅ Backend server running on port ${PORT}`);
    console.log(`📁 Upload directory: ${path.join(__dirname, 'uploads')}`);
    console.log(`🔗 Python ML service expected at: ${PYTHON_API_URL}`);
    console.log(`💬 WhatsApp Webhook ready at: http://localhost:${PORT}/api/whatsapp`);
});
