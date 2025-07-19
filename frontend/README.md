# VintageOptics GUI

A modern React-based graphical user interface for the VintageOptics lens characterization and synthesis system.

## Features

- **Image Processing**: Upload and process images with vintage lens effects
- **Lens Profiles**: Select from pre-configured vintage lens profiles
- **Defect Simulation**: Add authentic vintage defects like dust, fungus, and scratches
- **Correction Modes**: Choose between physical, ML, or hybrid correction
- **Real-time Preview**: See before/after comparisons
- **Processing Statistics**: View quality scores and processing metrics

## Setup

1. Install Node.js (version 14 or higher)

2. Run the setup script:
   ```bash
   ./setup_frontend.sh
   ```

3. Or manually install dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Running the Application

### Option 1: Run Everything (Recommended)

Use the integrated run script to start both backend and frontend:

```bash
./run_with_gui.sh
```

This will:
- Start the backend API on port 8000
- Start the frontend on port 3000
- Open your browser automatically

### Option 2: Run Separately

1. Start the backend API:
   ```bash
   python frontend_api.py
   ```

2. In a new terminal, start the frontend:
   ```bash
   cd frontend
   npm start
   ```

## Usage

1. **Upload an Image**: Click the upload area or drag and drop an image
2. **Select Lens Profile**: Choose from vintage lens profiles like Canon FD, Helios, etc.
3. **Add Defects** (Optional): Toggle vintage defects to simulate
4. **Choose Correction Mode**:
   - **No Correction**: Pure vintage simulation
   - **Physical Model**: Physics-based correction
   - **ML Enhancement**: Machine learning correction
   - **Hybrid**: Combined approach (recommended)
5. **Process**: Click the Process button
6. **Download**: Save the processed image

## Architecture

The GUI communicates with the VintageOptics backend through a REST API:

```
Frontend (React) <---> Backend API (FastAPI) <---> VintageOptics Core
```

### Technologies Used

- **Frontend**: React, Tailwind CSS, Lucide Icons
- **API Communication**: Axios
- **Backend**: FastAPI, OpenCV, NumPy
- **Image Processing**: VintageOptics core libraries

## Development

### Frontend Structure

```
frontend/
├── public/          # Static assets
├── src/
│   ├── components/  # React components
│   ├── services/    # API services
│   └── index.js     # Entry point
└── package.json
```

### Adding New Features

1. **New Lens Profile**: Add to `LENS_PROFILES` in `frontend_api.py`
2. **New Defect Type**: Update defect handling in both frontend and backend
3. **New Processing Mode**: Extend the correction modes in the API

## Troubleshooting

### Port Already in Use

If ports 3000 or 8000 are already in use:

```bash
# Kill processes on specific ports
lsof -ti:3000 | xargs kill -9
lsof -ti:8000 | xargs kill -9
```

### Backend Connection Issues

If the frontend can't connect to the backend:

1. Check the backend is running: `curl http://localhost:8000`
2. Verify CORS settings in `frontend_api.py`
3. Check browser console for error messages

### Image Processing Errors

- Ensure image files are in supported formats (JPEG, PNG)
- Check file size (recommended under 10MB)
- Verify Python dependencies are installed

## Performance Tips

- Process images at reasonable resolutions (2000-4000px wide)
- Use JPEG format for faster processing
- Enable GPU acceleration if available

## Contributing

See the main VintageOptics [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details
