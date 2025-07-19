import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const vintageOpticsAPI = {
  // Process an image with specified parameters
  processImage: async (file, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const params = new URLSearchParams({
      lens_profile: options.lensProfile || 'canon-50mm-f1.4',
      correction_mode: options.correctionMode || 'hybrid',
      ...Object.entries(options.defects || {}).reduce((acc, [key, value]) => {
        if (value) acc[`defect_${key}`] = 'true';
        return acc;
      }, {})
    });
    
    const response = await api.post(`/api/process?${params.toString()}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob',
    });
    
    return response;
  },

  // Get available lens profiles
  getLensProfiles: async () => {
    const response = await api.get('/api/lens-profiles');
    return response.data;
  },

  // Get processing statistics for a job
  getProcessingStats: async (jobId) => {
    const response = await api.get(`/api/stats/${jobId}`);
    return response.data;
  },

  // Synthesize a new lens profile
  synthesizeLens: async (parameters) => {
    const response = await api.post('/api/synthesize', parameters);
    return response.data;
  },

  // Get synthesis presets
  getSynthesisPresets: async () => {
    const response = await api.get('/api/synthesis-presets');
    return response.data;
  },
};

export default vintageOpticsAPI;
