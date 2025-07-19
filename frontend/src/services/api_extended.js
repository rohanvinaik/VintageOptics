// Add to src/services/api.js

export const vintageOpticsAPI = {
  // ... existing methods ...
  
  // Analyze image for defects and characteristics
  analyzeImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await api.post('/api/analyze', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  },
};
