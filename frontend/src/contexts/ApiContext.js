import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react';
import axios from 'axios';

// Create context
const ApiContext = createContext();

// Custom hook to use the API context
export const useApi = () => useContext(ApiContext);

// Provider component
export const ApiProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isResourcesLoaded, setIsResourcesLoaded] = useState(false);
  const [currentQuery, setCurrentQuery] = useState(null);
  const [numDocs, setNumDocs] = useState(5); // Default to 5 documents

  // Add a ref to track document updates
  const documentUpdateTimeoutRef = useRef(null);
  
  // Function to check if resources are loaded
  const refreshStatus = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await axios.get('/api/status');
      setIsResourcesLoaded(response.data.resources_loaded);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Check if resources are loaded on mount
  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  // Submit a query
  const submitQuery = useCallback(async (question) => {
    if (!isResourcesLoaded) {
      setError('Resources not loaded. Please wait for the system to initialize.');
      return null;
    }
    
    setIsLoading(true);
    setError(null);
    
    // Create an initial query result structure
    setCurrentQuery({
      question,
      documents: [],
      answer: null,
      structured_answer: null
    });
    
    try {
      // Use fetch for streaming instead of axios
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache',
          'X-Accel-Buffering': 'no'
        },
        body: JSON.stringify({
          question,
          num_docs: numDocs
        })
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      const reader = response.body.getReader();
      let buffer = '';
      const decoder = new TextDecoder();
      
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });
        
        // Split by newlines but keep the last potentially incomplete line
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep last line in buffer if incomplete
        
        // Process only complete lines
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            let data;
            
            // Handle both normal and double-escaped JSON (for Fly.io compatibility)
            try {
              // First try normal JSON parsing (works locally)
              data = JSON.parse(line);
            } catch (e) {
              // If that fails, check if it's double-escaped (Fly.io issue)
              if (line.startsWith('"') && line.endsWith('"')) {
                // Remove outer quotes and parse twice
                const unescaped = JSON.parse(line); // First parse removes escaping
                data = JSON.parse(unescaped);       // Second parse gets actual object
              } else {
                throw e; // Re-throw if neither format works
              }
            }
            
            if (data.error) {
              setError(data.error);
              continue;
            }
            
            switch (data.type) {
              case 'documents':
                // Update documents incrementally
                setCurrentQuery(prev => ({
                  ...prev,
                  documents: data.data
                }));
                break;
                
              case 'highlights':
                // Update documents with highlights
                setCurrentQuery(prev => {
                  // Create a map of document content to highlights
                  const highlightMap = {};
                  data.data.forEach(doc => {
                    highlightMap[doc.content] = doc.highlights || [];
                  });
                  
                  // Update each document with its highlights
                  const updatedDocs = prev.documents.map(doc => ({
                    ...doc,
                    highlights: highlightMap[doc.content] || []
                  }));
                  
                  return {
                    ...prev,
                    documents: updatedDocs
                  };
                });
                break;
                
              case 'answer':
                // Update with final answer
                setCurrentQuery(prev => ({
                  ...prev,
                  answer: data.data.answer,
                  structured_answer: data.data.structured_answer
                }));
                
                if (data.done) {
                  setIsLoading(false);
                }
                break;
                
              default:
                console.warn('Unknown response type:', data.type);
            }
          } catch (parseError) {
            console.error('Error parsing response:', parseError);
            console.error('Raw line:', line);
          }
        }
      }
      
      // Process any remaining buffer content
      if (buffer.trim()) {
        try {
          let data;
          
          // Handle both normal and double-escaped JSON (for Fly.io compatibility)
          try {
            data = JSON.parse(buffer);
          } catch (e) {
            if (buffer.startsWith('"') && buffer.endsWith('"')) {
              const unescaped = JSON.parse(buffer);
              data = JSON.parse(unescaped);
            } else {
              throw e;
            }
          }
          
          // Process final data if needed
          if (data.type === 'answer' && data.done) {
            setIsLoading(false);
          }
        } catch (parseError) {
          console.error('Error parsing final buffer:', parseError);
          console.error('Raw buffer:', buffer);
        }
      }
      
      return currentQuery;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [isResourcesLoaded, numDocs]);

  // Load resources with optional API key
  const loadResources = useCallback(async (apiKey = null) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const payload = apiKey ? { api_key: apiKey } : {};
      const response = await axios.post('/api/load-resources', payload);
      
      setIsResourcesLoaded(response.data.success);
      
      if (!response.data.success) {
        setError(response.data.message);
      }
      
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message;
      setError(errorMessage);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Update number of documents to retrieve - keep this for internal use
  const updateNumDocs = useCallback((num) => {
    setNumDocs(num);
  }, []);

  // Reset current query
  const resetQuery = useCallback(() => {
    setCurrentQuery(null);
  }, []);

  // Force UI updates when documents change
  useEffect(() => {
    if (currentQuery && currentQuery.documents && currentQuery.documents.length > 0 && isLoading) {
      // Clear any existing timeout
      if (documentUpdateTimeoutRef.current) {
        clearTimeout(documentUpdateTimeoutRef.current);
      }
      
      // Force a UI update after a short delay
      documentUpdateTimeoutRef.current = setTimeout(() => {
        // This is just to trigger a re-render
        setCurrentQuery(prev => ({...prev}));
      }, 50);
    }
    
    // Cleanup on unmount
    return () => {
      if (documentUpdateTimeoutRef.current) {
        clearTimeout(documentUpdateTimeoutRef.current);
      }
    };
  }, [currentQuery, isLoading]);

  // Value object
  const value = {
    isLoading,
    error,
    isResourcesLoaded,
    currentQuery,
    numDocs,
    submitQuery,
    loadResources,
    updateNumDocs,
    resetQuery,
    refreshStatus,
  };

  return <ApiContext.Provider value={value}>{children}</ApiContext.Provider>;
};

export default ApiContext; 