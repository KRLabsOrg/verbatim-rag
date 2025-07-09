import React, { createContext, useContext, useState, useCallback } from 'react';

// Create context
const DocumentsContext = createContext();

// Custom hook to use the documents context
export const useDocuments = () => {
  const context = useContext(DocumentsContext);
  if (!context) {
    throw new Error('useDocuments must be used within a DocumentsProvider');
  }
  return context;
};

// Provider component with proper state management
export const DocumentsProvider = ({ children }) => {
  const [selectedDocId, setSelectedDocIdState] = useState(null);
  const [viewMode, setViewMode] = useState('split'); // 'split', 'chat-only', 'doc-only'
  
  // Enhanced document selection handler
  const setSelectedDocId = useCallback((docInfo) => {
    setSelectedDocIdState(docInfo);
  }, []);

  // Reset document selection
  const resetDocumentSelection = useCallback(() => {
    setSelectedDocIdState(null);
  }, []);

  // Get document by index helper
  const getDocumentByIndex = useCallback((documents, index) => {
    if (!documents || !Array.isArray(documents) || index < 0 || index >= documents.length) {
      return null;
    }
    return documents[index];
  }, []);

  // Navigate to next/previous document
  const navigateDocument = useCallback((documents, direction) => {
    if (!selectedDocId || !documents || !Array.isArray(documents)) return;
    
    const currentIndex = selectedDocId.docIndex;
    const newIndex = currentIndex + direction;
    
    if (newIndex >= 0 && newIndex < documents.length) {
      setSelectedDocId({ docIndex: newIndex });
    }
  }, [selectedDocId, setSelectedDocId]);

  // Check if document has highlights
  const hasHighlights = useCallback((document) => {
    return document && document.highlights && document.highlights.length > 0;
  }, []);

  // Get highlight count for document
  const getHighlightCount = useCallback((document) => {
    return document && document.highlights ? document.highlights.length : 0;
  }, []);

  const value = {
    // State
    selectedDocId,
    viewMode,
    
    // Actions
    setSelectedDocId,
    resetDocumentSelection,
    setViewMode,
    
    // Helpers
    getDocumentByIndex,
    navigateDocument,
    hasHighlights,
    getHighlightCount,
  };

  return (
    <DocumentsContext.Provider value={value}>
      {children}
    </DocumentsContext.Provider>
  );
}; 