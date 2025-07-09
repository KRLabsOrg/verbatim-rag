import React, { useState } from 'react';
import { FaChevronLeft, FaSearch, FaChevronRight, FaFileAlt, FaSpinner, FaFolder } from 'react-icons/fa';
import { motion, AnimatePresence } from 'framer-motion';
import { useApi } from '../contexts/ApiContext';
import { useDocuments } from '../contexts/DocumentsContext';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { Badge } from './ui/Badge';
import { Spinner } from './ui/Spinner';
import DocumentViewer from './DocumentViewer';
import DocumentSelector from './DocumentSelector';

const DocumentPanel = ({ onBack }) => {
  const { isLoading, currentQuery } = useApi();
  const { selectedDocId, setSelectedDocId } = useDocuments();
  const [searchText, setSearchText] = useState('');
  
  // Determine if we're on mobile based on whether onBack is provided
  const isMobile = !!onBack;
  
  // Get the current document based on selectedDocId
  const document = currentQuery?.documents && selectedDocId?.docIndex !== undefined
    ? currentQuery.documents[selectedDocId.docIndex]
    : null;
  
  // Filter highlights if search text is provided
  const filteredDocument = document && searchText
    ? {
        ...document,
        highlights: document.highlights?.filter(h => 
          h.text.toLowerCase().includes(searchText.toLowerCase())
        ) || []
      }
    : document;

  const handleDocumentSelect = (docIndex) => {
    setSelectedDocId({ docIndex });
  };

  const navigateDocument = (direction) => {
    if (!selectedDocId || !currentQuery?.documents) return;
    
    const newIndex = selectedDocId.docIndex + direction;
    if (newIndex >= 0 && newIndex < currentQuery.documents.length) {
      setSelectedDocId({ docIndex: newIndex });
    }
  };
  
  return (
    <Card className="h-full flex flex-col">
      {/* Header */}
      <CardHeader className="border-b border-secondary-200">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            {isMobile && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onBack}
                className="mr-2"
              >
                <FaChevronLeft className="w-4 h-4" />
              </Button>
            )}
            <FaFileAlt className="w-5 h-5 text-primary-600" />
            <span>Source Documents</span>
          </CardTitle>
          
          {/* Document navigation */}
          {currentQuery?.documents && currentQuery.documents.length > 0 && selectedDocId && (
            <div className="flex items-center space-x-2">
              <span className="text-sm text-secondary-600">
                {selectedDocId.docIndex + 1} of {currentQuery.documents.length}
              </span>
              <div className="flex space-x-1">
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={selectedDocId.docIndex === 0}
                  onClick={() => navigateDocument(-1)}
                  className="h-8 w-8"
                >
                  <FaChevronLeft className="w-3 h-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={selectedDocId.docIndex === currentQuery.documents.length - 1}
                  onClick={() => navigateDocument(1)}
                  className="h-8 w-8"
                >
                  <FaChevronRight className="w-3 h-3" />
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardHeader>
      
      {/* Document selector */}
      {currentQuery?.documents && currentQuery.documents.length > 0 && (
        <DocumentSelector
          documents={currentQuery.documents}
          selectedDocIndex={selectedDocId?.docIndex}
          onDocumentSelect={handleDocumentSelect}
        />
      )}
      
      {/* Search bar - only show if we have a document */}
      {document && (
        <div className="border-b border-secondary-200 p-4 bg-white">
          <div className="relative">
            <FaSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-secondary-400" />
            <Input
              placeholder="Search in highlights..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              className="pl-10"
            />
          </div>
          {searchText && (
            <div className="flex items-center mt-2 space-x-2">
              <span className="text-xs text-secondary-500">Found:</span>
              <Badge variant={filteredDocument?.highlights?.length > 0 ? "success" : "danger"}>
                {filteredDocument?.highlights?.length || 0} highlight{filteredDocument?.highlights?.length !== 1 ? 's' : ''}
              </Badge>
            </div>
          )}
        </div>
      )}
      
      {/* Content */}
      <CardContent className="flex-1 overflow-y-auto p-4">
        <AnimatePresence mode="wait">
          {/* Initial loading state */}
          {isLoading && !currentQuery && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-64 space-y-4"
            >
              <Spinner size="lg" />
              <p className="text-secondary-600">Loading documents...</p>
            </motion.div>
          )}

          {/* Documents loading state */}
          {isLoading && currentQuery && !selectedDocId && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-64 space-y-4 text-center"
            >
              <div className="flex items-center space-x-2">
                <FaSpinner className="w-6 h-6 animate-spin text-primary-600" />
                <span className="text-lg font-medium text-secondary-700">Loading documents...</span>
              </div>
              {currentQuery.documents && currentQuery.documents.length > 0 && (
                <Badge variant="default" className="mt-2">
                  Found {currentQuery.documents.length} document{currentQuery.documents.length !== 1 ? 's' : ''}
                </Badge>
              )}
            </motion.div>
          )}

          {/* No query state */}
          {!currentQuery && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center justify-center h-64 space-y-4 text-center"
            >
              <FaFolder className="w-16 h-16 text-secondary-300" />
              <div>
                <h3 className="text-lg font-medium text-secondary-700 mb-2">No Documents Yet</h3>
                <p className="text-secondary-500">
                  Ask a question to see relevant source documents
                </p>
              </div>
            </motion.div>
          )}

          {/* No document selected state */}
          {currentQuery && !selectedDocId && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="flex flex-col items-center justify-center h-64 space-y-6"
            >
              <div className="text-center max-w-md">
                <div className="w-16 h-16 mx-auto mb-4 bg-primary-100 rounded-full flex items-center justify-center">
                  <FaFileAlt className="w-8 h-8 text-primary-600" />
                </div>
                <h3 className="text-lg font-semibold text-secondary-800 mb-2">
                  Source Documents Available
                </h3>
                <p className="text-secondary-600 text-sm mb-6">
                  {currentQuery.documents && currentQuery.documents.length > 0 
                    ? `${currentQuery.documents.length} documents were retrieved for your query. Select a document above to view its content.`
                    : "No documents were found for this query."}
                </p>
                
                {/* Quick access buttons */}
                {currentQuery.documents && currentQuery.documents.length > 0 && (
                  <div className="flex flex-wrap gap-2 justify-center">
                    {currentQuery.documents.slice(0, 3).map((doc, idx) => {
                      const hasHighlights = doc.highlights && doc.highlights.length > 0;
                      return (
                        <Button
                          key={idx}
                          variant="outline"
                          size="sm"
                          onClick={() => handleDocumentSelect(idx)}
                          className="flex items-center space-x-2"
                        >
                          <div className={`w-2 h-2 rounded-full ${hasHighlights ? 'bg-green-500' : 'bg-secondary-400'}`} />
                          <span>Doc {idx + 1}</span>
                          {hasHighlights && (
                            <Badge variant="success" className="ml-1">
                              {doc.highlights.length}
                            </Badge>
                          )}
                        </Button>
                      );
                    })}
                    {currentQuery.documents.length > 3 && (
                      <span className="text-xs text-secondary-500 mt-2">
                        + {currentQuery.documents.length - 3} more
                      </span>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}

          {/* Document viewer */}
          {document && (
            <motion.div
              key={selectedDocId?.docIndex}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              <DocumentViewer 
                document={filteredDocument} 
                selectedHighlightIndex={selectedDocId?.highlightIndex}
                isLoading={isLoading}
              />
            </motion.div>
          )}

          {/* Document not found state */}
          {selectedDocId && !document && !isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center justify-center h-64 text-secondary-500"
            >
              <div className="text-center">
                <FaFileAlt className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>Document not found</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
};

export default DocumentPanel; 