import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaHighlighter, FaInfoCircle, FaEye, FaEyeSlash } from 'react-icons/fa';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';
import { Card, CardContent } from './ui/Card';
import { Spinner } from './ui/Spinner';
import { cn } from '../utils/cn';

const DocumentViewer = ({ document: docData, selectedHighlightIndex, isLoading }) => {
  const contentRef = useRef(null);
  const highlightRefs = useRef([]);
  const [showFullContent, setShowFullContent] = useState(false);
  
  // Apply highlights to the document content
  useEffect(() => {
    if (!contentRef.current || !docData) return;
    
    const content = docData.content;
    
    // If there are no highlights, just show the plain content
    if (!docData.highlights || docData.highlights.length === 0) {
      contentRef.current.textContent = content;
      return;
    }
    
    const highlights = [...docData.highlights].sort((a, b) => a.start - b.start);
    
    // Create a new HTML content with highlights
    let html = '';
    let lastIndex = 0;
    
    // Reset highlight refs array
    highlightRefs.current = new Array(highlights.length);
    
    highlights.forEach((highlight, index) => {
      // Add text before the highlight
      html += content.substring(lastIndex, highlight.start);
      
      // Add the highlighted text with a data attribute for the index
      const isSelected = index === selectedHighlightIndex;
      const className = isSelected ? 'bg-blue-100 border-l-2 border-blue-400 shadow-sm' : 'bg-blue-50 border-b border-blue-300';
      
      html += `<mark 
        id="highlight-${index}" 
        class="highlight ${isSelected ? 'selected' : ''} ${className} px-1 py-0.5 rounded-sm transition-all duration-200" 
        data-index="${index}"
      >${highlight.text}</mark>`;
      
      // Update the last index
      lastIndex = highlight.end;
    });
    
    // Add any remaining text
    html += content.substring(lastIndex);
    
    // Set the HTML content
    contentRef.current.innerHTML = html;
    
    // Scroll to selected highlight if specified
    if (selectedHighlightIndex !== undefined && selectedHighlightIndex !== null) {
      setTimeout(() => {
        const selectedElement = document.getElementById(`highlight-${selectedHighlightIndex}`);
        if (selectedElement) {
          selectedElement.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
          setShowFullContent(true); // Auto-expand when a highlight is selected
        }
      }, 100);
    }
  }, [docData, selectedHighlightIndex]);
  
  if (!docData) {
    return (
      <div className="flex items-center justify-center h-64 text-secondary-500">
        <FaInfoCircle className="w-8 h-8 mr-2" />
        <span>No document selected</span>
      </div>
    );
  }
  
  const hasHighlights = docData.highlights && docData.highlights.length > 0;
  
  return (
    <div className="space-y-4">
      {/* Loading indicator */}
      {isLoading && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-end"
        >
          <Badge variant="warning" className="flex items-center space-x-2">
            <Spinner size="sm" />
            <span>Loading highlights...</span>
          </Badge>
        </motion.div>
      )}
      
      {/* No highlights alert */}
      {!hasHighlights && !isLoading && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="p-4 bg-blue-50 border border-blue-200 rounded-lg"
        >
          <div className="flex items-start space-x-3">
            <FaInfoCircle className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-blue-800">No highlights found</h4>
              <p className="text-xs text-blue-600 mt-1">
                This document doesn't contain any passages relevant to the current query.
              </p>
            </div>
          </div>
        </motion.div>
      )}
      
      {/* Document summary card */}
      <Card className={cn(
        "transition-all duration-200",
        hasHighlights ? "border-green-200 bg-green-50/50" : "border-secondary-200 bg-secondary-50/50"
      )}>
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <FaHighlighter className={cn(
                "w-4 h-4",
                hasHighlights ? "text-green-600" : "text-secondary-400"
              )} />
              <h3 className={cn(
                "text-sm font-semibold",
                hasHighlights ? "text-green-800" : "text-secondary-700"
              )}>
                Document Content
              </h3>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowFullContent(!showFullContent)}
              className="flex items-center space-x-2"
            >
              {showFullContent ? <FaEyeSlash className="w-3 h-3" /> : <FaEye className="w-3 h-3" />}
              <span className="text-xs">
                {showFullContent ? "Hide" : "Show"} Full Content
              </span>
            </Button>
          </div>
          
          {/* Document preview */}
          <div className="text-sm text-secondary-700 mb-3 font-mono leading-relaxed break-words">
            {docData.content.length > 300 
              ? docData.content.substring(0, 300) + '...' 
              : docData.content}
          </div>
          
          {/* Highlight badges */}
          {hasHighlights && (
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="text-xs font-medium text-green-700">
                  Relevant Passages ({docData.highlights.length}):
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {docData.highlights.map((highlight, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: idx * 0.05 }}
                  >
                    <Badge 
                      variant="warning" 
                      className="cursor-pointer hover:bg-yellow-200 transition-colors text-xs max-w-xs truncate"
                      onClick={() => {
                        setShowFullContent(true);
                        setTimeout(() => {
                          const selectedElement = document.getElementById(`highlight-${idx}`);
                          if (selectedElement) {
                            selectedElement.scrollIntoView({ 
                              behavior: 'smooth', 
                              block: 'center' 
                            });
                          }
                        }, 100);
                      }}
                    >
                      {highlight.text.length > 50 
                        ? highlight.text.substring(0, 50) + '...' 
                        : highlight.text}
                    </Badge>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
      
      {/* Full document content */}
      <AnimatePresence>
        {showFullContent && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <Card className={cn(
              "transition-all duration-200",
              isLoading ? "border-blue-200" : hasHighlights ? "border-green-200" : "border-secondary-200"
            )}>
              <CardContent className="p-0">
                <div
                  ref={contentRef}
                  className={cn(
                    "whitespace-pre-wrap text-sm font-mono leading-relaxed p-4 break-words",
                    "max-h-[50vh] sm:max-h-[60vh] md:max-h-96 overflow-y-auto scrollbar-webkit",
                    "text-secondary-900"
                  )}
                />
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
      
      {/* Loading indicator for highlights */}
      {docData.highlights && docData.highlights.length === 0 && isLoading && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex justify-center py-4"
        >
          <Badge variant="warning" className="flex items-center space-x-2">
            <Spinner size="sm" />
            <span>Searching for relevant passages...</span>
          </Badge>
        </motion.div>
      )}
    </div>
  );
};

export default DocumentViewer;