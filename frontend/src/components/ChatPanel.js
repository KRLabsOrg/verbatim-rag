import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaComments, FaExclamationTriangle } from 'react-icons/fa';
import { useApi } from '../contexts/ApiContext';
import { useDocuments } from '../contexts/DocumentsContext';
import { Card, CardHeader, CardTitle, CardContent } from './ui/Card';
import { Spinner } from './ui/Spinner';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import RelevantSentences from './RelevantSentences';

const ChatPanel = () => {
  const { isLoading, isResourcesLoaded, currentQuery, submitQuery, error } = useApi();
  const { setSelectedDocId } = useDocuments();
  const [selectedHighlight, setSelectedHighlight] = useState(null);
  const messagesEndRef = useRef(null);
  const answerRef = useRef(null);
  
  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentQuery]);
  
  // Format the answer with citations when available
  useEffect(() => {
    if (!answerRef.current || !currentQuery) return;
    
    const answer = currentQuery.answer;
    
    // If we have structured answer with citations, use it
    if (currentQuery.structured_answer && currentQuery.structured_answer.citations) {
      const citations = currentQuery.structured_answer.citations;
      
      // If we have a selected highlight, highlight it in the answer
      if (selectedHighlight) {
        const highlightText = selectedHighlight.text;
        
        // Create HTML with the highlight
        let html = '';
        let lastIndex = 0;
        let startIndex = answer.indexOf(highlightText);
        
        while (startIndex !== -1) {
          // Add text before the match
          html += answer.substring(lastIndex, startIndex);
          
          // Add the highlighted match
          html += `<mark style="background-color: #fef3c7; padding: 0 2px; border-radius: 2px;">${highlightText}</mark>`;
          
          // Update indices
          lastIndex = startIndex + highlightText.length;
          startIndex = answer.indexOf(highlightText, lastIndex);
        }
        
        // Add remaining text
        html += answer.substring(lastIndex);
        
        // Set the HTML
        answerRef.current.innerHTML = html;
      } else {
        // No selected highlight, just show the answer with citation markers
        let html = answer;
        
        // Add citation markers
        citations.forEach((citation, index) => {
          const citationText = citation.text;
          const citationMarker = `<sup><a href="#" data-citation="${index}" style="color: #2563eb; text-decoration: none; font-weight: bold;">[${index + 1}]</a></sup>`;
          
          // Replace the citation text with the citation text + marker
          // Only replace the first occurrence to avoid duplicate markers
          const citationIndex = html.indexOf(citationText);
          if (citationIndex !== -1) {
            html = html.substring(0, citationIndex + citationText.length) + 
                   citationMarker + 
                   html.substring(citationIndex + citationText.length);
          }
        });
        
        // Set the HTML
        answerRef.current.innerHTML = html;
        
        // Add click event listeners to citation markers
        const citationLinks = answerRef.current.querySelectorAll('a[data-citation]');
        citationLinks.forEach(link => {
          link.addEventListener('click', (e) => {
            e.preventDefault();
            const citationIndex = parseInt(link.getAttribute('data-citation'));
            const citation = citations[citationIndex];
            
            // Find the highlight in the document
            if (citation && currentQuery.documents[citation.doc_index]) {
              const doc = currentQuery.documents[citation.doc_index];
              if (doc.highlights && doc.highlights[citation.highlight_index]) {
                const highlight = {
                  ...doc.highlights[citation.highlight_index],
                  docIndex: citation.doc_index
                };
                setSelectedHighlight(highlight);
                setSelectedDocId({ docIndex: citation.doc_index });
              }
            }
          });
        });
      }
    } else {
      // No structured answer, just show the plain text
      answerRef.current.textContent = answer;
    }
    
    return () => {
      if (answerRef.current) {
        answerRef.current.textContent = currentQuery?.answer || '';
      }
    };
  }, [currentQuery, selectedHighlight, setSelectedDocId]);
  
  const handleSubmit = async (question) => {
    if (!question.trim() || !isResourcesLoaded) return;
    
    setSelectedHighlight(null);
    await submitQuery(question);
  };

  // Extract all unique highlights from all documents
  const getAllHighlights = () => {
    if (!currentQuery || !currentQuery.documents) return [];
    
    const allHighlights = [];
    currentQuery.documents.forEach((doc, docIndex) => {
      if (doc.highlights && doc.highlights.length > 0) {
        doc.highlights.forEach(highlight => {
          // Add document index to each highlight for reference
          allHighlights.push({
            ...highlight,
            docIndex
          });
        });
      }
    });
    
    // Sort by document index and then by start position
    return allHighlights.sort((a, b) => {
      if (a.docIndex !== b.docIndex) return a.docIndex - b.docIndex;
      return a.start - b.start;
    });
  };
  
  // Handle highlight selection
  const handleHighlightClick = (highlight) => {
    setSelectedHighlight(highlight === selectedHighlight ? null : highlight);
    setSelectedDocId({ docIndex: highlight.docIndex });
  };
  
  return (
    <Card className="h-full flex flex-col">
      {/* Header */}
      <CardHeader className="border-b border-border">
        <CardTitle className="flex items-center space-x-2">
          <FaComments className="w-5 h-5 text-primary" />
          <span>Conversation</span>
        </CardTitle>
      </CardHeader>
      
      {/* Messages */}
      <CardContent className="flex-1 overflow-y-auto p-4">
        <AnimatePresence>
          {/* Error state */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center space-x-2"
            >
              <FaExclamationTriangle className="w-4 h-4 text-red-600" />
              <span className="text-sm text-red-700">{error}</span>
            </motion.div>
          )}

          {/* Empty state */}
          {!currentQuery && !isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center py-12"
            >
              <FaComments className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
              <p className="text-foreground text-lg font-medium mb-2">
                {isResourcesLoaded
                  ? "Ready to answer your questions"
                  : "Loading resources..."}
              </p>
              <p className="text-muted-foreground text-sm">
                {isResourcesLoaded
                  ? "Ask a question to get started with the RAG system"
                  : "Please wait while we initialize the system"}
              </p>
            </motion.div>
          )}

          {/* Loading state */}
          {isLoading && !currentQuery && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-12"
            >
              <Spinner className="mx-auto mb-4" />
              <p className="text-foreground">Processing your question...</p>
            </motion.div>
          )}

          {/* Conversation */}
          {currentQuery && (
            <div className="space-y-4">
              {/* User question */}
              <ChatMessage 
                message={currentQuery.question} 
                isUser={true}
              />
              
              {/* Assistant response */}
              {currentQuery.answer ? (
                <div className="space-y-4">
                  <ChatMessage 
                    message={currentQuery.answer} 
                    isUser={false}
                  />
                  
                  {/* Enhanced answer with citations */}
                  <div className="bg-card border border-border rounded-lg p-4">
                    <div ref={answerRef} className="prose prose-sm max-w-none text-foreground" />
                    
                    {/* Relevant sentences */}
                    <RelevantSentences
                      highlights={getAllHighlights()}
                      onHighlightClick={handleHighlightClick}
                      selectedHighlight={selectedHighlight}
                      currentQuery={currentQuery}
                    />
                  </div>
                </div>
              ) : (
                <ChatMessage 
                  message="" 
                  isUser={false} 
                  isLoading={true}
                />
              )}
            </div>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </CardContent>
      
      {/* Input */}
      <ChatInput
        onSubmit={handleSubmit}
        isLoading={isLoading}
        isDisabled={!isResourcesLoaded}
        placeholder={
          isResourcesLoaded
            ? "Ask a question about your documents..."
            : "Loading resources, please wait..."
        }
      />
    </Card>
  );
};

export default ChatPanel;