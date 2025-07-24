import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, FileText, Sparkles } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card.jsx';
import { Badge } from './ui/badge.jsx';
import { Button } from './ui/button.jsx';
import { Input } from './ui/input.jsx';
import { ScrollArea } from './ui/scroll-area.jsx';
import { TooltipProvider } from './ui/tooltip.jsx';
import { useApi } from '../contexts/ApiContext';

const ImprovedFactInterface = () => {
  const { isLoading, isResourcesLoaded, currentQuery, submitQuery } = useApi();
  const [question, setQuestion] = useState('');
  const [selectedDocument, setSelectedDocument] = useState(0);
  const [highlightedFactId, setHighlightedFactId] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || !isResourcesLoaded) return;
    
    await submitQuery(question);
    setQuestion('');
    setSelectedDocument(0); // Reset to first document
  };

  // Extract facts from the current query
  const extractFacts = () => {
    if (!currentQuery?.structured_answer?.citations) return [];
    
    return currentQuery.structured_answer.citations.map((citation, index) => ({
      id: index,
      text: citation.text,
      docIndex: citation.doc_index,
      highlightIndex: citation.highlight_index,
      document: currentQuery.documents[citation.doc_index]
    }));
  };

  const facts = extractFacts();

  // Handle fact click - navigate to document and highlight the fact
  const handleFactClick = (fact) => {
    setSelectedDocument(fact.docIndex);
    setHighlightedFactId(fact.id);
  };

  // Create clickable answer with fact links
  const renderAnswerWithClickableFacts = () => {
    if (!currentQuery?.answer || facts.length === 0) {
      return <p className="text-base leading-relaxed">{currentQuery?.answer}</p>;
    }

    let answerText = currentQuery.answer;
    const factLinks = [];

    // Sort facts by text length (longest first) to avoid partial replacements
    const sortedFacts = [...facts].sort((a, b) => b.text.length - a.text.length);

    sortedFacts.forEach((fact) => {
      const factText = fact.text;
      if (answerText.includes(factText)) {
        const placeholder = `__FACT_${fact.id}__`;
        answerText = answerText.replace(factText, placeholder);
        factLinks.push({ placeholder, fact });
      }
    });

    // Split the answer by fact placeholders and render
    const parts = answerText.split(/(__FACT_\d+__)/);
    
    return (
      <div className="text-lg leading-8">
        {parts.map((part, index) => {
          const factLink = factLinks.find(fl => fl.placeholder === part);
          if (factLink) {
            return (
              <motion.button
                key={index}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleFactClick(factLink.fact)}
                className="inline-flex items-center gap-2 px-3 py-1 mx-1 bg-blue-50 border-2 border-blue-300 rounded-md text-base text-blue-800 hover:bg-blue-100 hover:border-blue-400 hover:shadow-sm transition-all duration-200 cursor-pointer font-medium"
              >
                {factLink.fact.text}
                <Badge variant="secondary" className="ml-1 text-xs bg-blue-200 text-blue-700">
                  [{factLink.fact.id + 1}]
                </Badge>
              </motion.button>
            );
          }
          return <span key={index}>{part}</span>;
        })}
      </div>
    );
  };

  // Render document with highlighting
  const DocumentWithHighlight = ({ document, docIndex, highlightedFact }) => {
    if (!document) return null;

    const content = document.content;
    const highlights = document.highlights || [];
    
    // Create a map of all highlights
    const highlightMap = highlights.map((highlight, index) => {
      const factForHighlight = facts.find(f => 
        f.docIndex === docIndex && f.text === highlight.text
      );
      return {
        ...highlight,
        factId: factForHighlight?.id,
        isHighlighted: factForHighlight?.id === highlightedFact
      };
    });

    // Sort highlights by start position
    highlightMap.sort((a, b) => a.start - b.start);

    let lastIndex = 0;
    const parts = [];

    highlightMap.forEach((highlight, index) => {
      // Add text before highlight
      if (highlight.start > lastIndex) {
        parts.push({
          type: 'text',
          content: content.substring(lastIndex, highlight.start)
        });
      }

      // Add highlighted text
      parts.push({
        type: 'highlight',
        content: highlight.text,
        factId: highlight.factId,
        isHighlighted: highlight.isHighlighted
      });

      lastIndex = highlight.end;
    });

    // Add remaining text
    if (lastIndex < content.length) {
      parts.push({
        type: 'text',
        content: content.substring(lastIndex)
      });
    }

    return (
      <div className="text-base leading-7 whitespace-pre-wrap break-words text-gray-700">
        {parts.map((part, index) => {
          if (part.type === 'highlight') {
            return (
              <mark
                key={index}
                className={`px-2 py-1 rounded transition-all duration-300 ${
                  part.isHighlighted 
                    ? 'bg-yellow-200 border-l-4 border-yellow-500 shadow-md font-medium' 
                    : 'bg-yellow-100 border-b-2 border-yellow-400 hover:bg-yellow-200'
                }`}
              >
                {part.content}
              </mark>
            );
          }
          return <span key={index}>{part.content}</span>;
        })}
      </div>
    );
  };

  return (
    <TooltipProvider>
      <div className="h-screen flex flex-col bg-gray-50">
        {/* Header */}
        <div className="bg-indigo-700 text-white p-4 shadow-md flex-shrink-0">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-indigo-600 rounded-lg">
                <MessageCircle className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Verbatim RAG 
                </h1>
                <p className="text-indigo-200">Click facts to jump to source documents</p>
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Badge variant={isResourcesLoaded ? "default" : "secondary"} className="bg-indigo-600 text-white">
                {isResourcesLoaded ? 'Ready' : 'Loading...'}
              </Badge>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Panel - Chat Interface */}
          <div className="w-full md:w-2/3 bg-white p-6 flex flex-col border-r border-gray-200">
            <div className="space-y-6 flex-1 overflow-y-auto">
              {/* Query Input */}
              <div className="mb-6">
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="text-lg font-medium mb-3 block text-gray-700">Ask a question about your documents</label>
                    <div className="flex gap-3">
                      <Input
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder="What would you like to know?"
                        className="flex-1 p-4 text-base border-2 border-gray-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200"
                        disabled={!isResourcesLoaded || isLoading}
                      />
                      <Button 
                        type="submit" 
                        disabled={!question.trim() || !isResourcesLoaded || isLoading}
                        className="px-8 py-4 text-base bg-indigo-600 hover:bg-indigo-700"
                      >
                        {isLoading ? 'Thinking...' : 'Ask'}
                      </Button>
                    </div>
                    <div className="mt-2 text-sm text-gray-500 flex items-center">
                      <span>Answers are generated from your document collection with exact citations</span>
                    </div>
                  </div>
                </form>
              </div>

              {/* Answer Section */}
              <AnimatePresence>
                {currentQuery && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="space-y-6"
                  >
                    {/* Question */}
                    <div className="bg-blue-50 p-6 rounded-lg border-l-4 border-blue-400">
                      <div className="flex items-start gap-3">
                        <div className="p-2 bg-blue-100 rounded-lg">
                          <MessageCircle className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-sm text-blue-600 font-medium mb-1">Your Question</p>
                          <p className="text-lg font-medium text-gray-800">{currentQuery.question}</p>
                        </div>
                      </div>
                    </div>

                    {/* Answer with Interactive Facts */}
                    {currentQuery.answer && (
                      <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
                        <div className="p-6 border-b border-gray-200">
                          <div className="flex items-center gap-2">
                            <Sparkles className="w-6 h-6 text-indigo-600" />
                            <h3 className="text-xl font-semibold text-gray-800">Answer</h3>
                            {facts.length > 0 && (
                              <Badge variant="secondary" className="ml-2">
                                {facts.length} fact{facts.length !== 1 ? 's' : ''} • Click to view source
                              </Badge>
                            )}
                          </div>
                        </div>
                        <div className="p-6">
                          <div className="prose prose-lg max-w-none text-gray-700 leading-relaxed">
                            {renderAnswerWithClickableFacts()}
                          </div>
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}

                {/* Empty State */}
                {!currentQuery && !isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center py-16"
                  >
                    <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
                      <MessageCircle className="w-10 h-10 text-blue-600" />
                    </div>
                    <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                      {isResourcesLoaded ? 'Ready to answer your questions' : 'Loading...'}
                    </h3>
                    <p className="text-gray-600 text-lg max-w-lg mx-auto">
                      {isResourcesLoaded 
                        ? 'Ask a question and click on facts in the answer to see their exact source context.'
                        : 'Please wait while we initialize the system.'}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Right Panel - Documents */}
          <div className="hidden md:block w-1/3 bg-gray-100 p-6 overflow-hidden flex flex-col">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                Source Documents
                {currentQuery?.documents && (
                  <Badge variant="outline" className="ml-2">
                    {currentQuery.documents.length} docs
                  </Badge>
                )}
              </h2>
            </div>

            <div className="bg-white rounded-lg shadow-sm flex-1 overflow-hidden flex flex-col">
              {currentQuery?.documents ? (
                <>
                  {/* Document Tabs */}
                  {currentQuery.documents.length > 1 && (
                    <div className="border-b px-4 py-2 flex-shrink-0">
                      <div className="flex gap-2 overflow-x-auto">
                        {currentQuery.documents.map((_, index) => (
                          <Button
                            key={index}
                            variant={selectedDocument === index ? "default" : "outline"}
                            size="sm"
                            onClick={() => setSelectedDocument(index)}
                            className="whitespace-nowrap"
                          >
                            Document {index + 1}
                            {currentQuery.documents[index].highlights?.length > 0 && (
                              <Badge variant="secondary" className="ml-1">
                                {currentQuery.documents[index].highlights.length}
                              </Badge>
                            )}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Document Content */}
                  <div className="flex-1 overflow-y-auto p-4">
                    {currentQuery.documents[selectedDocument] ? (
                      <DocumentWithHighlight
                        document={currentQuery.documents[selectedDocument]}
                        docIndex={selectedDocument}
                        highlightedFact={highlightedFactId}
                      />
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>No document selected</p>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="h-full flex items-center justify-center text-center p-8">
                  <div>
                    <FileText className="w-16 h-16 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <h3 className="text-lg font-medium mb-2">No Documents Yet</h3>
                    <p className="text-muted-foreground">
                      Ask a question to see relevant source documents
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
};

export default ImprovedFactInterface;