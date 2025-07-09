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
      <div className="text-base leading-relaxed">
        {parts.map((part, index) => {
          const factLink = factLinks.find(fl => fl.placeholder === part);
          if (factLink) {
            return (
              <motion.button
                key={index}
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                onClick={() => handleFactClick(factLink.fact)}
                className="inline-flex items-center gap-1 px-2 py-1 mx-0.5 bg-blue-50 border-b border-blue-300 rounded-sm text-sm text-blue-800 hover:bg-blue-100 hover:border-blue-400 transition-all duration-150 cursor-pointer"
              >
                {factLink.fact.text}
                <Badge variant="secondary" className="ml-1 text-xs opacity-60">
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
      <div className="text-sm leading-relaxed whitespace-pre-wrap break-words">
        {parts.map((part, index) => {
          if (part.type === 'highlight') {
            return (
              <mark
                key={index}
                className={`px-1 py-0.5 rounded-sm transition-all duration-300 ${
                  part.isHighlighted 
                    ? 'bg-blue-100 border-l-2 border-blue-400 shadow-sm' 
                    : 'bg-blue-50 border-b border-blue-300 hover:bg-blue-100'
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
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
        {/* Header */}
        <div className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-10">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg">
                  <MessageCircle className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    Verbatim RAG
                  </h1>
                  <p className="text-sm text-muted-foreground">Click facts to jump to source</p>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <Badge variant={isResourcesLoaded ? "default" : "secondary"}>
                  {isResourcesLoaded ? 'Ready' : 'Loading...'}
                </Badge>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-4 sm:py-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-8 max-w-7xl mx-auto">
            {/* Left Panel - Query & Answer */}
            <div className="space-y-6">
              {/* Query Input */}
              <Card>
                <CardContent className="p-6">
                  <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                      <label className="text-sm font-medium mb-2 block">Ask a question about your documents</label>
                      <div className="flex gap-3">
                        <Input
                          value={question}
                          onChange={(e) => setQuestion(e.target.value)}
                          placeholder="What would you like to know?"
                          className="flex-1"
                          disabled={!isResourcesLoaded || isLoading}
                        />
                        <Button 
                          type="submit" 
                          disabled={!question.trim() || !isResourcesLoaded || isLoading}
                          className="px-6"
                        >
                          {isLoading ? 'Thinking...' : 'Ask'}
                        </Button>
                      </div>
                    </div>
                  </form>
                </CardContent>
              </Card>

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
                    <Card>
                      <CardContent className="p-6">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-blue-100 rounded-lg">
                            <MessageCircle className="w-4 h-4 text-blue-600" />
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground mb-1">Your Question</p>
                            <p className="font-medium">{currentQuery.question}</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Answer with Interactive Facts */}
                    {currentQuery.answer && (
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center gap-2">
                            <Sparkles className="w-5 h-5 text-indigo-600" />
                            Answer
                            {facts.length > 0 && (
                              <Badge variant="secondary" className="ml-2">
                                {facts.length} fact{facts.length !== 1 ? 's' : ''} • Click to jump to source
                              </Badge>
                            )}
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="prose prose-sm max-w-none">
                            {renderAnswerWithClickableFacts()}
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </motion.div>
                )}

                {/* Empty State */}
                {!currentQuery && !isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center py-12"
                  >
                    <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
                      <MessageCircle className="w-8 h-8 text-blue-600" />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">
                      {isResourcesLoaded ? 'Ready to answer your questions' : 'Loading...'}
                    </h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      {isResourcesLoaded 
                        ? 'Ask a question and click on facts in the answer to see their source context.'
                        : 'Please wait while we initialize the system.'}
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Right Panel - Documents */}
            <div className="space-y-6">
              <Card className="h-[calc(100vh-8rem)] max-h-[calc(100vh-8rem)] overflow-hidden">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5 text-blue-600" />
                    Source Documents
                    {currentQuery?.documents && (
                      <Badge variant="outline" className="ml-2">
                        {currentQuery.documents.length} docs
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0 flex flex-col h-full">
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
                      <ScrollArea className="flex-1 px-4">
                        <div className="py-4">
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
                      </ScrollArea>
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
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
};

export default ImprovedFactInterface;