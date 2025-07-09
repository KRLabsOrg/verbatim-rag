import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, FileText, Sparkles, ChevronRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card.jsx';
import { Badge } from './ui/badge.jsx';
import { Button } from './ui/button.jsx';
import { Input } from './ui/input.jsx';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog.jsx';
import { ScrollArea } from './ui/scroll-area.jsx';
import { TooltipProvider } from './ui/tooltip.jsx';
import { useApi } from '../contexts/ApiContext';

const FactClickInterface = () => {
  const { isLoading, isResourcesLoaded, currentQuery, submitQuery } = useApi();
  const [question, setQuestion] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || !isResourcesLoaded) return;
    
    await submitQuery(question);
    setQuestion('');
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

  const FactPill = ({ fact, index }) => (
    <Dialog>
      <DialogTrigger asChild>
        <motion.button
          whileHover={{ scale: 1.01 }}
          whileTap={{ scale: 0.99 }}
          className="inline-flex items-center gap-1 px-3 py-1 bg-blue-50 border-b border-blue-300 rounded-sm text-sm text-blue-800 hover:bg-blue-100 hover:border-blue-400 transition-all duration-150 cursor-pointer group"
          onClick={() => {}}
        >
          <span className="max-w-[200px] truncate">{fact.text}</span>
          <Badge variant="secondary" className="ml-1 text-xs opacity-60">
            [{index + 1}]
          </Badge>
          <ChevronRight className="w-3 h-3 text-blue-400 group-hover:text-blue-600 opacity-60" />
        </motion.button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[80vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-blue-600" />
            Source Context - Fact [{index + 1}]
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {/* Fact highlight */}
          <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-600 font-medium mb-1">Selected Fact:</p>
            <p className="text-blue-900 font-medium">{fact.text}</p>
          </div>
          
          {/* Document context */}
          <div className="border rounded-lg overflow-hidden">
            <div className="bg-muted px-4 py-2 border-b">
              <p className="text-sm font-medium">Document {fact.docIndex + 1}</p>
            </div>
            <ScrollArea className="h-96">
              <div className="p-4">
                <DocumentWithHighlight 
                  content={fact.document.content}
                  highlightText={fact.text}
                />
              </div>
            </ScrollArea>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );

  const DocumentWithHighlight = ({ content, highlightText }) => {
    const parts = content.split(new RegExp(`(${highlightText.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')})`, 'gi'));
    
    return (
      <div className="text-sm leading-relaxed">
        {parts.map((part, index) => 
          part.toLowerCase() === highlightText.toLowerCase() ? (
            <mark key={index} className="bg-yellow-200 px-1 py-0.5 rounded font-medium">
              {part}
            </mark>
          ) : (
            <span key={index}>{part}</span>
          )
        )}
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
                  <p className="text-sm text-muted-foreground">Click facts to see source context</p>
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
        <div className="container mx-auto px-4 py-8 max-w-4xl">
          {/* Query Input */}
          <Card className="mb-8">
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
                            {facts.length} fact{facts.length !== 1 ? 's' : ''} • Click to explore
                          </Badge>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="prose prose-sm max-w-none mb-6">
                        <p className="text-base leading-relaxed">{currentQuery.answer}</p>
                      </div>
                      
                      {/* Interactive Facts */}
                      {facts.length > 0 && (
                        <div className="space-y-4">
                          <div className="border-t pt-4">
                            <p className="text-sm font-medium text-muted-foreground mb-3">
                              💡 Click any fact below to see it in context:
                            </p>
                            <div className="flex flex-wrap gap-2">
                              {facts.map((fact, index) => (
                                <FactPill key={fact.id} fact={fact} index={index} />
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
              </motion.div>
            )}
          </AnimatePresence>

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
                  ? 'Ask a question about your documents and click on facts in the answer to see their source context.'
                  : 'Please wait while we initialize the system.'}
              </p>
            </motion.div>
          )}
        </div>
      </div>
    </TooltipProvider>
  );
};

export default FactClickInterface;