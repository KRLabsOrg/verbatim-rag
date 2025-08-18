import React, { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageCircle, FileText, Sparkles, ExternalLink } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card.jsx';
import { Badge } from './ui/badge.jsx';
import { Button } from './ui/button.jsx';
import { Input } from './ui/input.jsx';
import { ScrollArea } from './ui/scroll-area.jsx';
import { TooltipProvider, Tooltip, TooltipTrigger, TooltipContent } from './ui/tooltip.jsx';
import { useApi } from '../contexts/ApiContext';

// Markdown components with custom styling
const MarkdownComponents = {
  h1: ({ children }) => <h1 className="text-2xl font-bold mt-4 mb-3">{children}</h1>,
  h2: ({ children }) => <h2 className="text-xl font-bold mt-4 mb-3">{children}</h2>,
  h3: ({ children }) => <h3 className="text-lg font-semibold mt-4 mb-2">{children}</h3>,
  p: ({ children }) => <p className="mb-2">{children}</p>,
  ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
  li: ({ children }) => <li className="mb-1">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-blue-300 pl-4 py-2 bg-blue-50 italic mb-2">
      {children}
    </blockquote>
  ),
  code: ({ inline, children }) => 
    inline 
      ? <code className="bg-gray-100 px-1 py-0.5 rounded text-sm font-mono">{children}</code>
      : <pre className="bg-gray-100 p-3 rounded text-sm font-mono overflow-x-auto mb-2"><code>{children}</code></pre>,
  table: ({ children }) => (
    <div className="overflow-x-auto mb-4 max-w-full">
      <table className="w-full border-collapse border border-gray-300 text-sm">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-gray-300 px-2 py-1 bg-gray-50 font-semibold text-left text-xs">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-gray-300 px-2 py-1 text-xs">
      {children}
    </td>
  ),
};

const CleanFactInterface = () => {
  const { isLoading, isResourcesLoaded, currentQuery, submitQuery } = useApi();
  const [question, setQuestion] = useState('');
  const [selectedDocument, setSelectedDocument] = useState(0);
  const [highlightedFactId, setHighlightedFactId] = useState(null);
  const documentScrollRef = useRef(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || !isResourcesLoaded) return;
    
    await submitQuery(question);
    setQuestion('');
    setSelectedDocument(0);
  };

  // Extract facts preserving backend ordering & types
  const extractFacts = () => {
    if (!currentQuery?.structured_answer?.citations) return [];
    return currentQuery.structured_answer.citations.map((c, i) => ({
      id: i,
      text: c.text,
      docIndex: c.doc_index,
      highlightIndex: c.highlight_index,
      citationNumber: (c.number) ? c.number : i + 1,
      factType: c.type || 'display'
    }));
  };

  // Group chunks by document_id (not by title/source)
  const groupDocuments = () => {
    if (!currentQuery?.documents) return [];
    
    const documentGroups = {};
    
    currentQuery.documents.forEach((chunk, chunkIndex) => {
      const docId = chunk.metadata?.document_id || `doc_${chunkIndex}`;
      
      if (!documentGroups[docId]) {
        documentGroups[docId] = {
          id: docId,
          title: chunk.title || chunk.source || `Document`,
          source: chunk.source || '',
          metadata: chunk.metadata || {},
          chunks: [],
          allHighlights: []
        };
      }
      
      documentGroups[docId].chunks.push({
        ...chunk,
        originalIndex: chunkIndex
      });
      
      // Collect highlights
      if (chunk.highlights) {
        chunk.highlights.forEach((highlight) => {
          documentGroups[docId].allHighlights.push({
            ...highlight,
            chunkIndex,
          });
        });
      }
    });
    
    return Object.values(documentGroups);
  };

  const facts = extractFacts();
  const groupedDocuments = groupDocuments();

  // Handle fact click with scroll-to-highlight
  const handleFactClick = (fact) => {
    const targetGroupIndex = groupedDocuments.findIndex(group => 
      group.chunks.some(chunk => chunk.originalIndex === fact.docIndex)
    );
    
    if (targetGroupIndex !== -1) {
      setSelectedDocument(targetGroupIndex);
      setHighlightedFactId(fact.id);
      
      // Scroll to highlight after a short delay to allow DOM update
      setTimeout(() => {
        const highlightElement = document.querySelector(`[data-highlight-id="${fact.id}"]`);
        if (highlightElement && documentScrollRef.current) {
          const scrollContainer = documentScrollRef.current.querySelector('[data-radix-scroll-area-viewport]');
          if (scrollContainer) {
            const elementTop = highlightElement.offsetTop;
            const containerHeight = scrollContainer.clientHeight;
            
            // Center the highlight in the viewport with enhanced smooth scrolling
            const targetScroll = elementTop - containerHeight / 2;
            scrollContainer.scrollTo({ 
              top: Math.max(0, targetScroll), 
              behavior: 'smooth' 
            });
            
            // Add a subtle flash effect to help user see the highlighted text
            setTimeout(() => {
              if (highlightElement) {
                highlightElement.style.animation = 'none';
                // Trigger reflow - ESLint disable because we need the side effect
                // eslint-disable-next-line no-unused-expressions
                highlightElement.offsetHeight;
                highlightElement.style.animation = 'flash 0.8s ease-in-out';
              }
            }, 300);
          }
        }
      }, 100);
    }
  };

  // Inline citation component - clean and simple
  const InlineCitation = ({ citationNumber, factType = 'display', onClick }) => {
    return (
      <sup
        onClick={onClick}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}
        role="button"
        tabIndex={0}
        aria-label={`Show citation ${citationNumber}`}
        className={`align-super text-[0.65em] ml-0.5 cursor-pointer select-none transition-colors duration-150 focus:outline-none focus:ring-1 focus:ring-blue-400 rounded font-medium ${factType === 'reference' ? 'text-blue-400 hover:text-blue-500' : 'text-blue-600 hover:text-blue-700'}`}
      >
        [{citationNumber}]
      </sup>
    );
  };

  // Render answer with inline clickable citations and ReactMarkdown support
  const renderAnswerWithFacts = () => {
    if (!currentQuery?.answer) {
      return null;
    }

    // remark plugin to turn [n] into link nodes we later render as superscript citations
    const remarkInlineCitations = () => (tree) => {
      const walk = (node, parent) => {
        if (!node) return;
        if (parent && (parent.type === 'code' || parent.type === 'inlineCode')) return;
        if (node.type === 'text') {
          const value = node.value;
            // quick check
          if (!/\[\d+\]/.test(value)) return;
          const regex = /\[(\d+)\]/g;
          let match; let last = 0; const newNodes = [];
          while ((match = regex.exec(value)) !== null) {
            const idx = match.index;
            if (idx > last) newNodes.push({ type: 'text', value: value.slice(last, idx) });
            const num = parseInt(match[1]);
            newNodes.push({
              type: 'link',
              url: `#citation-${num}`,
              data: { hProperties: { 'data-citation': num } },
              children: [{ type: 'text', value: `[${num}]` }]
            });
            last = idx + match[0].length;
          }
          if (last < value.length) newNodes.push({ type: 'text', value: value.slice(last) });
          if (newNodes.length) {
            const idxInParent = parent.children.indexOf(node);
            parent.children.splice(idxInParent, 1, ...newNodes);
          }
          return;
        }
        if (node.children) [...node.children].forEach(child => walk(child, node));
      };
      walk(tree, null);
    };

    return (
      <div className="text-lg leading-relaxed text-gray-700">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkInlineCitations]}
          components={{
            ...MarkdownComponents,
            a: ({ node, children, ...rest }) => {
              const citationNum = node?.properties?.['data-citation'];
              if (citationNum) {
                const num = parseInt(citationNum);
                const fact = facts.find(f => f.citationNumber === num);
                return (
                  <InlineCitation
                    citationNumber={num}
                    onClick={() => fact && handleFactClick(fact)}
                  />
                );
              }
              return <a {...rest}>{children}</a>;
            }
          }}
        >
          {currentQuery.answer}
        </ReactMarkdown>
      </div>
    );
  };

  // Render document with highlights
  const renderDocument = ({ documentGroup, highlightedFact }) => {
    if (!documentGroup) return null;

    const renderSourceLink = (source) => {
      if (source.startsWith('http://') || source.startsWith('https://')) {
        return (
          <a 
            href={source} 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 hover:underline"
          >
            <ExternalLink className="w-4 h-4" />
            View Full Document
          </a>
        );
      } else {
        return <span className="text-gray-600">{source.split('/').pop()}</span>;
      }
    };

    return (
      <div className="space-y-4">
        {/* Document header */}
        <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-r-lg">
          <h3 className="font-semibold text-lg text-gray-800 mb-2">{documentGroup.title}</h3>
          {documentGroup.source && (
            <div className="text-sm text-gray-600 mb-2">
              <span className="font-medium">Source:</span> {renderSourceLink(documentGroup.source)}
            </div>
          )}
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <span>{documentGroup.chunks.length} chunk{documentGroup.chunks.length !== 1 ? 's' : ''}</span>
            <span>{documentGroup.allHighlights.length} highlight{documentGroup.allHighlights.length !== 1 ? 's' : ''}</span>
          </div>
        </div>

        {/* Document chunks */}
        <div className="space-y-4">
          {documentGroup.chunks.map((chunk, chunkIndex) => {
            const highlights = chunk.highlights || [];
            
            // Create highlight map
            const highlightMap = highlights.map((highlight) => {
              const factForHighlight = facts.find(f => 
                f.docIndex === chunk.originalIndex && f.text === highlight.text
              );
              return {
                ...highlight,
                factId: factForHighlight?.id,
                isHighlighted: factForHighlight?.id === highlightedFact
              };
            });

            highlightMap.sort((a, b) => a.start - b.start);

            // Build text parts with highlights
            let lastIndex = 0;
            const parts = [];

            highlightMap.forEach((highlight) => {
              if (highlight.start > lastIndex) {
                parts.push({
                  type: 'text',
                  content: chunk.content.substring(lastIndex, highlight.start)
                });
              }

              parts.push({
                type: 'highlight',
                content: highlight.text,
                isHighlighted: highlight.isHighlighted
              });

              lastIndex = highlight.end;
            });

            if (lastIndex < chunk.content.length) {
              parts.push({
                type: 'text',
                content: chunk.content.substring(lastIndex)
              });
            }

            return (
              <div key={chunkIndex}>
                {/* Content break indicator */}
                {chunkIndex > 0 && (
                  <div className="flex items-center justify-center py-3">
                    <div className="flex-1 border-t border-dashed border-gray-300"></div>
                    <span className="px-3 text-xs text-gray-500 bg-gray-50 rounded-full">
                      Content break
                    </span>
                    <div className="flex-1 border-t border-dashed border-gray-300"></div>
                  </div>
                )}
                
                {/* Chunk content */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <div className="text-base leading-7 text-gray-700 max-w-full overflow-hidden">
                    {parts.map((part, index) => {
                      if (part.type === 'highlight') {
                        // For highlights, use a wrapper div with highlight styling
                        return (
                          <div
                            key={index}
                            data-highlight-id={facts.find(f => f.text === part.content)?.id}
                            className={`px-2 py-1 rounded transition-all duration-500 ease-out ${
                              part.isHighlighted 
                                ? 'bg-yellow-100 border-l-2 border-yellow-500 shadow-sm font-medium ring-1 ring-yellow-300' 
                                : 'bg-yellow-100 hover:bg-yellow-200 hover:shadow-sm'
                            }`}
                          >
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm]}
                              components={MarkdownComponents}
                            >
                              {part.content}
                            </ReactMarkdown>
                          </div>
                        );
                      }
                      // For non-highlighted content, use ReactMarkdown
                      return (
                        <div key={index}>
                          <ReactMarkdown 
                            remarkPlugins={[remarkGfm]}
                            components={MarkdownComponents}
                          >
                            {part.content}
                          </ReactMarkdown>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <TooltipProvider>
      <style>{`
        @keyframes flash {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
        
        @keyframes citationPulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }
      `}</style>
      <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <header className="bg-indigo-700 text-white p-4 shadow-lg flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600 rounded-lg">
              <MessageCircle className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Verbatim RAG</h1>
              <p className="text-indigo-200">Click facts to see their sources</p>
            </div>
          </div>
          
          <Badge variant={isResourcesLoaded ? "default" : "secondary"} className="bg-indigo-600 text-white">
            {isResourcesLoaded ? '✓ Ready' : '⏳ Loading...'}
          </Badge>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 min-h-0">
        {/* Left panel - Chat */}
        <div className="w-1/2 bg-white flex flex-col min-h-0">
          <ScrollArea className="flex-1">
            <div className="p-6">
            {/* Query input */}
            <div className="mb-8">
              <form onSubmit={handleSubmit} className="space-y-4">
                <label className="block text-lg font-medium text-gray-700 mb-3">
                  Ask a question about your documents
                </label>
                <div className="flex gap-3">
                  <Input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What would you like to know?"
                    className="flex-1 p-4 text-base"
                    disabled={!isResourcesLoaded || isLoading}
                  />
                  <Button 
                    type="submit" 
                    disabled={!question.trim() || !isResourcesLoaded || isLoading}
                    className="px-8 py-4 text-base"
                  >
                    {isLoading ? 'Thinking...' : 'Ask'}
                  </Button>
                </div>
                <p className="text-sm text-gray-500">
                  Answers include exact citations from your document collection
                </p>
              </form>
            </div>

            {/* Answer section */}
            <AnimatePresence>
              {currentQuery && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="space-y-6"
                >
                  {/* Question */}
                  <div className="bg-blue-50 border-l-4 border-blue-400 p-6 rounded-r-lg">
                    <div className="flex items-start gap-3">
                      <MessageCircle className="w-5 h-5 text-blue-600 mt-1" />
                      <div>
                        <p className="text-sm text-blue-600 font-medium mb-1">Your Question</p>
                        <p className="text-lg font-medium text-gray-800">{currentQuery.question}</p>
                      </div>
                    </div>
                  </div>

                  {/* Answer */}
                  {currentQuery.answer && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                          <Sparkles className="w-6 h-6 text-indigo-600" />
                          Answer
                          {facts.length > 0 && (
                            <Badge variant="secondary" className="ml-2">
                              {facts.length} citation{facts.length !== 1 ? 's' : ''}
                            </Badge>
                          )}
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        {renderAnswerWithFacts()}
                      </CardContent>
                    </Card>
                  )}
                </motion.div>
              )}

              {/* Empty state */}
              {!currentQuery && !isLoading && (
                <div className="text-center py-16">
                  <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full flex items-center justify-center">
                    <MessageCircle className="w-10 h-10 text-blue-600" />
                  </div>
                  <h3 className="text-2xl font-semibold mb-4 text-gray-800">
                    {isResourcesLoaded ? 'Ready to answer your questions' : 'Loading system...'}
                  </h3>
                  <p className="text-gray-600 text-lg max-w-lg mx-auto">
                    {isResourcesLoaded 
                      ? 'Ask a question and click on facts in the answer to see their exact source context.'
                      : 'Please wait while we initialize the system.'}
                  </p>
                </div>
              )}
            </AnimatePresence>
            </div>
          </ScrollArea>
        </div>

        {/* Right panel - Documents */}
        <div className="w-1/2 bg-gray-100 flex flex-col">
          <div className="p-4 border-b border-gray-200 bg-white flex-shrink-0">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-700 flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                Source Documents
              </h2>
              {groupedDocuments.length > 0 && (
                <span className="px-2 py-1 bg-gray-100 text-gray-600 text-sm rounded">
                  {groupedDocuments.length} document{groupedDocuments.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Document tabs */}
          {groupedDocuments.length > 1 && (
            <div className="p-4 bg-white border-b border-gray-200 flex-shrink-0">
              <div className="flex gap-2 overflow-x-auto">
                {groupedDocuments.map((docGroup, index) => (
                  <Tooltip key={index}>
                    <TooltipTrigger asChild>
                      <Button
                        onClick={() => setSelectedDocument(index)}
                        variant={selectedDocument === index ? "default" : "outline"}
                        size="sm"
                        className="whitespace-nowrap"
                      >
                        <span className="max-w-32 truncate">{docGroup.title}</span>
                        {docGroup.allHighlights.length > 0 && (
                          <Badge variant="secondary" className="ml-1">
                            {docGroup.allHighlights.length}
                          </Badge>
                        )}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>{docGroup.title}</p>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </div>
            </div>
          )}

          {/* Document content - INDEPENDENT SCROLLING */}
          <div className="flex-1 bg-white min-h-0">
            <ScrollArea ref={documentScrollRef} className="h-full">
              <div className="p-4">
              {groupedDocuments.length > 0 ? (
                groupedDocuments[selectedDocument] ? (
                  renderDocument({
                    documentGroup: groupedDocuments[selectedDocument],
                    highlightedFact: highlightedFactId
                  })
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <FileText className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No document selected</p>
                  </div>
                )
              ) : (
                <div className="text-center py-16 text-gray-500">
                  <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <h3 className="text-lg font-medium mb-2">No Documents Yet</h3>
                  <p>Ask a question to see relevant source documents</p>
                </div>
              )}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>
    </div>
    </TooltipProvider>
  );
};

export default CleanFactInterface;