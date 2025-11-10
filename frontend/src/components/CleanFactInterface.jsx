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
  h1: ({ children }) => <h1 className="text-2xl lg:text-3xl font-bold mt-4 lg:mt-5 mb-3 lg:mb-4">{children}</h1>,
  h2: ({ children }) => <h2 className="text-xl lg:text-2xl font-bold mt-4 lg:mt-5 mb-3 lg:mb-4">{children}</h2>,
  h3: ({ children }) => <h3 className="text-lg lg:text-xl font-semibold mt-4 mb-2 lg:mb-3">{children}</h3>,
  p: ({ children }) => <p className="mb-2 lg:mb-3">{children}</p>,
  ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
  li: ({ children }) => <li className="mb-1">{children}</li>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-primary/30 pl-4 py-2 bg-muted italic mb-2">
      {children}
    </blockquote>
  ),
  code: ({ inline, children }) => 
    inline 
      ? <code className="bg-muted px-1 py-0.5 rounded text-sm font-mono">{children}</code>
      : <pre className="bg-muted p-3 rounded text-sm font-mono overflow-x-auto mb-2"><code>{children}</code></pre>,
  table: ({ children }) => (
    <div className="overflow-x-auto mb-4 max-w-full">
      <table className="w-full border-collapse border border-border text-sm">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border px-2 py-1 bg-muted font-semibold text-left text-xs">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-2 py-1 text-xs">
      {children}
    </td>
  ),
};

const CleanFactInterface = () => {
  const { isLoading, isResourcesLoaded, currentQuery, submitQuery, resetQuery } = useApi();
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

  const goHome = () => {
    resetQuery();
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
        className={`align-super text-[0.65em] ml-0.5 cursor-pointer select-none transition-colors duration-150 focus:outline-none focus:ring-1 focus:ring-primary rounded font-medium ${factType === 'reference' ? 'text-primary/70 hover:text-primary/90' : 'text-primary hover:text-primary/80'}`}
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
      <div className="text-lg lg:text-xl leading-relaxed text-foreground">
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
            className="inline-flex items-center gap-1 text-primary hover:text-primary/80 hover:underline"
          >
            <ExternalLink className="w-4 h-4" />
            View Full Document
          </a>
        );
      } else {
        return <span className="text-muted-foreground">{source.split('/').pop()}</span>;
      }
    };

    return (
      <div className="space-y-4">
        {/* Document header */}
        <div className="bg-muted border-l-4 border-primary p-4 lg:p-6 rounded-r-lg">
          <h3 className="font-semibold text-xl lg:text-2xl text-foreground mb-2 lg:mb-3">{documentGroup.title}</h3>
          {documentGroup.source && (
            <div className="text-sm lg:text-base text-muted-foreground mb-2">
              <span className="font-medium">Source:</span> {renderSourceLink(documentGroup.source)}
            </div>
          )}
          <div className="flex items-center gap-4 text-sm lg:text-base text-muted-foreground">
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
                    <div className="flex-1 border-t border-dashed border-border"></div>
                    <span className="px-3 text-xs text-muted-foreground bg-muted rounded-full">
                      Content break
                    </span>
                    <div className="flex-1 border-t border-dashed border-border"></div>
                  </div>
                )}
                
                {/* Chunk content */}
                <div className="bg-card border border-border rounded-lg p-4 lg:p-6">
                  <div className="text-base lg:text-lg leading-7 lg:leading-8 text-foreground max-w-full overflow-hidden">
                    {parts.map((part, index) => {
                      if (part.type === 'highlight') {
                        // For highlights, use a wrapper div with highlight styling
                        return (
                          <div
                            key={index}
                            data-highlight-id={facts.find(f => f.text === part.content)?.id}
                            className={`px-2 py-1 rounded transition-all duration-500 ease-out ${
                              part.isHighlighted
                                ? 'bg-accent border-l-2 border-primary shadow-sm font-medium ring-1 ring-primary/30'
                                : 'bg-accent/50 hover:bg-accent hover:shadow-sm'
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
      <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <header className="bg-white border-b border-border py-3 lg:py-4 shadow-sm flex-shrink-0">
        <div className="max-w-[1800px] mx-auto px-4 lg:px-8 flex items-center justify-between">
          <button
            onClick={goHome}
            className="hover:opacity-70 transition-opacity cursor-pointer"
            title="Go home"
          >
            <h1 className="text-xl lg:text-2xl font-bold text-foreground">KR Labs <span className="text-muted-foreground">•</span> Verbatim RAG</h1>
          </button>

          <Badge variant={isResourcesLoaded ? "default" : "secondary"}>
            {isResourcesLoaded ? '✓ Ready' : '⏳ Loading...'}
          </Badge>
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 min-h-0 max-w-[1800px] mx-auto w-full">
        {/* Left panel - Chat */}
        <div className="w-1/2 bg-card flex flex-col min-h-0">
          <ScrollArea className="flex-1">
            <div className="p-6 lg:p-8 xl:p-10">
            {/* Query input */}
            <div className="mb-8">
              <form onSubmit={handleSubmit} className="space-y-4 lg:space-y-5">
                <label className="block text-xl lg:text-2xl font-medium text-foreground mb-4">
                  Ask a question about your documents
                </label>
                <div className="flex gap-3 lg:gap-4">
                  <Input
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="What would you like to know?"
                    className="flex-1 p-4 lg:p-5 text-base lg:text-lg"
                    disabled={!isResourcesLoaded || isLoading}
                  />
                  <Button
                    type="submit"
                    disabled={!question.trim() || !isResourcesLoaded || isLoading}
                    className="px-8 lg:px-10 py-4 lg:py-5 text-base lg:text-lg"
                  >
                    {isLoading ? 'Thinking...' : 'Ask'}
                  </Button>
                </div>
                <p className="text-sm lg:text-base text-muted-foreground">
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
                  <div className="bg-muted border-l-4 border-primary p-6 lg:p-8 rounded-r-lg">
                    <div className="flex items-start gap-3 lg:gap-4">
                      <MessageCircle className="w-6 h-6 lg:w-7 lg:h-7 text-primary mt-1" />
                      <div>
                        <p className="text-sm lg:text-base text-primary font-medium mb-1">Your Question</p>
                        <p className="text-lg lg:text-xl font-medium text-foreground">{currentQuery.question}</p>
                      </div>
                    </div>
                  </div>

                  {/* Answer */}
                  {currentQuery.answer && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-xl lg:text-2xl">
                          <Sparkles className="w-6 h-6 lg:w-7 lg:h-7 text-primary" />
                          Answer
                          {facts.length > 0 && (
                            <Badge variant="secondary" className="ml-2 text-sm lg:text-base">
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
                <div className="text-center py-16 lg:py-20">
                  <div className="w-24 h-24 lg:w-32 lg:h-32 mx-auto mb-6 lg:mb-8 bg-accent rounded-full flex items-center justify-center p-4 lg:p-6">
                    <img src="/chiliground-transparent.png" alt="Chili Mascot" className="w-full h-full object-contain" />
                  </div>
                  <h3 className="text-2xl lg:text-3xl font-semibold mb-4 lg:mb-6 text-foreground">
                    {isResourcesLoaded ? 'Ready to answer your questions' : 'Loading system...'}
                  </h3>
                  <p className="text-muted-foreground text-lg lg:text-xl max-w-2xl mx-auto mb-8 lg:mb-12">
                    {isResourcesLoaded
                      ? 'Ask a question and click on facts in the answer to see their exact source context.'
                      : 'Please wait while we initialize the system.'}
                  </p>

                  {/* Sample Questions */}
                  {isResourcesLoaded && (
                    <div className="max-w-3xl mx-auto space-y-3 lg:space-y-4">
                      <p className="text-sm lg:text-base text-muted-foreground font-medium mb-4">Try these example questions:</p>
                      {[
                        "What is the main contribution of the Verbatim RAG paper?",
                        "Cite the exact lines that define the extraction method.",
                        "Which datasets and metrics are used for evaluation?"
                      ].map((sampleQuestion, index) => (
                        <button
                          key={index}
                          onClick={() => setQuestion(sampleQuestion)}
                          className="w-full text-left p-4 lg:p-5 bg-card border-2 border-border hover:border-primary rounded-lg transition-all hover:shadow-md group"
                        >
                          <p className="text-base lg:text-lg text-foreground group-hover:text-primary transition-colors">
                            {sampleQuestion}
                          </p>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </AnimatePresence>
            </div>
          </ScrollArea>
        </div>

        {/* Right panel - Documents */}
        <div className="w-1/2 bg-secondary flex flex-col">
          <div className="p-4 lg:p-6 border-b border-border bg-card flex-shrink-0">
            <div className="flex items-center justify-between">
              <h2 className="text-xl lg:text-2xl font-semibold text-foreground flex items-center gap-2 lg:gap-3">
                <FileText className="w-6 h-6 lg:w-7 lg:h-7 text-primary" />
                Source Documents
              </h2>
              {groupedDocuments.length > 0 && (
                <span className="px-3 py-1.5 lg:px-4 lg:py-2 bg-secondary text-muted-foreground text-sm lg:text-base rounded">
                  {groupedDocuments.length} document{groupedDocuments.length !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </div>

          {/* Document tabs */}
          {groupedDocuments.length > 1 && (
            <div className="p-4 lg:p-6 bg-card border-b border-border flex-shrink-0">
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
          <div className="flex-1 bg-card min-h-0">
            <ScrollArea ref={documentScrollRef} className="h-full">
              <div className="p-4 lg:p-6 xl:p-8">
              {groupedDocuments.length > 0 ? (
                groupedDocuments[selectedDocument] ? (
                  renderDocument({
                    documentGroup: groupedDocuments[selectedDocument],
                    highlightedFact: highlightedFactId
                  })
                ) : (
                  <div className="text-center py-8 lg:py-12 text-muted-foreground">
                    <FileText className="w-12 h-12 lg:w-16 lg:h-16 mx-auto mb-3 lg:mb-4 opacity-50" />
                    <p className="text-base lg:text-lg">No document selected</p>
                  </div>
                )
              ) : (
                <div className="text-center py-16 lg:py-20 text-muted-foreground">
                  <FileText className="w-16 h-16 lg:w-20 lg:h-20 mx-auto mb-4 lg:mb-6 opacity-50" />
                  <h3 className="text-xl lg:text-2xl font-medium mb-2 lg:mb-3">No Documents Yet</h3>
                  <p className="text-base lg:text-lg">Ask a question to see relevant source documents</p>
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