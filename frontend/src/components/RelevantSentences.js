import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FaQuoteLeft, FaSearch, FaFileAlt, FaChevronDown, FaChevronUp } from 'react-icons/fa';
import { Input } from './ui/Input';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';
import { cn } from '../lib/utils';

const RelevantSentences = ({ 
  highlights = [], 
  onHighlightClick, 
  selectedHighlight,
  currentQuery 
}) => {
  const [searchText, setSearchText] = useState('');
  const [isExpanded, setIsExpanded] = useState(true);

  // Filter highlights by search text
  const filteredHighlights = highlights.filter(highlight => 
    !searchText || highlight.text.toLowerCase().includes(searchText.toLowerCase())
  );

  // Get citation index for a highlight
  const getCitationIndex = (highlight) => {
    if (!currentQuery?.structured_answer?.citations) return -1;
    
    return currentQuery.structured_answer.citations.findIndex(
      citation => citation.doc_index === highlight.docIndex && 
                 citation.highlight_index === currentQuery.documents[highlight.docIndex]?.highlights?.findIndex(
                   h => h.text === highlight.text
                 )
    );
  };

  if (highlights.length === 0) {
    return null;
  }

  return (
    <div className="mt-6 pt-4 border-t border-slate-200">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center space-x-2 p-0 h-auto font-medium text-slate-700"
        >
          <FaQuoteLeft className="w-4 h-4" />
          <span>Relevant Sentences</span>
          {isExpanded ? <FaChevronUp className="w-3 h-3" /> : <FaChevronDown className="w-3 h-3" />}
        </Button>
        
        <div className="flex items-center space-x-2">
          <Badge variant="secondary" className="text-xs">
            {highlights.length} found
          </Badge>
          {highlights.length > 0 && (
            <div className="relative">
              <Input
                placeholder="Search sentences..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                className="h-8 w-40 pr-8 text-xs"
              />
              <FaSearch className="absolute right-2 top-2 w-3 h-3 text-slate-400" />
            </div>
          )}
        </div>
      </div>

      {/* Collapsible content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="max-h-80 overflow-y-auto space-y-2 scrollbar-thin">
              {filteredHighlights.length === 0 ? (
                <div className="text-center py-8 text-slate-500">
                  <FaSearch className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">
                    {searchText ? 'No matching sentences found' : 'No relevant sentences available'}
                  </p>
                </div>
              ) : (
                filteredHighlights.map((highlight, idx) => {
                  const citationIndex = getCitationIndex(highlight);
                  const isSelected = selectedHighlight === highlight;
                  
                  return (
                    <motion.div
                      key={idx}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2, delay: idx * 0.05 }}
                      className={cn(
                        "group relative p-3 rounded-lg border cursor-pointer transition-all duration-200",
                        isSelected
                          ? "border-blue-300 bg-blue-50 shadow-sm"
                          : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                      )}
                      onClick={() => onHighlightClick(highlight)}
                    >
                      {/* Quote icon */}
                      <FaQuoteLeft className="absolute top-3 left-3 w-3 h-3 text-slate-400 group-hover:text-slate-600" />
                      
                      {/* Content */}
                      <div className="pl-6">
                        <p className="text-sm leading-relaxed text-slate-900 mb-2">
                          {highlight.text}
                        </p>
                        
                        {/* Footer */}
                        <div className="flex items-center justify-between text-xs text-slate-500">
                          <div className="flex items-center space-x-2">
                            <FaFileAlt className="w-3 h-3" />
                            <span>Document {highlight.docIndex + 1}</span>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            {citationIndex !== -1 && (
                              <Badge variant="default" className="text-xs">
                                [{citationIndex + 1}]
                              </Badge>
                            )}
                            <span className="text-xs text-slate-400">
                              Click to view
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Selection indicator */}
                      {isSelected && (
                        <div className="absolute right-2 top-2 w-2 h-2 rounded-full bg-blue-500" />
                      )}
                    </motion.div>
                  );
                })
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default RelevantSentences;