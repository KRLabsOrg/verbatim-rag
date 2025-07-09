import React from 'react';
import { motion } from 'framer-motion';
import { FaFileAlt } from 'react-icons/fa';
import { Button } from './ui/Button';
import { Badge } from './ui/Badge';
import { cn } from '../utils/cn';

const DocumentSelector = ({ 
  documents = [], 
  selectedDocIndex,
  onDocumentSelect 
}) => {
  if (documents.length === 0) {
    return null;
  }

  return (
    <div className="border-b border-secondary-200 p-4 bg-secondary-50/50">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-secondary-700">
          Source Documents ({documents.length})
        </h3>
        <div className="flex items-center space-x-3 text-xs text-secondary-500">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span>With highlights</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 rounded-full bg-secondary-400"></div>
            <span>No highlights</span>
          </div>
        </div>
      </div>
      
      <div className="flex flex-wrap gap-2">
        {documents.map((doc, idx) => {
          const hasHighlights = doc.highlights && doc.highlights.length > 0;
          const isSelected = selectedDocIndex === idx;
          
          return (
            <motion.div
              key={idx}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.2, delay: idx * 0.05 }}
            >
              <Button
                variant={isSelected ? "default" : "outline"}
                size="sm"
                onClick={() => onDocumentSelect(idx)}
                className={cn(
                  "relative flex items-center space-x-2 transition-all duration-200",
                  isSelected && "ring-2 ring-primary-500 ring-offset-2",
                  hasHighlights ? "border-green-300 hover:border-green-400" : "border-secondary-300"
                )}
              >
                <div className={cn(
                  "w-2 h-2 rounded-full",
                  hasHighlights ? "bg-green-500" : "bg-secondary-400"
                )} />
                <FaFileAlt className="w-3 h-3" />
                <span className="font-medium">Doc {idx + 1}</span>
                {hasHighlights && (
                  <Badge variant="success" className="ml-1 text-xs">
                    {doc.highlights.length}
                  </Badge>
                )}
              </Button>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default DocumentSelector;