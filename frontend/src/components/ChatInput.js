import React, { useState } from 'react';
import { FaPaperPlane, FaSpinner } from 'react-icons/fa';
import { Button } from './ui/Button';
import { Input } from './ui/Input';
import { cn } from '../lib/utils';

const ChatInput = ({ 
  onSubmit, 
  isLoading = false, 
  isDisabled = false,
  placeholder = "Ask a question..."
}) => {
  const [question, setQuestion] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim() || isDisabled || isLoading) return;
    
    onSubmit(question);
    setQuestion('');
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t border-slate-200 p-4 bg-white">
      <form onSubmit={handleSubmit} className="flex items-end space-x-3">
        <div className="flex-1 relative">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={isDisabled || isLoading}
            className={cn(
              "min-h-[44px] pr-12 resize-none",
              "focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            )}
          />
          
          {/* Character counter for long messages */}
          {question.length > 200 && (
            <div className="absolute bottom-2 right-2 text-xs text-slate-400">
              {question.length}/1000
            </div>
          )}
        </div>

        <Button
          type="submit"
          disabled={!question.trim() || isDisabled || isLoading}
          className="min-w-[44px] h-11 flex items-center justify-center"
        >
          {isLoading ? (
            <FaSpinner className="w-4 h-4 animate-spin" />
          ) : (
            <FaPaperPlane className="w-4 h-4" />
          )}
        </Button>
      </form>
    </div>
  );
};

export default ChatInput;