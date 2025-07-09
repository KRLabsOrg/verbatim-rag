import React from 'react';
import { motion } from 'framer-motion';
import { FaUser, FaRobot } from 'react-icons/fa';
import { cn } from '../lib/utils';

const ChatMessage = ({ message, isUser, isLoading = false }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex w-full mb-4",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      <div
        className={cn(
          "flex max-w-[80%] items-start space-x-3",
          isUser ? "flex-row-reverse space-x-reverse" : "flex-row"
        )}
      >
        {/* Avatar */}
        <div
          className={cn(
            "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium",
            isUser
              ? "bg-blue-600 text-white"
              : "bg-slate-100 text-slate-700"
          )}
        >
          {isUser ? <FaUser className="w-3 h-3" /> : <FaRobot className="w-3 h-3" />}
        </div>

        {/* Message bubble */}
        <div
          className={cn(
            "relative px-4 py-3 rounded-lg",
            isUser
              ? "bg-blue-600 text-white"
              : "bg-white border border-slate-200 text-slate-900"
          )}
        >
          {/* Message content */}
          <div className="text-sm leading-relaxed">
            {isLoading ? (
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-slate-500">Assistant is thinking...</span>
              </div>
            ) : (
              <div className="whitespace-pre-wrap">{message}</div>
            )}
          </div>

          {/* Message tail */}
          <div
            className={cn(
              "absolute top-4 w-0 h-0 border-t-8 border-b-8 border-transparent",
              isUser
                ? "right-[-8px] border-l-8 border-l-blue-600"
                : "left-[-8px] border-r-8 border-r-white"
            )}
          />
          {!isUser && (
            <div
              className="absolute left-[-9px] top-4 w-0 h-0 border-t-8 border-b-8 border-transparent border-r-8 border-r-slate-200"
            />
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default ChatMessage;