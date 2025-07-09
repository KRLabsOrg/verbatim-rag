import React from 'react';
import { FaGithub, FaQuestionCircle } from 'react-icons/fa';
import { motion } from 'framer-motion';
import { Button } from './ui/Button';
import { useApi } from '../contexts/ApiContext';
import { Badge } from './ui/Badge';

const Header = () => {
  const { isResourcesLoaded, isLoading } = useApi();

  return (
    <header className="sticky top-0 z-50 bg-white border-b border-slate-200 shadow-sm">
      <div className="container mx-auto px-4 py-3 max-w-7xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <motion.div
              initial={{ rotate: -10 }}
              animate={{ rotate: 0 }}
              transition={{ duration: 0.5 }}
              className="text-2xl font-bold"
              role="img"
              aria-label="Logo"
            >
              📚
            </motion.div>
            <div className="flex flex-col">
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Verbatim RAG
              </h1>
              <p className="text-xs text-slate-500 hidden sm:block">
                Intelligent Document Query System
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* Status indicator */}
            <div className="flex items-center space-x-2">
              <Badge 
                variant={isResourcesLoaded ? 'success' : isLoading ? 'warning' : 'secondary'}
                className="hidden sm:inline-flex"
              >
                {isResourcesLoaded ? 'Ready' : isLoading ? 'Loading...' : 'Initializing'}
              </Badge>
              <div className={`w-2 h-2 rounded-full ${
                isResourcesLoaded ? 'bg-green-500' : isLoading ? 'bg-yellow-500 animate-pulse' : 'bg-slate-300'
              }`} />
            </div>

            {/* Help button */}
            <Button 
              variant="ghost" 
              size="icon"
              className="text-slate-600 hover:text-slate-800"
              aria-label="Help"
            >
              <FaQuestionCircle className="w-4 h-4" />
            </Button>

            {/* GitHub link */}
            <a 
              href="https://github.com/your-repo" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none hover:bg-slate-100 h-10 w-10 text-slate-600 hover:text-slate-800"
              aria-label="GitHub"
            >
              <FaGithub className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 