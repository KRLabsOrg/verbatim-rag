import React from 'react';
import { ApiProvider } from './contexts/ApiContext';
import { DocumentsProvider } from './contexts/DocumentsContext';
import CleanFactInterface from './components/CleanFactInterface';
import ErrorBoundary from './components/ErrorBoundary';

function App() {
  return (
    <ErrorBoundary>
      <ApiProvider>
        <DocumentsProvider>
          <CleanFactInterface />
        </DocumentsProvider>
      </ApiProvider>
    </ErrorBoundary>
  );
}

export default App;